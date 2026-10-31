import CuboCore
@testable import CuboVirtual
import Foundation
import Metal
import Testing

/// Where the `.metal` reducers spend their time on a cube that is not resident (CuboKit#8). Skipped unless
/// `CUBOKIT_METAL_BENCH` is set: `CUBOKIT_METAL_BENCH=1 swift test -c release --filter MetalReducerBench`. On a
/// ``MemoryCube`` it prints the median and best of 11 calls of the public CPU and Metal reducers, then the mean-and-max
/// pass split into its parts over every tile: setup (an empty command buffer), reading the tile out of the source, the
/// upload (a new shared buffer filled from the tile, or a copy into one reused buffer), the kernel alone on a buffer that
/// is already there, and the readback with the Double combination.
/// A source that is not a ``MemoryCube``: the `.metal` path stages its tiles (as for a memory-mapped cube).
private actor StagedCube: CubeSource {
    nonisolated let shape: CubeShape
    private let inner: MemoryCube
    init(_ inner: MemoryCube) { self.inner = inner; self.shape = inner.shape }
    func readPattern(row: Int, column: Int) async throws -> DiffractionPattern { try await inner.readPattern(row: row, column: column) }
    func readScanTile(rows: Range<Int>) async throws -> ScanTile { try await inner.readScanTile(rows: rows) }
}

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["CUBOKIT_METAL_BENCH"] != nil, "set CUBOKIT_METAL_BENCH"))
struct MetalReducerBench {
    private func time(_ calls: Int = 11, _ body: () async throws -> Void) async rethrows -> (Double, Double) {
        try await body()
        var ts = [Double]()
        for _ in 0..<calls {
            let t0 = DispatchTime.now().uptimeNanoseconds
            try await body()
            ts.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
        }
        ts.sort()
        return (ts[ts.count / 2], ts[0])
    }

    private func report(_ name: String, _ t: (Double, Double)) {
        print(String(format: "METALBENCH %-34@ median %7.2f ms  best %7.2f ms", name as NSString, t.0, t.1))
    }

    private func cube(_ shape: CubeShape) throws -> MemoryCube {
        var state: UInt64 = 7
        let values = (0..<shape.elementCount).map { _ -> Float in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(state >> 40) / 1024
        }
        return try MemoryCube(shape: shape, values: values)
    }

    @Test(arguments: [(100, 100, 64, 64), (64, 64, 128, 128), (32, 32, 64, 64)])
    func memoryCube(_ dims: (Int, Int, Int, Int)) async throws {
        guard VirtualDiffraction.metalIsAvailable else { return }
        let shape = try CubeShape(scanRows: dims.0, scanColumns: dims.1, detectorRows: dims.2, detectorColumns: dims.3)
        let cube = try cube(shape)
        let metal = try MetalReducers.context()
        let n = shape.detectorCount
        let ranges = try ScanTiling.automatic.ranges(for: shape)
        print("METALBENCH shape \(dims) \(shape.elementCount * 4 / 1_000_000) MB, \(ranges.count) tiles")
        let disk = DetectorGeometry.circle(centerRow: Double(dims.2) / 2, centerColumn: Double(dims.3) / 2,
                                           radius: Double(dims.2) / 4)
        let take = (0..<shape.scanCount).map { $0 % 3 == 0 }

        report("cpu meanAndMax", try await time { _ = try await VirtualDiffraction.meanAndMax(of: cube, path: .cpu) })
        report("metal meanAndMax", try await time { _ = try await VirtualDiffraction.meanAndMax(of: cube, path: .metal) })
        report("cpu virtual image", try await time { _ = try await VirtualImaging.image(of: cube, geometry: disk, path: .cpu) })
        report("metal virtual image", try await time { _ = try await VirtualImaging.image(of: cube, geometry: disk, path: .metal) })
        report("cpu mask mean", try await time {
            _ = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .mask(take), path: .cpu)
        })
        report("metal mask mean", try await time {
            _ = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .mask(take), path: .metal)
        })
        report("cpu centre of mass", try await time { _ = try await CenterOfMass.measure(of: cube, path: .cpu) })
        report("metal centre of mass", try await time { _ = try await CenterOfMass.measure(of: cube, path: .metal) })
        // a source that is not a MemoryCube (its tiles are staged) and a resident cube, for comparison
        let staged = StagedCube(cube)
        report("staged metal meanAndMax", try await time { _ = try await VirtualDiffraction.meanAndMax(of: staged, path: .metal) })
        report("staged metal virtual image", try await time {
            _ = try await VirtualImaging.image(of: staged, geometry: disk, path: .metal)
        })
        report("staged metal centre of mass", try await time { _ = try await CenterOfMass.measure(of: staged, path: .metal) })
        let resident = try await ResidentCube(loading: cube)
        report("resident metal meanAndMax", try await time {
            _ = try await VirtualDiffraction.meanAndMax(of: resident, path: .metal)
        })
        report("resident metal virtual image", try await time {
            _ = try await VirtualImaging.image(of: resident, geometry: disk, path: .metal)
        })

        // the parts of one Metal mean-and-max pass, summed over the tiles
        report("setup: empty command buffer x tiles", try await time {
            for _ in ranges {
                guard let c = metal.queue.makeCommandBuffer() else { return }
                await withCheckedContinuation { (k: CheckedContinuation<Void, Never>) in
                    c.addCompletedHandler { _ in k.resume() }
                    c.commit()
                }
            }
        })
        report("read: readScanTile", try await time {
            for r in ranges { _ = try await cube.readScanTile(rows: r) }
        })
        report("read + upload: new buffer", try await time {
            for r in ranges { _ = try metal.buffer(try await cube.readScanTile(rows: r).values, label: "bench") }
        })
        let rowsPerTile = try ScanTiling.automatic.rowsPerTile(for: shape)
        let staging = try metal.buffer(floats: rowsPerTile * shape.elementsPerScanRow, label: "bench staging")
        report("upload alone: memcpy into reused", try await time {
            for r in ranges {
                cube.values.withUnsafeBufferPointer { v in
                    let count = r.count * shape.elementsPerScanRow
                    staging.contents().copyMemory(from: v.baseAddress! + r.lowerBound * shape.elementsPerScanRow,
                                                  byteCount: count * 4)
                }
            }
        })
        report("upload alone: makeBuffer(bytes:) cube", try await time {
            for r in ranges {
                cube.values.withUnsafeBufferPointer { v in
                    let count = r.count * shape.elementsPerScanRow
                    _ = metal.device.makeBuffer(bytes: v.baseAddress! + r.lowerBound * shape.elementsPerScanRow,
                                                length: count * 4, options: .storageModeShared)
                }
            }
        })
        let sums = try metal.buffer(floats: 2 * n, label: "bench sums")
        let maxima = try metal.buffer(floats: n, label: "bench maxima")
        var gpu = [Double]()
        report("kernel: sumAndMax on staging", try await time {
            for r in ranges {
                let positions = r.count * shape.scanColumns
                try await metal.run(metal.sumAndMax, threads: n, buffers: [staging, sums, maxima],
                                    dims: .init(positions: UInt32(positions), detectorRows: UInt32(shape.detectorRows),
                                                detectorColumns: UInt32(shape.detectorColumns)))
            }
        })
        for r in ranges {
            guard let c = metal.queue.makeCommandBuffer(), let e = c.makeComputeCommandEncoder() else { break }
            e.setComputePipelineState(metal.sumAndMax)
            e.setBuffer(staging, offset: 0, index: 0); e.setBuffer(sums, offset: 0, index: 1); e.setBuffer(maxima, offset: 0, index: 2)
            var d = MetalContext.Dims(positions: UInt32(r.count * shape.scanColumns), detectorRows: UInt32(shape.detectorRows),
                                      detectorColumns: UInt32(shape.detectorColumns))
            e.setBytes(&d, length: MemoryLayout<MetalContext.Dims>.stride, index: 3)
            let width = min(metal.sumAndMax.maxTotalThreadsPerThreadgroup, metal.sumAndMax.threadExecutionWidth * 4)
            e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            e.endEncoding()
            c.commit(); await c.completed()
            gpu.append((c.gpuEndTime - c.gpuStartTime) * 1e3)
        }
        print(String(format: "METALBENCH kernel GPU time summed             %7.2f ms", gpu.reduce(0, +)))
        var sum = [Double](repeating: 0, count: n)
        var maximum = [Float](repeating: -.infinity, count: n)
        report("readback + combine x tiles", try await time {
            for _ in ranges {
                let s = MetalContext.floats(sums, count: 2 * n), m = MetalContext.floats(maxima, count: n)
                for i in 0..<n {
                    sum[i] += MetalReducers.combined(s[2 * i], s[2 * i + 1])
                    maximum[i] = (maximum[i].isNaN || m[i].isNaN) ? .nan : Swift.max(maximum[i], m[i])
                }
            }
        })
        let page = Int(getpagesize())
        report("no-copy buffer over the cube", try await time {
            cube.values.withUnsafeBufferPointer { v in
                let start = UInt(bitPattern: v.baseAddress!)
                let aligned = start & ~UInt(page - 1)
                let end = (start + UInt(v.count * 4) + UInt(page - 1)) & ~UInt(page - 1)
                _ = metal.device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(bitPattern: aligned)!, length: Int(end - aligned),
                                            options: .storageModeShared, deallocator: nil)
            }
        })
    }
}
