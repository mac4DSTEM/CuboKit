//
//  MetalReducers.swift
//  The accelerated path (registration 3.2, 3.3): compute kernels compiled at run time from the source below, in Metal's
//  safe math mode (no fast-math reassociation), Float32 accumulation, Double combination across tiles on the host.
//  Held to the CPU path within relative 1e-5 (registration 3.4).
//
//  Adapted from mac4DSTEM 7448401b, Shaders/VirtualMask.metal (one thread per scan position over a detector mask) and
//  Shaders/DPStatistics.metal (one thread per detector pixel over the scan positions). Changed: the position sum is two
//  levels deep (detector rows, then the row sums; chunks of 64 positions, then the chunk sums), so the Float32 error
//  grows with the side, not the area; the tiles are combined in Double; the kernels are compiled from source so the
//  kit needs no .metallib and `swift build` alone builds it.
//

import CuboCore
import Foundation
import Metal

/// The Metal device, queue and the two pipelines, built once. Immutable after construction; Metal's device, queue and
/// pipeline objects are thread-safe, which is what the unchecked conformance states.
final class MetalContext: @unchecked Sendable {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let virtualImage: MTLComputePipelineState
    let sumAndMax: MTLComputePipelineState

    static let shared: Result<MetalContext, CuboVirtualError> = {
        guard let device = MTLCreateSystemDefaultDevice() else { return .failure(.noMetalDevice) }
        do { return .success(try MetalContext(device: device)) } catch let e as CuboVirtualError {
            return .failure(e)
        } catch { return .failure(.metal(error.localizedDescription)) }
    }()

    private init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw CuboVirtualError.metal("no command queue") }
        self.queue = queue
        let options = MTLCompileOptions()
        options.mathMode = .safe
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: Self.source, options: options) } catch {
            throw CuboVirtualError.metal("compiling the kernels: \(error.localizedDescription)")
        }
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw CuboVirtualError.metal("no function \(name)") }
            do { return try device.makeComputePipelineState(function: f) } catch {
                throw CuboVirtualError.metal("pipeline \(name): \(error.localizedDescription)")
            }
        }
        self.virtualImage = try pipeline("cubo_virtual_image")
        self.sumAndMax = try pipeline("cubo_sum_and_max")
    }

    /// Mirrors `struct CuboDims` in the source: four 4-byte fields.
    struct Dims {
        var positions: UInt32
        var detectorRows: UInt32
        var detectorColumns: UInt32
        var unused: UInt32 = 0
    }

    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct CuboDims { uint positions; uint detectorRows; uint detectorColumns; uint unused; };

    // One thread per scan position: each detector row summed, then the row sums (registration 3.2).
    kernel void cubo_virtual_image(const device float *data [[buffer(0)]],
                                   const device float *mask [[buffer(1)]],
                                   device float *image [[buffer(2)]],
                                   constant CuboDims &d [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]])
    {
        if (gid >= d.positions) { return; }
        const ulong n = (ulong)d.detectorRows * d.detectorColumns;
        const device float *p = data + (ulong)gid * n;
        float total = 0.0f;
        for (uint i = 0; i < d.detectorRows; ++i) {
            const ulong base = (ulong)i * d.detectorColumns;
            float row = 0.0f;
            for (uint j = 0; j < d.detectorColumns; ++j) {
                const float m = mask[base + j];
                if (m != 0.0f) { row += m * p[base + j]; }
            }
            total += row;
        }
        image[gid] = total;
    }

    // One thread per detector pixel: the tile's positions in chunks of 64 in scan order, then the chunk sums; the
    // maximum seeded at -infinity (registration 3.3).
    kernel void cubo_sum_and_max(const device float *data [[buffer(0)]],
                                 device float *sums [[buffer(1)]],
                                 device float *maxima [[buffer(2)]],
                                 constant CuboDims &d [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]])
    {
        const ulong n = (ulong)d.detectorRows * d.detectorColumns;
        if ((ulong)gid >= n) { return; }
        float total = 0.0f;
        float vmax = -INFINITY;
        for (uint start = 0; start < d.positions; start += 64) {
            const uint end = min(start + 64u, d.positions);
            float chunk = 0.0f;
            for (uint s = start; s < end; ++s) {
                const float v = data[(ulong)s * n + gid];
                chunk += v;
                vmax = max(vmax, v);
            }
            total += chunk;
        }
        sums[gid] = total;
        maxima[gid] = vmax;
    }
    """

    func buffer(_ values: [Float], label: String) throws -> MTLBuffer {
        guard let b = values.withUnsafeBytes({ raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: max(raw.count, 4), options: .storageModeShared)
        }) else { throw CuboVirtualError.metal("allocating \(label) (\(values.count) floats)") }
        b.label = label
        return b
    }

    func buffer(floats count: Int, label: String) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: max(count, 1) * MemoryLayout<Float>.stride,
                                        options: .storageModeShared) else {
            throw CuboVirtualError.metal("allocating \(label) (\(count) floats)")
        }
        b.label = label
        return b
    }

    /// Encodes one dispatch of `threads` threads and waits for it without blocking a cooperative thread.
    func run(_ pipeline: MTLComputePipelineState, threads: Int, buffers: [MTLBuffer], dims: Dims) async throws {
        guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            throw CuboVirtualError.metal("no command buffer")
        }
        encoder.setComputePipelineState(pipeline)
        for (index, b) in buffers.enumerated() { encoder.setBuffer(b, offset: 0, index: index) }
        var d = dims
        encoder.setBytes(&d, length: MemoryLayout<Dims>.stride, index: buffers.count)
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, max(1, pipeline.threadExecutionWidth * 4))
        encoder.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
            commands.addCompletedHandler { done in
                if let error = done.error {
                    continuation.resume(throwing: CuboVirtualError.metal(error.localizedDescription))
                } else if done.status != .completed {
                    continuation.resume(throwing: CuboVirtualError.metal("command buffer status \(done.status.rawValue)"))
                } else {
                    continuation.resume()
                }
            }
            commands.commit()
        }
    }

    static func floats(_ b: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: Float.self, capacity: count), count: count))
    }
}

enum MetalReducers {
    static func context() throws -> MetalContext { try MetalContext.shared.get() }

    static func virtualImage(of source: some CubeSource, mask: [Float], tiling: ScanTiling,
                             progress: (@Sendable (Double) -> Void)?) async throws -> ScanImage {
        let metal = try context()
        let shape = source.shape
        let maskBuffer = try metal.buffer(mask, label: "CuboKit detector mask")
        var out = [Double](repeating: 0, count: shape.scanCount)
        for range in try tiling.ranges(for: shape) {
            try Task.checkCancellation()
            let tile = try await source.readScanTile(rows: range)
            let positions = tile.positionCount
            let data = try metal.buffer(tile.values, label: "CuboKit tile rows \(range.lowerBound)..<\(range.upperBound)")
            let image = try metal.buffer(floats: positions, label: "CuboKit virtual image tile")
            try await metal.run(metal.virtualImage, threads: positions, buffers: [data, maskBuffer, image],
                                dims: .init(positions: UInt32(positions), detectorRows: UInt32(shape.detectorRows),
                                            detectorColumns: UInt32(shape.detectorColumns)))
            let first = range.lowerBound * shape.scanColumns
            for (k, v) in MetalContext.floats(image, count: positions).enumerated() { out[first + k] = Double(v) }
            progress?(Double(range.upperBound) / Double(shape.scanRows))
        }
        return ScanImage(scanRows: shape.scanRows, scanColumns: shape.scanColumns, values: out, path: .metal)
    }

    static func meanAndMax(of source: some CubeSource, tiling: ScanTiling,
                           progress: (@Sendable (Double) -> Void)?) async throws -> (DetectorImage, DetectorImage) {
        let metal = try context()
        let shape = source.shape
        let n = shape.detectorCount
        var sum = [Double](repeating: 0, count: n)
        var maximum = [Float](repeating: -.infinity, count: n)
        let sums = try metal.buffer(floats: n, label: "CuboKit tile sums")
        let maxima = try metal.buffer(floats: n, label: "CuboKit tile maxima")
        for range in try tiling.ranges(for: shape) {
            try Task.checkCancellation()
            let tile = try await source.readScanTile(rows: range)
            let data = try metal.buffer(tile.values, label: "CuboKit tile rows \(range.lowerBound)..<\(range.upperBound)")
            try await metal.run(metal.sumAndMax, threads: n, buffers: [data, sums, maxima],
                                dims: .init(positions: UInt32(tile.positionCount),
                                            detectorRows: UInt32(shape.detectorRows),
                                            detectorColumns: UInt32(shape.detectorColumns)))
            let s = MetalContext.floats(sums, count: n), m = MetalContext.floats(maxima, count: n)
            for i in 0..<n {
                sum[i] += Double(s[i])
                maximum[i] = Swift.max(maximum[i], m[i])
            }
            progress?(Double(range.upperBound) / Double(shape.scanRows))
        }
        let count = Double(shape.scanCount)
        return (DetectorImage(detectorRows: shape.detectorRows, detectorColumns: shape.detectorColumns,
                              values: sum.map { $0 / count }, path: .metal),
                DetectorImage(detectorRows: shape.detectorRows, detectorColumns: shape.detectorColumns,
                              values: maximum.map(Double.init), path: .metal))
    }
}
