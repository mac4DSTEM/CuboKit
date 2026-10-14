//
//  MetalReducers.swift
//  The accelerated path (registration 3.2, 3.3): compute kernels compiled at run time from the source below, in Metal's
//  safe math mode (no fast-math reassociation), compensated Float32 sums, Double combination on the host.
//  Held to the CPU path within relative 1e-5 (registration 3.4).
//
//  Adapted from mac4DSTEM 7448401b, Shaders/VirtualMask.metal (one thread per scan position over a detector mask) and
//  Shaders/DPStatistics.metal (one thread per detector pixel over the scan positions). Changed: every sum is compensated
//  (TwoSum and the fma TwoProduct, Dot2) and each thread returns its sum and its correction, which the host adds in
//  Double (the first version's plain two-level Float32 sums missed the registered 1e-5 on zero-mean data: Gate B,
//  2026-10-15); the tiles are combined in Double; the kernels are compiled from source so the
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

    // Error-free transformations (Knuth's TwoSum, the fma TwoProduct): in safe math mode nothing reassociates them, so
    // s + c holds the running sum with its rounding error carried exactly to the next step (Ogita, Rump and Oishi's
    // Dot2). The host adds s and c in Double. Registration 3.2, 3.3, amendment of 2026-10-15 (Gate B).
    inline void two_sum(thread float &s, thread float &c, float x) {
        const float t = s + x;
        const float z = t - s;
        c += (s - (t - z)) + (x - z);
        s = t;
    }

    // One thread per scan position: the compensated dot product of the pattern with the mask, every pixel (a zero
    // weight times a NaN is NaN, as on the CPU path).
    kernel void cubo_virtual_image(const device float *data [[buffer(0)]],
                                   const device float *mask [[buffer(1)]],
                                   device float2 *image [[buffer(2)]],
                                   constant CuboDims &d [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]])
    {
        if (gid >= d.positions) { return; }
        const ulong n = (ulong)d.detectorRows * d.detectorColumns;
        const device float *p = data + (ulong)gid * n;
        float s = 0.0f, c = 0.0f;
        for (ulong k = 0; k < n; ++k) {
            const float m = mask[k];
            const float x = p[k];
            const float product = m * x;
            const float error = fma(m, x, -product);
            two_sum(s, c, product);
            c += error;
        }
        image[gid] = float2(s, c);
    }

    // One thread per detector pixel: the tile's positions in scan order, compensated; the maximum seeded at -infinity
    // and NaN-propagating, as the CPU path.
    kernel void cubo_sum_and_max(const device float *data [[buffer(0)]],
                                 device float2 *sums [[buffer(1)]],
                                 device float *maxima [[buffer(2)]],
                                 constant CuboDims &d [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]])
    {
        const ulong n = (ulong)d.detectorRows * d.detectorColumns;
        if ((ulong)gid >= n) { return; }
        float s = 0.0f, c = 0.0f;
        float vmax = -INFINITY;
        for (uint k = 0; k < d.positions; ++k) {
            const float v = data[(ulong)k * n + gid];
            two_sum(s, c, v);
            vmax = (isnan(vmax) || isnan(v)) ? NAN : max(vmax, v);
        }
        sums[gid] = float2(s, c);
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
            let image = try metal.buffer(floats: 2 * positions, label: "CuboKit virtual image tile (sum, correction)")
            try await metal.run(metal.virtualImage, threads: positions, buffers: [data, maskBuffer, image],
                                dims: .init(positions: UInt32(positions), detectorRows: UInt32(shape.detectorRows),
                                            detectorColumns: UInt32(shape.detectorColumns)))
            let first = range.lowerBound * shape.scanColumns
            let pairs = MetalContext.floats(image, count: 2 * positions)
            for k in 0..<positions { out[first + k] = Double(pairs[2 * k]) + Double(pairs[2 * k + 1]) }
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
        let sums = try metal.buffer(floats: 2 * n, label: "CuboKit tile sums (sum, correction)")
        let maxima = try metal.buffer(floats: n, label: "CuboKit tile maxima")
        for range in try tiling.ranges(for: shape) {
            try Task.checkCancellation()
            let tile = try await source.readScanTile(rows: range)
            let data = try metal.buffer(tile.values, label: "CuboKit tile rows \(range.lowerBound)..<\(range.upperBound)")
            try await metal.run(metal.sumAndMax, threads: n, buffers: [data, sums, maxima],
                                dims: .init(positions: UInt32(tile.positionCount),
                                            detectorRows: UInt32(shape.detectorRows),
                                            detectorColumns: UInt32(shape.detectorColumns)))
            let s = MetalContext.floats(sums, count: 2 * n), m = MetalContext.floats(maxima, count: n)
            for i in 0..<n {
                sum[i] += Double(s[2 * i]) + Double(s[2 * i + 1])
                maximum[i] = (maximum[i].isNaN || m[i].isNaN) ? .nan : Swift.max(maximum[i], m[i])
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
