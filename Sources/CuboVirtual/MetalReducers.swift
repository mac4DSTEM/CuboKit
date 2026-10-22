//
//  MetalReducers.swift
//  The accelerated path (registration 3.2, 3.3): compute kernels compiled at run time from the source below, in Metal's
//  safe math mode (no fast-math reassociation), compensated Float32 sums, Double combination on the host.
//  Held to the CPU path within relative 1e-5 (registration 3.4).
//
//  Four kernels: one thread per scan position (a detector mask, the centre of mass, the origin) or per detector pixel over the scan
//  positions. Every sum is compensated
//  (TwoSum and the fma TwoProduct, Dot2) and each thread returns its sum and its correction, which the host adds in
//  Double (the first version's plain two-level Float32 sums missed the registered 1e-5 on zero-mean data: Gate B,
//  2026-10-15); the tiles are combined in Double; the kernels are compiled from source so the
//  kit needs no .metallib and `swift build` alone builds it.
//

import CuboCore
import Foundation
import Metal

/// The Metal device, queue and the pipelines, built once. Immutable after construction; Metal's device, queue and
/// pipeline objects are thread-safe, which is what the unchecked conformance states.
final class MetalContext: @unchecked Sendable {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let virtualImage: MTLComputePipelineState
    let sumAndMax: MTLComputePipelineState
    let centerOfMass: MTLComputePipelineState
    let measureOrigin: MTLComputePipelineState
    let weightedPattern: MTLComputePipelineState

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
        self.centerOfMass = try pipeline("cubo_center_of_mass")
        self.measureOrigin = try pipeline("cubo_measure_origin")
        self.weightedPattern = try pipeline("cubo_weighted_pattern")
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

    // Adds x to the double-float (hi, lo), renormalised every step (TwoSum, then a fast TwoSum), so lo stays below an
    // ulp of hi and the pair carries about 48 bits: integer sums stay exact up to 2^48. A non-finite sum is kept as it is.
    inline void df_add(thread float &hi, thread float &lo, float x) {
        const float s = hi + x;
        if (!isfinite(s)) { hi = s; lo = 0.0f; return; }
        const float z = s - hi;
        float e = (hi - (s - z)) + (x - z);
        e += lo;
        hi = s + e;
        lo = e - (hi - s);
    }

    // One thread per scan position: Σ wP, Σ (w i) P, Σ (w j) P (registration slice 2, 3.1; Gate B amendment). Each
    // coefficient arrives as a Float32 pair (hi, lo) with hi + lo its Double value; the product with hi is split exactly by
    // fma, the lo part is added rounded, and every term goes into a renormalised double-float accumulator returned as
    // (hi, lo) for the host to add in Double.
    kernel void cubo_center_of_mass(const device float *data [[buffer(0)]],
                                    const device float2 *w [[buffer(1)]],
                                    const device float2 *wi [[buffer(2)]],
                                    const device float2 *wj [[buffer(3)]],
                                    device float2 *out [[buffer(4)]],
                                    constant CuboDims &d [[buffer(5)]],
                                    uint gid [[thread_position_in_grid]])
    {
        if (gid >= d.positions) { return; }
        const ulong n = (ulong)d.detectorRows * d.detectorColumns;
        const device float *p = data + (ulong)gid * n;
        float s0 = 0.0f, c0 = 0.0f, s1 = 0.0f, c1 = 0.0f, s2 = 0.0f, c2 = 0.0f;
        for (ulong k = 0; k < n; ++k) {
            const float x = p[k];
            float h = w[k].x * x;   df_add(s0, c0, h); df_add(s0, c0, fma(w[k].x, x, -h));  df_add(s0, c0, w[k].y * x);
            h = wi[k].x * x;        df_add(s1, c1, h); df_add(s1, c1, fma(wi[k].x, x, -h)); df_add(s1, c1, wi[k].y * x);
            h = wj[k].x * x;        df_add(s2, c2, h); df_add(s2, c2, fma(wj[k].x, x, -h)); df_add(s2, c2, wj[k].y * x);
        }
        out[3 * gid] = float2(s0, c0);
        out[3 * gid + 1] = float2(s1, c1);
        out[3 * gid + 2] = float2(s2, c2);
    }

    // Mirrors `MetalContext.OriginParams`: eight 4-byte fields.
    struct CuboOriginParams {
        uint positions; uint detectorRows; uint detectorColumns; uint passes; uint binned;
        float radius; float radiusScale; float margin;
    };

    // Non-finite test on the bits, so no math mode can fold it away. A non-finite pixel reads as the median of its
    // finite 8 neighbours of the original pattern (0 when none), the CPU reference's rule.
    inline bool cubo_non_finite(float v) { return (as_type<uint>(v) & 0x7F800000u) == 0x7F800000u; }

    inline float cubo_filled(const device float *pat, int row, int col, uint qy, uint qx) {
        const float v = pat[uint(row) * qx + uint(col)];
        if (!cubo_non_finite(v)) { return v; }
        float nb[8];
        int k = 0;
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (dr == 0 && dc == 0) { continue; }
                const int rr = row + dr, cc = col + dc;
                if (rr < 0 || rr >= int(qy) || cc < 0 || cc >= int(qx)) { continue; }
                const float w = pat[uint(rr) * qx + uint(cc)];
                if (cubo_non_finite(w)) { continue; }
                int i = k++;
                while (i > 0 && nb[i - 1] > w) { nb[i] = nb[i - 1]; --i; }
                nb[i] = w;
            }
        }
        if (k == 0) { return 0.0f; }
        return (k % 2 == 1) ? nb[k / 2] : 0.5f * (nb[k / 2 - 1] + nb[k / 2]);
    }

    // One thread per scan position: the iterated-window origin (IteratedWindow.origin is the same algorithm operation by
    // operation; the multiply-adds are explicit fma on both sides). Returns (row, column).
    kernel void cubo_measure_origin(const device float *data [[buffer(0)]],
                                    device float2 *out [[buffer(1)]],
                                    constant CuboOriginParams &p [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]])
    {
        if (gid >= p.positions) { return; }
        const uint qy = p.detectorRows, qx = p.detectorColumns;
        const device float *pat = data + (ulong)gid * ((ulong)qy * qx);
        const uint bin = p.binned != 0u ? max(1u, uint(round(p.radius))) : 1u;
        float bestSum = -FLT_MAX;
        float coarseR = 0.5f * float(qy - 1), coarseC = 0.5f * float(qx - 1);
        for (uint by = 0; by < qy; by += bin) {
            const uint yEnd = min(by + bin, qy);
            for (uint bx = 0; bx < qx; bx += bin) {
                const uint xEnd = min(bx + bin, qx);
                float s = 0.0f;
                for (uint y = by; y < yEnd; ++y) {
                    for (uint x = bx; x < xEnd; ++x) { s += cubo_filled(pat, int(y), int(x), qy, qx); }
                }
                if (s > bestSum) {
                    bestSum = s;
                    coarseC = 0.5f * float(bx + xEnd - 1);
                    coarseR = 0.5f * float(by + yEnd - 1);
                }
            }
        }
        const float win = max(p.radius * p.radiusScale, p.radius + p.margin);
        const float win2 = win * win;
        float cr = coarseR, cc = coarseC;
        for (uint pass = 0; pass < p.passes; ++pass) {
            const int y0 = max(0, int(floor(cr - win)));
            const int y1 = min(int(qy) - 1, int(ceil(cr + win)));
            const int x0 = max(0, int(floor(cc - win)));
            const int x1 = min(int(qx) - 1, int(ceil(cc + win)));
            float sumI = 0.0f, sumIX = 0.0f, sumIY = 0.0f;
            for (int y = y0; y <= y1; ++y) {
                const float dy = float(y) - cr;
                const float dy2 = dy * dy;
                if (dy2 > win2) { continue; }
                for (int x = x0; x <= x1; ++x) {
                    const float dx = float(x) - cc;
                    if (fma(dx, dx, dy2) > win2) { continue; }
                    const float v = max(cubo_filled(pat, y, x, qy, qx), 0.0f);
                    sumI += v;
                    sumIX = fma(v, float(x), sumIX);
                    sumIY = fma(v, float(y), sumIY);
                }
            }
            if (!(sumI > 0.0f)) { break; }
            const float nc = sumIX / sumI, nr = sumIY / sumI;
            const bool settled = fabs(nc - cc) < 1e-4f && fabs(nr - cr) < 1e-4f;
            cc = nc; cr = nr;
            if (settled) { break; }
        }
        out[gid] = float2(cr, cc);
    }

    struct CuboWeightedDims { uint positions; uint detectorRows; uint detectorColumns; uint weighted; };

    // One thread per detector pixel over the tile's positions in scan order (selected-area diffraction, py4DSTEM's
    // get_virtual_diffraction with a real-space mask). weights[k] is the position's weight; with weighted == 0 it is a mask
    // (0 = not selected, skipped; anything else = selected, weight 1). The weighted sum is compensated with the fma
    // TwoProduct, as cubo_virtual_image; the maximum is of v * w, seeded at -infinity, NaN-propagating, as the CPU path.
    kernel void cubo_weighted_pattern(const device float *data [[buffer(0)]],
                                      const device float *weights [[buffer(1)]],
                                      device float2 *sums [[buffer(2)]],
                                      device float *maxima [[buffer(3)]],
                                      constant CuboWeightedDims &d [[buffer(4)]],
                                      uint gid [[thread_position_in_grid]])
    {
        const ulong n = (ulong)d.detectorRows * d.detectorColumns;
        if ((ulong)gid >= n) { return; }
        float s = 0.0f, c = 0.0f;
        float vmax = -INFINITY;
        for (uint k = 0; k < d.positions; ++k) {
            const float w = weights[k];
            const float x = data[(ulong)k * n + gid];
            float product = x;
            if (d.weighted != 0) {
                product = x * w;
                const float error = fma(x, w, -product);
                two_sum(s, c, product);
                c += error;
            } else {
                if (w == 0.0f) { continue; }
                two_sum(s, c, x);
            }
            vmax = (isnan(vmax) || isnan(product)) ? NAN : max(vmax, product);
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
    func run(_ pipeline: MTLComputePipelineState, threads: Int, buffers: [MTLBuffer], offsets: [Int] = [],
             dims: Dims) async throws {
        try await run(pipeline, threads: threads, buffers: buffers, offsets: offsets, constants: dims)
    }

    /// Mirrors `struct CuboOriginParams` in the source: eight 4-byte fields.
    struct OriginParams {
        var positions: UInt32
        var detectorRows: UInt32
        var detectorColumns: UInt32
        var passes: UInt32
        var binned: UInt32
        var radius: Float
        var radiusScale: Float
        var margin: Float
    }

    /// The same dispatch with any plain-old-data constants block after the buffers.
    func run<Constants: BitwiseCopyable>(_ pipeline: MTLComputePipelineState, threads: Int, buffers: [MTLBuffer],
                        offsets: [Int] = [], constants: Constants) async throws {
        guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            throw CuboVirtualError.metal("no command buffer")
        }
        encoder.setComputePipelineState(pipeline)
        for (index, b) in buffers.enumerated() { encoder.setBuffer(b, offset: index < offsets.count ? offsets[index] : 0, index: index) }
        var d = constants
        encoder.setBytes(&d, length: MemoryLayout<Constants>.stride, index: buffers.count)
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

    /// The Double coefficients as Float32 (hi, lo) pairs, interleaved: hi = Float(v), lo = Float(v − hi).
    static func split(_ v: [Double]) -> [Float] {
        var out = [Float](); out.reserveCapacity(2 * v.count)
        for x in v { let hi = Float(x); out.append(hi); out.append(Float(x - Double(hi))) }
        return out
    }

    static func centerOfMassSums(of source: some CubeSource, weights: [Float], tiling: ScanTiling,
                                 progress: (@Sendable (Double) -> Void)?) async throws
        -> (s: [Double], r: [Double], c: [Double]) {
        let metal = try context()
        let shape = source.shape
        let n = shape.detectorCount, qx = shape.detectorColumns
        let w = weights.map(Double.init)
        let wBuffer = try metal.buffer(split(w), label: "CuboKit CoM weights")
        let wiBuffer = try metal.buffer(split((0..<n).map { w[$0] * Double($0 / qx) }), label: "CuboKit CoM row weights")
        let wjBuffer = try metal.buffer(split((0..<n).map { w[$0] * Double($0 % qx) }), label: "CuboKit CoM column weights")
        var s = [Double](repeating: 0, count: shape.scanCount)
        var r = s, c = s
        for range in try tiling.ranges(for: shape) {
            try checkInterrupted()
            let (data, offset, positions) = try await tileBuffer(metal, source, range)
            let out = try metal.buffer(floats: 6 * positions, label: "CuboKit CoM sums")
            try await metal.run(metal.centerOfMass, threads: positions, buffers: [data, wBuffer, wiBuffer, wjBuffer, out],
                                offsets: [offset],
                                dims: .init(positions: UInt32(positions), detectorRows: UInt32(shape.detectorRows),
                                            detectorColumns: UInt32(shape.detectorColumns)))
            let v = MetalContext.floats(out, count: 6 * positions)
            let first = range.lowerBound * shape.scanColumns
            for k in 0..<positions {
                s[first + k] = Double(v[6 * k]) + Double(v[6 * k + 1])
                r[first + k] = Double(v[6 * k + 2]) + Double(v[6 * k + 3])
                c[first + k] = Double(v[6 * k + 4]) + Double(v[6 * k + 5])
            }
            progress?(Double(range.upperBound) / Double(shape.scanRows))
        }
        return (s, r, c)
    }

    /// The iterated-window origins of one tile, (row, column) per position, from `cubo_measure_origin`.
    static func iteratedWindowOrigins(_ metal: MetalContext, tile: [Float], positions: Int, shape: CubeShape, radius: Float,
                                      radiusScale: Float, passes: Int, margin: Float, binned: Bool) async throws -> [(Float, Float)] {
        let data = try metal.buffer(tile, label: "CuboKit origin tile")
        let out = try metal.buffer(floats: 2 * positions, label: "CuboKit origins (row, column)")
        try await metal.run(metal.measureOrigin, threads: positions, buffers: [data, out],
                            constants: MetalContext.OriginParams(
                                positions: UInt32(positions), detectorRows: UInt32(shape.detectorRows),
                                detectorColumns: UInt32(shape.detectorColumns), passes: UInt32(passes),
                                binned: binned ? 1 : 0, radius: radius, radiusScale: radiusScale, margin: margin))
        let v = MetalContext.floats(out, count: 2 * positions)
        return (0..<positions).map { (v[2 * $0], v[2 * $0 + 1]) }
    }

    static func virtualImage(of source: some CubeSource, mask: [Float], tiling: ScanTiling,
                             progress: (@Sendable (Double) -> Void)?) async throws -> ScanImage {
        let metal = try context()
        let shape = source.shape
        let maskBuffer = try metal.buffer(mask, label: "CuboKit detector mask")
        var out = [Double](repeating: 0, count: shape.scanCount)
        for range in try tiling.ranges(for: shape) {
            try checkInterrupted()
            let (data, offset, positions) = try await tileBuffer(metal, source, range)
            let image = try metal.buffer(floats: 2 * positions, label: "CuboKit virtual image tile (sum, correction)")
            try await metal.run(metal.virtualImage, threads: positions, buffers: [data, maskBuffer, image], offsets: [offset],
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
            try checkInterrupted()
            let (data, offset, positions) = try await tileBuffer(metal, source, range)
            try await metal.run(metal.sumAndMax, threads: n, buffers: [data, sums, maxima], offsets: [offset],
                                dims: .init(positions: UInt32(positions),
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

    /// The Float32 values of the scan rows `range` as a Metal buffer: the cube's own buffer at an offset when it is a
    /// ``ResidentCube`` (no copy), else a buffer filled from the source's tile.
    static func tileBuffer(_ metal: MetalContext, _ source: some CubeSource, _ range: Range<Int>) async throws
        -> (buffer: MTLBuffer, offset: Int, positions: Int) {
        let shape = source.shape
        if let resident = source as? ResidentCube {
            return (resident.storage.buffer, range.lowerBound * shape.elementsPerScanRow * MemoryLayout<Float>.stride,
                    range.count * shape.scanColumns)
        }
        let tile = try await source.readScanTile(rows: range)
        return (try metal.buffer(tile.values, label: "CuboKit tile rows \(range.lowerBound)..<\(range.upperBound)"), 0,
                tile.positionCount)
    }

    /// Mean or maximum of the patterns at the positions in `take` (`weights` multiplies each pattern when given): one
    /// thread per detector pixel, compensated Float32 sums per tile combined in Double.
    static func weightedPattern(of source: some CubeSource, statistic: DiffractionStatistic, take: [Bool], weights: [Double]?,
                                count: Double, tiling: ScanTiling, progress: (@Sendable (Double) -> Void)?) async throws
        -> DetectorImage {
        let metal = try context()
        let shape = source.shape
        let n = shape.detectorCount
        var sum = [Double](repeating: 0, count: n)
        var maximum = [Float](repeating: -.infinity, count: n)
        let sums = try metal.buffer(floats: 2 * n, label: "CuboKit selected-area sums (sum, correction)")
        let maxima = try metal.buffer(floats: n, label: "CuboKit selected-area maxima")
        for range in try tiling.ranges(for: shape) {
            try checkInterrupted()
            let first = range.lowerBound * shape.scanColumns
            let positions = range.count * shape.scanColumns
            guard (0..<positions).contains(where: { take[first + $0] }) else {
                progress?(Double(range.upperBound) / Double(shape.scanRows))
                continue
            }
            let w: [Float] = (0..<positions).map { k in
                take[first + k] ? (weights.map { Float($0[first + k]) } ?? 1) : 0
            }
            let weightBuffer = try metal.buffer(w, label: "CuboKit selection weights")
            let (data, offset, _) = try await tileBuffer(metal, source, range)
            try await metal.run(metal.weightedPattern, threads: n, buffers: [data, weightBuffer, sums, maxima],
                                offsets: [offset],
                                constants: WeightedDims(positions: UInt32(positions), detectorRows: UInt32(shape.detectorRows),
                                                        detectorColumns: UInt32(shape.detectorColumns),
                                                        weighted: weights == nil ? 0 : 1))
            let s = MetalContext.floats(sums, count: 2 * n), m = MetalContext.floats(maxima, count: n)
            for i in 0..<n {
                sum[i] += Double(s[2 * i]) + Double(s[2 * i + 1])
                maximum[i] = (maximum[i].isNaN || m[i].isNaN) ? .nan : Swift.max(maximum[i], m[i])
            }
            progress?(Double(range.upperBound) / Double(shape.scanRows))
        }
        return DetectorImage(detectorRows: shape.detectorRows, detectorColumns: shape.detectorColumns,
                             values: statistic == .mean ? sum.map { $0 / count } : maximum.map(Double.init), path: .metal)
    }

    /// Mirrors `struct CuboWeightedDims` in the source.
    struct WeightedDims: BitwiseCopyable {
        var positions: UInt32
        var detectorRows: UInt32
        var detectorColumns: UInt32
        var weighted: UInt32
    }
}
