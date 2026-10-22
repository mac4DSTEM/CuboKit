//
//  SelectedDiffraction.swift
//  The median pattern and virtual diffraction over a real-space selection (registration slice 4, 3.1): py4DSTEM's
//  get_dp_median and get_virtual_diffraction(method, mask, shift_center, subpixel) at f050d207, CPU, Double.
//
//  DEVIATIONS: an empty selection, weights summing to zero (for the mean) and non-finite weights are refused where
//  py4DSTEM returns NaN or ∞ or raises; the shifted maximum is seeded at −∞ (slice 3; py4DSTEM seeds it at 0). Complex
//  masks are not ported.
//

import CuboCore
import Foundation

/// How the patterns of a selection are combined (py4DSTEM's `method`).
public enum DiffractionStatistic: String, Sendable, Hashable, CaseIterable {
    case mean, max, median
}

/// Which scan positions a virtual diffraction pattern combines, row-major over the scan (py4DSTEM's real-space `mask`).
public enum ScanSelection: Sendable, Hashable {
    /// Every position.
    case all
    /// The positions where the mask is true (py4DSTEM's boolean mask).
    case mask([Bool])
    /// Every position, weighted (py4DSTEM's floating-point mask): the mean is Σ P·w / Σ w, the max and median are of P·w.
    case weights([Double])
}

extension VirtualDiffraction {
    /// The median pattern over every position (py4DSTEM `get_dp_median`): per pixel the middle value, an even count the
    /// mean of the two middle values in Double, NaN where any value is NaN.
    public static func median(of source: some CubeSource, tiling: ScanTiling = .automatic) async throws -> DetectorImage {
        try await pattern(of: source, statistic: .median, scan: .all, tiling: tiling)
    }

    /// A virtual diffraction pattern over a selection of positions, optionally centre-shifted (mean and max only).
    /// `.mean` and `.max` over `.all` without a shift are slice 1's `mean` and `max` on the CPU, bit for bit.
    /// `medianWorkingBytes` bounds the median's gathered block (256 MiB by default; it changes the passes, not the result).
    /// `path: .metal` runs the mean and the max over `.all`, `.mask` and `.weights` on the GPU (compensated Float32 sums
    /// combined in Double, held to the CPU path within relative 1e-5); the median and a centre shift are CPU only and
    /// throw with `.metal`. A ``ResidentCube`` is read in place.
    public static func pattern(of source: some CubeSource, statistic: DiffractionStatistic, scan: ScanSelection = .all,
                               shift: PatternShift? = nil, path: ComputePath = .cpu, tiling: ScanTiling = .automatic,
                               medianWorkingBytes: Int = 256 << 20) async throws -> DetectorImage {
        guard medianWorkingBytes > 0 else { throw CuboVirtualError.invalidInput("the median's working memory must be positive") }
        let shape = source.shape
        let weights: [Double]?
        let selected: [Int]
        switch scan {
        case .all:
            weights = nil; selected = Array(0..<shape.scanCount)
        case .mask(let m):
            guard m.count == shape.scanCount else {
                throw CuboVirtualError.maskLengthMismatch(expected: shape.scanCount, received: m.count)
            }
            weights = nil; selected = m.indices.filter { m[$0] }
        case .weights(let w):
            guard w.count == shape.scanCount else {
                throw CuboVirtualError.maskLengthMismatch(expected: shape.scanCount, received: w.count)
            }
            guard w.allSatisfy(\.isFinite) else { throw CuboVirtualError.invalidInput("the weights must be finite") }
            weights = w; selected = Array(0..<shape.scanCount)
        }
        guard !selected.isEmpty else { throw CuboVirtualError.invalidInput("the selection holds no scan position") }
        if path == .metal, shift != nil || statistic == .median {
            throw CuboVirtualError.invalidInput("the Metal path covers the mean and the max over a selection without a centre shift")
        }
        if let shift {
            guard statistic != .median else {
                throw CuboVirtualError.invalidInput("a centre-shifted median is not defined (py4DSTEM: mean and max only)")
            }
            return try await shifted(source, statistic, selected, weights, shift, tiling)
        }
        if case .all = scan, statistic != .median {
            return statistic == .mean ? try await mean(of: source, path: path, tiling: tiling)
                : try await max(of: source, path: path, tiling: tiling)
        }
        if statistic == .median { return try await median(source, selected, weights, tiling, medianWorkingBytes) }
        if path == .metal {
            var take = [Bool](repeating: false, count: shape.scanCount)
            for k in selected { take[k] = true }
            var count = Double(selected.count)
            if let weights, statistic == .mean {
                var total = Neumaier()
                for k in selected { total.add(weights[k]) }
                count = total.value
                guard count != 0 else { throw CuboVirtualError.invalidInput("the weights sum to zero") }
            }
            return try await MetalReducers.weightedPattern(of: source, statistic: statistic, take: take, weights: weights,
                                                           count: count, tiling: tiling, progress: nil)
        }
        return try await reduce(source, statistic, selected, weights, tiling)
    }

    /// Mean and max over a mask or weights, in scan order: compensated sums per pixel, the maximum with NaN winning.
    private static func reduce(_ source: some CubeSource, _ statistic: DiffractionStatistic, _ selected: [Int],
                               _ weights: [Double]?, _ tiling: ScanTiling) async throws -> DetectorImage {
        let shape = source.shape
        let n = shape.detectorCount
        var take = [Bool](repeating: false, count: shape.scanCount)
        for k in selected { take[k] = true }
        // Mean: a compensated sum per pixel (SIMD TwoSum, ``CompensatedSum``), the detector pixels in blocks on every core, each
        // pixel added in scan order; the sums start at +0 as the scalar Neumaier did.
        var sum = [Double](repeating: 0, count: statistic == .mean ? n : 0)
        var correction = [Double](repeating: 0, count: statistic == .mean ? n : 0)
        var maximum = [Double](repeating: -.infinity, count: statistic == .mean ? 0 : n)
        var total = Neumaier()
        let blockSize = 512
        let blocks = (n + blockSize - 1) / blockSize
        for range in try tiling.ranges(for: shape) {
            try Task.checkCancellation()
            let first = range.lowerBound * shape.scanColumns
            let picked = (0..<(range.count * shape.scanColumns)).filter { take[first + $0] }
            for k in picked { total.add(weights?[first + k] ?? 1.0) }
            try await withTileValues(of: source, rows: range) { values in
                if statistic == .mean {
                    sum.withUnsafeMutableBufferPointer { s in
                        correction.withUnsafeMutableBufferPointer { c in
                            let sp = SharedBuffer(base: s.baseAddress!), cp = SharedBuffer(base: c.baseAddress!)
                            let vp = SharedBuffer(base: UnsafeMutablePointer(mutating: values))
                            DispatchQueue.concurrentPerform(iterations: blocks) { b in
                                let lo = b * blockSize, hi = Swift.min(n, lo + blockSize)
                                for k in picked {
                                    CompensatedSum.add(vp.base + k * n + lo, weight: weights.map { $0[first + k] },
                                                       sum: sp.base + lo, correction: cp.base + lo, count: hi - lo)
                                }
                            }
                        }
                    }
                } else {
                    for k in picked {
                        let w = weights?[first + k] ?? 1.0
                        for i in 0..<n {
                            let v = weights == nil ? Double(values[k * n + i]) : Double(values[k * n + i]) * w
                            let m = maximum[i]    // np.max: NaN wins; of equal values (±0) the first is kept
                            maximum[i] = (m.isNaN || v.isNaN) ? .nan : (v > m ? v : m)
                        }
                    }
                }
            }
        }
        let values: [Double]
        if statistic == .mean {
            let count = weights == nil ? Double(selected.count) : total.value
            guard count != 0 else { throw CuboVirtualError.invalidInput("the weights sum to zero") }
            values = (0..<n).map { CompensatedSum.value(sum: sum[$0], correction: correction[$0]) / count }
        } else {
            values = maximum
        }
        return DetectorImage(detectorRows: shape.detectorRows, detectorColumns: shape.detectorColumns, values: values,
                             path: .cpu)
    }

    /// The median per pixel over the selected positions (of P·w with weights). Detector pixels in blocks that fit the
    /// working memory, each block gathered from every tile and sorted per pixel.
    private static func median(_ source: some CubeSource, _ selected: [Int], _ weights: [Double]?,
                               _ tiling: ScanTiling, _ workingBytes: Int) async throws -> DetectorImage {
        let shape = source.shape
        let n = shape.detectorCount, count = selected.count
        var slot = [Int](repeating: -1, count: shape.scanCount)
        for (s, k) in selected.enumerated() { slot[k] = s }
        let block = Swift.max(1, Swift.min(n, workingBytes / (8 * count)))
        var out = [Double](repeating: 0, count: n)
        let ranges = try tiling.ranges(for: shape)
        var start = 0
        while start < n {
            let end = Swift.min(start + block, n), width = end - start
            var gathered = [Double](repeating: 0, count: width * count)   // pixel-major: [pixel][position]
            for range in ranges {
                try Task.checkCancellation()
                let tile = try await source.readScanTile(rows: range)
                let first = range.lowerBound * shape.scanColumns
                for k in 0..<tile.positionCount where slot[first + k] >= 0 {
                    let s = slot[first + k], w = weights?[first + k]
                    for i in 0..<width {
                        let v = Double(tile.values[k * n + start + i])
                        gathered[i * count + s] = w.map { v * $0 } ?? v
                    }
                }
            }
            let medians = gathered.withUnsafeMutableBufferPointer { g -> [Double] in
                let shared = SharedBuffer(base: g.baseAddress!)
                return parallelMap(width) { i in
                    medianInPlace(UnsafeMutableBufferPointer(start: shared.base + i * count, count: count))
                }
            }
            for i in 0..<width { out[start + i] = medians[i] }
            start = end
        }
        return DetectorImage(detectorRows: shape.detectorRows, detectorColumns: shape.detectorColumns, values: out,
                             path: .cpu)
    }

    /// np.median of one pixel's values: NaN if any is NaN; else sorted, the middle one, or (a + b) / 2 of the two middle.
    static func medianInPlace(_ column: UnsafeMutableBufferPointer<Double>) -> Double {
        var v = column
        if v.contains(where: \.isNaN) { return .nan }
        v.sort()
        let c = v.count
        return c % 2 == 1 ? v[c / 2] : (v[c / 2 - 1] + v[c / 2]) / 2
    }

    /// The centre-shifted mean and max over a selection (slice 3, 3.4, with py4DSTEM's mask): each selected pattern moved
    /// by the map's mean minus its origin, then Σ moved (· w) / count (or Σ w) and the max of moved (· w), seeded at −∞.
    private static func shifted(_ source: some CubeSource, _ statistic: DiffractionStatistic, _ selected: [Int],
                                _ weights: [Double]?, _ shift: PatternShift, _ tiling: ScanTiling) async throws
        -> DetectorImage {
        let shape = source.shape
        let map: OriginMap, fourier: Bool
        switch shift { case .rounded(let m): map = m; fourier = false; case .fourier(let m): map = m; fourier = true }
        try VirtualImaging.require(map, matches: shape)
        let qy = shape.detectorRows, qx = shape.detectorColumns, n = shape.detectorCount
        var take = [Bool](repeating: false, count: shape.scanCount)
        for k in selected { take[k] = true }
        let fft = fourier ? FFT2D(rows: qy, columns: qx) : nil
        var moved = [Neumaier](repeating: Neumaier(), count: n)
        var top = [Double](repeating: -.infinity, count: n)
        var weight = Neumaier()
        for range in try tiling.ranges(for: shape) {
            try Task.checkCancellation()
            let tile = try await source.readScanTile(rows: range)
            let first = range.lowerBound * shape.scanColumns
            for k in 0..<tile.positionCount where take[first + k] {
                let sr = map.meanRow - map.row.values[first + k], sc = map.meanColumn - map.column.values[first + k]
                var p = (0..<n).map { Double(tile.values[k * n + $0]) }
                if let fft {
                    p = fourierShift(p, qy, qx, sr, sc, fft)
                } else {
                    p = VirtualImaging.roll(p, qy, qx, VirtualImaging.wrap(sr.rounded(.toNearestOrEven), qy),
                                            VirtualImaging.wrap(sc.rounded(.toNearestOrEven), qx))
                }
                let w = weights?[first + k] ?? 1.0
                weight.add(w)
                for i in 0..<n {
                    let v = weights == nil ? p[i] : p[i] * w
                    if statistic == .mean { moved[i].add(v) } else {
                        let m = top[i]        // np.maximum(acc, moved): of equal values the accumulated one
                        top[i] = (m.isNaN || v.isNaN) ? .nan : (v > m ? v : m)
                    }
                }
            }
        }
        if statistic == .max { return DetectorImage(detectorRows: qy, detectorColumns: qx, values: top, path: .cpu) }
        let norm = weights == nil ? Double(selected.count) : weight.value
        guard norm != 0 else { throw CuboVirtualError.invalidInput("the weights sum to zero") }
        return DetectorImage(detectorRows: qy, detectorColumns: qx, values: moved.map { $0.value / norm }, path: .cpu)
    }

    /// py4DSTEM's get_shifted_ar (Fourier): Re(IFFT2(FFT2(p) · exp(−2πi (sc · f_c + sr · f_r)))), f = fftfreq(n, 1).
    static func fourierShift(_ p: [Double], _ qy: Int, _ qx: Int, _ sr: Double, _ sc: Double, _ fft: FFT2D) -> [Double] {
        let fr = fftfreq(qy), fc = fftfreq(qx)
        var re = p, im = [Double](repeating: 0, count: p.count)
        fft.transform(&re, &im, inverse: false)
        for i in 0..<qy { for j in 0..<qx {
            let angle = -2 * Double.pi * (sc * fc[j] + sr * fr[i])
            let c = cos(angle), s = sin(angle), a = re[i * qx + j], b = im[i * qx + j]
            re[i * qx + j] = a * c - b * s; im[i * qx + j] = a * s + b * c
        } }
        fft.transform(&re, &im, inverse: true)
        return re
    }
}

/// A pointer handed to concurrent workers that each touch only their own elements.
struct SharedBuffer<T>: @unchecked Sendable {
    let base: UnsafeMutablePointer<T>
}

/// `(0..<count).map(body)` on every core (Dispatch's concurrentPerform); `body` must not share mutable state.
func parallelMap<T>(_ count: Int, _ body: @Sendable (Int) -> T) -> [T] {
    guard count > 0 else { return [] }
    return [T](unsafeUninitializedCapacity: count) { buffer, initialized in
        let shared = SharedBuffer(base: buffer.baseAddress!)
        DispatchQueue.concurrentPerform(iterations: count) { k in (shared.base + k).initialize(to: body(k)) }
        initialized = count
    }
}
