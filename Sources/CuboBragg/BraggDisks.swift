//
//  BraggDisks.swift
//  Bragg disk detection by probe-kernel template matching: py4DSTEM's find_Bragg_disks for one pattern, a stack, or a whole
//  cube (braggvectors/diskdetection.py at f050d207), CPU, Double. The numerical definition:
//
//      m  = FFT2(pattern) · conj(FFT2(kernel))
//      cc = |m|^p · exp(i·arg m)            (p = correlationPower; cc = m at p = 1)
//      CC = max(Re IFFT2(cc), 0)
//
//  then BraggMaxima on CC, with `cc` as the Fourier correlation for `.multicorr`.
//

import CuboCore
import CuboVirtual
import Foundation

/// A detector for one pattern shape and one kernel: the kernel's conjugate transform is computed once, so a detector can
/// serve many patterns, from many threads at once.
public struct BraggDetector: Sendable {
    public let detectorRows: Int
    public let detectorColumns: Int
    private let plan: FFTPlanBox
    private let kernelRe: [Double]
    private let kernelIm: [Double]   // conj(FFT2(kernel))

    /// `kernel` is corner-centred, as `ProbeKernel` builds it (the probe's centre at pixel (0, 0)), and as py4DSTEM's
    /// `template`.
    public init(kernel: DetectorImage) throws {
        detectorRows = kernel.detectorRows
        detectorColumns = kernel.detectorColumns
        guard detectorRows > 0, detectorColumns > 0 else { throw CuboBraggError.shapeMismatch("the kernel is empty") }
        plan = FFTPlanBox(FFT2D(rows: detectorRows, columns: detectorColumns))
        var re = kernel.values, im = [Double](repeating: 0, count: re.count)
        plan.fft.transform(&re, &im, inverse: false)
        kernelRe = re
        kernelIm = im.map { -$0 }
    }

    /// The correlation of one pattern: the real map `CC` and the Fourier-domain product `cc` (see the file header).
    public struct Correlation: Sendable, Equatable {
        public let real: [Double]
        public let fourierRe: [Double]
        public let fourierIm: [Double]
    }

    /// py4DSTEM's `get_cross_correlation_FT` on the (optionally smoothed) pattern, with the real map `max(Re IFFT, 0)`.
    public func correlate(_ pattern: DetectorImage, options o: BraggDiskOptions = BraggDiskOptions()) throws -> Correlation {
        try BraggMaxima.validate(o)
        return try correlate(values: try values(of: pattern), options: o)
    }

    /// The peaks of one pattern.
    public func detect(_ pattern: DetectorImage, options o: BraggDiskOptions = BraggDiskOptions()) throws -> [BraggPeak] {
        try BraggMaxima.validate(o)
        return try detect(values: try values(of: pattern), options: o)
    }

    func values(of pattern: DetectorImage) throws -> [Double] {
        guard pattern.detectorRows == detectorRows, pattern.detectorColumns == detectorColumns else {
            throw CuboBraggError.shapeMismatch("the pattern is \(pattern.detectorRows) × \(pattern.detectorColumns), the kernel \(detectorRows) × \(detectorColumns)")
        }
        return pattern.values
    }

    func detect(values: [Double], options o: BraggDiskOptions) throws -> [BraggPeak] {
        let c = try correlate(values: values, options: o)
        return try BraggMaxima.find(in: c.real, rows: detectorRows, columns: detectorColumns, options: o,
                                    fourier: (c.fourierRe, c.fourierIm))
    }

    func correlate(values input: [Double], options o: BraggDiskOptions) throws -> Correlation {
        var v = input
        if o.nonFinitePixels == .neighbourMedian { Self.fillNonFinite(&v, rows: detectorRows, columns: detectorColumns) }
        if o.patternSigma > 0 {
            v = try GaussianFilter(sigma: o.patternSigma).apply(v, rows: detectorRows, columns: detectorColumns, edge: .reflect)
        }
        var re = v, im = [Double](repeating: 0, count: v.count)
        plan.fft.transform(&re, &im, inverse: false)
        let p = o.correlationPower
        for k in re.indices {
            let a = re[k], b = im[k]
            var mr = a * kernelRe[k] - b * kernelIm[k]
            var mi = a * kernelIm[k] + b * kernelRe[k]
            if p != 1 {
                // |m|^p · exp(i arg m); a zero bin has no phase and NumPy gives it 0^p: 1 at p = 0, else 0.
                let magnitude = hypot(mr, mi)
                if magnitude == 0 {
                    mr = p == 0 ? 1 : 0; mi = 0
                } else {
                    let f = pow(magnitude, p - 1)
                    mr *= f; mi *= f
                }
            }
            re[k] = mr; im[k] = mi
        }
        let fr = re, fi = im
        plan.fft.transform(&re, &im, inverse: true)
        // np.maximum(x, 0) keeps a NaN.
        let real = re.map { $0 < 0 ? 0 : $0 }
        return Correlation(real: real, fourierRe: fr, fourierIm: fi)
    }

    /// mac4DSTEM's fill: every non-finite pixel becomes the median of its finite 8-neighbours (the mean of the middle two
    /// of an even count; 0 when none is finite). Neighbours are read before any replacement.
    static func fillNonFinite(_ v: inout [Double], rows: Int, columns: Int) {
        var fills: [(Int, Double)] = []
        for r in 0..<rows {
            for c in 0..<columns where !v[r * columns + c].isFinite {
                var neighbours: [Double] = []
                for dr in -1...1 {
                    for dc in -1...1 where dr != 0 || dc != 0 {
                        let rr = r + dr, cc = c + dc
                        guard rr >= 0, rr < rows, cc >= 0, cc < columns else { continue }
                        let x = v[rr * columns + cc]
                        if x.isFinite { neighbours.append(x) }
                    }
                }
                neighbours.sort()
                let n = neighbours.count
                let median = n == 0 ? 0 : (n % 2 == 1 ? neighbours[n / 2] : (neighbours[n / 2 - 1] + neighbours[n / 2]) / 2)
                fills.append((r * columns + c, median))
            }
        }
        for (i, x) in fills { v[i] = x }
    }
}

/// The 2-D transform plan, shared read-only between threads (its vDSP setups are immutable after creation).
private final class FFTPlanBox: @unchecked Sendable {
    let fft: FFT2D
    init(_ fft: FFT2D) { self.fft = fft }
}

public enum BraggDisks {
    /// py4DSTEM's `find_Bragg_disks(pattern, template)`: the peaks of one pattern, brightest first.
    public static func detect(_ pattern: DetectorImage, kernel: DetectorImage,
                              options: BraggDiskOptions = BraggDiskOptions()) throws -> [BraggPeak] {
        try BraggDetector(kernel: kernel).detect(pattern, options: options)
    }

    /// `find_Bragg_disks` on a 3-D stack: one list per pattern, in order.
    public static func detect(patterns: [DetectorImage], kernel: DetectorImage,
                              options: BraggDiskOptions = BraggDiskOptions()) throws -> [[BraggPeak]] {
        let detector = try BraggDetector(kernel: kernel)
        return try patterns.map { try detector.detect($0, options: options) }
    }

    /// `find_Bragg_disks(datacube, template)`: every scan position of `source`, a scan tile at a time, the patterns of a tile
    /// on every core. Patterns are read as Float32 and widened to Double. Throws `CancellationError` if the task is cancelled
    /// between tiles.
    public static func detectAll(in source: some CubeSource, kernel: DetectorImage,
                                 options: BraggDiskOptions = BraggDiskOptions(),
                                 tiling: ScanTiling = .automatic) async throws -> BraggVectors {
        try BraggMaxima.validate(options)
        let shape = source.shape
        guard shape.detectorRows == kernel.detectorRows, shape.detectorColumns == kernel.detectorColumns else {
            throw CuboBraggError.shapeMismatch("the cube's detector is \(shape.detectorRows) × \(shape.detectorColumns), the kernel \(kernel.detectorRows) × \(kernel.detectorColumns)")
        }
        let detector = try BraggDetector(kernel: kernel)
        let n = shape.detectorCount
        var all: [[BraggPeak]] = []
        all.reserveCapacity(shape.scanCount)
        for range in try tiling.ranges(for: shape) {
            try Task.checkCancellation()
            let tile = try await source.readScanTile(rows: range)
            let count = tile.positionCount
            var slots = [[BraggPeak]](repeating: [], count: count)
            var failures = [UInt8](repeating: 0, count: count)
            tile.values.withUnsafeBufferPointer { values in
                slots.withUnsafeMutableBufferPointer { out in
                    failures.withUnsafeMutableBufferPointer { fail in
                        let v = SharedBuffer(base: values.baseAddress!), o = SharedBuffer(base: out.baseAddress!),
                            f = SharedBuffer(base: fail.baseAddress!)
                        DispatchQueue.concurrentPerform(iterations: count) { k in
                            let pattern = (0..<n).map { Double(v.base[k * n + $0]) }
                            do {
                                o.base[k] = try detector.detect(values: pattern, options: options)
                            } catch {
                                f.base[k] = 1
                            }
                        }
                    }
                }
            }
            if failures.contains(1) { throw CuboBraggError.nonFiniteRefinement }
            all.append(contentsOf: slots)
        }
        return BraggVectors(scanRows: shape.scanRows, scanColumns: shape.scanColumns, peaks: all)
    }
}

/// A pointer handed to concurrent workers that each touch only their own elements.
private struct SharedBuffer<T>: @unchecked Sendable {
    let base: UnsafeMutablePointer<T>
    init(base: UnsafePointer<T>) { self.base = UnsafeMutablePointer(mutating: base) }
    init(base: UnsafeMutablePointer<T>) { self.base = base }
}
