//
//  BraggDisks.swift
//  Bragg disk detection by probe-kernel template matching: py4DSTEM's find_Bragg_disks for one pattern, a stack, or a whole
//  cube (braggvectors/diskdetection.py at f050d207), CPU, Double (or, opt in, Float32: SingleDetection.swift). The numerical definition:
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
    private let singleBox: SingleBox

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
        singleBox = SingleBox(rows: detectorRows, columns: detectorColumns, kernel: kernel.values)
    }

    /// Whether `BraggPrecision.single` can run on a detector of this size: vDSP's complex DFT must support both lengths
    /// (f·2ⁿ with f in 1, 3, 5, 15, ...: 64, 128, 192, 240, 256, 320, 384, 480, 512, ...). Other sizes are refused by the
    /// `.single` entry points with `CuboBraggError.invalidOption`; `.double` runs on any size.
    public static func supportsSinglePrecision(detectorRows: Int, detectorColumns: Int) -> Bool {
        SingleFFT2D.supports(rows: detectorRows, columns: detectorColumns)
    }

    /// The correlation of one pattern: the real map `CC` and the Fourier-domain product `cc` (see the file header).
    public struct Correlation: Sendable, Equatable {
        public let real: [Double]
        public let fourierRe: [Double]
        public let fourierIm: [Double]
    }

    /// py4DSTEM's `get_cross_correlation_FT` on the (optionally smoothed) pattern, with the real map `max(Re IFFT, 0)`.
    /// With `options.precision == .single` the map is computed in Float32 and widened.
    public func correlate(_ pattern: DetectorImage, options o: BraggDiskOptions = BraggDiskOptions()) throws -> Correlation {
        try BraggMaxima.validate(o)
        let v = try values(of: pattern)
        if o.precision == .single {
            let state = try singleBox.state()
            let floats = v.map { Float($0) }
            return try floats.withUnsafeBufferPointer { try state.correlation($0.baseAddress!, options: o) }
        }
        return try correlate(values: v, options: o).0
    }

    /// The peaks of one pattern.
    public func detect(_ pattern: DetectorImage, options o: BraggDiskOptions = BraggDiskOptions()) throws -> [BraggPeak] {
        try detectWithFunnel(pattern, options: o).peaks
    }

    /// The peaks of one pattern, as `detect`, with the count of maxima that survived each stage of the selection.
    public func detectWithFunnel(_ pattern: DetectorImage, options o: BraggDiskOptions = BraggDiskOptions()) throws
        -> (peaks: [BraggPeak], funnel: BraggFunnel) {
        try BraggMaxima.validate(o)
        let v = try values(of: pattern)
        if o.precision == .single {
            let state = try singleBox.state()
            let floats = v.map { Float($0) }
            return try floats.withUnsafeBufferPointer { try state.detect($0.baseAddress!, options: o) }
        }
        return try detect(values: v, options: o)
    }

    /// The peaks of one Float32 pattern (`detectorRows · detectorColumns` values, row-major) — the entry that never
    /// widens the pattern when `options.precision == .single`. With `.double` the pattern is widened first.
    public func detect(_ pattern: UnsafeBufferPointer<Float>, options o: BraggDiskOptions = BraggDiskOptions()) throws -> [BraggPeak] {
        try detectWithFunnel(pattern, options: o).peaks
    }

    /// `detect(_:options:)` for a Float32 pattern, with the funnel.
    public func detectWithFunnel(_ pattern: UnsafeBufferPointer<Float>, options o: BraggDiskOptions = BraggDiskOptions()) throws
        -> (peaks: [BraggPeak], funnel: BraggFunnel) {
        try BraggMaxima.validate(o)
        guard pattern.count == detectorRows * detectorColumns, let base = pattern.baseAddress else {
            throw CuboBraggError.shapeMismatch("the pattern holds \(pattern.count) values, the kernel \(detectorRows) × \(detectorColumns)")
        }
        if o.precision == .single { return try singleBox.state().detect(base, options: o) }
        return try detect(values: pattern.map { Double($0) }, options: o)
    }

    func values(of pattern: DetectorImage) throws -> [Double] {
        guard pattern.detectorRows == detectorRows, pattern.detectorColumns == detectorColumns else {
            throw CuboBraggError.shapeMismatch("the pattern is \(pattern.detectorRows) × \(pattern.detectorColumns), the kernel \(detectorRows) × \(detectorColumns)")
        }
        return pattern.values
    }

    /// The Double path.
    func detect(values: [Double], options o: BraggDiskOptions) throws -> (peaks: [BraggPeak], funnel: BraggFunnel) {
        let (c, replaced) = try correlate(values: values, options: o)
        var result = try BraggMaxima.findWithFunnel(in: c.real, rows: detectorRows, columns: detectorColumns, options: o,
                                                    fourier: (c.fourierRe, c.fourierIm))
        result.funnel.nonFinitePixelCount = replaced
        return result
    }

    /// The Float32 path's state for the whole-cube pass (refuses an unsupported size here, before any tile is read).
    func singleState() throws -> SingleState { try singleBox.state() }

    func correlate(values input: [Double], options o: BraggDiskOptions) throws -> (Correlation, Int) {
        var v = input
        var replaced = 0
        if o.nonFinitePixels == .neighbourMedian { replaced = Self.fillNonFinite(&v, rows: detectorRows, columns: detectorColumns) }
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
        return (Correlation(real: real, fourierRe: fr, fourierIm: fi), replaced)
    }

    /// mac4DSTEM's fill: every non-finite (NaN or infinite) pixel of the row-major `rows × columns` image `v` becomes the
    /// median of its finite 8-neighbours (the mean of the middle two of an even count; 0 when none is finite). Neighbours are
    /// read before any replacement, so replaced pixels never feed each other. Returns how many pixels it replaced (0 for a
    /// finite image, which is left untouched). `v.count` must equal `rows * columns`.
    @discardableResult
    public static func fillNonFinite(_ v: inout [Double], rows: Int, columns: Int) -> Int {
        precondition(v.count == rows * columns, "fillNonFinite: \(v.count) values for \(rows) × \(columns)")
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
        return fills.count
    }

    /// `fillNonFinite` for a Float32 image, in Float32 arithmetic (the median of two middle values is their mean in Float).
    @discardableResult
    public static func fillNonFinite(_ v: inout [Float], rows: Int, columns: Int) -> Int {
        precondition(v.count == rows * columns, "fillNonFinite: \(v.count) values for \(rows) × \(columns)")
        guard !v.isEmpty else { return 0 }
        return v.withUnsafeMutableBufferPointer { SingleNonFinite.fill($0.baseAddress!, rows: rows, columns: columns) }
    }
}

/// The Float32 state of a detector, built on first use (a `.double`-only caller never pays for it) and shared by copies of the
/// detector.
private final class SingleBox: @unchecked Sendable {
    private let rows: Int, columns: Int, kernel: [Double]
    private let lock = NSLock()
    private var built: SingleState?
    init(rows: Int, columns: Int, kernel: [Double]) { self.rows = rows; self.columns = columns; self.kernel = kernel }
    func state() throws -> SingleState {
        lock.lock(); defer { lock.unlock() }
        if let built { return built }
        let s = try SingleState(rows: rows, columns: columns, kernel: kernel)
        built = s
        return s
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
    /// on every core. Patterns are read as Float32 and, for `.double`, widened to Double; `.single` uses them as they are, one
    /// scratch per concurrent worker. Throws `CancellationError` if the task is cancelled
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
        let single = options.precision == .single ? try detector.singleState() : nil   // refuses an unsupported size before any read
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
                            do {
                                if let single {
                                    o.base[k] = try single.detect(v.base + k * n, options: options).peaks
                                } else {
                                    let pattern = (0..<n).map { Double(v.base[k * n + $0]) }
                                    o.base[k] = try detector.detect(values: pattern, options: options).peaks
                                }
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
