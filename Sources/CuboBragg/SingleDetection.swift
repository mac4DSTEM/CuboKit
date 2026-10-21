//
//  SingleDetection.swift
//  The `.single` path of BraggDetector: the correlation of docs/bragg.md in Float32 on Accelerate, with one reusable scratch per
//  concurrent worker. Pattern copy, non-finite fill, the Gaussian filters (vDSP_conv along reflect-padded rows, row-wise
//  multiply-adds across rows), the 2-D transform (SingleFFT.swift), the correlation product (vDSP_zvmul, or the hybrid
//  power), max(Re, 0) and the maxima search run in Float; the few accepted peaks are refined in Double (parabola from the
//  Float neighbourhood, matrix-DFT upsampling on the Float complex correlation widened to Double).
//

import CuboCore
import Accelerate
import CuboVirtual
import Foundation

/// A Gaussian's taps in Float: the Double kernel of `GaussianFilter` (scipy's radius int(4σ + 0.5)), rounded once.
struct SingleTaps {
    let radius: Int
    let values: [Float]
    init(sigma: Double) throws {
        let kernel = try GaussianFilter(sigma: sigma).kernel
        values = kernel.map { Float($0) }
        radius = (kernel.count - 1) / 2
    }
}

/// The buffers one worker needs for one pattern at a time. Reused from pattern to pattern; never shared.
final class SingleScratch: @unchecked Sendable {
    let count: Int
    let a0: UnsafeMutablePointer<Float>, a1: UnsafeMutablePointer<Float>
    let b0: UnsafeMutablePointer<Float>, b1: UnsafeMutablePointer<Float>
    /// The complex correlation, kept for `.multicorr` (transposed layout, see SingleFFT.swift).
    let d0: UnsafeMutablePointer<Float>, d1: UnsafeMutablePointer<Float>
    private var pad: UnsafeMutablePointer<Float>
    private var padCapacity: Int
    private var taps: [Double: SingleTaps] = [:]
    /// Double copies of the natural-layout complex correlation, for the matrix-DFT refinement.
    var fourierRe: [Double] = [], fourierIm: [Double] = []

    init(count: Int) {
        self.count = count
        func buffer() -> UnsafeMutablePointer<Float> {
            let p = UnsafeMutablePointer<Float>.allocate(capacity: count)
            p.initialize(repeating: 0, count: count)
            return p
        }
        a0 = buffer(); a1 = buffer(); b0 = buffer(); b1 = buffer(); d0 = buffer(); d1 = buffer()
        padCapacity = 1
        pad = UnsafeMutablePointer<Float>.allocate(capacity: 1)
    }

    deinit {
        for p in [a0, a1, b0, b1, d0, d1, pad] { p.deallocate() }
    }

    func taps(_ sigma: Double) throws -> SingleTaps {
        if let t = taps[sigma] { return t }
        let t = try SingleTaps(sigma: sigma)
        if taps.count > 4 { taps.removeAll() }
        taps[sigma] = t
        return t
    }

    func padded(_ capacity: Int) -> UnsafeMutablePointer<Float> {
        if capacity > padCapacity {
            pad.deallocate()
            pad = UnsafeMutablePointer<Float>.allocate(capacity: capacity)
            padCapacity = capacity
        }
        return pad
    }
}

/// What a detector builds once for `.single`: the plan, the kernel's conjugate-ready spectrum, and the scratch pool.
final class SingleState: @unchecked Sendable {
    let rows: Int, columns: Int
    let fft: SingleFFT2D
    /// FFT2(kernel), transposed layout; the correlation multiplies by its conjugate.
    private let kernelRe: UnsafeMutablePointer<Float>, kernelIm: UnsafeMutablePointer<Float>
    private let lock = NSLock()
    private var free: [SingleScratch] = []

    init(rows: Int, columns: Int, kernel: [Double]) throws {
        guard let fft = SingleFFT2D(rows: rows, columns: columns) else {
            throw CuboBraggError.invalidOption("precision .single needs a detector size vDSP's DFT supports on both axes (f·2ⁿ, f in 1, 3, 5, 15, ...); \(rows) × \(columns) is not one (use .double)")
        }
        self.rows = rows; self.columns = columns; self.fft = fft
        let n = rows * columns
        let scratch = SingleScratch(count: n)
        for i in 0..<n { scratch.a0[i] = Float(kernel[i]) }
        scratch.a1.update(repeating: 0, count: n)
        fft.forward(aRe: scratch.a0, aIm: scratch.a1, bRe: scratch.b0, bIm: scratch.b1)
        kernelRe = UnsafeMutablePointer<Float>.allocate(capacity: n)
        kernelIm = UnsafeMutablePointer<Float>.allocate(capacity: n)
        kernelRe.update(from: scratch.b0, count: n)
        kernelIm.update(from: scratch.b1, count: n)
        free = [scratch]
    }

    deinit { kernelRe.deallocate(); kernelIm.deallocate() }

    /// Scratches created so far and idle now (one per concurrent worker at most).
    var scratchCount: Int { lock.lock(); defer { lock.unlock() }; return free.count }

    func checkOut() -> SingleScratch {
        lock.lock(); defer { lock.unlock() }
        return free.popLast() ?? SingleScratch(count: rows * columns)
    }

    func giveBack(_ s: SingleScratch) {
        lock.lock(); free.append(s); lock.unlock()
    }

    /// Runs `body` with a scratch of its own (one per concurrent caller, reused afterwards).
    func withScratch<T>(_ body: (SingleScratch) throws -> T) rethrows -> T {
        let s = checkOut()
        defer { giveBack(s) }
        return try body(s)
    }

    /// The correlation of `pattern` (row-major `rows × columns` Float): afterwards `scratch.a0` holds
    /// CC = max(Re IFFT2(cc), 0) and, with `retain`, `scratch.d0/d1` the complex product `cc` (transposed layout). Returns the
    /// number of non-finite pixels replaced and where the smoothed correlation is (`scratch.a0` or `scratch.a1`).
    func correlate(_ pattern: UnsafePointer<Float>, options o: BraggDiskOptions, retain: Bool,
                   in s: SingleScratch) throws -> (nonFinite: Int, smooth: UnsafeMutablePointer<Float>) {
        let n = rows * columns
        var a0 = s.a0, a1 = s.a1
        let b0 = s.b0, b1 = s.b1
        a0.update(from: pattern, count: n)
        var replaced = 0
        if o.nonFinitePixels == .neighbourMedian { replaced = SingleNonFinite.fill(a0, rows: rows, columns: columns) }
        if o.patternSigma > 0 {
            SingleFilter.gaussian(src: a0, tmp: b0, dst: a1, rows: rows, columns: columns, taps: try s.taps(o.patternSigma), scratch: s)
            swap(&a0, &a1)
        }
        vDSP_vclr(a1, 1, vDSP_Length(n))
        fft.forward(aRe: a0, aIm: a1, bRe: b0, bIm: b1)

        // m = FFT2(pattern) · conj(FFT2(kernel)) into (a0, a1).
        let p = Float(o.correlationPower)
        // vDSP_zvmul with conjugate -1 is conj(A) · B.
        var k = DSPSplitComplex(realp: kernelRe, imagp: kernelIm)
        var x = DSPSplitComplex(realp: b0, imagp: b1)
        var m = DSPSplitComplex(realp: a0, imagp: a1)
        vDSP_zvmul(&k, 1, &x, 1, &m, 1, vDSP_Length(n), -1)
        if p != 1 {
            // |m|^p · exp(i arg m); a zero bin has no phase and NumPy gives it 0^p: 1 at p = 0, else 0.
            for i in 0..<n {
                let mr = a0[i], mi = a1[i]
                let magnitude = (mr * mr + mi * mi).squareRoot()
                if magnitude == 0 {
                    a0[i] = p == 0 ? 1 : 0; a1[i] = 0
                } else {
                    let f = powf(magnitude, p - 1)
                    a0[i] = mr * f; a1[i] = mi * f
                }
            }
        }
        if retain {
            s.d0.update(from: a0, count: n)
            s.d1.update(from: a1, count: n)
        }
        fft.inverse(aRe: a0, aIm: a1, bRe: b0, bIm: b1)
        // CC = max(Re / (rows · columns), 0); the Double path's NaN stays NaN there and finds no maxima here either.
        var scale = 1 / Float(n), zero: Float = 0
        vDSP_vsmul(b0, 1, &scale, b1, 1, vDSP_Length(n))
        vDSP_vthres(b1, 1, &zero, a0, 1, vDSP_Length(n))
        if o.correlationSigma > 0 {
            SingleFilter.gaussian(src: a0, tmp: b0, dst: a1, rows: rows, columns: columns, taps: try s.taps(o.correlationSigma), scratch: s)
            return (replaced, a1)
        }
        return (replaced, a0)
    }

    /// The peaks of one pattern and the funnel.
    func detect(_ pattern: UnsafePointer<Float>, options o: BraggDiskOptions) throws -> (peaks: [BraggPeak], funnel: BraggFunnel) {
        try withScratch { s in
            let retain = o.subpixel == .multicorr
            let (replaced, smooth) = try correlate(pattern, options: o, retain: retain, in: s)
            var maximum: Float = 0
            vDSP_maxv(smooth, 1, &maximum, vDSP_Length(rows * columns))
            var result = try BraggMaxima.select(
                smooth: UnsafePointer(smooth), rows: rows, columns: columns, options: o,
                correlationMaximum: maximum.isNaN ? 0 : Double(maximum)) {
                // Natural layout, widened to Double, for the matrix-DFT refinement (once, and only when a peak remains).
                let n = self.rows * self.columns
                self.fft.naturalSpectrum(fromRe: s.d0, fromIm: s.d1, toRe: s.b0, toIm: s.b1)
                if s.fourierRe.count != n { s.fourierRe = [Double](repeating: 0, count: n); s.fourierIm = s.fourierRe }
                vDSP_vspdp(s.b0, 1, &s.fourierRe, 1, vDSP_Length(n))
                vDSP_vspdp(s.b1, 1, &s.fourierIm, 1, vDSP_Length(n))
                return (s.fourierRe, s.fourierIm)
            }
            result.funnel.nonFinitePixelCount = replaced
            return result
        }
    }

    /// The real correlation map and the complex product, in Double and natural layout (the public `correlate`).
    func correlation(_ pattern: UnsafePointer<Float>, options o: BraggDiskOptions) throws -> BraggDetector.Correlation {
        try withScratch { s in
            let n = rows * columns
            var o2 = o
            o2.correlationSigma = 0   // the raw map is wanted, a0
            let (_, raw) = try correlate(pattern, options: o2, retain: true, in: s)
            var real = [Double](repeating: 0, count: n), fr = real, fi = real
            vDSP_vspdp(raw, 1, &real, 1, vDSP_Length(n))
            fft.naturalSpectrum(fromRe: s.d0, fromIm: s.d1, toRe: s.b0, toIm: s.b1)
            vDSP_vspdp(s.b0, 1, &fr, 1, vDSP_Length(n))
            vDSP_vspdp(s.b1, 1, &fi, 1, vDSP_Length(n))
            return BraggDetector.Correlation(real: real, fourierRe: fr, fourierIm: fi)
        }
    }
}

enum SingleNonFinite {
    /// The median-of-finite-neighbours fill on a Float image; returns how many pixels it replaced. A sum is finite only if
    /// every term is, so a finite sum skips the scan (an overflowing sum of finite values only costs the scan).
    static func fill(_ p: UnsafeMutablePointer<Float>, rows: Int, columns: Int) -> Int {
        var total: Float = 0
        vDSP_sve(p, 1, &total, vDSP_Length(rows * columns))
        if total.isFinite { return 0 }
        var fills: [(Int, Float)] = []
        for r in 0..<rows {
            for c in 0..<columns where !p[r * columns + c].isFinite {
                var neighbours: [Float] = []
                for dr in -1...1 {
                    for dc in -1...1 where dr != 0 || dc != 0 {
                        let rr = r + dr, cc = c + dc
                        guard rr >= 0, rr < rows, cc >= 0, cc < columns else { continue }
                        let x = p[rr * columns + cc]
                        if x.isFinite { neighbours.append(x) }
                    }
                }
                neighbours.sort()
                let k = neighbours.count
                let median: Float = k == 0 ? 0 : (k % 2 == 1 ? neighbours[k / 2] : (neighbours[k / 2 - 1] + neighbours[k / 2]) / 2)
                fills.append((r * columns + c, median))
            }
        }
        for (i, x) in fills { p[i] = x }
        return fills.count
    }
}

enum SingleFilter {
    /// scipy's gaussian_filter, mode `reflect`, in Float: rows by vDSP_conv on reflect-padded lines, then columns by
    /// multiply-adds of whole rows. `src` is read, `tmp` and `dst` written; the three are distinct.
    static func gaussian(src: UnsafePointer<Float>, tmp: UnsafeMutablePointer<Float>, dst: UnsafeMutablePointer<Float>,
                         rows: Int, columns: Int, taps: SingleTaps, scratch: SingleScratch) {
        let R = taps.radius, width = 2 * R + 1
        let pad = scratch.padded(columns + 2 * R)
        taps.values.withUnsafeBufferPointer { k in
            let kp = k.baseAddress!
            for r in 0..<rows {
                let line = src + r * columns
                (pad + R).update(from: line, count: columns)
                for i in 0..<R {
                    pad[i] = line[reflect(i - R, columns)]
                    pad[R + columns + i] = line[reflect(columns + i, columns)]
                }
                vDSP_conv(pad, 1, kp, 1, tmp + r * columns, 1, vDSP_Length(columns), vDSP_Length(width))
            }
            for r in 0..<rows {
                let out = dst + r * columns
                vDSP_vsmul(tmp + r * columns, 1, kp + R, out, 1, vDSP_Length(columns))
                for t in stride(from: 1, through: R, by: 1) {
                    vDSP_vsma(tmp + reflect(r - t, rows) * columns, 1, kp + (R - t), out, 1, out, 1, vDSP_Length(columns))
                    vDSP_vsma(tmp + reflect(r + t, rows) * columns, 1, kp + (R - t), out, 1, out, 1, vDSP_Length(columns))
                }
            }
        }
    }

    /// Half-sample symmetric reflection, folded repeatedly for an index far outside (d c b a | a b c d | d c b a).
    @inline(__always)
    static func reflect(_ i: Int, _ n: Int) -> Int {
        if i >= 0, i < n { return i }
        let m = ((i % (2 * n)) + 2 * n) % (2 * n)
        return m < n ? m : 2 * n - 1 - m
    }
}
