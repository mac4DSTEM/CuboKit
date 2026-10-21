//
//  DoubleFFT.swift
//  The exact N-point complex DFT in Double for any N: vDSP's vDSP_DFT_zop plans for the lengths it takes (2^n × {1, 3, 5, 15};
//  5-7× faster than vDSP_fft_zipD from 128² up), vDSP's radix-2 FFT (vDSP_fft_zipD) for the other powers of two, Bluestein's
//  chirp-z through a radix-2 FFT of length M ≥ 2N − 1 otherwise. Forward is NumPy's fft (e^{−2πi kn/N}, unnormalized);
//  inverse is NumPy's ifft (e^{+2πi kn/N}, divided by N). Double throughout, for the DPC iteration of slice 2
//  and the Fourier shifts of slice 3 (moved to CuboCore, package, in slice 3).
//

import Accelerate
import Foundation

package final class FFTPlan: @unchecked Sendable {   // immutable after init; vDSP setups are shareable across threads
    package let n: Int
    private let setup: FFTSetupD
    private let log2m: vDSP_Length
    private let m: Int
    private let chirpRe: [Double], chirpIm: [Double]          // e^{−iπ k²/n}, k = 0 … n−1 (Bluestein only)
    private let kernelRe: [Double], kernelIm: [Double]        // FFT of the conjugate chirp, wrapped (Bluestein only)
    fileprivate let dftForward: vDSP_DFT_SetupD?              // vDSP_DFT_zop plans, nil when vDSP does not take the length
    fileprivate let dftInverse: vDSP_DFT_SetupD?

    package init(_ n: Int) {
        precondition(n > 0)
        self.n = n
        dftForward = vDSP_DFT_zop_CreateSetupD(nil, vDSP_Length(n), .FORWARD)
        dftInverse = dftForward == nil ? nil : vDSP_DFT_zop_CreateSetupD(nil, vDSP_Length(n), .INVERSE)
        let pow2 = n & (n - 1) == 0
        m = pow2 ? n : Int(1) << Int(ceil(log2(Double(2 * n - 1))))
        log2m = vDSP_Length(Int(log2(Double(m)).rounded()))
        setup = vDSP_create_fftsetupD(log2m, FFTRadix(kFFTRadix2))!
        if pow2 {
            chirpRe = []; chirpIm = []; kernelRe = []; kernelIm = []
        } else {
            var cr = [Double](repeating: 0, count: n), ci = cr
            for k in 0..<n {
                let kk = (k * k) % (2 * n)                       // k² mod 2n keeps the angle exact
                let a = -Double.pi * Double(kk) / Double(n)
                cr[k] = cos(a); ci[k] = sin(a)
            }
            chirpRe = cr; chirpIm = ci
            var br = [Double](repeating: 0, count: m), bi = br
            for k in 0..<n {
                br[k] = cr[k]; bi[k] = -ci[k]
                if k > 0 { br[m - k] = cr[k]; bi[m - k] = -ci[k] }
            }
            Self.radix2(setup, log2m, &br, &bi, forward: true)
            kernelRe = br; kernelIm = bi
        }
    }

    deinit {
        vDSP_destroy_fftsetupD(setup)
        if let dftForward { vDSP_DFT_DestroySetupD(dftForward) }
        if let dftInverse { vDSP_DFT_DestroySetupD(dftInverse) }
    }

    /// True when the transform runs on vDSP_DFT_zop (the fast path, also what ``FFT2D`` batches).
    fileprivate var usesDFT: Bool { dftInverse != nil }

    static func radix2(_ setup: FFTSetupD, _ log2n: vDSP_Length, _ re: inout [Double], _ im: inout [Double],
                       forward: Bool) {
        re.withUnsafeMutableBufferPointer { r in
            im.withUnsafeMutableBufferPointer { i in
                var split = DSPDoubleSplitComplex(realp: r.baseAddress!, imagp: i.baseAddress!)
                vDSP_fft_zipD(setup, &split, 1, log2n, FFTDirection(forward ? kFFTDirection_Forward : kFFTDirection_Inverse))
            }
        }
    }

    /// In place, unnormalized forward DFT (NumPy's fft); `inverse` applies NumPy's ifft (the 1/N included).
    package func transform(_ re: inout [Double], _ im: inout [Double], inverse: Bool) {
        precondition(re.count == n && im.count == n)
        if let setup = inverse ? dftInverse : dftForward {
            var outRe = [Double](repeating: 0, count: n), outIm = outRe
            vDSP_DFT_ExecuteD(setup, re, im, &outRe, &outIm)
            if inverse {
                var s = 1.0 / Double(n)
                vDSP_vsmulD(outRe, 1, &s, &re, 1, vDSP_Length(n))
                vDSP_vsmulD(outIm, 1, &s, &im, 1, vDSP_Length(n))
            } else {
                re = outRe; im = outIm
            }
            return
        }
        if inverse {   // ifft(x) = conj(fft(conj(x))) / n
            for k in 0..<n { im[k] = -im[k] }
            transform(&re, &im, inverse: false)
            let s = 1.0 / Double(n)
            for k in 0..<n { re[k] *= s; im[k] = -im[k] * s }
            return
        }
        if m == n {
            Self.radix2(setup, log2m, &re, &im, forward: true)
            return
        }
        var ar = [Double](repeating: 0, count: m), ai = ar
        for k in 0..<n {
            ar[k] = re[k] * chirpRe[k] - im[k] * chirpIm[k]
            ai[k] = re[k] * chirpIm[k] + im[k] * chirpRe[k]
        }
        Self.radix2(setup, log2m, &ar, &ai, forward: true)
        for k in 0..<m {
            let r = ar[k] * kernelRe[k] - ai[k] * kernelIm[k]
            let i = ar[k] * kernelIm[k] + ai[k] * kernelRe[k]
            ar[k] = r; ai[k] = i
        }
        Self.radix2(setup, log2m, &ar, &ai, forward: false)
        let s = 1.0 / Double(m)
        for k in 0..<n {
            let r = ar[k] * s, i = ai[k] * s
            re[k] = r * chirpRe[k] - i * chirpIm[k]
            im[k] = r * chirpIm[k] + i * chirpRe[k]
        }
    }
}

/// A 2-D transform on a rows × columns grid (row-major), rows then columns.
package struct FFT2D {
    package let rows: Int, columns: Int
    private let rowPlan: FFTPlan, columnPlan: FFTPlan

    package init(rows: Int, columns: Int) {
        self.rows = rows; self.columns = columns
        columnPlan = FFTPlan(columns)                // transforms along a row (length = columns)
        rowPlan = rows == columns ? columnPlan : FFTPlan(rows)
    }

    package func transform(_ re: inout [Double], _ im: inout [Double], inverse: Bool) {
        if usesDFT { transformDFT(&re, &im, inverse: inverse); return }
        var lr = [Double](repeating: 0, count: columns), li = lr
        for r in 0..<rows {
            let base = r * columns
            for c in 0..<columns { lr[c] = re[base + c]; li[c] = im[base + c] }
            columnPlan.transform(&lr, &li, inverse: inverse)
            for c in 0..<columns { re[base + c] = lr[c]; im[base + c] = li[c] }
        }
        var cr = [Double](repeating: 0, count: rows), ci = cr
        for c in 0..<columns {
            for r in 0..<rows { cr[r] = re[r * columns + c]; ci[r] = im[r * columns + c] }
            rowPlan.transform(&cr, &ci, inverse: inverse)
            for r in 0..<rows { re[r * columns + c] = cr[r]; im[r * columns + c] = ci[r] }
        }
    }

    /// Rows, transpose, rows, transpose on vDSP_DFT_zop; scratch per call, so the value stays shareable across threads.
    private func transformDFT(_ re: inout [Double], _ im: inout [Double], inverse: Bool) {
        let count = rows * columns
        precondition(re.count == count && im.count == count)
        var scratch = [Double](repeating: 0, count: 4 * count)
        re.withUnsafeMutableBufferPointer { r in im.withUnsafeMutableBufferPointer { i in
            scratch.withUnsafeMutableBufferPointer { s in
                transformDFT(r.baseAddress!, i.baseAddress!, inverse: inverse, scratch: s.baseAddress!)
            }
        } }
    }

    /// True when ``transform(real:imaginary:inverse:scratch:)`` runs on vDSP_DFT_zop and needs its `4 * rows * columns` scratch.
    package var usesDFT: Bool { columnPlan.usesDFT && rowPlan.usesDFT }

    /// The same transform on caller-owned buffers of `rows * columns` values, with `4 * rows * columns` values of scratch
    /// (used only when ``usesDFT``): no allocation, for loops over many grids. Other lengths go through the array path.
    package func transform(real: UnsafeMutablePointer<Double>, imaginary: UnsafeMutablePointer<Double>, inverse: Bool,
                           scratch: UnsafeMutablePointer<Double>) {
        if usesDFT { transformDFT(real, imaginary, inverse: inverse, scratch: scratch); return }
        let count = rows * columns
        var r = Array(UnsafeBufferPointer(start: real, count: count)), i = Array(UnsafeBufferPointer(start: imaginary, count: count))
        transform(&r, &i, inverse: inverse)
        r.withUnsafeBufferPointer { real.update(from: $0.baseAddress!, count: count) }
        i.withUnsafeBufferPointer { imaginary.update(from: $0.baseAddress!, count: count) }
    }

    private func transformDFT(_ r: UnsafeMutablePointer<Double>, _ i: UnsafeMutablePointer<Double>, inverse: Bool,
                              scratch: UnsafeMutablePointer<Double>) {
        let count = rows * columns
        let columnSetup = (inverse ? columnPlan.dftInverse : columnPlan.dftForward)!
        let rowSetup = (inverse ? rowPlan.dftInverse : rowPlan.dftForward)!
        let aR = scratch, aI = scratch + count, bR = scratch + 2 * count, bI = scratch + 3 * count
        for row in 0..<rows {
            vDSP_DFT_ExecuteD(columnSetup, r + row * columns, i + row * columns, aR + row * columns, aI + row * columns)
        }
        vDSP_mtransD(aR, 1, bR, 1, vDSP_Length(columns), vDSP_Length(rows))
        vDSP_mtransD(aI, 1, bI, 1, vDSP_Length(columns), vDSP_Length(rows))
        for column in 0..<columns {
            vDSP_DFT_ExecuteD(rowSetup, bR + column * rows, bI + column * rows, aR + column * rows, aI + column * rows)
        }
        vDSP_mtransD(aR, 1, r, 1, vDSP_Length(rows), vDSP_Length(columns))
        vDSP_mtransD(aI, 1, i, 1, vDSP_Length(rows), vDSP_Length(columns))
        if inverse {
            var s = 1.0 / Double(count)
            vDSP_vsmulD(r, 1, &s, r, 1, vDSP_Length(count))
            vDSP_vsmulD(i, 1, &s, i, 1, vDSP_Length(count))
        }
    }
}
