//
//  FloatFFT.swift (CuboCore; the plan and `supports` public, the transforms package)
//  The Float32 2-D transform of the Float32 paths (Bragg `.single`, single-slice ptychography `.single`, the parallax shifter's
//  `shiftedSingle`): vDSP's complex DFT (vDSP_DFT_zop) along the rows, a transpose, and the DFT along what were the columns.
//  Forward is NumPy's fft2 (e^{-2πi…}, unnormalised); inverse is the unnormalised conjugate transform (the caller applies
//  1/(rows · columns)). The exact N-point DFT at the pattern's own size: nothing is padded. The public plan takes only the
//  lengths vDSP takes (f·2ⁿ, f ∈ {1, 3, 5, 15, …}; asked of vDSP itself at plan time); the package plan `init(anyRows:columns:)`
//  takes every length, running the others as Bluestein's chirp-z through a vDSP DFT of the shortest length M ≥ 2N − 1 vDSP takes
//  (384 for N = 165), with the chirp and its kernel computed in Double and rounded once.
//
//  Layout: the spectrum is left TRANSPOSED (columns × rows) — the second pass runs on the transposed grid and the transpose
//  back is skipped. A correlation multiplies two spectra elementwise and inverts, so every consumer that keeps to the
//  `forward` / `inverse` pair never sees the layout; `naturalSpectrum` undoes it for the one that needs the plain grid
//  (the multicorr refinement).
//

import Accelerate
import Foundation

/// A plan for one `rows × columns` grid, immutable after creation and shared by concurrent workers (vDSP's DFT setups are
/// shareable across threads); the buffers are the caller's.
/// Public: the plan and ``supports(rows:columns:)``, so a caller can ask before choosing a Float32 path; the transforms
/// themselves stay package access.
public final class FloatFFT2D: @unchecked Sendable {
    public let rows: Int, columns: Int
    private let rowLine: FloatLine        // length `columns`, along each row
    private let columnLine: FloatLine     // length `rows`, along each row of the transpose

    /// nil when vDSP has no DFT of one of the two lengths.
    public init?(rows: Int, columns: Int) {
        guard rows > 0, columns > 0, let r = FloatLine(direct: columns) else { return nil }
        let c: FloatLine
        if rows == columns { c = r } else if let other = FloatLine(direct: rows) { c = other } else { return nil }
        self.rows = rows; self.columns = columns
        rowLine = r; columnLine = c
    }

    /// Every positive size: a length vDSP's DFT takes runs on it as in ``init(rows:columns:)`` (the same calls, the same
    /// bits), any other through Bluestein.
    package init(anyRows rows: Int, columns: Int) {
        precondition(rows > 0 && columns > 0)
        self.rows = rows; self.columns = columns
        rowLine = FloatLine(direct: columns) ?? FloatLine(bluestein: columns)
        columnLine = rows == columns ? rowLine : (FloatLine(direct: rows) ?? FloatLine(bluestein: rows))
    }

    /// Whether vDSP can transform a grid of this size.
    public static func supports(rows: Int, columns: Int) -> Bool {
        FloatFFT2D(rows: rows, columns: columns) != nil
    }

    /// Transforms (aRe, aIm), a `rows × columns` grid; the result — transposed, `columns × rows` — is in (bRe, bIm). The
    /// input buffers are overwritten.
    package func forward(aRe: UnsafeMutablePointer<Float>, aIm: UnsafeMutablePointer<Float>,
                 bRe: UnsafeMutablePointer<Float>, bIm: UnsafeMutablePointer<Float>) {
        rowLine.pass(inverse: false, count: rows, aRe, aIm, bRe, bIm)
        vDSP_mtrans(bRe, 1, aRe, 1, vDSP_Length(columns), vDSP_Length(rows))
        vDSP_mtrans(bIm, 1, aIm, 1, vDSP_Length(columns), vDSP_Length(rows))
        columnLine.pass(inverse: false, count: columns, aRe, aIm, bRe, bIm)
    }

    /// The unnormalised inverse of `forward`'s layout: (aRe, aIm) holds a transposed spectrum; the natural-layout result is
    /// in (bRe, bIm). The input buffers are overwritten.
    package func inverse(aRe: UnsafeMutablePointer<Float>, aIm: UnsafeMutablePointer<Float>,
                 bRe: UnsafeMutablePointer<Float>, bIm: UnsafeMutablePointer<Float>) {
        columnLine.pass(inverse: true, count: columns, aRe, aIm, bRe, bIm)
        vDSP_mtrans(bRe, 1, aRe, 1, vDSP_Length(rows), vDSP_Length(columns))
        vDSP_mtrans(bIm, 1, aIm, 1, vDSP_Length(rows), vDSP_Length(columns))
        rowLine.pass(inverse: true, count: rows, aRe, aIm, bRe, bIm)
    }

    /// Copies a transposed spectrum (`columns × rows`) into the natural `rows × columns` layout.
    package func naturalSpectrum(fromRe: UnsafePointer<Float>, fromIm: UnsafePointer<Float>,
                         toRe: UnsafeMutablePointer<Float>, toIm: UnsafeMutablePointer<Float>) {
        vDSP_mtrans(fromRe, 1, toRe, 1, vDSP_Length(rows), vDSP_Length(columns))
        vDSP_mtrans(fromIm, 1, toIm, 1, vDSP_Length(rows), vDSP_Length(columns))
    }
}

/// One transform length: vDSP's DFT of length n, or Bluestein through vDSP's DFT of length m. Immutable after init.
private final class FloatLine {
    let n: Int
    private let forward: vDSP_DFT_Setup, inverse: vDSP_DFT_Setup    // length n (direct) or m (Bluestein)
    private let m: Int                                               // 0 when direct
    private let chirpRe: [Float], chirpIm: [Float], conjugateIm: [Float]   // w_k = e^{−iπ k²/n}; −Im w_k
    private let kernelRe: [Float], kernelIm: [Float]                 // DFT_m of conj(w), wrapped, divided by m

    init?(direct n: Int) {
        guard let f = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(n), .FORWARD) else { return nil }
        guard let i = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(n), .INVERSE) else { vDSP_DFT_DestroySetup(f); return nil }
        self.n = n; forward = f; inverse = i; m = 0
        chirpRe = []; chirpIm = []; conjugateIm = []; kernelRe = []; kernelIm = []
    }

    init(bluestein n: Int) {
        self.n = n
        var length = 0
        var f: vDSP_DFT_Setup?, i: vDSP_DFT_Setup?, d: vDSP_DFT_SetupD?
        for candidate in FloatLine.lengths(atLeast: 2 * n - 1) {   // the first length vDSP takes in Float both ways and in Double
            guard let cf = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(candidate), .FORWARD) else { continue }
            guard let ci = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(candidate), .INVERSE) else {
                vDSP_DFT_DestroySetup(cf); continue
            }
            guard let cd = vDSP_DFT_zop_CreateSetupD(nil, vDSP_Length(candidate), .FORWARD) else {
                vDSP_DFT_DestroySetup(cf); vDSP_DFT_DestroySetup(ci); continue
            }
            length = candidate; f = cf; i = ci; d = cd; break
        }
        guard let f, let i, let kernelSetup = d else { preconditionFailure("vDSP_DFT_zop takes the powers of two from 16") }
        defer { vDSP_DFT_DestroySetupD(kernelSetup) }
        m = length; forward = f; inverse = i
        // the chirp and the kernel in Double (k² mod 2n keeps the angle exact), rounded to Float once
        var cr = [Double](repeating: 0, count: n), ci = cr
        for k in 0..<n {
            let a = -Double.pi * Double((k * k) % (2 * n)) / Double(n)
            cr[k] = cos(a); ci[k] = sin(a)
        }
        var br = [Double](repeating: 0, count: length), bi = br
        for k in 0..<n {
            br[k] = cr[k]; bi[k] = -ci[k]
            if k > 0 { br[length - k] = cr[k]; bi[length - k] = -ci[k] }
        }
        var kr = [Double](repeating: 0, count: length), ki = kr
        vDSP_DFT_ExecuteD(kernelSetup, br, bi, &kr, &ki)
        let scale = 1 / Double(length)
        chirpRe = cr.map { Float($0) }; chirpIm = ci.map { Float($0) }; conjugateIm = ci.map { Float(-$0) }
        kernelRe = kr.map { Float($0 * scale) }; kernelIm = ki.map { Float($0 * scale) }
    }

    deinit { vDSP_DFT_DestroySetup(forward); vDSP_DFT_DestroySetup(inverse) }

    /// f · 2^k for f in {1, 3, 5, 15}, at least `minimum`, ascending (the first twelve of each, so the powers of two from 16
    /// that vDSP always takes are among them).
    static func lengths(atLeast minimum: Int) -> [Int] {
        var out = Set<Int>()
        for f in [1, 3, 5, 15] {
            var v = f
            while v < minimum { v *= 2 }
            for _ in 0..<12 { out.insert(v); v *= 2 }
        }
        return out.sorted()
    }

    /// `count` contiguous lines of length n, out of place; `inverse` is the unnormalised conjugate transform.
    func pass(inverse isInverse: Bool, count: Int,
              _ inRe: UnsafeMutablePointer<Float>, _ inIm: UnsafeMutablePointer<Float>,
              _ outRe: UnsafeMutablePointer<Float>, _ outIm: UnsafeMutablePointer<Float>) {
        let setup = isInverse ? inverse : forward
        if m == 0 {
            for line in 0..<count {
                vDSP_DFT_Execute(setup, inRe + line * n, inIm + line * n, outRe + line * n, outIm + line * n)
            }
            return
        }
        withUnsafeTemporaryAllocation(byteCount: 16 * m, alignment: 16) { raw in   // 16-byte aligned for vDSP
            let aR = raw.baseAddress!.bindMemory(to: Float.self, capacity: 4 * m), aI = aR + m, bR = aR + 2 * m, bI = aR + 3 * m
            chirpRe.withUnsafeBufferPointer { cr in chirpIm.withUnsafeBufferPointer { ci in
            conjugateIm.withUnsafeBufferPointer { cj in
            kernelRe.withUnsafeBufferPointer { kr in kernelIm.withUnsafeBufferPointer { ki in
                var w = DSPSplitComplex(realp: UnsafeMutablePointer(mutating: cr.baseAddress!),
                                        imagp: UnsafeMutablePointer(mutating: ci.baseAddress!))
                var wc = DSPSplitComplex(realp: UnsafeMutablePointer(mutating: cr.baseAddress!),
                                         imagp: UnsafeMutablePointer(mutating: cj.baseAddress!))
                var kernel = DSPSplitComplex(realp: UnsafeMutablePointer(mutating: kr.baseAddress!),
                                             imagp: UnsafeMutablePointer(mutating: ki.baseAddress!))
                var a = DSPSplitComplex(realp: aR, imagp: aI), b = DSPSplitComplex(realp: bR, imagp: bI)
                for line in 0..<count {
                    var x = DSPSplitComplex(realp: inRe + line * n, imagp: inIm + line * n)
                    var out = DSPSplitComplex(realp: outRe + line * n, imagp: outIm + line * n)
                    // a = x · w, or conj(x) · w for the inverse (vDSP_zvmul's −1 conjugates its first operand)
                    vDSP_zvmul(&x, 1, &w, 1, &a, 1, vDSP_Length(n), isInverse ? -1 : 1)
                    vDSP_vclr(aR + n, 1, vDSP_Length(m - n))
                    vDSP_vclr(aI + n, 1, vDSP_Length(m - n))
                    vDSP_DFT_Execute(forward, aR, aI, bR, bI)
                    vDSP_zvmul(&b, 1, &kernel, 1, &a, 1, vDSP_Length(m), 1)
                    vDSP_DFT_Execute(inverse, aR, aI, bR, bI)
                    if isInverse {   // conj(b · w) = conj(b) · conj(w)
                        vDSP_zvmul(&b, 1, &wc, 1, &out, 1, vDSP_Length(n), -1)
                    } else {
                        vDSP_zvmul(&b, 1, &w, 1, &out, 1, vDSP_Length(n), 1)
                    }
                }
            } } } } }
        }
    }
}
