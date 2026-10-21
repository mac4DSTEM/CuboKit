//
//  SingleFFT.swift
//  The Float32 2-D transform of the `.single` detection path: vDSP's complex DFT (vDSP_DFT_zop) along the rows, a transpose,
//  and the DFT along what were the columns. Forward is NumPy's fft2 (e^{-2πi…}, unnormalised); inverse is the unnormalised
//  conjugate transform (the caller applies 1/(rows · columns)). The exact N-point DFT at the pattern's own size: nothing is
//  padded, so only lengths vDSP supports (f·2ⁿ, f ∈ {1, 3, 5, 15, …}; asked of vDSP itself at plan time) are available.
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
final class SingleFFT2D: @unchecked Sendable {
    let rows: Int, columns: Int
    private let rowForward: vDSP_DFT_Setup, rowInverse: vDSP_DFT_Setup     // length `columns`, along each row
    private let columnForward: vDSP_DFT_Setup, columnInverse: vDSP_DFT_Setup   // length `rows`, along each row of the transpose

    /// nil when vDSP has no DFT of one of the two lengths.
    init?(rows: Int, columns: Int) {
        guard rows > 0, columns > 0,
              let rf = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(columns), .FORWARD),
              let ri = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(columns), .INVERSE),
              let cf = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(rows), .FORWARD),
              let ci = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(rows), .INVERSE) else { return nil }
        self.rows = rows; self.columns = columns
        rowForward = rf; rowInverse = ri; columnForward = cf; columnInverse = ci
    }

    deinit {
        vDSP_DFT_DestroySetup(rowForward); vDSP_DFT_DestroySetup(rowInverse)
        vDSP_DFT_DestroySetup(columnForward); vDSP_DFT_DestroySetup(columnInverse)
    }

    /// Whether vDSP can transform a grid of this size.
    static func supports(rows: Int, columns: Int) -> Bool {
        SingleFFT2D(rows: rows, columns: columns) != nil
    }

    /// Transforms (aRe, aIm), a `rows × columns` grid; the result — transposed, `columns × rows` — is in (bRe, bIm). The
    /// input buffers are overwritten.
    func forward(aRe: UnsafeMutablePointer<Float>, aIm: UnsafeMutablePointer<Float>,
                 bRe: UnsafeMutablePointer<Float>, bIm: UnsafeMutablePointer<Float>) {
        pass(rowForward, length: columns, count: rows, aRe, aIm, bRe, bIm)
        vDSP_mtrans(bRe, 1, aRe, 1, vDSP_Length(columns), vDSP_Length(rows))
        vDSP_mtrans(bIm, 1, aIm, 1, vDSP_Length(columns), vDSP_Length(rows))
        pass(columnForward, length: rows, count: columns, aRe, aIm, bRe, bIm)
    }

    /// The unnormalised inverse of `forward`'s layout: (aRe, aIm) holds a transposed spectrum; the natural-layout result is
    /// in (bRe, bIm). The input buffers are overwritten.
    func inverse(aRe: UnsafeMutablePointer<Float>, aIm: UnsafeMutablePointer<Float>,
                 bRe: UnsafeMutablePointer<Float>, bIm: UnsafeMutablePointer<Float>) {
        pass(columnInverse, length: rows, count: columns, aRe, aIm, bRe, bIm)
        vDSP_mtrans(bRe, 1, aRe, 1, vDSP_Length(rows), vDSP_Length(columns))
        vDSP_mtrans(bIm, 1, aIm, 1, vDSP_Length(rows), vDSP_Length(columns))
        pass(rowInverse, length: columns, count: rows, aRe, aIm, bRe, bIm)
    }

    /// Copies a transposed spectrum (`columns × rows`) into the natural `rows × columns` layout.
    func naturalSpectrum(fromRe: UnsafePointer<Float>, fromIm: UnsafePointer<Float>,
                         toRe: UnsafeMutablePointer<Float>, toIm: UnsafeMutablePointer<Float>) {
        vDSP_mtrans(fromRe, 1, toRe, 1, vDSP_Length(rows), vDSP_Length(columns))
        vDSP_mtrans(fromIm, 1, toIm, 1, vDSP_Length(rows), vDSP_Length(columns))
    }

    private func pass(_ setup: vDSP_DFT_Setup, length: Int, count: Int,
                      _ inRe: UnsafeMutablePointer<Float>, _ inIm: UnsafeMutablePointer<Float>,
                      _ outRe: UnsafeMutablePointer<Float>, _ outIm: UnsafeMutablePointer<Float>) {
        for line in 0..<count {
            vDSP_DFT_Execute(setup, inRe + line * length, inIm + line * length, outRe + line * length, outIm + line * length)
        }
    }
}
