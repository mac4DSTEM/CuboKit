//
//  MatrixDFTCorrelation.swift
//  Double-precision py4DSTEM-compatible matrix-DFT refinement.
//  Provenance and reference pin: NOTICE and docs/parallax-alignment.md.
//

import Accelerate
import Foundation
import CuboCore

package struct CorrelationPeak: Equatable, Sendable {
    package let row: Double
    package let column: Double

    package init(row: Double, column: Double) {
        self.row = row
        self.column = column
    }
}

/// Guizar-Sicairos matrix-multiply DFT refinement as used by py4DSTEM's
/// `align_images_fourier` / `upsampled_correlation` path.
package enum MatrixDFTCorrelation {
    /// Find the circular-correlation maximum, apply py4DSTEM's three-point
    /// parabolic estimate, round it to a half pixel, and refine it on a small
    /// matrix-DFT patch.
    static func refinedPeak(
        correlationRe: [Double],
        correlationIm: [Double],
        width: Int,
        height: Int,
        fft: FFT2D,
        upsampleFactor: Int
    ) throws -> CorrelationPeak {
        precondition(width > 0 && height > 0)
        precondition(correlationRe.count == width * height)
        precondition(correlationIm.count == correlationRe.count)
        precondition(fft.columns == width && fft.rows == height)

        var real = correlationRe
        var imaginary = correlationIm
        fft.transform(&real, &imaginary, inverse: true)
        guard real.allSatisfy(\.isFinite) else { throw ParallaxAlignmentError.nonfiniteValues }
        guard real.min() != real.max() else { throw ParallaxAlignmentError.degenerateCorrelation }
        var maximum = 0
        for index in 1..<real.count where real[index] > real[maximum] {
            maximum = index
        }
        let row = maximum / width
        let column = maximum % width
        let rowOffset = parabolicOffset(
            previous: real[positiveModulo(row - 1, height) * width + column],
            center: real[maximum],
            next: real[positiveModulo(row + 1, height) * width + column]
        )
        let columnOffset = parabolicOffset(
            previous: real[row * width + positiveModulo(column - 1, width)],
            center: real[maximum],
            next: real[row * width + positiveModulo(column + 1, width)]
        )
        return try refine(
            correlationRe: correlationRe, correlationIm: correlationIm,
            width: width, height: height,
            initial: CorrelationPeak(
                row: Double(row) + rowOffset,
                column: Double(column) + columnOffset
            ),
            upsampleFactor: upsampleFactor
        )
    }

    /// Refine the initial parabolic peak on a local matrix-DFT patch.
    package static func refine(
        correlationRe: [Double],
        correlationIm: [Double],
        width: Int,
        height: Int,
        initial: CorrelationPeak,
        upsampleFactor: Int
    ) throws -> CorrelationPeak {
        guard upsampleFactor > 2 else { return initial }
        precondition(width > 0 && height > 0)
        precondition(correlationRe.count == width * height)
        precondition(correlationIm.count == correlationRe.count)

        let u = Double(upsampleFactor)
        // NumPy uses ties-to-even rounding at both stages.
        var row = ((initial.row * 2).rounded(.toNearestOrEven) / 2)
        var column = ((initial.column * 2).rounded(.toNearestOrEven) / 2)
        row = (row * u).rounded(.toNearestOrEven) / u
        column = (column * u).rounded(.toNearestOrEven) / u

        let patchSize = Int(ceil(1.5 * u))
        let globalShift = floor(Double(patchSize) / 2)
        let patch = dftUpsampleOfConjugate(
            correlationRe: correlationRe, correlationIm: correlationIm,
            width: width, height: height,
            centerRow: globalShift - u * row,
            centerColumn: globalShift - u * column,
            upsampleFactor: upsampleFactor,
            patchSize: patchSize
        )

        guard patch.allSatisfy(\.isFinite) else { throw ParallaxAlignmentError.nonfiniteValues }
        var maximum = 0
        for index in 1..<patch.count where patch[index] > patch[maximum] {
            maximum = index
        }
        let peakRow = maximum / patchSize
        let peakColumn = maximum % patchSize
        var subRow = Double(peakRow)
        var subColumn = Double(peakColumn)
        // py4DSTEM falls back to zero polish when the 3x3 patch crosses an
        // edge. Keeping both offsets at zero matches that single try/except.
        if peakRow > 0, peakRow < patchSize - 1,
           peakColumn > 0, peakColumn < patchSize - 1 {
            let center = patch[maximum]
            subRow += parabolicOffset(
                previous: patch[maximum - patchSize], center: center,
                next: patch[maximum + patchSize]
            )
            subColumn += parabolicOffset(
                previous: patch[maximum - 1], center: center,
                next: patch[maximum + 1]
            )
        }
        return CorrelationPeak(
            row: row + (subRow - globalShift) / u,
            column: column + (subColumn - globalShift) / u
        )
    }

    /// real(rowKern · conj(correlation) · colKern). The center is expressed
    /// in upsampled-grid coordinates, matching py4DSTEM's `dftUpsample`.
    /// Computed as real matrix products on Accelerate, as py4DSTEM uses NumPy's
    /// complex matmul; the sums run in the library's order, not term by term.
    static func dftUpsampleOfConjugate(
        correlationRe: [Double],
        correlationIm: [Double],
        width: Int,
        height: Int,
        centerRow: Double,
        centerColumn: Double,
        upsampleFactor: Int,
        patchSize: Int
    ) -> [Double] {
        // vDSP reads the arrays through raw pointers.
        precondition(correlationRe.count == width * height && correlationIm.count == width * height)
        precondition(width > 0 && height > 0 && patchSize > 0)
        let p = patchSize

        // Stage 1, conj(C) · kernc, as two real products into one height x 2p block [Re | Im]:
        //   Re = Cre · cos + Cim · sin,   Im = Cre · sin - Cim · cos,
        // with B1 = [cos | sin] and B2 = [sin | -cos], each width x 2p.
        // Frequencies match numpy's ifftshift(arange(N)) - floor(N/2), as py4DSTEM's multicorr.py:186,195. On an odd axis
        // the middle bin is +(N-1)/2, not -(N+1)/2. The angles are formed as the scalar loop formed them; vForce takes the
        // sine and cosine.
        let columnScale = -2 * Double.pi / (Double(width) * Double(upsampleFactor))
        var columnAngle = [Double](repeating: 0, count: width * p)
        for frequencyColumn in 0..<width {
            let wrappedFrequency = Double(frequencyColumn < (width + 1) / 2 ? frequencyColumn : frequencyColumn - width)
            for patchColumn in 0..<p {
                columnAngle[frequencyColumn * p + patchColumn] = columnScale * wrappedFrequency
                    * (Double(patchColumn) - centerColumn)
            }
        }
        let (columnSin, columnCos) = sinCos(columnAngle)
        var kernelRe = [Double](repeating: 0, count: width * 2 * p)
        var kernelIm = [Double](repeating: 0, count: width * 2 * p)
        for frequencyColumn in 0..<width {
            let source = frequencyColumn * p, row = frequencyColumn * 2 * p
            for patchColumn in 0..<p {
                let cosine = columnCos[source + patchColumn], sine = columnSin[source + patchColumn]
                kernelRe[row + patchColumn] = cosine
                kernelRe[row + p + patchColumn] = sine
                kernelIm[row + patchColumn] = sine
                kernelIm[row + p + patchColumn] = -cosine
            }
        }
        var fromRe = [Double](repeating: 0, count: height * 2 * p)
        var fromIm = [Double](repeating: 0, count: height * 2 * p)
        vDSP_mmulD(correlationRe, 1, kernelRe, 1, &fromRe, 1,
                   vDSP_Length(height), vDSP_Length(2 * p), vDSP_Length(width))
        vDSP_mmulD(correlationIm, 1, kernelIm, 1, &fromIm, 1,
                   vDSP_Length(height), vDSP_Length(2 * p), vDSP_Length(width))
        // The intermediate, restacked as [Re; Im] (2 height x p) for stage 2.
        var intermediate = [Double](repeating: 0, count: 2 * height * p)
        for sourceRow in 0..<height {
            let row = sourceRow * 2 * p
            for patchColumn in 0..<p {
                intermediate[sourceRow * p + patchColumn] = fromRe[row + patchColumn] + fromIm[row + patchColumn]
                intermediate[(height + sourceRow) * p + patchColumn] = fromRe[row + p + patchColumn]
                    + fromIm[row + p + patchColumn]
            }
        }

        // Stage 2, real(kernr · intermediate) = [cos | -sin] · [Re; Im], kernr being p x height.
        let rowScale = -2 * Double.pi / (Double(height) * Double(upsampleFactor))
        var rowAngle = [Double](repeating: 0, count: p * height)
        for patchRow in 0..<p {
            for frequencyRow in 0..<height {
                let wrappedFrequency = Double(frequencyRow < (height + 1) / 2 ? frequencyRow : frequencyRow - height)
                rowAngle[patchRow * height + frequencyRow] = rowScale * (Double(patchRow) - centerRow) * wrappedFrequency
            }
        }
        let (rowSin, rowCos) = sinCos(rowAngle)
        var rowKernel = [Double](repeating: 0, count: p * 2 * height)
        for patchRow in 0..<p {
            for frequencyRow in 0..<height {
                rowKernel[patchRow * 2 * height + frequencyRow] = rowCos[patchRow * height + frequencyRow]
                rowKernel[patchRow * 2 * height + height + frequencyRow] = -rowSin[patchRow * height + frequencyRow]
            }
        }
        var patch = [Double](repeating: 0, count: p * p)
        vDSP_mmulD(rowKernel, 1, intermediate, 1, &patch, 1, vDSP_Length(p), vDSP_Length(p), vDSP_Length(2 * height))
        return patch
    }

    /// Sine and cosine of every angle, with vForce.
    private static func sinCos(_ angle: [Double]) -> (sine: [Double], cosine: [Double]) {
        var sine = [Double](repeating: 0, count: angle.count)
        var cosine = [Double](repeating: 0, count: angle.count)
        var count = Int32(angle.count)
        vvsincos(&sine, &cosine, angle, &count)
        return (sine, cosine)
    }

    private static func parabolicOffset(
        previous: Double, center: Double, next: Double
    ) -> Double {
        let denominator = 4 * center - 2 * next - 2 * previous
        let offset = (next - previous) / denominator
        return offset.isFinite ? offset : 0
    }

    private static func positiveModulo(_ value: Int, _ modulus: Int) -> Int {
        let remainder = value % modulus
        return remainder >= 0 ? remainder : remainder + modulus
    }
}
