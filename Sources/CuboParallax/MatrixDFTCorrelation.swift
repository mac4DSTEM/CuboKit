//
//  MatrixDFTCorrelation.swift
//  Double-precision py4DSTEM-compatible matrix-DFT refinement.
//  Provenance and reference pin: NOTICE and docs/parallax-alignment.md.
//

import Foundation
import CuboCore

struct CorrelationPeak: Equatable, Sendable {
    let row: Double
    let column: Double

    init(row: Double, column: Double) {
        self.row = row
        self.column = column
    }
}

/// Guizar-Sicairos matrix-multiply DFT refinement as used by py4DSTEM's
/// `align_images_fourier` / `upsampled_correlation` path.
enum MatrixDFTCorrelation {
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
    private static func refine(
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
        var intermediateRe = [Double](repeating: 0,
                                     count: height * patchSize)
        var intermediateIm = [Double](repeating: 0,
                                     count: height * patchSize)
        let columnScale = -2 * Double.pi
            / (Double(width) * Double(upsampleFactor))
        for frequencyColumn in 0..<width {
            // Matches numpy's ifftshift(arange(N)) - floor(N/2), as py4DSTEM's
            // multicorr.py:186,195. On an odd axis the middle bin is +(N-1)/2, not
            // -(N+1)/2.
            let wrappedFrequency = Double(
                frequencyColumn < (width + 1) / 2
                    ? frequencyColumn : frequencyColumn - width
            )
            for patchColumn in 0..<patchSize {
                let angle = columnScale * wrappedFrequency
                    * (Double(patchColumn) - centerColumn)
                let cosine = cos(angle), sine = sin(angle)
                for sourceRow in 0..<height {
                    let source = sourceRow * width + frequencyColumn
                    let destination = sourceRow * patchSize + patchColumn
                    // conj(correlation) × column kernel
                    intermediateRe[destination] += correlationRe[source] * cosine
                        + correlationIm[source] * sine
                    intermediateIm[destination] += -correlationIm[source] * cosine
                        + correlationRe[source] * sine
                }
            }
        }

        var patch = [Double](repeating: 0, count: patchSize * patchSize)
        let rowScale = -2 * Double.pi
            / (Double(height) * Double(upsampleFactor))
        for patchRow in 0..<patchSize {
            for frequencyRow in 0..<height {
                let wrappedFrequency = Double(
                    frequencyRow < (height + 1) / 2
                        ? frequencyRow : frequencyRow - height
                )
                let angle = rowScale * (Double(patchRow) - centerRow)
                    * wrappedFrequency
                let cosine = cos(angle), sine = sin(angle)
                for patchColumn in 0..<patchSize {
                    let source = frequencyRow * patchSize + patchColumn
                    patch[patchRow * patchSize + patchColumn]
                        += cosine * intermediateRe[source]
                            - sine * intermediateIm[source]
                }
            }
        }
        return patch
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
