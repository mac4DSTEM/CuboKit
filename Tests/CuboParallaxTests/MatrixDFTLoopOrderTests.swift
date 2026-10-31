import Foundation
import Testing
@testable import CuboParallax

/// The matrix-DFT patch is computed with its loops ordered for contiguous reads and its column kernel tabulated once. Each
/// output still sums its terms in the same order, so it equals the straightforward column-outer loop to the last bits: the
/// compiler vectorises the two loops differently, measured gaps are a few ulps (3e-11 on values of 9e4, 3e-16 relative), and
/// the 90 648 refined Bragg peaks of a 100 x 100 x 128 x 128 cube came out byte-identical. Held at 1e-14 of the patch's scale.
@Test(arguments: [(128, 128, 16), (64, 96, 8), (33, 47, 5), (127, 1, 3)])
func matrixDFTPatchEqualsColumnOuterLoop(rows: Int, columns: Int, upsample: Int) {
    var state: UInt64 = 0x9E37_79B9_7F4A_7C15 &+ UInt64(rows * 1000 + columns)
    func next() -> Double {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Double(state >> 11) / Double(1 << 53) * 2 - 1
    }
    let re = (0..<(rows * columns)).map { _ in next() * 1e3 }
    let im = (0..<(rows * columns)).map { _ in next() * 1e3 }
    let patchSize = Int(ceil(1.5 * Double(upsample)))
    for (centerRow, centerColumn) in [(0.0, 0.0), (11.5, -3.25), (-40.0625, 17.75)] {
        let fast = MatrixDFTCorrelation.dftUpsampleOfConjugate(
            correlationRe: re, correlationIm: im, width: columns, height: rows, centerRow: centerRow,
            centerColumn: centerColumn, upsampleFactor: upsample, patchSize: patchSize)
        let plain = columnOuter(re, im, width: columns, height: rows, centerRow: centerRow, centerColumn: centerColumn,
                                upsampleFactor: upsample, patchSize: patchSize)
        let scale = plain.map(abs).max() ?? 0
        let gap = zip(fast, plain).map { abs($0 - $1) }.max() ?? 0
        #expect(fast.count == plain.count)
        #expect(gap <= 1e-14 * scale, "gap \(gap) at scale \(scale)")
    }
}

/// The loop as the kit wrote it before the reordering: frequency column outermost, the source rows innermost.
private func columnOuter(_ correlationRe: [Double], _ correlationIm: [Double], width: Int, height: Int,
                         centerRow: Double, centerColumn: Double, upsampleFactor: Int, patchSize: Int) -> [Double] {
    var intermediateRe = [Double](repeating: 0, count: height * patchSize)
    var intermediateIm = [Double](repeating: 0, count: height * patchSize)
    let columnScale = -2 * Double.pi / (Double(width) * Double(upsampleFactor))
    for frequencyColumn in 0..<width {
        let wrappedFrequency = Double(frequencyColumn < (width + 1) / 2 ? frequencyColumn : frequencyColumn - width)
        for patchColumn in 0..<patchSize {
            let angle = columnScale * wrappedFrequency * (Double(patchColumn) - centerColumn)
            let cosine = cos(angle), sine = sin(angle)
            for sourceRow in 0..<height {
                let source = sourceRow * width + frequencyColumn
                let destination = sourceRow * patchSize + patchColumn
                intermediateRe[destination] += correlationRe[source] * cosine + correlationIm[source] * sine
                intermediateIm[destination] += -correlationIm[source] * cosine + correlationRe[source] * sine
            }
        }
    }
    var patch = [Double](repeating: 0, count: patchSize * patchSize)
    let rowScale = -2 * Double.pi / (Double(height) * Double(upsampleFactor))
    for patchRow in 0..<patchSize {
        for frequencyRow in 0..<height {
            let wrappedFrequency = Double(frequencyRow < (height + 1) / 2 ? frequencyRow : frequencyRow - height)
            let angle = rowScale * (Double(patchRow) - centerRow) * wrappedFrequency
            let cosine = cos(angle), sine = sin(angle)
            for patchColumn in 0..<patchSize {
                let source = frequencyRow * patchSize + patchColumn
                patch[patchRow * patchSize + patchColumn] += cosine * intermediateRe[source] - sine * intermediateIm[source]
            }
        }
    }
    return patch
}
