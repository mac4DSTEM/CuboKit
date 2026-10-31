import Foundation
import Testing
@testable import CuboParallax

/// The matrix-DFT patch is three real matrix products on Accelerate, as py4DSTEM computes `kernr @ imageCorr @ kernc` with
/// NumPy's matmul. The products sum in their own order, so the patch differs from the scalar loops below in the last bits
/// (measured at most 3e-15 of the patch's largest value); it is held to both at the tolerance the kit uses for this function
/// against py4DSTEM (1e-10 of the patch's largest value, with a unit floor; docs/parity.md).
@Test(arguments: [(128, 128, 16), (64, 96, 8), (33, 47, 5), (127, 1, 3), (1, 9, 3)])
func matrixDFTPatchEqualsScalarLoops(rows: Int, columns: Int, upsample: Int) {
    let (re, im) = randomCorrelation(rows: rows, columns: columns)
    let patchSize = Int(ceil(1.5 * Double(upsample)))
    for (centerRow, centerColumn) in [(0.0, 0.0), (11.5, -3.25), (-40.0625, 17.75)] {
        let fast = MatrixDFTCorrelation.dftUpsampleOfConjugate(
            correlationRe: re, correlationIm: im, width: columns, height: rows, centerRow: centerRow,
            centerColumn: centerColumn, upsampleFactor: upsample, patchSize: patchSize)
        for reference in [columnOuter, contiguousLoop] {
            let plain = reference(re, im, columns, rows, centerRow, centerColumn, upsample, patchSize)
            let scale = max(1, plain.map(abs).max() ?? 0)
            let gap = zip(fast, plain).map { abs($0 - $1) }.max() ?? 0
            #expect(fast.count == plain.count)
            #expect(fast.allSatisfy(\.isFinite) && gap <= 1e-10 * scale, "gap \(gap) at scale \(scale)")
        }
    }
}

/// The per-peak cost of the multicorr refinement: on a 128 x 128 correlation at upsampling 16, the matrix-product patch is
/// several times faster than the scalar loop it replaced (measured 7x: 37 against 260 microseconds a patch); held at 2x,
/// best of seven interleaved runs, so a busy machine does not flip it.
@Test func matrixDFTPatchIsFasterThanScalarLoop() {
    let rows = 128, columns = 128, upsample = 16
    let (re, im) = randomCorrelation(rows: rows, columns: columns)
    let patchSize = Int(ceil(1.5 * Double(upsample)))
    let clock = ContinuousClock()
    var fastBest = Duration.seconds(1_000), loopBest = Duration.seconds(1_000)
    var sink = 0.0
    for _ in 0..<7 {
        let fastTime = clock.measure {
            for k in 0..<20 {
                sink += MatrixDFTCorrelation.dftUpsampleOfConjugate(
                    correlationRe: re, correlationIm: im, width: columns, height: rows, centerRow: 11.5 + Double(k),
                    centerColumn: -3.25, upsampleFactor: upsample, patchSize: patchSize)[0]
            }
        }
        let loopTime = clock.measure {
            for k in 0..<20 {
                sink += contiguousLoop(re, im, columns, rows, 11.5 + Double(k), -3.25, upsample, patchSize)[0]
            }
        }
        fastBest = min(fastBest, fastTime)
        loopBest = min(loopBest, loopTime)
    }
    #expect(sink.isFinite)
    #expect(fastBest * 2 <= loopBest, "matrix product \(fastBest), scalar loop \(loopBest) per 20 patches")
}

private func randomCorrelation(rows: Int, columns: Int) -> ([Double], [Double]) {
    var state: UInt64 = 0x9E37_79B9_7F4A_7C15 &+ UInt64(rows * 1000 + columns)
    func next() -> Double {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Double(state >> 11) / Double(1 << 53) * 2 - 1
    }
    let re = (0..<(rows * columns)).map { _ in next() * 1e3 }
    let im = (0..<(rows * columns)).map { _ in next() * 1e3 }
    return (re, im)
}

/// The loop as the kit wrote it first: frequency column outermost, the source rows innermost.
private func columnOuter(_ correlationRe: [Double], _ correlationIm: [Double], _ width: Int, _ height: Int,
                         _ centerRow: Double, _ centerColumn: Double, _ upsampleFactor: Int, _ patchSize: Int) -> [Double] {
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
    return rowStage(intermediateRe, intermediateIm, height, centerRow, upsampleFactor, patchSize)
}

/// The scalar loop the matrix products replaced: source row, frequency column, patch column, the column kernel tabulated once.
private func contiguousLoop(_ correlationRe: [Double], _ correlationIm: [Double], _ width: Int, _ height: Int,
                            _ centerRow: Double, _ centerColumn: Double, _ upsampleFactor: Int, _ patchSize: Int) -> [Double] {
    let columnScale = -2 * Double.pi / (Double(width) * Double(upsampleFactor))
    var columnCos = [Double](repeating: 0, count: width * patchSize)
    var columnSin = [Double](repeating: 0, count: width * patchSize)
    for frequencyColumn in 0..<width {
        let wrappedFrequency = Double(frequencyColumn < (width + 1) / 2 ? frequencyColumn : frequencyColumn - width)
        for patchColumn in 0..<patchSize {
            let angle = columnScale * wrappedFrequency * (Double(patchColumn) - centerColumn)
            columnCos[frequencyColumn * patchSize + patchColumn] = cos(angle)
            columnSin[frequencyColumn * patchSize + patchColumn] = sin(angle)
        }
    }
    var intermediateRe = [Double](repeating: 0, count: height * patchSize)
    var intermediateIm = [Double](repeating: 0, count: height * patchSize)
    intermediateRe.withUnsafeMutableBufferPointer { outRe in
        intermediateIm.withUnsafeMutableBufferPointer { outIm in
            correlationRe.withUnsafeBufferPointer { inRe in
                correlationIm.withUnsafeBufferPointer { inIm in
                    columnCos.withUnsafeBufferPointer { kc in
                        columnSin.withUnsafeBufferPointer { ks in
                            for sourceRow in 0..<height {
                                let row = sourceRow * patchSize
                                for frequencyColumn in 0..<width {
                                    let re = inRe[sourceRow * width + frequencyColumn]
                                    let im = inIm[sourceRow * width + frequencyColumn]
                                    let kernel = frequencyColumn * patchSize
                                    for patchColumn in 0..<patchSize {
                                        let cosine = kc[kernel + patchColumn], sine = ks[kernel + patchColumn]
                                        outRe[row + patchColumn] += re * cosine + im * sine
                                        outIm[row + patchColumn] += -im * cosine + re * sine
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return rowStage(intermediateRe, intermediateIm, height, centerRow, upsampleFactor, patchSize)
}

/// The second stage both scalar loops share: the row kernel applied to the intermediate, real part kept.
private func rowStage(_ intermediateRe: [Double], _ intermediateIm: [Double], _ height: Int, _ centerRow: Double,
                      _ upsampleFactor: Int, _ patchSize: Int) -> [Double] {
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
