import CuboCore
import Foundation

extension ParallaxSubpixel {
    /// The source application's position-correction loop (``ParallaxPositionCorrectionVariant/application``).
    ///
    /// Offsets, steps, trial coordinates and scores are Float, as in the application; each deposition is the Double
    /// kernel density at the Float offsets, rounded to Float for scoring. Mean scores are Float sums in index order.
    static func correctPositionsApplication(images: [[Double]], rows: Int, columns: Int, shifts: [ParallaxShift],
        window doubleWindow: [Double], factor doubleFactor: Double, sigmaPixels: Double, lanczosOrder: Int?, lowpass: Bool,
        options: ParallaxPositionCorrectionOptions, isCancelled: (() -> Bool)?,
        progress: ((_ completed: Int, _ total: Int) -> Void)?) throws -> ParallaxPositionCorrectionResult {
        guard options.gaussianSigma == nil, options.lowpass == nil, options.highpass == nil,
            options.initialStep.isFinite, options.minimumStep.isFinite, options.stepFactor.isFinite,
            doubleWindow.allSatisfy(\.isFinite)
        else { throw ParallaxError.invalidInput }
        let n = rows * columns, factor = Float(doubleFactor)
        let window = doubleWindow.map(Float.init)
        let stack: [[Float]] = images.map { $0.map(Float.init) }
        let totalShifts = shifts.map { (row: Float($0.row), column: Float($0.column)) }
        var probeRows = [Float](repeating: 0, count: n), probeColumns = probeRows
        var step = [Float](repeating: Float(options.initialStep), count: n)
        let minimumStep = Float(options.minimumStep), stepFactor = Float(options.stepFactor)

        func deposit() throws -> (double: ParallaxSubpixelResult, float: [Float]) {
            try parallaxCheckCancelled(isCancelled)
            let result = try kernelDensity(images: images, rows: rows, columns: columns, shifts: shifts, factor: doubleFactor,
                sigmaPixels: sigmaPixels, lanczosOrder: lanczosOrder, lowpass: lowpass,
                probeOffsets: ParallaxProbeOffsets(rows: probeRows.map(Double.init), columns: probeColumns.map(Double.init)))
            return (result, result.image.map(Float.init))
        }
        func scoreMap(_ image: [Float], outRows: Int, outColumns: Int, _ dr: [Float], _ dc: [Float], window: [Float]?) throws -> [Float] {
            try parallaxCheckCancelled(isCancelled)
            var scores = [Float](repeating: 0, count: n)
            for row in 0..<rows { for column in 0..<columns {
                let position = row * columns + column
                var score: Float = 0
                for b in stack.indices {
                    let sampled = bilinear(image, rows: outRows, columns: outColumns,
                        row: (Float(row) + dr[position] + totalShifts[b].row) * factor,
                        column: (Float(column) + dc[position] + totalShifts[b].column) * factor)
                    score += abs(sampled - stack[b][position])
                }
                score /= Float(stack.count)
                scores[position] = score * (window?[position] ?? 1)
            } }
            return scores
        }
        func mean(_ values: [Float]) -> Float { values.reduce(0, +) / Float(values.count) }

        var (kde, pixels) = try deposit()
        var scores = try scoreMap(pixels, outRows: kde.rows, outColumns: kde.columns, probeRows, probeColumns, window: window)
        var history = [mean(scores)]
        let directions: [(Float, Float)] = options.checkerboard ? [(-1, 0), (1, 0), (0, -1), (0, 1)]
            : [(-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)]
        for iteration in 0..<options.iterations {
            try parallaxCheckCancelled(isCancelled)
            var trials: [[Float]] = []
            for direction in directions {
                trials.append(try scoreMap(pixels, outRows: kde.rows, outColumns: kde.columns,
                    zip(probeRows, step).map { $0.0 + direction.0 * $0.1 },
                    zip(probeColumns, step).map { $0.0 + direction.1 * $0.1 }, window: nil))
            }
            var update = [Bool](repeating: false, count: n)
            if options.checkerboard {
                for index in 0..<n {
                    var best = 0, bestScore = trials[0][index] * window[index]
                    for direction in 1..<4 {
                        let value = trials[direction][index] * window[index]
                        if value < bestScore { best = direction; bestScore = value }
                    }
                    if bestScore < scores[index] {
                        update[index] = true
                        probeRows[index] += directions[best].0 * step[index] * window[index]
                        probeColumns[index] += directions[best].1 * step[index] * window[index]
                    }
                }
            } else {
                var deltaRows = [Float](repeating: 0, count: n), deltaColumns = deltaRows
                for index in 0..<n {
                    let gradientRow = trials[0][index] - trials[1][index]
                    let gradientColumn = trials[2][index] - trials[3][index]
                    let denominator = hypot(gradientRow, gradientColumn) / step[index]
                    if denominator.isFinite, denominator > 0 {
                        deltaRows[index] = gradientRow * window[index] / denominator
                        deltaColumns[index] = gradientColumn * window[index] / denominator
                    }
                }
                let fixed = try scoreMap(pixels, outRows: kde.rows, outColumns: kde.columns,
                    zip(probeRows, deltaRows).map(+), zip(probeColumns, deltaColumns).map(+), window: window)
                for index in 0..<n where fixed[index] < scores[index] {
                    update[index] = true
                    probeRows[index] += deltaRows[index]
                    probeColumns[index] += deltaColumns[index]
                }
            }
            for index in 0..<n where !update[index] { step[index] = max(step[index] * stepFactor, minimumStep) }
            (kde, pixels) = try deposit()
            scores = try scoreMap(pixels, outRows: kde.rows, outColumns: kde.columns, probeRows, probeColumns, window: window)
            history.append(mean(scores))
            progress?(iteration + 1, options.iterations)
        }
        guard probeRows.allSatisfy(\.isFinite), probeColumns.allSatisfy(\.isFinite), history.allSatisfy(\.isFinite)
        else { throw ParallaxError.nonfiniteValues }
        return ParallaxPositionCorrectionResult(
            offsets: ParallaxProbeOffsets(rows: probeRows.map(Double.init), columns: probeColumns.map(Double.init)),
            scores: history.map(Double.init), image: kde)
    }

    private static func bilinear(_ image: [Float], rows: Int, columns: Int, row: Float, column: Float) -> Float {
        func wrap(_ value: Int, _ count: Int) -> Int { let r = value % count; return r >= 0 ? r : r + count }
        let rowFloor = Int(floor(row)), columnFloor = Int(floor(column))
        let rowFraction = row - Float(rowFloor), columnFraction = column - Float(columnFloor)
        let row0 = wrap(rowFloor, rows), row1 = wrap(rowFloor + 1, rows)
        let column0 = wrap(columnFloor, columns), column1 = wrap(columnFloor + 1, columns)
        return image[row0 * columns + column0] * (1 - rowFraction) * (1 - columnFraction)
            + image[row1 * columns + column0] * rowFraction * (1 - columnFraction)
            + image[row0 * columns + column1] * (1 - rowFraction) * columnFraction
            + image[row1 * columns + column1] * rowFraction * columnFraction
    }
}
