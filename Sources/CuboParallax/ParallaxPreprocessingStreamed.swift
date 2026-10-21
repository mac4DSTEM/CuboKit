import Foundation
import CuboCore

/// Float32 products of streamed constant-order parallax preprocessing.
///
/// Image stacks are flat, `[bright-field pixel][paddedRow][paddedColumn]`, so image `b` occupies
/// `b * rows * columns ..< (b + 1) * rows * columns`. Reciprocal vectors and probe angles are the Float32 values the
/// source application computes, widened to Double.
public struct ParallaxStreamedPreprocessingResult: Sendable {
    /// Padded image rows.
    public let rows: Int
    /// Padded image columns.
    public let columns: Int
    /// Scan rows before padding.
    public let scanRows: Int
    /// Scan columns before padding.
    public let scanColumns: Int
    /// Detector mask, `detectorRows * detectorColumns` values in row-major order (the one given, or the threshold's).
    public let detectorMask: [Bool]
    /// Selected detector pixels as row/column pairs, in row-major order.
    public let detectorIndices: [[Int]]
    /// Centered reciprocal vectors in inverse angstroms.
    public let reciprocalVectors: [ParallaxShift]
    /// Probe angles in milliradians (reciprocal vector times wavelength times 1000).
    public let probeAnglesMrad: [ParallaxShift]
    /// Unpadded, separable sine-squared edge window, `scanRows * scanColumns` values.
    public let window: [Float]
    /// Padded, edge-blended normalized images, `[bf][rows][columns]`.
    public let stack: [Float]
    /// Padded normalized images before blending, `[bf][rows][columns]`; for `.windowed` retention this equals `stack`
    /// (the same storage, not a second copy).
    public let unshiftedStack: [Float]
    /// Initial incoherent bright-field image, padded and edge blended, `rows * columns` values.
    public let incoherentImage: [Float]
    /// Normalized window-weighted absolute error.
    public let initialError: Float

    /// Selected bright-field detector pixels.
    public var brightFieldCount: Int { detectorIndices.count }

    /// Creates a result from its parts.
    public init(rows: Int, columns: Int, scanRows: Int, scanColumns: Int, detectorMask: [Bool],
                detectorIndices: [[Int]], reciprocalVectors: [ParallaxShift], probeAnglesMrad: [ParallaxShift],
                window: [Float], stack: [Float], unshiftedStack: [Float], incoherentImage: [Float],
                initialError: Float) {
        self.rows = rows; self.columns = columns; self.scanRows = scanRows; self.scanColumns = scanColumns
        self.detectorMask = detectorMask; self.detectorIndices = detectorIndices
        self.reciprocalVectors = reciprocalVectors; self.probeAnglesMrad = probeAnglesMrad
        self.window = window; self.stack = stack; self.unshiftedStack = unshiftedStack
        self.incoherentImage = incoherentImage; self.initialError = initialError
    }
}

extension ParallaxPreprocessing {
    /// Preprocesses a cube streamed in scan-row tiles, never holding it: pass 1 accumulates the mean pattern (skipped
    /// when `detectorMask` is given) and selects the bright field (`mean >= max * threshold`), pass 2 gathers only
    /// the bright-field pixels into the padded stacks.
    ///
    /// Arithmetic follows the source application: Float32 pixel arithmetic in scan order, a Double accumulator for
    /// the stack mean and the error sum. Results do not depend on `tiling` (bit for bit). Against the resident
    /// ``prepare(_:shape:reciprocalSampling:energyEV:threshold:edgeBlend:paddingRows:paddingColumns:detectorMask:retention:)``
    /// (Double) the stacks agree to Float32 rounding, regression tolerance `2e-5` relative (abs. `1e-5`).
    ///
    /// Only non-finite pixels inside the bright-field selection are an error (`nonfiniteValues`); elsewhere they are
    /// ignored by the mean. `maximumStackBytes` refuses (`stackTooLarge`) when the retained stacks (two for
    /// `.unwindowed`, one for `.windowed`) would exceed it. `progress` receives a fraction in `0...1`; `isCancelled` is
    /// polled per scan row and throws `CancellationError`.
    public static func prepare(source: some CubeSource,
        reciprocalSampling: ParallaxShift, energyEV: Double,
        threshold: Double = 0.8, edgeBlend: Double = 16,
        paddingRows: Int = 32, paddingColumns: Int = 32,
        detectorMask: [Bool]? = nil, retention: ParallaxStackRetention = .windowed,
        tiling: ScanTiling = .automatic, maximumStackBytes: Int? = nil,
        progress: (@Sendable (Double) -> Void)? = nil,
        isCancelled: (@Sendable () -> Bool)? = nil
    ) async throws -> ParallaxStreamedPreprocessingResult {
        let shape = source.shape
        let blend = Float(edgeBlend), thresholdF = Float(threshold)
        guard threshold.isFinite, threshold > 0, threshold <= 1, edgeBlend.isFinite, blend.isFinite, blend > 0,
              paddingRows >= 0, paddingColumns >= 0,
              reciprocalSampling.row.isFinite, reciprocalSampling.column.isFinite,
              reciprocalSampling.row > 0, reciprocalSampling.column > 0,
              maximumStackBytes == nil || maximumStackBytes! > 0,
              detectorMask == nil || detectorMask!.count == shape.detectorCount else { throw ParallaxError.invalidInput }
        let wavelength = try electronWavelength(energyEV: energyEV)
        let ranges = try tiling.ranges(for: shape)
        func cancelled() throws { if isCancelled?() == true { throw CancellationError() } }
        try cancelled()
        let detectorCount = shape.detectorCount, scanColumns = shape.scanColumns

        let (rows, ro) = shape.scanRows.addingReportingOverflow(paddingRows)
        let (columns, co) = shape.scanColumns.addingReportingOverflow(paddingColumns)
        let (planeCount, po) = rows.multipliedReportingOverflow(by: columns)
        guard !ro, !co, !po else { throw ParallaxError.invalidInput }

        // Pass 1: mean pattern (Float, scan order), then the bright-field selection.
        let mask: [Bool]
        if let detectorMask {
            mask = detectorMask
        } else {
            var mean = [Float](repeating: 0, count: detectorCount)
            var done = 0
            for range in ranges {
                try cancelled()
                let tile = try await source.readScanTile(rows: range)
                guard tile.values.count == range.count * shape.elementsPerScanRow else { throw ParallaxError.invalidInput }
                tile.values.withUnsafeBufferPointer { values in
                    for position in 0..<tile.positionCount {
                        let base = position * detectorCount
                        for q in 0..<detectorCount { mean[q] += values[base + q] }
                    }
                }
                done += range.count
                progress?(0.5 * Double(done) / Double(shape.scanRows))
            }
            let inverseCount = 1 / Float(shape.scanCount)
            for q in 0..<detectorCount { mean[q] *= inverseCount }
            guard let maximum = mean.filter(\.isFinite).max(), maximum > 0 else { throw ParallaxError.degenerateData }
            let cutoff = maximum * thresholdF
            mask = mean.map { $0.isFinite && $0 >= cutoff }
        }
        let selected = mask.indices.filter { mask[$0] }
        guard !selected.isEmpty else { throw ParallaxError.degenerateData }
        let count = selected.count
        let indices = selected.map { [$0 / shape.detectorColumns, $0 % shape.detectorColumns] }

        let centerR = Double(indices.reduce(0) { $0 + $1[0] }) / Double(count)
        let centerC = Double(indices.reduce(0) { $0 + $1[1] }) / Double(count)
        let vectors = indices.map {
            ParallaxShift(row: Double(Float((Double($0[0]) - centerR) * reciprocalSampling.row)),
                          column: Double(Float((Double($0[1]) - centerC) * reciprocalSampling.column)))
        }
        let toMrad = Float(wavelength * 1_000)
        let angles = vectors.map {
            ParallaxShift(row: Double(Float($0.row) * toMrad), column: Double(Float($0.column) * toMrad))
        }
        guard vectors.allSatisfy({ $0.row.isFinite && $0.column.isFinite }),
              angles.allSatisfy({ $0.row.isFinite && $0.column.isFinite }) else { throw ParallaxError.nonfiniteValues }

        func axis(_ n: Int) -> [Float] {
            (0..<n).map { i in
                let c = -1 + (Float(i) + 0.5) * 2 / Float(n)
                let s = sin(min(max((1 - abs(c)) * Float(n) / blend / 2, 0), 1) * .pi / 2)
                return s * s
            }
        }
        let wr = axis(shape.scanRows), wc = axis(scanColumns)
        var window = [Float](repeating: 0, count: shape.scanCount)
        for y in 0..<shape.scanRows { for x in 0..<scanColumns { window[y * scanColumns + x] = wr[y] * wc[x] } }
        let windowSum = window.reduce(0, +)
        guard windowSum.isFinite, windowSum > 0 else { throw ParallaxError.degenerateData }

        let (stackCount, so) = planeCount.multipliedReportingOverflow(by: count)
        let (stackBytes, bo) = stackCount.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
        let copies = retention == .unwindowed ? 2 : 1
        let (residentBytes, ro2) = stackBytes.multipliedReportingOverflow(by: copies)
        guard !so, !bo, !ro2 else { throw ParallaxError.invalidInput }
        if let maximumStackBytes, residentBytes > maximumStackBytes {
            throw ParallaxError.stackTooLarge(bytes: residentBytes, limit: maximumStackBytes)
        }

        // Pass 2: gather the bright-field pixels, accumulate the weighted sums.
        var raw = [Float](repeating: 1, count: stackCount)
        var weightedSums = [Float](repeating: 0, count: count)
        let top = paddingRows / 2, left = paddingColumns / 2
        let pass2Base = detectorMask == nil ? 0.5 : 0.0, pass2Span = detectorMask == nil ? 0.5 : 1.0
        var done = 0
        for range in ranges {
            try cancelled()
            let tile = try await source.readScanTile(rows: range)
            guard tile.values.count == range.count * shape.elementsPerScanRow else { throw ParallaxError.invalidInput }
            var bad = false
            try tile.values.withUnsafeBufferPointer { values in
                for localRow in 0..<range.count {
                    try cancelled()
                    let globalRow = range.lowerBound + localRow
                    for x in 0..<scanColumns {
                        let sourceBase = (localRow * scanColumns + x) * detectorCount
                        let w = window[globalRow * scanColumns + x]
                        let pixel = (top + globalRow) * columns + left + x
                        for (bf, q) in selected.enumerated() {
                            let value = values[sourceBase + q]
                            if !value.isFinite { bad = true; return }
                            raw[bf * planeCount + pixel] = value
                            weightedSums[bf] += value * w
                        }
                    }
                }
            }
            if bad { throw ParallaxError.nonfiniteValues }
            done += range.count
            progress?(pass2Base + pass2Span * Double(done) / Double(shape.scanRows))
        }
        try cancelled()

        let keepRaw = retention == .unwindowed
        var stack = raw  // the blended stack; `raw` becomes the unshifted one when retained
        if !keepRaw { raw = [] }  // release the shared buffer so `stack` mutates in place
        for bf in 0..<count {
            let weight = weightedSums[bf] / windowSum
            guard weight.isFinite, weight > 0 else { throw ParallaxError.degenerateData }
            for y in 0..<shape.scanRows {
                for x in 0..<scanColumns {
                    let w = window[y * scanColumns + x]
                    let offset = bf * planeCount + (top + y) * columns + left + x
                    let normalized = (keepRaw ? raw[offset] : stack[offset]) / weight
                    if keepRaw { raw[offset] = normalized }
                    stack[offset] = 1 - w + w * normalized
                }
            }
        }
        try cancelled()

        var total = 0.0
        for value in stack { total += Double(value) }
        let stackMean = Float(total / Double(stack.count))
        guard stackMean.isFinite, stackMean > 0 else { throw ParallaxError.degenerateData }
        var incoherent = [Float](repeating: stackMean, count: planeCount)
        var errorSum = 0.0
        for y in 0..<shape.scanRows {
            for x in 0..<scanColumns {
                let w = window[y * scanColumns + x]
                let pixel = (top + y) * columns + left + x
                var average: Float = 0
                for bf in 0..<count { average += stack[bf * planeCount + pixel] }
                average /= Float(count)
                let reference = stackMean * (1 - w) + average * w
                incoherent[pixel] = reference
                for bf in 0..<count { errorSum += Double(abs(stack[bf * planeCount + pixel] - reference) * w) }
            }
        }
        let initialError = Float(errorSum / Double(windowSum * Float(count)) / Double(stackMean))
        guard initialError.isFinite, incoherent.allSatisfy(\.isFinite) else { throw ParallaxError.nonfiniteValues }
        try cancelled()
        progress?(1)
        return ParallaxStreamedPreprocessingResult(
            rows: rows, columns: columns, scanRows: shape.scanRows, scanColumns: scanColumns,
            detectorMask: mask, detectorIndices: indices, reciprocalVectors: vectors, probeAnglesMrad: angles,
            window: window, stack: stack, unshiftedStack: retention == .unwindowed ? raw : stack,
            incoherentImage: incoherent, initialError: initialError)
    }
}
