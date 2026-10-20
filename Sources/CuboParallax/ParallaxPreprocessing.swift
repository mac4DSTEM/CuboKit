import Foundation
import CuboCore

/// Numerical input errors in the parallax processing stages.
public enum ParallaxError: Error, Sendable, Equatable {
    /// Dimensions, lengths, orders or physical sampling are inconsistent.
    case invalidInput
    /// A selected image has no positive normalization weight, or geometry is singular.
    case degenerateData
    /// Inputs or arithmetic results contain NaN or infinity.
    case nonfiniteValues
}

/// Which normalized image is retained for subsequent subpixel reconstruction.
public enum ParallaxStackRetention: Sendable {
    /// py4DSTEM default: retain the edge-windowed image.
    case windowed
    /// Retain the normalized image before blending, as the source application's KDE path does.
    case unwindowed
}

/// Calibrated numerical products of constant-order parallax preprocessing.
public struct ParallaxPreprocessingResult: Sendable {
    /// Padded image dimensions.
    public let rows: Int
    /// Padded image columns.
    public let columns: Int
    /// Selected detector pixels as row/column pairs, in row-major order.
    public let detectorIndices: [[Int]]
    /// Centered reciprocal vectors in inverse angstroms.
    public let reciprocalVectors: [ParallaxShift]
    /// Probe angles in radians (reciprocal vector times wavelength).
    public let probeAngles: [ParallaxShift]
    /// Unpadded, separable sine-squared edge window.
    public let window: [Double]
    /// Padded, blended normalized images, one per detector pixel.
    public let stack: [[Double]]
    /// Retained images according to the requested retention convention.
    public let unshiftedStack: [[Double]]
    /// Initial incoherent bright-field image, padded and edge blended.
    public let incoherentImage: [Double]
    /// Normalized window-weighted absolute error.
    public let initialError: Double
    /// Scan rows before padding.
    public let scanRows: Int
    /// Scan columns before padding.
    public let scanColumns: Int
    /// The edge window placed in the padded frame (zero in the padding), `rows * columns` values.
    ///
    /// This is py4DSTEM's `_window_pad`, the initial real-space mask of every image, and the
    /// window that ``ParallaxSubpixel/correctPositions(images:rows:columns:shifts:window:scanSampling:factor:sigmaPixels:lanczosOrder:lowpass:options:)``
    /// expects.
    public let paddedWindow: [Double]
}

/// Constant-order virtual bright-field normalization from `Parallax.preprocess`.
///
/// Eager CPU input in cube storage order. This entry point ports the source application's
/// threshold, coordinate, padding and edge-blending numbers. Descan correction,
/// polynomial normalization and real-space masks are not provided by this entry point.
public enum ParallaxPreprocessing: Sendable {
    /// Relativistic wavelength in angstroms, using the pinned reference constants.
    public static func electronWavelength(energyEV: Double) throws -> Double {
        guard energyEV.isFinite, energyEV > 0 else { throw ParallaxError.invalidInput }
        let result = 6.62607e-34 / sqrt(2 * 9.109383e-31 * 1.602177e-19 * energyEV)
            / sqrt(1 + 1.602177e-19 * energyEV / (2 * 9.109383e-31 * 299792458 * 299792458)) * 1e10
        guard result.isFinite, result > 0 else { throw ParallaxError.nonfiniteValues }
        return result
    }

    /// Prepares a stack; padding is the total added extent on each axis (odd padding is allowed).
    ///
    /// Sampling is inverse angstroms per detector pixel. The optional detector mask replaces
    /// threshold selection. The default retention matches Python; `.unwindowed` retains
    /// the source application's KDE input. Outputs use Double; the oracle's Float32 storage
    /// introduces the documented `2e-6` regression tolerance.
    public static func prepare(_ values: [Double], shape: CubeShape,
        reciprocalSampling: ParallaxShift, energyEV: Double,
        threshold: Double = 0.8, edgeBlend: Double = 16,
        paddingRows: Int = 32, paddingColumns: Int = 32,
        detectorMask: [Bool]? = nil, retention: ParallaxStackRetention = .windowed
    ) throws -> ParallaxPreprocessingResult {
        guard values.count == shape.elementCount, shape.scanRows >= 2, shape.scanColumns >= 2,
              threshold.isFinite, threshold > 0, threshold <= 1, edgeBlend.isFinite, edgeBlend > 0,
              paddingRows >= 0, paddingColumns >= 0,
              reciprocalSampling.row.isFinite, reciprocalSampling.column.isFinite,
              reciprocalSampling.row > 0, reciprocalSampling.column > 0,
              detectorMask == nil || detectorMask!.count == shape.detectorCount else { throw ParallaxError.invalidInput }
        guard values.allSatisfy(\.isFinite) else { throw ParallaxError.nonfiniteValues }
        let wavelength = try electronWavelength(energyEV: energyEV)
        let (rows, ro) = shape.scanRows.addingReportingOverflow(paddingRows)
        let (columns, co) = shape.scanColumns.addingReportingOverflow(paddingColumns)
        let (planeCount, po) = rows.multipliedReportingOverflow(by: columns)
        guard !ro, !co, !po else { throw ParallaxError.invalidInput }
        var mean = [Double](repeating: 0, count: shape.detectorCount)
        for p in 0..<shape.scanCount { for q in mean.indices { mean[q] += values[p * shape.detectorCount + q] / Double(shape.scanCount) } }
        let cutoff = mean.max()! * threshold
        let mask = detectorMask ?? mean.map { $0 >= cutoff }
        let selected = mask.indices.filter { mask[$0] }
        guard !selected.isEmpty else { throw ParallaxError.degenerateData }
        let (_, so) = planeCount.multipliedReportingOverflow(by: selected.count)
        guard !so else { throw ParallaxError.invalidInput }
        let indices = selected.map { [$0 / shape.detectorColumns, $0 % shape.detectorColumns] }
        let centerR = indices.reduce(0.0) { $0 + Double($1[0]) } / Double(indices.count)
        let centerC = indices.reduce(0.0) { $0 + Double($1[1]) } / Double(indices.count)
        let vectors = indices.map { ParallaxShift(row: (Double($0[0]) - centerR) * reciprocalSampling.row,
            column: (Double($0[1]) - centerC) * reciprocalSampling.column) }
        guard vectors.allSatisfy({ $0.row.isFinite && $0.column.isFinite }) else { throw ParallaxError.nonfiniteValues }
        let angles = vectors.map { ParallaxShift(row: $0.row * wavelength, column: $0.column * wavelength) }
        guard angles.allSatisfy({ $0.row.isFinite && $0.column.isFinite }) else { throw ParallaxError.nonfiniteValues }
        func axis(_ n: Int) -> [Double] {
            (0..<n).map { i in
                let x = -1 + (Double(i) + 0.5) * 2 / Double(n)
                let s = sin(min(max((1 - abs(x)) * Double(n) / edgeBlend / 2, 0), 1) * .pi / 2)
                return s * s
            }
        }
        let wr = axis(shape.scanRows), wc = axis(shape.scanColumns)
        let window = (0..<shape.scanCount).map { wr[$0 / shape.scanColumns] * wc[$0 % shape.scanColumns] }
        let windowSum = window.reduce(0,+)
        guard windowSum > 0 else { throw ParallaxError.degenerateData }
        var stack = [[Double]](), unshifted = [[Double]]()
        for q in selected {
            var image = [Double](repeating: 1, count: planeCount), rawImage = image
            var weight = 0.0
            for p in 0..<shape.scanCount { weight += values[p * shape.detectorCount + q] * window[p] }
            weight /= windowSum
            guard weight.isFinite, weight > 0 else { throw ParallaxError.degenerateData }
            for p in 0..<shape.scanCount {
                let target = (p / shape.scanColumns + paddingRows / 2) * columns + p % shape.scanColumns + paddingColumns / 2
                rawImage[target] = values[p * shape.detectorCount + q] / weight
                image[target] = 1 - window[p] + window[p] * rawImage[target]
            }
            stack.append(image); unshifted.append(retention == .windowed ? image : rawImage)
        }
        let stackMean = stack.reduce(0.0) { $0 + $1.reduce(0,+) } / Double(planeCount * selected.count)
        var incoherent = [Double](repeating: stackMean, count: planeCount), error = 0.0
        for p in 0..<shape.scanCount {
            let target = (p / shape.scanColumns + paddingRows / 2) * columns + p % shape.scanColumns + paddingColumns / 2
            let average = stack.reduce(0.0) { $0 + $1[target] } / Double(selected.count)
            incoherent[target] = stackMean * (1 - window[p]) + average * window[p]
            for image in stack { error += abs(image[target] - incoherent[target]) * window[p] }
        }
        error /= windowSum * Double(selected.count) * stackMean
        guard stack.flatMap({$0}).allSatisfy(\.isFinite), incoherent.allSatisfy(\.isFinite), error.isFinite else { throw ParallaxError.nonfiniteValues }
        var paddedWindow = [Double](repeating: 0, count: planeCount)
        for p in 0..<shape.scanCount {
            paddedWindow[(p / shape.scanColumns + paddingRows / 2) * columns + p % shape.scanColumns + paddingColumns / 2] = window[p]
        }
        return ParallaxPreprocessingResult(rows: rows, columns: columns, detectorIndices: indices,
            reciprocalVectors: vectors, probeAngles: angles, window: window, stack: stack,
            unshiftedStack: unshifted, incoherentImage: incoherent, initialError: error,
            scanRows: shape.scanRows, scanColumns: shape.scanColumns, paddedWindow: paddedWindow)
    }
}
