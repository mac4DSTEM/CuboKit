import Foundation
import CuboCore

/// What to do when the bin schedule is empty (the maximum bin is 1, so no binning is possible).
public enum ParallaxEmptyScheduleBehavior: Sendable {
    /// py4DSTEM: with `numIterAtMinBin > 1` its `bin_vals[-1]` raises `IndexError`, which becomes
    /// `ParallaxError.invalidInput`; with `numIterAtMinBin <= 1` the schedule is empty and nothing is aligned.
    case py4DSTEM
    /// The source application: one pass at the minimum bin.
    case singlePassAtMinimumBin
}

/// Options of ``ParallaxReconstruction/reconstruct(_:options:)``, named and defaulted as in py4DSTEM's
/// `Parallax.reconstruct`.
public struct ParallaxReconstructionOptions: Sendable {
    /// Largest alignment bin; `nil` uses the bright-field diameter in detector pixels. Larger values are clipped to it.
    public var maxAlignmentBin: Int? = nil
    /// Smallest alignment bin.
    public var minAlignmentBin = 1
    /// Passes at the smallest bin (the last bin is repeated `numIterAtMinBin - 1` times).
    public var numIterAtMinBin = 2
    /// Explicit schedule, clipped to `1...maxAlignmentBin`; `nil` builds `2^k` from the coarsest down.
    public var alignmentBinValues: [Double]? = nil
    /// `true` rounds `offset / bin` instead of `offset / bin + 0.5`.
    public var centeredAlignmentBins = false
    /// Matrix-DFT upsampling factor of each correlation, 3...64.
    public var crossCorrelationUpsampleFactor = 8
    /// Bernstein degrees (rows, columns), each 0...8, of the shift basis used when `regularizeShifts` is true.
    public var regularizerMatrixSize: (rows: Int, columns: Int) = (1, 1)
    /// Constrain the per-image shifts to the Bernstein basis instead of the linear reciprocal-vector basis.
    public var regularizeShifts = false
    /// Update the reference in a spiral from the centre as each subset is aligned.
    public var runningAverage = true
    /// Empty-schedule behaviour; py4DSTEM's by default.
    public var emptySchedule = ParallaxEmptyScheduleBehavior.py4DSTEM
    /// Creates py4DSTEM's defaults.
    public init() {}
}

/// Products of ``ParallaxReconstruction/reconstruct(_:options:)``.
public struct ParallaxReconstructionResult: Sendable {
    /// Padded image rows.
    public let rows: Int
    /// Padded image columns.
    public let columns: Int
    /// Aligned, windowed bright-field images (py4DSTEM `_stack_BF_shifted`), one per detector pixel.
    public let alignedStack: [[Double]]
    /// Shifted real-space masks (py4DSTEM `_stack_mask`), one per detector pixel.
    public let stackMask: [[Double]]
    /// Accumulated per-image shifts after median centering (py4DSTEM `_xy_shifts`), in padded-image pixels.
    /// Apply them to the preprocessing result's `unshiftedStack` for subpixel upsampling.
    public let shifts: [ParallaxShift]
    /// Reconstructed aligned bright-field image (py4DSTEM `recon_BF`).
    public let image: [Double]
    /// Mean of the shifted masks (py4DSTEM `_recon_mask`).
    public let mask: [Double]
    /// Normalized error after each alignment bin (py4DSTEM `error_iterations`).
    public let errors: [Double]
    /// The bin schedule that was run.
    public let binValues: [Double]
}

/// Port of py4DSTEM's `Parallax.reconstruct`: coarse-to-fine alignment of the bright-field stack.
///
/// For each bin, detector pixels are grouped by `round(offset / bin (+ 0.5))`, groups are visited from the
/// centre outward, each group's mean image is correlated against the reference with the matrix-DFT upsampled
/// peak, the per-image shifts are projected on the shift basis (least squares), the stack and masks are
/// Fourier-shifted, the shifts are median-centred by an integer roll, and the reconstruction and its error are
/// updated. The correlation, the matrix-DFT refinement and the Fourier shifts are those of ``ParallaxAlignment``.
public enum ParallaxReconstruction: Sendable {
    /// The bin schedule `reconstruct` would run for these detector pixels.
    public static func binSchedule(detectorIndices: [[Int]],
        options: ParallaxReconstructionOptions = .init()) throws -> [Double] {
        guard !detectorIndices.isEmpty, detectorIndices.allSatisfy({ $0.count == 2 }),
              options.minAlignmentBin >= 1, (options.maxAlignmentBin ?? 1) >= 1 else { throw ParallaxError.invalidInput }
        let rowValues = detectorIndices.map { $0[0] }, columnValues = detectorIndices.map { $0[1] }
        let diameter = max(rowValues.max()! - rowValues.min()!, columnValues.max()! - columnValues.min()!) + 1
        let maxBin = options.maxAlignmentBin.map { min(diameter, $0) } ?? diameter
        if let values = options.alignmentBinValues {
            guard !values.isEmpty, values.allSatisfy(\.isFinite) else { throw ParallaxError.invalidInput }
            return values.map { min(max($0, 1), Double(maxBin)) }
        }
        // The same log ratio as NumPy; exact powers of two are the only place the ceiling is sensitive.
        let lowest = Int((log(Double(options.minAlignmentBin)) / log(2.0)).rounded(.up))
        let highest = Int((log(Double(maxBin)) / log(2.0)).rounded(.up))
        var bins = lowest < highest ? (lowest..<highest).reversed().map { pow(2.0, Double($0)) } : []
        if options.numIterAtMinBin > 1 {
            if let last = bins.last {
                bins += [Double](repeating: last, count: options.numIterAtMinBin - 1)
            } else if options.emptySchedule == .py4DSTEM {
                throw ParallaxError.invalidInput
            }
        }
        if bins.isEmpty, options.emptySchedule == .singlePassAtMinimumBin { bins = [Double(options.minAlignmentBin)] }
        return bins
    }

    /// Aligns the preprocessed stack, starting from `Parallax.preprocess` with no initial aberrations.
    ///
    /// Input and output are Double; py4DSTEM runs float32. Subset order within equal radius is lexicographic
    /// (a stable sort). Continuing a previous reconstruction is not provided: call this once on the preprocessing result.
    public static func reconstruct(_ input: ParallaxPreprocessingResult,
        options: ParallaxReconstructionOptions = .init()) throws -> ParallaxReconstructionResult {
        let rows = input.rows, columns = input.columns, count = rows * columns
        let n = input.stack.count
        guard rows >= 3, columns >= 3, n >= 1, input.detectorIndices.count == n,
              input.reciprocalVectors.count == n, input.paddedWindow.count == count,
              input.stack.allSatisfy({ $0.count == count }),
              (3...64).contains(options.crossCorrelationUpsampleFactor),
              options.regularizerMatrixSize.rows >= 0, options.regularizerMatrixSize.columns >= 0,
              options.regularizerMatrixSize.rows <= 8, options.regularizerMatrixSize.columns <= 8
        else { throw ParallaxError.invalidInput }
        guard input.stack.allSatisfy({ $0.allSatisfy(\.isFinite) }), input.paddedWindow.allSatisfy(\.isFinite)
        else { throw ParallaxError.nonfiniteValues }
        let bins = try binSchedule(detectorIndices: input.detectorIndices, options: options)

        let stackMean = input.stack.reduce(0.0) { $0 + $1.reduce(0, +) } / Double(n * count)
        let maskSum = input.window.reduce(0, +) * Double(n)
        guard stackMean.isFinite, stackMean != 0, maskSum.isFinite, maskSum > 0 else { throw ParallaxError.degenerateData }

        let basis = try shiftBasis(input, options: options)
        let fft = FFT2D(rows: rows, columns: columns)
        var stack = input.stack
        var masks = [[Double]](repeating: input.paddedWindow, count: n)
        var shifts = [ParallaxShift](repeating: ParallaxShift(row: 0, column: 0), count: n)
        var (image, reconMask) = estimate(stack: stack, masks: masks, stackMean: stackMean)
        var errors: [Double] = []

        let indices = input.detectorIndices
        let medianRow = median(indices.map { Double($0[0]) }), medianColumn = median(indices.map { Double($0[1]) })
        let binShift = options.centeredAlignmentBins ? 0.0 : 0.5

        for bin in bins {
            var reference = try ParallaxAlignment.spectrum(image, fft: fft)
            var members: [GroupKey: [Int]] = [:]
            for (i, index) in indices.enumerated() {
                let key = GroupKey(row: Int(((Double(index[0]) - medianRow) / bin + binShift).rounded(.toNearestOrEven)),
                                   column: Int(((Double(index[1]) - medianColumn) / bin + binShift).rounded(.toNearestOrEven)))
                members[key, default: []].append(i)
            }
            let ordered = members.keys.sorted {
                let a = $0.row * $0.row + $0.column * $0.column, b = $1.row * $1.row + $1.column * $1.column
                if a != b { return a < b }
                return ($0.row, $0.column) < ($1.row, $1.column)
            }
            var update = [ParallaxShift](repeating: ParallaxShift(row: 0, column: 0), count: n)
            for (step, key) in ordered.enumerated() {
                let subset = members[key]!
                var mean = [Double](repeating: 0, count: count)
                for i in subset { for p in 0..<count { mean[p] += stack[i][p] } }
                for p in 0..<count { mean[p] /= Double(subset.count) }
                var moving = try ParallaxAlignment.spectrum(mean, fft: fft)
                let shift = try ParallaxAlignment.estimate(reference: reference, moving: moving, fft: fft,
                    upsampleFactor: options.crossCorrelationUpsampleFactor)
                for i in subset { update[i] = shift }
                if options.runningAverage {
                    ParallaxAlignment.apply(shift, to: &moving, rows: rows, columns: columns)
                    let weight = Double(step + 1), previous = Double(step)
                    for p in 0..<count {
                        reference.real[p] = reference.real[p] * previous / weight + moving.real[p] / weight
                        reference.imaginary[p] = reference.imaginary[p] * previous / weight + moving.imaginary[p] / weight
                    }
                }
            }

            // Constrain the update to the shift basis.
            let target = [shifts.indices.map { shifts[$0].row + update[$0].row }, shifts.indices.map { shifts[$0].column + update[$0].column }]
            let fitted = target.map { project($0, onto: basis) }
            for i in 0..<n {
                update[i] = ParallaxShift(row: fitted[0][i] - shifts[i].row, column: fitted[1][i] - shifts[i].column)
            }
            guard update.allSatisfy({ $0.row.isFinite && $0.column.isFinite }) else { throw ParallaxError.nonfiniteValues }

            for i in 0..<n {
                var spectrum = try ParallaxAlignment.spectrum(stack[i], fft: fft)
                ParallaxAlignment.apply(update[i], to: &spectrum, rows: rows, columns: columns)
                stack[i] = try ParallaxAlignment.inverse(spectrum, fft: fft)
                var maskSpectrum = try ParallaxAlignment.spectrum(masks[i], fft: fft)
                ParallaxAlignment.apply(update[i], to: &maskSpectrum, rows: rows, columns: columns)
                masks[i] = try ParallaxAlignment.inverse(maskSpectrum, fft: fft)
                shifts[i] = ParallaxShift(row: shifts[i].row + update[i].row, column: shifts[i].column + update[i].column)
            }

            // Centre the shifts by an integer roll.
            let rowRoll = Int(median(shifts.map(\.row)).rounded(.toNearestOrEven))
            let columnRoll = Int(median(shifts.map(\.column)).rounded(.toNearestOrEven))
            shifts = shifts.map { ParallaxShift(row: $0.row - Double(rowRoll), column: $0.column - Double(columnRoll)) }
            if rowRoll != 0 || columnRoll != 0 {
                for i in 0..<n {
                    stack[i] = roll(stack[i], rows: rows, columns: columns, rowOffset: rowRoll, columnOffset: columnRoll)
                    masks[i] = roll(masks[i], rows: rows, columns: columns, rowOffset: rowRoll, columnOffset: columnRoll)
                }
            }

            (image, reconMask) = estimate(stack: stack, masks: masks, stackMean: stackMean)
            var error = 0.0
            for i in 0..<n { for p in 0..<count { error += abs(stack[i][p] - image[p]) * masks[i][p] } }
            error = error / maskSum / stackMean
            guard error.isFinite, image.allSatisfy(\.isFinite) else { throw ParallaxError.nonfiniteValues }
            errors.append(error)
        }
        return ParallaxReconstructionResult(rows: rows, columns: columns, alignedStack: stack, stackMask: masks,
            shifts: shifts, image: image, mask: reconMask, errors: errors, binValues: bins)
    }

    // MARK: - private

    private struct GroupKey: Hashable { let row: Int; let column: Int }

    /// `(stackMean * (1 - clip(mask)) + mean(stack * masks)) / (mask + 1 - clip(mask))`, with `mask = mean(masks)`.
    private static func estimate(stack: [[Double]], masks: [[Double]], stackMean: Double) -> ([Double], [Double]) {
        let n = Double(stack.count), count = stack[0].count
        var mask = [Double](repeating: 0, count: count), weighted = mask
        for i in stack.indices { for p in 0..<count { mask[p] += masks[i][p]; weighted[p] += stack[i][p] * masks[i][p] } }
        var image = mask
        for p in 0..<count {
            mask[p] /= n
            let inverse = 1 - min(max(mask[p], 0), 1)
            image[p] = (stackMean * inverse + weighted[p] / n) / (mask[p] + inverse)
        }
        return (image, mask)
    }

    /// Columns of the shift basis: the reciprocal vectors, or the Bernstein tensor basis on `[0, 1]^2`.
    private static func shiftBasis(_ input: ParallaxPreprocessingResult,
        options: ParallaxReconstructionOptions) throws -> [[Double]] {
        let vectors = input.reciprocalVectors
        guard vectors.allSatisfy({ $0.row.isFinite && $0.column.isFinite }) else { throw ParallaxError.nonfiniteValues }
        guard options.regularizeShifts else { return [vectors.map(\.row), vectors.map(\.column)] }
        let maximum = vectors.map { hypot($0.row, $0.column) }.max() ?? 0
        guard maximum > 0 else { throw ParallaxError.degenerateData }
        let u = vectors.map { $0.row * 0.5 / maximum + 0.5 }, v = vectors.map { $0.column * 0.5 / maximum + 0.5 }
        let (degreeRow, degreeColumn) = options.regularizerMatrixSize
        func choose(_ n: Int, _ k: Int) -> Double { (0..<k).reduce(1.0) { $0 * Double(n - $1) / Double($1 + 1) } }
        var columns: [[Double]] = []
        for i in 0...degreeRow {
            for j in 0...degreeColumn {
                columns.append(vectors.indices.map { p in
                    choose(degreeRow, i) * pow(u[p], Double(i)) * pow(1 - u[p], Double(degreeRow - i))
                        * choose(degreeColumn, j) * pow(v[p], Double(j)) * pow(1 - v[p], Double(degreeColumn - j))
                })
            }
        }
        return columns
    }

    /// Least-squares fit `basis * lstsq(basis, y)`: the orthogonal projection of `y` onto the column space.
    /// Columns whose residual after Gram-Schmidt is below `rows * eps(Float)` of the largest column norm are dropped,
    /// like NumPy's `rcond=None` cut for rank-deficient bases (the projection does not depend on the null space).
    private static func project(_ y: [Double], onto basis: [[Double]]) -> [Double] {
        let scale = basis.map { sqrt($0.reduce(0) { $0 + $1 * $1 }) }.max() ?? 0
        let cut = Double(y.count) * Double(Float.ulpOfOne) * scale
        var orthonormal: [[Double]] = []
        for column in basis {
            var w = column
            for _ in 0..<2 {
                for q in orthonormal {
                    let dot = zip(w, q).reduce(0) { $0 + $1.0 * $1.1 }
                    for p in w.indices { w[p] -= dot * q[p] }
                }
            }
            let norm = sqrt(w.reduce(0) { $0 + $1 * $1 })
            if norm > cut { orthonormal.append(w.map { $0 / norm }) }
        }
        var fit = [Double](repeating: 0, count: y.count)
        for q in orthonormal {
            let dot = zip(y, q).reduce(0) { $0 + $1.0 * $1.1 }
            for p in fit.indices { fit[p] += dot * q[p] }
        }
        return fit
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted(), middle = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
    }

    /// `np.roll(image, (-rowOffset, -columnOffset), axis=(0, 1))`.
    private static func roll(_ image: [Double], rows: Int, columns: Int, rowOffset: Int, columnOffset: Int) -> [Double] {
        var out = image
        for r in 0..<rows {
            let sourceRow = ((r + rowOffset) % rows + rows) % rows
            for c in 0..<columns {
                out[r * columns + c] = image[sourceRow * columns + ((c + columnOffset) % columns + columns) % columns]
            }
        }
        return out
    }
}

// MARK: - End to end

/// Kernel-density upsampling and position correction after the alignment.
public struct ParallaxSubpixelOptions: Sendable {
    /// Upsampling factor; `nil` uses the reference's heuristic from the bright-field and dark-field limits.
    public var factor: Double? = nil
    /// Gaussian kernel bandwidth in non-upsampled pixels.
    public var sigmaPixels = 0.125
    /// Lanczos deposition order; `nil` deposits bilinearly.
    public var lanczosOrder: Int? = nil
    /// Divide the upsampled spectrum by the pixel sinc.
    public var lowpass = false
    /// Iterative probe-position correction after the first estimate; `nil` skips it.
    public var positionCorrection: ParallaxPositionCorrectionOptions? = nil
    /// Creates py4DSTEM's defaults.
    public init() {}
}

/// Options of ``ParallaxReconstruction/process(_:shape:reciprocalSampling:scanSampling:energyEV:options:)``.
public struct ParallaxProcessingOptions: Sendable {
    /// Bright-field threshold relative to the mean pattern's maximum.
    public var threshold = 0.8
    /// Edge-blend width in scan pixels.
    public var edgeBlend = 16.0
    /// Total padding added to the scan rows.
    public var paddingRows = 32
    /// Total padding added to the scan columns.
    public var paddingColumns = 32
    /// Detector mask replacing threshold selection.
    public var detectorMask: [Bool]? = nil
    /// Which stack is kept for subpixel upsampling.
    public var retention = ParallaxStackRetention.windowed
    /// Alignment options.
    public var reconstruction = ParallaxReconstructionOptions()
    /// Subpixel upsampling; `nil` stops after the alignment.
    public var subpixel: ParallaxSubpixelOptions? = nil
    /// Creates py4DSTEM's defaults (no subpixel stage).
    public init() {}
}

/// Every product of the end-to-end call.
public struct ParallaxProcessingResult: Sendable {
    /// Calibrated, normalized stack.
    public let preprocessing: ParallaxPreprocessingResult
    /// Aligned stack, shifts and reconstruction.
    public let reconstruction: ParallaxReconstructionResult
    /// Wavelength in angstroms used for the probe angles.
    public let wavelength: Double
    /// Bright-field and dark-field limits and the factor that was used; `nil` without a subpixel stage.
    public let limits: ParallaxUpsampleLimits?
    /// The upsampled image (with the corrected positions when requested); `nil` without a subpixel stage.
    public let subpixel: ParallaxSubpixelResult?
    /// Position-correction offsets and scores; `nil` unless requested.
    public let positionCorrection: ParallaxPositionCorrectionResult?
}

extension ParallaxReconstruction {
    /// Runs ``ParallaxPreprocessing/prepare(_:shape:reciprocalSampling:energyEV:threshold:edgeBlend:paddingRows:paddingColumns:detectorMask:retention:)``,
    /// ``reconstruct(_:options:)`` and, if requested, the kernel-density upsampling with its position correction,
    /// chaining the individual public steps unchanged. `reciprocalSampling` is inverse angstroms per detector pixel and
    /// `scanSampling` angstroms per scan pixel; the upsampling limits use the row components as py4DSTEM does.
    /// Aberration fitting and correction and depth sectioning stay separate calls on the returned result.
    public static func process(_ values: [Double], shape: CubeShape, reciprocalSampling: ParallaxShift,
        scanSampling: ParallaxShift, energyEV: Double,
        options: ParallaxProcessingOptions = .init()) throws -> ParallaxProcessingResult {
        let preprocessing = try ParallaxPreprocessing.prepare(values, shape: shape, reciprocalSampling: reciprocalSampling,
            energyEV: energyEV, threshold: options.threshold, edgeBlend: options.edgeBlend,
            paddingRows: options.paddingRows, paddingColumns: options.paddingColumns,
            detectorMask: options.detectorMask, retention: options.retention)
        let reconstruction = try reconstruct(preprocessing, options: options.reconstruction)
        let wavelength = try ParallaxPreprocessing.electronWavelength(energyEV: energyEV)
        guard let subpixel = options.subpixel else {
            return ParallaxProcessingResult(preprocessing: preprocessing, reconstruction: reconstruction,
                wavelength: wavelength, limits: nil, subpixel: nil, positionCorrection: nil)
        }
        guard scanSampling.row.isFinite, scanSampling.column.isFinite, scanSampling.row > 0, scanSampling.column > 0
        else { throw ParallaxError.invalidInput }
        let maximumVector = preprocessing.reciprocalVectors.map { hypot($0.row, $0.column) }.max() ?? 0
        let limits = try ParallaxSubpixel.upsampleLimits(scanSampling: scanSampling.row, maxReciprocalVector: maximumVector,
            reciprocalSampling: reciprocalSampling.row, detectorRows: shape.detectorRows, requested: subpixel.factor)
        let image: ParallaxSubpixelResult
        var correction: ParallaxPositionCorrectionResult?
        if let positions = subpixel.positionCorrection {
            let result = try ParallaxSubpixel.correctPositions(images: preprocessing.unshiftedStack,
                rows: preprocessing.rows, columns: preprocessing.columns, shifts: reconstruction.shifts,
                window: preprocessing.paddedWindow, scanSampling: scanSampling, factor: limits.factor,
                sigmaPixels: subpixel.sigmaPixels, lanczosOrder: subpixel.lanczosOrder, lowpass: subpixel.lowpass,
                options: positions)
            correction = result
            image = result.image
        } else {
            image = try ParallaxSubpixel.kernelDensity(images: preprocessing.unshiftedStack,
                rows: preprocessing.rows, columns: preprocessing.columns, shifts: reconstruction.shifts,
                factor: limits.factor, sigmaPixels: subpixel.sigmaPixels, lanczosOrder: subpixel.lanczosOrder,
                lowpass: subpixel.lowpass)
        }
        return ParallaxProcessingResult(preprocessing: preprocessing, reconstruction: reconstruction,
            wavelength: wavelength, limits: limits, subpixel: image, positionCorrection: correction)
    }
}
