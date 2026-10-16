import Foundation
import CuboCore

/// Result of a kernel-density upsampling of an aligned bright-field stack.
public struct ParallaxSubpixelResult: Sendable {
    /// Upsampled image, row-major, `rows * columns` values.
    public let image: [Double]
    /// Rows of the upsampled image.
    public let rows: Int
    /// Columns of the upsampled image.
    public let columns: Int
    /// The upsampling factor used.
    public let factor: Double
}

/// Upsampling factor limits and the default choice of py4DSTEM `subpixel_alignment`.
public struct ParallaxUpsampleLimits: Sendable, Equatable {
    /// Scan sampling over the bright-field Nyquist sampling.
    public let brightField: Double
    /// Clamped to at least one, as in the reference.
    public let darkField: Double
    /// The requested factor, or the reference heuristic's choice.
    public let factor: Double
}

/// Per-scan-position probe corrections in input scan pixels, row-major `rows * columns`.
public struct ParallaxProbeOffsets: Sendable, Equatable {
    public let rows: [Double]
    public let columns: [Double]
    public init(rows: [Double], columns: [Double]) { self.rows = rows; self.columns = columns }
}

/// Options for iterative probe-position correction (py4DSTEM `position_correction_*`).
public struct ParallaxPositionCorrectionOptions: Sendable {
    public var iterations: Int
    public var initialStep = 1.0
    public var minimumStep = 0.1
    public var stepFactor = 0.75
    public var checkerboard = false
    /// Gaussian regularisation sigmas in angstroms for the row and column offsets.
    public var gaussianSigma: ParallaxShift? = nil
    /// Butterworth cutoffs in 1/angstrom for the row and column offsets.
    public var lowpass: ParallaxShift? = nil
    public var highpass: ParallaxShift? = nil
    public var butterworthOrder = (2, 2)
    public init(iterations: Int) { self.iterations = iterations }
}

/// Probe offsets and the mean-score history (`iterations + 1` entries).
public struct ParallaxPositionCorrectionResult: Sendable {
    public let offsets: ParallaxProbeOffsets
    public let scores: [Double]
    /// Final upsampled image computed with the corrected positions.
    public let image: ParallaxSubpixelResult
}

/// Kernel-density subpixel upsampling (py4DSTEM `subpixel_alignment`, bilinear or Lanczos deposition).
public enum ParallaxSubpixel: Sendable {
    /// The reference's factor heuristic. `maxReciprocalVector` is the largest bright-field reciprocal
    /// vector length (1/angstrom), `reciprocalSampling` the detector sampling (1/angstrom per pixel).
    public static func upsampleLimits(scanSampling: Double, maxReciprocalVector: Double,
        reciprocalSampling: Double, detectorRows: Int, requested: Double? = nil) throws -> ParallaxUpsampleLimits {
        guard scanSampling.isFinite, scanSampling > 0, maxReciprocalVector.isFinite, maxReciprocalVector > 0,
            reciprocalSampling.isFinite, reciprocalSampling > 0, detectorRows > 0 else { throw ParallaxError.invalidInput }
        let bf = scanSampling / (1 / maxReciprocalVector / 2)
        let df = max(scanSampling / (1 / (reciprocalSampling * Double(detectorRows))), 1)
        let factor = requested ?? (bf * 1.5 > df ? df : bf * 1.5 > 1 ? bf * 1.5 : max(df * 2 / 3, 1))
        guard factor.isFinite, factor >= 1 else { throw ParallaxError.invalidInput }
        return .init(brightField: bf, darkField: df, factor: factor)
    }

    /// Deposits every image pixel at `(index + shift) * factor` (bilinear, or Lanczos of the given order with the
    /// reference's own second-axis window `sinc((i - dy)/order)`), wraps periodically, blurs with a Gaussian of
    /// `sigmaPixels * factor` (reflect boundaries, 4-sigma truncation) and divides by the blurred density
    /// (1 where the density is at most 1e-3). Optionally divides the spectrum by the pixel-sinc.
    public static func kernelDensity(images: [[Double]], rows: Int, columns: Int, shifts: [ParallaxShift],
        factor: Double, sigmaPixels: Double = 0.125, lanczosOrder: Int? = nil, lowpass: Bool = false,
        probeOffsets: ParallaxProbeOffsets? = nil) throws -> ParallaxSubpixelResult {
        guard !images.isEmpty, images.count == shifts.count, factor.isFinite, factor >= 1, sigmaPixels.isFinite,
            sigmaPixels >= 0, probeOffsets == nil || (probeOffsets!.rows.count == rows * columns && probeOffsets!.columns.count == rows * columns), lanczosOrder == nil || lanczosOrder! >= 1,
            shifts.allSatisfy({ $0.row.isFinite && $0.column.isFinite }) else { throw ParallaxError.invalidInput }
        for image in images { try validateParallaxImage(image, rows: rows, columns: columns) }
        let outRows = Int((Double(rows) * factor).rounded(.toNearestOrEven))
        let outColumns = Int((Double(columns) * factor).rounded(.toNearestOrEven))
        let (total, overflow) = outRows.multipliedReportingOverflow(by: outColumns)
        guard outRows > 0, outColumns > 0, !overflow else { throw ParallaxError.invalidInput }
        var count = [Double](repeating: 0, count: total), output = count
        func wrap(_ v: Int, _ n: Int) -> Int { let r = v % n; return r >= 0 ? r : r + n }
        func sinc(_ x: Double) -> Double { x == 0 ? 1 : sin(.pi * x) / (.pi * x) }
        for (b, image) in images.enumerated() {
            for r in 0..<rows { for c in 0..<columns {
                let o = r * columns + c
                let xr = (Double(r) + (probeOffsets?.rows[o] ?? 0) + shifts[b].row) * factor
                let xc = (Double(c) + (probeOffsets?.columns[o] ?? 0) + shifts[b].column) * factor
                let fr = Int(floor(xr)), fc = Int(floor(xc)), dr = xr - Double(fr), dc = xc - Double(fc)
                let v = image[r * columns + c]
                func add(_ i: Int, _ j: Int, _ w: Double) {
                    let t = wrap(fr + i, outRows) * outColumns + wrap(fc + j, outColumns)
                    count[t] += w; output[t] += w * v
                }
                if let order = lanczosOrder {
                    for i in (1 - order)...order { for j in (1 - order)...order {
                        add(i, j, sinc(Double(i) - dr) * sinc((Double(i) - dr) / Double(order))
                            * (sinc(Double(j) - dc) * sinc((Double(i) - dc) / Double(order))))
                    } }
                } else {
                    add(0, 0, (1 - dr) * (1 - dc)); add(1, 0, dr * (1 - dc)); add(0, 1, (1 - dr) * dc); add(1, 1, dr * dc)
                }
            } }
        }
        let sigma = sigmaPixels * factor
        count = gaussian(count, outRows, outColumns, sigma); output = gaussian(output, outRows, outColumns, sigma)
        for i in 0..<total { output[i] = count[i] > 1e-3 ? output[i] / count[i] : 1 }
        if lowpass {
            let fft = FFT2D(rows: outRows, columns: outColumns)
            var im = [Double](repeating: 0, count: total)
            fft.transform(&output, &im, inverse: false)
            for r in 0..<outRows { for c in 0..<outColumns {
                let d = sinc(parallaxFrequency(r, outRows)) * sinc(parallaxFrequency(c, outColumns)), i = r * outColumns + c
                output[i] /= d; im[i] /= d
            } }
            fft.transform(&output, &im, inverse: true)
        }
        guard output.allSatisfy(\.isFinite) else { throw ParallaxError.nonfiniteValues }
        return .init(image: output, rows: outRows, columns: outColumns, factor: factor)
    }

    /// `scipy.ndimage.gaussian_filter` (mode reflect, truncate 4) on a row-major image.
    static func gaussian(_ source: [Double], _ rows: Int, _ columns: Int, _ sigma: Double) -> [Double] {
        guard sigma > 0 else { return source }
        let radius = Int(4 * sigma + 0.5)
        guard radius > 0 else { return source }
        var taps = (-radius...radius).map { exp(-Double($0 * $0) / (2 * sigma * sigma)) }
        let sum = taps.reduce(0, +); taps = taps.map { $0 / sum }
        func reflect(_ v: Int, _ n: Int) -> Int {
            guard n > 1 else { return 0 }
            var i = v
            while i < 0 || i >= n { i = i < 0 ? -i - 1 : 2 * n - i - 1 }
            return i
        }
        var tmp = source, out = source
        for r in 0..<rows { for c in 0..<columns {
            var s = 0.0
            for o in -radius...radius { s += source[r * columns + reflect(c + o, columns)] * taps[o + radius] }
            tmp[r * columns + c] = s
        } }
        for r in 0..<rows { for c in 0..<columns {
            var s = 0.0
            for o in -radius...radius { s += tmp[reflect(r + o, rows) * columns + c] * taps[o + radius] }
            out[r * columns + c] = s
        } }
        return out
    }

    /// Mean absolute difference between the bilinearly sampled image and the stack, times `window`.
    static func scores(_ image: ParallaxSubpixelResult, images: [[Double]], rows: Int, columns: Int,
        shifts: [ParallaxShift], dr: [Double], dc: [Double], window: [Double]?) -> [Double] {
        func wrap(_ v: Int, _ n: Int) -> Int { let r = v % n; return r >= 0 ? r : r + n }
        var out = [Double](repeating: 0, count: rows * columns)
        for r in 0..<rows { for c in 0..<columns {
            let o = r * columns + c
            var sum = 0.0
            for (b, im) in images.enumerated() {
                let xr = (Double(r) + dr[o] + shifts[b].row) * image.factor
                let xc = (Double(c) + dc[o] + shifts[b].column) * image.factor
                let fr = Int(floor(xr)), fc = Int(floor(xc)), a = xr - Double(fr), q = xc - Double(fc)
                let r0 = wrap(fr, image.rows), r1 = wrap(fr + 1, image.rows)
                let c0 = wrap(fc, image.columns), c1 = wrap(fc + 1, image.columns)
                let v = image.image[r0 * image.columns + c0] * (1 - a) * (1 - q) + image.image[r1 * image.columns + c0] * a * (1 - q)
                    + image.image[r0 * image.columns + c1] * (1 - a) * q + image.image[r1 * image.columns + c1] * a * q
                sum += abs(v - im[o])
            }
            out[o] = sum / Double(images.count) * (window?[o] ?? 1)
        } }
        return out
    }

    /// Iterative probe-position correction, following py4DSTEM `subpixel_alignment` with
    /// `position_correction_num_iter`. Trial scores always use bilinear sampling; deposition uses
    /// `lanczosOrder` if given. `window` is the padded edge window (`ParallaxPreprocessingResult.window`).
    public static func correctPositions(images: [[Double]], rows: Int, columns: Int, shifts: [ParallaxShift],
        window: [Double], scanSampling: ParallaxShift, factor: Double, sigmaPixels: Double = 0.125,
        lanczosOrder: Int? = nil, lowpass: Bool = false, options: ParallaxPositionCorrectionOptions) throws -> ParallaxPositionCorrectionResult {
        let n = rows * columns
        guard window.count == n, options.iterations >= 0, options.initialStep > 0, options.minimumStep > 0,
            options.stepFactor > 0, options.stepFactor <= 1, scanSampling.row > 0, scanSampling.column > 0,
            options.butterworthOrder.0 >= 1, options.butterworthOrder.1 >= 1 else { throw ParallaxError.invalidInput }
        var dr = [Double](repeating: 0, count: n), dc = dr
        var step = [Double](repeating: options.initialStep, count: n)
        func kde() throws -> ParallaxSubpixelResult {
            try kernelDensity(images: images, rows: rows, columns: columns, shifts: shifts, factor: factor,
                sigmaPixels: sigmaPixels, lanczosOrder: lanczosOrder, lowpass: lowpass,
                probeOffsets: ParallaxProbeOffsets(rows: dr, columns: dc))
        }
        var image = try kde()
        var scores = Self.scores(image, images: images, rows: rows, columns: columns, shifts: shifts, dr: dr, dc: dc, window: window)
        var history = [scores.reduce(0, +) / Double(n)]
        let dirs: [(Double, Double)] = options.checkerboard ? [(-1, 0), (1, 0), (0, -1), (0, 1)] : [(-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)]
        for _ in 0..<options.iterations {
            let trials: [[Double]] = dirs.map { d in
                Self.scores(image, images: images, rows: rows, columns: columns, shifts: shifts,
                    dr: (0..<n).map { dr[$0] + d.0 * step[$0] }, dc: (0..<n).map { dc[$0] + d.1 * step[$0] }, window: nil)
            }
            var update = [Bool](repeating: false, count: n)
            if options.checkerboard {
                for i in 0..<n {
                    var best = 0, bestScore = trials[0][i] * window[i]
                    for k in 1..<4 { let v = trials[k][i] * window[i]; if v < bestScore { best = k; bestScore = v } }
                    if bestScore < scores[i] {
                        update[i] = true
                        dr[i] += dirs[best].0 * step[i] * window[i]; dc[i] += dirs[best].1 * step[i] * window[i]
                    }
                }
            } else {
                var ddr = [Double](repeating: 0, count: n), ddc = ddr
                for i in 0..<n {
                    let gr = trials[0][i] - trials[1][i], gc = trials[2][i] - trials[3][i]
                    let d = hypot(gr, gc) / step[i]
                    ddr[i] = gr * window[i] / d; ddc[i] = gc * window[i] / d
                }
                let fixed = Self.scores(image, images: images, rows: rows, columns: columns, shifts: shifts,
                    dr: (0..<n).map { dr[$0] + ddr[$0] }, dc: (0..<n).map { dc[$0] + ddc[$0] }, window: window)
                for i in 0..<n where fixed[i] < scores[i] { update[i] = true; dr[i] += ddr[i]; dc[i] += ddc[i] }
            }
            for i in 0..<n { if !update[i] { step[i] *= options.stepFactor }; step[i] = max(step[i], options.minimumStep) }
            if let g = options.gaussianSigma {
                dr = gaussian(dr, rows, columns, g.row / scanSampling.row)
                dc = gaussian(dc, rows, columns, g.column / scanSampling.column)
            }
            if options.lowpass != nil || options.highpass != nil {
                let fft = FFT2D(rows: rows, columns: columns)
                func filter(_ a: [Double], _ low: Double?, _ high: Double?, _ order: Int) -> [Double] {
                    let mean = a.reduce(0, +) / Double(n)
                    var re = a.map { $0 - mean }, im = [Double](repeating: 0, count: n)
                    fft.transform(&re, &im, inverse: false)
                    for r in 0..<rows { for c in 0..<columns {
                        let q = hypot(parallaxFrequency(r, rows) / scanSampling.row, parallaxFrequency(c, columns) / scanSampling.column)
                        var env = 1.0
                        if let h = high, h != 0 { env *= 1 - 1 / (1 + pow(q / h, Double(2 * order))) }
                        if let l = low, l != 0 { env *= 1 / (1 + pow(q / l, Double(2 * order))) }
                        re[r * columns + c] *= env; im[r * columns + c] *= env
                    } }
                    fft.transform(&re, &im, inverse: true)
                    return re.map { $0 + mean }
                }
                dr = filter(dr, options.lowpass?.row, options.highpass?.row, options.butterworthOrder.0)
                dc = filter(dc, options.lowpass?.column, options.highpass?.column, options.butterworthOrder.1)
            }
            image = try kde()
            scores = Self.scores(image, images: images, rows: rows, columns: columns, shifts: shifts, dr: dr, dc: dc, window: window)
            history.append(scores.reduce(0, +) / Double(n))
        }
        guard dr.allSatisfy(\.isFinite), dc.allSatisfy(\.isFinite) else { throw ParallaxError.nonfiniteValues }
        return .init(offsets: ParallaxProbeOffsets(rows: dr, columns: dc), scores: history, image: image)
    }
}
