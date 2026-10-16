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
        factor: Double, sigmaPixels: Double = 0.125, lanczosOrder: Int? = nil, lowpass: Bool = false) throws -> ParallaxSubpixelResult {
        guard !images.isEmpty, images.count == shifts.count, factor.isFinite, factor >= 1, sigmaPixels.isFinite,
            sigmaPixels >= 0, lanczosOrder == nil || lanczosOrder! >= 1,
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
                let xr = (Double(r) + shifts[b].row) * factor, xc = (Double(c) + shifts[b].column) * factor
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
}
