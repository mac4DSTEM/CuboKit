//
//  BraggMaxima.swift
//  py4DSTEM's get_maxima_2D and filter_2D_maxima (preprocess/utils.py at f050d207): Gaussian smoothing, the 8-neighbour
//  maxima (>= towards increasing indices, > towards decreasing, so a plateau yields its first pixel), the edge boundary, the
//  filter cascade (absolute, relative, spacing, count) and the subpixel refinement, in Double.
//
//  Ties between equal intensities are ordered by raster position by default
//  (BraggDiskOptions.tieOrder = .rasterOrder); .numpy reproduces py4DSTEM's order (larger row, then larger column).
//

import CuboCore
import CuboParallax
import CuboVirtual
import Foundation

public enum BraggMaxima {
    /// The maxima of a row-major `rows × columns` image `ar`, as py4DSTEM's `get_maxima_2D` with `ar` = the correlation.
    ///
    /// `fourier` is py4DSTEM's `_ar_FT`: the complex Fourier-domain correlation (real and imaginary parts) the `.multicorr`
    /// refinement works on; when nil, the transform of the smoothed `ar` is used. The filter order is absolute intensity,
    /// relative intensity, spacing, count; the edge boundary acts on the integer maxima, before the refinement.
    public static func find(in ar: [Double], rows: Int, columns: Int, options o: BraggDiskOptions,
                            fourier: (re: [Double], im: [Double])? = nil) throws -> [BraggPeak] {
        try findWithFunnel(in: ar, rows: rows, columns: columns, options: o, fourier: fourier).peaks
    }

    /// `find`, with the count of maxima that survived each stage (`BraggFunnel`); the peaks are those of `find`.
    /// `nonFinitePixelCount` of the funnel is 0 here: this function sees only the correlation.
    public static func findWithFunnel(in ar: [Double], rows: Int, columns: Int, options o: BraggDiskOptions,
                                      fourier: (re: [Double], im: [Double])? = nil) throws -> (peaks: [BraggPeak], funnel: BraggFunnel) {
        try validate(o)
        guard rows > 0, columns > 0, ar.count == rows * columns else {
            throw CuboBraggError.shapeMismatch("\(ar.count) values for \(rows) × \(columns)")
        }
        let smooth = o.correlationSigma > 0
            ? try GaussianFilter(sigma: o.correlationSigma).apply(ar, rows: rows, columns: columns, edge: .reflect) : ar
        let maximum = largest(smooth)
        return try smooth.withUnsafeBufferPointer { buffer in
            try select(smooth: buffer.baseAddress!, rows: rows, columns: columns, options: o, correlationMaximum: maximum) {
                if let fourier {
                    guard fourier.re.count == ar.count, fourier.im.count == ar.count else {
                        throw CuboBraggError.shapeMismatch("the Fourier correlation holds \(fourier.re.count) values for \(rows) × \(columns)")
                    }
                    return fourier
                }
                var re = smooth, im = [Double](repeating: 0, count: smooth.count)
                FFT2D(rows: rows, columns: columns).transform(&re, &im, inverse: false)
                return (re, im)
            }
        }
    }

    /// The largest value (0 for an empty or all-NaN map).
    static func largest(_ a: [Double]) -> Double {
        var m = -Double.infinity
        for x in a where x > m { m = x }
        return m == -Double.infinity ? 0 : m
    }

    /// The stages after the smoothing, on a smoothed correlation in either precision: the 8-neighbour maxima inside the edge
    /// boundary, the filter cascade, the parabolic refinement and (`.multicorr`) the matrix-DFT refinement on the complex
    /// correlation `fourier` supplies (called only when there is a peak to refine).
    static func select<S: BinaryFloatingPoint>(smooth: UnsafePointer<S>, rows: Int, columns: Int, options o: BraggDiskOptions,
                                               correlationMaximum: Double,
                                               fourier: () throws -> (re: [Double], im: [Double])) throws -> (peaks: [BraggPeak], funnel: BraggFunnel) {
        // Local maxima, edge boundary applied to the integer position.
        let eb = max(1, o.edgeBoundary)
        var found: [BraggPeak] = []
        if rows - eb > eb, columns - eb > eb {
            let w = columns
            for r in eb..<(rows - eb) {
                for c in eb..<(columns - eb) {
                    let i = r * w + c, v = smooth[i]
                    if v >= smooth[i + w], v > smooth[i - w], v >= smooth[i + 1], v > smooth[i - 1],
                       v >= smooth[i + w + 1], v > smooth[i + w - 1], v >= smooth[i - w + 1], v > smooth[i - w - 1] {
                        found.append(BraggPeak(row: Double(r), column: Double(c), intensity: Double(v)))
                    }
                }
            }
        }
        found = brightestFirst(found, o.tieOrder)
        guard !found.isEmpty else {
            return ([], BraggFunnel(localMaximumCount: 0, afterAbsoluteCutCount: 0, afterRelativeCutCount: 0, afterSpacingCount: 0,
                                    acceptedCount: 0, wasCountLimited: false, relativeReferenceIntensity: nil,
                                    relativeReferenceWasAvailable: o.minimumRelativeIntensity == 0,
                                    correlationMaximum: correlationMaximum, nonFinitePixelCount: 0))
        }
        let cascaded = cascade(found, o)
        found = cascaded.peaks
        let funnel = BraggFunnel(
            localMaximumCount: cascaded.local, afterAbsoluteCutCount: cascaded.afterAbsolute,
            afterRelativeCutCount: cascaded.afterRelative, afterSpacingCount: cascaded.afterSpacing,
            acceptedCount: found.count, wasCountLimited: cascaded.afterSpacing > o.maximumPeaks,
            relativeReferenceIntensity: cascaded.reference, relativeReferenceWasAvailable: cascaded.referenceAvailable,
            correlationMaximum: correlationMaximum, nonFinitePixelCount: 0)
        guard o.subpixel != .pixel else { return (found, funnel) }

        // Parabolic refinement on the smoothed image, intensity by bilinear interpolation there.
        let w = columns
        for k in found.indices {
            let r = Int(found[k].row), c = Int(found[k].column), i = r * w + c
            let i0 = Double(smooth[i])
            let up = Double(smooth[i + w]), down = Double(smooth[i - w]), right = Double(smooth[i + 1]), left = Double(smooth[i - 1])
            let dr = (up - down) / (4 * i0 - 2 * up - 2 * down)
            let dc = (right - left) / (4 * i0 - 2 * right - 2 * left)
            guard dr.isFinite, dc.isFinite else { throw CuboBraggError.nonFiniteRefinement }
            found[k].row += dr
            found[k].column += dc
            found[k].intensity = bilinear(smooth, columns: columns, row: found[k].row, column: found[k].column)
        }
        guard o.subpixel == .multicorr, !found.isEmpty else { return (found, funnel) }

        // Matrix-DFT upsampling of the unsmoothed complex correlation around the (half-pixel rounded) parabolic position.
        let (re, im) = try fourier()
        do {
            for k in found.indices {
                let refined = try MatrixDFTCorrelation.refine(
                    correlationRe: re, correlationIm: im, width: columns, height: rows,
                    initial: CorrelationPeak(row: found[k].row, column: found[k].column), upsampleFactor: o.upsampleFactor)
                found[k].row = refined.row
                found[k].column = refined.column
            }
        } catch {
            throw CuboBraggError.nonFiniteRefinement
        }
        return (o.multicorrOrder == .refinedIntensity ? brightestFirst(found, o.tieOrder) : found, funnel)
    }

    /// py4DSTEM's `filter_2D_maxima` on maxima already ordered brightest first (with mac4DSTEM's optional reference radius).
    static func filter(_ input: [BraggPeak], _ o: BraggDiskOptions) -> [BraggPeak] { cascade(input, o).peaks }

    /// The filter cascade with the count after each stage and the relative reference it used.
    static func cascade(_ input: [BraggPeak], _ o: BraggDiskOptions)
        -> (peaks: [BraggPeak], local: Int, afterAbsolute: Int, afterRelative: Int, afterSpacing: Int,
            reference: Double?, referenceAvailable: Bool) {
        var found = input
        let local = found.count
        if o.minimumAbsoluteIntensity > 0 { found.removeAll { $0.intensity < o.minimumAbsoluteIntensity } }
        let afterAbsolute = found.count
        var reference: Double?
        var referenceAvailable = o.minimumRelativeIntensity == 0
        if o.minimumRelativeIntensity > 0 {
            var references = found
            if o.relativeReferenceMinimumRadius > 0, let beam = found.first {
                references = found.filter {
                    hypot($0.row - beam.row, $0.column - beam.column) >= o.relativeReferenceMinimumRadius
                }
            }
            if references.count > o.relativeToPeak {
                let r = references[o.relativeToPeak].intensity
                reference = r
                referenceAvailable = true
                found.removeAll { $0.intensity / r < o.minimumRelativeIntensity }
            }
        }
        let afterRelative = found.count
        if o.minimumPeakSpacing > 0 {
            let limit = o.minimumPeakSpacing * o.minimumPeakSpacing
            var kept: [BraggPeak] = []
            outer: for candidate in found {
                for k in kept {
                    let dr = candidate.row - k.row, dc = candidate.column - k.column
                    if dr * dr + dc * dc < limit { continue outer }
                }
                kept.append(candidate)
            }
            found = kept
        }
        let afterSpacing = found.count
        if found.count > o.maximumPeaks { found = Array(found.prefix(o.maximumPeaks)) }
        return (found, local, afterAbsolute, afterRelative, afterSpacing, reference, referenceAvailable)
    }

    /// Brightest first; equal intensities keep their current order, or follow NumPy's (larger row, then larger column first).
    static func brightestFirst(_ peaks: [BraggPeak], _ ties: BraggTieOrder = .rasterOrder) -> [BraggPeak] {
        peaks.enumerated().sorted { a, b in
            if a.element.intensity != b.element.intensity { return a.element.intensity > b.element.intensity }
            if ties == .numpy {
                if a.element.row != b.element.row { return a.element.row > b.element.row }
                if a.element.column != b.element.column { return a.element.column > b.element.column }
            }
            return a.offset < b.offset
        }.map(\.element)
    }

    /// py4DSTEM's linear_interpolation_2D: floor and ceil neighbours.
    static func bilinear<S: BinaryFloatingPoint>(_ a: UnsafePointer<S>, columns: Int, row: Double, column: Double) -> Double {
        let r0 = Int(row.rounded(.down)), r1 = Int(row.rounded(.up))
        let c0 = Int(column.rounded(.down)), c1 = Int(column.rounded(.up))
        let dr = row - Double(r0), dc = column - Double(c0)
        return (1 - dr) * (1 - dc) * Double(a[r0 * columns + c0]) + dr * (1 - dc) * Double(a[r1 * columns + c0])
            + (1 - dr) * dc * Double(a[r0 * columns + c1]) + dr * dc * Double(a[r1 * columns + c1])
    }

    static func validate(_ o: BraggDiskOptions) throws {
        func bad(_ m: String) -> CuboBraggError { .invalidOption(m) }
        guard o.correlationPower.isFinite, o.correlationPower >= 0 else { throw bad("correlationPower must be finite and not negative (\(o.correlationPower))") }
        guard o.patternSigma.isFinite, o.patternSigma >= 0 else { throw bad("patternSigma must be finite and not negative (\(o.patternSigma))") }
        guard o.correlationSigma.isFinite, o.correlationSigma >= 0 else { throw bad("correlationSigma must be finite and not negative (\(o.correlationSigma))") }
        if o.subpixel == .multicorr {
            guard o.upsampleFactor > 2 else { throw bad("upsampleFactor must be above 2 for .multicorr (\(o.upsampleFactor))") }
        }
        guard o.minimumAbsoluteIntensity.isFinite, o.minimumRelativeIntensity.isFinite, o.minimumRelativeIntensity >= 0 else {
            throw bad("the intensity cuts must be finite, the relative cut not negative")
        }
        guard o.relativeToPeak >= 0 else { throw bad("relativeToPeak must not be negative (\(o.relativeToPeak))") }
        guard o.minimumPeakSpacing.isFinite, o.minimumPeakSpacing >= 0 else { throw bad("minimumPeakSpacing must be finite and not negative") }
        guard o.relativeReferenceMinimumRadius.isFinite, o.relativeReferenceMinimumRadius >= 0 else {
            throw bad("relativeReferenceMinimumRadius must be finite and not negative")
        }
        guard o.maximumPeaks >= 0 else { throw bad("maximumPeaks must not be negative (\(o.maximumPeaks))") }
    }
}
