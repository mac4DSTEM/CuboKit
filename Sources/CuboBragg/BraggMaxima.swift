//
//  BraggMaxima.swift
//  py4DSTEM's get_maxima_2D and filter_2D_maxima (preprocess/utils.py at f050d207): Gaussian smoothing, the 8-neighbour
//  maxima (>= towards increasing indices, > towards decreasing, so a plateau yields its first pixel), the edge boundary, the
//  filter cascade (absolute, relative, spacing, count) and the subpixel refinement, in Double.
//
//  Ties between equal intensities are ordered by raster position (py4DSTEM's order there is NumPy's unstable quicksort,
//  reversed; the pins have no ties).
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
        try validate(o)
        guard rows > 0, columns > 0, ar.count == rows * columns else {
            throw CuboBraggError.shapeMismatch("\(ar.count) values for \(rows) × \(columns)")
        }
        let smooth = o.correlationSigma > 0
            ? try GaussianFilter(sigma: o.correlationSigma).apply(ar, rows: rows, columns: columns, edge: .reflect) : ar

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
                        found.append(BraggPeak(row: Double(r), column: Double(c), intensity: v))
                    }
                }
            }
        }
        found = brightestFirst(found)
        guard !found.isEmpty else { return [] }
        found = filter(found, o)
        guard o.subpixel != .pixel else { return found }

        // Parabolic refinement on the smoothed image, intensity by bilinear interpolation there.
        let w = columns
        for k in found.indices {
            let r = Int(found[k].row), c = Int(found[k].column), i = r * w + c
            let i0 = smooth[i]
            let dr = (smooth[i + w] - smooth[i - w]) / (4 * i0 - 2 * smooth[i + w] - 2 * smooth[i - w])
            let dc = (smooth[i + 1] - smooth[i - 1]) / (4 * i0 - 2 * smooth[i + 1] - 2 * smooth[i - 1])
            guard dr.isFinite, dc.isFinite else { throw CuboBraggError.nonFiniteRefinement }
            found[k].row += dr
            found[k].column += dc
            found[k].intensity = bilinear(smooth, columns: columns, row: found[k].row, column: found[k].column)
        }
        guard o.subpixel == .multicorr else { return found }

        // Matrix-DFT upsampling of the unsmoothed complex correlation around the (half-pixel rounded) parabolic position.
        var re: [Double], im: [Double]
        if let fourier {
            guard fourier.re.count == ar.count, fourier.im.count == ar.count else {
                throw CuboBraggError.shapeMismatch("the Fourier correlation holds \(fourier.re.count) values for \(rows) × \(columns)")
            }
            (re, im) = fourier
        } else {
            re = smooth; im = [Double](repeating: 0, count: smooth.count)
            FFT2D(rows: rows, columns: columns).transform(&re, &im, inverse: false)
        }
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
        return o.multicorrOrder == .refinedIntensity ? brightestFirst(found) : found
    }

    /// py4DSTEM's `filter_2D_maxima` on maxima already ordered brightest first (with mac4DSTEM's optional reference radius).
    static func filter(_ input: [BraggPeak], _ o: BraggDiskOptions) -> [BraggPeak] {
        var found = input
        if o.minimumAbsoluteIntensity > 0 { found.removeAll { $0.intensity < o.minimumAbsoluteIntensity } }
        if o.minimumRelativeIntensity > 0 {
            var references = found
            if o.relativeReferenceMinimumRadius > 0, let beam = found.first {
                references = found.filter {
                    hypot($0.row - beam.row, $0.column - beam.column) >= o.relativeReferenceMinimumRadius
                }
            }
            if references.count > o.relativeToPeak {
                let reference = references[o.relativeToPeak].intensity
                found.removeAll { $0.intensity / reference < o.minimumRelativeIntensity }
            }
        }
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
        if found.count > o.maximumPeaks { found = Array(found.prefix(o.maximumPeaks)) }
        return found
    }

    /// Brightest first; equal intensities keep their current order.
    static func brightestFirst(_ peaks: [BraggPeak]) -> [BraggPeak] {
        peaks.enumerated().sorted { a, b in
            a.element.intensity != b.element.intensity ? a.element.intensity > b.element.intensity : a.offset < b.offset
        }.map(\.element)
    }

    /// py4DSTEM's linear_interpolation_2D: floor and ceil neighbours.
    static func bilinear(_ a: [Double], columns: Int, row: Double, column: Double) -> Double {
        let r0 = Int(row.rounded(.down)), r1 = Int(row.rounded(.up))
        let c0 = Int(column.rounded(.down)), c1 = Int(column.rounded(.up))
        let dr = row - Double(r0), dc = column - Double(c0)
        return (1 - dr) * (1 - dc) * a[r0 * columns + c0] + dr * (1 - dc) * a[r1 * columns + c0]
            + (1 - dr) * dc * a[r0 * columns + c1] + dr * dc * a[r1 * columns + c1]
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
