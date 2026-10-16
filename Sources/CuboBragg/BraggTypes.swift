//
//  BraggTypes.swift
//  The public vocabulary of Bragg disk detection: a peak, the options of py4DSTEM's find_Bragg_disks and the named
//  mac4DSTEM variants, and the errors. Rows and columns, never x and y: py4DSTEM's qx is the row here.
//

import Foundation

/// One detected Bragg reflection, in detector pixels. `row` is py4DSTEM's `qx`, `column` its `qy`.
public struct BraggPeak: Sendable, Equatable {
    public var row: Double
    public var column: Double
    /// The correlation intensity at the peak (on the smoothed correlation, bilinearly interpolated after refinement).
    public var intensity: Double

    public init(row: Double, column: Double, intensity: Double) {
        self.row = row
        self.column = column
        self.intensity = intensity
    }
}

/// py4DSTEM's `subpixel` choices.
public enum BraggSubpixel: Sendable, Hashable, CaseIterable {
    /// `"pixel"`: the integer position of the correlation maximum.
    case pixel
    /// `"poly"`: a three-point parabola in each direction on the smoothed correlation; the intensity is bilinearly
    /// interpolated there.
    case parabolic
    /// `"multicorr"`: `.parabolic`, then a matrix-DFT upsampling of the unsmoothed complex correlation around the
    /// half-pixel-rounded position (Guizar-Sicairos et al.), py4DSTEM's default.
    case multicorr
}

/// How the peaks of a `.multicorr` run are ordered.
public enum BraggPeakOrder: Sendable, Hashable {
    /// py4DSTEM: sorted again by the refined (bilinear) intensity, brightest first.
    case refinedIntensity
    /// mac4DSTEM variant: left in the order of the integer-pixel correlation maxima (not sorted again). The two differ
    /// only when refinement reorders two peaks.
    case correlationIntensity
}

/// What happens to a pattern pixel that is NaN or infinite.
public enum BraggNonFinitePixels: Sendable, Hashable {
    /// py4DSTEM: the value enters the FFT, spreads to every correlation bin, and the pattern yields no peak.
    case propagate
    /// mac4DSTEM variant: each such pixel is replaced by the median of its finite 8-neighbours (0 when it has none) before
    /// any smoothing or correlation.
    case neighbourMedian
}

/// The options of py4DSTEM's `find_Bragg_disks` with its names made explicit, defaults as in py4DSTEM f050d207
/// (`minRelativeIntensity` 0.005, `minPeakSpacing` 60, `edgeBoundary` 20, `maxNumPeaks` 70, `subpixel` multicorr), plus
/// the mac4DSTEM variants, each off by default.
public struct BraggDiskOptions: Sendable, Equatable {
    /// `corrPower`: 1 is a cross-correlation, 0 a phase correlation, between them a hybrid. Finite, not negative.
    public var correlationPower: Double = 1
    /// `sigma_dp`: Gaussian σ (pixels) applied to the pattern before correlating; 0 does nothing.
    public var patternSigma: Double = 0
    /// `sigma_cc` (or `sigma`): Gaussian σ applied to the correlation before the maxima are taken; 0 does nothing. The
    /// `.multicorr` refinement always uses the unsmoothed complex correlation.
    public var correlationSigma: Double = 2
    public var subpixel: BraggSubpixel = .multicorr
    /// `upsample_factor`, used by `.multicorr` only: an integer above 2.
    public var upsampleFactor: Int = 16
    /// `minAbsoluteIntensity`: maxima dimmer than this are dropped (0 keeps all).
    public var minimumAbsoluteIntensity: Double = 0
    /// `minRelativeIntensity`: maxima dimmer than this fraction of the reference maximum are dropped (0 keeps all).
    public var minimumRelativeIntensity: Double = 0.005
    /// `relativeToPeak`: the reference is the maximum of this rank (0 is the brightest) after the absolute cut.
    public var relativeToPeak: Int = 0
    /// `minPeakSpacing`: of two maxima closer than this (pixels), the dimmer is dropped (0 keeps all).
    public var minimumPeakSpacing: Double = 60
    /// `edgeBoundary`: maxima within this many pixels of the pattern's edge are not considered; values below 1 act as 1.
    public var edgeBoundary: Int = 20
    /// `maxNumPeaks`: at most this many peaks are returned (the brightest).
    public var maximumPeaks: Int = 70

    // mac4DSTEM variants (docs/bragg.md). Each defaults to py4DSTEM's behaviour.

    /// mac4DSTEM `relativeReferenceMinimumRadiusPx`: when above 0, the reference of the relative cut is the
    /// `relativeToPeak`-th brightest maximum among those at least this far (pixels) from the brightest one, so a saturated
    /// direct beam does not set the scale. 0 is py4DSTEM's rule (the brightest anywhere).
    public var relativeReferenceMinimumRadius: Double = 0
    /// mac4DSTEM's treatment of NaN and infinite pixels; py4DSTEM's is `.propagate`.
    public var nonFinitePixels: BraggNonFinitePixels = .propagate
    /// mac4DSTEM leaves `.multicorr` peaks in `.correlationIntensity` order; py4DSTEM sorts by `.refinedIntensity`.
    public var multicorrOrder: BraggPeakOrder = .refinedIntensity

    public init() {}

    /// The mac4DSTEM variants all on: neighbour-median fill, the multicorr order of the integer maxima, and the app's
    /// `minimumRelativeIntensity` default 0.0015 (the reference radius stays 0: the app default). Everything else is
    /// py4DSTEM's, as in the app.
    public static var mac4DSTEM: BraggDiskOptions {
        var o = BraggDiskOptions()
        o.nonFinitePixels = .neighbourMedian
        o.multicorrOrder = .correlationIntensity
        o.minimumRelativeIntensity = 0.0015
        return o
    }

    /// mac4DSTEM `DiskDetectionParams.detectorAdapted`: the pixel-scaled `minimumPeakSpacing` and `edgeBoundary` for a
    /// detector, py4DSTEM's literal 60 and 20 being sized for patterns of about 512 pixels. With a positive finite
    /// `probeRadius` the spacing is the radius rounded, at least 4 and at most the detector-scaled value
    /// max(4, round(min(rows, columns) / 8)); without it, that value. The edge is max(2, trunc(min(rows, columns) / 24)).
    /// Every other field of `base` is kept.
    public static func detectorAdapted(detectorRows: Int, detectorColumns: Int, probeRadius: Double? = nil,
                                       base: BraggDiskOptions = BraggDiskOptions()) -> BraggDiskOptions {
        var o = base
        let qMin = Double(min(detectorRows, detectorColumns))
        let scaled = max(4, (qMin / 8).rounded(.toNearestOrAwayFromZero))
        if let probeRadius, probeRadius.isFinite, probeRadius > 0 {
            o.minimumPeakSpacing = min(max(4, probeRadius.rounded(.toNearestOrAwayFromZero)), scaled)
        } else {
            o.minimumPeakSpacing = scaled
        }
        o.edgeBoundary = max(2, Int(qMin / 24))
        return o
    }
}

/// Bragg peaks for every scan position, row-major `[scanRow][scanColumn]`.
public struct BraggVectors: Sendable, Equatable {
    public let scanRows: Int
    public let scanColumns: Int
    /// `scanRows · scanColumns` lists, each in the order the detection returned.
    public let peaks: [[BraggPeak]]

    public init(scanRows: Int, scanColumns: Int, peaks: [[BraggPeak]]) {
        precondition(peaks.count == scanRows * scanColumns, "BraggVectors: \(peaks.count) lists for \(scanRows) × \(scanColumns)")
        self.scanRows = scanRows
        self.scanColumns = scanColumns
        self.peaks = peaks
    }

    /// The peaks at scan position (`row`, `column`).
    public subscript(row: Int, column: Int) -> [BraggPeak] { peaks[row * scanColumns + column] }
}

public enum CuboBraggError: Error, Equatable, Sendable, LocalizedError {
    /// An option py4DSTEM would turn into NaN, an exception or silence (a negative σ, an upsampling factor of 2 or less
    /// for `.multicorr`, ...).
    case invalidOption(String)
    /// The pattern, the kernel or the cube do not agree in shape.
    case shapeMismatch(String)
    /// A refinement produced a non-finite position (py4DSTEM raises there).
    case nonFiniteRefinement

    public var errorDescription: String? {
        switch self {
        case .invalidOption(let m): "Bragg disk detection: invalid option: \(m)"
        case .shapeMismatch(let m): "Bragg disk detection: shape mismatch: \(m)"
        case .nonFiniteRefinement: "Bragg disk detection: a subpixel refinement was not finite"
        }
    }
}
