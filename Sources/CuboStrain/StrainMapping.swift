//
//  StrainMapping.swift
//  The scan-level strain map after py4DSTEM f050d207: BraggVectors.histogram (the Bragg vector map),
//  StrainMap.choose_basis_vectors (maxima of the map, origin, g1, g2, indexed directions), fit_basis_vectors (indexing of
//  every scan position and the weighted lattice fit), get_reference_g1g2 over a scan region, and get_strain with the
//  coordinate rotation. Composed from `LatticeStrain` and `BraggMaxima`; Double, CPU, no plotting or state.
//
//  Frame: detector pixels, row = py4DSTEM x, column = y. Peaks are centred (the origin already subtracted), as
//  `BraggVectors.get_vectors(center=True)` returns them; the histogram adds the origin back, as `histogram(mode="cal")` does.
//  Scan position (r, c) is element r * scanColumns + c of every flat array.
//

import CuboBragg
import Foundation

/// One Bragg peak of one scan position, in detector pixels relative to the diffraction origin (centred and calibrated).
public struct StrainPeak: Sendable, Equatable {
    /// Row coordinate (py4DSTEM `qx`).
    public var row: Double
    /// Column coordinate (py4DSTEM `qy`).
    public var column: Double
    /// Intensity; the weight of the lattice fit. Finite and not negative.
    public var intensity: Double
    /// Creates a centred peak.
    public init(row: Double, column: Double, intensity: Double) { self.row = row; self.column = column; self.intensity = intensity }
}

/// A scan-level strain input that cannot be used, with the reason.
public enum StrainMappingError: Error, Sendable, Equatable {
    /// A shape, mask, weight, option or peak value is unusable; the text says which.
    case invalidInput(String)
}

/// py4DSTEM's Bragg vector map (`BraggVectors.histogram(mode="cal")`): peak intensity binned bilinearly on the detector grid.
public struct BraggVectorHistogram: Sendable, Equatable {
    /// Rows of the grid, `round(detectorRows * sampling)`.
    public let rows: Int
    /// Columns of the grid.
    public let columns: Int
    /// Row-major values, `rows * columns`.
    public let values: [Double]
    /// The diffraction origin on this grid, `sampling * origin`.
    public let origin: LatticeVector
    /// Grid cells per detector pixel.
    public let sampling: Double
    /// A map from values computed elsewhere (`rows * columns` of them), for ``StrainMapping/chooseBasisVectors(histogram:options:)``.
    public init(rows: Int, columns: Int, values: [Double], origin: LatticeVector, sampling: Double = 1) {
        precondition(rows > 0 && columns > 0 && values.count == rows * columns, "BraggVectorHistogram: \(values.count) values for \(rows) × \(columns)")
        self.rows = rows; self.columns = columns; self.values = values; self.origin = origin; self.sampling = sampling
    }
}

/// The `choose_basis_vectors` parameters, py4DSTEM's names and defaults (`maxNumPeaks` 10, `edgeBoundary` 1, `subpixel`
/// multicorr with `upsample_factor` 16, no smoothing or intensity cuts), mapped onto the kit's maxima finder.
public struct StrainBasisOptions: Sendable, Equatable {
    /// `subpixel`: pixel, parabolic (`"poly"`) or multicorr.
    public var subpixel: BraggSubpixel = .multicorr
    /// `upsample_factor`, multicorr only.
    public var upsampleFactor: Int = 16
    /// `sigma`: Gaussian σ (grid cells) applied to the map before the maxima are taken; 0 does nothing.
    public var sigma: Double = 0
    /// `minAbsoluteIntensity`.
    public var minimumAbsoluteIntensity: Double = 0
    /// `minRelativeIntensity`.
    public var minimumRelativeIntensity: Double = 0
    /// `relativeToPeak`.
    public var relativeToPeak: Int = 0
    /// `minSpacing`: of two maxima closer than this the dimmer is dropped.
    public var minimumSpacing: Double = 0
    /// `edgeBoundary`; values below 1 act as 1.
    public var edgeBoundary: Int = 1
    /// `maxNumPeaks`; nil keeps all.
    public var maximumPeaks: Int? = 10
    /// `index_origin`: the maximum taken as the lattice origin; nil guesses the one nearest the diffraction origin.
    public var indexOrigin: Int?
    /// `index_g1`: nil guesses the next nearest.
    public var indexG1: Int?
    /// `index_g2`: nil guesses by distance and angle.
    public var indexG2: Int?
    /// Creates py4DSTEM's defaults.
    public init() {}
}

/// The maxima, the chosen basis and the indexed lattice directions that fit_basis_vectors indexes against.
public struct StrainDirections: Sendable {
    /// The maxima of the Bragg vector map in grid coordinates, brightest first (py4DSTEM `g`); empty for ``lattice(basis:order:)``.
    public let maxima: [BraggPeak]
    /// Which maximum is the origin, g1 and g2 (nil for ``lattice(basis:order:)``).
    public let choice: LatticeStrain.BasisChoice?
    /// The basis relative to the lattice origin.
    public let basis: LatticeBasis
    /// The directions in the centred frame (the diffraction origin subtracted), with their integer indices h, k
    /// (py4DSTEM `braggdirections`).
    public let directions: [LatticePeak]

    /// All lattice points h·g1 + k·g2, |h|, |k| ≤ `order`, about the diffraction origin, for a basis given by the caller
    /// (a known lattice, or one from ``LatticeStrain/chooseBasisConsensus(_:origin:minimumRadius:)``).
    public static func lattice(basis: LatticeBasis, order: Int = 3) throws -> StrainDirections {
        guard order >= 1, order <= 64 else { throw StrainMappingError.invalidInput("order must be 1...64 (\(order))") }
        guard basis.g1.row.isFinite, basis.g1.column.isFinite, basis.g2.row.isFinite, basis.g2.column.isFinite,
              basis.g1.row * basis.g2.column - basis.g1.column * basis.g2.row != 0 else { throw LatticeStrainError.singularBasis }
        var d: [LatticePeak] = []
        for h in -order...order {
            for k in -order...order {
                d.append(.init(row: Double(h) * basis.g1.row + Double(k) * basis.g2.row,
                               column: Double(h) * basis.g1.column + Double(k) * basis.g2.column,
                               intensity: 1, h: Double(h), k: Double(k)))
            }
        }
        return .init(maxima: [], choice: nil, basis: basis, directions: d)
    }
}

/// How the reference lattice is chosen (`get_strain`'s `gvects`).
public enum StrainReference: Sendable, Equatable {
    /// `gvects=None`: the median of the fitted lattice over every fitted position.
    case fittedMedian
    /// A boolean scan array: the median over the fitted positions where it is true.
    case region([Bool])
    /// A lattice given directly (g1 and g2 in the centred frame).
    case supplied(LatticeBasis)
}

/// How each position's lattice is fitted.
public enum StrainFitMethod: Sendable, Equatable {
    /// py4DSTEM's `fit_lattice_vectors`: design and data multiplied by intensity, so the objective weights intensity squared.
    case py4DSTEM
    /// mac4DSTEM's intensity-once normal equations in Float32 (``LatticeStrain/fitIntensityOnce(_:)``).
    case intensityOnce
}

/// How the median reference is taken.
public enum StrainReferenceMethod: Sendable, Equatable {
    /// py4DSTEM's plain component medians.
    case median
    /// mac4DSTEM's rejection of distorted candidates first (``LatticeStrain/referenceMAD(_:)``).
    case madRejection
}

/// The `fit_basis_vectors` and `get_strain` parameters.
public struct StrainMapOptions: Sendable, Equatable {
    /// `max_peak_spacing`: a peak farther than this (pixels, inclusive) from every direction is not indexed or fitted.
    public var maximumPeakSpacing: Double
    /// `minNumPeaks` of `fit_lattice_vectors`: a position with fewer indexed peaks is not fitted.
    public var minimumPeaks: Int = 5
    /// `mask` of `fit_basis_vectors`: scan positions to fit (`scanRows * scanColumns` flags); nil fits all.
    public var scanMask: [Bool]?
    /// `gvects` of `get_strain`.
    public var reference: StrainReference = .fittedMedian
    /// Median (py4DSTEM) or the MAD-rejection variant, for ``StrainReference/fittedMedian`` and ``StrainReference/region(_:)``.
    public var referenceMethod: StrainReferenceMethod = .median
    /// py4DSTEM's fit or the intensity-once variant.
    public var fitMethod: StrainFitMethod = .py4DSTEM
    /// `coordinate_rotation`: degrees, counterclockwise; exx, eyy and exy are expressed in the rotated axes, theta is unchanged.
    public var coordinateRotation: Double = 0
    /// Creates options with py4DSTEM's defaults.
    public init(maximumPeakSpacing: Double) { self.maximumPeakSpacing = maximumPeakSpacing }
}

/// The maps and intermediate results of ``StrainMapping/compute(peaks:scanRows:scanColumns:detectorRows:detectorColumns:origin:directions:basisOptions:options:)``.
/// Every array is row-major `scanRows * scanColumns`; positions that were not fitted hold NaN.
public struct StrainMapResult: Sendable {
    /// Scan rows.
    public let scanRows: Int
    /// Scan columns.
    public let scanColumns: Int
    /// The Bragg vector map when it was built here, nil when the caller supplied the directions.
    public let histogram: BraggVectorHistogram?
    /// The maxima, basis and directions used for indexing.
    public let directions: StrainDirections
    /// True where the position is in the scan mask and was fitted (py4DSTEM's final `mask`).
    public let valid: [Bool]
    /// Peaks indexed at each position (those within the maximum peak spacing of a direction).
    public let indexedPeakCount: [Int]
    /// Fitted lattice origin, row (py4DSTEM `x0`).
    public let originRow: [Double]
    /// Fitted lattice origin, column (`y0`).
    public let originColumn: [Double]
    /// Fitted g1, row component (`g1x`).
    public let g1Row: [Double]
    /// Fitted g1, column component (`g1y`).
    public let g1Column: [Double]
    /// Fitted g2, row component (`g2x`).
    public let g2Row: [Double]
    /// Fitted g2, column component (`g2y`).
    public let g2Column: [Double]
    /// Intensity-weighted mean distance of the indexed peaks from the fitted lattice (`error`).
    public let residual: [Double]
    /// The reference lattice the strain is measured against.
    public let reference: LatticeBasis
    /// Positions that entered the median (0 for a supplied reference).
    public let referencePositionCount: Int
    /// How many of those positions the reference is the median of: all of them for the median method, the MAD-retained
    /// ones for `.madRejection` (see ``LatticeStrain/referenceMADResult(_:)``); 0 for a supplied reference.
    public let referenceInlierCount: Int
    /// Row-axis strain, `e_xx` (in the rotated axes when `coordinateRotation` is not 0).
    public let exx: [Double]
    /// Column-axis strain, `e_yy`.
    public let eyy: [Double]
    /// Shear strain, `e_xy`.
    public let exy: [Double]
    /// Rotation of the lattice relative to the reference, radians, py4DSTEM's sign.
    public let theta: [Double]
    /// The coordinate rotation applied, degrees.
    public let coordinateRotation: Double
}

/// Scan-level strain mapping after py4DSTEM's `StrainMap` (see `docs/strain.md`).
public enum StrainMapping {
    /// py4DSTEM's `BraggVectors.histogram(mode="cal")`: the Bragg vector map.
    ///
    /// Each centred peak is moved onto the grid (`(coordinate + origin) * sampling`) and its intensity shared bilinearly
    /// between the four surrounding cells; peaks whose neighbours fall outside the grid are dropped. The grid is
    /// `round(detector * sampling)` cells (ties to even). `scanMask` keeps the positions where it is true;
    /// `scanWeights` keeps those with weight above `weightsThreshold` and scales their intensities (not both).
    /// `peaks` holds one list per scan position, row-major.
    public static func histogram(peaks: [[StrainPeak]], scanRows: Int, scanColumns: Int,
                                 detectorRows: Int, detectorColumns: Int, origin: LatticeVector,
                                 sampling: Double = 1, scanMask: [Bool]? = nil, scanWeights: [Double]? = nil,
                                 weightsThreshold: Double = 0.005) throws -> BraggVectorHistogram {
        try checkScan(peaks, scanRows, scanColumns)
        guard detectorRows > 0, detectorColumns > 0 else { throw bad("the detector must be at least 1 × 1 (\(detectorRows) × \(detectorColumns))") }
        guard origin.row.isFinite, origin.column.isFinite else { throw bad("the origin must be finite") }
        guard sampling.isFinite, sampling > 0 else { throw bad("sampling must be finite and positive (\(sampling))") }
        let n = scanRows * scanColumns
        guard scanMask == nil || scanWeights == nil else { throw bad("give a scan mask or scan weights, not both") }
        if let scanMask { guard scanMask.count == n else { throw bad("the scan mask holds \(scanMask.count) flags for \(n) positions") } }
        if let scanWeights {
            guard scanWeights.count == n, scanWeights.allSatisfy(\.isFinite), weightsThreshold.isFinite else {
                throw bad("the scan weights must be finite, \(n) of them")
            }
        }
        let gridRows = Int((Double(detectorRows) * sampling).rounded(.toNearestOrEven))
        let gridColumns = Int((Double(detectorColumns) * sampling).rounded(.toNearestOrEven))
        guard gridRows > 0, gridColumns > 0, gridRows <= 1 << 15, gridColumns <= 1 << 15 else {
            throw bad("the histogram grid \(gridRows) × \(gridColumns) is empty or larger than 32768 on a side")
        }
        let size = gridRows * gridColumns
        var b00 = [Double](repeating: 0, count: size), b01 = b00, b10 = b00, b11 = b00
        for position in 0..<n {
            var weight = 1.0
            if let scanMask { guard scanMask[position] else { continue } }
            if let scanWeights { guard scanWeights[position] > weightsThreshold else { continue }; weight = scanWeights[position] }
            for p in peaks[position] {
                let intensity = scanWeights == nil ? p.intensity : p.intensity * weight
                let qr = (p.row + origin.row) * sampling, qc = (p.column + origin.column) * sampling
                guard qr.isFinite, qc.isFinite, intensity.isFinite else { throw bad("a peak at scan position \(position) is not finite") }
                guard qr > -1e9, qr < 1e9, qc > -1e9, qc < 1e9 else { continue }   // far outside the grid
                let fr = Int(qr.rounded(.down)), cr = Int(qr.rounded(.up))
                let fc = Int(qc.rounded(.down)), cc = Int(qc.rounded(.up))
                guard fr >= 0, fc >= 0, cr < gridRows, cc < gridColumns else { continue }
                let dr = qr - Double(fr), dc = qc - Double(fc)
                b00[fr * gridColumns + fc] += intensity * (1.0 - dr) * (1.0 - dc)
                b01[fr * gridColumns + cc] += intensity * (1.0 - dr) * dc
                b10[cr * gridColumns + fc] += intensity * dr * (1.0 - dc)
                b11[cr * gridColumns + cc] += intensity * dr * dc
            }
        }
        var values = [Double](repeating: 0, count: size)
        for i in 0..<size { values[i] = b00[i] + b01[i] + b10[i] + b11[i] }
        return BraggVectorHistogram(rows: gridRows, columns: gridColumns, values: values,
                                    origin: .init(row: sampling * origin.row, column: sampling * origin.column), sampling: sampling)
    }

    /// py4DSTEM's `StrainMap.choose_basis_vectors` (without the figure): the maxima of the Bragg vector map by
    /// `get_maxima_2D`, the origin, g1 and g2 by `options` (or guessed from the distance to the diffraction origin and the
    /// angle), and the indexing of every maximum. Requires a map of sampling 1, as `StrainMap` builds it.
    public static func chooseBasisVectors(histogram: BraggVectorHistogram,
                                          options: StrainBasisOptions = StrainBasisOptions()) throws -> StrainDirections {
        guard histogram.sampling == 1 else { throw bad("choose_basis_vectors needs a Bragg vector map of sampling 1 (\(histogram.sampling))") }
        var o = BraggDiskOptions()
        o.subpixel = options.subpixel
        o.upsampleFactor = options.upsampleFactor
        o.correlationSigma = options.sigma
        o.minimumAbsoluteIntensity = options.minimumAbsoluteIntensity
        o.minimumRelativeIntensity = options.minimumRelativeIntensity
        o.relativeToPeak = options.relativeToPeak
        o.minimumPeakSpacing = options.minimumSpacing
        o.edgeBoundary = options.edgeBoundary
        o.maximumPeaks = options.maximumPeaks ?? Int.max
        o.tieOrder = .numpy                                    // py4DSTEM's order of equal intensities
        let maxima: [BraggPeak]
        do {
            maxima = try BraggMaxima.find(in: histogram.values, rows: histogram.rows, columns: histogram.columns, options: o)
        } catch let error as CuboBraggError {
            throw bad("the maxima of the Bragg vector map: \(error)")
        }
        guard maxima.count >= 3 else { throw bad("the Bragg vector map has \(maxima.count) maxima; a basis needs at least 3") }
        let origin = histogram.origin
        let choice = try LatticeStrain.chooseBasis(
            maxima: maxima.map { .init(row: $0.row, column: $0.column) }, origin: origin,
            indexOrigin: options.indexOrigin, indexG1: options.indexG1, indexG2: options.indexG2)
        let b = choice.basis
        guard b.g1.row * b.g2.column - b.g1.column * b.g2.row != 0 else { throw LatticeStrainError.singularBasis }
        let relative = maxima.map { LatticeVector(row: $0.row - choice.g0.row, column: $0.column - choice.g0.column) }
        let indexed = try LatticeStrain.index(directions: relative, basis: b)
        let directions = zip(maxima, indexed).map { m, i in
            LatticePeak(row: m.row - origin.row, column: m.column - origin.column, intensity: 1, h: i.h, k: i.k)
        }
        return .init(maxima: maxima, choice: choice, basis: b, directions: directions)
    }

    /// The scan-level strain map: `StrainMap.choose_basis_vectors`, `fit_basis_vectors`, `get_strain`.
    ///
    /// - `peaks`: one list of centred peaks per scan position, row-major (`scanRows * scanColumns` lists).
    /// - `origin`: the diffraction origin in detector pixels (py4DSTEM `calibration.get_origin_mean()`), where the
    ///   centred peaks were measured from; it places the Bragg vector map and picks the lattice origin.
    /// - `directions`: when nil the Bragg vector map of every position is built and the basis chosen with `basisOptions`;
    ///   pass ``StrainDirections/lattice(basis:order:)`` or a result of ``chooseBasisVectors(histogram:options:)`` to skip that.
    ///
    /// A position is valid when it is in the scan mask and at least `minimumPeaks` of its peaks are indexed; the
    /// reference is the median of the valid positions in the reference region. A position whose weights sum to zero
    /// (all intensities 0) is treated as not fitted, where py4DSTEM would store NaN.
    public static func compute(peaks: [[StrainPeak]], scanRows: Int, scanColumns: Int,
                               detectorRows: Int, detectorColumns: Int, origin: LatticeVector,
                               directions suppliedDirections: StrainDirections? = nil,
                               basisOptions: StrainBasisOptions = StrainBasisOptions(),
                               options: StrainMapOptions) throws -> StrainMapResult {
        try checkScan(peaks, scanRows, scanColumns)
        let n = scanRows * scanColumns
        guard options.maximumPeakSpacing.isFinite, options.maximumPeakSpacing >= 0 else {
            throw bad("maximumPeakSpacing must be finite and not negative (\(options.maximumPeakSpacing))")
        }
        guard options.minimumPeaks >= 1 else { throw bad("minimumPeaks must be at least 1 (\(options.minimumPeaks))") }
        guard options.coordinateRotation.isFinite else { throw bad("coordinateRotation must be finite") }
        if let m = options.scanMask { guard m.count == n else { throw bad("the scan mask holds \(m.count) flags for \(n) positions") } }
        if case .region(let r) = options.reference { guard r.count == n else { throw bad("the reference region holds \(r.count) flags for \(n) positions") } }
        if case .supplied(let b) = options.reference {
            guard [b.g1.row, b.g1.column, b.g2.row, b.g2.column].allSatisfy(\.isFinite) else { throw bad("the supplied reference must be finite") }
        }

        var histogram: BraggVectorHistogram?
        let directions: StrainDirections
        if let suppliedDirections {
            directions = suppliedDirections
        } else {
            let h = try Self.histogram(peaks: peaks, scanRows: scanRows, scanColumns: scanColumns,
                                       detectorRows: detectorRows, detectorColumns: detectorColumns, origin: origin)
            histogram = h
            directions = try chooseBasisVectors(histogram: h, options: basisOptions)
        }
        guard !directions.directions.isEmpty else { throw bad("there are no lattice directions to index against") }

        // fit_basis_vectors: index each peak to its nearest direction, fit the lattice.
        let nan = Double.nan
        var valid = [Bool](repeating: false, count: n), indexedCount = [Int](repeating: 0, count: n)
        var x0 = [Double](repeating: nan, count: n), y0 = x0, g1r = x0, g1c = x0, g2r = x0, g2c = x0, err = x0
        var bases = [LatticeBasis?](repeating: nil, count: n)
        for position in 0..<n where options.scanMask?[position] ?? true {
            let measured = peaks[position].map { LatticePeak(row: $0.row, column: $0.column, intensity: $0.intensity) }
            let indexed = try LatticeStrain.index(peaks: measured, directions: directions.directions,
                                                  maximumSpacing: options.maximumPeakSpacing)
            indexedCount[position] = indexed.count
            let fit: LatticeFit?
            switch options.fitMethod {
            case .py4DSTEM:
                do { fit = try LatticeStrain.fit(indexed, minimumPeaks: options.minimumPeaks) }
                catch LatticeStrainError.invalidInput { fit = nil }   // the peaks were validated by the indexing: zero total weight
            case .intensityOnce:
                fit = indexed.count >= options.minimumPeaks ? try LatticeStrain.fitIntensityOnce(indexed) : nil
            }
            guard let fit else { continue }
            valid[position] = true
            x0[position] = fit.origin.row; y0[position] = fit.origin.column
            g1r[position] = fit.basis.g1.row; g1c[position] = fit.basis.g1.column
            g2r[position] = fit.basis.g2.row; g2c[position] = fit.basis.g2.column
            err[position] = fit.residual
            bases[position] = fit.basis
        }

        // get_reference_g1g2
        let reference: LatticeBasis, referenceCount: Int, inlierCount: Int
        switch options.reference {
        case .supplied(let b):
            reference = b; referenceCount = 0; inlierCount = 0
        case .fittedMedian, .region:
            var chosen: [LatticeBasis] = []
            for position in 0..<n {
                guard valid[position], let b = bases[position] else { continue }
                if case .region(let r) = options.reference, !r[position] { continue }
                chosen.append(b)
            }
            if options.referenceMethod == .median {
                reference = try LatticeStrain.reference(chosen); inlierCount = chosen.count
            } else {
                let mad = try LatticeStrain.referenceMADResult(chosen)
                reference = mad.reference; inlierCount = mad.inlierCount
            }
            referenceCount = chosen.count
        }

        // get_strain
        var exx = [Double](repeating: nan, count: n), eyy = exx, exy = exx, theta = exx
        for position in 0..<n {
            guard let local = bases[position] else { continue }
            let s = try LatticeStrain.strain(local: local, reference: reference, coordinateRotation: options.coordinateRotation)
            exx[position] = s.exx; eyy[position] = s.eyy; exy[position] = s.exy; theta[position] = s.theta
        }
        return StrainMapResult(scanRows: scanRows, scanColumns: scanColumns, histogram: histogram, directions: directions,
                               valid: valid, indexedPeakCount: indexedCount, originRow: x0, originColumn: y0,
                               g1Row: g1r, g1Column: g1c, g2Row: g2r, g2Column: g2c, residual: err,
                               reference: reference, referencePositionCount: referenceCount, referenceInlierCount: inlierCount,
                               exx: exx, eyy: eyy, exy: exy, theta: theta, coordinateRotation: options.coordinateRotation)
    }

    /// The scan-position lists of a `CuboBragg` detection result as centred strain peaks: `origin` is subtracted from each
    /// peak (pass the diffraction origin the detection was measured about, in detector pixels).
    public static func centredPeaks(_ vectors: BraggVectors, origin: LatticeVector) -> [[StrainPeak]] {
        vectors.peaks.map { $0.map { StrainPeak(row: $0.row - origin.row, column: $0.column - origin.column, intensity: $0.intensity) } }
    }

    private static func bad(_ m: String) -> StrainMappingError { .invalidInput(m) }

    private static func checkScan(_ peaks: [[StrainPeak]], _ rows: Int, _ columns: Int) throws {
        guard rows > 0, columns > 0, rows <= Int.max / columns else { throw bad("the scan must be at least 1 × 1 (\(rows) × \(columns))") }
        guard peaks.count == rows * columns else { throw bad("\(peaks.count) peak lists for a \(rows) × \(columns) scan") }
    }
}
