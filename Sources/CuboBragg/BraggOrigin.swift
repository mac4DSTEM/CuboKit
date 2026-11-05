//
//  BraggOrigin.swift
//  py4DSTEM's origin from the Bragg peaks: `BraggVectors.measure_origin` (braggvectors/braggvector_methods.py 320 at
//  f050d207) and `BraggVectors.fit_origin` (513, which calls process/calibration/origin.py's `fit_origin`). The direct
//  beam is picked among each position's detected peaks, so the origin follows the unscattered beam where a centre of mass
//  of the whole pattern is pulled by strong reflections. Rows and columns: py4DSTEM's qx is the row.
//

import Foundation
import CuboCore
import CuboVirtual

/// How `BraggOrigin.measure` picks the direct beam among a position's peaks (py4DSTEM's `score_method`).
public enum BraggOriginScore: Sendable, Hashable, CaseIterable {
    /// `"intensity"`: the brightest peak. py4DSTEM's default when no centre guess is given.
    case intensity
    /// `"distance"`: the peak closest to the centre guess. py4DSTEM's default when a guess is given.
    case distance
    /// `"intensity weighted distance"`: the largest intensity / (1 + squared distance to the guess).
    case intensityWeightedDistance
}

/// The per-position direct-beam positions of `BraggOrigin.measure`, row-major `[scanRow][scanColumn]`.
public struct BraggOriginMeasurement: Sendable, Equatable {
    public let scanRows: Int
    public let scanColumns: Int
    /// py4DSTEM's `qx0`: the chosen peak's row, or the guess's row at a position without peaks.
    public let row: [Double]
    /// py4DSTEM's `qy0`: the chosen peak's column, or the guess's column at a position without peaks.
    public let column: [Double]
    /// True where the position had at least one peak (py4DSTEM's intended `origin_meas_mask`).
    public let found: [Bool]
    /// The centre guess: the one given, or the brightest pixel of the smoothed raw Bragg vector map.
    public let guess: BraggDetectorCenter
    /// The score method used (the default resolved).
    public let score: BraggOriginScore

    public init(scanRows: Int, scanColumns: Int, row: [Double], column: [Double], found: [Bool],
                guess: BraggDetectorCenter, score: BraggOriginScore) {
        precondition(row.count == scanRows * scanColumns && column.count == row.count && found.count == row.count,
                     "BraggOriginMeasurement: arrays do not match \(scanRows) × \(scanColumns)")
        self.scanRows = scanRows
        self.scanColumns = scanColumns
        self.row = row
        self.column = column
        self.found = found
        self.guess = guess
        self.score = score
    }
}

/// The smooth origin of `BraggOrigin.fit`, row-major like the measurement.
public struct BraggOriginFit: Sendable, Equatable {
    public let fittedRow: [Double]
    public let fittedColumn: [Double]
    /// Measured minus fitted, at every position (py4DSTEM's `qx0_residuals`, `qy0_residuals`).
    public let residualRow: [Double]
    public let residualColumn: [Double]
    /// The positions that entered the fit: a peak was found there and its row is not 0.
    public let used: [Bool]

    public init(fittedRow: [Double], fittedColumn: [Double], residualRow: [Double], residualColumn: [Double], used: [Bool]) {
        self.fittedRow = fittedRow
        self.fittedColumn = fittedColumn
        self.residualRow = residualRow
        self.residualColumn = residualColumn
        self.used = used
    }
}

public enum BraggOrigin {
    /// py4DSTEM's `BraggVectors.measure_origin(center_guess, score_method, findcenter="max")` on the raw (uncalibrated)
    /// peaks of a `detectorRows × detectorColumns` detector.
    ///
    /// Without `centerGuess` the guess is the brightest pixel (first in raster order) of the raw Bragg vector map
    /// (`histogram(mode="raw")`, bilinear deposition) smoothed by scipy's `gaussian_filter(σ = 10)`, and `score` defaults to
    /// `.intensity`; with a guess it defaults to `.distance`. Each position's origin is its best-scoring peak (the first of
    /// equal scores, as `np.argmax`); a position without peaks gets the guess and `found` false.
    ///
    /// DEVIATION from py4DSTEM f050d207: (1) `measure_origin` hands the map object, not its array, to `gaussian_filter`,
    /// which returns a 0-d array, so its guess is always (0, 0); here it is the map's brightest smoothed pixel, as the
    /// docstring describes. Only a distance score and an empty position read the guess. (2) Its returned mask becomes the
    /// scalar `False` once any position is empty; `found` is per position. (3) `findcenter="CoM"` raises upstream
    /// (`get_CoM` of the map object) and is not offered. (4) A non-finite peak is refused (NumPy's argmax would pick it).
    public static func measure(_ vectors: BraggVectors, detectorRows: Int, detectorColumns: Int,
                               centerGuess: BraggDetectorCenter? = nil,
                               score: BraggOriginScore? = nil) throws -> BraggOriginMeasurement {
        guard detectorRows > 0, detectorColumns > 0 else {
            throw CuboBraggError.shapeMismatch("the detector must be at least 1 × 1 (\(detectorRows) × \(detectorColumns))")
        }
        for list in vectors.peaks {
            for p in list where !(p.row.isFinite && p.column.isFinite && p.intensity.isFinite) {
                throw CuboBraggError.invalidOption("a Bragg peak is not finite (\(p.row), \(p.column), \(p.intensity))")
            }
        }
        let guess: BraggDetectorCenter
        if let centerGuess {
            guard centerGuess.row.isFinite, centerGuess.column.isFinite else {
                throw CuboBraggError.invalidOption("the centre guess must be finite")
            }
            guess = centerGuess
        } else {
            let map = rawHistogram(vectors, rows: detectorRows, columns: detectorColumns)
            let smooth = try GaussianFilter(sigma: 10).apply(map, rows: detectorRows, columns: detectorColumns, edge: .reflect)
            var best = 0
            for i in smooth.indices where smooth[i] > smooth[best] { best = i }
            guess = BraggDetectorCenter(row: Double(best / detectorColumns), column: Double(best % detectorColumns))
        }
        let method = score ?? (centerGuess == nil ? .intensity : .distance)
        let n = vectors.scanRows * vectors.scanColumns
        var row = [Double](repeating: guess.row, count: n)
        var column = [Double](repeating: guess.column, count: n)
        var found = [Bool](repeating: false, count: n)
        for k in 0..<n {
            let list = vectors.peaks[k]
            guard !list.isEmpty else { continue }
            func value(_ p: BraggPeak) -> Double {
                let dr = p.row - guess.row, dc = p.column - guess.column
                switch method {
                case .intensity: return p.intensity
                case .distance: return -(dr * dr + dc * dc)      // argmin of r² is argmax of −r², first of equals kept
                case .intensityWeightedDistance: return p.intensity / (1 + (dr * dr + dc * dc))
                }
            }
            var best = 0, bestValue = value(list[0])
            for i in 1..<list.count {
                let v = value(list[i])
                if v > bestValue { best = i; bestValue = v }
            }
            row[k] = list[best].row
            column[k] = list[best].column
            found[k] = true
        }
        return BraggOriginMeasurement(scanRows: vectors.scanRows, scanColumns: vectors.scanColumns, row: row, column: column,
                                      found: found, guess: guess, score: method)
    }

    /// py4DSTEM's `BraggVectors.fit_origin(fitfunction, robust, robust_steps, robust_thresh)` with its default
    /// `mask_check_data=True` and `mask` = the found positions: the surface is fitted over the positions where a peak was
    /// found and whose measured row is not 0 (`used`), and evaluated everywhere. The fit is `OriginFit.fit`.
    ///
    /// DEVIATION from py4DSTEM f050d207: upstream fits every position its mask lets through, so an empty position enters
    /// with the guess it was given unless that guess's row is 0 (as upstream's always-(0, 0) guess is, so its default path
    /// and this one fit the same positions). Its `"constant"` model ignores the mask and averages every position; here it
    /// averages the `used` ones. The robust loop scores every position, unused ones included, as upstream's `fit_2D`.
    public static func fit(_ measurement: BraggOriginMeasurement, model: OriginFitModel = .plane, robust: Bool = false,
                           robustSteps: Int = 3, robustThreshold: Double = 2) throws -> BraggOriginFit {
        let m = measurement
        let used = m.found.indices.map { m.found[$0] && m.row[$0] != 0 }
        guard used.contains(true) else { throw OriginFitError.underdetermined(model) }
        let fittedRow: [Double], fittedColumn: [Double]
        if model == .constant {
            let r = NumPySum.mean(m.row.indices.filter { used[$0] }.map { m.row[$0] })
            let c = NumPySum.mean(m.column.indices.filter { used[$0] }.map { m.column[$0] })
            fittedRow = Array(repeating: r, count: used.count)
            fittedColumn = Array(repeating: c, count: used.count)
        } else {
            fittedRow = try OriginFit.fit(m.row, rows: m.scanRows, columns: m.scanColumns, model: model, mask: used,
                                          robust: robust, robustSteps: robustSteps, robustThreshold: robustThreshold)
            fittedColumn = try OriginFit.fit(m.column, rows: m.scanRows, columns: m.scanColumns, model: model, mask: used,
                                             robust: robust, robustSteps: robustSteps, robustThreshold: robustThreshold)
        }
        return BraggOriginFit(fittedRow: fittedRow, fittedColumn: fittedColumn,
                              residualRow: m.row.indices.map { m.row[$0] - fittedRow[$0] },
                              residualColumn: m.column.indices.map { m.column[$0] - fittedColumn[$0] },
                              used: used)
    }

    /// mac4DSTEM variant (not in py4DSTEM): a centre guess for `measure(centerGuess:score: .distance)` that finds the direct
    /// beam where a diffracted beam is brighter than it, the case where py4DSTEM's guess (the brightest point of the map) and
    /// its `.intensity` score both pick a reflection. A diffraction pattern is centrosymmetric about the direct beam
    /// (Friedel's law), and the extent of the detected disks is too, so the guess is the candidate about which the map of
    /// peak counts best matches its own point reflection.
    ///
    /// The map: every peak counted once (intensity 1), deposited as `rawHistogram`, smoothed by scipy's
    /// `gaussian_filter(σ = 1, mode="reflect")`. The candidates: its local maxima (equal to the maximum of the 5 × 5 window,
    /// scipy's `maximum_filter(size=5)`) above 5% of the largest value, the `candidates` highest (12 by default; ties in
    /// raster order). The score of a candidate pixel c: Σ min(M(p), M(2c − p)) over the pixels p whose mirror is on the
    /// detector. The best score wins, the first of equal scores. Peak counts, not intensities, so one strong reflection cannot
    /// outweigh the symmetry of all the others. On py4DSTEM's Si/SiGe tutorial cube (128 × 128 detector) it returns the
    /// direct beam at (64, 64) where the brightest point of the map is a first-order reflection at (49, 69).
    ///
    /// nil when no position has a peak.
    public static func symmetryCenterGuess(_ vectors: BraggVectors, detectorRows: Int, detectorColumns: Int,
                                           candidates: Int = 12) throws -> BraggDetectorCenter? {
        guard detectorRows > 0, detectorColumns > 0 else {
            throw CuboBraggError.shapeMismatch("the detector must be at least 1 × 1 (\(detectorRows) × \(detectorColumns))")
        }
        guard candidates > 0 else { throw CuboBraggError.invalidOption("at least one candidate (\(candidates))") }
        let counted = BraggVectors(scanRows: vectors.scanRows, scanColumns: vectors.scanColumns,
                                   peaks: vectors.peaks.map { $0.map { BraggPeak(row: $0.row, column: $0.column, intensity: 1) } })
        let rows = detectorRows, columns = detectorColumns
        let map = try GaussianFilter(sigma: 1).apply(rawHistogram(counted, rows: rows, columns: columns), rows: rows,
                                                     columns: columns, edge: .reflect)
        guard let top = map.max(), top > 0 else { return nil }
        var maxima: [(index: Int, value: Double)] = []
        for r in 0..<rows {
            for c in 0..<columns {
                let v = map[r * columns + c]
                guard v > 0.05 * top else { continue }
                var isMaximum = true
                for rr in max(0, r - 2)...min(rows - 1, r + 2) where isMaximum {
                    for cc in max(0, c - 2)...min(columns - 1, c + 2) where map[rr * columns + cc] > v {
                        isMaximum = false
                        break
                    }
                }
                if isMaximum { maxima.append((r * columns + c, v)) }
            }
        }
        // Stable: equal values keep raster order, as np.argsort(-values, kind="stable").
        let chosen = maxima.enumerated().sorted { $0.element.value != $1.element.value ? $0.element.value > $1.element.value
                                                                                       : $0.offset < $1.offset }
            .prefix(candidates).map(\.element.index)
        var best: (score: Double, index: Int)?
        for index in chosen {
            let r0 = index / columns, c0 = index % columns
            var score = 0.0
            for r in max(0, 2 * r0 - (rows - 1))...min(rows - 1, 2 * r0) {
                for c in max(0, 2 * c0 - (columns - 1))...min(columns - 1, 2 * c0) {
                    score += min(map[r * columns + c], map[(2 * r0 - r) * columns + (2 * c0 - c)])
                }
            }
            if best == nil || score > best!.score { best = (score, index) }
        }
        guard let best else { return nil }
        return BraggDetectorCenter(row: Double(best.index / columns), column: Double(best.index % columns))
    }

    /// py4DSTEM's `BraggVectors.histogram(mode="raw", sampling=1)`: every peak's intensity shared bilinearly between the four
    /// cells around it (floor and ceil of each coordinate); a peak with a neighbour outside the grid is dropped. The four
    /// `np.bincount` sums are added cell by cell, as upstream.
    static func rawHistogram(_ vectors: BraggVectors, rows: Int, columns: Int) -> [Double] {
        let size = rows * columns
        var b00 = [Double](repeating: 0, count: size), b01 = b00, b10 = b00, b11 = b00
        for list in vectors.peaks {
            for p in list {
                guard p.row > -1e9, p.row < 1e9, p.column > -1e9, p.column < 1e9 else { continue }
                let fr = Int(p.row.rounded(.down)), cr = Int(p.row.rounded(.up))
                let fc = Int(p.column.rounded(.down)), cc = Int(p.column.rounded(.up))
                guard fr >= 0, fc >= 0, cr < rows, cc < columns else { continue }
                let dr = p.row - Double(fr), dc = p.column - Double(fc)
                b00[fr * columns + fc] += p.intensity * (1.0 - dr) * (1.0 - dc)
                b01[fr * columns + cc] += p.intensity * (1.0 - dr) * dc
                b10[cr * columns + fc] += p.intensity * dr * (1.0 - dc)
                b11[cr * columns + cc] += p.intensity * dr * dc
            }
        }
        return (0..<size).map { b00[$0] + b01[$0] + b10[$0] + b11[$0] }
    }
}
