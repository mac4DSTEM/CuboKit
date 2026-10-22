//
//  PolarCube.swift
//  The polar-elliptical transform of every diffraction pattern (task 17): py4DSTEM's `DataCube.polar`, a `PolarDatacube`
//  (process/polar/polar_datacube.py at f050d207), in Double on the CPU.
//
//  A pattern is binned (not interpolated) onto `annularCount x radialCount` cells: each detector pixel spreads its value over
//  the four neighbouring (radius, angle) cells with bilinear weights, a Gaussian along the angle whose width follows the
//  arc length smooths the cells of large radius, the same transform of the mask divides the result, and cells whose
//  normalisation is below `maskThreshold` are invalid. The elliptical coordinates come from the stored ``Calibration/ellipse``
//  (a, b, theta) and the origin at the scan position; no ellipse (or `elliptical: false`) is a circle.
//
//  Differences from py4DSTEM, on purpose or by consequence of the language:
//    - py4DSTEM's x is the ROW and y the COLUMN; theta is the angle of the major axis from the +row axis towards the +column
//      axis, exactly the numbers py4DSTEM stores (no conversion is needed between `Calibration.ellipse` and this file).
//    - The data and the mask are binned in one pass over the pixels (py4DSTEM transforms them one after the other with the same
//      coordinates); the sums are formed in the same pixel order and with the same product order, so the cells are the same.
//    - A scan position's ellipse is read at that position (a per-position field works; py4DSTEM's `get_ellipse()` takes the
//      stored value whole and fails on an array).
//    - The radius of a cell is on the scale of the MINOR semi-axis: a ring of the ellipse sits at `b`, not `a` (py4DSTEM's
//      `rr = (b / a) * hypot(...)`). ``RadialBackground`` uses the scale of the major axis, as py4DSTEM does; see its notes.
//

import CuboCore
import Foundation

/// The sampling of a ``PolarCube`` (`PolarDatacube`'s arguments).
public struct PolarSampling: Sendable, Equatable {
    /// `qmin`: the smallest radius, in detector pixels.
    public var minimumRadius: Double = 0
    /// `qmax`; `nil` is `min(detector extents) / sqrt(2)`.
    public var maximumRadius: Double? = nil
    /// `qstep`: the width of a radial bin, in pixels.
    public var radialStep: Double = 1
    /// `n_annular`: the number of angular bins over 2 pi (over pi with ``twoFoldSymmetry``).
    public var annularCount: Int = 180
    /// `qscale`: multiplies each radial bin by `(q / q_last) ** power`; `nil` leaves the values as they are.
    public var radialPower: Double? = nil
    /// `ellipse`: `false` forces a circular transform even where the calibration has an ellipse.
    public var elliptical: Bool = true
    /// `two_fold_symmetry`: the angle is taken modulo pi (Friedel pairs share a cell).
    public var twoFoldSymmetry: Bool = false
    /// `mask_thresh`: cells whose normalisation, scaled by the angular density, is below this are invalid.
    public var maskThreshold: Double = 0.1

    public init(minimumRadius: Double = 0, maximumRadius: Double? = nil, radialStep: Double = 1, annularCount: Int = 180,
                radialPower: Double? = nil, elliptical: Bool = true, twoFoldSymmetry: Bool = false, maskThreshold: Double = 0.1) {
        self.minimumRadius = minimumRadius; self.maximumRadius = maximumRadius; self.radialStep = radialStep
        self.annularCount = annularCount; self.radialPower = radialPower; self.elliptical = elliptical
        self.twoFoldSymmetry = twoFoldSymmetry; self.maskThreshold = maskThreshold
    }
}

/// One pattern in polar coordinates: `annularCount x radialCount`, angle first (py4DSTEM's `polar_shape`), row-major.
public struct PolarPattern: Sendable, Equatable {
    public let annularCount: Int
    public let radialCount: Int
    /// The normalised, (optionally) radially scaled values; NaN in invalid cells.
    public let values: [Double]
    /// `false` where py4DSTEM's masked array is masked.
    public let valid: [Bool]
    /// The transform of the detector mask (`ans_norm`): how much of each cell the unmasked pixels cover.
    public let coverage: [Double]

    public subscript(annular: Int, radial: Int) -> Double { values[annular * radialCount + radial] }
}

/// The polar transform of a whole cube: `scanCount` patterns of ``PolarPattern`` shape, scan row-major.
public struct PolarCubeData: Sendable, Equatable {
    public let scanRows: Int, scanColumns: Int
    public let annularCount: Int, radialCount: Int
    public let values: [Double]
    public let valid: [Bool]

    /// The polar pattern at a scan position.
    public func pattern(row: Int, column: Int) -> (values: [Double], valid: [Bool]) {
        let n = annularCount * radialCount, k = (row * scanColumns + column) * n
        return (Array(values[k..<(k + n)]), Array(valid[k..<(k + n)]))
    }
}

/// The polar-elliptical sampling of a detector, and the transform of patterns onto it.
public struct PolarCube: Sendable {
    public let detectorRows: Int, detectorColumns: Int
    public let sampling: PolarSampling
    /// `radial_bins`: `np.arange(qmin, qmax, qstep)`, in detector pixels (the left edges of the bins).
    public let radialBins: [Double]
    /// `annular_bins`: `linspace(0, range, n_annular, endpoint=False)`, radians.
    public let annularBins: [Double]
    /// `annular_step`.
    public let annularStep: Double
    /// The detector pixel size that scales ``radialBins`` into `qq` (py4DSTEM's `Q_pixel_size`).
    public let detectorPixelSize: Double
    /// The mask applied to every pattern (`true` = keep), `detectorRows * detectorColumns`, or `nil`.
    public let detectorMask: [Bool]?

    private let annularRange: Double
    private let binDensity: [Double]       // _annular_bin_step: 1 / (annular_step * (radial_bins + qstep / 2))
    private let kdeSigma: [Double]         // _sigma_KDE
    private let radialScale: [Double]?     // _qscale_ar

    public var annularCount: Int { annularBins.count }
    public var radialCount: Int { radialBins.count }
    /// `qq`: the radial bins in the calibrated unit (bins times the detector pixel size).
    public var radialBinsCalibrated: [Double] { radialBins.map { $0 * detectorPixelSize } }

    /// Throws ``CuboVirtualError/invalidInput(_:)`` for a sampling py4DSTEM would fail on (no radial bin, fewer than two angular
    /// bins, a non-positive step, a mask of the wrong size).
    public init(detectorRows: Int, detectorColumns: Int, sampling: PolarSampling = PolarSampling(), detectorMask: [Bool]? = nil,
                detectorPixelSize: Double = 1) throws {
        guard detectorRows > 0, detectorColumns > 0 else { throw CuboVirtualError.invalidInput("a polar cube needs a detector") }
        guard sampling.radialStep.isFinite, sampling.radialStep > 0 else {
            throw CuboVirtualError.invalidInput("the radial step must be positive, not \(sampling.radialStep)")
        }
        guard sampling.annularCount >= 2 else {
            throw CuboVirtualError.invalidInput("a polar cube needs at least two angular bins, not \(sampling.annularCount)")
        }
        guard sampling.maskThreshold.isFinite, sampling.minimumRadius.isFinite else {
            throw CuboVirtualError.invalidInput("the polar sampling needs finite numbers")
        }
        if let m = detectorMask, m.count != detectorRows * detectorColumns {
            throw CuboVirtualError.maskLengthMismatch(expected: detectorRows * detectorColumns, received: m.count)
        }
        let qmax = sampling.maximumRadius ?? Double(min(detectorRows, detectorColumns)) / 2.0.squareRoot()
        guard qmax.isFinite else { throw CuboVirtualError.invalidInput("the maximum radius must be finite") }
        let bins = SciPyKernels.arange(sampling.minimumRadius, qmax, sampling.radialStep)
        guard !bins.isEmpty else { throw CuboVirtualError.invalidInput("no radial bin between \(sampling.minimumRadius) and \(qmax)") }
        guard detectorPixelSize.isFinite, detectorPixelSize > 0 else { throw CuboVirtualError.invalidPixelSize(detectorPixelSize) }
        self.detectorRows = detectorRows; self.detectorColumns = detectorColumns
        self.sampling = sampling; self.detectorMask = detectorMask; self.detectorPixelSize = detectorPixelSize
        radialBins = bins
        annularRange = sampling.twoFoldSymmetry ? Double.pi : 2 * Double.pi
        // np.linspace(0, range, n, endpoint=False): step = range / n, values i * step
        let n = sampling.annularCount
        let step = annularRange / Double(n)
        annularBins = (0..<n).map { Double($0) * step }
        let angularStep = annularBins[1] - annularBins[0]
        annularStep = angularStep
        let density = bins.map { 1 / (angularStep * ($0 + sampling.radialStep * 0.5)) }
        binDensity = density
        kdeSigma = density.map { $0 * 0.5 }
        if let p = sampling.radialPower {
            let qq = bins.map { $0 * detectorPixelSize }
            let last = qq[qq.count - 1]
            radialScale = qq.map { pow($0 / last, p) }
        } else {
            radialScale = nil
        }
    }

    // MARK: the geometry of one origin and ellipse

    /// Where every detector pixel lands: the cell indices and bilinear fractions, for one origin and ellipse. Reused for every
    /// pattern that shares them (a constant origin and ellipse over the scan), which changes nothing in the numbers.
    public struct Geometry: Sendable {
        fileprivate let radialFloor: [Int32]     // floor of the radial index; Int32.min when the pixel is not sampled at all
        fileprivate let angularFloor: [Int32]    // floor(tt / step) mod n_annular
        fileprivate let angularNext: [Int32]     // (floor + 1) mod n_annular
        fileprivate let dr: [Double], dt: [Double]
    }

    /// The geometry of `origin` (row, column, detector pixels) and `ellipse` (nil is a circle; ignored when the sampling is not
    /// elliptical).
    public func geometry(origin: (row: Double, column: Double), ellipse: (a: Double, b: Double, theta: Double)?) -> Geometry {
        let count = detectorRows * detectorColumns
        var rf = [Int32](repeating: Int32.min, count: count)
        var tf = [Int32](repeating: 0, count: count), tn = tf
        var drs = [Double](repeating: 0, count: count), dts = drs
        let n = annularBins.count, nr = radialBins.count
        let e = sampling.elliptical ? ellipse : nil
        let cosT = e.map { cos($0.theta) } ?? 1, sinT = e.map { sin($0.theta) } ?? 0
        for r in 0..<detectorRows {
            for c in 0..<detectorColumns {
                let x = Double(r) - origin.row, y = Double(c) - origin.column
                let rr: Double, tt: Double
                if let e {
                    let xc = x * cosT + y * sinT
                    let yc = (y * cosT - x * sinT) * (e.a / e.b)
                    rr = (e.b / e.a) * hypot(xc, yc)
                    tt = SciPyKernels.mod(atan2(yc, xc) + e.theta, annularRange)
                } else {
                    rr = (x * x + y * y).squareRoot()
                    tt = SciPyKernels.mod(atan2(y, x), annularRange)
                }
                let rInd = (rr - radialBins[0]) / sampling.radialStep
                let tInd = tt / annularStep
                guard rInd.isFinite, tInd.isFinite, abs(rInd) < 1e9, abs(tInd) < 1e9 else { continue }
                let rFloor = rInd.rounded(.down), tFloor = tInd.rounded(.down)
                let k = r * detectorColumns + c
                // the angular index is reduced modulo n (np.mod: always non-negative)
                let ti = Int(tFloor)
                let tm = ((ti % n) + n) % n
                let tp = (((ti + 1) % n) + n) % n
                let ri = Int(rFloor)
                guard ri >= -1, ri < nr else { continue }
                rf[k] = Int32(ri); tf[k] = Int32(tm); tn[k] = Int32(tp)
                drs[k] = rInd - rFloor; dts[k] = tInd - tFloor
            }
        }
        return Geometry(radialFloor: rf, angularFloor: tf, angularNext: tn, dr: drs, dt: dts)
    }

    /// The four `np.bincount` passes of py4DSTEM's `_transform_array` for one set of weights, then the Gaussian along the angle.
    private func bin(_ weights: [Double], _ g: Geometry) -> [Double] {
        let n = annularBins.count, nr = radialBins.count, size = n * nr
        var im = [Double](repeating: 0, count: size)
        var pass = [Double](repeating: 0, count: size)
        for k in 0..<weights.count {                       // pass 1: (r, t) with (1 - dr)(1 - dt)
            let rf = Int(g.radialFloor[k])
            guard rf != Int(Int32.min), rf >= 0, rf < nr else { continue }
            pass[rf + Int(g.angularFloor[k]) * nr] += weights[k] * (1 - g.dr[k]) * (1 - g.dt[k])
        }
        im = pass
        for i in pass.indices { pass[i] = 0 }
        for k in 0..<weights.count {                       // pass 2: (r, t + 1) with (1 - dr) dt
            let rf = Int(g.radialFloor[k])
            guard rf != Int(Int32.min), rf >= 0, rf < nr else { continue }
            pass[rf + Int(g.angularNext[k]) * nr] += weights[k] * (1 - g.dr[k]) * g.dt[k]
        }
        for i in 0..<size { im[i] += pass[i]; pass[i] = 0 }
        for k in 0..<weights.count {                       // pass 3: (r + 1, t) with dr (1 - dt)
            let rf = Int(g.radialFloor[k])
            guard rf != Int(Int32.min), rf >= -1, rf < nr - 1 else { continue }
            pass[rf + 1 + Int(g.angularFloor[k]) * nr] += weights[k] * g.dr[k] * (1 - g.dt[k])
        }
        for i in 0..<size { im[i] += pass[i]; pass[i] = 0 }
        for k in 0..<weights.count {                       // pass 4: (r + 1, t + 1) with dr dt
            let rf = Int(g.radialFloor[k])
            guard rf != Int(Int32.min), rf >= -1, rf < nr - 1 else { continue }
            pass[rf + 1 + Int(g.angularNext[k]) * nr] += weights[k] * g.dr[k] * g.dt[k]
        }
        for i in 0..<size { im[i] += pass[i] }
        // the kernel density estimate along the angle: a periodic Gaussian where the next bin would see at least 5% of the peak
        for a0 in 0..<nr where kdeSigma[a0] > 0.1669 {
            let column = (0..<n).map { im[$0 * nr + a0] }
            let f = SciPyKernels.gaussianFilter1D(column, sigma: kdeSigma[a0], boundary: .wrap)
            for t in 0..<n { im[t * nr + a0] = f[t] }
        }
        return im
    }

    /// The polar transform of one pattern with a prepared geometry (`transform(..., returnval="masked")`).
    public func transform(pattern: [Float], geometry g: Geometry) throws -> PolarPattern {
        guard pattern.count == detectorRows * detectorColumns else {
            throw CuboVirtualError.maskLengthMismatch(expected: detectorRows * detectorColumns, received: pattern.count)
        }
        var weights = [Double](repeating: 1, count: pattern.count)
        var data = [Double](repeating: 0, count: pattern.count)
        for k in pattern.indices {
            let keep = detectorMask?[k] ?? true
            weights[k] = keep ? 1 : 0
            data[k] = Double(pattern[k]) * weights[k]
        }
        let ans = bin(data, g), norm = bin(weights, g)
        let n = annularBins.count, nr = radialBins.count
        var values = [Double](repeating: .nan, count: n * nr)
        var valid = [Bool](repeating: false, count: n * nr)
        for t in 0..<n { for r in 0..<nr {
            let i = t * nr + r
            let ok = !(norm[i] * binDensity[r] < sampling.maskThreshold)
            valid[i] = ok
            if ok {
                var v = ans[i] / norm[i]
                if let s = radialScale { v *= s[r] }
                values[i] = v
            }
        } }
        return PolarPattern(annularCount: n, radialCount: nr, values: values, valid: valid, coverage: norm)
    }

    /// The polar transform of one pattern about `origin` with `ellipse` (nil is a circle).
    public func transform(pattern: [Float], origin: (row: Double, column: Double),
                          ellipse: (a: Double, b: Double, theta: Double)?) throws -> PolarPattern {
        try transform(pattern: pattern, geometry: geometry(origin: origin, ellipse: ellipse))
    }

    // MARK: from a cube and its calibration

    /// The origin and ellipse of `calibration` at a scan position (py4DSTEM `get_origin(rx, ry)` and `get_ellipse()`).
    /// Throws ``CalibrationError/notSet(_:)`` without an origin.
    public static func placement(of calibration: Calibration, scanRow: Int, scanColumn: Int) throws
        -> (origin: (row: Double, column: Double), ellipse: (a: Double, b: Double, theta: Double)?) {
        guard let origin = try calibration.origin(row: scanRow, column: scanColumn) else { throw CalibrationError.notSet("origin") }
        guard let e = calibration.ellipse else { return (origin, nil) }
        return (origin, (try e.a.value(row: scanRow, column: scanColumn), try e.b.value(row: scanRow, column: scanColumn),
                         try e.theta.value(row: scanRow, column: scanColumn)))
    }

    /// `polar.data[rx, ry]`: the pattern at a scan position, transformed about its calibrated origin and the calibrated ellipse.
    public func pattern(of source: some CubeSource, scanRow: Int, scanColumn: Int, calibration: Calibration) async throws -> PolarPattern {
        try requireDetector(of: source.shape)
        let p = try await source.readPattern(row: scanRow, column: scanColumn)
        let place = try Self.placement(of: calibration, scanRow: scanRow, scanColumn: scanColumn)
        return try transform(pattern: p.values, origin: place.origin, ellipse: place.ellipse)
    }

    /// Every pattern of `source` in polar coordinates, tile by tile and positions in parallel. A constant origin and ellipse
    /// share one geometry. `isCancelled` is asked before each tile and throws `CancellationError`; `progress` receives the
    /// fraction of scan rows done.
    public func transform(source: some CubeSource, calibration: Calibration, tiling: ScanTiling = .automatic,
                          isCancelled: (@Sendable () -> Bool)? = nil,
                          progress: (@Sendable (Double) -> Void)? = nil) async throws -> PolarCubeData {
        let shape = source.shape
        try requireDetector(of: shape)
        let size = annularCount * radialCount
        let constantPlacement: Bool = {
            guard calibration.originRow?.isConstant == true, calibration.originColumn?.isConstant == true else { return false }
            guard let e = calibration.ellipse else { return true }
            return e.a.isConstant && e.b.isConstant && e.theta.isConstant
        }()
        let shared: Geometry? = constantPlacement
            ? try { let p = try Self.placement(of: calibration, scanRow: 0, scanColumn: 0); return geometry(origin: p.origin, ellipse: p.ellipse) }()
            : nil
        var values = [Double](); values.reserveCapacity(shape.scanCount * size)
        var valid = [Bool](); valid.reserveCapacity(shape.scanCount * size)
        let dc = shape.detectorCount
        for range in try tiling.ranges(for: shape) {
            if isCancelled?() == true { throw CancellationError() }
            try Task.checkCancellation()
            let tile = try await source.readScanTile(rows: range)
            let first = range.lowerBound
            let places: [(origin: (row: Double, column: Double), ellipse: (a: Double, b: Double, theta: Double)?)] = shared != nil ? [] :
                try (0..<tile.positionCount).map { k in
                    try Self.placement(of: calibration, scanRow: first + k / shape.scanColumns, scanColumn: k % shape.scanColumns)
                }
            let tv = tile.values
            let found = parallelMap(tile.positionCount) { (k: Int) -> PolarPattern? in
                let g = shared ?? self.geometry(origin: places[k].origin, ellipse: places[k].ellipse)
                return try? self.transform(pattern: Array(tv[(k * dc)..<((k + 1) * dc)]), geometry: g)
            }
            for f in found {
                guard let f else { throw CuboVirtualError.invalidInput("a pattern could not be transformed") }
                values.append(contentsOf: f.values); valid.append(contentsOf: f.valid)
            }
            progress?(Double(range.upperBound) / Double(shape.scanRows))
        }
        return PolarCubeData(scanRows: shape.scanRows, scanColumns: shape.scanColumns, annularCount: annularCount,
                             radialCount: radialCount, values: values, valid: valid)
    }

    func requireDetector(of shape: CubeShape) throws {
        guard shape.detectorRows == detectorRows, shape.detectorColumns == detectorColumns else {
            throw CuboVirtualError.invalidInput("the polar cube samples a \(detectorRows) x \(detectorColumns) detector, the cube has \(shape.detectorRows) x \(shape.detectorColumns)")
        }
    }
}
