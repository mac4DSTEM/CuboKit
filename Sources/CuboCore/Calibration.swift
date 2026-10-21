//
//  Calibration.swift
//  The calibration model of a cube: pixel sizes and units of scan and detector, the origin, the diffraction ellipse, the
//  Q-R rotation and flip, and the probe numbers, with the coordinate vectors and grids built from them.
//
//  Reference: py4DSTEM `data.calibration.Calibration` and `DataCube.calibrate` (qxx, qyy, rxx, ryy, the raw grids, qxxs and
//  qyys). py4DSTEM's x is this kit's ROW and y its COLUMN. A value type, so the registration of targets that py4DSTEM
//  re-calibrates on every setter (`register_target`, `attach`) has no counterpart: a caller derives the axes it needs from the
//  value. Differences that are on purpose:
//    - py4DSTEM's grids come from `np.meshgrid(dim_x, dim_y)` and are shaped (Ny, Nx), the transpose of the cube's detector
//      plane (the four `*_raw` grids are `meshgrid(..., indexing="ij")`-like and are not transposed). The grids here are `rows x columns`, entry [row][column] (the transpose of py4DSTEM's array, values equal).
//    - py4DSTEM keeps `qx0`, `qy0` as a number or as an R-shaped array; a ``ScanField`` is both.
//    - Binning, thinning, padding and resampling update the pixel sizes as py4DSTEM does. py4DSTEM leaves the origin (in
//      detector pixels) untouched by those; so does ``Calibration``, which does not move it either (`cropped(...)` below is
//      the one place the origin is shifted: py4DSTEM's `crop_Q` does not, a caller that wants it shifts it explicitly).
//

import Foundation

/// A number that is either the same everywhere in the scan or one per scan position (py4DSTEM's scalar-or-R-shaped parameter).
public struct ScanField: Sendable, Equatable, Codable {
    /// `nil` for a single number; otherwise the scan the values belong to.
    public let scanRows: Int?, scanColumns: Int?
    /// One value, or `scanRows * scanColumns` values, row-major.
    public let values: [Double]

    private enum CodingKeys: String, CodingKey { case scanRows, scanColumns, values }

    // Decoded through the same checks as a constructed field, so a corrupt file cannot make a later lookup trap.
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let r = try c.decodeIfPresent(Int.self, forKey: .scanRows), k = try c.decodeIfPresent(Int.self, forKey: .scanColumns)
        let v = try c.decode([Double].self, forKey: .values)
        switch (r, k) {
        case (nil, nil):
            guard v.count == 1 else { throw CalibrationError.invalidField("a constant field holds \(v.count) values") }
            self.init(constant: v[0])
        case (let r?, let k?): try self.init(scanRows: r, scanColumns: k, values: v)
        default: throw CalibrationError.invalidField("a field needs both scan extents or neither")
        }
    }

    /// The same value at every scan position.
    public init(constant value: Double) { scanRows = nil; scanColumns = nil; values = [value] }

    /// One value per scan position, row-major; the count must be `scanRows * scanColumns`.
    public init(scanRows: Int, scanColumns: Int, values: [Double]) throws {
        guard scanRows > 0, scanColumns > 0, values.count == scanRows * scanColumns else {
            throw CalibrationError.invalidField("\(values.count) values for a \(scanRows) x \(scanColumns) scan")
        }
        self.scanRows = scanRows; self.scanColumns = scanColumns; self.values = values
    }

    /// Whether this is one number.
    public var isConstant: Bool { scanRows == nil }

    /// The value at a scan position (py4DSTEM `get_*(rx, ry)`; a constant returns itself for any position).
    public func value(row: Int, column: Int) throws -> Double {
        guard let rows = scanRows, let columns = scanColumns else { return values[0] }
        guard (0..<rows).contains(row), (0..<columns).contains(column) else {
            throw CalibrationError.outOfRange("scan position (\(row), \(column))")
        }
        return values[row * columns + column]
    }

    /// The mean (`np.mean`: NumPy sums pairwise; the kit sums in Double, the same to rounding).
    public var mean: Double { values.reduce(0, +) / Double(values.count) }

    /// The value minus the mean, at every position (py4DSTEM's `qx0_shift`).
    public var shift: ScanField {
        let m = mean
        if let r = scanRows, let c = scanColumns { return ScanField(unchecked: r, c, values.map { $0 - m }) }
        return ScanField(constant: values[0] - m)
    }

    private init(unchecked r: Int, _ c: Int, _ v: [Double]) { scanRows = r; scanColumns = c; values = v }
}

/// The units of the diffraction axes; py4DSTEM accepts these three for Q (`set_Q_pixel_units`) and any text for R.
public enum DetectorUnits: String, Sendable, Codable, Equatable {
    case pixels = "pixels"
    case inverseAngstroms = "A^-1"
    case milliradians = "mrad"
}

/// A `rows x columns` plane of numbers, row-major.
public struct CoordinateGrid: Sendable, Equatable {
    public let rows: Int, columns: Int
    public let values: [Double]
    public subscript(row: Int, column: Int) -> Double { values[row * columns + column] }
}

/// What a calibration refuses.
public enum CalibrationError: Error, Equatable, Sendable, LocalizedError {
    case invalidField(String)
    case outOfRange(String)
    /// A quantity the request needs has not been set (py4DSTEM raises for `qxxs` without an origin shift).
    case notSet(String)
    case invalidFactor(String)

    public var errorDescription: String? {
        switch self {
        case .invalidField(let d): return "The calibration field is invalid: \(d)."
        case .outOfRange(let d): return "\(d) is outside the calibration."
        case .notSet(let d): return "The calibration has no \(d)."
        case .invalidFactor(let d): return "The factor is invalid: \(d)."
        }
    }
}

/// The ellipse of the diffraction pattern, py4DSTEM's `a`, `b`, `theta` (semi-axes in pixels, angle in radians).
public struct EllipseCalibration: Sendable, Equatable, Codable {
    public var a: ScanField, b: ScanField, theta: ScanField
    public init(a: ScanField, b: ScanField, theta: ScanField) { self.a = a; self.b = b; self.theta = theta }
    public init(a: Double, b: Double, theta: Double) {
        self.init(a: ScanField(constant: a), b: ScanField(constant: b), theta: ScanField(constant: theta))
    }
}

/// The calibration of a cube. Defaults are py4DSTEM's: pixel sizes 1, units "pixels" (Q) and "pixels" (R), no flip.
public struct Calibration: Sendable, Equatable, Codable {
    /// Q pixel size and units (`Q_pixel_size`, `Q_pixel_units`).
    public var detectorPixelSize: Double = 1
    public var detectorUnits: DetectorUnits = .pixels
    /// R pixel size and units (`R_pixel_size`, `R_pixel_units`); py4DSTEM takes any text for R.
    public var scanPixelSize: Double = 1
    public var scanUnits: String = "pixels"

    /// The origin per scan position, in detector pixels (`qx0`, `qy0`). Set with ``setOrigin(row:column:)``.
    public private(set) var originRow: ScanField?
    public private(set) var originColumn: ScanField?
    /// The mean origin (`qx0_mean`, `qy0_mean`); set by ``setOrigin(row:column:)`` and by ``setProbeParameters(semiangle:centerRow:centerColumn:)``.
    public private(set) var originMeanRow: Double?
    public private(set) var originMeanColumn: Double?
    /// The measured origin and its mask (`qx0_meas`, `qy0_meas`, `origin_meas_mask`).
    public var measuredOriginRow: ScanField?, measuredOriginColumn: ScanField?
    public var measuredOriginMask: [Bool]?

    /// `a`, `b`, `theta`.
    public var ellipse: EllipseCalibration?

    /// Q-R rotation in radians and the flip (`QR_rotation`, `QR_flip`).
    public private(set) var qrRotationRadians: Double?
    public var qrFlip: Bool = false

    /// `probe_semiangle`, `probe_center` (row, column), `convergence_semiangle_pixels`, `convergence_semiangle_mrad`.
    public var probeSemiangle: Double?
    public var probeCenter: (row: Double, column: Double)?
    public var convergenceSemiangleInPixels: Double?
    public var convergenceSemiangleInMilliradians: Double?

    public init() {}

    public static func == (a: Calibration, b: Calibration) -> Bool {
        a.detectorPixelSize == b.detectorPixelSize && a.detectorUnits == b.detectorUnits && a.scanPixelSize == b.scanPixelSize
            && a.scanUnits == b.scanUnits && a.originRow == b.originRow && a.originColumn == b.originColumn
            && a.originMeanRow == b.originMeanRow && a.originMeanColumn == b.originMeanColumn
            && a.measuredOriginRow == b.measuredOriginRow && a.measuredOriginColumn == b.measuredOriginColumn
            && a.measuredOriginMask == b.measuredOriginMask && a.ellipse == b.ellipse && a.qrRotationRadians == b.qrRotationRadians
            && a.qrFlip == b.qrFlip && a.probeSemiangle == b.probeSemiangle && a.probeCenter?.row == b.probeCenter?.row
            && a.probeCenter?.column == b.probeCenter?.column && a.convergenceSemiangleInPixels == b.convergenceSemiangleInPixels
            && a.convergenceSemiangleInMilliradians == b.convergenceSemiangleInMilliradians
    }

    private enum CodingKeys: String, CodingKey {
        case detectorPixelSize, detectorUnits, scanPixelSize, scanUnits, originRow, originColumn, originMeanRow, originMeanColumn,
             measuredOriginRow, measuredOriginColumn, measuredOriginMask, ellipse, qrRotationRadians, qrFlip, probeSemiangle,
             probeCenterRow, probeCenterColumn, convergenceSemiangleInPixels, convergenceSemiangleInMilliradians
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        detectorPixelSize = try c.decode(Double.self, forKey: .detectorPixelSize)
        detectorUnits = try c.decode(DetectorUnits.self, forKey: .detectorUnits)
        scanPixelSize = try c.decode(Double.self, forKey: .scanPixelSize)
        scanUnits = try c.decode(String.self, forKey: .scanUnits)
        originRow = try c.decodeIfPresent(ScanField.self, forKey: .originRow)
        originColumn = try c.decodeIfPresent(ScanField.self, forKey: .originColumn)
        originMeanRow = try c.decodeIfPresent(Double.self, forKey: .originMeanRow)
        originMeanColumn = try c.decodeIfPresent(Double.self, forKey: .originMeanColumn)
        measuredOriginRow = try c.decodeIfPresent(ScanField.self, forKey: .measuredOriginRow)
        measuredOriginColumn = try c.decodeIfPresent(ScanField.self, forKey: .measuredOriginColumn)
        measuredOriginMask = try c.decodeIfPresent([Bool].self, forKey: .measuredOriginMask)
        ellipse = try c.decodeIfPresent(EllipseCalibration.self, forKey: .ellipse)
        qrRotationRadians = try c.decodeIfPresent(Double.self, forKey: .qrRotationRadians)
        qrFlip = try c.decode(Bool.self, forKey: .qrFlip)
        probeSemiangle = try c.decodeIfPresent(Double.self, forKey: .probeSemiangle)
        if let r = try c.decodeIfPresent(Double.self, forKey: .probeCenterRow),
           let k = try c.decodeIfPresent(Double.self, forKey: .probeCenterColumn) { probeCenter = (r, k) }
        convergenceSemiangleInPixels = try c.decodeIfPresent(Double.self, forKey: .convergenceSemiangleInPixels)
        convergenceSemiangleInMilliradians = try c.decodeIfPresent(Double.self, forKey: .convergenceSemiangleInMilliradians)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(detectorPixelSize, forKey: .detectorPixelSize); try c.encode(detectorUnits, forKey: .detectorUnits)
        try c.encode(scanPixelSize, forKey: .scanPixelSize); try c.encode(scanUnits, forKey: .scanUnits)
        try c.encodeIfPresent(originRow, forKey: .originRow); try c.encodeIfPresent(originColumn, forKey: .originColumn)
        try c.encodeIfPresent(originMeanRow, forKey: .originMeanRow); try c.encodeIfPresent(originMeanColumn, forKey: .originMeanColumn)
        try c.encodeIfPresent(measuredOriginRow, forKey: .measuredOriginRow)
        try c.encodeIfPresent(measuredOriginColumn, forKey: .measuredOriginColumn)
        try c.encodeIfPresent(measuredOriginMask, forKey: .measuredOriginMask)
        try c.encodeIfPresent(ellipse, forKey: .ellipse); try c.encodeIfPresent(qrRotationRadians, forKey: .qrRotationRadians)
        try c.encode(qrFlip, forKey: .qrFlip); try c.encodeIfPresent(probeSemiangle, forKey: .probeSemiangle)
        try c.encodeIfPresent(probeCenter?.row, forKey: .probeCenterRow)
        try c.encodeIfPresent(probeCenter?.column, forKey: .probeCenterColumn)
        try c.encodeIfPresent(convergenceSemiangleInPixels, forKey: .convergenceSemiangleInPixels)
        try c.encodeIfPresent(convergenceSemiangleInMilliradians, forKey: .convergenceSemiangleInMilliradians)
    }

    // MARK: origin

    /// py4DSTEM `set_origin((qx0, qy0))`: stores both and their means (`qx0_mean`, `qy0_mean`). The two fields must describe
    /// the same scan (or both be constant).
    public mutating func setOrigin(row: ScanField, column: ScanField) throws {
        guard row.scanRows == column.scanRows, row.scanColumns == column.scanColumns else {
            throw CalibrationError.invalidField("the origin's row and column fields describe different scans")
        }
        originRow = row; originColumn = column
        originMeanRow = row.mean; originMeanColumn = column.mean
    }

    /// The same origin at every scan position.
    public mutating func setOrigin(row: Double, column: Double) {
        originRow = ScanField(constant: row); originColumn = ScanField(constant: column)
        originMeanRow = row; originMeanColumn = column
    }

    /// `get_origin(rx, ry)`; `nil` when no origin is set.
    public func origin(row: Int, column: Int) throws -> (row: Double, column: Double)? {
        guard let r = originRow, let c = originColumn else { return nil }
        return (try r.value(row: row, column: column), try c.value(row: row, column: column))
    }

    /// `get_origin_mean`; `nil` when neither mean is set.
    public var originMean: (row: Double, column: Double)? {
        guard let r = originMeanRow, let c = originMeanColumn else { return nil }
        return (r, c)
    }

    /// `get_origin_shift(rx, ry)`: the origin minus its mean at one scan position; `nil` when no origin is set.
    public func originShift(row: Int, column: Int) throws -> (row: Double, column: Double)? {
        guard let r = originRow, let c = originColumn else { return nil }
        return (try r.value(row: row, column: column) - r.mean, try c.value(row: row, column: column) - c.mean)
    }

    // MARK: rotation, flip, probe

    /// `set_QR_rotation`: radians.
    public mutating func setQRRotation(radians: Double) { qrRotationRadians = radians }
    /// `set_QR_rotation_degrees`.
    public mutating func setQRRotation(degrees: Double) { qrRotationRadians = degrees * .pi / 180 }
    /// `QR_rotation_degrees`.
    public var qrRotationDegrees: Double? { qrRotationRadians.map { $0 * 180 / .pi } }
    /// `set_QR_rotflip_degrees((degrees, flip))`.
    public mutating func setQRRotationAndFlip(degrees: Double, flip: Bool) { setQRRotation(degrees: degrees); qrFlip = flip }

    /// `set_probe_param((semiangle, qx0, qy0))`: the semiangle and the MEAN origin (py4DSTEM's `set_qx0_mean`; the per-position
    /// origin is left as it is).
    public mutating func setProbeParameters(semiangle: Double, centerRow: Double, centerColumn: Double) {
        probeSemiangle = semiangle; originMeanRow = centerRow; originMeanColumn = centerColumn
    }

    // MARK: coordinate axes and grids (DataCube.calibrate)

    /// The calibrated scan-row axis, `arange(scanRows) * R_pixel_size` (py4DSTEM `dim_rx`).
    public func scanRowAxis(count: Int) -> [Double] { (0..<count).map { Double($0) * scanPixelSize } }
    /// The calibrated scan-column axis (`dim_ry`).
    public func scanColumnAxis(count: Int) -> [Double] { scanRowAxis(count: count) }
    /// The calibrated detector-row axis, `(arange(detectorRows) - mean origin row) * Q_pixel_size` (`dim_qx`); the origin is
    /// 0 when none is set.
    public func detectorRowAxis(count: Int) -> [Double] {
        let o = originMeanRow ?? 0
        return (0..<count).map { (Double($0) - o) * detectorPixelSize }
    }
    /// The calibrated detector-column axis (`dim_qy`).
    public func detectorColumnAxis(count: Int) -> [Double] {
        let o = originMeanColumn ?? 0
        return (0..<count).map { (Double($0) - o) * detectorPixelSize }
    }

    /// `qxx`, the calibrated row coordinate of every detector pixel (`rows x columns`, equal in value to the transpose of
    /// py4DSTEM's (Ny, Nx) array).
    public func detectorRowGrid(shape: CubeShape) -> CoordinateGrid {
        Self.rowGrid(detectorRowAxis(count: shape.detectorRows), columns: shape.detectorColumns)
    }
    /// `qyy`.
    public func detectorColumnGrid(shape: CubeShape) -> CoordinateGrid {
        Self.columnGrid(detectorColumnAxis(count: shape.detectorColumns), rows: shape.detectorRows)
    }
    /// `rxx`.
    public func scanRowGrid(shape: CubeShape) -> CoordinateGrid {
        Self.rowGrid(scanRowAxis(count: shape.scanRows), columns: shape.scanColumns)
    }
    /// `ryy`.
    public func scanColumnGrid(shape: CubeShape) -> CoordinateGrid {
        Self.columnGrid(scanColumnAxis(count: shape.scanColumns), rows: shape.scanRows)
    }
    /// `qxx_raw`: the pixel index of every detector pixel's row.
    public static func rawDetectorRowGrid(shape: CubeShape) -> CoordinateGrid {
        rowGrid((0..<shape.detectorRows).map(Double.init), columns: shape.detectorColumns)
    }
    /// `qyy_raw`.
    public static func rawDetectorColumnGrid(shape: CubeShape) -> CoordinateGrid {
        columnGrid((0..<shape.detectorColumns).map(Double.init), rows: shape.detectorRows)
    }
    /// `rxx_raw`.
    public static func rawScanRowGrid(shape: CubeShape) -> CoordinateGrid {
        rowGrid((0..<shape.scanRows).map(Double.init), columns: shape.scanColumns)
    }
    /// `ryy_raw`.
    public static func rawScanColumnGrid(shape: CubeShape) -> CoordinateGrid {
        columnGrid((0..<shape.scanColumns).map(Double.init), rows: shape.scanRows)
    }

    /// `qxxs(rx, ry)`: the row grid with the origin shift of one scan position removed. Throws `notSet` without an origin.
    public func shiftedDetectorRowGrid(shape: CubeShape, scanRow: Int, scanColumn: Int) throws -> CoordinateGrid {
        guard let s = try originShift(row: scanRow, column: scanColumn) else { throw CalibrationError.notSet("origin shift") }
        let g = detectorRowGrid(shape: shape)
        return CoordinateGrid(rows: g.rows, columns: g.columns, values: g.values.map { $0 - s.row })
    }
    /// `qyys(rx, ry)`.
    public func shiftedDetectorColumnGrid(shape: CubeShape, scanRow: Int, scanColumn: Int) throws -> CoordinateGrid {
        guard let s = try originShift(row: scanRow, column: scanColumn) else { throw CalibrationError.notSet("origin shift") }
        let g = detectorColumnGrid(shape: shape)
        return CoordinateGrid(rows: g.rows, columns: g.columns, values: g.values.map { $0 - s.column })
    }

    private static func rowGrid(_ axis: [Double], columns: Int) -> CoordinateGrid {
        var v = [Double](); v.reserveCapacity(axis.count * columns)
        for a in axis { v.append(contentsOf: repeatElement(a, count: columns)) }
        return CoordinateGrid(rows: axis.count, columns: columns, values: v)
    }
    private static func columnGrid(_ axis: [Double], rows: Int) -> CoordinateGrid {
        var v = [Double](); v.reserveCapacity(axis.count * rows)
        for _ in 0..<rows { v.append(contentsOf: axis) }
        return CoordinateGrid(rows: rows, columns: axis.count, values: v)
    }

    // MARK: pixel sizes after an edit (what py4DSTEM's preprocess functions write back)

    /// After `bin_Q(N)`: the Q pixel size times N. The origin is not rescaled (as upstream).
    public func afterBinningDetector(by factor: Int) throws -> Calibration {
        guard factor >= 1 else { throw CalibrationError.invalidFactor("bin factor \(factor)") }
        var c = self; c.detectorPixelSize *= Double(factor); return c
    }
    /// After `bin_R(N)`: the R pixel size times N.
    public func afterBinningScan(by factor: Int) throws -> Calibration {
        guard factor >= 1 else { throw CalibrationError.invalidFactor("bin factor \(factor)") }
        var c = self; c.scanPixelSize *= Double(factor); return c
    }
    /// After `thin_R(N)`: the R pixel size times N.
    public func afterThinningScan(by factor: Int) throws -> Calibration { try afterBinningScan(by: factor) }
    /// After `resample_Q` by `factor` (the row factor, as py4DSTEM uses `resampling_factor[0]`): the Q pixel size divided by it.
    public func afterResamplingDetector(rowFactor: Double) throws -> Calibration {
        guard rowFactor.isFinite, rowFactor > 0 else { throw CalibrationError.invalidFactor("resampling factor \(rowFactor)") }
        var c = self; c.detectorPixelSize /= rowFactor; return c
    }
}

/// py4DSTEM's `process.calibration.qpixelsize` and `rotation`: the pixel size from a measured and a known distance or from
/// indexed peaks, and the conversion of a direction between scan space and detector space. py4DSTEM's x is the ROW, y the
/// COLUMN.
public enum CalibrationMeasurement {
    /// `get_Q_pixel_size(q_meas, q_known, units)`: `1 / (q_meas * q_known)` and `units + "^-1"`.
    public static func detectorPixelSize(measuredPixels: Double, knownDistance: Double, units: String = "A") throws -> (size: Double, units: String) {
        guard measuredPixels.isFinite, knownDistance.isFinite, measuredPixels != 0, knownDistance != 0 else {
            throw CalibrationError.invalidFactor("measured \(measuredPixels) pixels for a known distance of \(knownDistance)")
        }
        return (1.0 / (measuredPixels * knownDistance), units + "^-1")
    }

    /// `get_dq_from_indexed_peaks(qs, hkl, a)`: the single scale `c` that minimises `Σ (q - c |hkl|)²` over the peaks whose
    /// indices are not (0, 0, 0) (the closed form, `c = Σ q |hkl| / Σ |hkl|²`; upstream reaches the same minimum with
    /// `scipy.optimize.leastsq`, to its 1.5e-8 tolerance), the pixel size `1 / (c a)`, the fitted positions `|hkl| / a` and the
    /// peaks that took part.
    public static func detectorPixelSize(peakPositions qs: [Double], indices hkl: [(Int, Int, Int)], cellSize a: Double) throws
        -> (size: Double, fittedPositions: [Double], indices: [(Int, Int, Int)]) {
        guard qs.count == hkl.count, !qs.isEmpty else { throw CalibrationError.invalidField("\(qs.count) positions for \(hkl.count) index triples") }
        guard a.isFinite, a != 0 else { throw CalibrationError.invalidFactor("cell size \(a)") }
        var used = [Int]()
        for (k, h) in hkl.enumerated() where (h.0, h.1, h.2) != (0, 0, 0) { used.append(k) }
        guard !used.isEmpty else { throw CalibrationError.invalidField("every peak has the index (0, 0, 0)") }
        let d = used.map { k -> Double in let h = hkl[k]; return (Double(h.0 * h.0 + h.1 * h.1 + h.2 * h.2)).squareRoot() }
        var num = 0.0, den = 0.0
        for (i, k) in used.enumerated() { num += qs[k] * d[i]; den += d[i] * d[i] }
        let c = num / den
        guard c != 0, c.isFinite else { throw CalibrationError.invalidField("the fitted scale is \(c)") }
        return (1 / (c * a), d.map { $0 / a }, used.map { hkl[$0] })
    }

    /// `get_Qvector_from_Rvector(vx, vy, QR_rotation)`: the normalised scan-space vector and the detector-space vector for a
    /// rotation in degrees (counterclockwise rotation of scan space with respect to the detector).
    public static func detectorVector(fromScanVector row: Double, _ column: Double, rotationDegrees: Double)
        -> (scan: (row: Double, column: Double), detector: (row: Double, column: Double)) {
        let phi = rotationDegrees * .pi / 180, length = hypot(row, column)
        let r = row / length, c = column / length
        return ((r, c), (cos(phi) * r + sin(phi) * c, -sin(phi) * r + cos(phi) * c))
    }

    /// `get_Rvector_from_Qvector(vx, vy, QR_rotation)`.
    public static func scanVector(fromDetectorVector row: Double, _ column: Double, rotationDegrees: Double)
        -> (scan: (row: Double, column: Double), detector: (row: Double, column: Double)) {
        let phi = rotationDegrees * .pi / 180, length = hypot(row, column)
        let r = row / length, c = column / length
        return ((cos(phi) * r - sin(phi) * c, sin(phi) * r + cos(phi) * c), (r, c))
    }
}
