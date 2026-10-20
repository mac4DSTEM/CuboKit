import Foundation
import CuboCore

/// Where each diffraction pattern's origin (the pixel that moves to the corner) is, in detector pixels.
public enum PtychographyOrigins: Sendable {
    /// One origin for every pattern.
    case uniform(row: Double, column: Double)
    /// One origin per scan position, row-major over the scan (`scanRows * scanColumns` values each),
    /// as py4DSTEM's fitted centre of mass (`com_fitted_x`, `com_fitted_y`).
    case perPosition(rows: [Double], columns: [Double])
}

/// Scan geometry and calibration for the scan positions: sampling in Å, the scan rotation in radians and the transpose
/// flag (py4DSTEM's `_rotation_best_rad` and `_rotation_best_transpose`, from the centre-of-mass rotation solve).
public struct PtychographyScanGeometry: Sendable {
    /// Scan step along the scan rows and along the scan columns, Å.
    public let scanSamplingRow: Double, scanSamplingColumn: Double
    /// Scan rotation, radians.
    public let rotation: Double
    /// Whether the scan axes are exchanged after the rotation.
    public let transpose: Bool
    /// Creates a geometry.
    public init(scanSamplingRow: Double, scanSamplingColumn: Double, rotation: Double = 0, transpose: Bool = false) {
        self.scanSamplingRow = scanSamplingRow; self.scanSamplingColumn = scanSamplingColumn
        self.rotation = rotation; self.transpose = transpose
    }
}

/// How the object canvas is padded around the scan positions (pixels).
public struct PtychographyPadding: Sendable, Equatable {
    /// Offset added to the row and column of every position (top and left padding).
    public let startRow: Double, startColumn: Double
    /// Padding added to the largest row and column when sizing the object (bottom and right).
    public let extentRow: Double, extentColumn: Double
    /// Creates a padding.
    public init(startRow: Double, startColumn: Double, extentRow: Double, extentColumn: Double) {
        self.startRow = startRow; self.startColumn = startColumn
        self.extentRow = extentRow; self.extentColumn = extentColumn
    }
    /// py4DSTEM's default for an `r x c` detector, quirks included: both axes start at `r / 2` and both extents add
    /// `c / 2` (`object_padding_px = (roi / 2, roi / 2)` read at `[0][0]`, `[1][0]` and `[0][1]`, `[1][1]`).
    public static func reference(detectorRows r: Int, detectorColumns c: Int) -> PtychographyPadding {
        PtychographyPadding(startRow: Double(r) / 2, startColumn: Double(r) / 2,
                            extentRow: Double(c) / 2, extentColumn: Double(c) / 2)
    }
    /// The app's choice: each axis padded by its own half detector extent at both ends. Equal to `reference` on a
    /// square detector.
    public static func perAxis(detectorRows r: Int, detectorColumns c: Int) -> PtychographyPadding {
        PtychographyPadding(startRow: Double(r) / 2, startColumn: Double(c) / 2,
                            extentRow: Double(r) / 2, extentColumn: Double(c) / 2)
    }
}

/// How a diffraction pattern is shifted so its origin sits on the corner pixel.
public enum PtychographyShift: Sendable {
    /// Bilinear resampling with wrap-around (py4DSTEM `shifting_interpolation_order=1`, `get_shifted_ar(bilinear=True)`).
    case bilinear
    /// Cubic B-spline interpolation with wrap-around (`shifting_interpolation_order=3`, py4DSTEM's default;
    /// SciPy `ndimage.shift(order=3, mode="grid-wrap")`).
    case cubicSpline
}

/// Square-root amplitudes of a cube, corner-centred, with the mean intensity.
public struct PtychographyAmplitudes: Sendable {
    /// `[position][row][column]`, positions row-major over the scan.
    public let values: [Double]
    /// Number of scan positions and detector shape.
    public let positionCount: Int, detectorRows: Int, detectorColumns: Int
    /// Mean over positions of the summed shifted intensity, negative values included (py4DSTEM `mean_intensity`).
    public let meanIntensity: Double
    /// The same sum with negative shifted values set to zero first (the app's `meanIntensity`).
    public let clippedMeanIntensity: Double
    /// py4DSTEM's error normalisation for the engines: `.meanIntensity(meanIntensity)`.
    public var errorNormalization: PtychographyErrorNormalization { .meanIntensity(meanIntensity) }
}

/// Options of ``PtychographyPreparation/prepare(source:origins:geometry:objectSamplingRow:objectSamplingColumn:wavelength:semiangle:options:)``.
public struct PtychographyPreparationOptions: Sendable {
    /// Pattern shift interpolation.
    public var shift: PtychographyShift
    /// Padding convention; `nil` means py4DSTEM's (`PtychographyPadding.reference`).
    public var padding: PtychographyPadding?
    /// Move the positions' mean to the object centre (py4DSTEM `center_positions_in_fov`, on by default).
    public var centerPositionsInField: Bool
    /// Normalise the probe to the clipped mean intensity (the app) instead of py4DSTEM's unclipped mean.
    public var useClippedMeanIntensity: Bool
    /// Probe aperture rolloff, milliradians.
    public var rolloff: Double
    /// Probe defocus (Å), two-fold astigmatism (Å) and angle (rad), spherical aberration C30 (Å).
    public var defocus: Double, astigmatism: Double, astigmatismAngle: Double, sphericalAberration: Double
    /// Tiling of the pass over the cube.
    public var tiling: ScanTiling
    /// py4DSTEM's defaults: cubic spline, its padding, centred positions, unclipped mean, 2 mrad rolloff.
    public init(shift: PtychographyShift = .cubicSpline, padding: PtychographyPadding? = nil,
                centerPositionsInField: Bool = true, useClippedMeanIntensity: Bool = false, rolloff: Double = 2,
                defocus: Double = 0, astigmatism: Double = 0, astigmatismAngle: Double = 0,
                sphericalAberration: Double = 0, tiling: ScanTiling = .automatic) {
        self.shift = shift; self.padding = padding; self.centerPositionsInField = centerPositionsInField
        self.useClippedMeanIntensity = useClippedMeanIntensity; self.rolloff = rolloff
        self.defocus = defocus; self.astigmatism = astigmatism; self.astigmatismAngle = astigmatismAngle
        self.sphericalAberration = sphericalAberration; self.tiling = tiling
    }
    /// The app's preparer: bilinear shifts, per-axis padding, positions not re-centred, clipped mean intensity.
    public static func mac4DSTEM(detectorRows: Int, detectorColumns: Int, rolloff: Double = 2) -> PtychographyPreparationOptions {
        PtychographyPreparationOptions(shift: .bilinear,
                                       padding: .perAxis(detectorRows: detectorRows, detectorColumns: detectorColumns),
                                       centerPositionsInField: false, useClippedMeanIntensity: true, rolloff: rolloff)
    }
}

/// What a reconstruction starts from.
public struct PreparedPtychography: Sendable {
    /// Corner-centred amplitudes.
    public let amplitudes: PtychographyAmplitudes
    /// Scan positions in object pixels.
    public let positions: [PtychographyPosition]
    /// Object canvas of ones.
    public let object: PtychographyArray
    /// Probe normalised so that `sum |fft2(probe)|^2` equals the mean intensity.
    public let probe: PtychographyArray
}

/// Amplitude stack, scan positions, object canvas and starting probe from a 4D source, after py4DSTEM's preprocessing
/// (`_normalize_diffraction_intensities`, `_calculate_scan_positions_in_pixels`, `_initialize_object`,
/// `_initialize_probe`). Centre-of-mass fits, the rotation search, cropping, padding of patterns, resampling and
/// position masks are not part of it: the origins and the rotation are inputs. Rows and columns follow py4DSTEM's
/// first and second array axis; an app that names its axes the other way relabels them before the call.
public enum PtychographyPreparation {
    /// Object sampling in Å from the wavelength (Å), the angular sampling (mrad per detector pixel) and the pattern
    /// size, `wavelength * 1e3 / dk / n` (py4DSTEM `sampling`).
    public static func objectSampling(wavelength: Double, angularSampling: Double, size: Int) -> Double {
        wavelength * 1e3 / angularSampling / Double(size)
    }

    /// Shifts every pattern of `source` so its origin is at the corner and takes the square root of the clipped
    /// values. The shift follows `scipy.ndimage.shift(pattern, -origin)`: `out[i, j] = pattern[i + originRow, j + originColumn]`,
    /// interpolated, with wrap-around.
    public static func amplitudes(from source: some CubeSource, origins: PtychographyOrigins,
                                  shift: PtychographyShift = .cubicSpline,
                                  tiling: ScanTiling = .automatic) async throws -> PtychographyAmplitudes {
        let shape = source.shape
        let rows = shape.detectorRows, columns = shape.detectorColumns
        let scanColumns = shape.scanColumns, count = shape.scanRows * shape.scanColumns
        let n = rows * columns
        guard count > 0, count <= Int.max / n, rows > 1, columns > 1 else { throw PtychographyError.invalidInput }
        let originRows: [Double], originColumns: [Double]
        switch origins {
        case .uniform(let r, let c):
            originRows = [Double](repeating: r, count: count); originColumns = [Double](repeating: c, count: count)
        case .perPosition(let r, let c):
            originRows = r; originColumns = c
        }
        guard originRows.count == count, originColumns.count == count,
              originRows.allSatisfy({ $0.isFinite && abs($0) < 1e9 }),
              originColumns.allSatisfy({ $0.isFinite && abs($0) < 1e9 }) else { throw PtychographyError.invalidInput }
        var out = [Double](repeating: 0, count: count * n)
        var total = 0.0, clipped = 0.0
        let fft = FFT2D(rows: rows, columns: columns)
        for range in try tiling.ranges(for: shape) {
            let tile = try await source.readScanTile(rows: range)
            for local in 0..<tile.positionCount {
                let index = range.lowerBound * scanColumns + local
                var pattern = [Double](repeating: 0, count: n)
                for k in 0..<n {
                    let v = tile.values[local * n + k]
                    guard v.isFinite else { throw PtychographyError.invalidInput }
                    pattern[k] = Double(v)
                }
                let shifted = shiftPattern(pattern, rows: rows, columns: columns, originRow: originRows[index],
                                           originColumn: originColumns[index], method: shift, fft: fft)
                for k in 0..<n {
                    total += shifted[k]; clipped += max(shifted[k], 0)
                    out[index * n + k] = max(shifted[k], 0).squareRoot()
                }
            }
        }
        return PtychographyAmplitudes(values: out, positionCount: count, detectorRows: rows, detectorColumns: columns,
                                      meanIntensity: total / Double(count), clippedMeanIntensity: clipped / Double(count))
    }

    /// Scan positions in object pixels (py4DSTEM `_calculate_scan_positions_in_pixels` for a raster scan): the raster
    /// in Å, rotated about its mean, optionally transposed with the sampling pair, moved to non-negative values,
    /// divided by the object sampling and offset by the start padding. The rotation matrix and the rotation origin are
    /// rounded to Float32 as py4DSTEM does (`AffineTransform`).
    public static func scanPositions(scanRows: Int, scanColumns: Int, geometry: PtychographyScanGeometry,
                                     objectSamplingRow: Double, objectSamplingColumn: Double,
                                     padding: PtychographyPadding) throws -> [PtychographyPosition] {
        guard scanRows > 0, scanColumns > 0, scanRows <= Int.max / scanColumns,
              [geometry.scanSamplingRow, geometry.scanSamplingColumn, objectSamplingRow, objectSamplingColumn]
                .allSatisfy({ $0.isFinite && $0 > 0 }), geometry.rotation.isFinite,
              [padding.startRow, padding.startColumn, padding.extentRow, padding.extentColumn].allSatisfy(\.isFinite)
        else { throw PtychographyError.invalidInput }
        let count = scanRows * scanColumns
        var p0 = [Double](repeating: 0, count: count), p1 = p0
        for r in 0..<scanRows { for c in 0..<scanColumns {
            p0[r*scanColumns+c] = Double(r) * geometry.scanSamplingRow
            p1[r*scanColumns+c] = Double(c) * geometry.scanSamplingColumn
        } }
        let o0 = Double(Float(p0.reduce(0, +) / Double(count))), o1 = Double(Float(p1.reduce(0, +) / Double(count)))
        let cs = Double(Float(cos(geometry.rotation))), sn = Double(Float(sin(geometry.rotation)))
        for k in 0..<count {
            let u = p0[k] - o0, v = p1[k] - o1
            p0[k] = u*cs + v*sn + o0
            p1[k] = -u*sn + v*cs + o1
        }
        var samplingRow = objectSamplingRow, samplingColumn = objectSamplingColumn
        if geometry.transpose { swap(&p0, &p1); swap(&samplingRow, &samplingColumn) }
        let m0 = min(p0.min()!, 0), m1 = min(p1.min()!, 0)
        return (0..<count).map { k in
            PtychographyPosition(row: (p0[k] - m0) / samplingRow + padding.startRow,
                                 column: (p1[k] - m1) / samplingColumn + padding.startColumn)
        }
    }

    /// Object canvas size for `positions` (py4DSTEM `_initialize_object`): the rounded largest position plus the extent
    /// padding, rounded again, at least the detector size; ties to even as NumPy's `round`.
    public static func objectShape(positions: [PtychographyPosition], detectorRows: Int, detectorColumns: Int,
                                   padding: PtychographyPadding) throws -> (rows: Int, columns: Int) {
        guard let maxRow = positions.map(\.row).max(), let maxColumn = positions.map(\.column).max(),
              maxRow.isFinite, maxColumn.isFinite, abs(maxRow) < 1e6, abs(maxColumn) < 1e6 else { throw PtychographyError.invalidInput }
        let p = max((maxRow.rounded(.toNearestOrEven) + padding.extentRow).rounded(.toNearestOrEven), Double(detectorRows))
        let q = max((maxColumn.rounded(.toNearestOrEven) + padding.extentColumn).rounded(.toNearestOrEven), Double(detectorColumns))
        return (Int(p), Int(q))
    }

    /// Moves the positions' mean to the object centre (`center_positions_in_fov`); positions are held in Float32 as
    /// py4DSTEM holds them.
    public static func centerPositions(_ positions: [PtychographyPosition], objectRows: Int,
                                       objectColumns: Int) -> [PtychographyPosition] {
        let r = positions.map { Double(Float($0.row)) }, c = positions.map { Double(Float($0.column)) }
        let n = Double(max(1, positions.count))
        let dr = r.reduce(0, +) / n - Double(objectRows) / 2, dc = c.reduce(0, +) / n - Double(objectColumns) / 2
        return zip(r, c).map { PtychographyPosition(row: $0 - dr, column: $1 - dc) }
    }

    /// The aperture probe at the object sampling, scaled so `sum |fft2(probe)|^2 = meanIntensity`
    /// (py4DSTEM `_initialize_probe` without a vacuum probe). `semiangle` and `rolloff` are in milliradians.
    public static func initialProbe(rows: Int, columns: Int, samplingRow: Double, samplingColumn: Double,
                                    wavelength: Double, semiangle: Double, rolloff: Double = 2,
                                    defocus: Double = 0, astigmatism: Double = 0, astigmatismAngle: Double = 0,
                                    sphericalAberration: Double = 0, meanIntensity: Double) throws -> PtychographyArray {
        guard meanIntensity.isFinite, meanIntensity > 0 else { throw PtychographyError.invalidInput }
        let built = try PtychographyProbe.build(rows: rows, columns: columns, rowSampling: samplingRow,
                                                columnSampling: samplingColumn, wavelength: wavelength,
                                                cutoff: semiangle / 1e3, rolloff: rolloff / 1e3, defocus: defocus,
                                                astigmatism: astigmatism, astigmatismAngle: astigmatismAngle,
                                                sphericalAberration: sphericalAberration)
        var re = built.real, im = built.imaginary
        FFT2D(rows: rows, columns: columns).transform(&re, &im, inverse: false)
        let energy = zip(re, im).reduce(0) { $0 + $1.0*$1.0 + $1.1*$1.1 }
        let scale = (meanIntensity / energy).squareRoot()
        return try PtychographyArray(rows: rows, columns: columns, real: built.real.map { $0 * scale },
                                     imaginary: built.imaginary.map { $0 * scale })
    }

    /// Amplitudes, positions, object canvas and probe in one call.
    /// `objectSamplingRow/Column` are Å per object pixel (see ``objectSampling(wavelength:angularSampling:size:)``);
    /// `semiangle` is the probe aperture in milliradians.
    public static func prepare(source: some CubeSource, origins: PtychographyOrigins, geometry: PtychographyScanGeometry,
                               objectSamplingRow: Double, objectSamplingColumn: Double, wavelength: Double,
                               semiangle: Double,
                               options: PtychographyPreparationOptions = .init()) async throws -> PreparedPtychography {
        let shape = source.shape
        let a = try await amplitudes(from: source, origins: origins, shift: options.shift, tiling: options.tiling)
        let padding = options.padding ?? .reference(detectorRows: shape.detectorRows, detectorColumns: shape.detectorColumns)
        var positions = try scanPositions(scanRows: shape.scanRows, scanColumns: shape.scanColumns, geometry: geometry,
                                          objectSamplingRow: objectSamplingRow, objectSamplingColumn: objectSamplingColumn,
                                          padding: padding)
        let size = try objectShape(positions: positions, detectorRows: shape.detectorRows,
                                   detectorColumns: shape.detectorColumns, padding: padding)
        if options.centerPositionsInField { positions = centerPositions(positions, objectRows: size.rows, objectColumns: size.columns) }
        let mean = options.useClippedMeanIntensity ? a.clippedMeanIntensity : a.meanIntensity
        let probe = try initialProbe(rows: shape.detectorRows, columns: shape.detectorColumns,
                                     samplingRow: objectSamplingRow, samplingColumn: objectSamplingColumn,
                                     wavelength: wavelength, semiangle: semiangle, rolloff: options.rolloff,
                                     defocus: options.defocus, astigmatism: options.astigmatism,
                                     astigmatismAngle: options.astigmatismAngle,
                                     sphericalAberration: options.sphericalAberration, meanIntensity: mean)
        let n = size.rows * size.columns
        let object = try PtychographyArray(rows: size.rows, columns: size.columns, real: [Double](repeating: 1, count: n),
                                           imaginary: [Double](repeating: 0, count: n))
        return PreparedPtychography(amplitudes: a, positions: positions, object: object, probe: probe)
    }
}

// MARK: - pattern shifts

private func floorMod(_ v: Int, _ n: Int) -> Int { let m = v % n; return m < 0 ? m + n : m }

private func shiftPattern(_ p: [Double], rows: Int, columns: Int, originRow: Double, originColumn: Double,
                          method: PtychographyShift, fft: FFT2D) -> [Double] {
    switch method {
    case .bilinear:
        var out = [Double](repeating: 0, count: p.count)
        func taps(_ origin: Double, _ i: Int, _ n: Int) -> (Int, Int, Double) {
            let x = Double(i) + origin, f = x.rounded(.down)
            return (floorMod(Int(f), n), floorMod(Int(f) + 1, n), x - f)
        }
        for r in 0..<rows {
            let (r0, r1, fr) = taps(originRow, r, rows)
            for c in 0..<columns {
                let (c0, c1, fc) = taps(originColumn, c, columns)
                out[r*columns+c] = p[r0*columns+c0]*(1-fr)*(1-fc) + p[r1*columns+c0]*fr*(1-fc)
                    + p[r0*columns+c1]*(1-fr)*fc + p[r1*columns+c1]*fr*fc
            }
        }
        return out
    case .cubicSpline:
        // Periodic cubic B-spline coefficients by division in Fourier space (the cubic kernel's DFT is
        // (4 + 2 cos(2 pi k / n)) / 6), then four taps per axis.
        var re = p, im = [Double](repeating: 0, count: p.count)
        fft.transform(&re, &im, inverse: false)
        for r in 0..<rows {
            let hr = (4 + 2 * cos(2 * Double.pi * Double(r) / Double(rows))) / 6
            for c in 0..<columns {
                let h = hr * (4 + 2 * cos(2 * Double.pi * Double(c) / Double(columns))) / 6
                re[r*columns+c] /= h; im[r*columns+c] /= h
            }
        }
        fft.transform(&re, &im, inverse: true)
        func weights(_ origin: Double, _ i: Int, _ n: Int) -> ([Int], [Double]) {
            let x = Double(i) + origin, f = x.rounded(.down), t = x - f
            let w = [(1-t)*(1-t)*(1-t)/6, (3*t*t*t - 6*t*t + 4)/6, (-3*t*t*t + 3*t*t + 3*t + 1)/6, t*t*t/6]
            return ((-1...2).map { floorMod(Int(f) + $0, n) }, w)
        }
        var out = [Double](repeating: 0, count: p.count)
        let cw = (0..<columns).map { weights(originColumn, $0, columns) }
        for r in 0..<rows {
            let (ri, rw) = weights(originRow, r, rows)
            for c in 0..<columns {
                let (ci, w) = cw[c]
                var s = 0.0
                for a in 0..<4 { for b in 0..<4 { s += rw[a] * w[b] * re[ri[a]*columns + ci[b]] } }
                out[r*columns+c] = s
            }
        }
        return out
    }
}
