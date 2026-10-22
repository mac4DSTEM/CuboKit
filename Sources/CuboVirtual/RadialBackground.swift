//
//  RadialBackground.swift
//  The radial background of a diffraction pattern from its polar transform (task 17): py4DSTEM's `DataCube.get_radial_bkgrnd`,
//  `get_radial_bksb_dp` and `get_local_ave_dp(radial_bksb=True)` (datacube/datacube.py at f050d207), in Double on the CPU.
//
//  The background is the median over the angle of the valid polar cells at each radius (zero where none is valid), smoothed
//  along the radius by a Gaussian (`sigma`, mode `reflect`), and laid back on the detector as rings of constant value about the
//  origin, interpolating linearly (extrapolating beyond the first and last bin) in the elliptical radius.
//
//  Port notes (what a reader comparing with py4DSTEM should know):
//    - The elliptical radius of a detector pixel here is on the scale of the MAJOR semi-axis,
//      `sqrt(u^2 + v^2 / (b / a)^2)` with u, v along the axes, while ``PolarCube`` bins radii on the scale of the MINOR one
//      (`(b / a) * hypot(...)`). py4DSTEM does exactly this, so a ring of the ellipse sits at `a` in the lookup and at `b` in
//      the polar bins: for `a != b` the background is laid out stretched by `a / b` against the polar profile. It is ported as
//      it is (D17); a circle is not affected. No ellipse in the calibration is the circle `(1, 1, 0)`, whatever the sampling
//      says; the sampling's `elliptical: false` switches the polar transform only, as in py4DSTEM.
//    - Patterns are Float32 in the cube; the subtraction runs in Double and zeroes negative values (NaN stays NaN).
//    - `localAverage` clips the 3 x 3 Gaussian kernel at the scan edge and renormalises, as py4DSTEM does; py4DSTEM's own edge
//      branches fail for a scan one position wide, this one does not.
//

import CuboCore
import Foundation

public enum RadialBackground {
    /// The one-dimensional background of a polar pattern: per radial bin the median of the valid cells over the angle (0 where
    /// no cell is valid), then `gaussian_filter1d(sigma)` when `sigma > 0`.
    public static func profile(of polar: PolarPattern, sigma: Double = 2) -> [Double] {
        var median = [Double](repeating: 0, count: polar.radialCount)
        for r in 0..<polar.radialCount {
            var column = [Double]()
            for t in 0..<polar.annularCount where polar.valid[t * polar.radialCount + r] { column.append(polar.values[t * polar.radialCount + r]) }
            median[r] = SciPyKernels.median(of: column) ?? 0
        }
        guard sigma > 0 else { return median }
        return SciPyKernels.gaussianFilter1D(median, sigma: sigma, boundary: .reflect)
    }

    /// The elliptical radius of every detector pixel about `origin`, on the scale of the major semi-axis (`qrr`).
    public static func ellipticalRadius(detectorRows: Int, detectorColumns: Int, origin: (row: Double, column: Double),
                                        ellipse: (a: Double, b: Double, theta: Double)?) -> [Double] {
        let e = ellipse ?? (1, 1, 0)
        let cosT = cos(e.theta), sinT = sin(e.theta)
        let ratio = e.b / e.a
        let ratio2 = ratio * ratio
        var out = [Double](repeating: 0, count: detectorRows * detectorColumns)
        for r in 0..<detectorRows { for c in 0..<detectorColumns {
            let x = Double(r) - origin.row, y = Double(c) - origin.column
            let u = x * cosT + y * sinT, v = x * sinT - y * cosT
            out[r * detectorColumns + c] = (u * u + v * v / ratio2).squareRoot()
        } }
        return out
    }

    /// `get_radial_bkgrnd(rx, ry, sigma)`: the radial background image of the pattern at a scan position, from the polar transform
    /// of that pattern. Throws ``CalibrationError/notSet(_:)`` without an origin and ``CuboVirtualError/invalidInput(_:)`` for
    /// fewer than two radial bins (nothing to interpolate).
    public static func background(of source: some CubeSource, polar: PolarCube, calibration: Calibration, scanRow: Int,
                                  scanColumn: Int, sigma: Double = 2) async throws -> DetectorImage {
        let pattern = try await source.readPattern(row: scanRow, column: scanColumn)
        return try background(of: pattern, polar: polar, calibration: calibration, scanRow: scanRow, scanColumn: scanColumn, sigma: sigma)
    }

    static func background(of pattern: DiffractionPattern, polar: PolarCube, calibration: Calibration, scanRow: Int, scanColumn: Int,
                           sigma: Double) throws -> DetectorImage {
        guard polar.radialCount >= 2 else { throw CuboVirtualError.invalidInput("a radial background needs at least two radial bins") }
        guard sigma.isFinite else { throw CuboVirtualError.invalidInput("the smoothing sigma must be finite") }
        guard pattern.detectorRows == polar.detectorRows, pattern.detectorColumns == polar.detectorColumns else {
            throw CuboVirtualError.invalidInput("the polar cube samples a \(polar.detectorRows) x \(polar.detectorColumns) detector, the pattern has \(pattern.detectorRows) x \(pattern.detectorColumns)")
        }
        let place = try PolarCube.placement(of: calibration, scanRow: scanRow, scanColumn: scanColumn)
        let transformed = try polar.transform(pattern: pattern.values, origin: place.origin, ellipse: place.ellipse)
        let oneD = profile(of: transformed, sigma: sigma)
        let qrr = ellipticalRadius(detectorRows: polar.detectorRows, detectorColumns: polar.detectorColumns, origin: place.origin,
                                   ellipse: calibration.ellipse == nil ? nil : place.ellipse)
        let values = qrr.map { SciPyKernels.interpolateLinear(x: polar.radialBins, y: oneD, at: $0) }
        return DetectorImage(detectorRows: polar.detectorRows, detectorColumns: polar.detectorColumns, values: values, path: .cpu)
    }

    /// `get_radial_bksb_dp(rx, ry, sigma)`: the pattern minus its radial background, negative values set to zero.
    public static func subtracted(from source: some CubeSource, polar: PolarCube, calibration: Calibration, scanRow: Int,
                                  scanColumn: Int, sigma: Double = 2) async throws -> DetectorImage {
        let pattern = try await source.readPattern(row: scanRow, column: scanColumn)
        return try subtract(pattern, polar: polar, calibration: calibration, scanRow: scanRow, scanColumn: scanColumn, sigma: sigma)
    }

    static func subtract(_ pattern: DiffractionPattern, polar: PolarCube, calibration: Calibration, scanRow: Int, scanColumn: Int,
                         sigma: Double) throws -> DetectorImage {
        let bg = try background(of: pattern, polar: polar, calibration: calibration, scanRow: scanRow, scanColumn: scanColumn, sigma: sigma)
        var out = [Double](repeating: 0, count: bg.values.count)
        for i in out.indices {
            let v = Double(pattern.values[i]) - bg.values[i]
            out[i] = v < 0 ? 0 : v
        }
        return DetectorImage(detectorRows: bg.detectorRows, detectorColumns: bg.detectorColumns, values: out, path: .cpu)
    }

    /// `get_local_ave_dp(rx, ry, radial_bksb=True, sigma)`: the radial-background-subtracted patterns of the 3 x 3 neighbourhood
    /// averaged with the kernel `[[1, 2, 1], [2, 4, 2], [1, 2, 1]] / 16` (cut at the scan edge and renormalised). `isCancelled`
    /// is asked before each neighbour and throws `CancellationError`.
    ///
    /// `keeping` is py4DSTEM's `braggmask=True` together with `radial_bksb=True`: for a scan position it returns which detector
    /// pixels take part (`true`; `BraggMasking.mask` outside the Bragg disks, in CuboBragg, which this module cannot call). Each
    /// neighbour then contributes only where it keeps the pixel and the sum is divided by the summed weights; a pixel no neighbour
    /// keeps is NaN (py4DSTEM's masked entry).
    public static func localAverage(of source: some CubeSource, polar: PolarCube, calibration: Calibration, scanRow: Int,
                                    scanColumn: Int, sigma: Double = 2,
                                    keeping: (@Sendable (_ scanRow: Int, _ scanColumn: Int) throws -> [Bool])? = nil,
                                    isCancelled: (@Sendable () -> Bool)? = nil) async throws -> DetectorImage {
        let s = source.shape
        try polar.requireDetector(of: s)
        try s.requirePosition(row: scanRow, column: scanColumn)
        let rows = (max(0, scanRow - 1)...min(s.scanRows - 1, scanRow + 1)).map { $0 }
        let columns = (max(0, scanColumn - 1)...min(s.scanColumns - 1, scanColumn + 1)).map { $0 }
        let kernel: [[Double]] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]].map { $0.map { Double($0) / 16.0 } }
        var weights = [[Double]]()
        var total = 0.0
        for r in rows { var line = [Double](); for c in columns { let w = kernel[r - scanRow + 1][c - scanColumn + 1]; line.append(w); total += w }; weights.append(line) }
        var ans = [Double](repeating: 0, count: s.detectorCount)
        var summedWeights = [Double](repeating: 0, count: keeping == nil ? 0 : s.detectorCount)
        for (i, r) in rows.enumerated() { for (j, c) in columns.enumerated() {
            if isCancelled?() == true { throw CancellationError() }
            try Task.checkCancellation()
            let w = weights[i][j] / total
            let dp = try await subtracted(from: source, polar: polar, calibration: calibration, scanRow: r, scanColumn: c, sigma: sigma)
            if let keeping {
                let keep = try keeping(r, c)
                guard keep.count == s.detectorCount else {
                    throw CuboVirtualError.maskLengthMismatch(expected: s.detectorCount, received: keep.count)
                }
                for k in ans.indices { let wk = keep[k] ? w : 0; ans[k] += dp.values[k] * wk; summedWeights[k] += wk }
            } else {
                for k in ans.indices { ans[k] += dp.values[k] * w }
            }
        } }
        if keeping != nil { for k in ans.indices { ans[k] = summedWeights[k] > 0 ? ans[k] / summedWeights[k] : .nan } }
        return DetectorImage(detectorRows: s.detectorRows, detectorColumns: s.detectorColumns, values: ans, path: .cpu)
    }
}
