//
//  ShiftedImaging.swift
//  Centred, calibrated and centre-shifted virtual images, and centre-shifted mean and maximum patterns (registration
//  slice 3, 3.3, 3.4): py4DSTEM's get_virtual_image(centered, calibrated, shift_center, subpixel) and
//  get_virtual_diffraction(shift_center, subpixel) at f050d207, CPU, Double.
//
//  DEVIATIONS: centred and calibrated together are refused (py4DSTEM divides the pixel origin by the pixel size); the
//  shifted maximum is seeded at −∞ (py4DSTEM seeds it at 0, which reads a negative pixel as 0).
//

import CuboCore
import Foundation

/// How the detector mask follows the beam from position to position.
public enum MaskShift: Sendable {
    case none
    /// By the origin minus the map's mean, rounded half to even, rolled periodically (py4DSTEM `shift_center`).
    case rounded(OriginMap)
    /// By the same shift, bilinear between four rolls (py4DSTEM `subpixel=True`).
    case bilinear(OriginMap)
}

/// How each pattern is moved before the mean and maximum.
public enum PatternShift: Sendable {
    /// By the map's mean minus the origin, rounded half to even, rolled.
    case rounded(OriginMap)
    /// By the same shift, through the Fourier shift theorem (py4DSTEM's get_shifted_ar default).
    case fourier(OriginMap)
}

extension VirtualImaging {
    /// A virtual image with py4DSTEM's centring (`centeredOn`, normally the origin map's mean), calibration (`pixelSize`,
    /// Å⁻¹ per pixel, the geometry given in Å⁻¹) and centre shift. A shift runs on the CPU only.
    public static func image(of source: some CubeSource, geometry: DetectorGeometry,
                             centeredOn origin: (row: Double, column: Double)?, pixelSize: Double? = nil,
                             shift: MaskShift = .none, path: ComputePath = .cpu,
                             tiling: ScanTiling = .automatic) async throws -> ScanImage {
        let shape = source.shape
        let g = try transform(geometry, origin: origin, pixelSize: pixelSize)
        switch shift {
        case .none:
            return try await image(of: source, geometry: g, path: path, tiling: tiling)
        case .rounded(let map), .bilinear(let map):
            guard path == .cpu else { throw CuboVirtualError.unsupportedOnMetal("a shifted virtual image") }
            try require(map, matches: shape)
            let mask = try DetectorMask.make(g, shape: shape).map(Double.init)
            let bilinear: Bool
            if case .bilinear = shift { bilinear = true } else { bilinear = false }
            let qy = shape.detectorRows, qx = shape.detectorColumns, n = shape.detectorCount
            var out = [Double](repeating: 0, count: shape.scanCount)
            for range in try tiling.ranges(for: shape) {
                try Task.checkCancellation()
                let tile = try await source.readScanTile(rows: range)
                let first = range.lowerBound * shape.scanColumns
                for k in 0..<tile.positionCount {
                    let dr = map.row.values[first + k] - map.meanRow, dc = map.column.values[first + k] - map.meanColumn
                    let m: [Double]
                    if bilinear {
                        let fr = dr.rounded(.down), fc = dc.rounded(.down), wr = dr - fr, wc = dc - fc
                        let ir = wrap(fr, qy), ic = wrap(fc, qx)
                        let a = roll(mask, qy, qx, ir, ic), b = roll(mask, qy, qx, ir + 1, ic)
                        let c = roll(mask, qy, qx, ir, ic + 1), d = roll(mask, qy, qx, ir + 1, ic + 1)
                        let w00 = (1 - wr) * (1 - wc), w10 = wr * (1 - wc), w01 = (1 - wr) * wc, w11 = wr * wc
                        m = (0..<n).map { a[$0] * w00 + b[$0] * w10 + c[$0] * w01 + d[$0] * w11 }
                    } else {
                        m = roll(mask, qy, qx, wrap(dr.rounded(.toNearestOrEven), qy), wrap(dc.rounded(.toNearestOrEven), qx))
                    }
                    var s = Neumaier()
                    for i in 0..<n { s.add(Double(tile.values[k * n + i]) * m[i]) }
                    out[first + k] = s.value
                }
            }
            return ScanImage(scanRows: shape.scanRows, scanColumns: shape.scanColumns, values: out, path: .cpu)
        }
    }

    /// py4DSTEM's get_calibrated_detector_geometry: the origin added, then every coordinate divided by the pixel size.
    static func transform(_ g: DetectorGeometry, origin: (row: Double, column: Double)?, pixelSize: Double?) throws -> DetectorGeometry {
        if origin != nil && pixelSize != nil { throw CuboVirtualError.centeredAndCalibrated }
        if let p = pixelSize { guard p > 0, p.isFinite else { throw CuboVirtualError.invalidPixelSize(p) } }
        let r0 = origin?.row ?? 0, c0 = origin?.column ?? 0, s = pixelSize ?? 1
        switch g {
        case .mask: return g
        case .point(let row, let column):
            // py4DSTEM: int(g[0] + x0_mean) and int(g / p), truncating toward zero (a non-finite or huge value is refused,
            // as Python's int(nan) raises)
            let pr = ((Double(row) + r0) / s).rounded(.towardZero), pc = ((Double(column) + c0) / s).rounded(.towardZero)
            guard pr.isFinite, pc.isFinite, abs(pr) < 1e15, abs(pc) < 1e15 else { throw CuboVirtualError.nonFiniteGeometry }
            return .point(row: Int(pr), column: Int(pc))
        case .circle(let cr, let cc, let r):
            return .circle(centerRow: (cr + r0) / s, centerColumn: (cc + c0) / s, radius: r / s)
        case .annulus(let cr, let cc, let i, let o):
            return .annulus(centerRow: (cr + r0) / s, centerColumn: (cc + c0) / s, inner: i / s, outer: o / s)
        case .rectangle(let a, let b, let c, let d):
            return .rectangle(rowMin: (a + r0) / s, rowMax: (b + r0) / s, columnMin: (c + c0) / s, columnMax: (d + c0) / s)
        }
    }

    static func require(_ map: OriginMap, matches shape: CubeShape) throws {
        guard map.row.scanRows == shape.scanRows, map.row.scanColumns == shape.scanColumns else {
            throw CuboVirtualError.originMapMismatch("the origin map is \(map.row.scanRows) × \(map.row.scanColumns), the scan \(shape.scanRows) × \(shape.scanColumns)")
        }
        guard (map.row.values + map.column.values).allSatisfy(\.isFinite) else {
            throw CuboVirtualError.originMapMismatch("the origin map has non-finite values")
        }
    }

    /// An integral shift reduced modulo the axis length in Double first, so a shift of any finite size never traps.
    static func wrap(_ v: Double, _ n: Int) -> Int {
        let m = v.truncatingRemainder(dividingBy: Double(n))
        return Int(m < 0 ? m + Double(n) : m) % n
    }

    /// np.roll(a, (dr, dc), axis=(0, 1)): out[i, j] = a[(i − dr) mod qy, (j − dc) mod qx].
    static func roll(_ a: [Double], _ qy: Int, _ qx: Int, _ dr: Int, _ dc: Int) -> [Double] {
        var out = [Double](repeating: 0, count: a.count)
        let sr = ((dr % qy) + qy) % qy, sc = ((dc % qx) + qx) % qx
        for i in 0..<qy { for j in 0..<qx {
            out[((i + sr) % qy) * qx + (j + sc) % qx] = a[i * qx + j]
        } }
        return out
    }
}

extension VirtualDiffraction {
    /// The mean and maximum of the patterns each moved by the map's mean minus its origin. CPU.
    public static func meanAndMax(of source: some CubeSource, shift: PatternShift, tiling: ScanTiling = .automatic)
        async throws -> (mean: DetectorImage, max: DetectorImage) {
        let shape = source.shape
        let map: OriginMap, fourier: Bool
        switch shift { case .rounded(let m): map = m; fourier = false; case .fourier(let m): map = m; fourier = true }
        try VirtualImaging.require(map, matches: shape)
        let qy = shape.detectorRows, qx = shape.detectorColumns, n = shape.detectorCount
        let fft = fourier ? FFT2D(rows: qy, columns: qx) : nil
        let fr = fftfreq(qy), fc = fftfreq(qx)
        var sums = [Neumaier](repeating: Neumaier(), count: n)
        var maximum = [Double](repeating: -.infinity, count: n)
        for range in try tiling.ranges(for: shape) {
            try Task.checkCancellation()
            let tile = try await source.readScanTile(rows: range)
            let first = range.lowerBound * shape.scanColumns
            for k in 0..<tile.positionCount {
                let sr = map.meanRow - map.row.values[first + k], sc = map.meanColumn - map.column.values[first + k]
                var p = (0..<n).map { Double(tile.values[k * n + $0]) }
                if let fft {
                    var im = [Double](repeating: 0, count: n)
                    fft.transform(&p, &im, inverse: false)
                    for i in 0..<qy { for j in 0..<qx {
                        let angle = -2 * Double.pi * (sc * fc[j] + sr * fr[i])
                        let c = cos(angle), s = sin(angle), a = p[i * qx + j], b = im[i * qx + j]
                        p[i * qx + j] = a * c - b * s; im[i * qx + j] = a * s + b * c
                    } }
                    fft.transform(&p, &im, inverse: true)
                } else {
                    p = VirtualImaging.roll(p, qy, qx, VirtualImaging.wrap(sr.rounded(.toNearestOrEven), qy),
                                            VirtualImaging.wrap(sc.rounded(.toNearestOrEven), qx))
                }
                for i in 0..<n {
                    sums[i].add(p[i])
                    maximum[i] = (maximum[i].isNaN || p[i].isNaN) ? .nan : Swift.max(maximum[i], p[i])
                }
            }
        }
        let count = Double(shape.scanCount)
        return (DetectorImage(detectorRows: qy, detectorColumns: qx, values: sums.map { $0.value / count }, path: .cpu),
                DetectorImage(detectorRows: qy, detectorColumns: qx, values: maximum, path: .cpu))
    }

    static func fftfreq(_ n: Int) -> [Double] {
        let scale = 1.0 / Double(n), positive = (n - 1) / 2 + 1
        return (0..<n).map { Double($0 < positive ? $0 : $0 - n) * scale }
    }
}
