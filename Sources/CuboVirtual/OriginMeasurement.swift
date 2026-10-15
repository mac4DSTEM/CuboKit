//
//  OriginMeasurement.swift
//  Per-position origins (registration slice 3, 3.1): py4DSTEM's get_origin (process/calibration/origin.py 222 at f050d207)
//  in Double: scipy's gaussian_filter (σ = r, truncate 4, mode nearest, rows then columns), the first argmax, and the centre
//  of mass of the pattern within r · rscale of it.
//
//  Seeds: mac4DSTEM 7f192da9 Core/Analysis/OriginCalibration.swift and Shaders/OriginMeasure.metal (Float32, a CoM window
//  without the Gaussian argmax); the kit follows py4DSTEM.
//

import CuboCore
import Foundation

/// An origin per scan position, in detector pixels, with its means (NumPy's `np.mean`, as py4DSTEM's calibration keeps).
public struct OriginMap: Sendable, Equatable {
    public let row: ScanImage
    public let column: ScanImage
    public let meanRow: Double
    public let meanColumn: Double

    public init(row: ScanImage, column: ScanImage) throws {
        guard row.scanRows == column.scanRows, row.scanColumns == column.scanColumns else {
            throw CuboVirtualError.originMapMismatch("the row and column maps differ in shape")
        }
        self.row = row
        self.column = column
        meanRow = NumPySum.mean(row.values)
        meanColumn = NumPySum.mean(column.values)
    }
}

public enum OriginMeasurement {
    /// py4DSTEM's get_origin: `radius` defaults to the probe size of the maximum pattern (py4DSTEM's `dp_max` default).
    public static func measure(of source: some CubeSource, radius: Double? = nil, radiusScale: Double = 1.2,
                               tiling: ScanTiling = .automatic) async throws -> (origins: OriginMap, radius: Double) {
        let shape = source.shape
        let r: Double
        if let radius { r = radius } else {
            let max = try await VirtualDiffraction.max(of: source, tiling: tiling)
            r = try ProbeSize.estimate(max).radius
        }
        guard radiusScale > 0, radiusScale.isFinite else { throw CuboVirtualError.invalidRadius(radiusScale) }
        let qy = shape.detectorRows, qx = shape.detectorColumns, n = shape.detectorCount
        let filter = try GaussianFilter(sigma: r)
        var rows = [Double](repeating: 0, count: shape.scanCount), cols = rows
        let limit = r * radiusScale
        for range in try tiling.ranges(for: shape) {
            try Task.checkCancellation()
            let tile = try await source.readScanTile(rows: range)
            let first = range.lowerBound * shape.scanColumns
            for k in 0..<tile.positionCount {
                let p = (0..<n).map { Double(tile.values[k * n + $0]) }
                let g = filter.apply(p, rows: qy, columns: qx)
                // np.argmax: the first maximum, or the first NaN if there is one (NaN compares as the largest)
                var best = 0
                if let nan = g.firstIndex(where: \.isNaN) { best = nan } else { for i in 1..<n where g[i] > g[best] { best = i } }
                let i0 = Double(best / qx), j0 = Double(best % qx)
                var s = Neumaier(), mr = Neumaier(), mc = Neumaier()
                for i in 0..<n {   // get_CoM(dp · mask): every pixel times its 0/1 weight, so a NaN anywhere is NaN
                    let di = Double(i / qx) - i0, dj = Double(i % qx) - j0
                    let v = p[i] * (hypot(di, dj) < limit ? 1.0 : 0.0)
                    s.add(v); mr.add(Double(i / qx) * v); mc.add(Double(i % qx) * v)
                }
                rows[first + k] = mr.value / s.value
                cols[first + k] = mc.value / s.value
            }
        }
        let img = { (v: [Double]) in ScanImage(scanRows: shape.scanRows, scanColumns: shape.scanColumns, values: v, path: .cpu) }
        return (try OriginMap(row: img(rows), column: img(cols)), r)
    }
}

/// scipy.ndimage.gaussian_filter in Double: kernel exp((−0.5/σ²)·t²), t = −R…R, R = int(4σ + 0.5), normalised by its
/// sum; one pass along the rows' axis then the columns', edge values repeated (mode nearest).
public struct GaussianFilter: Sendable {
    public let sigma: Double
    public let kernel: [Double]
    public var radius: Int { (kernel.count - 1) / 2 }

    /// σ must be positive, finite, with σ² representable (above about 1.5e-154) and a radius below 10⁶ pixels.
    public init(sigma: Double) throws {
        guard sigma > 0, sigma.isFinite, sigma * sigma >= Double.leastNormalMagnitude, 4.0 * sigma + 0.5 < 1e6 else {
            throw CuboVirtualError.invalidRadius(sigma)
        }
        self.sigma = sigma
        let R = Int(4.0 * sigma + 0.5)
        let w = (-R...R).map { exp(-0.5 / (sigma * sigma) * Double($0 * $0)) }
        let sum = NumPySum.pairwise(w)                   // phi_x / phi_x.sum()
        kernel = w.map { $0 / sum }
    }

    /// Filters a `rows × columns` pattern (row-major) in scipy's order for a symmetric kernel (ni_filters.c
    /// NI_Correlate1D): the centre weight times the pixel, then (p[i − t] + p[i + t]) · k[t] for t = R down to 1. The
    /// pairing makes a mirror-symmetric pattern filter to exactly symmetric values, so exact ties stay ties.
    public func apply(_ p: [Double], rows: Int, columns: Int) -> [Double] {
        let R = radius, k0 = kernel[R]
        var a = [Double](repeating: 0, count: p.count)
        for i in 0..<rows { for j in 0..<columns {
            var s = p[i * columns + j] * k0
            for t in stride(from: R, through: 1, by: -1) {
                s += (p[max(i - t, 0) * columns + j] + p[min(i + t, rows - 1) * columns + j]) * kernel[R - t]
            }
            a[i * columns + j] = s
        } }
        var b = [Double](repeating: 0, count: p.count)
        for i in 0..<rows { for j in 0..<columns {
            var s = a[i * columns + j] * k0
            for t in stride(from: R, through: 1, by: -1) {
                s += (a[i * columns + max(j - t, 0)] + a[i * columns + min(j + t, columns - 1)]) * kernel[R - t]
            }
            b[i * columns + j] = s
        } }
        return b
    }
}
