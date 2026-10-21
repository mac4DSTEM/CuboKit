//
//  BraggMasking.swift
//  The mask around Bragg disks and the locally averaged pattern: py4DSTEM's `DataCube.get_braggmask` and
//  `DataCube.get_local_ave_dp` (without the radial background, which needs the polar data cube: DiffractoKit).
//
//  py4DSTEM's x is the ROW and y the COLUMN. Float32 patterns are averaged in Double (tensordot promotes the Float32 data
//  against the Float64 kernel).
//

import CuboCore
import CuboVirtual
import Foundation

public enum BraggMasking {
    /// `get_braggmask(braggvectors, rx, ry, radius)`: true except within `radius` (distance `<= radius` is masked, as upstream
    /// keeps `qr > radius`) of every Bragg peak at the scan position. Row-major `detectorRows x detectorColumns`.
    public static func mask(vectors: BraggVectors, scanRow: Int, scanColumn: Int, detectorRows: Int, detectorColumns: Int,
                            radius: Double) throws -> [Bool] {
        guard (0..<vectors.scanRows).contains(scanRow), (0..<vectors.scanColumns).contains(scanColumn) else {
            throw CuboBraggError.shapeMismatch("scan position (\(scanRow), \(scanColumn)) is outside the \(vectors.scanRows) x \(vectors.scanColumns) scan")
        }
        guard detectorRows > 0, detectorColumns > 0, radius.isFinite else {
            throw CuboBraggError.invalidOption("a mask needs a positive detector and a finite radius")
        }
        var mask = [Bool](repeating: true, count: detectorRows * detectorColumns)
        for peak in vectors[scanRow, scanColumn] {
            for r in 0..<detectorRows { for c in 0..<detectorColumns {
                if !(hypot(Double(r) - peak.row, Double(c) - peak.column) > radius) {
                    mask[r * detectorColumns + c] = false   // upstream keeps `qr > radius`; a NaN peak masks
                }
            } }
        }
        return mask
    }

    /// A locally averaged pattern; `valid` is false where a Bragg mask removed every neighbour (upstream returns a masked
    /// array there) and `values` holds NaN.
    public struct AveragedPattern: Sendable, Equatable {
        public let detectorRows: Int, detectorColumns: Int
        public let values: [Double]
        public let valid: [Bool]
    }

    /// `get_local_ave_dp(rx, ry, radial_bksb=False, braggmask=...)`: the pattern at the scan position averaged with its
    /// neighbours under the 3 x 3 Gaussian kernel `[[1,2,1],[2,4,2],[1,2,1]] / 16`, cut at the scan edge and renormalised.
    /// With `braggMask` (vectors and radius) each neighbour contributes only where it is outside the Bragg mask and the
    /// result is divided by the summed weights.
    public static func localAverage(of source: some CubeSource, scanRow: Int, scanColumn: Int,
                                    braggMask: (vectors: BraggVectors, radius: Double)? = nil) async throws -> AveragedPattern {
        let s = source.shape
        guard (0..<s.scanRows).contains(scanRow), (0..<s.scanColumns).contains(scanColumn) else {
            throw CuboBraggError.shapeMismatch("scan position (\(scanRow), \(scanColumn)) is outside the cube")
        }
        let kernel: [[Double]] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]].map { $0.map { Double($0) / 16 } }
        let rows = (max(0, scanRow - 1)...min(s.scanRows - 1, scanRow + 1)).map { $0 }
        let cols = (max(0, scanColumn - 1)...min(s.scanColumns - 1, scanColumn + 1)).map { $0 }
        var total = 0.0
        for r in rows { for c in cols { total += kernel[r - scanRow + 1][c - scanColumn + 1] } }
        let n = s.detectorCount
        var sum = [Double](repeating: 0, count: n), weights = [Double](repeating: 0, count: n)
        for r in rows { for c in cols {
            let w = kernel[r - scanRow + 1][c - scanColumn + 1] / total
            let p = try await source.readPattern(row: r, column: c).values
            if let braggMask {
                let m = try mask(vectors: braggMask.vectors, scanRow: r, scanColumn: c, detectorRows: s.detectorRows,
                                 detectorColumns: s.detectorColumns, radius: braggMask.radius)
                for i in 0..<n where m[i] { sum[i] += Double(p[i]) * w; weights[i] += w }
            } else {
                for i in 0..<n { sum[i] += Double(p[i]) * w }
            }
        } }
        if braggMask == nil { return AveragedPattern(detectorRows: s.detectorRows, detectorColumns: s.detectorColumns, values: sum, valid: [Bool](repeating: true, count: n)) }
        let valid = weights.map { $0 > 0 }
        return AveragedPattern(detectorRows: s.detectorRows, detectorColumns: s.detectorColumns,
                               values: (0..<n).map { valid[$0] ? sum[$0] / weights[$0] : .nan }, valid: valid)
    }

    /// `make_bragg_mask(Qshape, g1, g2, radius, origin, max_q, return_sum=True, include_origin, rotation_deg)`: how many of the
    /// disks of radius `radius` around the lattice points `origin + h g1 + k g2` (every point shorter than `maximumDistance`
    /// from the origin, g1 and g2 rotated by `rotationDegrees`) cover each pixel; row-major. Vectors are (row, column).
    public static func latticeMask(detectorRows: Int, detectorColumns: Int, g1: (row: Double, column: Double),
                                   g2: (row: Double, column: Double), radius: Double, origin: (row: Double, column: Double),
                                   maximumDistance: Double, includeOrigin: Bool = true, rotationDegrees: Double = 0) throws -> [Int] {
        guard maximumDistance.isFinite, maximumDistance > 0, radius.isFinite, radius >= 0 else {
            throw CuboBraggError.invalidOption("a lattice mask needs a positive maximum distance and a non-negative radius")
        }
        let t = rotationDegrees * .pi / 180, cost = cos(t), sint = sin(t)
        // np.dot(g, [[cost, sint], [-sint, cost]])
        func rotate(_ g: (row: Double, column: Double)) -> (row: Double, column: Double) {
            (g.row * cost + g.column * -sint, g.row * sint + g.column * cost)
        }
        let a = rotate(g1), b = rotate(g2)
        let l1 = (a.row * a.row + a.column * a.column).squareRoot()
        let l2 = hypot(-b.row * a.column, b.column * a.row) / l1
        guard l1 > 0, l2 > 0, l1.isFinite, l2.isFinite else { throw CuboBraggError.invalidOption("the lattice vectors are degenerate") }
        let hf = maximumDistance / l1, kf = maximumDistance / l2
        guard hf < 1e4, kf < 1e4 else { throw CuboBraggError.invalidOption("the lattice vectors are too short for the maximum distance (more than 1e4 points per axis)") }
        let h = Int(hf) + 1, k = Int(kf) + 1
        var total = [Int](repeating: 0, count: detectorRows * detectorColumns)
        for hh in -h...h { for kk in -k...k {
            if hh == 0, kk == 0, !includeOrigin { continue }
            let v = (row: Double(hh) * a.row + Double(kk) * b.row, column: Double(hh) * a.column + Double(kk) * b.column)
            guard (v.row * v.row + v.column * v.column).squareRoot() < maximumDistance else { continue }
            let disk = try DetectorMask.make(.circle(centerRow: origin.row + v.row, centerColumn: origin.column + v.column, radius: radius),
                                             detectorRows: detectorRows, detectorColumns: detectorColumns)
            for i in disk.indices where disk[i] != 0 { total[i] += 1 }
        } }
        return total
    }
}
