//
//  RotationSolver.swift
//  The relative rotation (and transpose) between the scan and the detector, by minimising the curl of the
//  centre-of-mass field or maximising its divergence: a port of py4DSTEM's
//  `_solve_for_center_of_mass_relative_rotation` (process/phase/phase_base_class.py 769 at f050d207), in Double.
//

import CuboCore
import Foundation

public enum RotationCriterion: Sendable, Hashable {
    /// py4DSTEM's default: the curl of a gradient field vanishes at the right rotation.
    case minimizeCurl
    case maximizeDivergence
}

public struct RotationSolution: Sendable, Equatable {
    /// The chosen angle, from the list searched (or the forced one), in degrees and radians.
    public let degrees: Double
    public let radians: Double
    /// Whether the detector components are swapped before the rotation.
    public let transposed: Bool
    /// The angles searched and the criterion for each orientation that was evaluated (empty when not evaluated).
    public let anglesDegrees: [Double]
    public let curve: [Double]
    public let curveTransposed: [Double]
    /// The normalized field rotated (and transposed) at the solution: py4DSTEM's `com_x`, `com_y`.
    public let corrected: ScanVectorField
}

public enum RotationSolver {
    /// py4DSTEM's default search: −89 … 89 degrees in steps of 1.
    public static let defaultAnglesDegrees: [Double] = (0..<179).map { Double($0) - 89 }

    /// The criterion at each angle for one orientation, exactly as `solve` computes it (its `curve` for
    /// `transpose: false`, its `curveTransposed` for `true`), the angles evaluated in parallel. Throws
    /// `CancellationError` when `isCancelled` returns true (asked per angle); never a partial curve.
    public static func curve(_ normalized: ScanVectorField, anglesDegrees: [Double],
                             criterion: RotationCriterion = .minimizeCurl, transpose: Bool,
                             isCancelled: (@Sendable () -> Bool)? = nil) throws -> [Double] {
        try requireUsable(normalized)
        let angles = anglesDegrees.filter(\.isFinite)
        guard !angles.isEmpty, angles.count == anglesDegrees.count else { throw CuboPhaseError.invalidAngles }
        return try scores(normalized, angles, [transpose], maximize: criterion == .maximizeDivergence,
                          isCancelled: isCancelled)[0]
    }

    public static func solve(_ normalized: ScanVectorField, anglesDegrees: [Double] = defaultAnglesDegrees,
                             criterion: RotationCriterion = .minimizeCurl, forceRotationDegrees: Double? = nil,
                             forceTranspose: Bool? = nil, isCancelled: (@Sendable () -> Bool)? = nil) throws -> RotationSolution {
        try requireUsable(normalized)
        let R = normalized.scanRows, C = normalized.scanColumns
        let maximize = criterion == .maximizeDivergence
        func score(_ deg: Double, _ transpose: Bool) -> Double {
            scoreAt(normalized, deg, transpose, maximize: maximize)
        }
        func better(_ a: Double, than b: Double) -> Bool { maximize ? a > b : a < b }
        func best(_ curve: [Double]) -> Int {
            var k = 0
            for i in curve.indices where better(curve[i], than: curve[k]) { k = i }
            return k
        }
        if isCancelled?() == true { throw CancellationError() }
        var degrees: Double, transposed: Bool
        var angles: [Double] = [], plain: [Double] = [], trans: [Double] = []
        if let forced = forceRotationDegrees {
            guard forced.isFinite else { throw CuboPhaseError.invalidAngles }
            degrees = forced
            if let t = forceTranspose {
                transposed = t
            } else {
                let p = score(forced, false), q = score(forced, true)
                angles = [forced]; plain = [p]; trans = [q]
                transposed = better(q, than: p)
            }
        } else {
            angles = anglesDegrees.filter(\.isFinite)
            guard !angles.isEmpty, angles.count == anglesDegrees.count else { throw CuboPhaseError.invalidAngles }
            if let t = forceTranspose {
                transposed = t
                let curve = try scores(normalized, angles, [t], maximize: maximize, isCancelled: isCancelled)[0]
                if t { trans = curve } else { plain = curve }
                degrees = angles[best(curve)]
            } else {
                let both = try scores(normalized, angles, [false, true], maximize: maximize, isCancelled: isCancelled)
                plain = both[0]; trans = both[1]
                let i = best(plain), j = best(trans)
                // py4DSTEM keeps the plain orientation on a tie (≤ for the curl, ≥ for the divergence).
                transposed = better(trans[j], than: plain[i])
                degrees = angles[transposed ? j : i]
            }
        }
        let radians = degrees * (.pi / 180)    // NumPy's deg2rad and Python's math.radians: deg · (π/180)
        let (x, y) = rotate(normalized, radians: radians, transpose: transposed)
        return RotationSolution(degrees: degrees, radians: radians, transposed: transposed, anglesDegrees: angles,
                                curve: plain, curveTransposed: trans,
                                corrected: try ScanVectorField(scanRows: R, scanColumns: C, row: x, column: y))
    }

    /// The criterion curve over the angles for each requested orientation, every (orientation, angle) on its own core;
    /// each value is `scoreAt`, so the curve is bit identical to the sequential one.
    private static func scores(_ normalized: ScanVectorField, _ angles: [Double], _ transposes: [Bool], maximize: Bool,
                               isCancelled: (@Sendable () -> Bool)?) throws -> [[Double]] {
        let n = angles.count
        let flat = try parallelScores(n * transposes.count, isCancelled: isCancelled) { (k: Int) -> Double in
            scoreAt(normalized, angles[k % n], transposes[k / n], maximize: maximize)
        }
        return transposes.indices.map { (o: Int) -> [Double] in Array(flat[(o * n)..<((o + 1) * n)]) }
    }

    /// The criterion at one angle and orientation.
    private static func scoreAt(_ normalized: ScanVectorField, _ deg: Double, _ transpose: Bool, maximize: Bool) -> Double {
        let R = normalized.scanRows, C = normalized.scanColumns
        let (x, y) = rotate(normalized, radians: deg * (.pi / 180), transpose: transpose)
        return maximize ? divergence(x, y, R, C) : curl(x, y, R, C)
    }

    /// The scan must be at least 3 × 3 and the field finite.
    private static func requireUsable(_ normalized: ScanVectorField) throws {
        let R = normalized.scanRows, C = normalized.scanColumns
        guard R >= 3, C >= 3 else { throw CuboPhaseError.scanTooSmallForRotation(rows: R, columns: C) }
        guard normalized.row.allSatisfy(\.isFinite), normalized.column.allSatisfy(\.isFinite) else {
            throw CuboPhaseError.invalidField("the rotation search needs a finite field (normalize applies nan_to_num)")
        }
    }

    /// x' = cos·x − sin·y, y' = sin·x + cos·y; transposed, the components swapped first.
    static func rotate(_ f: ScanVectorField, radians: Double, transpose: Bool) -> ([Double], [Double]) {
        let c = cos(radians), s = sin(radians)
        let a = transpose ? f.column : f.row, b = transpose ? f.row : f.column
        return (zip(a, b).map { c * $0 - s * $1 }, zip(a, b).map { s * $0 + c * $1 })
    }

    /// mean |(y[r+1,c] − y[r−1,c]) − (x[r,c+1] − x[r,c−1])| over the interior.
    static func curl(_ x: [Double], _ y: [Double], _ R: Int, _ C: Int) -> Double {
        var v = [Double](); v.reserveCapacity((R - 2) * (C - 2))
        for r in 1..<(R - 1) { for c in 1..<(C - 1) {
            v.append(abs((y[(r + 1) * C + c] - y[(r - 1) * C + c]) - (x[r * C + c + 1] - x[r * C + c - 1])))
        } }
        return NumPySum.mean(v)
    }

    /// mean |(x[r+1,c] − x[r−1,c]) + (y[r,c+1] − y[r,c−1])| over the interior.
    static func divergence(_ x: [Double], _ y: [Double], _ R: Int, _ C: Int) -> Double {
        var v = [Double](); v.reserveCapacity((R - 2) * (C - 2))
        for r in 1..<(R - 1) { for c in 1..<(C - 1) {
            v.append(abs((x[(r + 1) * C + c] - x[(r - 1) * C + c]) + (y[r * C + c + 1] - y[r * C + c - 1])))
        } }
        return NumPySum.mean(v)
    }
}
