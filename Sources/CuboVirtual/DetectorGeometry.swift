//
//  DetectorGeometry.swift
//  py4DSTEM's virtual detectors in detector pixels, from the corner, and their masks (registration 1.2, 3.1).
//
//  Ported from py4DSTEM f050d207, py4DSTEM/datacube/virtualimage.py, DataCubeVirtualImager.make_detector (line 588):
//      circle    (qx - x0)**2 + (qy - y0)**2 <  r**2
//      annulus   (qx - x0)**2 + (qy - y0)**2 >  r_in**2  and  < r_out**2   (r_out > r_in asserted)
//      rectangle mask[round(xmin):round(xmax), round(ymin):round(ymax)] = 1   (np.round: half to even)
//      point     mask[qx, qy] = 1
//      mask      the given array
//  py4DSTEM's qx (first detector axis) is this kit's row, its qy the column, so the tuples pass through unchanged.
//  Evaluated in Double, as NumPy does on np.indices minus a Python float; mac4DSTEM 7448401b
//  (Core/Analysis/VirtualDetector.swift, makeMask and fillRadial) evaluates in Float32, which differs from py4DSTEM on a
//  pixel whose distance lies within one Float32 ulp of the radius (the pins carry one).
//
//  DEVIATION from py4DSTEM, in the refusing direction only: a point off the grid, a negative rectangle bound, a negative
//  radius and a non-finite value are refused, where NumPy would wrap a negative index or square a negative radius.
//

import CuboCore

/// A virtual detector in detector pixels; rows and columns as `CubeShape` names them.
public enum DetectorGeometry: Sendable, Hashable {
    /// One detector pixel.
    case point(row: Int, column: Int)
    /// Pixels whose distance from the centre is strictly below `radius` (bright field, or dark field on one spot).
    case circle(centerRow: Double, centerColumn: Double, radius: Double)
    /// Pixels whose distance is strictly between `inner` and `outer` (annular dark field); `outer > inner`.
    case annulus(centerRow: Double, centerColumn: Double, inner: Double, outer: Double)
    /// Rows `[round(rowMin), round(rowMax))`, columns `[round(columnMin), round(columnMax))`, half to even, clipped.
    case rectangle(rowMin: Double, rowMax: Double, columnMin: Double, columnMax: Double)
    /// Any weights on the detector grid, row-major, `detectorRows · detectorColumns` of them.
    case mask([Float])
}

public enum DetectorMask {
    /// The mask of `geometry` on the detector grid of `shape`, row-major: 0 or 1, or the given weights.
    public static func make(_ geometry: DetectorGeometry, shape: CubeShape) throws -> [Float] {
        try make(geometry, detectorRows: shape.detectorRows, detectorColumns: shape.detectorColumns)
    }

    /// The mask of `geometry` on a `detectorRows × detectorColumns` grid, row-major.
    public static func make(_ geometry: DetectorGeometry, detectorRows qy: Int, detectorColumns qx: Int) throws
        -> [Float] {
        guard qy > 0, qx > 0 else { throw CuboError.invalidShape([qy, qx]) }
        var mask = [Float](repeating: 0, count: qy * qx)
        switch geometry {
        case .point(let row, let column):
            guard (0..<qy).contains(row), (0..<qx).contains(column) else {
                throw CuboVirtualError.pointOutsideDetector(row: row, column: column)
            }
            mask[row * qx + column] = 1
        case .circle(let cr, let cc, let r):
            guard cr.isFinite, cc.isFinite, r.isFinite else { throw CuboVirtualError.nonFiniteGeometry }
            guard r >= 0 else { throw CuboVirtualError.negativeRadius(r) }
            let r2 = r * r
            fill(&mask, qy: qy, qx: qx, cr: cr, cc: cc) { $0 < r2 }
        case .annulus(let cr, let cc, let inner, let outer):
            guard cr.isFinite, cc.isFinite, inner.isFinite, outer.isFinite else {
                throw CuboVirtualError.nonFiniteGeometry
            }
            guard inner >= 0 else { throw CuboVirtualError.negativeRadius(inner) }
            guard outer > inner else { throw CuboVirtualError.innerNotBelowOuter(inner: inner, outer: outer) }
            let in2 = inner * inner, out2 = outer * outer
            fill(&mask, qy: qy, qx: qx, cr: cr, cc: cc) { $0 > in2 && $0 < out2 }
        case .rectangle(let rowMin, let rowMax, let columnMin, let columnMax):
            let bounds = [rowMin, rowMax, columnMin, columnMax]
            guard bounds.allSatisfy(\.isFinite) else { throw CuboVirtualError.nonFiniteGeometry }
            let rounded = bounds.map { $0.rounded(.toNearestOrEven) }
            guard rounded.allSatisfy({ $0 >= 0 }) else { throw CuboVirtualError.negativeRectangleBound }
            // NumPy slicing: a stop past the end is clipped; a start at or past the stop selects nothing.
            let r0 = Int(min(rounded[0], Double(qy))), r1 = Int(min(rounded[1], Double(qy)))
            let c0 = Int(min(rounded[2], Double(qx))), c1 = Int(min(rounded[3], Double(qx)))
            if r0 < r1, c0 < c1 {
                for i in r0..<r1 { for j in c0..<c1 { mask[i * qx + j] = 1 } }
            }
        case .mask(let weights):
            guard weights.count == qy * qx else {
                throw CuboVirtualError.maskLengthMismatch(expected: qy * qx, received: weights.count)
            }
            guard weights.allSatisfy(\.isFinite) else { throw CuboVirtualError.nonFiniteGeometry }
            mask = weights
        }
        return mask
    }

    /// Sets the pixels whose squared distance, in Double and in py4DSTEM's order of operations, passes `predicate`.
    private static func fill(_ mask: inout [Float], qy: Int, qx: Int, cr: Double, cc: Double,
                             _ predicate: (Double) -> Bool) {
        for i in 0..<qy {
            let di = Double(i) - cr
            let di2 = di * di
            for j in 0..<qx {
                let dj = Double(j) - cc
                if predicate(di2 + dj * dj) { mask[i * qx + j] = 1 }
            }
        }
    }
}
