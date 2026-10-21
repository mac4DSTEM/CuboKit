import Foundation

/// Re-expression of a strain tensor field in another frame (the scan frame, once the R–Q rotation is known).
public enum StrainFrame {
    /// Rotates a strain field from the detector frame into the scan frame: first the optional transpose (swap exx and
    /// eyy, negate theta), then the tensor rotation by `rotation` (radians), with c = cos, s = sin:
    /// exx' = c²·a − 2sc·exy + s²·b, eyy' = s²·a + 2sc·exy + c²·b, exy' = sc·(a − b) + (c² − s²)·exy, where (a, b) is
    /// (exx, eyy), or (eyy, exx) when transposed. Theta is unchanged by the rotation.
    ///
    /// Sign convention: `rotation` is the angle `a` of `LatticeStrain.strain(local:reference:coordinateRotation:)`, whose
    /// degrees `r` give `a = -r·π/180` (a counterclockwise `coordinateRotation` of r degrees is `rotation = -r·π/180`
    /// here), so for `transposed == false` rotating a `coordinateRotation == 0` strain by `rotation` equals asking for
    /// `coordinateRotation = -rotation·180/π`. Element-wise over arrays of equal length (a NaN stays NaN).
    /// Double; the same formulas as the Float32 original, relative agreement about 1e-6.
    public static func rotate(exx: [Double], eyy: [Double], exy: [Double], theta: [Double], rotation: Double, transposed: Bool)
        -> (exx: [Double], eyy: [Double], exy: [Double], theta: [Double]) {
        precondition(exx.count == eyy.count && exx.count == exy.count && exx.count == theta.count,
                     "StrainFrame.rotate: the four components must have the same length")
        let n = exx.count
        let c = cos(rotation), s = sin(rotation)
        let c2 = c * c, s2 = s * s, sc = s * c
        var oxx = [Double](repeating: 0, count: n), oyy = oxx, oxy = oxx
        for i in 0..<n {
            let a = transposed ? eyy[i] : exx[i], b = transposed ? exx[i] : eyy[i], d = exy[i]
            oxx[i] = c2 * a - 2 * sc * d + s2 * b
            oyy[i] = s2 * a + 2 * sc * d + c2 * b
            oxy[i] = sc * (a - b) + (c2 - s2) * d
        }
        return (oxx, oyy, oxy, transposed ? theta.map { -$0 } : theta)
    }
}
