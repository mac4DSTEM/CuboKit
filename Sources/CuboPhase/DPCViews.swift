//
//  DPCViews.swift
//  The scalar views of a centre-of-mass (DPC) vector field and the sign convention of the R-Q rotation.
//
//  Magnitude and angle follow py4DSTEM's complex centre of mass (`show_complex_CoM`: `com_x + 1j * com_y`, x the first
//  detector axis, which is this kit's row): magnitude `hypot`, angle `np.angle`. The angle is wrapped to [0, 2 pi) with a
//  canonical +0 (never -0, never 2 pi).
//

import CuboCore
import Foundation

/// Magnitude and direction images of a ``ScanVectorField``, in Double.
public enum DPCViews {
    /// Which component pair the direction is measured from.
    public enum AngleConvention: Sendable, Hashable {
        /// py4DSTEM's complex CoM `row + i column`: `atan2(column, row)`.
        case py4DSTEM
        /// `atan2(row, column)`: the direction from the column axis towards the row axis (an image-frame reading;
        /// mac4DSTEM's `atan2(cy, cx)` with `cx` along the detector columns).
        case rowOverColumn
    }

    /// `hypot(row, column)` per scan position, row-major (NumPy `hypot`).
    public static func magnitude(of field: ScanVectorField) -> [Double] {
        zip(field.row, field.column).map { hypot($0, $1) }
    }

    /// The direction per scan position in radians. `wrapped` gives [0, 2 pi) with +0 canonical, the zero vector 0;
    /// otherwise `atan2`'s own (-pi, pi].
    public static func angle(of field: ScanVectorField, convention: AngleConvention = .py4DSTEM,
                             wrapped: Bool = true) -> [Double] {
        (0..<field.row.count).map { i in
            var a = convention == .py4DSTEM ? atan2(field.column[i], field.row[i]) : atan2(field.row[i], field.column[i])
            if wrapped {
                if a < 0 { a += 2 * Double.pi }
                if a == 0 { a = 0 }
            }
            return a
        }
    }

    /// Small-angle conversion `theta = lambda q`: milliradians per detector pixel from the wavelength (Å) and the
    /// reciprocal sampling (Å^-1 per detector pixel). `nil` unless both are positive and finite.
    public static func milliradiansPerPixel(wavelength: Double, reciprocalSampling: Double) -> Double? {
        guard wavelength.isFinite, wavelength > 0, reciprocalSampling.isFinite, reciprocalSampling > 0 else { return nil }
        return 1000 * wavelength * reciprocalSampling
    }
}

/// The sign of the R-Q rotation between conventions. The kit's rotation is py4DSTEM's (`QR_rotation`, the angle
/// ``RotationSolver`` returns and a forced angle expects). A caller that reads a file's axes as (scan row, scan
/// column, detector row, detector column) with its image-frame x along the columns swaps both axis pairs, which turns
/// a rotation R(theta) into R(-theta): its angle is the negative of py4DSTEM's on the same data; the transpose flag is the same.
public enum RotationConvention {
    /// py4DSTEM's angle (radians) for the image-frame caller's.
    public static func kit(fromImageFrame radians: Double) -> Double { -radians }
    /// The image-frame caller's angle (radians) for py4DSTEM's.
    public static func imageFrame(fromKit radians: Double) -> Double { -radians }

    /// A rotation in degrees as printed: a value that rounds to zero at `decimals` prints "0.0", never "-0.0"; non-finite
    /// input prints "—", never a number.
    public static func degreesText(_ degrees: Double, decimals: Int = 1) -> String {
        guard degrees.isFinite else { return "—" }
        let text = String(format: "%.\(decimals)f", degrees)
        let isZero = !text.contains { "123456789".contains($0) }
        return (isZero ? String(format: "%.\(decimals)f", 0.0) : text) + "°"
    }
}
