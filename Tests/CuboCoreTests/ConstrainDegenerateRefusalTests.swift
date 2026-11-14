import Foundation
import Testing
import CuboCore

// CuboKit#92 item 6: `constrainDegenerate` on an annulus without four spots gave NaN semi-axes; it throws `notFinite`, as the
// kit's other ellipse fits do.

@Suite struct ConstrainDegenerateRefusalTests {
    /// An empty Bragg-vector map: every centre of mass is 0/0. Mutation: drop the guard - (NaN, NaN) comes back, red.
    @Test func anEmptyAnnulusIsRefused() throws {
        let plane = try EllipseFitPlane(rows: 48, columns: 52, values: [Double](repeating: 0, count: 48 * 52))
        let ellipse = EllipseParameters(centerRow: 24, centerColumn: 26, a: 14, b: 10, theta: 0.5)
        #expect(throws: EllipseFitError.notFinite) {
            _ = try EllipseFit.constrainDegenerate(plane, ellipse: ellipse, innerRadius: 5, outerRadius: 20, phiKnown: 1)
        }
    }
}
