import Foundation
import Testing
import CuboPhase

private struct Pin: Decodable {
    let rows: Int, columns: Int, row: [Double], column: [Double], magnitude: [Double]
    let py4dstemAngle: [Double], rowOverColumnAngle: [Double], unwrappedPy4dstem: [Double]
}
private func pin() throws -> Pin {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/dpcviews.json")
    return try JSONDecoder().decode(Pin.self, from: Data(contentsOf: url))
}
private func field(_ p: Pin) throws -> ScanVectorField { try ScanVectorField(scanRows: p.rows, scanColumns: p.columns, row: p.row, column: p.column) }

@Suite struct DPCViewsTests {
    // NumPy's hypot and arctan2 in Double: tolerance 1e-15 relative (both libm); the wrapped angle's signs of zero are exact.
    @Test func magnitudeAndAngleAgainstNumPy() throws {
        let p = try pin(), f = try field(p)
        let m = DPCViews.magnitude(of: f)
        #expect(zip(m, p.magnitude).allSatisfy { abs($0 - $1) <= 1e-15 * max(1, abs($1)) })
        for (convention, expected) in [(DPCViews.AngleConvention.py4DSTEM, p.py4dstemAngle), (.rowOverColumn, p.rowOverColumnAngle)] {
            let a = DPCViews.angle(of: f, convention: convention)
            #expect(zip(a, expected).allSatisfy { abs($0 - $1) <= 1e-15 * max(1, abs($1)) })
            #expect(a.allSatisfy { $0 >= 0 && $0 < 2 * Double.pi && $0.sign == .plus })       // [0, 2 pi), never -0
        }
        let raw = DPCViews.angle(of: f, wrapped: false)
        #expect(zip(raw, p.unwrappedPy4dstem).allSatisfy { abs($0 - $1) <= 1e-15 })
        // a vector along -row with a signed-zero column is wrapped to pi, never -pi
        #expect(DPCViews.angle(of: try ScanVectorField(scanRows: 1, scanColumns: 1, row: [-1], column: [-0.0]))[0] == Double.pi)
    }

    @Test func milliradiansPerPixelIsTheSmallAngleLaw() {
        #expect(DPCViews.milliradiansPerPixel(wavelength: 0.0251, reciprocalSampling: 0.02) == 1000 * 0.0251 * 0.02)
        #expect(DPCViews.milliradiansPerPixel(wavelength: 0, reciprocalSampling: 0.02) == nil)
        #expect(DPCViews.milliradiansPerPixel(wavelength: 0.0251, reciprocalSampling: .nan) == nil)
    }

    // The convention made executable: a field in the image frame (x along the columns, y along the rows) rotated by the image-frame
    // formula x' = cos a - sin b, y' = sin a + cos b equals the kit's solver forced at the NEGATIVE angle (py4DSTEM's sign).
    @Test func imageFrameRotationEqualsTheKitAtTheNegatedAngle() throws {
        let rows = 4, columns = 5, n = rows * columns
        let cx = (0..<n).map { sin(Double($0) * 0.7) * 2 + 0.3 }, cy = (0..<n).map { cos(Double($0) * 0.45) - 0.2 }
        let f = try ScanVectorField(scanRows: rows, scanColumns: columns, row: cy, column: cx)
        for degrees in [-37.5, 0.0, 12.0, 80.0] {
            let theta = degrees * Double.pi / 180
            for transpose in [false, true] {
                let solved = try RotationSolver.solve(f, forceRotationDegrees: RotationConvention.kit(fromImageFrame: theta) * 180 / .pi,
                                                       forceTranspose: transpose)
                for i in 0..<n {
                    let a = transpose ? cy[i] : cx[i], b = transpose ? cx[i] : cy[i]
                    let x = cos(theta) * a - sin(theta) * b, y = sin(theta) * a + cos(theta) * b
                    #expect(abs(solved.corrected.column[i] - x) < 1e-13 && abs(solved.corrected.row[i] - y) < 1e-13)
                }
            }
        }
        #expect(RotationConvention.imageFrame(fromKit: RotationConvention.kit(fromImageFrame: 0.3)) == 0.3)
    }

    @Test func degreesTextNeverPrintsMinusZero() {
        #expect(RotationConvention.degreesText(-0.04) == "0.0°")
        #expect(RotationConvention.degreesText(-80.06) == "-80.1°")
        #expect(RotationConvention.degreesText(.nan) == "—")
        #expect(RotationConvention.degreesText(12.345, decimals: 2) == "12.35°" || RotationConvention.degreesText(12.345, decimals: 2) == "12.34°")
    }
}
