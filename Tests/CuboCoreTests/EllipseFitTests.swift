//
//  EllipseFitTests.swift
//  The ellipse fits (task 18) against py4DSTEM 0.14.19 (tools/polar-ellipse/pins.py) and against the planted ellipses of
//  tools/polar-ellipse/planted.py. py4DSTEM has no test for these functions: the direct evaluations are held to
//  `assert_allclose`'s Float64 default (rtol 1e-7); an iterative fit is compared with py4DSTEM's result at the convergence
//  tolerance of the optimiser both run (`leastsq`: xtol = 1.49012e-8, relative to the parameter, floor 1), and with the planted
//  truth at the stated truth tolerances (the algebraic 1D fit is biased by about 1% in the semi-axes by construction).
//

import Foundation
import Testing
import CuboCore

private struct EllipsePin: Decodable { let r0: Double, c0: Double, a: Double, b: Double, theta: Double }
private struct AmorphousPin: Decodable { let ellipse: [Double], p: [Double] }
private struct AmorphousP0Pin: Decodable { let p0: [Double?], ellipse: [Double], p: [Double] }
private struct FitPin: Decodable {
    let name: String, truthPy4DSTEM: EllipsePin, shape: [Int], ring: [Double], full: [Double]
    let centerGuess: [Double], fitRadii: [Double], ignoreMask: [Int], keepMask: [Int]
    let fit1D: [Double], fit1DMasked: [Double], fit1DDefaults: [Double]
    let amorphous: AmorphousPin, amorphousMasked: AmorphousPin, amorphousP0: AmorphousP0Pin
}
private struct ConvertPin: Decodable { let abc: [Double], aba: [Double] }
private struct ConvertRPin: Decodable { let aba: [Double], abc: [Double] }
private struct FunctionsPin: Decodable {
    let points: [Double], values: [Double], pEllipse: [Double], pGaussian: [Double]
    let ellipseErr: [Double], doubleSidedGaussian: [Double], doubleSidedGaussianFitErr: [Double]
    let convert: [ConvertPin], convertR: [ConvertRPin]
}
private struct ConstrainCase: Decodable {
    let pEllipse: [Double], rInner: Double, rOuter: Double, phiKnown: Double, fitRad: Double, aConstrained: Double, bConstrained: Double
}
private struct ConstrainPin: Decodable { let shape: [Int], data: [Double], cases: [ConstrainCase] }
private struct Pin: Decodable { let fitImageShape: [Int], fits: [FitPin], functions: FunctionsPin, constrain: ConstrainPin }

private let pin: Pin = {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/ellipse-resample-pins.json")
    do { return try JSONDecoder().decode(Pin.self, from: Data(contentsOf: url)) } catch { fatalError("ellipse-resample-pins.json: \(error)") }
}()

private func agrees(_ a: Double, _ b: Double) -> Bool { abs(a - b) <= 1e-12 + 1e-7 * abs(b) }
/// The optimiser's own tolerance on a fitted number (xtol = 1.49012e-8, relative to the value, floor 1).
private func withinFitTolerance(_ a: Double, _ b: Double) -> Bool { abs(a - b) <= 1.49012e-8 * max(abs(b), 1) }
/// The difference of two ellipse angles, which are the same ellipse modulo pi.
private func angleDifference(_ a: Double, _ b: Double) -> Double {
    var d = (a - b).truncatingRemainder(dividingBy: .pi)
    if d > .pi / 2 { d -= .pi } else if d < -.pi / 2 { d += .pi }
    return abs(d)
}
private func plane(_ f: FitPin, _ values: [Double]) throws -> EllipseFitPlane {
    try EllipseFitPlane(rows: f.shape[0], columns: f.shape[1], values: values)
}
private func ellipse(_ v: [Double]) -> EllipseParameters { EllipseParameters(centerRow: v[0], centerColumn: v[1], a: v[2], b: v[3], theta: v[4]) }

@Suite struct EllipseConversionAndFunctionTests {
    @Test func testConversionsAndFunctionsAgreeWithPy4DSTEM() throws {
        for c in pin.functions.convert {
            let got = EllipseFit.semiAxes(A: c.abc[0], B: c.abc[1], C: c.abc[2])
            #expect(agrees(got.a, c.aba[0]) && agrees(got.b, c.aba[1]) && agrees(got.theta, c.aba[2]), "convert \(c.abc)")
        }
        for c in pin.functions.convertR {
            let got = EllipseFit.canonical(a: c.aba[0], b: c.aba[1], theta: c.aba[2])
            #expect(agrees(got.A, c.abc[0]) && agrees(got.B, c.abc[1]) && agrees(got.C, c.abc[2]), "convertR \(c.aba)")
        }
        let f = pin.functions
        let xs = stride(from: 0, to: f.points.count, by: 2).map { f.points[$0] }
        let ys = stride(from: 1, to: f.points.count, by: 2).map { f.points[$0] }
        let err = try EllipseFit.ellipseError(parameters: f.pEllipse, rows: xs, columns: ys, values: f.values)
        #expect(zip(err, f.ellipseErr).allSatisfy { agrees($0, $1) })
        let g = try EllipseFit.doubleSidedGaussian(parameters: f.pGaussian, rows: xs, columns: ys)
        #expect(zip(g, f.doubleSidedGaussian).allSatisfy { agrees($0, $1) })
        let ge = try EllipseFit.doubleSidedGaussianFitError(parameters: f.pGaussian, rows: xs, columns: ys, values: f.values)
        #expect(zip(ge, f.doubleSidedGaussianFitErr).allSatisfy { agrees($0, $1) })
    }

    /// The two conversions are inverse to each other (a round trip through the canonical form).
    @Test func testConversionRoundTrip() {
        for (a, b, t) in [(14.0, 10.0, 0.9), (12.0, 9.0, -1.2), (15.0, 12.0, 0.0), (20.0, 5.0, 1.4)] {
            let abc = EllipseFit.canonical(a: a, b: b, theta: t)
            let back = EllipseFit.semiAxes(A: abc.A, B: abc.B, C: abc.C)
            #expect(abs(back.a - a) < 1e-12 && abs(back.b - b) < 1e-12 && angleDifference(back.theta, t) < 1e-12)
        }
    }

    @Test func testParametersFeedTheCalibration() {
        let e = EllipseParameters(centerRow: 1, centerColumn: 2, a: 14, b: 10, theta: 0.5)
        var cal = Calibration()
        cal.ellipse = e.calibration
        #expect(cal.ellipse == EllipseCalibration(a: 14, b: 10, theta: 0.5))
    }
}

@Suite struct EllipseFitPinTests {
    /// Agreement of two fits of the same problem. py4DSTEM's own result is determined only to the level at which MINPACK's path
    /// is insensitive to rounding: restarting `leastsq` from the same point scaled by 1 + 1e-15 moves the amorphous-ring ellipse
    /// by 1e-7 (tilted), 2e-7 (flat) and 1e-12 (near circle), and by 7e-4 where the path bifurcates (one ulp, tilted). So the
    /// centre and semi-axes are compared to `relative`, the canonical form (A, B, C), which stays defined where theta does not
    /// (a circle, B = 0), to 1e-5 of its largest element (see below), and theta to 1e-5 rad where a - b exceeds 10% of a.
    private func check(_ got: EllipseFitResult, _ want: [Double], relative: Double, _ label: String) {
        let e = got.ellipse
        let v = [e.centerRow, e.centerColumn, e.a, e.b, e.theta]
        for k in 0..<4 { #expect(abs(v[k] - want[k]) <= relative * max(abs(want[k]), 1), "\(label) element \(k): \(v[k]) vs \(want[k])") }
        let g = e.canonical, w = EllipseFit.canonical(a: want[2], b: want[3], theta: want[4])
        let scale = max(abs(w.A), abs(w.B), abs(w.C))
        // a poorly determined direction (B of an untilted ellipse is zero in truth) is fixed by `leastsq` only to about
        // sqrt(ftol) = 1.2e-4 of the scale; 1e-5 is held
        #expect(abs(g.A - w.A) <= 1e-5 * scale && abs(g.B - w.B) <= 1e-5 * scale && abs(g.C - w.C) <= 1e-5 * scale,
                "\(label) canonical form")
        if (want[2] - want[3]) > 0.1 * want[2] { #expect(angleDifference(v[4], want[4]) <= 1e-5, "\(label) theta \(v[4]) vs \(want[4])") }
        #expect(got.converged, "\(label) info \(got.info)")
    }

    /// The sum of squares of the double-sided Gaussian's residuals over the annulus: both fits must be minima of it.
    private func cost(_ p: [Double], _ data: EllipseFitPlane, center: (Double, Double), radii: (Double, Double)) throws -> Double {
        var xs = [Double](), ys = [Double](), vs = [Double]()
        for r in 0..<data.rows { for c in 0..<data.columns {
            let rr = hypot(Double(r) - center.0, Double(c) - center.1)
            guard rr > radii.0, rr < radii.1 else { continue }
            xs.append(Double(r)); ys.append(Double(c)); vs.append(data[r, c])
        } }
        return try EllipseFit.doubleSidedGaussianFitError(parameters: p, rows: xs, columns: ys, values: vs).reduce(0) { $0 + $1 * $1 }
    }

    @Test func testFit1DAgreesWithPy4DSTEM() throws {
        var worst = 0.0
        for f in pin.fits {
            let center = (f.centerGuess[0], f.centerGuess[1]), radii = (f.fitRadii[0], f.fitRadii[1])
            let ring = try plane(f, f.ring)
            let got = try EllipseFit.fit1D(ring, center: center, fitRadii: radii)
            check(got, f.fit1D, relative: 1e-7, "\(f.name) fit1D")
            print("\(f.name) fit1D: \(got.evaluations) evaluations, info \(got.info)")
            let masked = try EllipseFit.fit1D(ring, center: center, fitRadii: radii, ignoring: f.ignoreMask.map { $0 != 0 })
            check(masked, f.fit1DMasked, relative: 1e-7, "\(f.name) fit1D masked")
            let defaults = try EllipseFit.fit1D(ring, fitRadii: (0, Double(min(f.shape[0], f.shape[1])) / 2))
            check(defaults, f.fit1DDefaults, relative: 1e-7, "\(f.name) fit1D default centre")
            // the default radii (py4DSTEM's raise a TypeError): the intended (0, min(shape) / 2) gives the same fit
            let intended = try EllipseFit.fit1D(ring)
            check(intended, f.fit1DDefaults, relative: 1e-7, "\(f.name) fit1D default radii")
            let w = f.fit1D
            let e = got.ellipse
            worst = max(worst, abs(e.a - w[2]) / w[2], abs(e.b - w[3]) / w[3], abs(e.centerRow - w[0]) / w[0])
        }
        print("fit_ellipse_1D vs py4DSTEM: worst relative difference \(worst)")
    }

    @Test func testAmorphousRingAgreesWithPy4DSTEM() throws {
        var worstCostGap = 0.0
        for f in pin.fits {
            let data = try plane(f, f.full)
            let center = (f.centerGuess[0], f.centerGuess[1])
            let radii = (f.truthPy4DSTEM.b * 0.75, f.truthPy4DSTEM.a * 1.25)
            let got = try EllipseFit.fitAmorphousRing(data, center: center, fitRadii: radii)
            check(got, f.amorphous.ellipse, relative: 1e-6, "\(f.name) amorphous")
            print("\(f.name) amorphous: \(got.evaluations) evaluations, info \(got.info)")
            // both are minima of the same sum of squares, to the optimiser's ftol
            let mine = try cost(got.parameters, data, center: center, radii: radii), theirs = try cost(f.amorphous.p, data, center: center, radii: radii)
            worstCostGap = max(worstCostGap, (mine - theirs) / theirs)
            let w = f.amorphous.ellipse, e = got.ellipse
            print("\(f.name) amorphous vs py4DSTEM: centre/axes relative \(max(abs(e.centerRow - w[0]) / w[0], abs(e.centerColumn - w[1]) / w[1], abs(e.a - w[2]) / w[2], abs(e.b - w[3]) / w[3])), theta \(angleDifference(e.theta, w[4])) rad")
            #expect(mine <= theirs * (1 + 1e-6), "\(f.name) cost \(mine) vs py4DSTEM \(theirs)")
            let masked = try EllipseFit.fitAmorphousRing(data, center: center, fitRadii: radii, keeping: f.keepMask.map { $0 != 0 })
            check(masked, f.amorphousMasked.ellipse, relative: 1e-6, "\(f.name) amorphous masked")
            let withP0 = try EllipseFit.fitAmorphousRing(data, center: center, fitRadii: radii, initial: f.amorphousP0.p0)
            check(withP0, f.amorphousP0.ellipse, relative: 1e-6, "\(f.name) amorphous with p0")
        }
        print("fit_ellipse_amorphous_ring: the kit's residual sum of squares against py4DSTEM's, worst relative gap \(worstCostGap)")
    }

    /// The fit recovers the planted ellipse to the fit's own accuracy: the amorphous-ring model to 0.02 px in the semi-axes and
    /// 5e-3 px in the centre; the algebraic 1D fit of a 2 px wide ring to 1.5% in the semi-axes (it weights the pixels of the
    /// annulus by their value, which biases the radius upwards), 5e-3 px in the centre, 2e-3 rad in the tilt.
    @Test func testFitsRecoverThePlantedEllipse() throws {
        for f in pin.fits {
            let t = f.truthPy4DSTEM
            let one = try EllipseFit.fit1D(try plane(f, f.ring), center: (f.centerGuess[0], f.centerGuess[1]), fitRadii: (f.fitRadii[0], f.fitRadii[1])).ellipse
            let am = try EllipseFit.fitAmorphousRing(try plane(f, f.full), center: (f.centerGuess[0], f.centerGuess[1]),
                                                     fitRadii: (t.b * 0.75, t.a * 1.25)).ellipse
            print("\(f.name): 1D a \(one.a) b \(one.b) (truth \(t.a), \(t.b)); amorphous a \(am.a) b \(am.b), centre \(am.centerRow - t.r0), \(am.centerColumn - t.c0), theta \(angleDifference(am.theta, t.theta))")
            #expect(abs(one.centerRow - t.r0) < 5e-3 && abs(one.centerColumn - t.c0) < 5e-3, "\(f.name) 1D centre")
            #expect(abs(one.a - t.a) < 0.015 * t.a && abs(one.b - t.b) < 0.015 * t.b, "\(f.name) 1D axes")
            #expect(angleDifference(one.theta, t.theta) < 2e-3, "\(f.name) 1D theta")
            #expect(abs(am.centerRow - t.r0) < 5e-3 && abs(am.centerColumn - t.c0) < 5e-3, "\(f.name) amorphous centre")
            #expect(abs(am.a - t.a) < 0.02 && abs(am.b - t.b) < 0.02, "\(f.name) amorphous axes")
            #expect(angleDifference(am.theta, t.theta) < 1e-3, "\(f.name) amorphous theta")
            // a circle assumption would be wrong by the difference of the axes
            if t.a - t.b > 1 { #expect(one.a - one.b > 0.5 * (t.a - t.b)) }
        }
    }

    @Test func testConstrainDegenerateEllipseAgreesWithPy4DSTEM() throws {
        let c = pin.constrain
        let data = try EllipseFitPlane(rows: c.shape[0], columns: c.shape[1], values: c.data)
        for k in c.cases {
            let p = k.pEllipse
            let got = try EllipseFit.constrainDegenerate(data, ellipse: ellipse(p), innerRadius: k.rInner, outerRadius: k.rOuter,
                                                         phiKnown: k.phiKnown, fitRadius: k.fitRad)
            print("constrain_degenerate_ellipse: a \(got.a) vs \(k.aConstrained), b \(got.b) vs \(k.bConstrained)")
            #expect(agrees(got.a, k.aConstrained) && agrees(got.b, k.bConstrained), "phi \(k.phiKnown)")
        }
    }

    @Test func testRefusalsAndCancellation() throws {
        let f = pin.fits[0]
        let ring = try plane(f, f.ring)
        #expect(throws: EllipseFitError.self) { _ = try EllipseFitPlane(rows: 2, columns: 2, values: [1]) }
        #expect(throws: EllipseFitError.self) { _ = try EllipseFit.fit1D(ring, fitRadii: (100, 200)) }                       // no pixel in the annulus
        #expect(throws: EllipseFitError.self) { _ = try EllipseFit.fit1D(ring, ignoring: [true]) }                            // wrong mask size
        #expect(throws: EllipseFitError.self) { _ = try EllipseFit.fit1D(ring, center: (.nan, 1)) }
        #expect(throws: EllipseFitError.self) { _ = try EllipseFit.fitAmorphousRing(ring, center: (24, 26), fitRadii: (10, 20), initial: [1]) }
        #expect(throws: EllipseFitError.self) { _ = try EllipseFit.fitAmorphousRing(ring, center: (24, 26), fitRadii: (200, 300)) }
        #expect(throws: EllipseFitError.self) { _ = try EllipseFit.fitAmorphousRing(ring, center: (24, 26), fitRadii: (0.2, 0.4)) }   // no radial bin
        #expect(throws: CancellationError.self) { _ = try EllipseFit.fit1D(ring, center: (24, 26), fitRadii: (5, 20), isCancelled: { true }) }
        #expect(throws: CancellationError.self) {
            _ = try EllipseFit.fitAmorphousRing(ring, center: (24, 26), fitRadii: (5, 20), isCancelled: { true })
        }
        #expect(throws: CancellationError.self) {
            _ = try EllipseFit.constrainDegenerate(ring, ellipse: ellipse([24, 26, 14, 10, 0.5]), innerRadius: 5, outerRadius: 20, phiKnown: 1,
                                                   isCancelled: { true })
        }
    }
}

@Suite struct MINPACKTests {
    /// Fitting an exponential to exact data from a far start: the optimiser converges to the truth (info 1 to 4).
    @Test func testExponentialFitConverges() throws {
        let t = (0..<12).map { Double($0) * 0.5 }
        let y = t.map { 3.5 * exp(-0.7 * $0) + 1.25 }
        let r = try MINPACKLeastSquares.leastSquares([1, 0.1, 0]) { p in zip(t, y).map { p[0] * exp(-p[1] * $0) + p[2] - $1 } }
        #expect(r.converged)
        #expect(abs(r.x[0] - 3.5) < 1e-7 && abs(r.x[1] - 0.7) < 1e-7 && abs(r.x[2] - 1.25) < 1e-7, "\(r.x)")
    }

    /// Rosenbrock's residuals: the classic MINPACK start (-1.2, 1) reaches (1, 1).
    @Test func testRosenbrock() throws {
        let r = try MINPACKLeastSquares.leastSquares([-1.2, 1]) { p in [10 * (p[1] - p[0] * p[0]), 1 - p[0]] }
        #expect(r.converged && abs(r.x[0] - 1) < 1e-7 && abs(r.x[1] - 1) < 1e-7, "\(r.x)")
    }

    @Test func testFewerResidualsThanParametersThrows() {
        #expect(throws: MINPACKLeastSquares.Failure.fewerResidualsThanParameters(residuals: 1, parameters: 2)) {
            _ = try MINPACKLeastSquares.leastSquares([0, 0]) { _ in [1.0] }
        }
    }
}
