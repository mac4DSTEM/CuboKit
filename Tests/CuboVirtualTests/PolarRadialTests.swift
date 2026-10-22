//
//  PolarRadialTests.swift
//  The polar data cube and the radial background (task 17) against py4DSTEM 0.14.19 (tools/polar-ellipse/pins.py, run on
//  planted cubes) and against the planted truth of tools/polar-ellipse/planted.py. No upstream test exists for these functions, so
//  the tolerance is `assert_allclose`'s Float64 default (rtol 1e-7) with atol 1e-12 (the rule of CalibrationTests); the port runs
//  the same operations in the same order and the measured difference is printed.
//

import Foundation
import Testing
import CuboKitTestSupport
import CuboCore
import CuboVirtual

private struct PolarPin: Decodable { let row: Int, column: Int, values: [Double?], mask: [Int], norm: [Double] }
private struct PolarCase: Decodable {
    let name: String, ellipse: [Double]?
    let params: [String: Param]
    let polarShape: [Int], radialBins: [Double], annularBins: [Double], sigmaKDE: [Double]
    let patterns: [PolarPin]
}
private enum Param: Decodable {
    case number(Double), flag(Bool), text(String)
    init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let b = try? c.decode(Bool.self) { self = .flag(b) } else if let d = try? c.decode(Double.self) { self = .number(d) } else { self = .text(try c.decode(String.self)) }
    }
    var double: Double? { if case .number(let d) = self { d } else { nil } }
    var bool: Bool? { if case .flag(let b) = self { b } else { nil } }
}
private struct SigmaCase: Decodable { let row: Int, column: Int, sigma: Double, background: [Double], bksb: [Double] }
private struct LocalCase: Decodable { let row: Int, column: Int, sigma: Double, values: [Double] }
private struct NoEllipse: Decodable { let row: Int, column: Int, sigma: Double, background: [Double] }
private struct Radial: Decodable { let sigmaCases: [SigmaCase], localAverage: [LocalCase], noEllipse: NoEllipse }
private struct Pin: Decodable {
    let scan: [Int], detector: [Int], cube: [Double], originRow: [Double], originColumn: [Double], qPixelSize: Double
    let ellipsePy4DSTEM: [Double], mask: [Int]
    let polar: [PolarCase], radial: Radial
}

private let pin: Pin = {
    do { return try JSONDecoder().decode(Pin.self, from: Data(contentsOf: Fixtures.url("polar-radial-pins.json"))) }
    catch { fatalError("polar-radial-pins.json: \(error)") }
}()

private func close(_ a: Double, _ b: Double) -> Bool { abs(a - b) <= 1e-12 + 1e-7 * abs(b) }
/// The largest |a - b| / (1e-12 + |b|) over a pair of vectors.
private func worst(_ a: [Double], _ b: [Double]) -> Double { zip(a, b).map { abs($0 - $1) / (1e-12 + abs($1)) }.max() ?? 0 }

private func fixtureCube() throws -> (MemoryCube, Calibration) {
    let shape = try CubeShape(scanRows: pin.scan[0], scanColumns: pin.scan[1], detectorRows: pin.detector[0], detectorColumns: pin.detector[1])
    let cube = try MemoryCube(shape: shape, values: pin.cube.map { Float($0) })
    var cal = Calibration()
    cal.detectorPixelSize = pin.qPixelSize
    try cal.setOrigin(row: ScanField(scanRows: shape.scanRows, scanColumns: shape.scanColumns, values: pin.originRow),
                      column: ScanField(scanRows: shape.scanRows, scanColumns: shape.scanColumns, values: pin.originColumn))
    return (cube, cal)
}

private func sampling(_ c: PolarCase) -> PolarSampling {
    var s = PolarSampling()
    if let v = c.params["n_annular"]?.double { s.annularCount = Int(v) }
    if let v = c.params["qmin"]?.double { s.minimumRadius = v }
    if let v = c.params["qmax"]?.double { s.maximumRadius = v }
    if let v = c.params["qstep"]?.double { s.radialStep = v }
    if let v = c.params["qscale"]?.double { s.radialPower = v }
    if let v = c.params["mask_thresh"]?.double { s.maskThreshold = v }
    if let v = c.params["ellipse"]?.bool { s.elliptical = v }
    if let v = c.params["two_fold_symmetry"]?.bool { s.twoFoldSymmetry = v }
    return s
}

private func calibration(_ base: Calibration, _ c: PolarCase) -> Calibration {
    var cal = base
    if let e = c.ellipse { cal.ellipse = EllipseCalibration(a: e[0], b: e[1], theta: e[2]) }
    return cal
}

private func polarCube(_ c: PolarCase) throws -> PolarCube {
    let usesMask = c.params["mask"] != nil
    return try PolarCube(detectorRows: pin.detector[0], detectorColumns: pin.detector[1], sampling: sampling(c),
                         detectorMask: usesMask ? pin.mask.map { $0 != 0 } : nil, detectorPixelSize: pin.qPixelSize)
}

@Suite struct PolarCubeTests {
    /// The sampling: radial bins (np.arange), annular bins, the KDE widths and the polar shape are py4DSTEM's.
    @Test func testSamplingIsPy4DSTEMs() throws {
        for c in pin.polar {
            let p = try polarCube(c)
            #expect([p.annularCount, p.radialCount] == c.polarShape, "\(c.name)")
            #expect(p.radialBins == c.radialBins, "\(c.name) radial bins")
            #expect(zip(p.annularBins, c.annularBins).allSatisfy { close($0, $1) }, "\(c.name) annular bins")
        }
    }

    /// Every pinned pattern: the validity mask exactly, the values and the coverage to the float64 default.
    @Test func testPolarPatternsAgreeWithPy4DSTEM() async throws {
        let (cube, base) = try fixtureCube()
        var worstValue = 0.0, worstNorm = 0.0
        for c in pin.polar {
            let p = try polarCube(c)
            let cal = calibration(base, c)
            for want in c.patterns {
                let got = try await p.pattern(of: cube, scanRow: want.row, scanColumn: want.column, calibration: cal)
                #expect(got.valid == want.mask.map { $0 == 0 }, "\(c.name) (\(want.row), \(want.column)) validity")
                for i in got.values.indices {
                    if let w = want.values[i] {
                        #expect(close(got.values[i], w), "\(c.name) (\(want.row), \(want.column)) cell \(i): \(got.values[i]) vs \(w)")
                        worstValue = max(worstValue, abs(got.values[i] - w) / (1e-12 + abs(w)))
                    } else {
                        #expect(got.values[i].isNaN, "\(c.name) cell \(i) should be invalid")
                    }
                }
                #expect(zip(got.coverage, want.norm).allSatisfy { close($0, $1) }, "\(c.name) coverage")
                worstNorm = max(worstNorm, worst(got.coverage, want.norm))
            }
        }
        print("polar vs py4DSTEM: worst relative value difference \(worstValue), coverage \(worstNorm)")
        #expect(worstValue < 1e-9)
    }

    /// The whole-cube transform (shared and per-position geometry, tiles, parallel positions) equals the per-pattern one.
    @Test func testWholeCubeTransformEqualsPerPatternTransform() async throws {
        let (cube, base) = try fixtureCube()
        let c = pin.polar[0]
        let p = try polarCube(c)
        let cal = calibration(base, c)
        let all = try await p.transform(source: cube, calibration: cal, tiling: .rows(2))
        #expect(all.scanRows == 3 && all.scanColumns == 3 && all.annularCount == 36 && all.radialCount == 29)
        for want in c.patterns {
            let one = try await p.pattern(of: cube, scanRow: want.row, scanColumn: want.column, calibration: cal)
            let got = all.pattern(row: want.row, column: want.column)
            #expect(got.valid == one.valid)
            #expect(zip(got.values, one.values).allSatisfy { $0 == $1 || ($0.isNaN && $1.isNaN) })
        }
        // a constant origin and ellipse share one geometry and give the same numbers as the per-position path
        var constant = cal
        constant.setOrigin(row: 19.6, column: 22.4)
        let shared = try await p.transform(source: cube, calibration: constant)
        let one = try await p.pattern(of: cube, scanRow: 2, scanColumn: 1, calibration: constant)
        let got = shared.pattern(row: 2, column: 1)
        #expect(zip(got.values, one.values).allSatisfy { $0 == $1 || ($0.isNaN && $1.isNaN) })
    }

    @Test func testRefusals() async throws {
        let (cube, base) = try fixtureCube()
        let c = pin.polar[0]
        let p = try polarCube(c)
        // no origin
        await #expect(throws: CalibrationError.notSet("origin")) { _ = try await p.pattern(of: cube, scanRow: 0, scanColumn: 0, calibration: Calibration()) }
        // a polar cube for another detector
        let other = try PolarCube(detectorRows: 30, detectorColumns: 30)
        await #expect(throws: CuboVirtualError.self) { _ = try await other.pattern(of: cube, scanRow: 0, scanColumn: 0, calibration: base) }
        // a sampling py4DSTEM would fail on
        #expect(throws: CuboVirtualError.self) { _ = try PolarCube(detectorRows: 8, detectorColumns: 8, sampling: PolarSampling(annularCount: 1)) }
        #expect(throws: CuboVirtualError.self) { _ = try PolarCube(detectorRows: 8, detectorColumns: 8, sampling: PolarSampling(radialStep: 0)) }
        #expect(throws: CuboVirtualError.self) { _ = try PolarCube(detectorRows: 8, detectorColumns: 8, sampling: PolarSampling(minimumRadius: 5, maximumRadius: 3)) }
        #expect(throws: CuboVirtualError.self) { _ = try PolarCube(detectorRows: 8, detectorColumns: 8, detectorMask: [true]) }
        // cancellation: asked before the first tile
        await #expect(throws: CancellationError.self) { _ = try await p.transform(source: cube, calibration: base, isCancelled: { true }) }
    }

    @Test func testProgressReachesOne() async throws {
        let (cube, base) = try fixtureCube()
        let p = try polarCube(pin.polar[0])
        let last = LastValue()
        _ = try await p.transform(source: cube, calibration: calibration(base, pin.polar[0]), tiling: .rows(1), progress: { last.set($0) })
        #expect(last.value == 1)
    }
}

final class LastValue: @unchecked Sendable {
    private let lock = NSLock()
    private var v = 0.0
    func set(_ x: Double) { lock.lock(); v = x; lock.unlock() }
    var value: Double { lock.lock(); defer { lock.unlock() }; return v }
}

@Suite struct RadialBackgroundTests {
    @Test func testBackgroundAndSubtractionAgreeWithPy4DSTEM() async throws {
        let (cube, base) = try fixtureCube()
        let c = pin.polar[0]
        let p = try polarCube(c)
        let cal = calibration(base, c)
        var worstBg = 0.0, worstSb = 0.0
        for want in pin.radial.sigmaCases {
            let bg = try await RadialBackground.background(of: cube, polar: p, calibration: cal, scanRow: want.row, scanColumn: want.column, sigma: want.sigma)
            let sb = try await RadialBackground.subtracted(from: cube, polar: p, calibration: cal, scanRow: want.row, scanColumn: want.column, sigma: want.sigma)
            #expect(bg.path == .cpu && bg.detectorRows == 40 && bg.detectorColumns == 44)
            #expect(zip(bg.values, want.background).allSatisfy { close($0, $1) }, "background (\(want.row), \(want.column)) sigma \(want.sigma)")
            #expect(zip(sb.values, want.bksb).allSatisfy { close($0, $1) }, "bksb (\(want.row), \(want.column))")
            #expect(sb.values.allSatisfy { $0 >= 0 })
            worstBg = max(worstBg, worst(bg.values, want.background)); worstSb = max(worstSb, worst(sb.values, want.bksb))
        }
        print("radial background vs py4DSTEM: worst relative difference \(worstBg), subtracted \(worstSb)")
    }

    @Test func testLocalAverageWithRadialBackgroundSubtraction() async throws {
        let (cube, base) = try fixtureCube()
        let c = pin.polar[0]
        let p = try polarCube(c)
        let cal = calibration(base, c)
        var w = 0.0
        for want in pin.radial.localAverage {   // a corner, the interior and two edge positions of a 3 x 3 scan
            let got = try await RadialBackground.localAverage(of: cube, polar: p, calibration: cal, scanRow: want.row, scanColumn: want.column, sigma: want.sigma)
            #expect(zip(got.values, want.values).allSatisfy { close($0, $1) }, "local average (\(want.row), \(want.column))")
            w = max(w, worst(got.values, want.values))
        }
        print("local average (radial_bksb) vs py4DSTEM: worst relative difference \(w)")
        await #expect(throws: CancellationError.self) {
            _ = try await RadialBackground.localAverage(of: cube, polar: p, calibration: cal, scanRow: 1, scanColumn: 1, isCancelled: { true })
        }
    }

    /// No ellipse in the calibration: the polar transform is circular and the lookup radius is the plain distance.
    @Test func testNoEllipseIsACircle() async throws {
        let (cube, base) = try fixtureCube()
        let n = pin.radial.noEllipse
        let p = try PolarCube(detectorRows: 40, detectorColumns: 44, sampling: PolarSampling(radialStep: 2, annularCount: 24), detectorPixelSize: pin.qPixelSize)
        let bg = try await RadialBackground.background(of: cube, polar: p, calibration: base, scanRow: n.row, scanColumn: n.column, sigma: n.sigma)
        #expect(zip(bg.values, n.background).allSatisfy { close($0, $1) })
    }

    @Test func testRefusals() async throws {
        let (cube, base) = try fixtureCube()
        let one = try PolarCube(detectorRows: 40, detectorColumns: 44, sampling: PolarSampling(minimumRadius: 3, maximumRadius: 3.5))
        await #expect(throws: CuboVirtualError.self) { _ = try await RadialBackground.background(of: cube, polar: one, calibration: base, scanRow: 0, scanColumn: 0) }
        let p = try PolarCube(detectorRows: 40, detectorColumns: 44)
        await #expect(throws: CalibrationError.notSet("origin")) { _ = try await RadialBackground.background(of: cube, polar: p, calibration: Calibration(), scanRow: 0, scanColumn: 0) }
        await #expect(throws: CuboError.self) { _ = try await RadialBackground.localAverage(of: cube, polar: p, calibration: base, scanRow: 9, scanColumn: 0) }
    }
}

// MARK: planted truth (tools/polar-ellipse/planted.py, in Swift)

/// The planted pattern: rings at rho = 1 and 1.6 of an ellipse (r0, c0, a, b) whose major axis lies along (sin t, cos t) in
/// (row, column) (planted.py's theta, from +column towards +row), plus the exponential background of the circularised radius.
private struct Planted {
    var r0 = 31.3, c0 = 32.6, a = 16.0, b = 12.0, theta = 0.6
    let size = 64
    func value(_ r: Double, _ c: Double) -> Double {
        let dr = r - r0, dc = c - c0
        let u = dr * sin(theta) + dc * cos(theta), v = dr * cos(theta) - dc * sin(theta)
        let rho = ((u / a) * (u / a) + (v / b) * (v / b)).squareRoot()
        let w = 0.04
        return 100 * exp(-pow(rho - 1, 2) / (2 * w * w)) + 60 * exp(-pow(rho - 1.6, 2) / (2 * w * w)) + 80 * exp(-a * rho / 25) + 5
    }
    func cube() throws -> MemoryCube {
        var v = [Float](); v.reserveCapacity(size * size)
        for r in 0..<size { for c in 0..<size { v.append(Float(value(Double(r), Double(c)))) } }
        return try MemoryCube(shape: try CubeShape(scanRows: 1, scanColumns: 1, detectorRows: size, detectorColumns: size), values: v)
    }
    /// py4DSTEM's theta: the major axis from the +row axis towards the +column axis.
    var thetaPy4DSTEM: Double { Double.pi / 2 - theta }
}

@Suite struct PolarPlantedTruthTests {
    /// The spread over the angle of a radial bin, relative to its mean, for the bins whose radius lies in `radii`.
    private func spread(_ polar: PolarPattern, in radii: Range<Int>) -> Double {
        var worst = 0.0
        for r in radii {
            let column = (0..<polar.annularCount).filter { polar.valid[$0 * polar.radialCount + r] }.map { polar.values[$0 * polar.radialCount + r] }
            guard column.count > polar.annularCount / 2, let lo = column.min(), let hi = column.max() else { continue }
            worst = max(worst, (hi - lo) / (column.reduce(0, +) / Double(column.count)))
        }
        return worst
    }

    private func transform(_ p: Planted, ellipse: (a: Double, b: Double, theta: Double)?) async throws -> PolarPattern {
        let cube = try p.cube()
        var cal = Calibration()
        cal.setOrigin(row: p.r0, column: p.c0)
        if let e = ellipse { cal.ellipse = EllipseCalibration(a: e.a, b: e.b, theta: e.theta) }
        let polar = try PolarCube(detectorRows: p.size, detectorColumns: p.size, sampling: PolarSampling(annularCount: 90))
        return try await polar.pattern(of: cube, scanRow: 0, scanColumn: 0, calibration: cal)
    }

    /// With the right ellipse the polar cube is constant in phi: radial bins away from the rings (the background alone, 60% of
    /// the radius range) vary by well under a per cent, and a ring is a single column. With a circle assumption on the same
    /// data the bins vary by tens of per cent. (Fails before the ellipse is applied; passes with it.)
    @Test func testRightEllipseIsConstantInPhiAndACircleIsNot() async throws {
        let p = Planted()
        let right = try await transform(p, ellipse: (p.a, p.b, p.thetaPy4DSTEM))
        let circle = try await transform(p, ellipse: nil)
        // bins on the scale of b = 12: ring 1 sits at 12 (bins 12), ring 2 at 19.2; the cells are valid up to the detector edge
        let freeBins = 2..<8                   // rho 0.17 to 0.67: only background
        let rightSpread = spread(right, in: freeBins), circleSpread = spread(circle, in: freeBins)
        print("polar planted tilted: spread over phi with the right ellipse \(rightSpread), with a circle \(circleSpread)")
        #expect(rightSpread < 0.01)
        #expect(circleSpread > 0.05)
        #expect(circleSpread > 10 * rightSpread)
        // the ring: its radial centroid (bins 8 to 16 of each angular row) does not depend on phi with the right ellipse, and
        // wanders by a pixel and more with a circle
        func centroidRange(_ polar: PolarPattern) -> Double {
            var cs = [Double]()
            for t in 0..<polar.annularCount {
                var num = 0.0, den = 0.0
                for r in 8..<17 where polar.valid[t * polar.radialCount + r] {
                    let w = polar.values[t * polar.radialCount + r] - 20   // above the background of the window
                    if w > 0 { num += w * (Double(r) + 0.5); den += w }
                }
                if den > 0 { cs.append(num / den) }
            }
            return cs.max()! - cs.min()!
        }
        let rightRange = centroidRange(right), circleRange = centroidRange(circle)
        print("ring centroid range over phi: right ellipse \(rightRange) px, circle \(circleRange) px")
        #expect(rightRange < 0.3)
        #expect(circleRange > 0.5)
        #expect(circleRange > 20 * rightRange)
    }

    /// The whole chain on the planted tilted case: fit the ring of the pattern (task 18), store the fit in the calibration
    /// (`Calibration.ellipse`), transform (task 17): the result is constant in phi, as with the true ellipse.
    @Test func testFittedEllipseFeedsThePolarCube() async throws {
        let p = Planted()
        let cube = try p.cube()
        let tile = try await cube.readPattern(row: 0, column: 0)
        let data = try EllipseFitPlane(rows: p.size, columns: p.size, values: tile.values.map(Double.init))
        let fit = try EllipseFit.fitAmorphousRing(data, center: (p.r0 + 0.5, p.c0 - 0.4), fitRadii: (p.b * 0.75, p.a * 1.25))
        let e = fit.ellipse
        print("fitted ellipse: centre (\(e.centerRow), \(e.centerColumn)) a \(e.a) b \(e.b) theta \(e.theta); truth \(p.r0), \(p.c0), \(p.a), \(p.b), \(p.thetaPy4DSTEM)")
        #expect(fit.converged)
        #expect(abs(e.a - p.a) < 0.1 && abs(e.b - p.b) < 0.1 && abs(e.centerRow - p.r0) < 0.05 && abs(e.centerColumn - p.c0) < 0.05)
        var cal = Calibration()
        cal.setOrigin(row: e.centerRow, column: e.centerColumn)
        cal.ellipse = e.calibration
        let polar = try PolarCube(detectorRows: p.size, detectorColumns: p.size, sampling: PolarSampling(annularCount: 90))
        let fitted = try await polar.pattern(of: cube, scanRow: 0, scanColumn: 0, calibration: cal)
        let truth = try await transform(p, ellipse: (p.a, p.b, p.thetaPy4DSTEM))
        let circle = try await transform(p, ellipse: nil)
        let freeBins = 2..<8
        print("spread over phi: fitted \(spread(fitted, in: freeBins)), true ellipse \(spread(truth, in: freeBins)), circle \(spread(circle, in: freeBins))")
        #expect(spread(fitted, in: freeBins) < 0.015)
        #expect(spread(fitted, in: freeBins) < spread(circle, in: freeBins) / 5)
    }

    /// The ring position: the phi mean of the right-ellipse transform peaks at b (bin 12 holds radius 12 to 13, centre 12.5 at the
    /// bin's left edge convention: the bin value is the density at its left edge), within 0.25 pixel.
    @Test func testRingPositionIsTheMinorAxisOnTheTiltedCase() async throws {
        let p = Planted()
        let right = try await transform(p, ellipse: (p.a, p.b, p.thetaPy4DSTEM))
        var mean = [Double](repeating: 0, count: right.radialCount)
        for r in 0..<right.radialCount {
            let v = (0..<right.annularCount).filter { right.valid[$0 * right.radialCount + r] }.map { right.values[$0 * right.radialCount + r] }
            mean[r] = v.reduce(0, +) / Double(max(v.count, 1))
        }
        // subpixel peak by the parabola through the maximum bin and its neighbours, in bins of width 1 (radius = bin + 0.5 is the
        // bin centre; py4DSTEM labels a bin by its left edge, and the binning spreads a pixel over the two bins around it)
        let k = mean.indices.dropFirst(4).max { mean[$0] < mean[$1] }!
        let denom = mean[k - 1] - 2 * mean[k] + mean[k + 1]
        let peak = Double(k) + 0.5 * (mean[k - 1] - mean[k + 1]) / denom
        print("ring 1 peak at bin \(peak) (truth b = \(p.b))")
        #expect(abs(peak - p.b) <= 0.25)
    }

    /// The radial background of the tilted case is the polar profile laid out on the major-axis scale, as py4DSTEM does. For a
    /// circle that is the planted exponential; for the ellipse the pixel at elliptical radius q takes the profile at bin q.
    @Test func testRadialBackgroundOfACircle() async throws {
        var p = Planted(); p.b = p.a; p.theta = 0
        let cube = try p.cube()
        var cal = Calibration(); cal.setOrigin(row: p.r0, column: p.c0)
        cal.ellipse = EllipseCalibration(a: p.a, b: p.a, theta: 0)
        let polar = try PolarCube(detectorRows: p.size, detectorColumns: p.size, sampling: PolarSampling(annularCount: 90))
        let bg = try await RadialBackground.background(of: cube, polar: polar, calibration: cal, scanRow: 0, scanColumn: 0, sigma: 0)
        // away from the rings (|rho - 1| > 0.25, |rho - 1.6| > 0.25) and inside the sampled radius the background is the
        // planted exponential plus the (rings' tails are nil there); the binning is not an interpolation, so 1.5%
        var worst = 0.0, count = 0
        for r in 0..<p.size { for c in 0..<p.size {
            let q = hypot(Double(r) - p.r0, Double(c) - p.c0)
            let rho = q / p.a
            guard q > 3, q < 28, abs(rho - 1) > 0.25, abs(rho - 1.6) > 0.25 else { continue }
            let truth = 80 * exp(-q / 25) + 5
            worst = max(worst, abs(bg.values[r * p.size + c] - truth) / truth); count += 1
        } }
        print("circle: radial background against the planted exponential, worst relative \(worst) over \(count) pixels")
        #expect(count > 500)
        #expect(worst < 0.015)
    }
}
