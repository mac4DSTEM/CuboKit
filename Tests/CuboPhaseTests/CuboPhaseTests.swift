//
//  CuboPhaseTests.swift
//  The centre-of-mass reference, the R–Q rotation and DPC against py4DSTEM f050d207 and planted truth (registration
//  slice 2, 3.2–3.4, 4, 5). The pins are CuboVirtualTests/Fixtures/slice-2-pins.json; the real data are slice 2's
//  exports in $CUBOKIT_REAL_DATA.
//

import Foundation
import Testing
import CuboCore
import CuboPhase
import CuboVirtual

enum Pins2 {
    static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("CuboVirtualTests/Fixtures")
    nonisolated(unsafe) static let pins: [String: Any] = {
        try! JSONSerialization.jsonObject(with: Data(contentsOf: fixtures.appendingPathComponent("slice-2-pins.json")))
            as! [String: Any]
    }()
    static func doubles(_ any: Any?) -> [Double] { (any as? [String] ?? []).map { Double($0)! } }
    static func double(_ any: Any?) -> Double { Double(any as! String)! }
    static var realData: URL {
        if let env = ProcessInfo.processInfo.environment["CUBOKIT_REAL_DATA"], !env.isEmpty { return URL(fileURLWithPath: env) }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Caches/CuboKit/real-data")
    }
    static func realIsPresent(_ name: String) -> Bool {
        FileManager.default.fileExists(atPath: realData.appendingPathComponent("\(name)-s2-pins.json").path)
    }
}

/// Relative with slice 1's floor (1e-6 · max|want|).
@discardableResult
func near(_ got: [Double], _ want: [Double], _ t: Double, _ what: String,
          sourceLocation: SourceLocation = SourceLocation(fileID: #fileID, filePath: #filePath, line: #line, column: #column)) -> Double {
    #expect(got.count == want.count, "\(what): count \(got.count) vs \(want.count)", sourceLocation: sourceLocation)
    guard got.count == want.count, !want.isEmpty else { return 0 }
    let floor = 1e-6 * (want.map(abs).max() ?? 0)
    var worst = 0.0
    for k in want.indices {
        let d = abs(got[k] - want[k]) / max(abs(want[k]), floor, .leastNormalMagnitude)
        if !(d <= worst) { worst = d }
    }
    #expect(worst <= t, "\(what): \(worst) > \(t)", sourceLocation: sourceLocation)
    return worst
}

/// max|got − want| / max|want|.
@discardableResult
func ofMax(_ got: [Double], _ want: [Double], _ t: Double, _ what: String,
           sourceLocation: SourceLocation = SourceLocation(fileID: #fileID, filePath: #filePath, line: #line, column: #column)) -> Double {
    #expect(got.count == want.count, "\(what): count", sourceLocation: sourceLocation)
    guard got.count == want.count, !want.isEmpty else { return 0 }
    let d = zip(got, want).map { abs($0 - $1) }.max()! / max(want.map(abs).max()!, .leastNormalMagnitude)
    #expect(d <= t, "\(what): \(d) > \(t)", sourceLocation: sourceLocation)
    return d
}

/// Checks one chain (fit, rotation, DPC) against the float64 chain and py4DSTEM's own decisions.
func checkChain(measured: ScanVectorField, reference: CenterOfMassReference, pins ch: [String: Any], _ name: String) throws {
    let (fitted, normalized) = try CenterOfMassReference.normalize(measured, reference: reference)
    if case .planeFit = reference {
        near(fitted.row, Pins2.doubles(ch["fitted_x"]), 1e-10, "\(name) fitted row")
        near(fitted.column, Pins2.doubles(ch["fitted_y"]), 1e-10, "\(name) fitted column")
        // py4DSTEM's curve_fit plane is recorded, not asserted (amendment 2026-10-15: up to 8.3e-6 px from the exact
        // least-squares plane of the same input; its stopping tolerance, not the definition).
        print("slice2 \(name): py4DSTEM curve_fit plane vs least squares of its own input \(ch["fit_distance_py4dstem_vs_lstsq_px"] ?? "n/a") px")
    }
    // the normalized shift: 1e-10 px (absolute: it can be near zero everywhere)
    let dn = zip(normalized.row, Pins2.doubles(ch["normalized_x"])).map { abs($0 - $1) }.max() ?? 0
    #expect(dn <= 1e-10, "\(name): normalized shift \(dn) px")
    let solution = try RotationSolver.solve(normalized)
    #expect(solution.degrees == ch["degrees"] as! Double, "\(name): \(solution.degrees) vs \(ch["degrees"]!)")
    #expect(solution.transposed == ch["transposed"] as! Bool, "\(name): transposed")
    #expect(solution.degrees == (ch["degrees_py4dstem"] as! Double).rounded(), "\(name): py4DSTEM chose \(ch["degrees_py4dstem"]!)")
    #expect(solution.transposed == ch["transposed_py4dstem"] as! Bool, "\(name): py4DSTEM's transpose")
    near(solution.curve, Pins2.doubles(ch["curve"]), 1e-10, "\(name) curve")
    near(solution.curveTransposed, Pins2.doubles(ch["curve_transposed"]), 1e-10, "\(name) curve transposed")
    let dpc = try DPC.reconstruct(solution.corrected)
    #expect(dpc.accepted == ch["accepted"] as! [Bool], "\(name): backtracking sequence")
    near(dpc.errors, Pins2.doubles(ch["errors"]), 1e-9, "\(name) errors")
    let d = ofMax(dpc.phase, Pins2.doubles(ch["phase"]), 1e-8, "\(name) phase")
    print("slice2 chain \(name): rotation \(solution.degrees) T=\(solution.transposed); phase kit vs float64 \(d) of max; py4DSTEM own vs float64 \(ch["phase_distance_py4dstem_own_vs_f64"]!)")
}

@Suite struct CuboPhasePinsTests {
    /// The full chain from the kit's own centre of mass on the synthetic cubes (plane fit, as py4DSTEM's default).
    @Test func testChainOnCubesAgreesWithTheFloat64ChainAndPy4DSTEM() async throws {
        for e in (Pins2.pins["com"] as! [[String: Any]]) where e["chain"] != nil {
            let name = e["cube"] as! String
            let cube = try MappedCube(npy: Pins2.fixtures.appendingPathComponent("\(name).npy"))
            let com = try await CenterOfMass.measure(of: cube)
            let measured = try ScanVectorField(scanRows: com.row.scanRows, scanColumns: com.row.scanColumns,
                                               row: com.row.values, column: com.column.values)
            try checkChain(measured: measured, reference: .planeFit, pins: e["chain"] as! [String: Any], name)
        }
    }

    /// The planted fields (5.1): the solver returns the planted angle and transpose, as py4DSTEM does.
    @Test func testChainOnPlantedFields() throws {
        for f in Pins2.pins["fields"] as! [[String: Any]] {
            let shape = f["shape"] as! [Int]
            let field = try ScanVectorField(scanRows: shape[0], scanColumns: shape[1],
                                            row: Pins2.doubles(f["measured_x"]), column: Pins2.doubles(f["measured_y"]))
            try checkChain(measured: field, reference: .none, pins: f, "field \(f["theta"]!)")
            let solution = try RotationSolver.solve(field)
            #expect(solution.degrees == f["theta"] as! Double && solution.transposed == f["transpose"] as! Bool)
        }
    }
}

@Suite struct CuboPhaseTruthTests {
    /// 5.1: the reconstruction of a known phase gradient correlates with the phase at ≥ 0.99.
    @Test func testKnownPhaseIsReconstructed() throws {
        for f in Pins2.pins["fields"] as! [[String: Any]] where f["theta"] as! Double == 0 {
            let shape = f["shape"] as! [Int]
            let field = try ScanVectorField(scanRows: shape[0], scanColumns: shape[1],
                                            row: Pins2.doubles(f["measured_x"]), column: Pins2.doubles(f["measured_y"]))
            let phase = try DPC.reconstruct(field).phase
            let phi = Pins2.doubles(f["phi"])
            let mp = phase.reduce(0, +) / Double(phase.count), mq = phi.reduce(0, +) / Double(phi.count)
            var spq = 0.0, spp = 0.0, sqq = 0.0
            for k in phi.indices { let p = phase[k] - mp, q = phi[k] - mq; spq += p * q; spp += p * p; sqq += q * q }
            let pearson = spq / (spp * sqq).squareRoot()
            #expect(pearson >= 0.99, "pearson \(pearson)")
            let e = try DPC.reconstruct(field).errors
            #expect(zip(e, e.dropFirst()).allSatisfy { $1 <= $0 }, "accepted errors must not increase")
        }
    }

    /// The curl of an exact gradient vanishes at θ = 0 (to rounding); the divergence criterion picks the same angle.
    @Test func testCurlOfAGradientAndTheDivergenceCriterion() throws {
        let f = (Pins2.pins["fields"] as! [[String: Any]]).first { $0["theta"] as! Double == 23 }!
        let shape = f["shape"] as! [Int]
        let field = try ScanVectorField(scanRows: shape[0], scanColumns: shape[1],
                                        row: Pins2.doubles(f["measured_x"]), column: Pins2.doubles(f["measured_y"]))
        let curl = try RotationSolver.solve(field)
        let k = curl.anglesDegrees.firstIndex(of: 23)!
        #expect(curl.curve[k] <= 1e-12 * curl.curve.max()!, "curl at the planted angle \(curl.curve[k])")
        let forced = try RotationSolver.solve(field, forceRotationDegrees: 23)
        #expect(forced.transposed == false && forced.degrees == 23)
        let both = try RotationSolver.solve(field, forceRotationDegrees: 23, forceTranspose: true)
        #expect(both.transposed && both.curve.isEmpty)
        let onlyT = try RotationSolver.solve(field, forceTranspose: false)
        #expect(onlyT.degrees == 23 && onlyT.curveTransposed.isEmpty)
    }

    /// A pure plane is removed by the plane fit to rounding; `.constant` removes the mean; `.given` subtracts the field.
    @Test func testReferences() throws {
        let p = Pins2.pins["plane"] as! [String: Any]
        let shape = p["shape"] as! [Int]
        let plane = try ScanVectorField(scanRows: shape[0], scanColumns: shape[1], row: Pins2.doubles(p["x"]),
                                        column: Pins2.doubles(p["y"]))
        let n = try CenterOfMassReference.normalize(plane, reference: .planeFit).normalized
        #expect((n.row + n.column).allSatisfy { abs($0) <= 1e-12 })
        let c = try CenterOfMassReference.normalize(plane, reference: .constant, reciprocalSampling: 2).normalized
        #expect(abs(c.row.reduce(0, +)) <= 1e-12)
        let g = try CenterOfMassReference.normalize(plane, reference: .given(plane)).normalized
        #expect((g.row + g.column).allSatisfy { $0 == 0 })
        // NaN → 0 (nan_to_num), and a NaN position is left out of the plane fit
        var withNaN = plane.row; withNaN[4] = .nan
        let field = try ScanVectorField(scanRows: shape[0], scanColumns: shape[1], row: withNaN, column: plane.column)
        let m = try CenterOfMassReference.normalize(field, reference: .planeFit).normalized
        #expect(m.row[4] == 0 && m.row.allSatisfy { abs($0) <= 1e-12 })
    }

    /// py4DSTEM's tie rules: a field whose curl is the same at every angle and orientation (zero) keeps the plain
    /// orientation and the first angle (argmin's first index; ≤ for the orientation).
    @Test func testTiesKeepThePlainOrientationAndTheFirstAngle() throws {
        let zero = try ScanVectorField(scanRows: 4, scanColumns: 5, row: Array(repeating: 0, count: 20),
                                       column: Array(repeating: 0, count: 20))
        let s = try RotationSolver.solve(zero)
        #expect(s.transposed == false && s.degrees == -89)
        let d = try RotationSolver.solve(zero, criterion: .maximizeDivergence)
        #expect(d.transposed == false && d.degrees == -89)
    }

    /// Backtracking halves the step: a step of 8 overshoots on the planted field, so iterations are rejected, the
    /// final step is 8 / 2^(rejections), and the accepted errors never increase.
    @Test func testBacktrackingHalvesTheStep() throws {
        let f = (Pins2.pins["fields"] as! [[String: Any]]).first { $0["theta"] as! Double == 0 }!
        let shape = f["shape"] as! [Int]
        let field = try ScanVectorField(scanRows: shape[0], scanColumns: shape[1],
                                        row: Pins2.doubles(f["measured_x"]), column: Pins2.doubles(f["measured_y"]))
        let r = try DPC.reconstruct(field, maxIterations: 40, stepSize: 8)
        let rejected = r.accepted.filter { !$0 }.count
        #expect(rejected > 0, "a step of 8 must overshoot")
        #expect(r.finalStepSize == 8 / Double(1 << rejected))
        #expect(zip(r.errors, r.errors.dropFirst()).allSatisfy { $1 <= $0 })
        let off = try DPC.reconstruct(field, maxIterations: 40, stepSize: 8, backtrack: false)
        #expect(off.accepted.allSatisfy { $0 } && off.finalStepSize == 8)
    }

    @Test func testRefusals() throws {
        let small = try ScanVectorField(scanRows: 2, scanColumns: 5, row: Array(repeating: 1, count: 10),
                                        column: Array(repeating: 0, count: 10))
        #expect(throws: CuboPhaseError.scanTooSmallForRotation(rows: 2, columns: 5)) { try RotationSolver.solve(small) }
        let zero = try ScanVectorField(scanRows: 3, scanColumns: 3, row: Array(repeating: 0, count: 9),
                                       column: Array(repeating: 0, count: 9))
        #expect(throws: CuboPhaseError.zeroField) { try DPC.reconstruct(zero) }
        #expect(throws: CuboPhaseError.planeFitDegenerate) {
            try CenterOfMassReference.normalize(try ScanVectorField(scanRows: 1, scanColumns: 4, row: [1, 2, 3, 4],
                                                                    column: [0, 0, 0, 0]), reference: .planeFit)
        }
        #expect(throws: CuboPhaseError.self) { try ScanVectorField(scanRows: 2, scanColumns: 2, row: [1], column: [1]) }
        #expect(throws: CuboPhaseError.invalidAngles) { try RotationSolver.solve(zero, anglesDegrees: []) }
    }
}

/// Serialized: each test maps a real cube; real data absent → skipped by name (tools/test.sh then writes no stamp).
@Suite(.serialized) struct CuboPhaseRealDataTests {
    static func check(_ name: String) async throws {
        let dir = Pins2.realData
        let info = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("\(name)-s2-pins.json"))) as! [String: Any]
        func a(_ key: String) throws -> [Double] { try NPYFile.readDoubles(dir.appendingPathComponent("\(name)-s2-\(key).npy")).values }
        let cube = try MappedCube(npy: dir.appendingPathComponent("\(name).npy"))
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("\(name)-pins.json"))) as! [String: Any]
        let bfSpec = ((meta["geometries"] as! [[String: Any]]).first { $0["name"] as! String == "bf" }!["geometry"]) as! [String: Any]
        let bf = DetectorGeometry.circle(centerRow: bfSpec["centerRow"] as! Double, centerColumn: bfSpec["centerColumn"] as! Double,
                                         radius: bfSpec["radius"] as! Double)
        var log = ["real slice 2 \(name)"]
        for (mname, mask) in [("none", DetectorGeometry?.none), ("bf", bf)] {
            let com = try await CenterOfMass.measure(of: cube, mask: mask)
            near(com.intensity.values, try a("com_\(mname)_intensity"), 1e-10, "\(name)/\(mname) intensity")
            let conditioned = try a("com_\(mname)_conditioned")
            let r = try a("com_\(mname)_row"), c = try a("com_\(mname)_column")
            var worst = 0.0
            for k in r.indices where conditioned[k] == 1 {
                worst = max(worst, abs(com.row.values[k] - r[k]), abs(com.column.values[k] - c[k]))
            }
            #expect(worst <= 1e-9, "\(name)/\(mname): CoM \(worst) px")
            var line = "CoM \(mname): kit vs float64 \(worst) px (ill-conditioned \(conditioned.filter { $0 == 0 }.count))"
            if VirtualDiffraction.metalIsAvailable {
                let gpu = try await CenterOfMass.measure(of: cube, mask: mask, path: .metal)
                let dm = zip(gpu.row.values, com.row.values).enumerated().filter { conditioned[$0.offset] == 1 }
                    .map { abs($0.element.0 - $0.element.1) }.max() ?? 0
                #expect(dm <= 1e-9, "\(name)/\(mname): Metal CoM \(dm) px")
                line += ", metal vs cpu \(dm) px"
            }
            log.append(line)
            if mname == "none" {
                let measured = try ScanVectorField(scanRows: com.row.scanRows, scanColumns: com.row.scanColumns,
                                                   row: com.row.values, column: com.column.values)
                let chain = info["chain"] as! [String: Any]
                let (fitted, normalized) = try CenterOfMassReference.normalize(measured, reference: .planeFit)
                let df = ofMax(fitted.row, try a("chain_fitted_x"), 1e-10, "\(name) fitted")
                let solution = try RotationSolver.solve(normalized)
                #expect(solution.degrees == chain["degrees"] as! Double && solution.transposed == chain["transposed"] as! Bool,
                        "\(name): rotation \(solution.degrees) \(solution.transposed) vs float64 \(chain["degrees"]!) \(chain["transposed"]!)")
                #expect(solution.degrees == (chain["degrees_py4dstem"] as! Double).rounded()
                        && solution.transposed == chain["transposed_py4dstem"] as! Bool,
                        "\(name): py4DSTEM chose \(chain["degrees_py4dstem"]!) \(chain["transposed_py4dstem"]!)")
                near(solution.curve, try a("chain_curve"), 1e-10, "\(name) curve")
                let dpc = try DPC.reconstruct(solution.corrected)
                #expect(dpc.accepted == chain["accepted"] as! [Bool], "\(name): backtracking")
                let dp = ofMax(dpc.phase, try a("chain_phase"), 1e-8, "\(name) phase")
                log.append("chain: fit \(df), rotation \(solution.degrees) T=\(solution.transposed), phase kit vs float64 \(dp) of max, py4DSTEM own vs float64 \(chain["phase_distance_py4dstem_own_vs_f64"]!), plane vs curve_fit \(chain["fit_distance_py4dstem_vs_lstsq_px"]!) px")
            }
        }
        let (mean, max) = try await VirtualDiffraction.meanAndMax(of: cube)
        for (kind, pattern) in [("mean", mean), ("max", max)] {
            let e = (info["probe"] as! [String: Any])[kind] as! [String: Any]
            if e["refused"] as! Bool { #expect(throws: CuboVirtualError.self) { try ProbeSize.estimate(pattern) }; continue }
            let p = try ProbeSize.estimate(pattern)
            #expect(p.radius.bitPattern == Pins2.double(e["radius"]).bitPattern
                    && p.centerRow.bitPattern == Pins2.double(e["center_row"]).bitPattern
                    && p.centerColumn.bitPattern == Pins2.double(e["center_column"]).bitPattern, "\(name) probe \(kind)")
            log.append("probe \(kind): r \(p.radius) at (\(p.centerRow), \(p.centerColumn))")
        }
        print(log.joined(separator: "\n  "))
    }

    @Test(.enabled(if: Pins2.realIsPresent("almgsi"), "real data absent: run tools/slice-2/pins.py real"))
    func testAlMgSiStudyCubeSlice2() async throws { try await Self.check("almgsi") }
    @Test(.enabled(if: Pins2.realIsPresent("auref"), "real data absent: run tools/slice-2/pins.py real"))
    func testAuReferenceCubeSlice2() async throws { try await Self.check("auref") }
    @Test(.enabled(if: Pins2.realIsPresent("simau"), "real data absent: run tools/slice-2/pins.py real"))
    func testSimulatedPolycrystallineGoldSlice2() async throws { try await Self.check("simau") }
    @Test(.enabled(if: Pins2.realIsPresent("particle"), "real data absent: run tools/slice-2/pins.py real"))
    func testParticleCubeSlice2() async throws { try await Self.check("particle") }
    @Test(.enabled(if: Pins2.realIsPresent("py4dstem_small"), "real data absent: run tools/slice-2/pins.py real"))
    func testPy4DSTEMSmallDatacubeSlice2() async throws { try await Self.check("py4dstem_small") }
}
