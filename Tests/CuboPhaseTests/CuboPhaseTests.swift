//
//  CuboPhaseTests.swift
//  The centre-of-mass reference, the R–Q rotation and DPC against py4DSTEM f050d207 and planted truth. The pins are
//  CuboVirtualTests/Fixtures/com-dpc-pins.json; the real data are slice 2's exports in $CUBOKIT_REAL_DATA.
//

import Foundation
import Testing
import CuboKitTestSupport
import CuboCore
import CuboPhase
import CuboVirtual

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
    /// The reconstruction of a known phase gradient correlates with the phase at ≥ 0.99.
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
        let (mf, m) = try CenterOfMassReference.normalize(field, reference: .planeFit)
        #expect(m.row[4] == 0 && m.row.allSatisfy { abs($0) <= 1e-12 })
        // the plane is fitted to the finite positions only: it is the input plane everywhere, the NaN position included
        #expect(mf.row.indices.allSatisfy { abs(mf.row[$0] - plane.row[$0]) <= 1e-12 }, "the fitted plane must stay finite")
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

    /// Degrees become radians as NumPy's deg2rad and Python's math.radians do, deg · (π/180), asserted exact today
    /// (deg · π / 180 differs by an ulp for 38 of the 179 default angles); a non-finite field is refused.
    @Test func testRadiansAndNonFiniteFields() throws {
        let f = (Pins2.pins["fields"] as! [[String: Any]]).first { $0["theta"] as! Double == 23 }!
        let shape = f["shape"] as! [Int]
        let field = try ScanVectorField(scanRows: shape[0], scanColumns: shape[1],
                                        row: Pins2.doubles(f["measured_x"]), column: Pins2.doubles(f["measured_y"]))
        for deg in RotationSolver.defaultAnglesDegrees {
            let s = try RotationSolver.solve(field, forceRotationDegrees: deg, forceTranspose: false)
            #expect(s.radians.bitPattern == (deg * (Double.pi / 180)).bitPattern, "\(deg)")
        }
        var bad = field.row; bad[7] = .infinity
        #expect(throws: CuboPhaseError.self) {
            try RotationSolver.solve(try ScanVectorField(scanRows: shape[0], scanColumns: shape[1], row: bad, column: field.column))
        }
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
