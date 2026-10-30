//
//  RotationSolverParallelTests.swift
//  `RotationSolver.curve` and the parallel angle grid in `solve`: exactly equal to `solve`'s own curves, and the
//  cancellation. Tier 1 (the numbers must not move).
//

import Foundation
import Synchronization
import Testing
import CuboKitTestSupport
import CuboCore
import CuboPhase

private func bits(_ v: [Double]) -> [UInt64] { v.map { (d: Double) -> UInt64 in d.bitPattern } }

private func randomField(_ rows: Int, _ columns: Int, seed: UInt64) throws -> ScanVectorField {
    var s = seed
    func next() -> Double {
        s = s &* 6364136223846793005 &+ 1442695040888963407
        return Double(s >> 11) / Double(1 << 53) - 0.5
    }
    let a = (0..<(rows * columns)).map { (_: Int) -> Double in next() }
    let b = (0..<(rows * columns)).map { (_: Int) -> Double in next() }
    return try ScanVectorField(scanRows: rows, scanColumns: columns, row: a, column: b)
}

private func pinnedFields() throws -> [(String, ScanVectorField)] {
    try (ComDPCPins.pins["fields"] as! [[String: Any]]).map { (f: [String: Any]) -> (String, ScanVectorField) in
        let shape = f["shape"] as! [Int]
        let field = try ScanVectorField(scanRows: shape[0], scanColumns: shape[1],
                                        row: ComDPCPins.doubles(f["measured_x"]), column: ComDPCPins.doubles(f["measured_y"]))
        return ("field \(f["theta"]!)", field)
    }
}

@Suite struct RotationSolverParallelTests {
    private func expectCurvesMatch(_ field: ScanVectorField, _ name: String) throws {
        for criterion in [RotationCriterion.minimizeCurl, .maximizeDivergence] {
            let solution = try RotationSolver.solve(field, criterion: criterion)
            let angles = RotationSolver.defaultAnglesDegrees
            let plain = try RotationSolver.curve(field, anglesDegrees: angles, criterion: criterion, transpose: false)
            let trans = try RotationSolver.curve(field, anglesDegrees: angles, criterion: criterion, transpose: true)
            #expect(!plain.isEmpty && plain.count == angles.count)
            #expect(bits(plain) == bits(solution.curve), "\(name) \(criterion): curve")
            #expect(bits(trans) == bits(solution.curveTransposed), "\(name) \(criterion): curveTransposed")
            // a closure that never cancels changes nothing
            let again = try RotationSolver.solve(field, criterion: criterion, isCancelled: { false })
            #expect(bits(again.curve) == bits(solution.curve) && bits(again.curveTransposed) == bits(solution.curveTransposed))
            #expect(again.degrees == solution.degrees && again.transposed == solution.transposed)
            // a short grid, in a different order from the default
            let few: [Double] = [41, -89, 0, 23.5, 88]
            let fewPlain = try RotationSolver.curve(field, anglesDegrees: few, criterion: criterion, transpose: false)
            let forced = try few.map { (a: Double) -> Double in
                try RotationSolver.solve(field, anglesDegrees: [a], criterion: criterion, forceTranspose: false).curve[0]
            }
            #expect(bits(fewPlain) == bits(forced), "\(name) \(criterion): short grid")
        }
    }

    @Test func testCurveEqualsSolveOnThePinnedFields() throws {
        let fields = try pinnedFields()
        #expect(!fields.isEmpty)
        for (name, field) in fields { try expectCurvesMatch(field, name) }
    }

    @Test func testCurveEqualsSolveOnARandomField() throws {
        try expectCurvesMatch(try randomField(37, 41, seed: 20261015), "random 37x41")
    }

    @Test func testCurveRejectsWhatSolveRejects() throws {
        let field = try randomField(5, 6, seed: 1)
        #expect(throws: CuboPhaseError.invalidAngles) {
            try RotationSolver.curve(field, anglesDegrees: [], transpose: false)
        }
        #expect(throws: CuboPhaseError.invalidAngles) {
            try RotationSolver.curve(field, anglesDegrees: [1, .nan], transpose: true)
        }
        let small = try randomField(2, 5, seed: 2)
        #expect(throws: CuboPhaseError.scanTooSmallForRotation(rows: 2, columns: 5)) {
            try RotationSolver.curve(small, anglesDegrees: [0], transpose: false)
        }
    }

    @Test func testCancellationThrowsAndNeverReturnsAPartialResult() throws {
        let field = try randomField(37, 41, seed: 7)
        #expect(throws: CancellationError.self) { try RotationSolver.solve(field, isCancelled: { true }) }
        #expect(throws: CancellationError.self) {
            try RotationSolver.solve(field, forceTranspose: true, isCancelled: { true })
        }
        #expect(throws: CancellationError.self) {
            try RotationSolver.solve(field, forceRotationDegrees: 10, isCancelled: { true })
        }
        #expect(throws: CancellationError.self) {
            try RotationSolver.curve(field, anglesDegrees: RotationSolver.defaultAnglesDegrees, transpose: false,
                                     isCancelled: { true })
        }
        // cancelled part-way: the 50th question onwards says yes
        let asked = Atomic<Int>(0)
        #expect(throws: CancellationError.self) {
            try RotationSolver.curve(field, anglesDegrees: RotationSolver.defaultAnglesDegrees, transpose: true,
                                     isCancelled: { asked.add(1, ordering: .relaxed).newValue > 50 })
        }
        #expect(asked.load(ordering: .relaxed) > 50)
    }
}
