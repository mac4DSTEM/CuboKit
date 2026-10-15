//
//  PolicyParameterTests.swift
//  The generality pass (0.2.0): the plane fit's degeneracy tolerance is a parameter, 1e-12 by default (tier 2).
//

import Testing
import CuboCore
import CuboPhase

@Suite struct CenterOfMassReferencePolicyTests {
    @Test func testDegeneracyToleranceIsAParameter() throws {
        let rows: [Double] = (0..<20).map { (k: Int) -> Double in 31 + 0.01 * Double(k / 5) }
        let columns: [Double] = (0..<20).map { (k: Int) -> Double in 32 - 0.02 * Double(k % 5) }
        let field = try ScanVectorField(scanRows: 4, scanColumns: 5, row: rows, column: columns)
        let byDefault = try CenterOfMassReference.normalize(field, reference: .planeFit)
        let explicit = try CenterOfMassReference.normalize(field, reference: .planeFit, degeneracyTolerance: 1e-12)
        #expect(byDefault.fitted.row == explicit.fitted.row && byDefault.normalized.column == explicit.normalized.column)
        // the determinant never exceeds Σu² · Σv²: a tolerance of 1 refuses every plane
        #expect(throws: CuboPhaseError.planeFitDegenerate) {
            try CenterOfMassReference.normalize(field, reference: .planeFit, degeneracyTolerance: 1)
        }
        // refuter 0.2.0, F1: a negative tolerance hid a degenerate fit behind nan_to_num; refused now
        for bad in [-1.0, -.infinity, .nan] {
            #expect(throws: CuboPhaseError.self) {
                try CenterOfMassReference.normalize(field, reference: .planeFit, degeneracyTolerance: bad)
            }
        }
    }
}
