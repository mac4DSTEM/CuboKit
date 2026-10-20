//
//  PolicyParameterTests.swift
//  The generality pass (0.2.0): a policy the kit used to fix is a parameter whose default is the old value. The default
//  path is pinned by the slice tests (D12); these check that the parameter takes effect and is validated (tier 2).
//

import Testing
import CuboCore

@Suite struct OriginFitPolicyTests {
    @Test func testRankToleranceIsAParameter() throws {
        let plane: [Double] = (0..<12).map { (k: Int) -> Double in
            let r = Double(k / 4), c = Double(k % 4)
            return 0.3 + 0.02 * r - 0.07 * c
        }
        let byDefault = try OriginFit.fit(plane, rows: 3, columns: 4, model: .plane)
        let explicit = try OriginFit.fit(plane, rows: 3, columns: 4, model: .plane, rankTolerance: 1e-12)
        #expect(byDefault == explicit)
        // a tolerance of 1 finds every column negligible: nothing is left to fit, refused
        #expect(throws: OriginFitError.underdetermined(.plane)) {
            try OriginFit.fit(plane, rows: 3, columns: 4, model: .plane, rankTolerance: 1)
        }
        #expect(throws: OriginFitError.self) { try OriginFit.fit(plane, rows: 3, columns: 4, model: .plane, rankTolerance: -1) }
    }
}
