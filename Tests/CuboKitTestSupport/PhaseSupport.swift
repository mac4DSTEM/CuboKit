//
//  PhaseSupport.swift
//  Role: Slice 2 pins (Pins2), the near/ofMax comparisons and the chain check, shared by the fast tests and the real-data parity suites.
//

import Foundation
import Testing
import CuboCore
import CuboVirtual
import CuboPhase

package enum Pins2 {
    package static let fixtures = FixtureDirs.fixtures(of: "CuboVirtualTests")
    package nonisolated(unsafe) static let pins: [String: Any] = {
        try! JSONSerialization.jsonObject(with: Data(contentsOf: fixtures.appendingPathComponent("slice-2-pins.json")))
            as! [String: Any]
    }()
    package static func doubles(_ any: Any?) -> [Double] { (any as? [String] ?? []).map { Double($0)! } }
    package static func double(_ any: Any?) -> Double { Double(any as! String)! }
    package static var realData: URL {
        if let env = ProcessInfo.processInfo.environment["CUBOKIT_REAL_DATA"], !env.isEmpty { return URL(fileURLWithPath: env) }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Caches/CuboKit/real-data")
    }
    package static func realIsPresent(_ name: String) -> Bool {
        FileManager.default.fileExists(atPath: realData.appendingPathComponent("\(name)-s2-pins.json").path)
    }
}

/// Relative with slice 1's floor (1e-6 · max|want|).
@discardableResult
package func near(_ got: [Double], _ want: [Double], _ t: Double, _ what: String,
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
package func ofMax(_ got: [Double], _ want: [Double], _ t: Double, _ what: String,
           sourceLocation: SourceLocation = SourceLocation(fileID: #fileID, filePath: #filePath, line: #line, column: #column)) -> Double {
    #expect(got.count == want.count, "\(what): count", sourceLocation: sourceLocation)
    guard got.count == want.count, !want.isEmpty else { return 0 }
    let d = zip(got, want).map { abs($0 - $1) }.max()! / max(want.map(abs).max()!, .leastNormalMagnitude)
    #expect(d <= t, "\(what): \(d) > \(t)", sourceLocation: sourceLocation)
    return d
}

/// Checks one chain (fit, rotation, DPC) against the float64 chain and py4DSTEM's own decisions.
package func checkChain(measured: ScanVectorField, reference: CenterOfMassReference, pins ch: [String: Any], _ name: String) throws {
    let (fitted, normalized) = try CenterOfMassReference.normalize(measured, reference: reference)
    if case .planeFit = reference {
        near(fitted.row, Pins2.doubles(ch["fitted_x"]), 1e-10, "\(name) fitted row")
        near(fitted.column, Pins2.doubles(ch["fitted_y"]), 1e-10, "\(name) fitted column")
        // py4DSTEM's curve_fit plane is recorded, not asserted (up to 8.3e-6 px from the exact
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
