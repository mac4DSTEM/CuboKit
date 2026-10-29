//
//  CuboPhaseRealDataTests.swift
//  Role: Centre-of-mass, rotation, DPC and probe checks on the real cubes in $CUBOKIT_REAL_DATA (skipped by name when absent).
//

import Foundation
import Testing
import CuboCore
import CuboPhase
import CuboVirtual
import CuboKitTestSupport

/// Serialized: each test maps a real cube; real data absent → skipped by name (tools/test.sh then writes no stamp).
@Suite(.serialized) struct CuboPhaseRealDataTests {
    static func check(_ name: String) async throws {
        let dir = Fixtures.realData
        let info = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("\(name)-s2-pins.json"))) as! [String: Any]
        func a(_ key: String) throws -> [Double] { try NPYFile.readDoubles(dir.appendingPathComponent("\(name)-s2-\(key).npy")).values }
        let cube = try MappedCube(npy: dir.appendingPathComponent("\(name).npy"))
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("\(name)-pins.json"))) as! [String: Any]
        let bfSpec = ((meta["geometries"] as! [[String: Any]]).first { $0["name"] as! String == "bf" }!["geometry"]) as! [String: Any]
        let bf = DetectorGeometry.circle(centerRow: bfSpec["centerRow"] as! Double, centerColumn: bfSpec["centerColumn"] as! Double,
                                         radius: bfSpec["radius"] as! Double)
        var log = ["real com-dpc \(name)"]
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
            #expect(p.radius.bitPattern == ComDPCPins.double(e["radius"]).bitPattern
                    && p.centerRow.bitPattern == ComDPCPins.double(e["center_row"]).bitPattern
                    && p.centerColumn.bitPattern == ComDPCPins.double(e["center_column"]).bitPattern, "\(name) probe \(kind)")
            log.append("probe \(kind): r \(p.radius) at (\(p.centerRow), \(p.centerColumn))")
        }
        print(log.joined(separator: "\n  "))
    }

    @Test(.enabled(if: ComDPCPins.realIsPresent("almgsi"), "real data absent: run tools/com-dpc/pins.py real"))
    func testAlMgSiStudyCube() async throws { try await Self.check("almgsi") }
    @Test(.enabled(if: ComDPCPins.realIsPresent("auref"), "real data absent: run tools/com-dpc/pins.py real"))
    func testAuReferenceCube() async throws { try await Self.check("auref") }
    @Test(.enabled(if: ComDPCPins.realIsPresent("simau"), "real data absent: run tools/com-dpc/pins.py real"))
    func testSimulatedPolycrystallineGold() async throws { try await Self.check("simau") }
    @Test(.enabled(if: ComDPCPins.realIsPresent("particle"), "real data absent: run tools/com-dpc/pins.py real"))
    func testParticleCube() async throws { try await Self.check("particle") }
    @Test(.enabled(if: ComDPCPins.realIsPresent("py4dstem_small"), "real data absent: run tools/com-dpc/pins.py real"))
    func testPy4DSTEMSmallDatacube() async throws { try await Self.check("py4dstem_small") }
}
