//
//  ProbesRealDataTests.swift
//  Role: Median, selection, Friedel-origin, beamstop and vacuum-probe checks of the real cubes against the float64 pins in $CUBOKIT_REAL_DATA (skipped by name when absent).
//

import Foundation
import Testing
import CuboCore
import CuboVirtual
import CuboKitTestSupport

@Suite(.serialized) struct ProbesRealDataTests {
    static func run(_ name: String) async throws {
        let dir = Fixtures.realData
        let e = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("\(name)-s4-pins.json"))) as! [String: Any]
        try await ProbesPins.check(e, cube: try MappedCube(npy: dir.appendingPathComponent("\(name).npy")))
    }
    @Test(.enabled(if: ProbesPins.realIsPresent("bullseye"), "real data absent: run tools/probes/pins.py export-bullseye and real"))
    func testBullseyeVacuumProbe() async throws { try await Self.run("bullseye") }
    @Test(.enabled(if: ProbesPins.realIsPresent("almgsi"), "real data absent: run tools/probes/pins.py real"))
    func testAlMgSiStudyCube() async throws { try await Self.run("almgsi") }
    @Test(.enabled(if: ProbesPins.realIsPresent("auref"), "real data absent: run tools/probes/pins.py real"))
    func testAuReferenceCube() async throws { try await Self.run("auref") }
    @Test(.enabled(if: ProbesPins.realIsPresent("simau"), "real data absent: run tools/probes/pins.py real"))
    func testSimulatedPolycrystallineGold() async throws { try await Self.run("simau") }
    @Test(.enabled(if: ProbesPins.realIsPresent("particle"), "real data absent: run tools/probes/pins.py real"))
    func testParticleCube() async throws { try await Self.run("particle") }
    @Test(.enabled(if: ProbesPins.realIsPresent("py4dstem_small"), "real data absent: run tools/probes/pins.py real"))
    func testPy4DSTEMSmallDatacube() async throws { try await Self.run("py4dstem_small") }
}
