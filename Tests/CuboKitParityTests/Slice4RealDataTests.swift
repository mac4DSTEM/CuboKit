//
//  Slice4RealDataTests.swift
//  Role: Slice 4 checks of the real cubes against the float64 pins in $CUBOKIT_REAL_DATA (skipped by name when absent).
//

import Foundation
import Testing
import CuboCore
import CuboVirtual
import CuboKitTestSupport

@Suite(.serialized) struct Slice4RealDataTests {
    static func run(_ name: String) async throws {
        let dir = Fixtures.realData
        let e = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("\(name)-s4-pins.json"))) as! [String: Any]
        try await S4.check(e, cube: try MappedCube(npy: dir.appendingPathComponent("\(name).npy")))
    }
    @Test(.enabled(if: S4.realIsPresent("bullseye"), "real data absent: run tools/slice-4/pins.py export-bullseye and real"))
    func testBullseyeVacuumProbeSlice4() async throws { try await Self.run("bullseye") }
    @Test(.enabled(if: S4.realIsPresent("almgsi"), "real data absent: run tools/slice-4/pins.py real"))
    func testAlMgSiStudyCubeSlice4() async throws { try await Self.run("almgsi") }
    @Test(.enabled(if: S4.realIsPresent("auref"), "real data absent: run tools/slice-4/pins.py real"))
    func testAuReferenceCubeSlice4() async throws { try await Self.run("auref") }
    @Test(.enabled(if: S4.realIsPresent("simau"), "real data absent: run tools/slice-4/pins.py real"))
    func testSimulatedPolycrystallineGoldSlice4() async throws { try await Self.run("simau") }
    @Test(.enabled(if: S4.realIsPresent("particle"), "real data absent: run tools/slice-4/pins.py real"))
    func testParticleCubeSlice4() async throws { try await Self.run("particle") }
    @Test(.enabled(if: S4.realIsPresent("py4dstem_small"), "real data absent: run tools/slice-4/pins.py real"))
    func testPy4DSTEMSmallDatacubeSlice4() async throws { try await Self.run("py4dstem_small") }
}
