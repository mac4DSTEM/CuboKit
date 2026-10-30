//
//  OriginFitRealDataTests.swift
//  Role: Origin and origin-fit checks of the real cubes against the float64 pins in $CUBOKIT_REAL_DATA (skipped by name when absent).
//

import Foundation
import Testing
import CuboCore
import CuboVirtual
import CuboKitTestSupport

@Suite(.serialized) struct OriginFitRealDataTests {
    static func run(_ name: String) async throws {
        let dir = Fixtures.realData
        let e = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("\(name)-s3-pins.json"))) as! [String: Any]
        let cube = try MappedCube(npy: dir.appendingPathComponent("\(name).npy"))
        try await OriginFitPins.check(e, cube: cube, integer: cube.elementType != .float32 && cube.elementType != .float64)
    }
    @Test(.enabled(if: OriginFitPins.realIsPresent("almgsi"), "real data absent: run tools/origin-fit/pins.py real"))
    func testAlMgSiStudyCube() async throws { try await Self.run("almgsi") }
    @Test(.enabled(if: OriginFitPins.realIsPresent("auref"), "real data absent: run tools/origin-fit/pins.py real"))
    func testAuReferenceCube() async throws { try await Self.run("auref") }
    @Test(.enabled(if: OriginFitPins.realIsPresent("simau"), "real data absent: run tools/origin-fit/pins.py real"))
    func testSimulatedPolycrystallineGold() async throws { try await Self.run("simau") }
    @Test(.enabled(if: OriginFitPins.realIsPresent("particle"), "real data absent: run tools/origin-fit/pins.py real"))
    func testParticleCube() async throws { try await Self.run("particle") }
    @Test(.enabled(if: OriginFitPins.realIsPresent("py4dstem_small"), "real data absent: run tools/origin-fit/pins.py real"))
    func testPy4DSTEMSmallDatacube() async throws { try await Self.run("py4dstem_small") }
}
