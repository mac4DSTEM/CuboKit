//
//  Slice3RealDataTests.swift
//  Role: Slice 3 checks of the real cubes against the float64 pins in $CUBOKIT_REAL_DATA (skipped by name when absent).
//

import Foundation
import Testing
import CuboCore
import CuboVirtual
import CuboKitTestSupport

@Suite(.serialized) struct Slice3RealDataTests {
    static func run(_ name: String) async throws {
        let dir = Fixtures.realData
        let e = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("\(name)-s3-pins.json"))) as! [String: Any]
        let cube = try MappedCube(npy: dir.appendingPathComponent("\(name).npy"))
        try await S3.check(e, cube: cube, integer: cube.elementType != .float32 && cube.elementType != .float64)
    }
    @Test(.enabled(if: S3.realIsPresent("almgsi"), "real data absent: run tools/origin-fit/pins.py real"))
    func testAlMgSiStudyCubeSlice3() async throws { try await Self.run("almgsi") }
    @Test(.enabled(if: S3.realIsPresent("auref"), "real data absent: run tools/origin-fit/pins.py real"))
    func testAuReferenceCubeSlice3() async throws { try await Self.run("auref") }
    @Test(.enabled(if: S3.realIsPresent("simau"), "real data absent: run tools/origin-fit/pins.py real"))
    func testSimulatedPolycrystallineGoldSlice3() async throws { try await Self.run("simau") }
    @Test(.enabled(if: S3.realIsPresent("particle"), "real data absent: run tools/origin-fit/pins.py real"))
    func testParticleCubeSlice3() async throws { try await Self.run("particle") }
    @Test(.enabled(if: S3.realIsPresent("py4dstem_small"), "real data absent: run tools/origin-fit/pins.py real"))
    func testPy4DSTEMSmallDatacubeSlice3() async throws { try await Self.run("py4dstem_small") }
}
