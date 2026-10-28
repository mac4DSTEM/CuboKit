//
//  RealDataTests.swift
//  Registration 5.3, 5.4, gate 3: real cubes and py4DSTEM's test datacube, exported as .npy with their pins by
//  `tools/slice-1/pins.py real` into the real-data directory OUTSIDE the repository. Skipped by name when it is absent,
//  so tools/test.sh is green on a clone and complete on a machine that has the exports. Every distance measured is printed, so the
//  run's log is the evidence docs/status.md cites.
//

import Foundation
import Testing
import CuboKitTestSupport
import CuboCore
import CuboVirtual

/// Serialized: each test maps a cube of up to 1.8 GB and holds an eager copy; one at a time keeps memory and the timings
/// clean.
@Suite(.serialized) struct RealDataTests {
    struct Meta: Decodable {
        let name: String
        let shape: [Int]
        let dtype: String
        let exactInFloat32: Bool
        let meanDistancePy4dstemOwnVsF64: Double
        let geometries: [G]
        struct G: Decodable {
            let name: String; let geometry: GeometrySpec; let maskCount: Int
            let imageDistancePy4dstemOwnVsF64: Double
            enum CodingKeys: String, CodingKey {
                case name, geometry
                case maskCount = "mask_count"
                case imageDistancePy4dstemOwnVsF64 = "image_distance_py4dstem_own_vs_f64"
            }
        }
        enum CodingKeys: String, CodingKey {
            case name, shape, dtype, geometries
            case exactInFloat32 = "exact_in_float32"
            case meanDistancePy4dstemOwnVsF64 = "mean_distance_py4dstem_own_vs_f64"
        }
        var isInteger: Bool { dtype.hasPrefix("uint") || dtype.hasPrefix("int") }
    }

    func load(_ name: String) throws -> (Meta, MappedCube, (String) throws -> [Double]) {
        let dir = Fixtures.realData
        let pins = dir.appendingPathComponent("\(name)-pins.json")
        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: pins))
        let cube = try MappedCube(npy: dir.appendingPathComponent("\(name).npy"))
        #expect(cube.shape.extents == meta.shape)
        let array: (String) throws -> [Double] = { key in
            try NPYFile.readDoubles(dir.appendingPathComponent("\(name)-\(key).npy")).values
        }
        return (meta, cube, array)
    }

    func check(_ name: String) async throws {
        let (meta, cube, array) = try load(name)
        let clock = ContinuousClock()
        var log = ["real \(name) \(meta.dtype) \(cube.shape)"]

        let t0 = clock.now
        let cpu = try await VirtualDiffraction.meanAndMax(of: cube, path: .cpu)
        log.append("cpu mean+max \(clock.now - t0)")
        assertIdentical(cpu.max.values, try array("max"), "\(name) max")
        let dMean = assertRelative(cpu.mean.values, try array("mean_f64"), 1e-10, "\(name) mean against float64")
        if meta.isInteger && meta.exactInFloat32 {
            assertIdentical(cpu.mean.values, try array("mean_py4dstem_native"), "\(name) mean against py4DSTEM")
        }
        log.append("mean: kit vs float64 \(dMean), py4DSTEM own vs float64 \(meta.meanDistancePy4dstemOwnVsF64)")

        let oneRow = try await VirtualDiffraction.meanAndMax(of: cube, path: .cpu, tiling: .rows(1))
        assertIdentical(oneRow.mean.values, cpu.mean.values, "\(name) mean, one row per tile")

        let metal = VirtualDiffraction.metalIsAvailable
        if metal {
            let t1 = clock.now
            let gpu = try await VirtualDiffraction.meanAndMax(of: cube, path: .metal)
            log.append("metal mean+max \(clock.now - t1)")
            assertIdentical(gpu.max.values, cpu.max.values, "\(name) max metal")
            let d = assertRelative(gpu.mean.values, cpu.mean.values, 1e-5, "\(name) mean metal")
            log.append("mean: metal vs cpu \(d)")
        }

        for g in meta.geometries {
            let geometry = g.geometry.geometry
            #expect(try DetectorMask.make(geometry, shape: cube.shape).filter { $0 != 0 }.count == g.maskCount)
            let t2 = clock.now
            let image = try await VirtualImaging.image(of: cube, geometry: geometry)
            let tCPU = clock.now - t2
            let d = assertRelative(image.values, try array("\(g.name)-image_f64"), 1e-10, "\(name)/\(g.name) against float64")
            if meta.isInteger && meta.exactInFloat32 {
                assertIdentical(image.values, try array("\(g.name)-image_py4dstem_native"), "\(name)/\(g.name) against py4DSTEM")
            }
            var line = "\(g.name) (\(g.maskCount) px): kit vs float64 \(d), py4DSTEM own vs float64 "
                + "\(g.imageDistancePy4dstemOwnVsF64), cpu \(tCPU)"
            if metal {
                let t3 = clock.now
                let m = try await VirtualImaging.image(of: cube, geometry: geometry, path: .metal)
                let tGPU = clock.now - t3
                let dm = assertRelative(m.values, image.values, 1e-5, "\(name)/\(g.name) metal")
                line += ", metal vs cpu \(dm), metal \(tGPU)"
            }
            log.append(line)
        }

        // Eager equals lazy, exactly, on the real cube.
        let eager = try await MemoryCube(loading: cube)
        let e = try await VirtualDiffraction.meanAndMax(of: eager, path: .cpu)
        assertIdentical(e.mean.values, cpu.mean.values, "\(name) mean eager")
        if let first = meta.geometries.first {
            let a = try await VirtualImaging.image(of: eager, geometry: first.geometry.geometry)
            let b = try await VirtualImaging.image(of: cube, geometry: first.geometry.geometry)
            assertIdentical(a.values, b.values, "\(name)/\(first.name) eager")
        }
        log.append("eager = lazy: checked")
        print(log.joined(separator: "\n  "))
    }

    // Enabled only where the real-data directory holds the export; a skipped run says "real data absent" (tools/test.sh
    // reads that and writes no stamp).
    @Test(.enabled(if: realDataIsPresent("almgsi"), "real data absent: run tools/slice-1/pins.py real"))
    func testAlMgSiStudyCube() async throws { try await check("almgsi") }
    @Test(.enabled(if: realDataIsPresent("auref"), "real data absent: run tools/slice-1/pins.py real"))
    func testAuReferenceCube() async throws { try await check("auref") }
    @Test(.enabled(if: realDataIsPresent("simau"), "real data absent: run tools/slice-1/pins.py real"))
    func testSimulatedPolycrystallineGold() async throws { try await check("simau") }
    @Test(.enabled(if: realDataIsPresent("particle"), "real data absent: run tools/slice-1/pins.py real"))
    func testParticleCube() async throws { try await check("particle") }
    @Test(.enabled(if: realDataIsPresent("py4dstem_small"), "real data absent: run tools/slice-1/pins.py real"))
    func testPy4DSTEMSmallDatacube() async throws { try await check("py4dstem_small") }
}
