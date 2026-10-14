//
//  PinsTests.swift
//  Agreement with py4DSTEM f050d207 on the synthetic cubes (registration 3.4, 4, 5.1, 5.2), read from slice-1-pins.json.
//  Agreement is agreement, not truth (TruthTests); a difference is a failure of the port, never a tolerance to widen.
//

import CryptoKit
import Foundation
import Testing
import CuboCore
import CuboVirtual

@Suite struct PinsTests {
    let pins = Pins.shared

    @Test func testPinsCarryTheirVersionsAndEveryCube() {
        #expect(pins.versions["py4dstem"] == "0.14.19")
        #expect(pins.versions["py4dstem_commit"] == "f050d207851f8accd3aa550365baaf1f38230428")
        #expect(pins.versions["numpy"] == "2.5.0")
        #expect(pins.versions["h5py"] == "3.11.0")
        #expect(pins.seed == 20261014)
        #expect(pins.cubes.map(\.name) == ["digits", "gaussian", "counts16", "counts32", "thirds", "negative", "boundary"])
        #expect(pins.cubes.reduce(0) { $0 + $1.geometries.count } == 71)
        #expect(pins.boundaryPixelsFloat32Disagrees >= 1)
    }

    @Test func testFixturesAreTheGeneratorsBytes() throws {
        for cube in pins.cubes {
            let digest = SHA256.hash(data: try Data(contentsOf: cube.url)).map { String(format: "%02x", $0) }.joined()
            #expect(digest == cube.sha256, "\(cube.name)")
            let mapped = try cube.mapped()
            #expect(mapped.shape.extents == cube.shape, "\(cube.name)")
            #expect(mapped.elementType.rawValue == cube.dtype, "\(cube.name)")
        }
    }

    /// Registration 3.1: the kit's mask selects exactly make_detector's pixels, with the given weights for `.mask`.
    @Test func testMasksAreMakeDetectors() throws {
        for cube in pins.cubes {
            let shape = try CubeShape(cube.shape)
            for g in cube.geometries {
                let mask = try DetectorMask.make(g.geometry.geometry, shape: shape)
                let selected = mask.indices.filter { mask[$0] != 0 }
                #expect(selected == g.maskPixels, "\(cube.name)/\(g.name)")
                #expect(selected.count == g.maskCount, "\(cube.name)/\(g.name)")
                if case .mask(let weights) = g.geometry.geometry { #expect(mask == weights) }
            }
        }
    }

    /// Registration 3.4: CPU virtual images. Integer cubes bit for bit against py4DSTEM's own call; every cube within
    /// relative 1e-10 of the float64 evaluation; py4DSTEM's own float32 distance printed beside the kit's.
    @Test func testVirtualImagesOnTheCPUAgreeWithPy4DSTEM() async throws {
        for cube in pins.cubes {
            let source = try cube.mapped()
            for g in cube.geometries {
                let image = try await VirtualImaging.image(of: source, geometry: g.geometry.geometry)
                #expect(image.path == .cpu)
                #expect([image.scanRows, image.scanColumns] == Array(cube.shape.prefix(2)))
                let kit = assertRelative(image.values, doubles(g.imageF64), 1e-10, "\(cube.name)/\(g.name) against float64")
                if cube.isInteger && cube.exactInFloat32 {
                    assertIdentical(image.values, doubles(g.imagePy4dstemNative), "\(cube.name)/\(g.name) against py4DSTEM")
                }
                print("pins \(cube.name)/\(g.name): kit vs float64 \(kit); py4DSTEM own vs float64 "
                      + "\(g.imageDistancePy4dstemOwnVsF64)")
            }
        }
    }

    /// Registration 3.4: CPU mean (integer cubes bit for bit against py4DSTEM's get_dp_mean; float cubes within 1e-10 of
    /// the float64 mean) and max (bit for bit everywhere).
    @Test func testMeanAndMaxOnTheCPUAgreeWithPy4DSTEM() async throws {
        for cube in pins.cubes {
            let source = try cube.mapped()
            let (mean, max) = try await VirtualDiffraction.meanAndMax(of: source)
            assertIdentical(max.values, doubles(cube.max), "\(cube.name) max")
            let kit = assertRelative(mean.values, doubles(cube.meanF64), 1e-10, "\(cube.name) mean against float64")
            if cube.isInteger && cube.exactInFloat32 {
                assertIdentical(mean.values, doubles(cube.meanPy4dstemNative), "\(cube.name) mean against py4DSTEM")
                assertIdentical(mean.values, doubles(cube.meanF64), "\(cube.name) mean against float64")
            }
            print("pins \(cube.name) mean: kit vs float64 \(kit); py4DSTEM own vs float64 \(cube.meanDistancePy4dstemOwnVsF64)")
        }
    }

    /// Registration 3.4: Metal within relative 1e-5 of the CPU path; max bit for bit.
    @Test(.enabled(if: VirtualDiffraction.metalIsAvailable, "no Metal device on this machine"))
    func testMetalAgreesWithTheCPUWithinTheRegisteredTolerance() async throws {
        for cube in pins.cubes {
            let source = try cube.mapped()
            for tiling in [ScanTiling.automatic, .rows(1)] {
                let cpu = try await VirtualDiffraction.meanAndMax(of: source, path: .cpu, tiling: tiling)
                let gpu = try await VirtualDiffraction.meanAndMax(of: source, path: .metal, tiling: tiling)
                #expect(gpu.mean.path == .metal)
                assertIdentical(gpu.max.values, cpu.max.values, "\(cube.name) max metal \(tiling)")
                assertRelative(gpu.mean.values, cpu.mean.values, 1e-5, "\(cube.name) mean metal \(tiling)")
                for g in cube.geometries {
                    let c = try await VirtualImaging.image(of: source, geometry: g.geometry.geometry, path: .cpu,
                                                           tiling: tiling)
                    let m = try await VirtualImaging.image(of: source, geometry: g.geometry.geometry, path: .metal,
                                                           tiling: tiling)
                    assertRelative(m.values, c.values, 1e-5, "\(cube.name)/\(g.name) metal \(tiling)")
                }
            }
        }
    }

    /// Registration 3.4: the eager cube and the lazy one of the same bytes give the same bits, on both paths, and the CPU
    /// path gives the same bits at every tiling (the Metal virtual image too: it is per position).
    @Test func testLazyEagerAndEveryTilingGiveTheSameBits() async throws {
        let paths: [ComputePath] = VirtualDiffraction.metalIsAvailable ? [.cpu, .metal] : [.cpu]
        for cube in pins.cubes {
            let lazy = try cube.mapped()
            let eager = try await MemoryCube(loading: lazy, tiling: .rows(1))
            for path in paths {
                let reference = try await VirtualDiffraction.meanAndMax(of: lazy, path: path, tiling: .automatic)
                let fromEager = try await VirtualDiffraction.meanAndMax(of: eager, path: path, tiling: .automatic)
                assertIdentical(fromEager.mean.values, reference.mean.values, "\(cube.name) mean eager \(path)")
                assertIdentical(fromEager.max.values, reference.max.values, "\(cube.name) max eager \(path)")
                for g in cube.geometries {
                    let a = try await VirtualImaging.image(of: lazy, geometry: g.geometry.geometry, path: path)
                    let b = try await VirtualImaging.image(of: eager, geometry: g.geometry.geometry, path: path)
                    assertIdentical(b.values, a.values, "\(cube.name)/\(g.name) eager \(path)")
                    for rows in [1, 2] {
                        let t = try await VirtualImaging.image(of: lazy, geometry: g.geometry.geometry, path: path,
                                                               tiling: .rows(rows))
                        assertIdentical(t.values, a.values, "\(cube.name)/\(g.name) rows \(rows) \(path)")
                    }
                }
            }
            for rows in [1, 2, 3] {
                let t = try await VirtualDiffraction.meanAndMax(of: lazy, path: .cpu, tiling: .rows(rows))
                let a = try await VirtualDiffraction.meanAndMax(of: lazy, path: .cpu)
                assertIdentical(t.mean.values, a.mean.values, "\(cube.name) mean cpu rows \(rows)")
                assertIdentical(t.max.values, a.max.values, "\(cube.name) max cpu rows \(rows)")
            }
        }
    }
}
