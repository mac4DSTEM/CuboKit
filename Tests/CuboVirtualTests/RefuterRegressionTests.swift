//
//  RefuterRegressionTests.swift
//  The Gate B refuter's cases on 85318cd (2026-10-15; registration section 9), kept so they gate. F1: the Metal path missed
//  the registered relative 1e-5 (floor 1e-6 · max) by up to 8e-3 on zero-mean, dark-subtracted noise and on cancelling
//  data; fixed by compensated kernels. F3 and F4: the CPU mean turned ±∞ into NaN and dropped the sign of −0; Metal
//  dropped NaN from the maximum.
//

import Foundation
import Testing
import CuboKitTestSupport
import CuboCore
import CuboVirtual

/// Deterministic normal deviates: SplitMix64 and Box–Muller, so the cubes are the same on every run and machine.
struct Normals {
    var state: UInt64
    mutating func uniform() -> Double {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return Double((z ^ (z >> 31)) >> 11) / Double(1 << 53)
    }
    mutating func next(_ sigma: Double) -> Float {
        let u = max(uniform(), 1e-300), v = uniform()
        return Float(sigma * (-2 * log(u)).squareRoot() * cos(2 * .pi * v))
    }
}

@Suite(.enabled(if: VirtualDiffraction.metalIsAvailable, "no Metal device on this machine"))
struct RefuterRegressionTests {
    static let noiseShapes: [[Int]] = [[32, 32, 64, 64], [128, 128, 32, 32], [20, 20, 128, 128]]

    /// F1: pure N(0, 500) noise, no signal; plain and randomly weighted detectors; Metal within 1e-5 of the CPU path
    /// under the registered floor, the max exact.
    @Test(arguments: noiseShapes)
    func testMetalMeetsTheToleranceOnZeroMeanNoise(_ extents: [Int]) async throws {
        let shape = try CubeShape(extents)
        var g = Normals(state: UInt64(extents.reduce(17, *)))
        let cube = try MemoryCube(shape: shape, values: (0..<shape.elementCount).map { _ in g.next(500) })
        let qy = Double(shape.detectorRows), qx = Double(shape.detectorColumns)
        let geometries: [DetectorGeometry] = [
            .circle(centerRow: (qy - 1) / 2, centerColumn: (qx - 1) / 2, radius: 0.1 * qy),
            .annulus(centerRow: (qy - 1) / 2, centerColumn: (qx - 1) / 2, inner: 0.28 * qy, outer: 0.47 * qy),
            .circle(centerRow: 0.3 * qy, centerColumn: 0.6 * qx, radius: 3),
            .rectangle(rowMin: 0.2 * qy, rowMax: 0.8 * qy, columnMin: 0.2 * qx, columnMax: 0.8 * qx),
            .mask((0..<shape.detectorCount).map { _ in g.next(1) }),
        ]
        for geometry in geometries {
            let cpu = try await VirtualImaging.image(of: cube, geometry: geometry, path: .cpu)
            let gpu = try await VirtualImaging.image(of: cube, geometry: geometry, path: .metal)
            let d = assertRelative(gpu.values, cpu.values, 1e-5, "\(extents) image metal")
            print("refuter F1 \(extents) image: metal vs cpu \(d)")
        }
        let cpu = try await VirtualDiffraction.meanAndMax(of: cube, path: .cpu)
        let gpu = try await VirtualDiffraction.meanAndMax(of: cube, path: .metal, tiling: .rows(1))
        let d = assertRelative(gpu.mean.values, cpu.mean.values, 1e-5, "\(extents) mean metal")
        assertIdentical(gpu.max.values, cpu.max.values, "\(extents) max metal")
        print("refuter F1 \(extents) mean: metal vs cpu \(d)")
    }

    /// F1: cancelling data, ±1e6 pairs under an all-ones mask, and a bright disk on noise across several tiles.
    @Test func testMetalMeetsTheToleranceOnCancellingData() async throws {
        let shape = try CubeShape([6, 7, 32, 32])
        var g = Normals(state: 99)
        var values = [Float]()
        for _ in 0..<shape.scanCount {
            for k in 0..<(shape.detectorCount / 2) { let a = Float(1_000_000 + k) + g.next(3); values += [a, -a + g.next(50)] }
        }
        let cube = try MemoryCube(shape: shape, values: values)
        let ones = DetectorGeometry.mask([Float](repeating: 1, count: shape.detectorCount))
        let cpu = try await VirtualImaging.image(of: cube, geometry: ones, path: .cpu)
        let gpu = try await VirtualImaging.image(of: cube, geometry: ones, path: .metal)
        assertRelative(gpu.values, cpu.values, 1e-5, "cancelling pairs")

        let disk = try CubeShape([48, 48, 64, 64])
        var h = Normals(state: 5)
        var v = [Float]()
        for _ in 0..<disk.scanCount {
            for i in 0..<64 { for j in 0..<64 {
                let r2 = Double((i - 31) * (i - 31) + (j - 33) * (j - 33))
                v.append(h.next(500) + (r2 < 64 ? 1e6 : 0))
            } }
        }
        let bright = try MemoryCube(shape: disk, values: v)
        for tiling in [ScanTiling.rows(1), .rows(7), .automatic] {
            let c = try await VirtualDiffraction.meanAndMax(of: bright, path: .cpu, tiling: tiling)
            let m = try await VirtualDiffraction.meanAndMax(of: bright, path: .metal, tiling: tiling)
            assertRelative(m.mean.values, c.mean.values, 1e-5, "disk mean \(tiling)")
            for geometry: DetectorGeometry in [.annulus(centerRow: 31, centerColumn: 33, inner: 9, outer: 30),
                                               .circle(centerRow: 10, centerColumn: 50, radius: 6)] {
                let a = try await VirtualImaging.image(of: bright, geometry: geometry, path: .cpu, tiling: tiling)
                let b = try await VirtualImaging.image(of: bright, geometry: geometry, path: .metal, tiling: tiling)
                assertRelative(b.values, a.values, 1e-5, "disk \(geometry) \(tiling)")
            }
        }
    }

    /// F3, F4: ±∞ stays ∞ in the CPU mean (NumPy's answer), −0 keeps its sign, and both paths propagate NaN in the max.
    @Test func testNonFiniteAndSignedZeroBehaviour() async throws {
        let shape = try CubeShape([2, 1, 1, 4])
        let cube = try MemoryCube(shape: shape, values: [.infinity, -0.0, .nan, 1, 2, -0.0, 3, -.infinity])
        let cpu = try await VirtualDiffraction.meanAndMax(of: cube, path: .cpu)
        #expect(cpu.mean.values[0] == .infinity)
        #expect(cpu.mean.values[1] == 0 && cpu.mean.values[1].sign == .minus)
        #expect(cpu.mean.values[2].isNaN && cpu.max.values[2].isNaN)
        #expect(cpu.mean.values[3] == -.infinity)
        let gpu = try await VirtualDiffraction.meanAndMax(of: cube, path: .metal)
        #expect(gpu.max.values[2].isNaN)
        #expect(gpu.max.values[0] == .infinity && gpu.max.values[3] == 1)
    }
}
