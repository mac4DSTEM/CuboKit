//
//  IteratedWindowOriginTests.swift
//  The iterated-window origin estimator: the Metal kernel against the Swift Float32 reference (relative 1e-5, as the kit's
//  other Metal pins), recovery of a planted origin, and the window sensitivity on cases with a known answer.
//

import Foundation
import Testing
import CuboCore
import CuboVirtual

@Suite struct IteratedWindowOriginTests {
    /// A scan of `rows × columns` patterns of `q × q`: a Gaussian-edged disk of radius 6 centred at (c0 + dr, c0 + dc)
    /// with a per-position sub-pixel offset, a small background; `ring` adds a ring outside the core (a bullseye).
    static func cube(rows: Int = 4, columns: Int = 5, q: Int = 40, ring: Bool = false, nan: Bool = false) throws
        -> (cube: MemoryCube, truth: [(Double, Double)]) {
        var values = [Float](); var truth = [(Double, Double)]()
        for sr in 0..<rows { for sc in 0..<columns {
            let cr = 19.3 + 0.21 * Double(sr) - 0.07 * Double(sc), cc = 20.6 - 0.13 * Double(sr) + 0.17 * Double(sc)
            truth.append((cr, cc))
            for i in 0..<q { for j in 0..<q {
                let d = ((Double(i) - cr) * (Double(i) - cr) + (Double(j) - cc) * (Double(j) - cc)).squareRoot()
                var v = 100 / (1 + exp((d - 6) / 0.5)) + 1 + 0.01 * Double((i * 7 + j * 3 + sr + sc) % 5)
                if ring { v += 40 / (1 + exp((abs(d - 14) - 1.5) / 0.4)) }
                values.append(Float(v))
            } }
            if nan {   // a few non-finite pixels inside and beside the disk, one in a corner
                let base = values.count - q * q
                values[base + 19 * q + 20] = .nan; values[base + 21 * q + 18] = .infinity; values[base + 3] = -.infinity
                values[base] = .nan
            }
        } }
        return (try MemoryCube(shape: try CubeShape([rows, columns, q, q]), values: values), truth)
    }

    @Test func testMetalAgreesWithTheCPUReference() async throws {
        guard VirtualDiffraction.metalIsAvailable else {
            Issue.record("No Metal device on this machine: the Metal-versus-CPU pin was not run")
            return
        }
        let (cube, _) = try Self.cube(nan: true)
        for estimator in [OriginEstimator.iteratedWindow(), .iteratedWindow(passes: 2, minimumMargin: 0.5, binnedCoarse: false)] {
            let cpu = try await OriginMeasurement.measure(of: cube, radius: 6.2, estimator: estimator, path: .cpu, tiling: .rows(2))
            let gpu = try await OriginMeasurement.measure(of: cube, radius: 6.2, estimator: estimator, path: .metal, tiling: .rows(3))
            #expect(cpu.row.path == .cpu && gpu.row.path == .metal)
            for (a, b) in [(cpu.row.values, gpu.row.values), (cpu.column.values, gpu.column.values)] {
                for (x, y) in zip(a, b) { #expect(abs(x - y) <= 1e-5 * Swift.max(abs(x), 1), "\(x) vs \(y)") }
            }
        }
    }

    @Test func testPlantedOriginIsRecovered() async throws {
        let (cube, truth) = try Self.cube(nan: true)
        let map = try await OriginMeasurement.measure(of: cube, radius: 6.2, estimator: .iteratedWindow())
        for k in truth.indices {
            #expect(abs(map.row.values[k] - truth[k].0) < 0.1 && abs(map.column.values[k] - truth[k].1) < 0.1,
                    "\(k): \(map.row.values[k]), \(map.column.values[k]) vs \(truth[k])")
        }
        // the default estimator is unchanged: the py4DSTEM overload through the new entry point gives the same maps
        let (clean, _) = try Self.cube()
        let old = try await OriginMeasurement.measure(of: clean, radius: 6.2).origins
        let viaNew = try await OriginMeasurement.measure(of: clean, radius: 6.2, estimator: .py4DSTEM)
        #expect(old == viaNew)
    }

    @Test func testWindowSensitivityKnownAnswers() async throws {
        // a compact probe does not care which window it gets; a ringed one walks (Float32, so a small floor)
        let (compact, _) = try Self.cube()
        let small = try #require(try await OriginMeasurement.windowSensitivity(of: compact, radius: 6.2))
        #expect(small < 0.1, "\(small)")
        let (ringed, _) = try Self.cube(ring: true)
        let large = try #require(try await OriginMeasurement.windowSensitivity(of: ringed, radius: 6.2))
        #expect(large > 0.2 && large > 20 * small, "\(large)")
        // by hand: the same shipped and wide scale give distance exactly 0
        let zero = try await OriginMeasurement.windowSensitivity(of: ringed, radius: 6.2, shippedScale: 1.2, wideScale: 1.2)
        #expect(zero == 0)
        // the strided sample: 20 positions with a limit of 7 take every 3rd (7 positions); the result is the same on Metal
        if VirtualDiffraction.metalIsAvailable {
            let cpu = try await OriginMeasurement.windowSensitivity(of: ringed, radius: 6.2, sampleLimit: 7)
            let gpu = try await OriginMeasurement.windowSensitivity(of: ringed, radius: 6.2, sampleLimit: 7, path: .metal)
            #expect(abs(try #require(cpu) - (try #require(gpu))) <= 1e-4 * Swift.max(try #require(cpu), 1))
        }
        #expect(try await OriginMeasurement.windowSensitivity(of: ringed, radius: -1) == nil)
    }
}
