//
//  LowDoseOriginTests.swift
//  The iterated-window origin at very low dose: deterministic Poisson cubes with a planted origin plane, a flat probe disk
//  of radius 3 px at 1 and 3 counts per pixel on a background of 0.2 and 0.5. The brightest-disk seed recovers the plane;
//  the block seed (the default, kept as it was) is shown drifting on the same cubes.
//

import Foundation
import Testing
import CuboCore
import CuboVirtual

@Suite struct LowDoseOriginTests {
    /// Knuth's Poisson sampler on a 64-bit LCG: the same cube on every machine.
    struct Counts {
        var state: UInt64
        mutating func uniform() -> Double {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return (Double(state >> 11) + 0.5) / Double(UInt64(1) << 53)
        }
        mutating func poisson(_ mean: Double) -> Float {
            let limit = exp(-mean)
            var k = 0, p = 1.0
            repeat { k += 1; p *= uniform() } while p > limit
            return Float(k - 1)
        }
    }

    /// 12 × 12 positions of 64 × 64 pixels; origin (row, column) = a plane over the scan; a disk of radius 3 with a
    /// 0.5-px linear edge at `peak` counts per pixel on `background`.
    static func cube(peak: Double, background: Double, seed: UInt64) throws -> (MemoryCube, [(Double, Double)]) {
        let s = 12, q = 64
        var counts = Counts(state: seed)
        var values = [Float](), truth = [(Double, Double)]()
        values.reserveCapacity(s * s * q * q)
        for sr in 0..<s { for sc in 0..<s {
            let cr = 30.4 + 0.15 * Double(sr) - 0.10 * Double(sc), cc = 33.7 - 0.12 * Double(sr) + 0.20 * Double(sc)
            truth.append((cr, cc))
            for i in 0..<q { for j in 0..<q {
                let d = ((Double(i) - cr) * (Double(i) - cr) + (Double(j) - cc) * (Double(j) - cc)).squareRoot()
                let disk = min(max(3.25 - d, 0), 0.5) / 0.5
                values.append(counts.poisson(background + peak * disk))
            } }
        } }
        return (try MemoryCube(shape: try CubeShape([s, s, q, q]), values: values), truth)
    }

    /// The largest distance between the truth and the least-squares plane through the measured origins (row and column
    /// each fitted as a + b·scanRow + c·scanColumn), and the median per-position distance.
    static func errors(_ map: OriginMap, _ truth: [(Double, Double)], scan s: Int = 12) -> (plane: Double, median: Double) {
        func fit(_ v: [Double]) -> [Double] {
            var ata = [[Double]](repeating: [0, 0, 0], count: 3), atb = [0.0, 0, 0]
            for k in v.indices {
                let x = [1, Double(k / s), Double(k % s)]
                for i in 0..<3 { atb[i] += x[i] * v[k]; for j in 0..<3 { ata[i][j] += x[i] * x[j] } }
            }
            // Cramer's rule on the 3 × 3 normal equations
            func det(_ m: [[Double]]) -> Double {
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            }
            let d = det(ata)
            let coef = (0..<3).map { c -> Double in
                var m = ata
                for i in 0..<3 { m[i][c] = atb[i] }
                return det(m) / d
            }
            return v.indices.map { k in coef[0] + coef[1] * Double(k / s) + coef[2] * Double(k % s) }
        }
        let pr = fit(map.row.values), pc = fit(map.column.values)
        var plane = 0.0, each = [Double]()
        for k in truth.indices {
            plane = max(plane, ((pr[k] - truth[k].0) * (pr[k] - truth[k].0) + (pc[k] - truth[k].1) * (pc[k] - truth[k].1)).squareRoot())
            let dr = map.row.values[k] - truth[k].0, dc = map.column.values[k] - truth[k].1
            each.append((dr * dr + dc * dc).squareRoot())
        }
        each.sort()
        return (plane, each[each.count / 2])
    }

    /// Measured (plane error in px, disk seed / block seed): peak 3 on 0.5: 0.10 / 0.11; peak 1 on 0.2: 0.19 / 1.55;
    /// peak 2 on 1.0: 0.23 / 1.99; peak 1 on 0.5: 1.11 / 5.67 (28 disk counts on 14 of background: a few positions still
    /// seed on noise, and the least-squares plane follows them).
    @Test(arguments: [(3.0, 0.5, UInt64(11), 0.3), (1.0, 0.2, UInt64(12), 0.3), (2.0, 1.0, UInt64(14), 0.3),
                      (1.0, 0.5, UInt64(13), 1.5)])
    func brightestDiskSeedHoldsThePlantedPlaneAtLowDose(peak: Double, background: Double, seed: UInt64,
                                                           bound: Double) async throws {
        let (cube, truth) = try Self.cube(peak: peak, background: background, seed: seed)
        let disk = try await OriginMeasurement.measure(of: cube, radius: 3, estimator: .iteratedWindow(seed: .brightestDisk))
        let block = try await OriginMeasurement.measure(of: cube, radius: 3, estimator: .iteratedWindow())
        let e = Self.errors(disk, truth), b = Self.errors(block, truth)
        #expect(e.plane < bound, "plane error \(e.plane) px (block seed \(b.plane))")
        #expect(e.median < 0.6, "median error \(e.median) px")
        #expect(e.plane <= b.plane + 0.01, "the disk seed is no worse than the block seed: \(e.plane) vs \(b.plane)")
        if peak < 3 { #expect(b.plane > 1, "the block seed drifts at this dose (\(b.plane) px)") }
    }

    @Test func brightestDiskSeedIsRefusedOnMetal() async throws {
        let (cube, _) = try Self.cube(peak: 3, background: 0.5, seed: 1)
        await #expect(throws: CuboVirtualError.self) {
            _ = try await OriginMeasurement.measure(of: cube, radius: 3, estimator: .iteratedWindow(seed: .brightestDisk),
                                                    path: .metal)
        }
    }

    @Test func aRadiusFarLargerThanTheDetectorStaysCheap() async throws {
        let (cube, _) = try Self.cube(peak: 3, background: 0.5, seed: 3)
        let map = try await OriginMeasurement.measure(of: cube, radius: 1e5, estimator: .iteratedWindow(seed: .brightestDisk))
        #expect(map.row.values.allSatisfy(\.isFinite) && map.column.values.allSatisfy(\.isFinite))
    }

    @Test func brightestDiskOnACleanPatternIsTheDiskCentre() async throws {
        let (cube, truth) = try Self.cube(peak: 200, background: 0, seed: 2)
        let map = try await OriginMeasurement.measure(of: cube, radius: 3, estimator: .iteratedWindow(seed: .brightestDisk))
        #expect(Self.errors(map, truth).median < 0.15)
    }
}
