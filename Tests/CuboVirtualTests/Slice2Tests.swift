//
//  Slice2Tests.swift
//  Centre of mass and probe size against py4DSTEM f050d207 and planted truth (registration slice 2, 3.1, 3.5, 4, 5),
//  read from slice-2-pins.json (tools/slice-2/pins.py, run before any Swift).
//

import Foundation
import Testing
import CuboCore
import CuboVirtual

enum Slice2 {
    nonisolated(unsafe) static let pins: [String: Any] = {
        let data = try! Data(contentsOf: Fixtures.url("slice-2-pins.json"))
        return try! JSONSerialization.jsonObject(with: data) as! [String: Any]
    }()
    static func doubles(_ any: Any?) -> [Double] { (any as? [String] ?? []).map { Double($0)! } }
    static func double(_ any: Any?) -> Double { Double(any as! String)! }
    static var com: [[String: Any]] { pins["com"] as! [[String: Any]] }
    static var probe: [[String: Any]] { pins["probe"] as! [[String: Any]] }
    static func cubeURL(_ name: String) -> URL { Fixtures.url("\(name).npy") }
    static func mask(_ entry: [String: Any]) -> DetectorGeometry? {
        (entry["weights"] as? [String]).map { .mask($0.map { Float(Double($0)!) }) }
    }
    static let integerCubes: Set<String> = ["counts16", "counts32"]
}

@Suite struct Slice2PinsTests {
    @Test func testSlice2PinsCarryTheirVersions() {
        let v = Slice2.pins["versions"] as! [String: String]
        #expect(v["py4dstem_commit"] == "f050d207851f8accd3aa550365baaf1f38230428")
        #expect(v["numpy"] == "2.5.0")
        #expect(Slice2.com.count == 24 && Slice2.probe.count == 20)
    }

    /// 3.1, 4: the sums within 1e-10 of the float64 definition (integer cubes bit for bit), the centre of mass within
    /// 1e-9 px where the denominator is conditioned; py4DSTEM's own float32 distance printed.
    @Test func testCenterOfMassAgreesWithTheFloat64Definition() async throws {
        for e in Slice2.com {
            let name = e["cube"] as! String, mask = e["mask"] as! String
            let cube = try MappedCube(npy: Slice2.cubeURL(name))
            let com = try await CenterOfMass.measure(of: cube, mask: Slice2.mask(e))
            let s = Slice2.doubles(e["intensity"]), r = Slice2.doubles(e["row"]), c = Slice2.doubles(e["column"])
            if Slice2.integerCubes.contains(name) {
                assertIdentical(com.intensity.values, s, "\(name)/\(mask) intensity")
                assertIdentical(com.momentRow.values, Slice2.doubles(e["moment_row"]), "\(name)/\(mask) moment row")
                assertIdentical(com.row.values, r, "\(name)/\(mask) row")
                assertIdentical(com.column.values, c, "\(name)/\(mask) column")
            } else {
                assertRelative(com.intensity.values, s, 1e-10, "\(name)/\(mask) intensity")
                assertRelative(com.momentRow.values, Slice2.doubles(e["moment_row"]), 1e-10, "\(name)/\(mask) moment row")
                assertRelative(com.momentColumn.values, Slice2.doubles(e["moment_column"]), 1e-10, "\(name)/\(mask) moment col")
            }
            // conditioned (registration 4): |S| ≥ 1e-6 · Σ|w P|, from the cube itself
            let eager = try await MemoryCube(loading: cube)
            let n = cube.shape.detectorCount
            let w = try DetectorMask.make(Slice2.mask(e) ?? .mask([Float](repeating: 1, count: n)), shape: cube.shape)
            var worst = 0.0, ill = 0
            for k in r.indices {
                var absSum = 0.0
                for i in 0..<n { absSum += abs(Double(w[i]) * Double(eager.values[k * n + i])) }
                guard abs(s[k]) >= 1e-6 * absSum else { ill += 1; continue }
                if r[k].isNaN { #expect(com.row.values[k].isNaN); continue }
                worst = max(worst, abs(com.row.values[k] - r[k]), abs(com.column.values[k] - c[k]))
            }
            #expect(worst <= 1e-9, "\(name)/\(mask): worst CoM distance \(worst) px")
            print("slice2 com \(name)/\(mask): kit vs float64 \(worst) px (ill-conditioned \(ill)); py4DSTEM own vs float64 \(e["com_distance_py4dstem_own_vs_f64_px"] ?? "n/a") px")
        }
    }

    @Test(.enabled(if: VirtualDiffraction.metalIsAvailable, "no Metal device on this machine"))
    func testCenterOfMassMetalAgreesWithTheCPU() async throws {
        for e in Slice2.com {
            let name = e["cube"] as! String
            let cube = try MappedCube(npy: Slice2.cubeURL(name))
            for tiling in [ScanTiling.automatic, .rows(1)] {
                let cpu = try await CenterOfMass.measure(of: cube, mask: Slice2.mask(e), path: .cpu, tiling: tiling)
                let gpu = try await CenterOfMass.measure(of: cube, mask: Slice2.mask(e), path: .metal, tiling: tiling)
                #expect(gpu.row.path == .metal)
                assertRelative(gpu.intensity.values, cpu.intensity.values, 1e-10, "\(name) metal intensity")
                assertRelative(gpu.momentRow.values, cpu.momentRow.values, 1e-10, "\(name) metal moment row")
                assertRelative(gpu.momentColumn.values, cpu.momentColumn.values, 1e-10, "\(name) metal moment column")
            }
        }
    }

    /// 3.5: py4DSTEM's get_probe_size is float64 on these patterns, so the port is held to it bit for bit; where it
    /// returns NaN the kit refuses.
    @Test func testProbeSizeIsPy4DSTEMsBitForBit() async throws {
        for e in Slice2.probe {
            let name = e["cube"] as! String, kind = e["pattern"] as! String
            let pattern: DetectorImage
            if kind == "given" {
                let shape = e["shape"] as! [Int]
                pattern = DetectorImage(detectorRows: shape[0], detectorColumns: shape[1],
                                        values: Slice2.doubles(e["pattern_values"]), path: .cpu)
            } else {
                let cube = try MappedCube(npy: Slice2.cubeURL(name))
                let (mean, max) = try await VirtualDiffraction.meanAndMax(of: cube)
                pattern = kind == "mean" ? mean : max
            }
            let lower = e["lower"] as? Double ?? 0.01, upper = e["upper"] as? Double ?? 0.99, count = e["count"] as? Int ?? 100
            if e["refused"] as! Bool {
                #expect(throws: CuboVirtualError.self) { try ProbeSize.estimate(pattern, lowerThreshold: lower, upperThreshold: upper, count: count) }
                continue
            }
            let p = try ProbeSize.estimate(pattern, lowerThreshold: lower, upperThreshold: upper, count: count)
            #expect(p.radius.bitPattern == Slice2.double(e["radius"]).bitPattern, "\(name)/\(kind) radius \(p.radius) vs \(Slice2.double(e["radius"]))")
            #expect(p.centerRow.bitPattern == Slice2.double(e["center_row"]).bitPattern, "\(name)/\(kind) row \(p.centerRow)")
            #expect(p.centerColumn.bitPattern == Slice2.double(e["center_column"]).bitPattern, "\(name)/\(kind) column")
            #expect(p.threshold.bitPattern == Slice2.double(e["threshold"]).bitPattern, "\(name)/\(kind) threshold \(p.threshold)")
        }
    }
}

@Suite struct Slice2TruthTests {
    /// 5.1: two pixels a at (i0, j0) and b at (i0+1, j0+1): row = i0 + b/(a+b), column = j0 + b/(a+b).
    @Test func testPlantedShiftIsRecovered() async throws {
        let cube = try MappedCube(npy: Slice2.cubeURL("shift"))
        let truth = Slice2.pins["shift_truth"] as! [String: Any]
        for path in VirtualDiffraction.metalIsAvailable ? [ComputePath.cpu, .metal] : [.cpu] {
            let com = try await CenterOfMass.measure(of: cube, path: path)
            assertRelative(com.row.values, Slice2.doubles(truth["row"]), 1e-12, "shift row \(path)")
            assertRelative(com.column.values, Slice2.doubles(truth["column"]), 1e-12, "shift column \(path)")
        }
    }

    /// 5.1: a uniform disk: radius sqrt(n/π), centre exactly the disk's, on a pixel and a half-pixel.
    @Test func testUniformDiskProbeTruth() throws {
        for e in Slice2.probe where (e["cube"] as! String).hasPrefix("disk_") {
            let shape = e["shape"] as! [Int]
            let p = try ProbeSize.estimate(DetectorImage(detectorRows: shape[0], detectorColumns: shape[1],
                                                         values: Slice2.doubles(e["pattern_values"]), path: .cpu))
            #expect(abs(p.radius - Slice2.double(e["truth_radius"])) <= 1e-12)
            #expect(abs(p.centerRow - Slice2.double(e["truth_row"])) <= 1e-12)
            #expect(abs(p.centerColumn - Slice2.double(e["truth_column"])) <= 1e-12)
        }
    }

    /// A zero-intensity pattern has no centre of mass: NaN, not a substituted value.
    @Test func testZeroPatternIsNaN() async throws {
        let shape = try CubeShape([1, 2, 3, 3])
        let cube = try MemoryCube(shape: shape, values: [0, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 5, 0, 0, 0, 0])
        let com = try await CenterOfMass.measure(of: cube)
        #expect(com.row.values[0].isNaN && com.column.values[0].isNaN)
        #expect(com.row.values[1] == 1 && com.column.values[1] == 1)
    }

    @Test func testProbeRefusals() {
        #expect(throws: CuboVirtualError.self) {
            try ProbeSize.estimate(DetectorImage(detectorRows: 2, detectorColumns: 2, values: [0, 0, 0, 0], path: .cpu))
        }
        #expect(throws: CuboVirtualError.self) {
            try ProbeSize.estimate(DetectorImage(detectorRows: 1, detectorColumns: 2, values: [1, 2], path: .cpu), count: 1)
        }
    }
}

/// The Gate B refuter's slice-2 cases on 84815c9 (registration slice 2, section 9), kept so they gate.
@Suite(.enabled(if: VirtualDiffraction.metalIsAvailable, "no Metal device on this machine"))
struct Slice2RefuterRegressionTests {
    /// F1: on a 1024² detector the row moment of an integer cube passes 3e10; the first Metal kernel's Float32 correction
    /// lost it (2e-4 px). Integer cubes are exact on Metal now (renormalised double-float sums, exact to 2^48).
    @Test func testMetalCenterOfMassIsExactOnLargeIntegerCubes() async throws {
        let shape = try CubeShape([2, 2, 1024, 1024])
        var state: UInt64 = 7
        let values: [Float] = (0..<shape.elementCount).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float((state >> 33) % 4001)
        }
        let cube = try MemoryCube(shape: shape, values: values)
        let cpu = try await CenterOfMass.measure(of: cube, path: .cpu)
        let gpu = try await CenterOfMass.measure(of: cube, path: .metal)
        assertIdentical(gpu.intensity.values, cpu.intensity.values, "1024² intensity")
        assertIdentical(gpu.momentRow.values, cpu.momentRow.values, "1024² row moment")
        assertIdentical(gpu.momentColumn.values, cpu.momentColumn.values, "1024² column moment")
        #expect(cpu.momentRow.values.max()! > 3e10)
    }

    /// F1: dark-subtracted data (a blob minus a wider blob plus noise): Metal within 1e-9 px of the CPU where
    /// |S| ≥ 1e-3 · Σ|w P| (the Metal condition of the amendment).
    @Test func testMetalCenterOfMassOnDarkSubtractedData() async throws {
        let q = 128, shape = try CubeShape([3, 3, q, q])
        var g = Normals(state: 31)
        var values = [Float]()
        for s in 0..<shape.scanCount {
            for i in 0..<q { for j in 0..<q {
                let r2 = Double((i - 60 - s) * (i - 60 - s) + (j - 64) * (j - 64))
                values.append(Float(5000 * exp(-r2 / 50) - 4000 * exp(-r2 / 400)) + g.next(30))
            } }
        }
        let cube = try MemoryCube(shape: shape, values: values)
        let cpu = try await CenterOfMass.measure(of: cube, path: .cpu)
        let gpu = try await CenterOfMass.measure(of: cube, path: .metal)
        var worst = 0.0, claimed = 0
        for k in 0..<shape.scanCount {
            let absSum = values[(k * q * q)..<((k + 1) * q * q)].reduce(0.0) { $0 + abs(Double($1)) }
            guard abs(cpu.intensity.values[k]) >= 1e-3 * absSum else { continue }
            claimed += 1
            worst = max(worst, abs(gpu.row.values[k] - cpu.row.values[k]), abs(gpu.column.values[k] - cpu.column.values[k]))
        }
        #expect(claimed > 0 && worst <= 1e-9, "Metal vs CPU \(worst) px over \(claimed) conditioned positions")
        assertRelative(gpu.intensity.values, cpu.intensity.values, 1e-10, "dark-subtracted intensity")
    }
}
