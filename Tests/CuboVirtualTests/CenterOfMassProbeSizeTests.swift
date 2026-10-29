//
//  CenterOfMassProbeSizeTests.swift
//  Centre of mass and probe size against py4DSTEM f050d207 and planted truth,
//  read from com-dpc-pins.json (tools/com-dpc/pins.py, run before any Swift).
//

import Foundation
import Testing
import CuboKitTestSupport
import CuboCore
import CuboVirtual

enum CenterOfMassPins {
    nonisolated(unsafe) static let pins: [String: Any] = {
        let data = try! Data(contentsOf: Fixtures.url("com-dpc-pins.json"))
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

@Suite struct CenterOfMassPinsTests {
    @Test func testPinsCarryTheirVersions() {
        let v = CenterOfMassPins.pins["versions"] as! [String: String]
        #expect(v["py4dstem_commit"] == "f050d207851f8accd3aa550365baaf1f38230428")
        #expect(v["numpy"] == "2.5.0")
        #expect(CenterOfMassPins.com.count == 24 && CenterOfMassPins.probe.count == 20)
    }

    /// The sums within 1e-10 of the float64 definition (integer cubes exact today), the centre of mass within
    /// 1e-9 px where the denominator is conditioned; py4DSTEM's own float32 distance printed.
    @Test func testCenterOfMassAgreesWithTheFloat64Definition() async throws {
        for e in CenterOfMassPins.com {
            let name = e["cube"] as! String, mask = e["mask"] as! String
            let cube = try MappedCube(npy: CenterOfMassPins.cubeURL(name))
            let com = try await CenterOfMass.measure(of: cube, mask: CenterOfMassPins.mask(e))
            let s = CenterOfMassPins.doubles(e["intensity"]), r = CenterOfMassPins.doubles(e["row"]), c = CenterOfMassPins.doubles(e["column"])
            if CenterOfMassPins.integerCubes.contains(name) {
                assertIdentical(com.intensity.values, s, "\(name)/\(mask) intensity")
                assertIdentical(com.momentRow.values, CenterOfMassPins.doubles(e["moment_row"]), "\(name)/\(mask) moment row")
                assertIdentical(com.row.values, r, "\(name)/\(mask) row")
                assertIdentical(com.column.values, c, "\(name)/\(mask) column")
            } else {
                assertRelative(com.intensity.values, s, 1e-10, "\(name)/\(mask) intensity")
                assertRelative(com.momentRow.values, CenterOfMassPins.doubles(e["moment_row"]), 1e-10, "\(name)/\(mask) moment row")
                assertRelative(com.momentColumn.values, CenterOfMassPins.doubles(e["moment_column"]), 1e-10, "\(name)/\(mask) moment col")
            }
            // conditioned: |S| ≥ 1e-6 · Σ|w P|, from the cube itself
            let eager = try await MemoryCube(loading: cube)
            let n = cube.shape.detectorCount
            let w = try DetectorMask.make(CenterOfMassPins.mask(e) ?? .mask([Float](repeating: 1, count: n)), shape: cube.shape)
            var worst = 0.0, ill = 0
            for k in r.indices {
                var absSum = 0.0
                for i in 0..<n { absSum += abs(Double(w[i]) * Double(eager.values[k * n + i])) }
                guard abs(s[k]) >= 1e-6 * absSum else { ill += 1; continue }
                if r[k].isNaN { #expect(com.row.values[k].isNaN); continue }
                worst = max(worst, abs(com.row.values[k] - r[k]), abs(com.column.values[k] - c[k]))
            }
            #expect(worst <= 1e-9, "\(name)/\(mask): worst CoM distance \(worst) px")
            print("com-dpc com \(name)/\(mask): kit vs float64 \(worst) px (ill-conditioned \(ill)); py4DSTEM own vs float64 \(e["com_distance_py4dstem_own_vs_f64_px"] ?? "n/a") px")
        }
    }

    @Test(.enabled(if: VirtualDiffraction.metalIsAvailable, "no Metal device on this machine"))
    func testCenterOfMassMetalAgreesWithTheCPU() async throws {
        for e in CenterOfMassPins.com {
            let name = e["cube"] as! String
            let cube = try MappedCube(npy: CenterOfMassPins.cubeURL(name))
            for tiling in [ScanTiling.automatic, .rows(1)] {
                let cpu = try await CenterOfMass.measure(of: cube, mask: CenterOfMassPins.mask(e), path: .cpu, tiling: tiling)
                let gpu = try await CenterOfMass.measure(of: cube, mask: CenterOfMassPins.mask(e), path: .metal, tiling: tiling)
                #expect(gpu.row.path == .metal)
                assertRelative(gpu.intensity.values, cpu.intensity.values, 1e-10, "\(name) metal intensity")
                assertRelative(gpu.momentRow.values, cpu.momentRow.values, 1e-10, "\(name) metal moment row")
                assertRelative(gpu.momentColumn.values, cpu.momentColumn.values, 1e-10, "\(name) metal moment column")
            }
        }
    }

    /// Py4DSTEM's get_probe_size is float64 on these patterns, so the test asserts it exactly today (a tolerance would be allowed); where it
    /// returns NaN the kit refuses.
    @Test func testProbeSizeIsPy4DSTEMsBitForBit() async throws {
        for e in CenterOfMassPins.probe {
            let name = e["cube"] as! String, kind = e["pattern"] as! String
            let pattern: DetectorImage
            if kind == "given" {
                let shape = e["shape"] as! [Int]
                pattern = DetectorImage(detectorRows: shape[0], detectorColumns: shape[1],
                                        values: CenterOfMassPins.doubles(e["pattern_values"]), path: .cpu)
            } else {
                let cube = try MappedCube(npy: CenterOfMassPins.cubeURL(name))
                let (mean, max) = try await VirtualDiffraction.meanAndMax(of: cube)
                pattern = kind == "mean" ? mean : max
            }
            let lower = e["lower"] as? Double ?? 0.01, upper = e["upper"] as? Double ?? 0.99, count = e["count"] as? Int ?? 100
            if e["refused"] as! Bool {
                #expect(throws: CuboVirtualError.self) { try ProbeSize.estimate(pattern, lowerThreshold: lower, upperThreshold: upper, count: count) }
                continue
            }
            let p = try ProbeSize.estimate(pattern, lowerThreshold: lower, upperThreshold: upper, count: count)
            #expect(p.radius.bitPattern == CenterOfMassPins.double(e["radius"]).bitPattern, "\(name)/\(kind) radius \(p.radius) vs \(CenterOfMassPins.double(e["radius"]))")
            #expect(p.centerRow.bitPattern == CenterOfMassPins.double(e["center_row"]).bitPattern, "\(name)/\(kind) row \(p.centerRow)")
            #expect(p.centerColumn.bitPattern == CenterOfMassPins.double(e["center_column"]).bitPattern, "\(name)/\(kind) column")
            #expect(p.threshold.bitPattern == CenterOfMassPins.double(e["threshold"]).bitPattern, "\(name)/\(kind) threshold \(p.threshold)")
        }
    }
}

@Suite struct CenterOfMassTruthTests {
    /// Two pixels a at (i0, j0) and b at (i0+1, j0+1): row = i0 + b/(a+b), column = j0 + b/(a+b).
    @Test func testPlantedShiftIsRecovered() async throws {
        let cube = try MappedCube(npy: CenterOfMassPins.cubeURL("shift"))
        let truth = CenterOfMassPins.pins["shift_truth"] as! [String: Any]
        for path in VirtualDiffraction.metalIsAvailable ? [ComputePath.cpu, .metal] : [.cpu] {
            let com = try await CenterOfMass.measure(of: cube, path: path)
            assertRelative(com.row.values, CenterOfMassPins.doubles(truth["row"]), 1e-12, "shift row \(path)")
            assertRelative(com.column.values, CenterOfMassPins.doubles(truth["column"]), 1e-12, "shift column \(path)")
        }
    }

    /// A uniform disk: radius sqrt(n/π), centre exactly the disk's, on a pixel and a half-pixel.
    @Test func testUniformDiskProbeTruth() throws {
        for e in CenterOfMassPins.probe where (e["cube"] as! String).hasPrefix("disk_") {
            let shape = e["shape"] as! [Int]
            let p = try ProbeSize.estimate(DetectorImage(detectorRows: shape[0], detectorColumns: shape[1],
                                                         values: CenterOfMassPins.doubles(e["pattern_values"]), path: .cpu))
            #expect(abs(p.radius - CenterOfMassPins.double(e["truth_radius"])) <= 1e-12)
            #expect(abs(p.centerRow - CenterOfMassPins.double(e["truth_row"])) <= 1e-12)
            #expect(abs(p.centerColumn - CenterOfMassPins.double(e["truth_column"])) <= 1e-12)
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

/// Slice-2 cases an adversarial review found on 84815c9, kept so they gate.
@Suite(.enabled(if: VirtualDiffraction.metalIsAvailable, "no Metal device on this machine"))
struct CenterOfMassMetalRegressionTests {
    /// On a 1024² detector the row moment of an integer cube passes 3e10; the first Metal kernel's Float32 correction
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

    /// Dark-subtracted data (a blob minus a wider blob plus noise): Metal within 1e-9 px of the CPU where
    /// |S| ≥ 1e-3 · Σ|w P| (the Metal path's conditioning bar).
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
