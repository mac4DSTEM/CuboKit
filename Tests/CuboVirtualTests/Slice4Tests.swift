//
//  Slice4Tests.swift
//  The median pattern, virtual diffraction over a selection, origins with a scan mask, the Friedel origin, the beamstop
//  mask, the vacuum probe and its kernels (registration slice 4) against py4DSTEM f050d207 and planted truth:
//  slice-4-pins.json and, on this machine, the real-data exports (<name>-s4-pins.json).
//

import Foundation
import Testing
import CuboCore
import CuboVirtual

enum S4 {
    nonisolated(unsafe) static let pins: [String: Any] = {
        try! JSONSerialization.jsonObject(with: Data(contentsOf: Fixtures.url("slice-4-pins.json"))) as! [String: Any]
    }()
    static func d(_ a: Any?) -> [Double] { (a as? [String] ?? []).map { Double($0)! } }
    static func b(_ a: Any?) -> [Bool] { (a as? [Int] ?? []).map { $0 != 0 } }
    static func n(_ a: Any?) -> Double { (a as? NSNumber)?.doubleValue ?? Double(a as! String)! }
    static func realIsPresent(_ name: String) -> Bool {
        FileManager.default.fileExists(atPath: Fixtures.realData.appendingPathComponent("\(name)-s4-pins.json").path)
    }
    static func image(_ v: [Double], _ rows: Int, _ columns: Int) -> DetectorImage {
        DetectorImage(detectorRows: rows, detectorColumns: columns, values: v, path: .cpu)
    }
    static func map(_ r: [Double], _ c: [Double], _ shape: CubeShape) throws -> OriginMap {
        try OriginMap(row: ScanImage(scanRows: shape.scanRows, scanColumns: shape.scanColumns, values: r, path: .cpu),
                      column: ScanImage(scanRows: shape.scanRows, scanColumns: shape.scanColumns, values: c, path: .cpu))
    }
    /// max |got − want| / max |want| (the normwise claim).
    static func normwise(_ got: [Double], _ want: [Double]) -> Double {
        let top = want.map(abs).max() ?? 0
        return zip(got, want).map { abs($0 - $1) }.max()! / Swift.max(top, .leastNonzeroMagnitude)
    }
    /// The largest distance between two origin maps where `want` is finite; NaN must meet NaN.
    static func distance(_ got: [Double], _ want: [Double], skip: Set<Int> = []) -> Double {
        var worst = 0.0
        for k in want.indices where !skip.contains(k) {
            if want[k].isNaN || got[k].isNaN { if want[k].isNaN != got[k].isNaN { return .infinity }; continue }
            worst = Swift.max(worst, abs(got[k] - want[k]))
        }
        return worst
    }

    /// Every claim of section 4 on one pinned cube; prints what it measured.
    static func check(_ e: [String: Any], cube: MappedCube) async throws {
        let name = e["cube"] as! String, integer = e["integer"] as! Bool
        let shape = cube.shape, qy = shape.detectorRows, qx = shape.detectorColumns
        var log = ["slice4 \(name)"]
        // 3.1 selections
        let left = b(e["scan_mask"]), weights = d(e["weights"])
        let sel = e["selections"] as! [String: [String: Any]]
        for (key, scan) in [("all", ScanSelection.all), ("mask", .mask(left)), ("weights", .weights(weights))] {
            for stat in DiffractionStatistic.allCases {
                guard let ent = sel["\(key)_\(stat.rawValue)"] else { continue }
                let got = try await VirtualDiffraction.pattern(of: cube, statistic: stat, scan: scan).values
                let want = d(ent["value"]), own = d(ent["value_py4dstem"])
                if stat == .mean {
                    assertRelative(got, want, 1e-10, "\(name) \(key) mean")
                    if !own.isEmpty, key == "weights" { assertRelative(got, own, 1e-10, "\(name) \(key) mean vs py4DSTEM") }
                    if integer, key == "mask" {    // exact sums of integers, one division: bit for bit
                        assertIdentical(got, want, "\(name) \(key) mean (integer)")
                        assertIdentical(got, own, "\(name) \(key) mean vs py4DSTEM (integer)")
                    }
                } else {
                    assertIdentical(got, want, "\(name) \(key) \(stat)")
                    if !own.isEmpty, stat == .max || key == "weights" || integer {
                        assertIdentical(got, own, "\(name) \(key) \(stat) vs py4DSTEM")
                    }
                }
                if let dist = ent["distance_py4dstem"] { log.append("\(key) \(stat): py4DSTEM own vs float64 \(dist)") }
            }
        }
        // shifted with a selection
        let map = try S4.map(d(e["origin_row"]), d(e["origin_column"]), shape)
        let sh = e["shifted"] as! [String: [String: Any]]
        for (mode, shift) in [("rounded", PatternShift.rounded(map)), ("fourier", .fourier(map))] {
            for (key, scan) in [("mask", ScanSelection.mask(left)), ("weights", .weights(weights))] {
                for stat in [DiffractionStatistic.mean, .max] {
                    let ent = sh["\(mode)_\(key)_\(stat.rawValue)"]!
                    let got = try await VirtualDiffraction.pattern(of: cube, statistic: stat, scan: scan, shift: shift).values
                    let want = d(ent["value"]), own = d(ent["value_py4dstem"])
                    if mode == "fourier" {
                        #expect(normwise(got, want) <= 1e-12, "\(name) \(mode) \(key) \(stat): \(normwise(got, want))")
                    } else if stat == .mean {
                        assertRelative(got, want, 1e-10, "\(name) \(mode) \(key) mean")
                        assertRelative(got, own, 1e-10, "\(name) \(mode) \(key) mean vs py4DSTEM")
                    } else {
                        assertIdentical(got, want, "\(name) \(mode) \(key) max")
                        let k = got.indices.first { got[$0] >= 0 && got[$0].bitPattern != own[$0].bitPattern }
                        #expect(k == nil, "\(name): \(mode) \(key) max vs py4DSTEM where ≥ 0, at \(k ?? -1)")
                    }
                }
            }
        }
        // 3.2 origins with the scan mask, the fast centre, remove_NaN
        let om = e["origins_masked"] as! [String: Any]
        let (masked, radius) = try await OriginMeasurement.measure(of: cube, scanMask: left)
        #expect(radius.bitPattern == Double(e["radius"] as! String)!.bitPattern, "\(name): radius")
        let dm = Swift.max(distance(masked.row.values, d(om["row"])), distance(masked.column.values, d(om["column"])))
        #expect(dm <= 1e-9, "\(name): masked origins \(dm) px")
        let pinnedMap = try S4.map(d(om["row"]), d(om["column"]), shape).replacingNaNWithMean()
        assertIdentical(pinnedMap.row.values, d(om["filled_row"]), "\(name) remove_NaN row")
        assertIdentical(pinnedMap.column.values, d(om["filled_column"]), "\(name) remove_NaN column")
        let fc = e["fast_center"] as! [String: Any]
        let (fast, _) = try await OriginMeasurement.measure(of: cube, scanMask: nil, fastCenter: true)
        assertIdentical(fast.row.values, d(fc["row"]), "\(name) fast centre row")
        assertIdentical(fast.column.values, d(fc["column"]), "\(name) fast centre column")
        log.append("origins: masked \(dm) px (py4DSTEM own \(om["distance_py4dstem_px"]!) px), fast centre: py4DSTEM's float32 differs at \(fc["differ_py4dstem"]!)")
        // 3.4 beamstop on the float64 mean
        let mean = image(d(e["mean_f64"]), qy, qx)
        for (key, ent) in e["beamstop"] as! [String: [String: Any]] {
            let kw = ent["kw"] as! [String: Any]
            let call = {
                try BeamstopMask.make(from: mean, threshold: (kw["threshold"] as? NSNumber)?.doubleValue ?? 0.25,
                                      distanceEdge: (kw["distance_edge"] as? NSNumber)?.doubleValue ?? 2,
                                      includeEdges: (kw["include_edges"] as? Bool) ?? true,
                                      sigma: (kw["sigma"] as? NSNumber)?.doubleValue ?? 0,
                                      scaleRadial: (kw["scale_radial"] as? NSNumber)?.doubleValue)
            }
            if ent["refused"] != nil { #expect(throws: CuboVirtualError.self) { try call() }; continue }
            let got = try call()
            #expect(got == b(ent["mask"]), "\(name): beamstop \(key)")
            if integer { #expect(got == b(ent["mask_py4dstem"]), "\(name): beamstop \(key) vs py4DSTEM") }
            log.append("beamstop \(key): py4DSTEM own differs at \(ent["differ_py4dstem"]!)")
        }
        // 3.3 Friedel
        for (key, ent) in e["friedel"] as! [String: [String: Any]] {
            let mask = ent["mask"] is NSNull || ent["mask"] == nil ? nil : b(ent["mask"])
            let got = try await FriedelOrigin.measure(of: cube, mask: mask, upsampleFactor: (ent["upsample"] as! NSNumber).intValue)
            let gap = d(ent["gap"])
            let ties = Set(gap.indices.filter { gap[$0] <= 1e-12 })
            let df = Swift.max(distance(got.row.values, d(ent["row"]), skip: ties), distance(got.column.values, d(ent["column"]), skip: ties))
            #expect(df <= 1e-9, "\(name): Friedel \(key) \(df) px")
            log.append("Friedel \(key): \(df) px, near ties \(ties.count), py4DSTEM own \(ent["distance_py4dstem_own_px"]!) px")
        }
        // 3.5 the vacuum probe, 3.6 its kernels
        let v = e["vacuum"] as! [String: Any]
        try await checkVacuum(v, cube: cube, name: name, log: &log)
        let probe = image(d(v["probe"]), qy, qx)
        try checkKernels(e["kernels"] as! [[String: Any]], probe: probe, name: name, log: &log)
        print(log.joined(separator: "\n  "))
    }

    static func checkVacuum(_ v: [String: Any], cube: MappedCube, name: String, log: inout [String]) async throws {
        let shape = cube.shape
        let roi = v["roi"] is NSNull || v["roi"] == nil ? nil : b(v["roi"])
        let dmask = v["detector_mask"] is NSNull || v["detector_mask"] == nil ? nil : d(v["detector_mask"])
        let got = try await VacuumProbe.measure(of: cube, scanMask: roi, align: v["align"] as! Bool,
                                                zeroVacuum: v["zero_vacuum"] as! Bool, detectorMask: dmask)
        assertRelative(got.averaged.values, d(v["averaged"]), 1e-12, "\(name) vacuum averaged")
        let pinned = try ProbeSize.estimate(image(d(v["averaged"]), shape.detectorRows, shape.detectorColumns))
        #expect(pinned.radius.bitPattern == Double(v["radius"] as! String)!.bitPattern, "\(name): vacuum radius")
        #expect(pinned.centerRow.bitPattern == Double(v["center_row"] as! String)!.bitPattern, "\(name): vacuum centre row")
        #expect(pinned.centerColumn.bitPattern == Double(v["center_column"] as! String)!.bitPattern, "\(name): vacuum centre column")
        #expect(abs(got.estimate.radius - pinned.radius) <= 1e-12 * pinned.radius, "\(name): vacuum radius from the kit's average")
        assertRelative(got.probe.values, d(v["probe"]), 1e-12, "\(name) vacuum probe")
        log.append("vacuum: py4DSTEM own vs float64 \(v["distance_py4dstem_own"]!) normwise, near ties \(v["near_ties"]!)")
    }

    static func checkKernels(_ cases: [[String: Any]], probe: DetectorImage, name: String, log: inout [String]) throws {
        var worst = 0.0
        for c in cases {
            let origin = (c["origin"] as? [NSNumber]).map { (row: $0[0].doubleValue, column: $0[1].doubleValue) }
            let kind = c["kind"] as! String
            let k: DetectorImage
            switch kind {
            case "flat":
                k = try ProbeKernel.flat(probe, origin: origin, bilinear: (c["bilinear"] as? Bool) ?? false)
            case "gaussian":
                k = try ProbeKernel.gaussian(probe, sigma: n(c["sigma"]), origin: origin, bilinear: (c["bilinear"] as? Bool) ?? true)
            default:
                let r = c["radii"] as! [NSNumber]
                k = try ProbeKernel.sigmoid(probe, innerRadius: r[0].doubleValue, outerRadius: r[1].doubleValue,
                                            profile: (c["profile"] as? String) == "logistic" ? .logistic : .sineSquared,
                                            origin: origin, bilinear: (c["bilinear"] as? Bool) ?? true)
            }
            let dk = normwise(k.values, d(c["kernel"]))
            #expect(dk <= 1e-12, "\(name): kernel \(c.filter { $0.key != "kernel" }) \(dk)")
            worst = Swift.max(worst, dk)
        }
        log.append("kernels: worst \(worst) normwise of \(cases.count)")
    }
}

@Suite struct Slice4Tests {
    @Test(arguments: ["planted_f32", "planted_u16", "gaussian", "counts16"])
    func testSlice4PinsOnSyntheticCubes(_ name: String) async throws {
        let e = (S4.pins["cubes"] as! [[String: Any]]).first { $0["cube"] as! String == name }!
        try await S4.check(e, cube: try MappedCube(npy: Fixtures.url("\(name).npy")))
    }

    /// 5: the median is the closed form; one outlier position moves it at most one rank.
    @Test func testMedianTruth() async throws {
        for t in S4.pins["median_truth"] as! [[String: Any]] {
            let nm = t["cube"] as! String
            let m = try await VirtualDiffraction.median(of: try MappedCube(npy: Fixtures.url("\(nm).npy")))
            assertIdentical(m.values, S4.d(t["median"]), "\(nm) median")
            let o = try await VirtualDiffraction.median(of: try MappedCube(npy: Fixtures.url("\(nm)_outlier.npy")))
            assertIdentical(o.values, S4.d(t["median_outlier"]), "\(nm) median with an outlier")
            let mean = try await VirtualDiffraction.mean(of: try MappedCube(npy: Fixtures.url("\(nm)_outlier.npy")))
            #expect(mean.values.allSatisfy { $0 > 1e27 }, "the outlier moves the mean")
        }
    }

    /// 5: one selected position gives its pattern; weights all one give the mean; one-hot weights give the pattern and max(P, 0).
    @Test func testSelectionTruth() async throws {
        let cube = try MappedCube(npy: Fixtures.url("counts16.npy"))
        let shape = cube.shape, N = shape.scanCount
        let p = try await cube.readPattern(row: 1, column: 2).values.map(Double.init)
        let k = shape.scanIndex(row: 1, column: 2)
        var one = [Bool](repeating: false, count: N); one[k] = true
        for stat in DiffractionStatistic.allCases {
            let got = try await VirtualDiffraction.pattern(of: cube, statistic: stat, scan: .mask(one))
            #expect(got.values == p, "one position: \(stat)")
        }
        var hot = [Double](repeating: 0, count: N); hot[k] = 1
        #expect(try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .weights(hot)).values == p)
        #expect(try await VirtualDiffraction.pattern(of: cube, statistic: .max, scan: .weights(hot)).values == p.map { Swift.max($0, 0) })
        let ones = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .weights(Array(repeating: 1, count: N)))
        let mean = try await VirtualDiffraction.mean(of: cube)
        assertRelative(ones.values, mean.values, 1e-15, "weights all one")
        let all = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .all)
        #expect(all.values == mean.values, "the .all mean is slice 1's, bit for bit")
        let mx = try await VirtualDiffraction.max(of: cube)
        #expect(try await VirtualDiffraction.pattern(of: cube, statistic: .max).values == mx.values)
    }

    /// 3.1, 3.3, 3.4, 3.6: the refusals where py4DSTEM returns NaN or ∞ or raises.
    @Test func testRefusals() async throws {
        let cube = try MappedCube(npy: Fixtures.url("counts16.npy"))
        let N = cube.shape.scanCount
        await #expect(throws: CuboVirtualError.self) {
            try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .mask(Array(repeating: false, count: N)))
        }
        await #expect(throws: CuboVirtualError.self) {
            var w = [Double](repeating: 1, count: N); w[0] = -Double(N - 1)
            return try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .weights(w))
        }
        await #expect(throws: CuboVirtualError.maskLengthMismatch(expected: N, received: 3)) {
            try await VirtualDiffraction.pattern(of: cube, statistic: .max, scan: .mask([true, true, true]))
        }
        await #expect(throws: CuboVirtualError.self) {
            var w = [Double](repeating: 1, count: N); w[2] = .nan
            return try await VirtualDiffraction.pattern(of: cube, statistic: .max, scan: .weights(w))
        }
        let map = try S4.map(Array(repeating: 3, count: N), Array(repeating: 4, count: N), cube.shape)
        await #expect(throws: CuboVirtualError.self) {
            try await VirtualDiffraction.pattern(of: cube, statistic: .median, shift: .rounded(map))
        }
        await #expect(throws: CuboVirtualError.self) { try await FriedelOrigin.measure(of: cube, upsampleFactor: 2) }
        await #expect(throws: CuboVirtualError.self) { try await FriedelOrigin.measure(of: cube, upsampleFactor: 0) }
        let flat = S4.image(Array(repeating: 1, count: 64), 8, 8)
        #expect(throws: CuboVirtualError.self) { try BeamstopMask.make(from: flat, threshold: 1.0) }
        #expect(throws: CuboVirtualError.self) { try BeamstopMask.make(from: flat, threshold: 0, includeEdges: false) }
        #expect(throws: CuboVirtualError.self) { try ProbeKernel.sigmoid(flat, innerRadius: 3, outerRadius: 3, origin: (row: 4, column: 4)) }
        #expect(throws: CuboVirtualError.self) { try ProbeKernel.flat(S4.image(Array(repeating: 0, count: 64), 8, 8), origin: (row: 4, column: 4)) }
        await #expect(throws: CuboVirtualError.self) {
            try await VacuumProbe.measure(of: cube, scanMask: Array(repeating: false, count: N))
        }
    }

    /// By hand, the edges of the definitions: NaN in a median; the maximum of negative data; the Friedel origin reported
    /// modulo the detector (np.remainder); the threshold index rounded half to even; holes filled with 4-connectivity.
    @Test func testEdgesByHand() async throws {
        let nan = try MemoryCube(shape: CubeShape(scanRows: 1, scanColumns: 3, detectorRows: 1, detectorColumns: 2),
                                 values: [1, 5, .nan, 6, 3, 7])
        let m = try await VirtualDiffraction.median(of: nan)
        #expect(m.values[0].isNaN && m.values[1] == 6)
        let negative = try MemoryCube(shape: CubeShape(scanRows: 2, scanColumns: 2, detectorRows: 1, detectorColumns: 2),
                                      values: [-4, -9, -3, -8, -5, -7, -6, -10])
        let all = [true, true, true, true]
        #expect(try await VirtualDiffraction.pattern(of: negative, statistic: .max, scan: .mask(all)).values == [-3, -7])
        #expect(try await VirtualDiffraction.pattern(of: negative, statistic: .max, scan: .weights([1, 1, 1, 1])).values == [-3, -7])
        // rows 0 and 1 only, symmetric about column 3.5: cc peaks at (0, 7) with cc[1, 7] = −40 and cc[−1, 7] = 0, so the
        // row refinement is −40/880 and the origin row (−40/880)/2 lands at 8 − 1/44 modulo the detector
        var p = [Double](repeating: 0, count: 64)
        p[3] = 10; p[4] = 10; p[11] = -1; p[12] = -1
        let o = try FriedelOrigin.origin(of: S4.image(p, 8, 8))
        #expect(abs(o.row - (8 - 40.0 / 880 / 2)) <= 1e-9 && abs(o.column - 3.5) <= 1e-9, "\(o)")
        // N · threshold = 2.5: index 2 (half to even), the cut at the third-dimmest pixel
        var b = [Double](repeating: 100, count: 64)
        b[0] = 1; b[1] = 2; b[2] = 3
        let half = try BeamstopMask.make(from: S4.image(b, 8, 8), threshold: 2.5 / 64, distanceEdge: 0.5, includeEdges: false)
        #expect(half.indices.filter { half[$0] } == [0, 1])
        // a dark pixel touching the dark corner only diagonally is a hole (4-connectivity): filled
        var h = [Double](repeating: 100, count: 64)
        h[0] = 1; h[9] = 1
        let hole = try BeamstopMask.make(from: S4.image(h, 8, 8), threshold: 0.5, distanceEdge: 0.5, includeEdges: false)
        #expect(hole.indices.filter { hole[$0] } == [0])
    }

    /// Gate B on bc41094 (registration 9): the refusals for an all-false Friedel mask, an upsampling factor above 1024 and
    /// an empty pattern or probe (each trapped or was ill-posed); NumPy's keep-first rule for equal maxima (±0); the
    /// kernels' messages name the normalisation that failed.
    @Test func testGateBRegressions() async throws {
        let cube = try MappedCube(npy: Fixtures.url("friedel_plain.npy"))
        await #expect(throws: CuboVirtualError.invalidInput("the mask keeps no pixel of the pattern")) {
            try await FriedelOrigin.measure(of: cube, mask: Array(repeating: false, count: 32 * 34))
        }
        await #expect(throws: CuboVirtualError.self) { try await FriedelOrigin.measure(of: cube, upsampleFactor: 1025) }
        _ = try await FriedelOrigin.measure(of: cube, upsampleFactor: 3)
        let empty = S4.image([], 0, 5)
        #expect(throws: CuboVirtualError.self) { try FriedelOrigin.origin(of: empty) }
        #expect(throws: CuboVirtualError.self) { try ProbeKernel.gaussian(empty, sigma: 2, origin: (row: 0, column: 0)) }
        #expect(throws: CuboVirtualError.self) {
            try ProbeKernel.sigmoid(empty, innerRadius: 1, outerRadius: 2, origin: (row: 0, column: 0))
        }
        do {   // a 1 × 1 probe: σ_q² = 0, the Gaussian's sum is NaN; the probe's sum (2.5) is fine
            _ = try ProbeKernel.gaussian(S4.image([2.5], 1, 1), sigma: 1, origin: (row: 0, column: 0))
            Issue.record("a 1 × 1 Gaussian kernel must be refused")
        } catch let CuboVirtualError.invalidInput(why) {
            #expect(why.hasPrefix("the Gaussian sums to"), "\(why)")
        }
        // ±0: np.max keeps the first of equal values, at each pixel in scan order
        let zeros = try MemoryCube(shape: CubeShape(scanRows: 1, scanColumns: 2, detectorRows: 1, detectorColumns: 2),
                                   values: [0, -0.0, -0.0, 0])
        let mx = try await VirtualDiffraction.pattern(of: zeros, statistic: .max, scan: .mask([true, true]))
        #expect(mx.values[0].sign == .plus && mx.values[1].sign == .minus, "\(mx.values.map(\.sign))")
    }

    /// 5: the fast centre of the planted disk cube is the planted centre; excluded positions are NaN, selected ones
    /// equal the unmasked run bit for bit; remove_NaN fills with the mean of the rest.
    @Test func testOriginsTruth() async throws {
        let s3 = (S3.pins["cubes"] as! [[String: Any]]).first { $0["cube"] as! String == "planted_f32" }!
        let cube = try MappedCube(npy: Fixtures.url("planted_f32.npy"))
        let (fast, _) = try await OriginMeasurement.measure(of: cube, scanMask: nil, fastCenter: true)
        #expect(fast.row.values == S3.d(s3["truth_row"]) && fast.column.values == S3.d(s3["truth_column"]))
        let N = cube.shape.scanCount
        let mask = (0..<N).map { $0 % 3 != 1 }
        let (all, _) = try await OriginMeasurement.measure(of: cube)
        let (some, _) = try await OriginMeasurement.measure(of: cube, scanMask: mask)
        for k in 0..<N {
            if mask[k] { #expect(some.row.values[k].bitPattern == all.row.values[k].bitPattern) } else {
                #expect(some.row.values[k].isNaN && some.column.values[k].isNaN)
            }
        }
        let filled = some.replacingNaNWithMean()
        let kept = (0..<N).filter { mask[$0] }.map { all.row.values[$0] }
        let fill = kept.reduce(0, +) / Double(kept.count)
        #expect((0..<N).filter { !mask[$0] }.allSatisfy { abs(filled.row.values[$0] - fill) <= 1e-12 })
        #expect(filled.meanRow.isFinite && some.meanRow.isNaN)
    }

    /// 5: point-symmetric patterns give their centres; behind a planted beamstop the masked form keeps the peak at twice
    /// the centre; Gaussians at fractional centres within 0.05 px with u = 16.
    @Test func testFriedelTruth() async throws {
        let t = S4.pins["friedel_truth"] as! [String: Any]
        let centres = (t["centres"] as! [[NSNumber]]).map { ($0[0].doubleValue, $0[1].doubleValue) }
        let plain = try await FriedelOrigin.measure(of: try MappedCube(npy: Fixtures.url("friedel_plain.npy")))
        for (k, c) in centres.enumerated() {
            #expect(abs(plain.row.values[k] - c.0) <= 1e-9 && abs(plain.column.values[k] - c.1) <= 1e-9, "centre \(k)")
        }
        let maskedCube = try MappedCube(npy: Fixtures.url("friedel_masked.npy"))
        let m = S4.b(t["mask"])
        let masked = try await FriedelOrigin.measure(of: maskedCube, mask: m)
        let unmasked = try await FriedelOrigin.measure(of: maskedCube)
        let dm = (masked.row.values.map { abs($0 - 15.5) } + masked.column.values.map { abs($0 - 16.5) }).max()!
        let du = (unmasked.row.values.map { abs($0 - 15.5) } + unmasked.column.values.map { abs($0 - 16.5) }).max()!
        #expect(dm <= 0.04 && du >= 1, "behind the beamstop: masked \(dm) px, unmasked \(du) px")
        #expect(t["masked_peak_is_twice_centre"] as! Bool)
        let g = S4.pins["friedel_gauss"] as! [String: Any]
        let gc = (g["centres"] as! [[NSNumber]]).map { ($0[0].doubleValue, $0[1].doubleValue) }
        let gauss = try await FriedelOrigin.measure(of: try MappedCube(npy: Fixtures.url("friedel_gauss.npy")), upsampleFactor: 16)
        for (k, c) in gc.enumerated() {
            #expect(abs(gauss.row.values[k] - c.0) <= 0.05 && abs(gauss.column.values[k] - c.1) <= 0.05, "Gaussian \(k)")
        }
        // the single-pattern entry equals the cube's
        let p = try await maskedCube.readPattern(row: 0, column: 1)
        let one = try FriedelOrigin.origin(of: S4.image(p.values.map(Double.init), 32, 34), mask: m)
        #expect(one.row == masked.row.values[1] && one.column == masked.column.values[1])
    }

    /// 5: the planted beamstop mask by hand: the arm and the border dilated by 2, the hole filled, the speck removed.
    @Test func testBeamstopTruth() throws {
        let t = S4.pins["beamstop_truth"] as! [String: Any]
        let shape = t["shape"] as! [Int]
        let got = try BeamstopMask.make(from: S4.image(S4.d(t["pattern"]), shape[0], shape[1]))
        #expect(got == S4.b(t["mask"]))
        #expect(got.filter { $0 }.count == (t["count"] as! Int))
        #expect(!got[11 * 24 + 6 + 12] && got[11 * 24 + 6], "the speck is under the mask, the region beside the arm is not")
    }

    /// 5: copies of one disk rolled by known shifts: the aligned average is the disk at p₀; p₀'s hot pixel is absent;
    /// without alignment the mean of p₁ … p_{L−1}; the damping and zeroing as pinned.
    @Test func testVacuumTruth() async throws {
        let t = S4.pins["vacuum_truth"] as! [String: Any]
        let cube = try MappedCube(npy: Fixtures.url("vacuum_planted.npy"))
        let a = try await VacuumProbe.measure(of: cube, zeroVacuum: false)
        assertRelative(a.averaged.values, S4.d(t["disk"]), 1e-13, "aligned average is the disk")
        #expect(a.averaged.values[2 * 32 + 28] == 0, "the hot pixel of p₀ is absent")
        let u = try await VacuumProbe.measure(of: cube, align: false, zeroVacuum: false)
        assertRelative(u.averaged.values, S4.d(t["plain_mean"]), 1e-14, "unaligned: the mean of p₁ … p_{L−1}")
        var log = ["slice4 vacuum truth"]
        for key in ["aligned", "aligned_damped", "unaligned", "detector_mask"] {
            try await S4.checkVacuum(t[key] as! [String: Any], cube: cube, name: "planted \(key)", log: &log)
        }
        let avg = S4.image(S4.d((t["aligned_damped"] as! [String: Any])["averaged"]), 32, 32)
        let z = t["zero_vacuum"] as! [String: Any], c = z["center"] as! [NSNumber]
        let zv = VacuumProbe.zeroVacuum(avg, center: (row: c[0].doubleValue, column: c[1].doubleValue), alpha: S4.n(z["alpha"]), alphaMax: 1.2)
        assertRelative(zv.values, S4.d(z["value"]), 1e-12, "zero_vacuum")
        assertRelative(zv.values, S4.d(z["value_py4dstem"]), 1e-12, "zero_vacuum vs py4DSTEM")
        let dv = t["damp_vacuum"] as! [String: Any], dc = dv["center"] as! [NSNumber]
        let dd = VacuumProbe.dampVacuum(avg, center: (row: dc[0].doubleValue, column: dc[1].doubleValue), alpha: S4.n(dv["alpha"]),
                                        start: S4.n(dv["s"]), width: S4.n(dv["w"]), power: S4.n(dv["p"]))
        assertRelative(dd.values, S4.d(dv["value"]), 1e-12, "zero_vacuum_sinusoid")
        assertRelative(dd.values, S4.d(dv["value_py4dstem"]), 1e-12, "zero_vacuum_sinusoid vs py4DSTEM")
        print(log.joined(separator: "\n  "))
    }

    /// 5: synthetic probes as py4DSTEM's (1/2 on the radius); the flat kernel sums to 1, the others to 0; the kernels pinned.
    @Test func testKernelTruth() throws {
        let sps = S4.pins["synthetic_probes"] as! [[String: Any]]
        for sp in sps {
            let shape = sp["shape"] as! [Int]
            let p = try SyntheticProbe.make(radius: S4.n(sp["radius"]), width: S4.n(sp["width"]),
                                            detectorRows: shape[0], detectorColumns: shape[1])
            #expect(S4.normwise(p.values, S4.d(sp["probe"])) <= 1e-12)
        }
        let probe = try SyntheticProbe.make(radius: 6, width: 2, detectorRows: 32, detectorColumns: 30)
        #expect(probe[22, 15] == 0.5, "on the radius")
        let kt = S4.pins["kernels"] as! [String: Any]
        var log = ["slice4 kernel truth"]
        try S4.checkKernels(kt["cases"] as! [[String: Any]], probe: probe, name: "synthetic", log: &log)
        try S4.checkKernels(kt["cases_given_origin"] as! [[String: Any]], probe: probe, name: "synthetic, origin given", log: &log)
        #expect(abs(try ProbeKernel.flat(probe).values.reduce(0, +) - 1) <= 1e-12)
        #expect(abs(try ProbeKernel.gaussian(probe, sigma: 4).values.reduce(0, +)) <= 1e-12)
        #expect(abs(try ProbeKernel.sigmoid(probe, innerRadius: 6, outerRadius: 12).values.reduce(0, +)) <= 1e-12)
        // a probe point-symmetric about an integer centre: the flat kernel is symmetric about the corner
        let sym = try SyntheticProbe.make(radius: 5, width: 1.5, detectorRows: 24, detectorColumns: 24)
        let k = try ProbeKernel.flat(sym, origin: (row: 12, column: 12), bilinear: true)
        #expect((0..<24).allSatisfy { i in (0..<24).allSatisfy { j in abs(k[i, j] - k[(24 - i) % 24, (24 - j) % 24]) <= 1e-15 } })
        print(log.joined(separator: "\n  "))
    }
}

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
