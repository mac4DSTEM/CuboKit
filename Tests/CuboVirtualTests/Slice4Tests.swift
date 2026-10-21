//
//  Slice4Tests.swift
//  The median pattern, virtual diffraction over a selection, origins with a scan mask, the Friedel origin, the beamstop
//  mask, the vacuum probe and its kernels (registration slice 4) against py4DSTEM f050d207 and planted truth:
//  slice-4-pins.json and, on this machine, the real-data exports (<name>-s4-pins.json).
//

import Foundation
import Testing
import CuboKitTestSupport
import CuboCore
import CuboVirtual


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
        #expect(all.values == mean.values, "the .all mean is slice 1's, exactly")
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
        // NaN first among five: a sort leaves it in front and the middle finite, so only the NaN rule makes it NaN
        let nan = try MemoryCube(shape: CubeShape(scanRows: 1, scanColumns: 5, detectorRows: 1, detectorColumns: 2),
                                 values: [.nan, 5, 1, 6, 2, 7, 3, 8, 4, 9])
        let m = try await VirtualDiffraction.median(of: nan)
        #expect(m.values[0].isNaN && m.values[1] == 7)
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
    /// equal the unmasked run exactly; remove_NaN fills with the mean of the rest.
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
