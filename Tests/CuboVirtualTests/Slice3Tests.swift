//
//  Slice3Tests.swift
//  Per-position origins, fits, centred, calibrated and shifted virtual imaging (registration slice 3) against py4DSTEM
//  f050d207 and planted truth: slice-3-pins.json and, on this machine, the real-data exports (<name>-s3-pins.json).
//

import Foundation
import Testing
import CuboCore
import CuboVirtual

enum S3 {
    nonisolated(unsafe) static let pins: [String: Any] = {
        try! JSONSerialization.jsonObject(with: Data(contentsOf: Fixtures.url("slice-3-pins.json"))) as! [String: Any]
    }()
    static func d(_ a: Any?) -> [Double] { (a as? [String] ?? []).map { Double($0)! } }
    static func geometry(_ g: [String: Any]) -> DetectorGeometry {
        let n = { (k: String) in (g[k] as! NSNumber).doubleValue }
        return g["kind"] as! String == "circle"
            ? .circle(centerRow: n("centerRow"), centerColumn: n("centerColumn"), radius: n("radius"))
            : .annulus(centerRow: n("centerRow"), centerColumn: n("centerColumn"), inner: n("inner"), outer: n("outer"))
    }
    static func realIsPresent(_ name: String) -> Bool {
        FileManager.default.fileExists(atPath: Fixtures.realData.appendingPathComponent("\(name)-s3-pins.json").path)
    }
    static let fitModels: [String: OriginFitModel] = ["constant": .constant, "plane": .plane, "parabola": .parabola,
                                                      "bezier_two": .bezierTwo]

    /// Every claim of section 4 on one pinned cube; prints what it measured.
    static func check(_ e: [String: Any], cube: MappedCube, integer: Bool) async throws {
        let name = e["cube"] as! String
        let shape = cube.shape
        let r = Double(e["radius"] as! String)!
        var log = ["slice3 \(name)"]
        // origins: the default radius is the probe size of the max pattern, bit for bit; origins 1e-9 px
        let (measured, radius) = try await OriginMeasurement.measure(of: cube)
        #expect(radius.bitPattern == r.bitPattern, "\(name): radius \(radius) vs \(r)")
        let orow = d(e["origin_row"]), ocol = d(e["origin_column"])
        let dO = zip(measured.row.values + measured.column.values, orow + ocol).map { abs($0 - $1) }.max()!
        #expect(dO <= 1e-9, "\(name): origins \(dO) px")
        log.append("origins: kit vs float64 \(dO) px, py4DSTEM own vs float64 \(e["origin_distance_py4dstem_own_vs_f64_px"]!) px")
        // the filter
        let p0 = try await cube.readPattern(row: 0, column: 0)
        let g = GaussianFilter(sigma: r).apply(p0.values.map(Double.init), rows: shape.detectorRows, columns: shape.detectorColumns)
        assertRelative(g, d(e["filtered_00"]), 1e-12, "\(name) filtered")
        // fits on the float64 origins
        for (key, value) in e["fits"] as! [String: [String: Any]] {
            let robust = key.hasSuffix("_robust")
            let model = fitModels[robust ? String(key.dropLast(7)) : key]!
            let fr = try OriginFit.fit(orow, rows: shape.scanRows, columns: shape.scanColumns, model: model, robust: robust)
            let df = zip(fr, d(value["row"])).map { abs($0 - $1) }.max()!
            #expect(df <= 1e-9, "\(name): fit \(key) \(df) px")
        }
        // images
        let map = try OriginMap(row: ScanImage(scanRows: shape.scanRows, scanColumns: shape.scanColumns, values: orow, path: .cpu),
                                column: ScanImage(scanRows: shape.scanRows, scanColumns: shape.scanColumns, values: ocol, path: .cpu))
        for im in e["images"] as! [[String: Any]] {
            let geo = geometry(im["geometry"] as! [String: Any])
            let origin = (row: map.meanRow, column: map.meanColumn)
            let c = try await VirtualImaging.image(of: cube, geometry: geo, centeredOn: origin)
            let ro = try await VirtualImaging.image(of: cube, geometry: geo, centeredOn: origin, shift: .rounded(map))
            let bl = try await VirtualImaging.image(of: cube, geometry: geo, centeredOn: origin, shift: .bilinear(map))
            let cal = try await VirtualImaging.image(of: cube, geometry: geometry(im["calibrated_geometry"] as! [String: Any]),
                                                     centeredOn: nil, pixelSize: (im["pixel_size"] as! NSNumber).doubleValue)
            if integer {
                assertIdentical(c.values, d(im["centered_py4dstem"]), "\(name) centred vs py4DSTEM")
                assertIdentical(ro.values, d(im["rounded_py4dstem"]), "\(name) rounded shift vs py4DSTEM")
                assertIdentical(cal.values, d(im["calibrated_py4dstem"]), "\(name) calibrated vs py4DSTEM")
            }
            assertRelative(c.values, d(im["centered"]), 1e-10, "\(name) centred")
            assertRelative(ro.values, d(im["rounded"]), 1e-10, "\(name) rounded shift")
            assertRelative(bl.values, d(im["bilinear"]), 1e-10, "\(name) bilinear shift")
            assertRelative(cal.values, d(im["calibrated"]), 1e-10, "\(name) calibrated")
        }
        // shifted patterns
        let pats = e["patterns"] as! [String: [String: Any]]
        for (key, shift) in [("rounded", PatternShift.rounded(map)), ("fourier", PatternShift.fourier(map))] {
            let (mean, max) = try await VirtualDiffraction.meanAndMax(of: cube, shift: shift)
            let pm = pats[key]!
            assertRelative(mean.values, d(pm["mean"]), 1e-10, "\(name) \(key) mean")
            if key == "rounded" {
                assertIdentical(max.values, d(pm["max"]), "\(name) rounded max")
                assertRelative(mean.values, d(pm["mean_py4dstem"]), 1e-10, "\(name) rounded mean vs py4DSTEM")
                let own = d(pm["max_py4dstem"]), mine = max.values
                let k = mine.indices.first { mine[$0] >= 0 && mine[$0].bitPattern != own[$0].bitPattern }
                #expect(k == nil, "\(name): rounded max vs py4DSTEM where ≥ 0")
            } else {
                assertRelative(max.values, d(pm["max"]), 1e-10, "\(name) fourier max")
            }
            log.append("\(key) mean: py4DSTEM own vs float64 \(pm["mean_distance_py4dstem"]!)")
        }
        print(log.joined(separator: "\n  "))
    }
}

@Suite struct Slice3Tests {
    @Test(arguments: ["planted_f32", "planted_u16", "gaussian", "counts16"])
    func testSlice3PinsOnSyntheticCubes(_ name: String) async throws {
        let e = (S3.pins["cubes"] as! [[String: Any]]).first { $0["cube"] as! String == name }!
        try await S3.check(e, cube: try MappedCube(npy: Fixtures.url("\(name).npy")), integer: name == "planted_u16" || name == "counts16")
    }

    /// 5: the origins of the planted cubes are the planted centres; the rounded shifted mean is the unshifted disk.
    @Test(arguments: ["planted_f32", "planted_u16"])
    func testPlantedBeamShiftIsRecovered(_ name: String) async throws {
        let e = (S3.pins["cubes"] as! [[String: Any]]).first { $0["cube"] as! String == name }!
        let cube = try MappedCube(npy: Fixtures.url("\(name).npy"))
        let (o, _) = try await OriginMeasurement.measure(of: cube)
        let dr = zip(o.row.values, S3.d(e["truth_row"])).map { abs($0 - $1) }.max()!
        let dc = zip(o.column.values, S3.d(e["truth_column"])).map { abs($0 - $1) }.max()!
        #expect(dr <= 1e-9 && dc <= 1e-9, "planted centres \(dr), \(dc)")
        let (mean, _) = try await VirtualDiffraction.meanAndMax(of: cube, shift: .rounded(o))
        let p0 = try await cube.readPattern(row: 0, column: 0)
        let disk = VirtualImagingTestRoll.roll(p0.values.map(Double.init), 24, 25,
                                               Int((o.meanRow - o.row.values[0]).rounded(.toNearestOrEven)),
                                               Int((o.meanColumn - o.column.values[0]).rounded(.toNearestOrEven)))
        #expect(mean.values == disk)
    }

    /// 5: exact surfaces recovered by their own model; one outlier removed by the robust fit.
    @Test func testFitTruth() throws {
        for (key, t) in S3.pins["fit_truth"] as! [String: [String: Any]] {
            let model = S3.fitModels[key]!, shape = t["shape"] as! [Int]
            let s = S3.d(t["surface"]), o = S3.d(t["with_outlier"])
            let f = try OriginFit.fit(s, rows: shape[0], columns: shape[1], model: model)
            #expect(zip(f, s).allSatisfy { abs($0 - $1) <= 1e-10 }, "\(key)")
            let rf = try OriginFit.fit(o, rows: shape[0], columns: shape[1], model: model, robust: true)
            #expect(zip(rf, s).allSatisfy { abs($0 - $1) <= 1e-9 }, "robust \(key)")
            let plain = try OriginFit.fit(o, rows: shape[0], columns: shape[1], model: model)
            #expect(zip(plain, s).contains { abs($0 - $1) > 1e-3 }, "the outlier must move the plain \(key) fit")
        }
    }

    /// 3.1 by hand: a constant pattern's argmax is the first pixel (0, 0); the mask keeps |q − argmax| < r·rscale
    /// strictly (with r = 3, rscale = 1 the axial pixels at distance exactly 3 are out).
    @Test func testArgmaxTiesAndTheStrictMaskBoundary() async throws {
        let q = 9
        let flat = try MemoryCube(shape: try CubeShape([1, 1, q, q]), values: Array(repeating: 2, count: q * q))
        let (o, _) = try await OriginMeasurement.measure(of: flat, radius: 3, radiusScale: 1)
        var n = 0.0, sr = 0.0, sc = 0.0
        for i in 0..<q { for j in 0..<q where (Double(i * i + j * j)).squareRoot() < 3 { n += 1; sr += Double(i); sc += Double(j) } }
        #expect(abs(o.row.values[0] - sr / n) <= 1e-12 && abs(o.column.values[0] - sc / n) <= 1e-12, "\(o.row.values[0]) vs \(sr / n)")
        // a bright pixel at (4, 4) on a gradient: the axial pixels at distance exactly 3 carry the gradient
        let values: [Float] = (0..<(q * q)).map { $0 == 4 * q + 4 ? 1000 : Float(1 + $0 / q) }
        let cube = try MemoryCube(shape: try CubeShape([1, 1, q, q]), values: values)
        let (b, _) = try await OriginMeasurement.measure(of: cube, radius: 3, radiusScale: 1)
        var s = 0.0, m = 0.0
        for k in 0..<(q * q) {
            let di = Double(k / q - 4), dj = Double(k % q - 4)
            guard (di * di + dj * dj).squareRoot() < 3 else { continue }
            s += Double(values[k]); m += Double(k / q) * Double(values[k])
        }
        #expect(abs(b.row.values[0] - m / s) <= 1e-12, "\(b.row.values[0]) vs \(m / s)")
    }

    /// 3.3: shifts of exactly ±0.5 round half to even (to 0): the rounded shifted image is the unshifted one.
    @Test func testHalfShiftsRoundToEven() async throws {
        let cube = try MappedCube(npy: Fixtures.url("gaussian.npy"))
        let rows = (0..<30).map { Double($0 % 2) }          // mean 0.5: every shift is ±0.5
        let map = try OriginMap(row: ScanImage(scanRows: 6, scanColumns: 5, values: rows, path: .cpu),
                                column: ScanImage(scanRows: 6, scanColumns: 5, values: rows, path: .cpu))
        let g = DetectorGeometry.circle(centerRow: 7, centerColumn: 8, radius: 4.5)
        let shifted = try await VirtualImaging.image(of: cube, geometry: g, centeredOn: nil, shift: .rounded(map))
        let plain = try await VirtualImaging.image(of: cube, geometry: g)
        assertRelative(shifted.values, plain.values, 1e-12, "half shifts")
    }

    @Test func testKernelCentringAndRefusals() async throws {
        let k = GaussianFilter(sigma: 2.6).kernel
        #expect(abs(k.reduce(0, +) - 1) <= 1e-15 && k == k.reversed() && k.count == 2 * Int(4 * 2.6 + 0.5) + 1)
        let flat = GaussianFilter(sigma: 1.7).apply(Array(repeating: 3.25, count: 63), rows: 7, columns: 9)
        #expect(flat.allSatisfy { abs($0 - 3.25) <= 1e-14 })
        let cube = try MappedCube(npy: Fixtures.url("gaussian.npy"))
        let a = try await VirtualImaging.image(of: cube, geometry: .circle(centerRow: 0, centerColumn: 0, radius: 3.3),
                                               centeredOn: (row: 7.25, column: 8.5))
        let b = try await VirtualImaging.image(of: cube, geometry: .circle(centerRow: 7.25, centerColumn: 8.5, radius: 3.3))
        #expect(a.values == b.values)
        await #expect(throws: CuboVirtualError.centeredAndCalibrated) {
            try await VirtualImaging.image(of: cube, geometry: .circle(centerRow: 0, centerColumn: 0, radius: 1),
                                           centeredOn: (row: 1, column: 1), pixelSize: 0.02)
        }
        let map = try OriginMap(row: ScanImage(scanRows: 6, scanColumns: 5, values: Array(repeating: 7, count: 30), path: .cpu),
                                column: ScanImage(scanRows: 6, scanColumns: 5, values: Array(repeating: 8, count: 30), path: .cpu))
        await #expect(throws: CuboVirtualError.unsupportedOnMetal("a shifted virtual image")) {
            try await VirtualImaging.image(of: cube, geometry: .circle(centerRow: 0, centerColumn: 0, radius: 1),
                                           centeredOn: nil, shift: .rounded(map), path: .metal)
        }
        #expect(throws: OriginFitError.self) { try OriginFit.fit([1, 2], rows: 1, columns: 2, model: .parabola) }
    }
}

enum VirtualImagingTestRoll {
    static func roll(_ a: [Double], _ qy: Int, _ qx: Int, _ dr: Int, _ dc: Int) -> [Double] {
        var out = [Double](repeating: 0, count: a.count)
        for i in 0..<qy { for j in 0..<qx { out[(((i + dr) % qy + qy) % qy) * qx + ((j + dc) % qx + qx) % qx] = a[i * qx + j] } }
        return out
    }
}

@Suite(.serialized) struct Slice3RealDataTests {
    static func run(_ name: String) async throws {
        let dir = Fixtures.realData
        let e = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("\(name)-s3-pins.json"))) as! [String: Any]
        let cube = try MappedCube(npy: dir.appendingPathComponent("\(name).npy"))
        try await S3.check(e, cube: cube, integer: cube.elementType != .float32 && cube.elementType != .float64)
    }
    @Test(.enabled(if: S3.realIsPresent("almgsi"), "real data absent: run tools/slice-3/pins.py real"))
    func testAlMgSiStudyCubeSlice3() async throws { try await Self.run("almgsi") }
    @Test(.enabled(if: S3.realIsPresent("auref"), "real data absent: run tools/slice-3/pins.py real"))
    func testAuReferenceCubeSlice3() async throws { try await Self.run("auref") }
    @Test(.enabled(if: S3.realIsPresent("simau"), "real data absent: run tools/slice-3/pins.py real"))
    func testSimulatedPolycrystallineGoldSlice3() async throws { try await Self.run("simau") }
    @Test(.enabled(if: S3.realIsPresent("particle"), "real data absent: run tools/slice-3/pins.py real"))
    func testParticleCubeSlice3() async throws { try await Self.run("particle") }
    @Test(.enabled(if: S3.realIsPresent("py4dstem_small"), "real data absent: run tools/slice-3/pins.py real"))
    func testPy4DSTEMSmallDatacubeSlice3() async throws { try await Self.run("py4dstem_small") }
}
