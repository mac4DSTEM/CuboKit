//
//  Slice3Tests.swift
//  Per-position origins, fits, centred, calibrated and shifted virtual imaging against py4DSTEM
//  f050d207 and planted truth: slice-3-pins.json and, on this machine, the real-data exports (<name>-s3-pins.json).
//

import Foundation
import Testing
import CuboKitTestSupport
import CuboCore
import CuboVirtual

@Suite struct Slice3Tests {
    @Test(arguments: ["planted_f32", "planted_u16", "gaussian", "counts16"])
    func testSlice3PinsOnSyntheticCubes(_ name: String) async throws {
        let e = (S3.pins["cubes"] as! [[String: Any]]).first { $0["cube"] as! String == name }!
        try await S3.check(e, cube: try MappedCube(npy: Fixtures.url("\(name).npy")), integer: name == "planted_u16" || name == "counts16")
    }

    /// The origins of the planted cubes are the planted centres; the rounded shifted mean is the unshifted disk.
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

    /// Exact surfaces recovered by their own model; one outlier removed by the robust fit.
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

    /// Shifts of exactly ±0.5 round half to even (to 0): the rounded shifted image is the unshifted one.
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

    /// A mirror-symmetric pattern filters to exactly symmetric values in scipy's order, so the first
    /// of the tied maxima wins: a disk centred at (11.5, 12.5) has its origin at (11, 12).
    @Test func testSymmetricTiesKeepTheFirstArgmax() async throws {
        let q = (24, 25)
        let values: [Float] = (0..<(q.0 * q.1)).map { k in
            let i = Double(k / q.1) - 11.5, j = Double(k % q.1) - 12.5
            return i * i + j * j < 6.1 * 6.1 ? 1000 : 0
        }
        let cube = try MemoryCube(shape: try CubeShape([1, 1, q.0, q.1]), values: values)
        let (o, _) = try await OriginMeasurement.measure(of: cube, radius: 3.3, radiusScale: 1.2)
        #expect(o.row.values[0] == 11 && o.column.values[0] == 12, "\(o.row.values[0]), \(o.column.values[0])")
        // NaN anywhere makes the origin NaN (get_CoM(dp · mask)); the filter refuses an unusable sigma
        var withNaN = values; withNaN[0] = .nan
        let nanCube = try MemoryCube(shape: try CubeShape([1, 1, q.0, q.1]), values: withNaN)
        let (n, _) = try await OriginMeasurement.measure(of: nanCube, radius: 3.3)
        #expect(n.row.values[0].isNaN)
        for bad in [-1.0, 0, .nan, .infinity, 1e300, 1e-160] { #expect(throws: CuboVirtualError.self) { try GaussianFilter(sigma: bad) } }
        // a point centred on a NaN origin is refused, not trapped; a rank-deficient plane fit (1 × 5) is the projection
        await #expect(throws: CuboVirtualError.self) {
            try await VirtualImaging.image(of: cube, geometry: .point(row: 0, column: 0), centeredOn: (row: .nan, column: 0))
        }
        let line = try OriginFit.fit([1, 2, 3.5, 4, 5], rows: 1, columns: 5, model: .plane)
        #expect(line.count == 5 && line.allSatisfy(\.isFinite))
    }

    @Test func testKernelCentringAndRefusals() async throws {
        let k = try GaussianFilter(sigma: 2.6).kernel
        #expect(abs(k.reduce(0, +) - 1) <= 1e-15 && k == k.reversed() && k.count == 2 * Int(4 * 2.6 + 0.5) + 1)
        // scipy's radius int(4σ + 0.5): σ = 1.7 gives 7, where int(4σ) would give 6
        #expect(try GaussianFilter(sigma: 1.7).radius == 7)
        #expect(try GaussianFilter(sigma: 2.6).radius == 10)
        let flat = try GaussianFilter(sigma: 1.7).apply(Array(repeating: 3.25, count: 63), rows: 7, columns: 9)
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
        // rank-deficient: the projection (here the data itself), as NumPy's lstsq; no finite value: refused
        let two = try OriginFit.fit([1, 2], rows: 1, columns: 2, model: .parabola)
        #expect(abs(two[0] - 1) <= 1e-12 && abs(two[1] - 2) <= 1e-12)
        #expect(throws: OriginFitError.self) { try OriginFit.fit([.nan, .nan], rows: 1, columns: 2, model: .plane) }
    }
}

enum VirtualImagingTestRoll {
    static func roll(_ a: [Double], _ qy: Int, _ qx: Int, _ dr: Int, _ dc: Int) -> [Double] {
        var out = [Double](repeating: 0, count: a.count)
        for i in 0..<qy { for j in 0..<qx { out[(((i + dr) % qy + qy) % qy) * qx + ((j + dc) % qx + qx) % qx] = a[i * qx + j] } }
        return out
    }
}
