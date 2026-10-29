//
//  S3Support.swift
//  Role: Slice 3 pins and the check of one pinned cube, shared by the fast tests and the real-data parity suites.
//

import Foundation
import Testing
import CuboCore
import CuboVirtual
import CuboPhase

package enum S3 {
    package nonisolated(unsafe) static let pins: [String: Any] = {
        try! JSONSerialization.jsonObject(with: Data(contentsOf: Fixtures.url("origin-fit-pins.json"))) as! [String: Any]
    }()
    package static func d(_ a: Any?) -> [Double] { (a as? [String] ?? []).map { Double($0)! } }
    package static func geometry(_ g: [String: Any]) -> DetectorGeometry {
        let n = { (k: String) in (g[k] as! NSNumber).doubleValue }
        return g["kind"] as! String == "circle"
            ? .circle(centerRow: n("centerRow"), centerColumn: n("centerColumn"), radius: n("radius"))
            : .annulus(centerRow: n("centerRow"), centerColumn: n("centerColumn"), inner: n("inner"), outer: n("outer"))
    }
    package static func realIsPresent(_ name: String) -> Bool {
        FileManager.default.fileExists(atPath: Fixtures.realData.appendingPathComponent("\(name)-s3-pins.json").path)
    }
    package static let fitModels: [String: OriginFitModel] = ["constant": .constant, "plane": .plane, "parabola": .parabola,
                                                      "bezier_two": .bezierTwo]

    /// Every pinned claim on one cube; prints what it measured.
    package static func check(_ e: [String: Any], cube: MappedCube, integer: Bool) async throws {
        let name = e["cube"] as! String
        let shape = cube.shape
        let r = Double(e["radius"] as! String)!
        var log = ["slice3 \(name)"]
        // origins: the default radius is the probe size of the max pattern, asserted exact today; origins 1e-9 px
        let (measured, radius) = try await OriginMeasurement.measure(of: cube)
        #expect(radius.bitPattern == r.bitPattern, "\(name): radius \(radius) vs \(r)")
        let orow = d(e["origin_row"]), ocol = d(e["origin_column"])
        let dO = zip(measured.row.values + measured.column.values, orow + ocol).map { abs($0 - $1) }.max()!
        #expect(dO <= 1e-9, "\(name): origins \(dO) px")
        log.append("origins: kit vs float64 \(dO) px, py4DSTEM own vs float64 \(e["origin_distance_py4dstem_own_vs_f64_px"]!) px")
        // the filter
        let p0 = try await cube.readPattern(row: 0, column: 0)
        let g = try GaussianFilter(sigma: r).apply(p0.values.map(Double.init), rows: shape.detectorRows, columns: shape.detectorColumns)
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
            if key == "rounded" {
                assertRelative(mean.values, d(pm["mean"]), 1e-10, "\(name) \(key) mean")
            } else {   // the Fourier shift's error is normwise: |Δ| ≤ 1e-12 · max|want|
                let want = d(pm["mean"]), top = want.map(abs).max()!
                #expect(zip(mean.values, want).allSatisfy { abs($0 - $1) <= 1e-12 * top }, "\(name) fourier mean")
            }
            if key == "rounded" {
                assertIdentical(max.values, d(pm["max"]), "\(name) rounded max")
                assertRelative(mean.values, d(pm["mean_py4dstem"]), 1e-10, "\(name) rounded mean vs py4DSTEM")
                let own = d(pm["max_py4dstem"]), mine = max.values
                let k = mine.indices.first { mine[$0] >= 0 && mine[$0].bitPattern != own[$0].bitPattern }
                #expect(k == nil, "\(name): rounded max vs py4DSTEM where ≥ 0")
            } else {
                let want = d(pm["max"]), top = want.map(abs).max()!
                #expect(zip(max.values, want).allSatisfy { abs($0 - $1) <= 1e-12 * top }, "\(name) fourier max")
            }
            log.append("\(key) mean: py4DSTEM own vs float64 \(pm["mean_distance_py4dstem"]!)")
        }
        print(log.joined(separator: "\n  "))
    }
}
