//
//  S4Support.swift
//  Role: Slice 4 pins and the check of one pinned cube, shared by the fast tests and the real-data parity suites.
//

import Foundation
import Testing
import CuboCore
import CuboVirtual
import CuboPhase

package enum S4 {
    package nonisolated(unsafe) static let pins: [String: Any] = {
        try! JSONSerialization.jsonObject(with: Data(contentsOf: Fixtures.url("probes-pins.json"))) as! [String: Any]
    }()
    package static func d(_ a: Any?) -> [Double] { (a as? [String] ?? []).map { Double($0)! } }
    package static func b(_ a: Any?) -> [Bool] { (a as? [Int] ?? []).map { $0 != 0 } }
    package static func n(_ a: Any?) -> Double { (a as? NSNumber)?.doubleValue ?? Double(a as! String)! }
    package static func realIsPresent(_ name: String) -> Bool {
        FileManager.default.fileExists(atPath: Fixtures.realData.appendingPathComponent("\(name)-s4-pins.json").path)
    }
    package static func image(_ v: [Double], _ rows: Int, _ columns: Int) -> DetectorImage {
        DetectorImage(detectorRows: rows, detectorColumns: columns, values: v, path: .cpu)
    }
    package static func map(_ r: [Double], _ c: [Double], _ shape: CubeShape) throws -> OriginMap {
        try OriginMap(row: ScanImage(scanRows: shape.scanRows, scanColumns: shape.scanColumns, values: r, path: .cpu),
                      column: ScanImage(scanRows: shape.scanRows, scanColumns: shape.scanColumns, values: c, path: .cpu))
    }
    /// max |got − want| / max |want| (the normwise claim).
    package static func normwise(_ got: [Double], _ want: [Double]) -> Double {
        let top = want.map(abs).max() ?? 0
        return zip(got, want).map { abs($0 - $1) }.max()! / Swift.max(top, .leastNonzeroMagnitude)
    }
    /// The largest distance between two origin maps where `want` is finite; NaN must meet NaN.
    package static func distance(_ got: [Double], _ want: [Double], skip: Set<Int> = []) -> Double {
        var worst = 0.0
        for k in want.indices where !skip.contains(k) {
            if want[k].isNaN || got[k].isNaN { if want[k].isNaN != got[k].isNaN { return .infinity }; continue }
            worst = Swift.max(worst, abs(got[k] - want[k]))
        }
        return worst
    }

    /// Every pinned claim on one cube; prints what it measured.
    package static func check(_ e: [String: Any], cube: MappedCube) async throws {
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
                    if integer, key == "mask" {    // exact sums of integers, one division: exact today
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

    package static func checkVacuum(_ v: [String: Any], cube: MappedCube, name: String, log: inout [String]) async throws {
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

    package static func checkKernels(_ cases: [[String: Any]], probe: DetectorImage, name: String, log: inout [String]) throws {
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
