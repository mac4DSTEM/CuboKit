import Foundation
import Testing
import CuboCore

/// py4DSTEM has no test for these calibration numbers, so the tolerance is `assert_allclose`'s default for Float64, rtol 1e-7, with
/// atol 1e-12 so that a value that is zero in one arithmetic and 1e-17 in the other agrees (0.8.1; the pins were 1e-12 to 1e-15 before,
/// stricter than the package). Stored values and indices stay exact (`==`).
private func agrees(_ a: Double, _ b: Double) -> Bool { abs(a - b) <= 1e-12 + 1e-7 * abs(b) }

private struct Op: Decodable {
    let name: String, shape: [Int], values: [Double]
    let rPixelSize: Double?, qPixelSize: Double?
}
private struct Xs: Decodable { let r: Int, c: Int, values: [Double], valuesY: [Double] }
private struct AfterBin: Decodable { let qPixelSize: Double, originMean: [Double] }
private struct Cal: Decodable {
    let qPixelSize: Double, qPixelUnits: String, rPixelSize: Double, rPixelUnits: String
    let qx0: [Double], qy0: [Double], qx0Mean: Double, qy0Mean: Double, qx0Shift: [Double], qy0Shift: [Double]
    let qxxShape: [Int], qxx: [Double], qyy: [Double], rxx: [Double], ryy: [Double]
    let qxxRaw: [Double], qyyRaw: [Double], rxxRaw: [Double], ryyRaw: [Double]
    let qxxs: [Xs]
    let ellipse: [Double], qrRotationDegrees: Double, qrRotationRadians: Double, qrFlip: Bool, qrRotation30Radians: Double
    let noOriginQxx: [Double]
    let afterBinQ: AfterBin
}
private struct Rot: Decodable { let vx: Double, vy: Double, rotation: Double, q: [Double], r: [Double] }
private struct Processes: Decodable {
    let pixelSize: Double, pixelSizeUnits: String, dq: Double, qsFit: [Double], hklFit: [[Int]]
    let qs: [Double], hkl: [[Int]], a: Double, rotation: [Rot]
}
private struct Pin: Decodable { let sourceShape: [Int], floats: [Double], ops: [Op], calibration: Cal, processes: Processes }
private func pin() throws -> Pin {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/calibration.json")
    return try JSONDecoder().decode(Pin.self, from: Data(contentsOf: url))
}
private func transposed(_ a: [Double], rows: Int, columns: Int) -> [Double] {
    // py4DSTEM's meshgrid arrays are (columns x rows); the kit's grids are rows x columns
    var o = [Double](repeating: 0, count: a.count)
    for r in 0..<rows { for c in 0..<columns { o[r * columns + c] = a[c * rows + r] } }
    return o
}
private func gap(_ a: [Float], _ b: [Double]) -> Double {
    zip(a, b).map { abs(Double($0) - $1) }.max() ?? .infinity
}

@Suite struct CalibrationTests {
    private func cube(_ p: Pin) throws -> MemoryCube {
        try MemoryCube(shape: CubeShape(p.sourceShape), values: p.floats.map { Float($0) })
    }
    private func check(_ e: EditedCube, _ op: Op, tolerance: Double) async throws {
        #expect(e.cube.shape.extents == op.shape, "\(op.name) shape")
        let v = e.cube.values
        #expect(v.count == op.values.count, "\(op.name) count")
        let scale = max(1, op.values.map(abs).max() ?? 1)
        let g = gap(v, op.values)
        print("CALIBRATION \(op.name) max gap \(g) (scale \(scale))")
        if tolerance == 0 { #expect(g == 0, "\(op.name)") } else { #expect(g / scale <= tolerance, "\(op.name) gap \(g)") }
    }
    private func op(_ p: Pin, _ name: String) throws -> Op {
        guard let o = p.ops.first(where: { $0.name == name }) else { throw CuboError.outOfRange(name) }
        return o
    }

    // set_scan_shape, swap_* are index moves: exact
    @Test func reshapeAndSwapsAreExact() async throws {
        let p = try pin(), c = try cube(p)
        try await check(try await CubeEditing.setScanShape(of: c, rows: 6, columns: 4), try op(p, "setScanShape"), tolerance: 0)
        try await check(try await CubeEditing.setScanShape(of: c, rows: 24, columns: 1), try op(p, "setScanShapeFlat"), tolerance: 0)
        try await check(try await CubeEditing.swapScanAndDetector(of: c), try op(p, "swapRQ"), tolerance: 0)
        try await check(try await CubeEditing.swapScanAxes(of: c), try op(p, "swapRxy"), tolerance: 0)
        try await check(try await CubeEditing.swapDetectorAxes(of: c), try op(p, "swapQxy"), tolerance: 0)
        await #expect(throws: CuboError.self) { _ = try await CubeEditing.setScanShape(of: c, rows: 5, columns: 5) }
    }

    // bin_R sums in Float32 in NumPy; the kit sums in Double and rounds once: relative 1e-6 of the largest value.
    @Test func binAndThinScan() async throws {
        let p = try pin(), c = try cube(p)
        for b in [2, 3, 4] {
            let e = try await CubeEditing.binScan(of: c, by: b, calibration: { var k = Calibration(); k.scanPixelSize = 1.5; return k }())
            let o = try op(p, "binR\(b)")
            try await check(e, o, tolerance: 1e-6)
            #expect(e.calibration.scanPixelSize == o.rPixelSize)
            let t = try await CubeEditing.thinScan(of: c, by: b, calibration: { var k = Calibration(); k.scanPixelSize = 1.5; return k }())
            let to = try op(p, "thinR\(b)")
            try await check(t, to, tolerance: 0)
            #expect(t.calibration.scanPixelSize == to.rPixelSize)
        }
        await #expect(throws: CuboError.self) { _ = try await CubeEditing.binScan(of: c, by: 5) }
        await #expect(throws: CuboError.self) { _ = try await CubeEditing.thinScan(of: c, by: 0) }
    }

    @Test func padDetector() async throws {
        let p = try pin(), c = try cube(p)
        try await check(try await CubeEditing.padDetector(of: c, .factor(rows: 2, columns: 2)), try op(p, "padFactor2"), tolerance: 0)
        try await check(try await CubeEditing.padDetector(of: c, .factor(rows: 1.5, columns: 2)), try op(p, "padFactor1p5x2"), tolerance: 0)
        try await check(try await CubeEditing.padDetector(of: c, .outputSize(rows: 12, columns: 14)), try op(p, "padSize"), tolerance: 0)
        await #expect(throws: CuboError.self) { _ = try await CubeEditing.padDetector(of: c, .factor(rows: 0.9, columns: 1)) }
        await #expect(throws: CuboError.self) { _ = try await CubeEditing.padDetector(of: c, .outputSize(rows: 6, columns: 9)) }
        let lead = try CubeEditing.padLeading(rows: 7, columns: 9, specification: .outputSize(rows: 12, columns: 14))
        #expect(lead.rows == 2 && lead.columns == 2)
    }

    // scipy's zoom stores Float32 after a Float64 interpolation: Float32 agreement, relative 1e-6 of the largest value.
    @Test func resampleDetectorBilinear() async throws {
        let p = try pin(), c = try cube(p)
        var k = Calibration(); k.detectorPixelSize = 0.04
        let cases: [(String, CubeEditing.ResampleSpecification, Bool)] = [
            ("resampleHalf", .factor(rows: 0.5, columns: 0.5), false), ("resampleDouble", .factor(rows: 2, columns: 2), false),
            ("resample1p5", .factor(rows: 1.5, columns: 1.5), false), ("resampleAniso", .factor(rows: 0.5, columns: 2), false),
            ("resampleSize", .outputSize(rows: 10, columns: 5), false), ("resampleHalfConserve", .factor(rows: 0.5, columns: 0.5), true),
            ("resampleSizeConserve", .outputSize(rows: 10, columns: 5), true)]
        for (name, spec, conserve) in cases {
            let e = try await CubeEditing.resampleDetector(of: c, spec, conserveSums: conserve, calibration: k)
            let o = try op(p, name)
            try await check(e, o, tolerance: 1e-6)
            #expect(agrees(e.calibration.detectorPixelSize, o.qPixelSize!), "\(name) pixel size")
        }
    }

    @Test func medianFilterMaskedPixels() async throws {
        let p = try pin(), c = try cube(p)
        var mask = [Bool](repeating: false, count: 63)
        for (r, q) in [(0, 0), (3, 4), (3, 5), (6, 8), (0, 4)] { mask[r * 9 + q] = true }
        for k in [3, 5] { try await check(try await CubeEditing.medianFilterMaskedPixels(of: c, mask: mask, kernelWidth: k), try op(p, "medianMasked\(k)"), tolerance: 0) }
        for k in [2, 4] { try await check(try await CubeEditing.medianFilterMaskedPixels(of: c, mask: mask, kernelWidth: k), try op(p, "medianMasked\(k)"), tolerance: 1e-6) }
        await #expect(throws: CuboError.self) { _ = try await CubeEditing.medianFilterMaskedPixels(of: c, mask: [true], kernelWidth: 3) }
    }

    // DataCube.calibrate: axes and grids (the raw grids are rows x columns in py4DSTEM already); the kit's grids are the transpose of py4DSTEM's, equal in value (to 1 ulp of Double).
    @Test func coordinatesAgainstDataCubeCalibrate() throws {
        let p = try pin(), k = p.calibration
        let shape = try CubeShape(p.sourceShape)
        var cal = Calibration()
        cal.detectorPixelSize = k.qPixelSize; cal.detectorUnits = DetectorUnits(rawValue: k.qPixelUnits)!
        cal.scanPixelSize = k.rPixelSize; cal.scanUnits = k.rPixelUnits
        try cal.setOrigin(row: ScanField(scanRows: 4, scanColumns: 6, values: k.qx0), column: ScanField(scanRows: 4, scanColumns: 6, values: k.qy0))
        #expect(agrees(cal.originMean!.row, k.qx0Mean) && agrees(cal.originMean!.column, k.qy0Mean))
        func close(_ a: [Double], _ b: [Double], _ what: String) {
            #expect(a.count == b.count, "\(what)")
            #expect(zip(a, b).allSatisfy { agrees($0, $1) }, "\(what)")
        }
        let qr = cal.detectorRowGrid(shape: shape), qc = cal.detectorColumnGrid(shape: shape)
        close(qr.values, transposed(k.qxx, rows: 7, columns: 9), "qxx")
        close(qc.values, transposed(k.qyy, rows: 7, columns: 9), "qyy")
        close(cal.scanRowGrid(shape: shape).values, transposed(k.rxx, rows: 4, columns: 6), "rxx")
        close(cal.scanColumnGrid(shape: shape).values, transposed(k.ryy, rows: 4, columns: 6), "ryy")
        close(Calibration.rawDetectorRowGrid(shape: shape).values, k.qxxRaw, "qxx_raw")
        close(Calibration.rawDetectorColumnGrid(shape: shape).values, k.qyyRaw, "qyy_raw")
        close(Calibration.rawScanRowGrid(shape: shape).values, k.rxxRaw, "rxx_raw")
        close(Calibration.rawScanColumnGrid(shape: shape).values, k.ryyRaw, "ryy_raw")
        for x in k.qxxs {
            close(try cal.shiftedDetectorRowGrid(shape: shape, scanRow: x.r, scanColumn: x.c).values, transposed(x.values, rows: 7, columns: 9), "qxxs")
            close(try cal.shiftedDetectorColumnGrid(shape: shape, scanRow: x.r, scanColumn: x.c).values, transposed(x.valuesY, rows: 7, columns: 9), "qyys")
        }
        close(try ScanField(scanRows: 4, scanColumns: 6, values: k.qx0).shift.values, k.qx0Shift, "qx0 shift")
        let s = try cal.originShift(row: 2, column: 3)!
        #expect(agrees(s.row, k.qx0Shift[2 * 6 + 3]) && agrees(s.column, k.qy0Shift[2 * 6 + 3]))
        // no origin: the axes start at 0 and the shifted grids are refused
        var bare = Calibration(); bare.detectorPixelSize = 0.1
        close(bare.detectorRowGrid(shape: shape).values, transposed(k.noOriginQxx, rows: 7, columns: 9), "no-origin qxx")
        #expect(throws: CalibrationError.self) { _ = try bare.shiftedDetectorRowGrid(shape: shape, scanRow: 0, scanColumn: 0) }
        #expect(throws: CalibrationError.self) { _ = try cal.originShift(row: 4, column: 0) }
    }

    @Test func settersAndPixelSizeBookkeeping() throws {
        let p = try pin(), k = p.calibration
        var cal = Calibration()
        #expect(cal.detectorPixelSize == 1 && cal.detectorUnits == .pixels && cal.scanUnits == "pixels" && !cal.qrFlip)
        cal.setQRRotationAndFlip(degrees: -12.5, flip: true)
        #expect(agrees(cal.qrRotationDegrees!, k.qrRotationDegrees) && cal.qrFlip == k.qrFlip)
        cal.setQRRotation(degrees: 30)
        #expect(agrees(cal.qrRotationRadians!, k.qrRotation30Radians))
        // bin_Q: the pixel size multiplies and the origin is left alone
        var b = Calibration(); b.detectorPixelSize = 0.05; b.setOrigin(row: 3, column: 4)
        let binned = try b.afterBinningDetector(by: 2)
        #expect(agrees(binned.detectorPixelSize, k.afterBinQ.qPixelSize))
        #expect(binned.originMean!.row == k.afterBinQ.originMean[0] && binned.originMean!.column == k.afterBinQ.originMean[1])
        #expect(throws: CalibrationError.self) { _ = try b.afterBinningDetector(by: 0) }
        // set_probe_param sets the semiangle and the MEAN origin only
        var q = Calibration(); q.setOrigin(row: 1, column: 2)
        q.setProbeParameters(semiangle: 5, centerRow: 10, centerColumn: 11)
        let q0 = try q.origin(row: 0, column: 0)!
        #expect(q.probeSemiangle == 5 && q.originMean!.row == 10 && q0.row == 1)
        // a calibration survives Codable
        var e = Calibration(); e.ellipse = EllipseCalibration(a: k.ellipse[0], b: k.ellipse[1], theta: k.ellipse[2])
        e.probeCenter = (1.5, 2.5); e.measuredOriginMask = [true, false]
        #expect(throws: CalibrationError.self) {
            try e.setOrigin(row: ScanField(scanRows: 1, scanColumns: 2, values: [1, 2]), column: ScanField(constant: 3))
        }
        try e.setOrigin(row: ScanField(scanRows: 1, scanColumns: 2, values: [1, 2]), column: ScanField(scanRows: 1, scanColumns: 2, values: [3, 5]))
        let back = try JSONDecoder().decode(Calibration.self, from: JSONEncoder().encode(e))
        #expect(back == e && back.originMean!.row == 1.5 && back.probeCenter!.column == 2.5)
    }

    // process.calibration.qpixelsize / rotation. leastsq stops at its 1.5e-8 tolerance, the closed form is the exact
    // minimum: relative 1e-7. The rotations and positions are closed-form arithmetic: `agrees` (assert_allclose default).
    @Test func pixelSizeAndRotationProcesses() throws {
        let k = try pin().processes
        let s = try CalibrationMeasurement.detectorPixelSize(measuredPixels: 25, knownDistance: 3.905)
        #expect(agrees(s.size, k.pixelSize) && s.units == k.pixelSizeUnits)
        let f = try CalibrationMeasurement.detectorPixelSize(peakPositions: k.qs, indices: k.hkl.map { ($0[0], $0[1], $0[2]) }, cellSize: k.a)
        #expect(abs(f.size - k.dq) <= 1e-7 * k.dq, "dq \(f.size) against \(k.dq)")
        #expect(f.fittedPositions.count == k.qsFit.count && zip(f.fittedPositions, k.qsFit).allSatisfy { agrees($0, $1) })
        #expect(f.indices.map { [$0.0, $0.1, $0.2] } == k.hklFit)
        for v in k.rotation {
            let q = CalibrationMeasurement.detectorVector(fromScanVector: v.vx, v.vy, rotationDegrees: v.rotation)
            #expect([q.scan.row, q.scan.column, q.detector.row, q.detector.column].enumerated().allSatisfy { agrees($1, v.q[$0]) })
            let r = CalibrationMeasurement.scanVector(fromDetectorVector: v.vx, v.vy, rotationDegrees: v.rotation)
            #expect([r.scan.row, r.scan.column, r.detector.row, r.detector.column].enumerated().allSatisfy { agrees($1, v.r[$0]) })
        }
        #expect(throws: CalibrationError.self) { _ = try CalibrationMeasurement.detectorPixelSize(measuredPixels: 0, knownDistance: 1) }
        #expect(throws: CalibrationError.self) {
            _ = try CalibrationMeasurement.detectorPixelSize(peakPositions: [1, 2], indices: [(0, 0, 0), (0, 0, 0)], cellSize: 4)
        }
    }
}

@Suite struct CalibrationHardeningTests {
    @Test func corruptFieldsAreRefusedOnDecode() {
        for bad in [#"{"values":[]}"#, #"{"scanRows":2,"scanColumns":2,"values":[1]}"#, #"{"scanRows":2,"values":[1,2]}"#] {
            #expect(throws: (any Error).self) { _ = try JSONDecoder().decode(ScanField.self, from: Data(bad.utf8)) }
        }
        #expect(throws: Never.self) { _ = try JSONDecoder().decode(ScanField.self, from: Data(#"{"values":[2]}"#.utf8)) }
    }
    @Test func maskedMedianWithNaNIsNaN() async throws {
        let shape = try CubeShape(scanRows: 1, scanColumns: 1, detectorRows: 3, detectorColumns: 3)
        var v = [Float](repeating: 1, count: 9); v[0] = .nan
        var mask = [Bool](repeating: false, count: 9); mask[4] = true
        let out = try await CubeEditing.medianFilterMaskedPixels(of: MemoryCube(shape: shape, values: v), mask: mask)
        #expect(out.cube.values[4].isNaN)
    }
}
