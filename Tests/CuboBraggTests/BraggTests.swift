import CuboCore
import CuboVirtual
import Foundation
import Testing
@testable import CuboBragg

/// Pins from tools/bragg/pins.py. Two references: py4DSTEM 0.14.19 `find_Bragg_disks` (the primary behaviour) and the
/// mac4DSTEM source at the commit named in Fixtures/bragg.json, compiled unchanged and run in Float32 (the named options).
struct Fixture {
    let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")
    let meta: [String: Any]
    let blob: [Double]

    init() throws {
        meta = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("bragg.json"))) as! [String: Any]
        let data = try Data(contentsOf: directory.appendingPathComponent("bragg.f64"))
        blob = data.withUnsafeBytes { raw in
            stride(from: 0, to: raw.count, by: 8).map { Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: $0, as: UInt64.self))) }
        }
    }

    func array(_ name: String) -> [Double] {
        let entry = (meta["arrays"] as! [String: Any])[name] as! [String: Any]
        let offset = entry["offset"] as! Int, shape = entry["shape"] as! [Int]
        return Array(blob[offset..<(offset + shape.reduce(1, *))])
    }
    func scene(_ name: String) -> (rows: Int, columns: Int, kernel: DetectorImage, patterns: [DetectorImage]) {
        let s = (meta["scenes"] as! [String: Any])[name] as! [String: Any]
        let rows = s["rows"] as! Int, columns = s["columns"] as! Int
        let all = array(s["patterns"] as! String), n = rows * columns
        let patterns = (0..<(all.count / n)).map {
            DetectorImage(detectorRows: rows, detectorColumns: columns, values: Array(all[($0 * n)..<(($0 + 1) * n)]), path: .cpu)
        }
        return (rows, columns, DetectorImage(detectorRows: rows, detectorColumns: columns, values: array(s["kernel"] as! String), path: .cpu), patterns)
    }
    var cases: [[String: Any]] { meta["cases"] as! [[String: Any]] }
}

/// py4DSTEM's find_Bragg_disks keywords, as the generator recorded them, as kit options.
func options(_ o: [String: Any]) -> BraggDiskOptions {
    func d(_ k: String) -> Double { (o[k] as! NSNumber).doubleValue }
    var b = BraggDiskOptions()
    b.correlationPower = d("corrPower"); b.patternSigma = d("sigma_dp"); b.correlationSigma = d("sigma_cc")
    b.subpixel = [BraggSubpixel.pixel, .parabolic, .multicorr][Int(d("subpixel"))]
    b.upsampleFactor = Int(d("upsample_factor"))
    b.minimumAbsoluteIntensity = d("minAbsoluteIntensity"); b.minimumRelativeIntensity = d("minRelativeIntensity")
    b.relativeToPeak = Int(d("relativeToPeak")); b.minimumPeakSpacing = d("minPeakSpacing")
    b.edgeBoundary = Int(d("edgeBoundary")); b.maximumPeaks = Int(d("maxNumPeaks"))
    b.relativeReferenceMinimumRadius = d("relativeReferenceMinimumRadiusPx")
    return b
}

func peaks(_ a: [Double]) -> [BraggPeak] {
    stride(from: 0, to: a.count, by: 3).map { BraggPeak(row: a[$0], column: a[$0 + 1], intensity: a[$0 + 2]) }
}

/// The largest position gap (pixels) and relative intensity gap between two peak lists of equal length, in order.
func gaps(_ a: [BraggPeak], _ b: [BraggPeak]) -> (position: Double, intensity: Double) {
    var position = 0.0, intensity = 0.0
    for (p, q) in zip(a, b) {
        position = max(position, abs(p.row - q.row), abs(p.column - q.column))
        intensity = max(intensity, abs(p.intensity - q.intensity) / max(abs(q.intensity), 1e-300))
    }
    return (position, intensity)
}

@Suite struct BraggDiskTests {
    let fixture: Fixture
    init() throws { fixture = try Fixture() }

    /// Every py4DSTEM case: same peaks, same order, to the registered tolerance.
    @Test func matchesPy4DSTEM() throws {
        var worstPosition = 0.0, worstIntensity = 0.0, checked = 0
        for c in fixture.cases where (c["reference"] as! String) == "py4dstem" {
            let id = c["id"] as! String
            let s = fixture.scene(c["scene"] as! String)
            let pattern = s.patterns[c["pattern"] as! Int]
            let expected = peaks(fixture.array(c["expected"] as! String))
            let actual = try BraggDisks.detect(pattern, kernel: s.kernel, options: options(c["options"] as! [String: Any]))
            #expect(actual.count == expected.count, "\(id): \(actual.count) peaks, py4DSTEM \(expected.count)")
            guard actual.count == expected.count else { continue }
            let g = gaps(actual, expected)
            #expect(g.position <= 1e-9 && g.intensity <= 1e-10, "\(id): position gap \(g.position), intensity gap \(g.intensity)")
            worstPosition = max(worstPosition, g.position); worstIntensity = max(worstIntensity, g.intensity); checked += 1
        }
        print("CuboBragg py4DSTEM pins: \(checked) cases, maximum position gap \(worstPosition) px (tolerance 1e-9), maximum relative intensity gap \(worstIntensity) (tolerance 1e-10)")
        #expect(checked == 30)
    }

    /// The pins must not be vacuous: a few cases have to hold several peaks, subpixel offsets and each filter has to bite.
    @Test func pinsAreInformative() throws {
        func count(_ id: String) -> Int { fixture.array("e_" + id).count / 3 }
        #expect(count("s1p0_poly") >= 5)
        #expect(count("s1p0_poly") > count("s1p0_edge30"))
        #expect(count("s1p0_maxPeaks3") == 3)
        #expect(count("s1p2_spacing_off") > count("s1p2_spacing8"))
        #expect(count("s1p2_edge0") > count("s1p2_edge1") - 1)
        #expect(count("s1p1_absolute") < count("s1p1_poly"))
        #expect(count("s3p2_nonfinite") == 0)
        #expect(fixture.array("e_s1p0_pixel") != fixture.array("e_s1p0_poly"))
        #expect(fixture.array("e_s1p0_poly") != fixture.array("e_s1p0_multicorr"))
    }

    /// The mac4DSTEM source, Float32: the named options reproduce it, to Float32's resolution.
    @Test func matchesMac4DSTEMVariants() throws {
        var worstPosition = 0.0, worstIntensity = 0.0, checked = 0
        for c in fixture.cases where (c["reference"] as! String) == "app" {
            let id = c["id"] as! String
            let s = fixture.scene(c["scene"] as! String)
            let pattern = s.patterns[c["pattern"] as! Int]
            var o = options(c["options"] as! [String: Any])
            o.nonFinitePixels = .neighbourMedian
            o.multicorrOrder = .correlationIntensity
            let expected = peaks(fixture.array(c["expected"] as! String))
            let actual = try BraggDisks.detect(pattern, kernel: s.kernel, options: o)
            #expect(actual.count == expected.count, "\(id): \(actual.count) peaks, app \(expected.count)")
            guard actual.count == expected.count else { continue }
            let g = gaps(actual, expected)
            #expect(g.position <= 1e-3 && g.intensity <= 1e-5, "\(id): position gap \(g.position), intensity gap \(g.intensity)")
            worstPosition = max(worstPosition, g.position); worstIntensity = max(worstIntensity, g.intensity); checked += 1
        }
        print("CuboBragg mac4DSTEM pins: \(checked) cases, maximum position gap \(worstPosition) px (tolerance 1e-3), maximum relative intensity gap \(worstIntensity) (tolerance 1e-5)")
        #expect(checked == 10)
    }

    /// Each variant is an option, not a replacement: with it off the kit gives py4DSTEM's different answer on the same case.
    @Test func variantsAreOptionsNotDefaults() throws {
        func run(_ id: String, _ edit: (inout BraggDiskOptions) -> Void = { _ in }) throws -> [BraggPeak] {
            let c = fixture.cases.first { ($0["id"] as! String) == id }!
            let s = fixture.scene(c["scene"] as! String)
            var o = options(c["options"] as! [String: Any]); edit(&o)
            return try BraggDisks.detect(s.patterns[c["pattern"] as! Int], kernel: s.kernel, options: o)
        }
        // relativeReferenceMinimumRadius: the brightest maximum sets the scale unless the radius excludes it.
        let near = try run("app_s3p0_refradius10") { $0.relativeReferenceMinimumRadius = 0 }
        let far = try run("app_s3p0_refradius10")
        #expect(near.count == 1 && far.count == 3)
        // NaN / infinity: py4DSTEM finds nothing, the median fill finds the disks.
        #expect(try run("app_s3p2_nonfinite_fill").isEmpty)
        #expect(try run("app_s3p2_nonfinite_fill") { $0.nonFinitePixels = .neighbourMedian }.count == fixture.array("e_app_s3p2_nonfinite_fill").count / 3)
        #expect(fixture.array("e_app_s3p2_nonfinite_fill").count / 3 >= 3)
        // The order of multicorr peaks: sorted by refined intensity, or left in integer-maximum order.
        let sorted = try run("app_s3p1_multicorr")
        let unsorted = try run("app_s3p1_multicorr") { $0.multicorrOrder = .correlationIntensity }
        #expect(sorted.map(\.intensity) == sorted.map(\.intensity).sorted(by: >))
        #expect(unsorted.map(\.intensity) != unsorted.map(\.intensity).sorted(by: >))
        #expect(Set(sorted.map(\.row)) == Set(unsorted.map(\.row)))
    }

    @Test func detectorAdaptedMatchesMac4DSTEM() throws {
        for a in fixture.meta["adapted"] as! [[String: Any]] {
            let i = a["input"] as! [Double], e = a["expected"] as! [Double]
            let o = BraggDiskOptions.detectorAdapted(detectorRows: Int(i[0]), detectorColumns: Int(i[1]), probeRadius: i[2] < 0 ? nil : i[2])
            #expect(o.minimumPeakSpacing == e[0] && Double(o.edgeBoundary) == e[1], "\(i): got \(o.minimumPeakSpacing), \(o.edgeBoundary), app \(e)")
        }
    }

    @Test func probeRingHintMatchesMac4DSTEM() throws {
        var seenHint = false, seenNone = false
        for h in fixture.meta["hints"] as! [[String: Any]] {
            let name = h["name"] as! String
            let image = DetectorImage(detectorRows: 64, detectorColumns: 64, values: fixture.array(h["pattern"] as! String), path: .cpu)
            let expected = h["expected"] as! [Double]
            let actual = ProbeRingHint.outerEdge(of: image)
            if expected.isEmpty {
                #expect(actual == nil, "\(name)")
                seenNone = true
            } else {
                let hint = try #require(actual, "\(name)")
                #expect(Double(hint.outerRadius) == expected[0], "\(name)")
                #expect(abs(hint.dip - expected[1]) <= 1e-5, "\(name): dip \(hint.dip) vs \(expected[1])")
                #expect(hint.isWorthOffering(currentRadius: 5.4) && !hint.isWorthOffering(currentRadius: 8.3))
                seenHint = true
            }
        }
        #expect(seenHint && seenNone)
    }

    @Test func correlationDefinition() throws {
        let s = fixture.scene("S1")
        let detector = try BraggDetector(kernel: s.kernel)
        // Cross-correlation of a pattern with a delta kernel is the pattern itself (positive part).
        var delta = [Double](repeating: 0, count: 64 * 64); delta[0] = 1
        let identity = try BraggDetector(kernel: DetectorImage(detectorRows: 64, detectorColumns: 64, values: delta, path: .cpu))
        let c = try identity.correlate(s.patterns[0])
        let gap = zip(c.real, s.patterns[0].values).map { abs($0 - max($1, 0)) }.max()!
        #expect(gap < 1e-9, "delta-kernel correlation gap \(gap)")
        // Phase correlation of an all-zero pattern: py4DSTEM's 0**0 = 1 in every bin.
        let zero = DetectorImage(detectorRows: 64, detectorColumns: 64, values: [Double](repeating: 0, count: 64 * 64), path: .cpu)
        var power0 = BraggDiskOptions(); power0.correlationPower = 0
        let p = try detector.correlate(zero, options: power0)
        #expect(p.fourierRe.allSatisfy { $0 == 1 } && p.fourierIm.allSatisfy { $0 == 0 })
        #expect(try detector.detect(zero, options: power0).isEmpty)
    }

    @Test func refusals() throws {
        let s = fixture.scene("S1")
        var o = BraggDiskOptions()
        o.upsampleFactor = 2
        #expect(throws: CuboBraggError.self) { try BraggDisks.detect(s.patterns[0], kernel: s.kernel, options: o) }
        o = BraggDiskOptions(); o.correlationSigma = -1
        #expect(throws: CuboBraggError.self) { try BraggDisks.detect(s.patterns[0], kernel: s.kernel, options: o) }
        o = BraggDiskOptions(); o.correlationPower = .nan
        #expect(throws: CuboBraggError.self) { try BraggDisks.detect(s.patterns[0], kernel: s.kernel, options: o) }
        let other = fixture.scene("S2")
        #expect(throws: CuboBraggError.self) { try BraggDisks.detect(other.patterns[0], kernel: s.kernel) }
        // upsampleFactor is irrelevant (and unchecked) unless multicorr is asked for, as in py4DSTEM.
        o = BraggDiskOptions(); o.subpixel = .parabolic; o.upsampleFactor = 1
        _ = try BraggDisks.detect(s.patterns[0], kernel: s.kernel, options: o)
    }

    /// The whole-cube pass is the per-pattern pass, position by position.
    @Test func wholeCubeEqualsPerPattern() async throws {
        let s = fixture.scene("S1")
        let shape = try CubeShape(scanRows: 3, scanColumns: 1, detectorRows: 64, detectorColumns: 64)
        let floats = s.patterns.flatMap { $0.values.map { Float($0) } }
        let cube = try MemoryCube(shape: shape, values: floats)
        var o = BraggDiskOptions(); o.minimumPeakSpacing = 8; o.edgeBoundary = 5
        let all = try await BraggDisks.detectAll(in: cube, kernel: s.kernel, options: o, tiling: .rows(2))
        #expect(all.scanRows == 3 && all.scanColumns == 1)
        for (k, pattern) in s.patterns.enumerated() {
            let widened = DetectorImage(detectorRows: 64, detectorColumns: 64, values: pattern.values.map { Double(Float($0)) }, path: .cpu)
            #expect(all[k, 0] == (try BraggDisks.detect(widened, kernel: s.kernel, options: o)))
        }
        #expect(all.peaks.map(\.count).reduce(0, +) > 10)
        let wrong = DetectorImage(detectorRows: 40, detectorColumns: 56, values: fixture.scene("S2").kernel.values, path: .cpu)
        await #expect(throws: CuboBraggError.self) { _ = try await BraggDisks.detectAll(in: cube, kernel: wrong) }
    }
}
