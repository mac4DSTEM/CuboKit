import Foundation
import Testing
import CuboBragg

// Pins from py4DSTEM 0.14.19 `BraggVectors.get_virtual_image` with all calibrations off (tools/classification/pins.py). The membership
// test is a comparison of hypot distances with the radii (exact) and the sums follow NumPy's order, so the images are compared with
// np.allclose defaults (rtol 1e-5, atol 1e-8); the measured gap is printed.

private struct CasePin: Decodable { let label: String, mode: String?, geometry: Geometry?, image: [Double] }
/// `geometry` is `[centre, radius or range]`, each possibly empty; decoded loosely.
private struct Geometry: Decodable {
    let centre: [Double]
    let range: [Double]
    let scalar: Double?
    init(from decoder: Decoder) throws {
        var c = try decoder.unkeyedContainer()
        centre = try c.decode([Double].self)
        if let s = try? c.decode(Double.self) { scalar = s; range = [] } else { scalar = nil; range = try c.decode([Double].self) }
    }
}
private struct Pin: Decodable { let scan: [Int], peaks: [[[Double]]], cases: [CasePin] }
private let pin: Pin = {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/virtualimage.json")
    let decoder = JSONDecoder()
    decoder.nonConformingFloatDecodingStrategy = .convertFromString(positiveInfinity: "Infinity", negativeInfinity: "-Infinity", nan: "NaN")
    return try! decoder.decode(Pin.self, from: Data(contentsOf: url))
}()

private func detector(_ c: CasePin) -> BraggVirtualDetector {
    // py4DSTEM's rules: mode None uses geometry[1] as a radius (a number), a range, or nothing; no geometry means (0, inf).
    guard let g = c.geometry else { return .annulus(center: nil, inner: 0, outer: .infinity) }
    let centre = g.centre.count == 2 ? BraggDetectorCenter(row: g.centre[0], column: g.centre[1]) : nil
    switch c.mode {
    case "circle", "circular":
        return .circle(center: centre, radius: g.scalar!)
    case "annulus", "annular":
        return .annulus(center: centre, inner: g.range[0], outer: g.range[1])
    default:
        if let s = g.scalar { return .annulus(center: centre, inner: 0, outer: s) }
        if g.range.isEmpty { return .allPeaks }
        return .annulus(center: centre, inner: g.range[0], outer: g.range[1])
    }
}

@Suite struct BraggVirtualImageTests {
    @Test func virtualImagesMatchPy4DSTEM() {
        let vectors = BraggVectors(scanRows: pin.scan[0], scanColumns: pin.scan[1],
                                   peaks: pin.peaks.map { $0.map { BraggPeak(row: $0[0], column: $0[1], intensity: $0[2]) } })
        for c in pin.cases {
            let image = vectors.virtualImage(detector: detector(c))
            #expect(image.values.count == c.image.count, "\(c.label)")
            var worst = 0.0
            for (g, w) in zip(image.values, c.image) {
                if g.isNaN && w.isNaN { continue }
                if g.isNaN != w.isNaN { worst = .infinity; continue }
                worst = max(worst, abs(g - w) / (1e-8 + 1e-5 * abs(w)))
            }
            print("virtual image \(c.label): worst gap / allclose bound = \(worst)")
            #expect(worst <= 1, "\(c.label)")
        }
        #expect(pin.cases.contains { $0.image.contains { $0.isNaN } }, "the pin holds a NaN intensity")
        #expect(pin.cases.contains { $0.image.contains(0) && $0.image.contains { $0 > 0 } })
    }

    @Test func aPlantedDiskIsFoundInItsAnnulus() {
        // One bright peak at radius 10 from the centre (32, 32) at one position, a dim one at radius 25 at another.
        let rows = 3, columns = 4
        var peaks = [[BraggPeak]](repeating: [], count: rows * columns)
        peaks[5] = [BraggPeak(row: 32, column: 42, intensity: 7), BraggPeak(row: 32, column: 32, intensity: 100)]
        peaks[6] = [BraggPeak(row: 7, column: 32, intensity: 2)]
        let v = BraggVectors(scanRows: rows, scanColumns: columns, peaks: peaks)
        let centre = BraggDetectorCenter(row: 32, column: 32)
        let ring = v.virtualImage(detector: .annulus(center: centre, inner: 5, outer: 15))
        #expect(ring.values.enumerated().allSatisfy { $0.offset == 5 ? $0.element == 7 : $0.element == 0 })
        let bright = v.virtualImage(detector: .circle(center: centre, radius: 1))
        #expect(bright[1, 1] == 100 && bright.values.reduce(0, +) == 100)
        let outer = v.virtualImage(detector: .annulus(center: centre, inner: 20, outer: 30))
        #expect(outer[1, 2] == 2 && outer.values.reduce(0, +) == 2)
        #expect(v.virtualImage(detector: .allPeaks)[1, 1] == 107)
    }
}
