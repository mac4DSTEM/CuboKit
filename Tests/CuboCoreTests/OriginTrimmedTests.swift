import Foundation
import Testing
import CuboCore

// OriginFit.fitTrimmed against (a) a Float32 NumPy transcription of the app's fitOriginTrimmed / fit2D and (b) py4DSTEM's masked
// fit_origin on the mask the transcription ends with (tools/origin/trimmed_pins.py). The mask, the share excluded and the
// shape are exact; the surfaces agree with the transcription within 2e-6 of max(1, max|surface|) (the app rounds its surfaces
// and residuals to Float32, whose half ulp at 40 is 1.9e-6 absolute), the residual RMS within 1e-5 absolute (measured 2.6e-6; Float32 residuals
// quantised at 3.8e-6), and with py4DSTEM's Double fit within 5e-8 (measured 1.3e-8: py4DSTEM reaches the least-squares solution
// with curve_fit, within that solver's convergence; the kit solves it by QR).

private struct Surfaces: Decodable { let fittedRow: [Double], fittedColumn: [Double] }
private struct App: Decodable {
    let fittedRow: [Double], fittedColumn: [Double], kept: [Bool], keptResidual: Double, fullScanResidual: Double
}
private struct Case: Decodable {
    let name: String, model: String, trimSigma: Double, rounds: Int, originRow: [Double], originColumn: [Double]
    let app: App, py4dstem: Surfaces?
}
private struct Pin: Decodable { let cases: [Case], shapes: [String: [Int]] }
private func pin() throws -> Pin {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/origin-trimmed.json")
    return try JSONDecoder().decode(Pin.self, from: Data(contentsOf: url))
}
private func gap(_ a: [Double], _ b: [Double]) -> Double {
    guard a.count == b.count else { return .infinity }
    return zip(a, b).map { abs($0 - $1) }.max()! / max(1, b.map(abs).max()!)
}
private func model(_ name: String) -> OriginFitModel {
    switch name { case "plane": .plane; case "parabola": .parabola; case "bezierTwo": .bezierTwo; default: .constant }
}

@Suite struct OriginTrimmedTests {
    @Test func trimmedFitAgainstTheAppAndPy4DSTEM() throws {
        let p = try pin()
        #expect(p.cases.count >= 14)
        for c in p.cases {
            let shape = p.shapes[c.name]!
            let r = try OriginFit.fitTrimmed(originRow: c.originRow, originColumn: c.originColumn, rows: shape[0], columns: shape[1],
                                             model: model(c.model), trimSigma: c.trimSigma, rounds: c.rounds)
            #expect(r.kept == c.app.kept, "\(c.name): mask")
            let g = max(gap(r.fittedRow, c.app.fittedRow), gap(r.fittedColumn, c.app.fittedColumn))
            let keptGap = abs(r.keptResidual - c.app.keptResidual), fullGap = abs(r.fullScanResidual - c.app.fullScanResidual)
            var line = "TRIMMED \(c.name) kept \(r.kept.filter { $0 }.count)/\(r.kept.count) surfaces \(g) RMS \(keptGap) \(fullGap)"
            #expect(g < 2e-6, "\(c.name): surfaces")
            #expect(keptGap < 1e-5 && fullGap < 1e-5 * max(1, c.app.fullScanResidual), "\(c.name): residuals")
            #expect(r.excludedFraction == 1 - Double(c.app.kept.filter { $0 }.count) / Double(c.app.kept.count))
            if let q = c.py4dstem {
                let pg = max(gap(r.fittedRow, q.fittedRow), gap(r.fittedColumn, q.fittedColumn))
                line += " py4DSTEM \(pg)"
                #expect(pg < 5e-8, "\(c.name): against py4DSTEM's masked fit")
            }
            print(line)
        }
    }

    // Planted truth, independent of both references: a plane with scattered failures is recovered exactly from the kept positions, and
    // the plain least-squares plane is not.
    @Test func plantedPlaneIsRecoveredAndPlainFitIsNot() throws {
        let rows = 14, columns = 11
        func truth(_ r: Int, _ c: Int) -> (Double, Double) { (20 + 0.7*Double(r) - 0.3*Double(c), 33 - 0.2*Double(r) + 0.5*Double(c)) }
        var a = [Double](), b = [Double]()
        for r in 0..<rows { for c in 0..<columns { let t = truth(r, c); a.append(t.0); b.append(t.1) } }
        let bad = [5, 19, 33, 47, 61, 75, 89, 103, 117, 131, 145]                       // 11 of 154, scattered
        for (i, k) in bad.enumerated() { a[k] += (i % 2 == 0 ? 25 : -18); b[k] += (i % 3 == 0 ? -22 : 15) }
        let r = try OriginFit.fitTrimmed(originRow: a, originColumn: b, rows: rows, columns: columns, model: .plane)
        #expect(r.kept.enumerated().allSatisfy { $0.element == !bad.contains($0.offset) })
        for rr in 0..<rows { for cc in 0..<columns {
            let t = truth(rr, cc), k = rr*columns + cc
            #expect(abs(r.fittedRow[k] - t.0) < 1e-10 && abs(r.fittedColumn[k] - t.1) < 1e-10)
        } }
        #expect(r.keptResidual < 1e-10 && r.fullScanResidual > 5)
        let plain = try OriginFit.fit(a, rows: rows, columns: columns, model: .plane)
        #expect(zip(plain, (0..<a.count).map { truth($0 / columns, $0 % columns).0 }).map { abs($0 - $1) }.max()! > 1)
    }

    @Test func nonFiniteMeasurementsAreNeverKeptAndDoNotChangeTheOthers() throws {
        let p = try pin()
        let c = p.cases.first { $0.name == "planeScattered15" }!
        let shape = p.shapes[c.name]!
        let clean = try OriginFit.fitTrimmed(originRow: c.originRow, originColumn: c.originColumn, rows: shape[0], columns: shape[1], model: .plane)
        var a = c.originRow, b = c.originColumn
        let dropped = clean.kept.firstIndex(of: false)!, other = clean.kept.firstIndex(of: true)!
        a[dropped] = .nan; b[other] = .infinity
        let r = try OriginFit.fitTrimmed(originRow: a, originColumn: b, rows: shape[0], columns: shape[1], model: .plane)
        #expect(!r.kept[dropped] && !r.kept[other])
        #expect(r.fittedRow.allSatisfy(\.isFinite) && r.fittedColumn.allSatisfy(\.isFinite) && r.fullScanResidual.isFinite)
        #expect(throws: OriginFitError.self) {
            try OriginFit.fitTrimmed(originRow: [.nan, .nan, .nan, .nan], originColumn: [1, 2, 3, 4], rows: 2, columns: 2, model: .plane)
        }
    }

    @Test func degenerateAndInvalidInputs() throws {
        // empty / mismatched / bad parameters throw; the app returned empty arrays for some of these
        for bad in [(try? OriginFit.fitTrimmed(originRow: [], originColumn: [], rows: 0, columns: 0, model: .plane)),
                    (try? OriginFit.fitTrimmed(originRow: [1, 2, 3, 4], originColumn: [1, 2, 3], rows: 2, columns: 2, model: .plane)),
                    (try? OriginFit.fitTrimmed(originRow: [1, 2, 3, 4], originColumn: [1, 2, 3, 4], rows: 2, columns: 2, model: .plane, trimSigma: 0)),
                    (try? OriginFit.fitTrimmed(originRow: [1, 2, 3, 4], originColumn: [1, 2, 3, 4], rows: 2, columns: 2, model: .plane, madScale: .nan)),
                    (try? OriginFit.fitTrimmed(originRow: [1, 2, 3, 4], originColumn: [1, 2, 3, 4], rows: Int.max, columns: 2, model: .plane))] {
            #expect(bad == nil)
        }
        // a single position is itself; its fit is exact and nothing is excluded
        let one = try OriginFit.fitTrimmed(originRow: [3], originColumn: [4], rows: 1, columns: 1, model: .plane)
        #expect(one.fittedRow == [3] && one.fittedColumn == [4] && one.kept == [true] && one.excludedFraction == 0 && one.keptResidual == 0)
    }
}
