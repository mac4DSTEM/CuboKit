import Foundation
import Testing
import CuboCore
import CuboBragg

// Pins from py4DSTEM 0.14.19 `BraggVectors.measure_origin` and `fit_origin` (tools/bragg-origin/pins.py) on planted peaks: a
// descan plane, reflections brighter than the direct beam at some positions, an empty position and a chosen peak at row 0.
// The chosen peaks are copied coordinates, so the measurement is exact; the raw map and the fitted surfaces are compared with
// assert_allclose's Float64 default (rtol 1e-7) and atol 1e-12 for values near zero (residuals).

private struct FitPin: Decodable {
    let label: String, fittedRow: [Double], fittedColumn: [Double], residualRow: [Double], residualColumn: [Double]
}
private struct MeasurePin: Decodable {
    let label: String, row: [Double], column: [Double], centerGuess: [Double]?, score: String?, fits: [FitPin]
}
private struct SymmetryPin: Decodable { let peaks: [[[Double]]], guess: [Double], row: [Double], column: [Double], planedGuess: [Double] }
private struct Pin: Decodable {
    let symmetry: SymmetryPin
    let scan: [Int], detector: [Int], peaks: [[[Double]]], histogram: [Double], maxGuess: [Double], measures: [MeasurePin]
}
private let pin: Pin = {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/braggorigin.json")
    let decoder = JSONDecoder()
    decoder.nonConformingFloatDecodingStrategy = .convertFromString(positiveInfinity: "Infinity", negativeInfinity: "-Infinity", nan: "NaN")
    return try! decoder.decode(Pin.self, from: Data(contentsOf: url))
}()

private let vectors = BraggVectors(scanRows: pin.scan[0], scanColumns: pin.scan[1],
                                   peaks: pin.peaks.map { $0.map { BraggPeak(row: $0[0], column: $0[1], intensity: $0[2]) } })

/// The largest |got − want| / (atol + rtol·|want|); at most 1 passes.
private func worst(_ got: [Double], _ want: [Double], rtol: Double = 1e-7, atol: Double = 1e-12) -> Double {
    guard got.count == want.count else { return .infinity }
    return zip(got, want).reduce(0.0) { max($0, abs($1.0 - $1.1) / (atol + rtol * abs($1.1))) }
}

private func score(_ name: String?) -> BraggOriginScore? {
    switch name {
    case "distance": .distance
    case "intensity": .intensity
    case "intensity weighted distance": .intensityWeightedDistance
    default: nil
    }
}

private func model(_ label: String) -> (OriginFitModel, Bool) {
    switch label {
    case "parabola": (.parabola, false)
    case "plane-robust": (.plane, true)
    default: (.plane, false)
    }
}

@Suite struct BraggOriginTests {
    @Test func rawHistogramMatchesPy4DSTEM() {
        let map = BraggOrigin.rawHistogram(vectors, rows: pin.detector[0], columns: pin.detector[1])
        let w = worst(map, pin.histogram)
        print("raw Bragg vector map: worst gap / bound = \(w)")
        #expect(w <= 1)
    }

    @Test func measureAndFitMatchPy4DSTEM() throws {
        for c in pin.measures {
            let guess = c.centerGuess.map { BraggDetectorCenter(row: $0[0], column: $0[1]) }
            let m = try BraggOrigin.measure(vectors, detectorRows: pin.detector[0], detectorColumns: pin.detector[1],
                                            centerGuess: guess, score: score(c.score))
            if guess == nil {
                #expect(m.guess == BraggDetectorCenter(row: pin.maxGuess[0], column: pin.maxGuess[1]), "\(c.label)")
            }
            #expect(m.row == c.row, "\(c.label) row")
            #expect(m.column == c.column, "\(c.label) column")
            #expect(m.found.filter { !$0 }.count == 1, "\(c.label): one empty position")
            for f in c.fits {
                let (fitModel, robust) = model(f.label)
                let fit = try BraggOrigin.fit(m, model: fitModel, robust: robust)
                let gaps = [worst(fit.fittedRow, f.fittedRow), worst(fit.fittedColumn, f.fittedColumn),
                            worst(fit.residualRow, f.residualRow), worst(fit.residualColumn, f.residualColumn)]
                print("Bragg origin \(c.label) / \(f.label): worst gap / bound = \(gaps.max()!)")
                #expect(gaps.max()! <= 1, "\(c.label) / \(f.label)")
            }
        }
    }

    @Test func positionsWithoutPeaksOrAtRowZeroStayOutOfTheFit() throws {
        let m = try BraggOrigin.measure(vectors, detectorRows: pin.detector[0], detectorColumns: pin.detector[1])
        let fit = try BraggOrigin.fit(m)
        #expect(!fit.used[13] && !m.found[13])
        #expect(m.row[40] == 0 && !fit.used[40])
        #expect(fit.used.filter { $0 }.count == m.row.count - 2)
    }

    @Test func refusesBadInput() {
        let bad = BraggVectors(scanRows: 1, scanColumns: 1, peaks: [[BraggPeak(row: .nan, column: 1, intensity: 1)]])
        #expect(throws: CuboBraggError.self) { try BraggOrigin.measure(bad, detectorRows: 8, detectorColumns: 8) }
        #expect(throws: CuboBraggError.self) { try BraggOrigin.measure(vectors, detectorRows: 0, detectorColumns: 8) }
        let empty = BraggVectors(scanRows: 2, scanColumns: 2, peaks: [[], [], [], []])
        #expect(throws: OriginFitError.self) {
            try BraggOrigin.fit(try BraggOrigin.measure(empty, detectorRows: 8, detectorColumns: 8))
        }
    }

    @Test func symmetryGuessFindsTheBeamUnderABrighterReflection() throws {
        let bright = BraggVectors(scanRows: pin.scan[0], scanColumns: pin.scan[1],
                                  peaks: pin.symmetry.peaks.map { $0.map { BraggPeak(row: $0[0], column: $0[1], intensity: $0[2]) } })
        let guess = try #require(try BraggOrigin.symmetryCenterGuess(bright, detectorRows: pin.detector[0],
                                                                     detectorColumns: pin.detector[1]))
        #expect(guess == BraggDetectorCenter(row: pin.symmetry.guess[0], column: pin.symmetry.guess[1]))
        let m = try BraggOrigin.measure(bright, detectorRows: pin.detector[0], detectorColumns: pin.detector[1],
                                        centerGuess: guess, score: .distance)
        #expect(m.row == pin.symmetry.row && m.column == pin.symmetry.column)
        // py4DSTEM's default picks the brighter reflection at every position here.
        let byIntensity = try BraggOrigin.measure(bright, detectorRows: pin.detector[0], detectorColumns: pin.detector[1])
        #expect(zip(byIntensity.row, m.row).allSatisfy { abs($0 - $1) > 5 })
        let planted = try #require(try BraggOrigin.symmetryCenterGuess(vectors, detectorRows: pin.detector[0],
                                                                       detectorColumns: pin.detector[1]))
        #expect(planted == BraggDetectorCenter(row: pin.symmetry.planedGuess[0], column: pin.symmetry.planedGuess[1]))
        let none = BraggVectors(scanRows: 1, scanColumns: 2, peaks: [[], []])
        #expect(try BraggOrigin.symmetryCenterGuess(none, detectorRows: 8, detectorColumns: 8) == nil)
    }

    @Test func originFitMaskLeavesPositionsOut() throws {
        // A plane with one wild value: masked out, the fit is the plane exactly.
        var values = (0..<12).map { k in 2.0 + 0.5 * Double(k / 4) - 0.25 * Double(k % 4) }
        let clean = values
        values[5] = 100
        var mask = [Bool](repeating: true, count: 12)
        mask[5] = false
        let fit = try OriginFit.fit(values, rows: 3, columns: 4, model: .plane, mask: mask)
        #expect(worst(fit, clean, rtol: 1e-12, atol: 1e-12) <= 1)
        #expect(throws: OriginFitError.self) { try OriginFit.fit(values, rows: 3, columns: 4, model: .plane, mask: [true]) }
    }
}
