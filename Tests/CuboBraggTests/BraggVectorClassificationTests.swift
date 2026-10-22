import Foundation
import Testing
import CuboCore
import CuboBragg

// Pins from py4DSTEM 0.14.19 (BraggVectorClassification, classutils.get_class_DP) on a planted 12 x 14 scan of three phases and
// nine reference points (tools/classification/pins.py). Labels, class sets and the 0/1 matrix W are compared exactly; every other
// number with np.allclose defaults (rtol 1e-5, atol 1e-8), the default for these functions upstream, the worst measured ratio of
// the gap to that bound printed for each group.

private struct LabelPin: Decodable { let maxDist: Double?, labels: [[Int]], xBoolean: [Double], xIntensity: [Double] }
private struct InitialPin: Decodable { struct Args: Decodable { let thresh: Double?, BP_fraction_thresh: Double?, max_iterations: Int?, n_corr_init: Int? }
    let args: Args, classes: [[Int]] }
private struct StatePin: Decodable { let classCount: Int, w: [Double], h: [Double] }
private struct CasePin: Decodable { struct Args: Decodable { let thresh: Double?, n_corr_init: Int?, booleanData: Bool? }
    let name: String, args: Args?, images: [[Double]]?, state: StatePin }
private struct OpPin: Decodable {
    let op: String, maxIterations: Int?, sigma: Double?, thresholdSplit: Double?, expandMask: Int?, minimumPixels: Int?
    let thresholdPeaks: Double?, thresholdScan: Double?, index: Int?, i: Int?, j: Int?
}
private struct ResultPin: Decodable { let op: OpPin, candidate: StatePin?, accepted: StatePin?, error: String? }
private struct OpsPin: Decodable { let name: String, classCount: Int, w: [Double], h: [Double], results: [ResultPin] }
private struct DPCase: Decodable {
    struct Args: Decodable { let threshold: Double?, dark: Bool?, shifts: StringOrBool?, integerShifts: Bool?, radius: Double?, origin: [Double]? }
    let args: Args, dp: [Double]
}
/// `shifts` is `true` or the string "halves".
private struct StringOrBool: Decodable {
    let value: String
    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let b = try? c.decode(Bool.self) { value = b ? "true" : "false" } else { value = try c.decode(String.self) }
    }
}
private struct ClassPatternPin: Decodable {
    let patterns: [Double], detector: [Int], image: [Double], dark: [Double], rowShifts: [Double], columnShifts: [Double]
    let rowHalves: [Double], columnHalves: [Double], cases: [DPCase]
}
private struct MaskedPin: Decodable { let peaks: [[[Double]]], origin: [Double], cases: [DPCase] }
private struct Pins: Decodable {
    let scan: [Int], peaks: [[[Double]]], Qx: [Double], Qy: [Double], truth: [String: [Double]]
    let labels: [LabelPin], initialClasses: [InitialPin], cases: [CasePin], states: [OpsPin]
    let classPattern: ClassPatternPin, classPatternMasked: MaskedPin
}
private let pins: Pins = {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/classification.json")
    return try! JSONDecoder().decode(Pins.self, from: Data(contentsOf: url))
}()

private func vectors(_ peaks: [[[Double]]], _ scan: [Int]) -> BraggVectors {
    BraggVectors(scanRows: scan[0], scanColumns: scan[1], peaks: peaks.map { $0.map { BraggPeak(row: $0[0], column: $0[1], intensity: $0[2]) } })
}

@discardableResult
private func allclose(_ label: String, _ got: [Double], _ want: [Double], rtol: Double = 1e-5, atol: Double = 1e-8) -> Double {
    #expect(got.count == want.count, "\(label): \(got.count) values against \(want.count)")
    var worst = 0.0
    for (g, w) in zip(got, want) {
        if g.isNaN && w.isNaN { continue }
        worst = max(worst, abs(g - w) / (atol + rtol * abs(w)))
    }
    print("classification \(label): worst gap / allclose bound = \(worst)")
    #expect(worst <= 1, "\(label)")
    return worst
}

private func classification(booleanData: Bool = true, maximumDistance: Double? = nil) throws -> BraggVectorClassification {
    try BraggVectorClassification(vectors: vectors(pins.peaks, pins.scan), referenceRows: pins.Qx, referenceColumns: pins.Qy,
                                  booleanData: booleanData, maximumDistance: maximumDistance)
}

@Suite struct BraggVectorClassificationTests {
    @Test func labelsAndDataMatrixMatchPy4DSTEM() throws {
        for pin in pins.labels {
            let c = try classification(maximumDistance: pin.maxDist)
            #expect(c.peakLabels == pin.labels, "labels, max_dist \(String(describing: pin.maxDist))")
            #expect(c.data == pin.xBoolean, "boolean X")
            let i = try classification(booleanData: false, maximumDistance: pin.maxDist)
            #expect(i.data == pin.xIntensity, "intensity X (an intensity is copied, not computed)")
        }
        #expect(pins.labels[1].labels != pins.labels[0].labels, "the pin holds a peak that max_dist removes")
    }

    @Test func initialClassesMatchPy4DSTEM() throws {
        let labels = try BraggVectorClassification.peakLabels(vectors: vectors(pins.peaks, pins.scan), referenceRows: pins.Qx,
                                                              referenceColumns: pins.Qy)
        for pin in pins.initialClasses {
            let got = try BraggVectorClassification.initialClasses(
                peakLabels: labels, featureCount: pins.Qx.count, threshold: pin.args.thresh ?? 0.3,
                fractionThreshold: pin.args.BP_fraction_thresh ?? 0.1, maximumIterations: pin.args.max_iterations ?? 200,
                correlationOrder: pin.args.n_corr_init ?? 2)
            #expect(got == pin.classes, "initial classes \(pin.args)")
        }
    }

    private func build(_ pin: CasePin) throws -> BraggVectorClassification {
        var c = try classification(booleanData: pin.args?.booleanData ?? true)
        if let images = pin.images {
            try c.setInitialClasses(images: images)
        } else {
            try c.setInitialClassesByCooccurrence(threshold: pin.args?.thresh ?? 0.3, correlationOrder: pin.args?.n_corr_init ?? 2)
        }
        return c
    }

    @Test func initialClassesStatesMatchPy4DSTEM() throws {
        for pin in pins.cases {
            let c = try build(pin)
            #expect(c.classCount == pin.state.classCount, "\(pin.name) class count")
            allclose("\(pin.name) W", c.w, pin.state.w)
            allclose("\(pin.name) H", c.h, pin.state.h)
            if pin.images == nil { #expect(c.w == pin.state.w, "\(pin.name): W is 0/1, exact") }
        }
    }

    private func apply(_ op: OpPin, to c: inout BraggVectorClassification) throws {
        switch op.op {
        case "nmf": try c.nmf(maximumIterations: op.maxIterations!)
        case "split": try c.split(sigma: op.sigma!, thresholdSplit: op.thresholdSplit!, expandMask: op.expandMask!, minimumPixels: op.minimumPixels!)
        case "splitIndex": try c.split(classIndex: op.index!, sigma: op.sigma!, thresholdSplit: op.thresholdSplit!, expandMask: op.expandMask!, minimumPixels: op.minimumPixels!)
        case "merge": try c.merge(thresholdPeaks: op.thresholdPeaks!, thresholdScan: op.thresholdScan!)
        case "mergeIndex": try c.merge(classIndices: op.i!, op.j!)
        default: try c.removeClass(op.index!)
        }
    }

    @Test func refinementsMatchPy4DSTEM() throws {
        var compared = 0, errors = 0
        for state in pins.states {
            for result in state.results {
                var c = try classification()
                try c.setClasses(w: state.w, h: state.h, classCount: state.classCount)
                let tag = "\(state.name) \(result.op.op) \(result.op.sigma.map { "sigma \($0)" } ?? "")"
                if let error = result.error {
                    #expect(throws: CuboBraggError.self, "\(tag) is \(error) upstream") { try apply(result.op, to: &c) }
                    errors += 1
                    continue
                }
                try apply(result.op, to: &c)
                let candidate = try #require(c.candidate, "\(tag): no candidate")
                let want = try #require(result.candidate)
                #expect(candidate.classCount == want.classCount, "\(tag): \(candidate.classCount) classes against \(want.classCount)")
                if candidate.classCount == want.classCount {
                    allclose(tag + " W_next", candidate.w, want.w)
                    allclose(tag + " H_next", candidate.h, want.h)
                }
                c.accept()
                #expect(c.candidate == nil)
                let accepted = try #require(result.accepted)
                #expect(c.classCount == accepted.classCount)
                allclose(tag + " accepted W", c.w, accepted.w)
                allclose(tag + " accepted H", c.h, accepted.h)
                compared += 1
            }
        }
        #expect(compared >= 20 && errors >= 2)
    }

    @Test func rejectAndAcceptKeepTheState() throws {
        var c = try classification()
        try c.setInitialClassesByCooccurrence()
        let w = c.w, h = c.h
        try c.removeClass(0)
        #expect(c.candidate?.classCount == c.classCount - 1)
        c.reject()
        #expect(c.candidate == nil && c.w == w && c.h == h)
        c.accept()                                    // nothing to accept
        #expect(c.w == w && c.classCount == 3)
        var empty = try classification()
        #expect(throws: CuboBraggError.self) { try empty.nmf() }
    }

    // MARK: planted classes

    @Test func plantedPhasesAreRecovered() throws {
        var c = try classification()
        try c.setInitialClassesByCooccurrence()
        #expect(c.classCount == 3)
        // the class holding reflection 1 is phase A, 4 phase B, 7 phase C: its image is large inside the phase, small outside
        for (phase, label) in [("A", 1), ("B", 4), ("C", 7)] {
            let k = try #require((0..<c.classCount).first { try c.classPeakWeights($0)[label] == 1 })
            let image = try c.classImage(k)
            let truth = pins.truth[phase]!
            let inside = zip(image, truth).filter { $0.1 == 1 }.map(\.0), outside = zip(image, truth).filter { $0.1 == 0 }.map(\.0)
            let meanInside = inside.reduce(0, +) / Double(inside.count), meanOutside = outside.reduce(0, +) / Double(outside.count)
            #expect(meanInside > 0.6 && meanOutside < 0.15, "phase \(phase): inside \(meanInside), outside \(meanOutside)")
        }
        // the NMF keeps the three classes; splitting the three-phase state separates phase C's two patches
        try c.nmf(maximumIterations: 200)
        c.accept()
        #expect(c.classCount == 3)
        try c.split()
        #expect(c.candidate?.classCount == 4, "phase C lies in two separate patches")
    }

    @Test func mergeRefusesASingleClass() throws {
        var c = try classification()
        try c.setClasses(w: [Double](repeating: 1, count: 9), h: [Double](repeating: 1, count: 168), classCount: 1)
        #expect(throws: CuboBraggError.self) { try c.merge() }
    }

    @Test func cancelsIterativeSteps() throws {
        var c = try classification()
        #expect(throws: CancellationError.self) { try c.setInitialClassesByCooccurrence(isCancelled: { true }) }
        try c.setInitialClassesByCooccurrence()
        #expect(throws: CancellationError.self) { try c.nmf(maximumIterations: 5, isCancelled: { true }) }
        #expect(throws: CancellationError.self) { try c.merge(isCancelled: { true }) }
    }

    @Test func refusesBadInput() throws {
        #expect(throws: CuboBraggError.self) {
            _ = try BraggVectorClassification(vectors: vectors(pins.peaks, pins.scan), referenceRows: [1, 2], referenceColumns: [1])
        }
        var c = try classification()
        #expect(throws: CuboBraggError.self) { try c.setInitialClasses(images: [[1, 2, 3]]) }
        #expect(throws: CuboBraggError.self) { try c.setInitialClassesByCooccurrence(correlationOrder: 4) }
        #expect(throws: CuboBraggError.self) { try c.setInitialClassesByCooccurrence(threshold: 1.5) }
    }

    // MARK: class diffraction patterns

    private func cube(_ detector: [Int]) throws -> MemoryCube {
        try MemoryCube(shape: CubeShape(scanRows: 12, scanColumns: 14, detectorRows: detector[0], detectorColumns: detector[1]),
                       values: pins.classPattern.patterns.map { Float($0) })
    }

    @Test func classPatternsMatchPy4DSTEM() async throws {
        let p = pins.classPattern
        let source = try cube(p.detector)
        for pin in p.cases {
            let a = pin.args
            let halves = a.shifts?.value == "halves"
            let shifted = a.shifts != nil
            let got = try await BraggVectorClassification.classPattern(
                of: source, classImage: p.image, threshold: a.threshold ?? 0.01,
                rowShifts: shifted ? (halves ? p.rowHalves : p.rowShifts) : nil,
                columnShifts: shifted ? (halves ? p.columnHalves : p.columnShifts) : nil,
                darkReference: a.dark == true ? p.dark : nil, integerShifts: a.integerShifts ?? true)
            allclose("get_class_DP \(a)", got, pin.dp)
        }
    }

    @Test func maskedClassPatternsMatchPy4DSTEM() async throws {
        let p = pins.classPattern, m = pins.classPatternMasked
        let source = try cube(p.detector)
        let v = vectors(m.peaks, pins.scan)
        for pin in m.cases {
            let a = pin.args
            let got = try await BraggVectorClassification.classPatternWithoutBraggScattering(
                of: source, classImage: p.image, vectors: v, radius: a.radius!, originRow: (a.origin ?? m.origin)[0], originColumn: (a.origin ?? m.origin)[1],
                threshold: a.threshold ?? 0.01, rowShifts: a.shifts != nil ? p.rowShifts : nil,
                columnShifts: a.shifts != nil ? p.columnShifts : nil, darkReference: a.dark == true ? p.dark : nil,
                integerShifts: a.integerShifts ?? true)
            allclose("get_class_DP_without_Bragg_scattering \(a)", got, pin.dp)
        }
    }
}
