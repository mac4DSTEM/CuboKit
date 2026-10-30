import Foundation
import Testing
import CuboCore
@testable import CuboParallax

/// Counts polls and trips after a number of them.
private final class Trip: @unchecked Sendable {
    private let lock = NSLock()
    private var polls = 0
    let after: Int
    init(after: Int) { self.after = after }
    var count: Int { lock.lock(); defer { lock.unlock() }; return polls }
    func poll() -> Bool { lock.lock(); defer { lock.unlock() }; polls += 1; return polls > after }
}
private final class Log: @unchecked Sendable {
    var calls: [[Int]] = []
    func add(_ done: Int, _ total: Int) { calls.append([done, total]) }
}

struct AppPins: Decodable {
    struct Correlation: Decodable {
        struct Case: Decodable { let factor: Int; let planted, shift: [Double]; let moving: [[Double]] }
        let rows, columns: Int; let reference: [[Double]]; let cases: [Case]
    }
    struct Transfer: Decodable { let coefs, real, imag, image: [Double]; let minAbsSin: Double }
    struct PositionCase: Decodable {
        let name: String; let iterations: Int; let factor, sigma: Double; let checkerboard: Bool
        let initialStep, minimumStep, stepFactor: Double; let dr, dc, scores, image: [Double]
    }
    struct Position: Decodable {
        let rows, columns, count: Int; let images, shifts, window: [Double]
        let cases: [PositionCase]; let flat: PositionCase
    }
    let correlation: [Correlation]; let transfer: [Transfer]; let position: Position
    static func load() throws -> AppPins {
        let url = try #require(parallaxFixture("appvariants"))
        return try JSONDecoder().decode(AppPins.self, from: Data(contentsOf: url))
    }
}
func maxGap(_ a: [Double], _ b: [Double], floor: Double = 1) -> Double {
    #expect(a.count == b.count)
    return zip(a, b).map { abs($0 - $1) / max(floor, abs($1)) }.max() ?? .infinity
}

// MARK: plan-reusing shifter

@Suite struct FourierShifterTests {
    // Same code path as ParallaxAlignment.shifted: results are bit-identical (exact equality, no tolerance).
    @Test func bitIdenticalToShifted() throws {
        for (rows, columns) in [(8, 9), (12, 12), (7, 5), (16, 10)] {
            let shifter = try ParallaxFourierShifter(rows: rows, columns: columns)
            let images = LegacyInputs.images(3, rows: rows, columns: columns, seed: 31)
            let shifts = [ParallaxShift(row: 1.3, column: -2.6), ParallaxShift(row: -0.45, column: 3.25), ParallaxShift(row: 0, column: 0)]
            let batch = try shifter.shifted(images, by: shifts)
            for i in images.indices {
                let reference = try ParallaxAlignment.shifted(images[i], rows: rows, columns: columns, by: shifts[i])
                #expect(try shifter.shifted(images[i], by: shifts[i]) == reference)
                #expect(batch[i] == reference)
            }
        }
    }
    @Test func concurrentUseIsBitIdentical() throws {
        let rows = 12, columns = 10
        let shifter = try ParallaxFourierShifter(rows: rows, columns: columns)
        let images = LegacyInputs.images(16, rows: rows, columns: columns, seed: 5)
        let shifts = (0..<16).map { ParallaxShift(row: Double($0) * 0.37 - 3, column: 2 - Double($0) * 0.21) }
        let expected = try (0..<16).map { try ParallaxAlignment.shifted(images[$0], rows: rows, columns: columns, by: shifts[$0]) }
        final class Box: @unchecked Sendable { var out = [[Double]](repeating: [], count: 16) }
        let box = Box(), lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: 16) { i in
            let r = try! shifter.shifted(images[i], by: shifts[i])
            lock.lock(); box.out[i] = r; lock.unlock()
        }
        #expect(box.out == expected)
    }
    @Test func spectrumRoundTripAndPhaseRamp() throws {
        let shifter = try ParallaxFourierShifter(rows: 9, columns: 8)
        let image = LegacyInputs.values(72, seed: 3)
        let s = try shifter.spectrum(of: image)
        let moved = try shifter.inverse(shifter.shifting(s, by: .init(row: 0.7, column: -1.2)))
        #expect(moved == (try shifter.shifted(image, by: .init(row: 0.7, column: -1.2))))
    }
    @Test func cancellationProgressAndErrors() throws {
        let shifter = try ParallaxFourierShifter(rows: 8, columns: 8)
        let images = LegacyInputs.images(5, rows: 8, columns: 8, seed: 1)
        let shifts = [ParallaxShift](repeating: .init(row: 1, column: 1), count: 5)
        let log = Log()
        _ = try shifter.shifted(images, by: shifts, progress: log.add)
        #expect(log.calls == (1...5).map { [$0, 5] })
        let trip = Trip(after: 2)
        #expect(throws: CancellationError.self) { try shifter.shifted(images, by: shifts, isCancelled: trip.poll) }
        #expect(trip.count == 3)
        #expect(throws: ParallaxAlignmentError.inconsistentShape) { try shifter.shifted(images, by: Array(shifts.dropLast())) }
        #expect(throws: ParallaxAlignmentError.invalidDimensions) { _ = try ParallaxFourierShifter(rows: 2, columns: 8) }
        #expect(throws: ParallaxAlignmentError.inconsistentShape) { try shifter.shifted([1, 2, 3], by: shifts[0]) }
    }
    @Test func alignCancellationAndProgress() throws {
        let imgs = LegacyInputs.images(5, rows: 8, columns: 9, seed: 11)
        let log = Log()
        let plain = try ParallaxAlignment.align(Array(imgs[1...]), reference: imgs[0], rows: 8, columns: 9)
        let reported = try ParallaxAlignment.align(Array(imgs[1...]), reference: imgs[0], rows: 8, columns: 9,
                                                   isCancelled: { false }, progress: log.add)
        #expect(log.calls == (1...4).map { [$0, 4] })
        #expect(plain.images == reported.images && plain.shifts == reported.shifts)
        let trip = Trip(after: 1)
        #expect(throws: CancellationError.self) {
            try ParallaxAlignment.align(Array(imgs[1...]), reference: imgs[0], rows: 8, columns: 9, isCancelled: trip.poll)
        }
    }
}

// MARK: spectrum-level shift estimate

@Suite struct CorrelationShiftTests {
    // Float64 transcription of the application's correlationShift and MatrixDFTCorrelation (tools/parallax/app_pins.py).
    // Both sides are Double; the matrix-DFT sums differ in order only: measured worst gap 3.3e-14 px, written tolerance 1e-11 px.
    static let tolerance = 1e-11
    @Test func pinnedToApplicationTranscription() throws {
        var worst = 0.0
        for group in try AppPins.load().correlation {
            let shifter = try ParallaxFourierShifter(rows: group.rows, columns: group.columns)
            let reference = ParallaxSpectrum(rows: group.rows, columns: group.columns, real: group.reference[0], imaginary: group.reference[1])
            for c in group.cases {
                let moving = ParallaxSpectrum(rows: group.rows, columns: group.columns, real: c.moving[0], imaginary: c.moving[1])
                for result in [try shifter.correlationShift(reference: reference, moving: moving, upsampleFactor: c.factor),
                               try ParallaxAlignment.correlationShift(reference: reference, moving: moving, upsampleFactor: c.factor)] {
                    let gap = max(abs(result.row - c.shift[0]), abs(result.column - c.shift[1]))
                    worst = max(worst, gap)
                    #expect(gap <= Self.tolerance, "\(group.rows)x\(group.columns) factor \(c.factor): \(gap)")
                }
            }
        }
        print("correlationShift worst gap vs application transcription \(worst)")
    }
    // The real-space `shift` is the same estimate: spectra of the images give bit-identical translations.
    @Test func equalsImageLevelShift() throws {
        let rows = 10, columns = 9
        let shifter = try ParallaxFourierShifter(rows: rows, columns: columns)
        let images = LegacyInputs.images(3, rows: rows, columns: columns, seed: 71)
        for factor in [3, 8, 20] {
            let viaImages = try ParallaxAlignment.shift(reference: images[0], moving: images[1], rows: rows, columns: columns, upsampleFactor: factor)
            let viaSpectra = try shifter.correlationShift(reference: shifter.spectrum(of: images[0]), moving: shifter.spectrum(of: images[1]),
                                                          upsampleFactor: factor)
            #expect(viaImages == viaSpectra)
        }
    }
    // And py4DSTEM's pinned alignment (align_images_fourier, factor 8) through the spectra.
    @Test func matchesPy4DSTEMPins() throws {
        struct Pins: Decodable { struct Case: Decodable { let rows, columns: Int; let reference: [Double]; let frames, shifts: [[Double]]; let running: Bool } ; let cases: [Case] }
        let url = try #require(parallaxFixture("alignment"))
        let pins = try JSONDecoder().decode(Pins.self, from: Data(contentsOf: url))
        var checked = 0
        for item in pins.cases where !item.running {
            let shifter = try ParallaxFourierShifter(rows: item.rows, columns: item.columns)
            let reference = try shifter.spectrum(of: item.reference)
            for (i, frame) in item.frames.enumerated() {
                let shift = try ParallaxAlignment.correlationShift(reference: reference, moving: shifter.spectrum(of: frame))
                #expect(abs(shift.row - item.shifts[i][0]) < 1e-10 && abs(shift.column - item.shifts[i][1]) < 1e-10)
                checked += 1
            }
        }
        #expect(checked > 0)
    }
    @Test func rejectsBadInput() throws {
        let s = ParallaxSpectrum(rows: 6, columns: 6, real: LegacyInputs.values(36, seed: 1), imaginary: LegacyInputs.values(36, seed: 2))
        #expect(throws: ParallaxAlignmentError.invalidUpsampleFactor) { try ParallaxAlignment.correlationShift(reference: s, moving: s, upsampleFactor: 0) }
        #expect(throws: ParallaxAlignmentError.invalidUpsampleFactor) { try ParallaxAlignment.correlationShift(reference: s, moving: s, upsampleFactor: 65) }
        let small = ParallaxSpectrum(rows: 5, columns: 6, real: LegacyInputs.values(30, seed: 1), imaginary: LegacyInputs.values(30, seed: 2))
        #expect(throws: ParallaxAlignmentError.inconsistentShape) { try ParallaxAlignment.correlationShift(reference: s, moving: small) }
        let flat = ParallaxSpectrum(rows: 6, columns: 6, real: [Double](repeating: 0, count: 36), imaginary: [Double](repeating: 0, count: 36))
        #expect(throws: ParallaxAlignmentError.degenerateCorrelation) { try ParallaxAlignment.correlationShift(reference: s, moving: flat, upsampleFactor: 2) }
        var bad = s; bad.real[3] = .nan
        #expect(throws: ParallaxAlignmentError.nonfiniteValues) { try ParallaxAlignment.correlationShift(reference: s, moving: bad) }
    }
}

// MARK: transfer function

@Suite struct TransferFunctionTests {
    // Float64 transcription of the application's correction (app_pins.py). The sign of sin(chi_even) is the only
    // discontinuity; the pin's smallest |sin| is recorded in the fixture (0.004), far above the 1e-13 phase differences.
    static let tolerance = 1e-12   // measured worst gap 2.6e-15
    @Test func transferMatchesTranscription() throws {
        let ab = try JSONDecoder().decode(AberrationPin.self, from: Data(contentsOf: #require(parallaxFixture("aberrations"))))
        let terms = ab.terms.map { ParallaxAberrationTerm(radialOrder: $0[0], angularOrder: $0[1], component: $0[2]) }
        var worst = 0.0
        for t in try AppPins.load().transfer {
            let r = try ParallaxAberrationCorrection.correct(ab.image, rows: ab.rows, columns: ab.columns, sampling: .init(row: 1.25, column: 1.25),
                wavelength: ab.wavelength, terms: terms, coefficients: t.coefs, lowpass: 0.3, highpass: 0.12, highpassBehavior: .apply)
            #expect(r.transferReal.count == ab.rows * ab.columns && r.transferImaginary.count == ab.rows * ab.columns)
            let gap = max(maxGap(r.transferReal, t.real, floor: 1), maxGap(r.transferImaginary, t.imag, floor: 1))
            let imageGap = maxGap(r.image, t.image)
            worst = max(worst, gap, imageGap)
            #expect(gap <= Self.tolerance && imageGap <= Self.tolerance, "transfer \(gap) image \(imageGap)")
            #expect(r.transferReal[0] == 0 && r.transferImaginary[0] == 0)
        }
        print("transfer function worst gap \(worst)")
    }
    // The returned transfer is the multiplier of the image spectrum: applying it reproduces the corrected image.
    @Test func transferReproducesImage() throws {
        let image = LegacyInputs.values(8 * 9, seed: 11)
        let r = try ParallaxAberrationCorrection.correct(image, rows: 8, columns: 9, sampling: .init(row: 1.25, column: 1.25),
            wavelength: 0.025, terms: LegacyInputs.terms, coefficients: LegacyInputs.coefficients, lowpass: 0.3)
        let shifter = try ParallaxFourierShifter(rows: 8, columns: 9)
        var s = try shifter.spectrum(of: image)
        for i in 0..<72 {
            let re = s.real[i], im = s.imaginary[i]
            s.real[i] = re * r.transferReal[i] - im * r.transferImaginary[i]
            s.imaginary[i] = re * r.transferImaginary[i] + im * r.transferReal[i]
        }
        let again = try shifter.inverse(s)
        #expect(maxGap(again, r.image) < 1e-13)
    }
    struct AberrationPin: Decodable { let rows, columns: Int; let image: [Double]; let wavelength: Double; let terms: [[Int]] }
}

// MARK: depth sectioning, plane at a time

@Suite struct DepthSectionerTests {
    static func setup() throws -> ([[Double]], [ParallaxShift]) {
        (Array(LegacyInputs.images(4, rows: 8, columns: 9, seed: 11)[1...]), LegacyInputs.angles)
    }
    @Test func planesEqualSection() throws {
        let (stack, angles) = try Self.setup()
        let depths = [-50.0, 0, 120, 300]
        for full in [false, true] {
            let ctf: ParallaxDepthCTF = full ? .fullFit(terms: LegacyInputs.terms, coefficientsAngstrom: LegacyInputs.coefficients) : .defocusOnly(c1Angstrom: 80)
            let sectioner = try ParallaxDepthSectioner(stack: stack, rows: 8, columns: 9, probeAngles: angles, sampling: .init(row: 1.25, column: 1.25),
                wavelength: 0.025, ctf: ctf, informationLimit: 0.4, informationPower: 2)
            let all = try ParallaxDepthSectioning.section(stack: stack, rows: 8, columns: 9, probeAngles: angles, sampling: .init(row: 1.25, column: 1.25),
                wavelength: 0.025, ctf: ctf, depthsAngstrom: depths, informationLimit: 0.4, informationPower: 2)
            for (i, depth) in depths.enumerated() { #expect(try sectioner.plane(atDepth: depth) == all[i]) }   // bit-identical
            #expect(sectioner.imageCount == 3 && sectioner.rows == 8 && sectioner.columns == 9)
        }
    }
    @Test func planesCanBeFormedConcurrently() throws {
        let (stack, angles) = try Self.setup()
        let sectioner = try ParallaxDepthSectioner(stack: stack, rows: 8, columns: 9, probeAngles: angles, sampling: .init(row: 1.25, column: 1.25),
            wavelength: 0.025, ctf: .defocusOnly(c1Angstrom: 80))
        let depths = (0..<12).map { Double($0) * 40 - 200 }
        let expected = try depths.map { try sectioner.plane(atDepth: $0) }
        final class Box: @unchecked Sendable { var out = [[Double]](repeating: [], count: 12) }
        let box = Box(), lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: 12) { i in
            let r = try! sectioner.plane(atDepth: depths[i]); lock.lock(); box.out[i] = r; lock.unlock()
        }
        #expect(box.out == expected)
    }
    @Test func progressAndCancellation() throws {
        let (stack, angles) = try Self.setup()
        let log = Log()
        _ = try ParallaxDepthSectioning.section(stack: stack, rows: 8, columns: 9, probeAngles: angles, sampling: .init(row: 1.25, column: 1.25),
            wavelength: 0.025, ctf: .defocusOnly(c1Angstrom: 80), depthsAngstrom: [0, 10, 20], progress: log.add)
        #expect(log.calls == [[1, 3], [2, 3], [3, 3]])
        let trip = Trip(after: 5)
        #expect(throws: CancellationError.self) {
            try ParallaxDepthSectioning.section(stack: stack, rows: 8, columns: 9, probeAngles: angles, sampling: .init(row: 1.25, column: 1.25),
                wavelength: 0.025, ctf: .defocusOnly(c1Angstrom: 80), depthsAngstrom: [0, 10, 20], isCancelled: trip.poll)
        }
        let sectioner = try ParallaxDepthSectioner(stack: stack, rows: 8, columns: 9, probeAngles: angles, sampling: .init(row: 1.25, column: 1.25),
            wavelength: 0.025, ctf: .defocusOnly(c1Angstrom: 80))
        #expect(throws: CancellationError.self) { try sectioner.plane(atDepth: 0, isCancelled: { true }) }
        #expect(throws: ParallaxError.invalidInput) { try sectioner.plane(atDepth: .nan) }
        #expect(throws: CancellationError.self) {
            try ParallaxDepthSectioner(stack: stack, rows: 8, columns: 9, probeAngles: angles, sampling: .init(row: 1.25, column: 1.25),
                wavelength: 0.025, ctf: .defocusOnly(c1Angstrom: 80), isCancelled: { true })
        }
    }
}

// MARK: position correction: application variant, cancellation, progress

@Suite struct PositionCorrectionVariantTests {
    // Float32 NumPy transcription of the application's loop (app_pins.py). Both sides run the same float32 operations in
    // the same order on the same Double deposition rounded to Float; a one-ulp difference in the deposition can flip a
    // comparison; measured gaps are 0 for offsets and scores, and the image (Double here, rounded to float32 in the transcription)
    // differs by 5.5e-8 relative. Written tolerances: offsets 1e-6 px, scores 1e-6 relative, image 1.2e-7 (float32 rounding).
    static let offsetTolerance = 1e-6, scoreTolerance = 1e-6, imageTolerance = 1.2e-7

    static func run(_ c: AppPins.PositionCase, images: [[Double]], p: AppPins.Position, shifts: [ParallaxShift], window: [Double],
                    variant: ParallaxPositionCorrectionVariant, isCancelled: (() -> Bool)? = nil,
                    progress: ((Int, Int) -> Void)? = nil) throws -> ParallaxPositionCorrectionResult {
        var o = ParallaxPositionCorrectionOptions(iterations: c.iterations)
        o.checkerboard = c.checkerboard; o.initialStep = c.initialStep; o.minimumStep = c.minimumStep; o.stepFactor = c.stepFactor
        o.variant = variant
        return try ParallaxSubpixel.correctPositions(images: images, rows: p.rows, columns: p.columns, shifts: shifts, window: window,
            scanSampling: .init(row: 1.25, column: 1.25), factor: c.factor, sigmaPixels: c.sigma, options: o,
            isCancelled: isCancelled, progress: progress)
    }
    static func inputs(_ p: AppPins.Position) -> ([[Double]], [ParallaxShift]) {
        let n = p.rows * p.columns
        return ((0..<p.count).map { Array(p.images[$0 * n..<($0 + 1) * n]) },
                stride(from: 0, to: p.shifts.count, by: 2).map { .init(row: p.shifts[$0], column: p.shifts[$0 + 1]) })
    }

    @Test func applicationVariantMatchesTranscription() throws {
        let p = try AppPins.load().position
        let (images, shifts) = Self.inputs(p)
        for c in p.cases {
            let r = try Self.run(c, images: images, p: p, shifts: shifts, window: p.window, variant: .application)
            let g = (offsets: maxGap(r.offsets.rows + r.offsets.columns, c.dr + c.dc, floor: 1),
                     scores: maxGap(r.scores, c.scores, floor: 1e-30), image: maxGap(r.image.image, c.image))
            print("application variant \(c.name): offsets \(g.offsets) scores \(g.scores) image \(g.image)")
            #expect(r.scores.count == c.iterations + 1)
            #expect(g.offsets <= Self.offsetTolerance && g.scores <= Self.scoreTolerance && g.image <= Self.imageTolerance, "\(c.name)")
        }
    }
    // Where the step floor exceeds the initial step the two loops differ (the application floors rejected sites only).
    @Test func variantsDifferWhereTheStepFloorMatters() throws {
        let p = try AppPins.load().position
        let (images, shifts) = Self.inputs(p)
        let c = try #require(p.cases.first { $0.name == "floor-above-initial" })
        let app = try Self.run(c, images: images, p: p, shifts: shifts, window: p.window, variant: .application)
        let ref = try Self.run(c, images: images, p: p, shifts: shifts, window: p.window, variant: .py4DSTEM)
        #expect(maxGap(app.offsets.rows + app.offsets.columns, ref.offsets.rows + ref.offsets.columns, floor: 1) > 1e-4)
    }
    // A zero gradient (flat scores) gives a zero move in the application's loop; py4DSTEM's 0/0 is not finite and is refused.
    @Test func zeroGradientIsGuarded() throws {
        let p = try AppPins.load().position
        let images = [[Double]](repeating: [Double](repeating: 1, count: 36), count: 3)
        let shifts = [ParallaxShift](repeating: .init(row: 0, column: 0), count: 3)
        let c = p.flat
        let r = try Self.run(c, images: images, p: .init(rows: 6, columns: 6, count: 3, images: [], shifts: [], window: [], cases: [], flat: c),
                             shifts: shifts, window: [Double](repeating: 1, count: 36), variant: .application)
        #expect((r.offsets.rows + r.offsets.columns).allSatisfy { $0 == 0 })
        #expect(maxGap(r.scores, c.scores, floor: 1e-30) == 0 && maxGap(r.offsets.rows, c.dr) == 0)
        #expect(throws: ParallaxError.nonfiniteValues) {
            try Self.run(c, images: images, p: .init(rows: 6, columns: 6, count: 3, images: [], shifts: [], window: [], cases: [], flat: c),
                         shifts: shifts, window: [Double](repeating: 1, count: 36), variant: .py4DSTEM)
        }
    }
    @Test func applicationVariantRefusesRegularisation() throws {
        let p = try AppPins.load().position
        let (images, shifts) = Self.inputs(p)
        for mutate in [{ (o: inout ParallaxPositionCorrectionOptions) in o.gaussianSigma = .init(row: 1, column: 1) },
                       { (o: inout ParallaxPositionCorrectionOptions) in o.lowpass = .init(row: 1, column: 1) },
                       { (o: inout ParallaxPositionCorrectionOptions) in o.highpass = .init(row: 0.1, column: 0.1) }] {
            var o = ParallaxPositionCorrectionOptions(iterations: 1); o.variant = .application; mutate(&o)
            #expect(throws: ParallaxError.invalidInput) {
                try ParallaxSubpixel.correctPositions(images: images, rows: p.rows, columns: p.columns, shifts: shifts, window: p.window,
                    scanSampling: .init(row: 1.25, column: 1.25), factor: 2, options: o)
            }
        }
    }
    @Test func progressAndCancellationBothVariants() throws {
        let p = try AppPins.load().position
        let (images, shifts) = Self.inputs(p)
        let c = p.cases[0]
        for variant in [ParallaxPositionCorrectionVariant.py4DSTEM, .application] {
            let log = Log()
            let plain = try Self.run(c, images: images, p: p, shifts: shifts, window: p.window, variant: variant)
            let reported = try Self.run(c, images: images, p: p, shifts: shifts, window: p.window, variant: variant,
                                        isCancelled: { false }, progress: log.add)
            #expect(log.calls == (1...3).map { [$0, 3] })
            #expect(plain.offsets == reported.offsets && plain.scores == reported.scores && plain.image.image == reported.image.image)
            let trip = Trip(after: 4)
            #expect(throws: CancellationError.self) {
                try Self.run(c, images: images, p: p, shifts: shifts, window: p.window, variant: variant, isCancelled: trip.poll)
            }
            #expect(throws: CancellationError.self) {
                try Self.run(c, images: images, p: p, shifts: shifts, window: p.window, variant: variant, isCancelled: { true })
            }
        }
    }
}

// MARK: reconstruct and process: cancellation and progress

@Suite struct ReconstructControlTests {
    @Test func reconstructProgressAndCancellation() throws {
        let c = try #require(try ParallaxReconstructTests.pin().cases.first)
        let prep = try ParallaxReconstructTests.prepared(c)
        let plain = try ParallaxReconstruction.reconstruct(prep, options: c.options.swift)
        let log = Log()
        let reported = try ParallaxReconstruction.reconstruct(prep, options: c.options.swift, isCancelled: { false }, progress: log.add)
        #expect(log.calls == (1...plain.binValues.count).map { [$0, plain.binValues.count] })
        #expect(plain.image == reported.image && plain.shifts == reported.shifts && plain.errors == reported.errors)
        for after in [0, 3, 40] {
            let trip = Trip(after: after)
            #expect(throws: CancellationError.self) { try ParallaxReconstruction.reconstruct(prep, options: c.options.swift, isCancelled: trip.poll) }
        }
    }
    @Test func processProgressAndCancellation() throws {
        let (options, e, k) = try DefaultUnchangedTests.endToEndSetup()
        func run(isCancelled: (() -> Bool)? = nil, progress: ((Int, Int) -> Void)? = nil) throws -> ParallaxProcessingResult {
            try ParallaxReconstruction.process(e.cube, shape: CubeShape(e.scan + e.detector),
                reciprocalSampling: .init(row: k.reciprocalSampling, column: k.reciprocalSampling),
                scanSampling: .init(row: k.scanSampling, column: k.scanSampling), energyEV: 200000, options: options,
                isCancelled: isCancelled, progress: progress)
        }
        let log = Log()
        let r = try run(progress: log.add)
        let total = r.reconstruction.binValues.count + 2
        #expect(log.calls == (1...total).map { [$0, total] })
        for after in [0, 5, 60] {
            let trip = Trip(after: after)
            #expect(throws: CancellationError.self) { try run(isCancelled: trip.poll) }
        }
    }
}
