import Foundation
import Testing
import CuboCore
@testable import CuboParallax

// Every long call gained optional `isCancelled` and `progress` parameters and the position correction a `variant`; with the
// defaults, their results are those of CuboParallax 0.5.0 (deb33d9). Fixture legacy050.json holds the 0.5.0 outputs on the
// deterministic inputs of LegacyInputs (written once by a throwaway generator run on that commit).
// Tolerance: 1e-12 of max(1, max|expected|) -- the code paths are unchanged, so on one machine the gap is zero; the
// allowance covers a different vDSP build only.
@Suite struct DefaultUnchangedTests {
    static func golden() throws -> [String: [Double]] {
        let url = try #require(Bundle.module.url(forResource: "legacy050", withExtension: "json", subdirectory: "Fixtures"))
        return try JSONDecoder().decode([String: [Double]].self, from: Data(contentsOf: url))
    }
    static func same(_ actual: [Double], _ expected: [Double], _ name: String, sourceLocation: SourceLocation = #_sourceLocation) {
        #expect(actual.count == expected.count, "\(name) length", sourceLocation: sourceLocation)
        let scale = max(1, expected.map(abs).max() ?? 0)
        let gap = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
        #expect(gap <= 1e-12 * scale, "\(name) gap \(gap)", sourceLocation: sourceLocation)
    }
    static let rows = 8, columns = 9

    @Test func fourierShiftAndAlignment() throws {
        let g = try Self.golden()
        let imgs = LegacyInputs.images(4, rows: Self.rows, columns: Self.columns, seed: 11)
        Self.same(try ParallaxAlignment.shifted(imgs[0], rows: Self.rows, columns: Self.columns, by: .init(row: 1.3, column: -2.6)),
                  g["shifted"]!, "shifted")
        for running in [false, true] {
            let r = try ParallaxAlignment.align(Array(imgs[1...]), reference: imgs[0], rows: Self.rows, columns: Self.columns,
                                                runningAverage: running)
            Self.same(r.shifts.flatMap { [$0.row, $0.column] } + r.images.flatMap { $0 } + r.mean, g["align\(running)"]!, "align \(running)")
        }
    }
    @Test func correctionKeepsImageAndSurfaces() throws {
        let g = try Self.golden()
        let imgs = LegacyInputs.images(1, rows: Self.rows, columns: Self.columns, seed: 11)
        let c = try ParallaxAberrationCorrection.correct(imgs[0], rows: Self.rows, columns: Self.columns, sampling: .init(row: 1.25, column: 1.25),
            wavelength: 0.025, terms: LegacyInputs.terms, coefficients: LegacyInputs.coefficients, lowpass: 0.3, highpass: 0.1, highpassBehavior: .apply)
        Self.same(c.image + c.chiEven + c.chiOdd, g["correct"]!, "correct")
    }
    @Test func depthSectioning() throws {
        let g = try Self.golden()
        let stack = Array(LegacyInputs.images(4, rows: Self.rows, columns: Self.columns, seed: 11)[1...])
        for full in [false, true] {
            let planes = try ParallaxDepthSectioning.section(stack: stack, rows: Self.rows, columns: Self.columns, probeAngles: LegacyInputs.angles,
                sampling: .init(row: 1.25, column: 1.25), wavelength: 0.025,
                ctf: full ? .fullFit(terms: LegacyInputs.terms, coefficientsAngstrom: LegacyInputs.coefficients) : .defocusOnly(c1Angstrom: 80),
                depthsAngstrom: [-50, 0, 120], informationLimit: 0.4, informationPower: 2)
            Self.same(planes.flatMap { $0 }, g["section\(full)"]!, "section \(full)")
        }
    }
    @Test func positionCorrectionDefaultVariant() throws {
        let g = try Self.golden()
        let images = LegacyInputs.images(3, rows: 6, columns: 6, seed: 99)
        let shifts = [ParallaxShift(row: 0.3, column: -0.4), ParallaxShift(row: -0.2, column: 0.6), ParallaxShift(row: 0.5, column: 0.1)]
        for checker in [false, true] {
            var o = ParallaxPositionCorrectionOptions(iterations: 2); o.checkerboard = checker
            #expect(o.variant == .py4DSTEM)
            let r = try ParallaxSubpixel.correctPositions(images: images, rows: 6, columns: 6, shifts: shifts,
                window: LegacyInputs.values(36, seed: 5, low: 0.2, high: 1), scanSampling: .init(row: 1.25, column: 1.25),
                factor: 2, sigmaPixels: 0.3, options: o)
            Self.same(r.offsets.rows + r.offsets.columns + r.scores + r.image.image, g["position\(checker)"]!, "position \(checker)")
        }
    }
    @Test func endToEndProcess() throws {
        let g = try Self.golden()
        let (options, e, k) = try Self.endToEndSetup()
        let r = try ParallaxReconstruction.process(e.cube, shape: CubeShape(e.scan + e.detector),
            reciprocalSampling: .init(row: k.reciprocalSampling, column: k.reciprocalSampling),
            scanSampling: .init(row: k.scanSampling, column: k.scanSampling), energyEV: 200000, options: options)
        let values = r.reconstruction.shifts.flatMap { [$0.row, $0.column] } + r.reconstruction.image + r.reconstruction.errors
            + r.subpixel!.image + r.positionCorrection!.scores + r.positionCorrection!.offsets.rows
        Self.same(values, g["process"]!, "process")
    }
    /// The end-to-end case with checkerboard position correction from the reconstruct fixture.
    static func endToEndSetup() throws -> (ParallaxProcessingOptions, ParallaxReconstructTests.EndToEnd, ParallaxReconstructTests.E2E) {
        let e = try ParallaxReconstructTests.pin().endToEnd
        var options = ParallaxProcessingOptions()
        options.threshold = 0.5; options.edgeBlend = 2; options.paddingRows = e.padding[0]; options.paddingColumns = e.padding[1]
        var sub = ParallaxSubpixelOptions(); sub.factor = 2; sub.sigmaPixels = 0.3
        var pc = ParallaxPositionCorrectionOptions(iterations: 2); pc.checkerboard = true; sub.positionCorrection = pc
        options.subpixel = sub
        return (options, e, e.cases[1])
    }
}
