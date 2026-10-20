import Foundation
import Testing
import CuboCore
@testable import CuboParallax

// Fixture: Tests/CuboParallaxTests/Fixtures/reconstruct.json, from tools/parallax/reconstruct_pins.py
// (py4DSTEM 0.14.19 `Parallax.preprocess().reconstruct()`, float32; the kit is Double).
@Suite struct ParallaxReconstructTests {
    struct Pin: Decodable { let cases: [Case]; let endToEnd: EndToEnd }
    struct Case: Decodable {
        let name: String
        let scan, detector, padding: [Int]
        let reciprocalSampling, threshold, edgeBlend: Double
        let cube: [Double]
        let options: Options
        let bins: [Double]
        let nImages: Int
        let indices: [Int]
        let shifts, image, mask, errors, stackNorms: [Double]
        let maskSum: Double
        let stack: [Double]?
    }
    struct Options: Decodable {
        let max_alignment_bin: Int?
        let min_alignment_bin: Int?
        let num_iter_at_min_bin: Int?
        let alignment_bin_values: [Double]?
        let centered_alignment_bins: Bool?
        let cross_correlation_upsample_factor: Int?
        let regularizer_matrix_size: [Int]?
        let regularize_shifts: Bool?
        let running_average: Bool?
        var swift: ParallaxReconstructionOptions {
            var o = ParallaxReconstructionOptions()
            o.maxAlignmentBin = max_alignment_bin
            o.minAlignmentBin = min_alignment_bin ?? 1
            o.numIterAtMinBin = num_iter_at_min_bin ?? 2
            o.alignmentBinValues = alignment_bin_values
            o.centeredAlignmentBins = centered_alignment_bins ?? false
            o.crossCorrelationUpsampleFactor = cross_correlation_upsample_factor ?? 8
            if let s = regularizer_matrix_size { o.regularizerMatrixSize = (s[0], s[1]) }
            o.regularizeShifts = regularize_shifts ?? false
            o.runningAverage = running_average ?? true
            return o
        }
    }
    struct EndToEnd: Decodable { let cube: [Double]; let scan, detector, padding: [Int]; let cases: [E2E] }
    struct E2E: Decodable {
        let reciprocalSampling, scanSampling, usedFactor, bfLimit, dfLimit: Double
        let shifts, image: [Double]
        let shape: [Int]
        let dr, dc: [Double]?
        let subpixel: Sub
    }
    struct Sub: Decodable { let kde_upsample_factor: Double?; let kde_sigma_px: Double?; let position_correction_num_iter: Int?; let position_correction_checkerboard_steps: Bool? }

    static func pin() throws -> Pin {
        let url = try #require(Bundle.module.url(forResource: "reconstruct", withExtension: "json", subdirectory: "Fixtures"))
        return try JSONDecoder().decode(Pin.self, from: Data(contentsOf: url))
    }

    static func prepared(_ c: Case) throws -> ParallaxPreprocessingResult {
        try ParallaxPreprocessing.prepare(c.cube, shape: CubeShape(c.scan + c.detector),
            reciprocalSampling: .init(row: c.reciprocalSampling, column: c.reciprocalSampling), energyEV: 200000,
            threshold: c.threshold, edgeBlend: c.edgeBlend, paddingRows: c.padding[0], paddingColumns: c.padding[1])
    }

    /// Largest absolute difference over `max(1, max|expected|)`.
    static func gap(_ a: [Double], _ e: [Double]) -> Double {
        #expect(a.count == e.count)
        let scale = max(1, e.map(abs).max() ?? 0)
        return zip(a, e).map { abs($0 - $1) }.max().map { $0 / scale } ?? .infinity
    }

    // Tolerances, written per quantity (measured gaps are in docs/parallax-alignment.md). The reference keeps the stack,
    // masks, shifts and reconstruction in float32 (eps 1.2e-7) through up to three rounds of Fourier shifts and a
    // median-centred roll: measured gaps are 1e-7 to 9e-7, so shifts (px) are held to 5e-6, images, masks, stack norms
    // and the mask sum to 2e-6 (normwise-style: scaled by max(1, max|expected|)), the per-bin errors to 5e-7.
    static let shiftTolerance = 5e-6, imageTolerance = 2e-6, errorTolerance = 5e-7

    @Test func pinnedReconstruction() throws {
        for c in try Self.pin().cases {
            let prep = try Self.prepared(c)
            #expect(prep.detectorIndices.flatMap { $0 } == c.indices, "\(c.name) detector pixels")
            let r = try ParallaxReconstruction.reconstruct(prep, options: c.options.swift)
            #expect(r.binValues == c.bins, "\(c.name) schedule")
            #expect(r.errors.count == c.bins.count)
            let g = (
                shifts: Self.gap(r.shifts.flatMap { [$0.row, $0.column] }, c.shifts),
                image: Self.gap(r.image, c.image), mask: Self.gap(r.mask, c.mask),
                errors: Self.gap(r.errors, c.errors),
                norms: Self.gap(r.alignedStack.map { sqrt($0.reduce(0) { $0 + $1 * $1 }) }, c.stackNorms),
                maskSum: abs(r.stackMask.flatMap { $0 }.reduce(0, +) - c.maskSum) / c.maskSum,
                stack: c.stack.map { Self.gap(r.alignedStack.flatMap { $0 }, $0) } ?? 0)
            print("reconstruct gap \(c.name): shifts \(g.shifts) image \(g.image) mask \(g.mask) errors \(g.errors) norms \(g.norms) maskSum \(g.maskSum) stack \(g.stack)")
            #expect(g.shifts < Self.shiftTolerance, "\(c.name) shifts")
            #expect(g.image < Self.imageTolerance, "\(c.name) image")
            #expect(g.mask < Self.imageTolerance, "\(c.name) mask")
            #expect(g.errors < Self.errorTolerance, "\(c.name) errors")
            #expect(g.norms < Self.imageTolerance, "\(c.name) stack norms")
            #expect(g.maskSum < Self.imageTolerance, "\(c.name) mask sum")
            #expect(g.stack < Self.imageTolerance, "\(c.name) stack")
        }
    }

    @Test func pinnedEndToEnd() throws {
        let p = try Self.pin().endToEnd
        let shape = try CubeShape(p.scan + p.detector)
        for (k, c) in p.cases.enumerated() {
            var options = ParallaxProcessingOptions()
            options.threshold = 0.5; options.edgeBlend = 2
            options.paddingRows = p.padding[0]; options.paddingColumns = p.padding[1]
            var sub = ParallaxSubpixelOptions()
            sub.factor = c.subpixel.kde_upsample_factor
            sub.sigmaPixels = c.subpixel.kde_sigma_px ?? 0.125
            if let n = c.subpixel.position_correction_num_iter {
                var pc = ParallaxPositionCorrectionOptions(iterations: n)
                pc.checkerboard = c.subpixel.position_correction_checkerboard_steps ?? false
                sub.positionCorrection = pc
            }
            options.subpixel = sub
            let r = try ParallaxReconstruction.process(p.cube, shape: shape,
                reciprocalSampling: .init(row: c.reciprocalSampling, column: c.reciprocalSampling),
                scanSampling: .init(row: c.scanSampling, column: c.scanSampling), energyEV: 200000, options: options)
            let limits = try #require(r.limits), image = try #require(r.subpixel)
            #expect(abs(limits.brightField - c.bfLimit) < 1e-6 && abs(limits.darkField - c.dfLimit) < 1e-6
                    && abs(limits.factor - c.usedFactor) < 1e-6, "limits \(k)")
            #expect([image.rows, image.columns] == c.shape)
            let shiftGap = Self.gap(r.reconstruction.shifts.flatMap { [$0.row, $0.column] }, c.shifts)
            let imageGap = Self.gap(image.image, c.image)
            var offsetGap = 0.0
            if let dr = c.dr, let dc = c.dc {
                let o = try #require(r.positionCorrection).offsets
                offsetGap = Self.gap(o.rows + o.columns, dr + dc)
            }
            print("end-to-end gap \(k): shifts \(shiftGap) image \(imageGap) offsets \(offsetGap)")
            #expect(shiftGap < Self.shiftTolerance, "shifts \(k)")
            // Measured 2e-7 (image) and 4e-8 (offsets); the upsampled image divides by a blurred density, so 5e-6.
            #expect(imageGap < 5e-6, "image \(k)")
            #expect(offsetGap < 5e-6, "offsets \(k)")
        }
    }

    /// Nine images that are known integer shifts of one image are brought onto one another. The image has odd
    /// extents (an even axis loses its Nyquist component in every fractional Fourier shift, so the stack could not
    /// become identical), and 30 passes at the finest bin: the iteration converges geometrically (about 0.37 per pass).
    @Test func plantedIntegerShiftsAreRecovered() throws {
        let rows = 11, columns = 13
        var state: UInt64 = 0x2026_1020
        func uniform() -> Double {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Double(state >> 11) / Double(1 << 53)
        }
        let base = (0..<rows * columns).map { _ in 1 + uniform() }
        let offsets = (-1...1).flatMap { a in (-1...1).map { b in (a, b) } }
        // Offset (a, b) moves the content by (a, 2b) pixels: linear in the detector offset, as parallax is.
        let planted = offsets.map { ($0.0, 2 * $0.1) }
        func rolled(_ dr: Int, _ dc: Int) -> [Double] {
            var out = base
            for r in 0..<rows { for c in 0..<columns {
                out[((r + dr) % rows + rows) % rows * columns + ((c + dc) % columns + columns) % columns] = base[r * columns + c]
            } }
            return out
        }
        let stack = planted.map { rolled($0.0, $0.1) }
        let prep = ParallaxPreprocessingResult(rows: rows, columns: columns,
            detectorIndices: offsets.map { [$0.0 + 1, $0.1 + 1] },
            reciprocalVectors: offsets.map { .init(row: Double($0.0), column: Double($0.1)) },
            probeAngles: offsets.map { .init(row: Double($0.0), column: Double($0.1)) },
            window: [Double](repeating: 1, count: rows * columns), stack: stack, unshiftedStack: stack,
            incoherentImage: base, initialError: 0, scanRows: rows, scanColumns: columns,
            paddedWindow: [Double](repeating: 1, count: rows * columns))
        var options = ParallaxReconstructionOptions()
        options.numIterAtMinBin = 30
        let r = try ParallaxReconstruction.reconstruct(prep, options: options)
        // Aligning undoes the planted shift; the planted set is symmetric, so the median is zero and nothing is rolled.
        for (i, shift) in r.shifts.enumerated() {
            #expect(abs(shift.row + Double(planted[i].0)) < 1e-6 && abs(shift.column + Double(planted[i].1)) < 1e-6,
                    "image \(i): \(shift) vs planted \(planted[i])")
        }
        for image in r.alignedStack {
            #expect(zip(image, base).allSatisfy { abs($0 - $1) < 1e-6 })
        }
        #expect(zip(r.image, base).allSatisfy { abs($0 - $1) < 1e-6 })
        #expect(r.errors.last! < 1e-10)
    }

    @Test func schedules() throws {
        let disk = (0..<8).flatMap { r in (0..<8).map { [r, $0] } }   // diameter 8
        #expect(try ParallaxReconstruction.binSchedule(detectorIndices: disk) == [4, 2, 1, 1])
        var o = ParallaxReconstructionOptions()
        o.numIterAtMinBin = 1
        #expect(try ParallaxReconstruction.binSchedule(detectorIndices: disk, options: o) == [4, 2, 1])
        o.minAlignmentBin = 2; o.maxAlignmentBin = 100
        #expect(try ParallaxReconstruction.binSchedule(detectorIndices: disk, options: o) == [4, 2])
        o.maxAlignmentBin = 3
        #expect(try ParallaxReconstruction.binSchedule(detectorIndices: disk, options: o) == [2])  // maximum 3: ceil(log2 3) = 2, so 2^arange(1, 2)
        o.alignmentBinValues = [0.5, 3, 9]
        #expect(try ParallaxReconstruction.binSchedule(detectorIndices: disk, options: o) == [1, 3, 3])
        // Single detector pixel: diameter 1, no binning is possible.
        let one = [[2, 3]]
        #expect(throws: ParallaxError.invalidInput) { try ParallaxReconstruction.binSchedule(detectorIndices: one) }
        var single = ParallaxReconstructionOptions()
        single.numIterAtMinBin = 1
        #expect(try ParallaxReconstruction.binSchedule(detectorIndices: one, options: single) == [])
        single.emptySchedule = .singlePassAtMinimumBin
        #expect(try ParallaxReconstruction.binSchedule(detectorIndices: one, options: single) == [1])
        var app = ParallaxReconstructionOptions()
        app.emptySchedule = .singlePassAtMinimumBin
        #expect(try ParallaxReconstruction.binSchedule(detectorIndices: one, options: app) == [1])
    }

    @Test func rejectsInvalidInput() throws {
        let c = try #require(Self.pin().cases.first)
        let prep = try Self.prepared(c)
        var o = ParallaxReconstructionOptions()
        o.crossCorrelationUpsampleFactor = 2
        #expect(throws: ParallaxError.invalidInput) { try ParallaxReconstruction.reconstruct(prep, options: o) }
        o = ParallaxReconstructionOptions(); o.minAlignmentBin = 0
        #expect(throws: ParallaxError.invalidInput) { try ParallaxReconstruction.reconstruct(prep, options: o) }
        o = ParallaxReconstructionOptions(); o.alignmentBinValues = []
        #expect(throws: ParallaxError.invalidInput) { try ParallaxReconstruction.reconstruct(prep, options: o) }
        o = ParallaxReconstructionOptions(); o.alignmentBinValues = [.nan]
        #expect(throws: ParallaxError.invalidInput) { try ParallaxReconstruction.reconstruct(prep, options: o) }
        o = ParallaxReconstructionOptions(); o.regularizerMatrixSize = (-1, 1)
        #expect(throws: ParallaxError.invalidInput) { try ParallaxReconstruction.reconstruct(prep, options: o) }
        // A single bright-field pixel cannot be regularised (all reciprocal vectors are zero).
        let shape = try CubeShape(c.scan + c.detector)
        var mask = [Bool](repeating: false, count: shape.detectorCount); mask[27] = true
        let one = try ParallaxPreprocessing.prepare(c.cube, shape: shape, reciprocalSampling: .init(row: 0.08, column: 0.08),
            energyEV: 200000, edgeBlend: 2, paddingRows: 4, paddingColumns: 4, detectorMask: mask)
        o = ParallaxReconstructionOptions(); o.regularizeShifts = true; o.numIterAtMinBin = 1
        o.alignmentBinValues = [1]
        #expect(throws: ParallaxError.degenerateData) { try ParallaxReconstruction.reconstruct(one, options: o) }
        // End to end: validation of the preprocessing is passed through.
        #expect(throws: ParallaxError.invalidInput) {
            try ParallaxReconstruction.process([], shape: shape, reciprocalSampling: .init(row: 1, column: 1),
                scanSampling: .init(row: 1, column: 1), energyEV: 200000)
        }
        var bad = ParallaxProcessingOptions(); bad.subpixel = ParallaxSubpixelOptions()
        #expect(throws: ParallaxError.invalidInput) {
            try ParallaxReconstruction.process(c.cube, shape: shape, reciprocalSampling: .init(row: 0.08, column: 0.08),
                scanSampling: .init(row: 0, column: 1), energyEV: 200000, options: bad)
        }
    }

    /// The end-to-end call is exactly the chain of the individual public steps.
    @Test func endToEndEqualsChainedSteps() throws {
        let p = try Self.pin().endToEnd
        let shape = try CubeShape(p.scan + p.detector)
        let reciprocal = ParallaxShift(row: 0.2, column: 0.2), scan = ParallaxShift(row: 1.25, column: 1.25)
        var options = ParallaxProcessingOptions()
        options.threshold = 0.5; options.edgeBlend = 2; options.paddingRows = 4; options.paddingColumns = 4
        options.reconstruction.regularizeShifts = true
        var sub = ParallaxSubpixelOptions()
        sub.factor = 2; sub.sigmaPixels = 0.3; sub.lanczosOrder = 2
        sub.positionCorrection = ParallaxPositionCorrectionOptions(iterations: 2)
        options.subpixel = sub
        let all = try ParallaxReconstruction.process(p.cube, shape: shape, reciprocalSampling: reciprocal,
            scanSampling: scan, energyEV: 200000, options: options)

        let prep = try ParallaxPreprocessing.prepare(p.cube, shape: shape, reciprocalSampling: reciprocal, energyEV: 200000,
            threshold: 0.5, edgeBlend: 2, paddingRows: 4, paddingColumns: 4)
        let recon = try ParallaxReconstruction.reconstruct(prep, options: options.reconstruction)
        let limits = try ParallaxSubpixel.upsampleLimits(scanSampling: 1.25,
            maxReciprocalVector: prep.reciprocalVectors.map { hypot($0.row, $0.column) }.max()!,
            reciprocalSampling: 0.2, detectorRows: 8, requested: 2)
        var pc = ParallaxPositionCorrectionOptions(iterations: 2)
        pc.checkerboard = false
        let corrected = try ParallaxSubpixel.correctPositions(images: prep.unshiftedStack, rows: prep.rows, columns: prep.columns,
            shifts: recon.shifts, window: prep.paddedWindow, scanSampling: scan, factor: limits.factor, sigmaPixels: 0.3,
            lanczosOrder: 2, options: pc)
        #expect(all.preprocessing.stack == prep.stack)
        #expect(all.reconstruction.shifts == recon.shifts)
        #expect(all.reconstruction.image == recon.image)
        #expect(all.reconstruction.alignedStack == recon.alignedStack)
        #expect(all.reconstruction.errors == recon.errors)
        #expect(all.limits == limits)
        #expect(all.subpixel?.image == corrected.image.image)
        #expect(all.positionCorrection?.offsets == corrected.offsets)

        // Without position correction the stage is the plain kernel density estimate; without a subpixel stage it is skipped.
        options.subpixel?.positionCorrection = nil
        let plain = try ParallaxReconstruction.process(p.cube, shape: shape, reciprocalSampling: reciprocal,
            scanSampling: scan, energyEV: 200000, options: options)
        let kde = try ParallaxSubpixel.kernelDensity(images: prep.unshiftedStack, rows: prep.rows, columns: prep.columns,
            shifts: recon.shifts, factor: 2, sigmaPixels: 0.3, lanczosOrder: 2)
        #expect(plain.subpixel?.image == kde.image && plain.positionCorrection == nil)
        options.subpixel = nil
        let none = try ParallaxReconstruction.process(p.cube, shape: shape, reciprocalSampling: reciprocal,
            scanSampling: scan, energyEV: 200000, options: options)
        #expect(none.subpixel == nil && none.limits == nil && none.reconstruction.image == recon.image)
    }
}
