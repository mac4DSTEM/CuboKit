import Foundation
import Testing
import CuboParallax

@Suite struct ParallaxSubpixelTests {
    struct Pin: Decodable {
        let rows, columns, count, detectorRows: Int
        let stack, shifts, kr, window: [Double]; let positionCases: [PosCase]; let reciprocalSampling, scanSampling: Double; let cases: [Case]
    }
    struct PosCase: Decodable { let dx, dy, output: [Double] }
    struct Case: Decodable {
        let factor: Double?; let sigma: Double; let lanczos: Int?; let lowpass: Bool
        let usedFactor, bfLimit, dfLimit: Double; let shape: [Int]; let output: [Double]
    }
    @Test func pinnedKernelDensity() throws {
        let p = try JSONDecoder().decode(Pin.self, from: Data(contentsOf: parallaxFixture("subpixel")!))
        let n = p.rows * p.columns
        let images = (0..<p.count).map { Array(p.stack[$0 * n..<($0 + 1) * n]) }
        let shifts = stride(from: 0, to: p.shifts.count, by: 2).map { ParallaxShift(row: p.shifts[$0], column: p.shifts[$0 + 1]) }
        for c in p.cases {
            let limits = try ParallaxSubpixel.upsampleLimits(scanSampling: p.scanSampling, maxReciprocalVector: p.kr.max()!,
                reciprocalSampling: p.reciprocalSampling, detectorRows: p.detectorRows, requested: c.factor)
            #expect(abs(limits.brightField - c.bfLimit) < 1e-6 && abs(limits.darkField - c.dfLimit) < 1e-6 && abs(limits.factor - c.usedFactor) < 1e-6)
            let r = try ParallaxSubpixel.kernelDensity(images: images, rows: p.rows, columns: p.columns, shifts: shifts,
                factor: limits.factor, sigmaPixels: c.sigma, lanczosOrder: c.lanczos, lowpass: c.lowpass)
            #expect([r.rows, r.columns] == c.shape)
            let gap = zip(r.image, c.output).map { abs($0 - $1) / max(1, abs($1)) }.max() ?? 0
            print("subpixel gap \(gap)")
            #expect(gap < 1e-4) // reference accumulates in Float32; Double here.
        }
    }
    @Test func pinnedPositionCorrection() throws {
        let p = try JSONDecoder().decode(Pin.self, from: Data(contentsOf: parallaxFixture("subpixel")!))
        let n = p.rows * p.columns
        let images = (0..<p.count).map { Array(p.stack[$0 * n..<($0 + 1) * n]) }
        let shifts = stride(from: 0, to: p.shifts.count, by: 2).map { ParallaxShift(row: p.shifts[$0], column: p.shifts[$0 + 1]) }
        for (k, c) in p.positionCases.enumerated() {
            var o = ParallaxPositionCorrectionOptions(iterations: 3)
            o.checkerboard = k == 1
            if k == 2 {
                o.gaussianSigma = .init(row: 1.5, column: 1.5); o.lowpass = .init(row: 0.5, column: 0.5); o.highpass = .init(row: 0.02, column: 0.02)
            }
            let r = try ParallaxSubpixel.correctPositions(images: images, rows: p.rows, columns: p.columns, shifts: shifts, window: p.window,
                scanSampling: .init(row: 1.25, column: 1.25), factor: 2, sigmaPixels: 0.3, options: o)
            let g1 = zip(r.offsets.rows + r.offsets.columns, c.dx + c.dy).map { abs($0 - $1) }.max() ?? 1
            let g2 = zip(r.image.image, c.output).map { abs($0 - $1) / max(1, abs($1)) }.max() ?? 1
            print("position gap \(k) offsets \(g1) image \(g2)")
            #expect(r.scores.count == 4)
            #expect(g1 < 1e-4 && g2 < 1e-4) // reference runs Float32; tie-breaking could flip a step.
        }
    }
    @Test func invalidInput() {
        #expect(throws: ParallaxError.invalidInput) {
            try ParallaxSubpixel.kernelDensity(images: [], rows: 4, columns: 4, shifts: [], factor: 2)
        }
    }
}
