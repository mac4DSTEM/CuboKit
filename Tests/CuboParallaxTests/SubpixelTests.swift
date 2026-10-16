import Foundation
import Testing
import CuboParallax

@Suite struct ParallaxSubpixelTests {
    struct Pin: Decodable {
        let rows, columns, count, detectorRows: Int
        let stack, shifts, kr: [Double]; let reciprocalSampling, scanSampling: Double; let cases: [Case]
    }
    struct Case: Decodable {
        let factor: Double?; let sigma: Double; let lanczos: Int?; let lowpass: Bool
        let usedFactor, bfLimit, dfLimit: Double; let shape: [Int]; let output: [Double]
    }
    @Test func pinnedKernelDensity() throws {
        let p = try JSONDecoder().decode(Pin.self, from: Data(contentsOf: Bundle.module.url(forResource: "subpixel", withExtension: "json", subdirectory: "Fixtures")!))
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
    @Test func invalidInput() {
        #expect(throws: ParallaxError.invalidInput) {
            try ParallaxSubpixel.kernelDensity(images: [], rows: 4, columns: 4, shifts: [], factor: 2)
        }
    }
}
