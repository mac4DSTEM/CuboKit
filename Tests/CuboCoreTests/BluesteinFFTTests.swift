import CuboCore
import Foundation
import Testing

/// CuboKit#9: `FloatFFT2D(anyRows:columns:)` takes lengths vDSP's DFT does not (the padded parallax sizes, 133 + 32 = 165) through
/// Bluestein. Its forward and inverse transforms are checked against the Double ``FFT2D`` (the float64 reference) at 1e-6 of the
/// largest coefficient (float32: rtol 1e-6), on the padded sizes and edge lengths (1, 2, primes, a DFT length beside a Bluestein
/// one); for lengths vDSP takes it gives the public plan's bits.
@Suite struct BluesteinFFTTests {
    private func signal(_ count: Int, seed: UInt64) -> [Double] {
        var state = seed
        return (0..<count).map { k in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return sin(Double(k) / 3) + Double(state >> 11) / Double(1 << 53) - 0.5
        }
    }

    /// The Float plan's natural-layout spectrum (forward) or unnormalised inverse of a real Float grid, back in Double.
    private func floatTransform(_ plan: FloatFFT2D, _ x: [Float], inverse: Bool) -> ([Double], [Double]) {
        let n = plan.rows * plan.columns
        var aRe = x, aIm = [Float](repeating: 0, count: n), bRe = aIm, bIm = aIm, cRe = aIm, cIm = aIm
        aRe.withUnsafeMutableBufferPointer { ar in aIm.withUnsafeMutableBufferPointer { ai in
        bRe.withUnsafeMutableBufferPointer { br in bIm.withUnsafeMutableBufferPointer { bi in
        cRe.withUnsafeMutableBufferPointer { cr in cIm.withUnsafeMutableBufferPointer { ci in
            if inverse {
                // the inverse reads a transposed spectrum: hand it the transposed real grid
                for r in 0..<plan.rows { for c in 0..<plan.columns { br[c * plan.rows + r] = ar[r * plan.columns + c] } }
                plan.inverse(aRe: br.baseAddress!, aIm: bi.baseAddress!, bRe: cr.baseAddress!, bIm: ci.baseAddress!)
            } else {
                plan.forward(aRe: ar.baseAddress!, aIm: ai.baseAddress!, bRe: br.baseAddress!, bIm: bi.baseAddress!)
                plan.naturalSpectrum(fromRe: br.baseAddress!, fromIm: bi.baseAddress!, toRe: cr.baseAddress!, toIm: ci.baseAddress!)
            }
        } } } } } }
        return (cRe.map(Double.init), cIm.map(Double.init))
    }

    @Test(arguments: [(165, 165), (132, 132), (197, 197), (165, 197), (160, 165), (1, 1), (1, 7), (2, 3), (7, 9), (3, 16), (17, 257)])
    func matchesTheDoubleTransform(rows: Int, columns: Int) {
        let x = signal(rows * columns, seed: UInt64(rows * 1000 + columns)).map { Double(Float($0)) }
        let plan = FloatFFT2D(anyRows: rows, columns: columns)
        let fft = FFT2D(rows: rows, columns: columns)
        for inverse in [false, true] {
            var re = x, im = [Double](repeating: 0, count: x.count)
            fft.transform(&re, &im, inverse: inverse)
            if inverse { re = re.map { $0 * Double(rows * columns) }; im = im.map { $0 * Double(rows * columns) } }
            let (fr, fi) = floatTransform(plan, x.map { Float($0) }, inverse: inverse)
            var worst = 0.0, scale = 0.0
            for k in re.indices {
                worst = max(worst, abs(fr[k] - re[k]), abs(fi[k] - im[k]))
                scale = max(scale, abs(re[k]), abs(im[k]))
            }
            #expect(worst <= 1e-6 * scale, "\(rows)x\(columns) inverse=\(inverse): \(worst) of \(scale)")
        }
    }

    @Test(arguments: [(160, 160), (64, 48), (16, 16), (96, 64)])
    func supportedSizesKeepThePublicPlansBits(rows: Int, columns: Int) throws {
        let x = signal(rows * columns, seed: 3).map { Float($0) }
        let any = FloatFFT2D(anyRows: rows, columns: columns)
        let direct = try #require(FloatFFT2D(rows: rows, columns: columns))
        for inverse in [false, true] {
            let a = floatTransform(any, x, inverse: inverse), b = floatTransform(direct, x, inverse: inverse)
            #expect(a.0 == b.0 && a.1 == b.1)
        }
        #expect(!FloatFFT2D.supports(rows: 165, columns: 165))
    }
}
