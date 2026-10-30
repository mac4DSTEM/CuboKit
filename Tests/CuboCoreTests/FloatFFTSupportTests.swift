import Testing
import CuboCore

// `FloatFFT2D.supports(rows:columns:)` is public: a caller asks before it picks a Float32 path. vDSP's DFT takes f · 2^n
// lengths (f in 1, 3, 5, 15); a prime like 7 has none.
@Suite struct FloatFFTSupportTests {
    @Test func supportsAnswersForVDSPLengths() {
        #expect(FloatFFT2D.supports(rows: 64, columns: 48))
        #expect(FloatFFT2D.supports(rows: 120, columns: 80))
        #expect(!FloatFFT2D.supports(rows: 7, columns: 64))
        #expect(!FloatFFT2D.supports(rows: 0, columns: 64))
        let plan = FloatFFT2D(rows: 64, columns: 48)
        #expect(plan?.rows == 64 && plan?.columns == 48)
    }
}
