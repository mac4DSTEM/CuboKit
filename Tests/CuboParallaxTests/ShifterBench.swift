import CuboCore
import CuboParallax
import Foundation
import Testing

/// Timing of the Fourier shifter on padded parallax sizes (CuboKit#9). Skipped unless `CUBOKIT_SHIFT_BENCH` is set:
/// `CUBOKIT_SHIFT_BENCH=1 swift test -c release --filter ShifterBench`. Prints the median and best of seven runs of a
/// 49-image batch per size, after one warm-up; 160 x 160 is the vDSP DFT size beside the Bluestein ones.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["CUBOKIT_SHIFT_BENCH"] != nil, "set CUBOKIT_SHIFT_BENCH"))
struct ShifterBench {
    private func image(_ rows: Int, _ columns: Int, seed: Int) -> [Double] {
        (0..<rows * columns).map { k in sin(Double(k % columns + seed) / 5) * cos(Double(k / columns) / 7) + Double(k % 13) / 13 }
    }

    @Test(arguments: [(165, 165), (132, 132), (197, 197), (165, 197), (160, 160)])
    func batchOf49(rows: Int, columns: Int) throws {
        let shifter = try ParallaxFourierShifter(rows: rows, columns: columns)
        let images = (0..<49).map { image(rows, columns, seed: $0) }
        let shifts = (0..<49).map { ParallaxShift(row: Double($0 % 7) * 0.37 - 1.1, column: Double($0 / 7) * 0.41 - 1.3) }
        _ = try shifter.shifted(images, by: shifts)
        var times: [Double] = []
        var checksum = 0.0
        for _ in 0..<7 {
            let start = DispatchTime.now().uptimeNanoseconds
            let out = try shifter.shifted(images, by: shifts)
            times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
            checksum = out[17][rows * columns / 3]
        }
        times.sort()
        var fftTimes: [Double] = []
        let fft = FFT2D(rows: rows, columns: columns)
        var re = images[0], im = [Double](repeating: 0, count: rows * columns)
        for _ in 0..<7 {
            let start = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<49 { fft.transform(&re, &im, inverse: false); fft.transform(&re, &im, inverse: true) }
            fftTimes.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
        }
        fftTimes.sort()
        var singleTimes: [Double] = []
        var worst = 0.0
        if shifter.supportsSingle {
            let floats = images.map { $0.map { Float($0) } }
            let reference = try shifter.shifted(images, by: shifts), single = try shifter.shiftedSingle(floats, by: shifts)
            for (a, b) in zip(reference, single) {
                let range = a.max()! - a.min()!
                worst = max(worst, zip(a, b).map { abs($0 - Double($1)) }.max()! / range)
            }
            for _ in 0..<7 {
                let start = DispatchTime.now().uptimeNanoseconds
                _ = try shifter.shiftedSingle(floats, by: shifts)
                singleTimes.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
            }
            singleTimes.sort()
        }
        let single = singleTimes.isEmpty ? "unsupported"
            : String(format: "median %.1f ms best %.1f ms, worst gap to Double %.2e of range", singleTimes[3], singleTimes[0], worst)
        print(String(format: "SHIFTBENCH %dx%d shifted(49) Double: median %.1f ms best %.1f ms | shiftedSingle(49): %@ | fft2 fwd+inv x49: median %.1f ms best %.1f ms | check %.6f",
                     rows, columns, times[3], times[0], single, fftTimes[3], fftTimes[0], checksum))
    }
}
