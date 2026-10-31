import CuboPtycho
import Foundation
import Testing

/// Timing of single-slice ptychography on detector sizes the vDSP DFT does not take (CuboKit#9). Skipped unless
/// `CUBOKIT_PTYCHO_BENCH` is set: `CUBOKIT_PTYCHO_BENCH=1 swift test -c release --filter PtychoSingleBench`. Prints the
/// median and best of five runs of 2 iterations over 49 positions per size, Double and Float32, after one warm-up;
/// 160 x 160 is the vDSP DFT size beside the Bluestein ones.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["CUBOKIT_PTYCHO_BENCH"] != nil, "set CUBOKIT_PTYCHO_BENCH"))
struct PtychoSingleBench {
    @Test(arguments: [(165, 165), (132, 132), (160, 160)])
    func twoIterations(rows: Int, columns: Int) throws {
        let n = rows * columns, size = 2 * max(rows, columns) + 64, count = 49
        var probeRe = [Double](repeating: 0, count: n)
        for r in 0..<rows { for c in 0..<columns {
            let rr = Double(min(r, rows - r)), cc = Double(min(c, columns - c)); probeRe[r * columns + c] = exp(-(rr * rr + cc * cc) / 400)
        } }
        let probe = try PtychographyArray(rows: rows, columns: columns, real: probeRe, imaginary: [Double](repeating: 0, count: n))
        let object = try PtychographyArray(rows: size, columns: size, real: [Double](repeating: 1, count: size * size),
                                           imaginary: [Double](repeating: 0, count: size * size))
        let positions = (0..<count).map { PtychographyPosition(row: 40 + Double($0 / 7) * 9.3, column: 40 + Double($0 % 7) * 8.7) }
        let amplitudes = (0..<count * n).map { k in 0.5 + 0.4 * sin(Double(k % 977) / 31) }
        func time(_ precision: PtychographyPrecision) throws -> (median: Double, best: Double, result: SingleslicePtychography.Result) {
            var result = try SingleslicePtychography.reconstruct(amplitudes: amplitudes, positions: positions, object: object,
                                                                 probe: probe, iterations: 2, precision: precision)
            var times: [Double] = []
            for _ in 0..<5 {
                let start = DispatchTime.now().uptimeNanoseconds
                result = try SingleslicePtychography.reconstruct(amplitudes: amplitudes, positions: positions, object: object,
                                                                 probe: probe, iterations: 2, precision: precision)
                times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
            }
            times.sort()
            return (times[2], times[0], result)
        }
        // timing only: the amplitudes are not a forward model, so the two paths are not compared here
        // (SinglePrecisionNonSquareTests checks the agreement on Bluestein detectors, 165 x 40 among them)
        let double = try time(.double), single = try time(.single)
        print(String(format: "PTYCHOBENCH %dx%d 2 iterations x 49: Double median %.1f ms best %.1f ms | single median %.1f ms best %.1f ms | errors %.4f %.4f",
                     rows, columns, double.median, double.best, single.median, single.best,
                     double.result.errors.last!, single.result.errors.last!))
    }
}
