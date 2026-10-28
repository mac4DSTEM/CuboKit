import Foundation
import Testing
import CuboParallax

/// 0.7.0: the Double FFT runs on vDSP_DFT_zop where vDSP takes the length (checked here against a naive DFT, with a length it
/// does not take beside them), and the Float32 shifter agrees with the Double one (the float32 tolerance: 1e-5 of the image range, written
/// per test).
@Suite struct DFTAndSingleShifterTests {
    private func image(_ rows: Int, _ columns: Int) -> [Double] {
        var state: UInt64 = 0x9E3779B97F4A7C15
        return (0..<rows*columns).map { k in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let noise = Double(state >> 40) / Double(1 << 24)
            return 3 * sin(Double(k % columns) / 5) * cos(Double(k / columns) / 7) + noise
        }
    }

    @Test(arguments: [(16, 16), (48, 32), (24, 40), (30, 50), (7, 9), (8, 8), (4, 16), (16, 8)])
    func spectrumMatchesNaiveDFT(rows: Int, columns: Int) throws {
        let data = image(rows, columns)
        let shifter = try ParallaxFourierShifter(rows: rows, columns: columns)
        let spectrum = try shifter.spectrum(of: data)
        var worst = 0.0, scale = 0.0
        for u in 0..<rows { for v in 0..<columns {
            var re = 0.0, im = 0.0
            for r in 0..<rows { for c in 0..<columns {
                let angle = -2 * Double.pi * (Double(u * r) / Double(rows) + Double(v * c) / Double(columns))
                re += data[r * columns + c] * cos(angle); im += data[r * columns + c] * sin(angle)
            } }
            worst = max(worst, abs(re - spectrum.real[u * columns + v]), abs(im - spectrum.imaginary[u * columns + v]))
            scale = max(scale, abs(re), abs(im))
        } }
        // float64: 1e-12 of the largest coefficient (naive summation error dominates)
        #expect(worst <= 1e-12 * scale, "\(rows)x\(columns): worst \(worst) of \(scale)")
    }

    @Test(arguments: [(16, 16), (48, 32), (64, 64), (160, 160)])
    func inverseRoundTrips(rows: Int, columns: Int) throws {
        let data = image(rows, columns)
        let shifter = try ParallaxFourierShifter(rows: rows, columns: columns)
        let back = try shifter.inverse(try shifter.spectrum(of: data))
        let worst = zip(data, back).map { abs($0 - $1) }.max()!
        #expect(worst <= 1e-12, "\(rows)x\(columns): \(worst)")
    }

    @Test func singleShifterAgreesWithDouble() throws {
        let rows = 64, columns = 96
        let data = image(rows, columns)
        let shifter = try ParallaxFourierShifter(rows: rows, columns: columns)
        #expect(shifter.supportsSingle)
        let range = data.max()! - data.min()!
        for shift in [ParallaxShift(row: 1.37, column: -2.21), ParallaxShift(row: 0, column: 0), ParallaxShift(row: -31.5, column: 47.9)] {
            let expected = try shifter.shifted(data, by: shift)
            let actual = try shifter.shiftedSingle(data.map { Float($0) }, by: shift)
            let worst = zip(expected, actual).map { abs($0 - Double($1)) }.max()!
            #expect(worst <= 1e-5 * range, "shift \(shift): \(worst) of range \(range)")
        }
    }

    @Test func singleShifterRefusesUnsupportedSizesAndBadInput() throws {
        let odd = try ParallaxFourierShifter(rows: 30, columns: 50)
        #expect(!odd.supportsSingle)
        #expect(throws: ParallaxAlignmentError.invalidDimensions) { try odd.shiftedSingle([Float](repeating: 0, count: 1500), by: ParallaxShift(row: 1, column: 1)) }
        let fine = try ParallaxFourierShifter(rows: 16, columns: 16)
        #expect(throws: ParallaxAlignmentError.inconsistentShape) { try fine.shiftedSingle([Float](repeating: 0, count: 10), by: ParallaxShift(row: 1, column: 1)) }
        #expect(throws: ParallaxAlignmentError.nonfiniteValues) { try fine.shiftedSingle([Float](repeating: 0, count: 256), by: ParallaxShift(row: .nan, column: 1)) }
    }

    @Test func singleBatchCancelsAndReportsProgress() throws {
        let shifter = try ParallaxFourierShifter(rows: 16, columns: 16)
        let images = (0..<4).map { _ in image(16, 16).map { Float($0) } }
        let shifts = (0..<4).map { ParallaxShift(row: Double($0) * 0.5, column: 0.25) }
        nonisolated(unsafe) var seen = [Int]()
        let out = try shifter.shiftedSingle(images, by: shifts, progress: { done, total in #expect(total == 4); seen.append(done) })
        #expect(out.count == 4 && seen == [1, 2, 3, 4])
        nonisolated(unsafe) var polls = 0
        #expect(throws: CancellationError.self) { try shifter.shiftedSingle(images, by: shifts, isCancelled: { polls += 1; return polls > 2 }) }
    }
}

/// The plans are shared by threads (Bragg's concurrentPerform, the engines): eight threads on one shifter agree with the serial
/// result exactly, Double and Float32.
@Suite struct SharedPlanConcurrencyTests {
    @Test func eightThreadsOnePlan() async throws {
        let rows = 48, columns = 64
        let shifter = try ParallaxFourierShifter(rows: rows, columns: columns)
        let images = (0..<8).map { t in (0..<rows * columns).map { k in sin(Double(k * (t + 3)) / 17) } }
        let serial = try images.map { try shifter.shifted($0, by: ParallaxShift(row: 1.3, column: -0.7)) }
        let serialSingle = try images.map { try shifter.shiftedSingle($0.map { Float($0) }, by: ParallaxShift(row: 1.3, column: -0.7)) }
        nonisolated(unsafe) var parallel = [[Double]](repeating: [], count: 8), parallelSingle = [[Float]](repeating: [], count: 8)
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: 64) { n in
            let t = n % 8
            let a = try! shifter.shifted(images[t], by: ParallaxShift(row: 1.3, column: -0.7))
            let b = try! shifter.shiftedSingle(images[t].map { Float($0) }, by: ParallaxShift(row: 1.3, column: -0.7))
            lock.lock(); parallel[t] = a; parallelSingle[t] = b; lock.unlock()
        }
        #expect(parallel == serial && parallelSingle == serialSingle)
    }
}
