//
//  FourierResampleTests.swift
//  `resample_Q(method="fourier")` and `fourier_resample` (task 18) against py4DSTEM 0.14.19 (tools/polar-ellipse/pins.py) and
//  against the exact band-limited truth of tools/polar-ellipse/planted.py. py4DSTEM has no test for them. Its transform is
//  single precision (NumPy 2 `fft2` of Float32 is complex64), so the pinned comparison holds the kit to single-precision
//  agreement, relative 1e-6 of the largest value of a case (the rule of the other resample rows); the exact truth, where the
//  transform is exact in Double, is held to 1e-10.
//

import Foundation
import Testing
import CuboCore

private struct Args: Decodable {
    let scale: Double?, output_size: [Int]?, conserve_array_sums: Bool?
    let bandlimit_nyquist: Double?, bandlimit_power: Double?, force_nonnegative: Bool?
}
private struct Case: Decodable { let name: String, args: Args, shape: [Int], values: [Double] }
private struct TwoD: Decodable { let input: [Double], shape: [Int], outputShape: [Int], values: [Double] }
private struct Rdd: Decodable { let qPixelSize: Double, shape: [Int] }
private struct RddPins: Decodable { let factor2: Rdd, size15x9: Rdd }
private struct Resample: Decodable {
    let inputShape: [Int], input: [Double], cases: [Case]
    let oddInputShape: [Int], oddInput: [Double], odd: [Case]
    let twoD: TwoD, resampleDataDiffraction: RddPins
}
private struct Pin: Decodable { let resample: Resample }

private let pin: Resample = {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/ellipse-resample-pins.json")
    do { return try JSONDecoder().decode(Pin.self, from: Data(contentsOf: url)).resample } catch { fatalError("ellipse-resample-pins.json: \(error)") }
}()

private func cube(_ shape: [Int], _ values: [Double]) throws -> MemoryCube {
    try MemoryCube(shape: try CubeShape(shape), values: values.map { Float($0) })
}
private func spec(_ a: Args) -> FourierResampleSpecification {
    if let s = a.scale { return .scale(s) }
    return .outputSize(rows: a.output_size![0], columns: a.output_size![1])
}
private func options(_ a: Args) -> FourierResampleOptions {
    FourierResampleOptions(bandlimitNyquist: a.bandlimit_nyquist, bandlimitPower: a.bandlimit_power ?? 2,
                           forceNonnegative: a.force_nonnegative ?? false, conserveSums: a.conserve_array_sums ?? false)
}
/// The relative-to-the-largest-value distance of two arrays.
private func distance(_ got: [Float], _ want: [Double]) -> Double {
    let scale = max(want.map(abs).max() ?? 1, 1e-300)
    return zip(got, want).map { abs(Double($0) - $1) / scale }.max() ?? 0
}

@Suite struct FourierResampleTests {
    @Test func testCubesAgreeWithPy4DSTEM() async throws {
        let source = try cube(pin.inputShape, pin.input)
        var worst = 0.0
        for c in pin.cases {
            let edited = try await CubeEditing.resampleDetectorFourier(of: source, spec(c.args), options: options(c.args))
            #expect(edited.cube.shape.extents == c.shape, "\(c.name) shape")
            let d = distance(edited.cube.values, c.values)
            worst = max(worst, d)
            #expect(d <= 1e-6, "\(c.name): \(d) of the largest value")
        }
        for c in pin.odd {   // odd extents in and out
            let edited = try await CubeEditing.resampleDetectorFourier(of: try cube(pin.oddInputShape, pin.oddInput), spec(c.args), options: options(c.args))
            #expect(edited.cube.shape.extents == c.shape, "\(c.name) shape")
            let d = distance(edited.cube.values, c.values)
            worst = max(worst, d)
            #expect(d <= 1e-6, "\(c.name): \(d) of the largest value")
        }
        print("fourier_resample vs py4DSTEM (single precision): worst difference \(worst) of the largest value")
    }

    @Test func testTwoDimensionalReferenceAgreesWithPy4DSTEM() throws {
        let t = pin.twoD
        let got = try CubeEditing.fourierResample(t.input, rows: t.shape[0], columns: t.shape[1], outputRows: t.outputShape[0], outputColumns: t.outputShape[1])
        #expect(got.count == t.values.count)
        let scale = t.values.map(abs).max()!
        let d = zip(got, t.values).map { abs($0 - $1) / scale }.max()!
        print("fourierResample (Double) vs py4DSTEM 2-D: \(d) of the largest value")
        #expect(d <= 1e-6)
    }

    /// `resample_data_diffraction(method="fourier")` updates the pixel size by the scale itself, or by rows out / rows in.
    @Test func testPixelSizeFollowsResampleDataDiffraction() async throws {
        let source = try cube(pin.inputShape, pin.input)
        var cal = Calibration(); cal.detectorPixelSize = 0.05
        let a = try await CubeEditing.resampleDetectorFourier(of: source, .scale(2), calibration: cal)
        #expect(a.calibration.detectorPixelSize == pin.resampleDataDiffraction.factor2.qPixelSize)
        #expect(a.cube.shape.extents == pin.resampleDataDiffraction.factor2.shape)
        let b = try await CubeEditing.resampleDetectorFourier(of: source, .outputSize(rows: 15, columns: 9), calibration: cal)
        #expect(abs(b.calibration.detectorPixelSize - pin.resampleDataDiffraction.size15x9.qPixelSize) < 1e-17)
        // a scale that truncates the extent still divides the pixel size by the scale (py4DSTEM's resampling_factor)
        let odd = try cube(pin.oddInputShape, pin.oddInput)
        let c = try await CubeEditing.resampleDetectorFourier(of: odd, .scale(0.5), calibration: cal)
        #expect(c.cube.shape.extents == [1, 2, 4, 5] && c.calibration.detectorPixelSize == 0.1)
    }

    // MARK: the exact band-limited truth of planted.py

    private static let modes: [(kr: Double, kc: Double, amp: Double, phase: Double)] =
        [(1, 0, 3.0, 0.3), (0, 2, 2.0, 1.1), (3, -2, 1.5, -0.7), (-4, 5, 1.0, 2.0), (6, 6, 0.7, 0.4)]
    private func bandLimited(_ n0: Int, _ n1: Int) -> [Double] {
        var out = [Double](repeating: 2, count: n0 * n1)
        for i in 0..<n0 { for j in 0..<n1 {
            let r = Double(i) / Double(n0), c = Double(j) / Double(n1)
            for m in Self.modes { out[i * n1 + j] += m.amp * cos(2 * .pi * (m.kr * r + m.kc * c) + m.phase) }
        } }
        return out
    }

    /// Every mode of the planted signal is below the Nyquist frequency of every size, so zero-padding or cropping the spectrum
    /// reproduces the continuous signal at the new sampling: atol 1e-10, for up, down and mixed resizing.
    @Test func testRecoversTheBandLimitedTruth() throws {
        for (from, to) in [((32, 32), (48, 48)), ((48, 48), (32, 32)), ((40, 24), (64, 20))] {
            let got = try CubeEditing.fourierResample(bandLimited(from.0, from.1), rows: from.0, columns: from.1, outputRows: to.0, outputColumns: to.1)
            let want = bandLimited(to.0, to.1)
            let error = zip(got, want).map { abs($0 - $1) }.max()!
            print("fourier resample \(from) to \(to): max error against the continuous signal \(error)")
            #expect(error < 1e-10, "\(from) to \(to)")
        }
    }

    // MARK: behaviours

    @Test func testConservedSumsAndNonnegativeAndIdentity() async throws {
        let source = try cube(pin.inputShape, pin.input)
        let up = try await CubeEditing.resampleDetectorFourier(of: source, .scale(2), options: FourierResampleOptions(conserveSums: true))
        let before = pin.input.prefix(120).reduce(0, +)           // pattern (0, 0): 10 x 12
        let after = up.cube.values.prefix(480).reduce(0) { $0 + Double($1) }
        #expect(abs(before - after) < 1e-4 * abs(before), "\(before) vs \(after)")
        let clamped = try await CubeEditing.resampleDetectorFourier(of: source, .outputSize(rows: 16, columns: 18), options: FourierResampleOptions(forceNonnegative: true))
        #expect(clamped.cube.values.allSatisfy { $0 >= 0 })
        let same = try await CubeEditing.resampleDetectorFourier(of: source, .outputSize(rows: 10, columns: 12))
        #expect(distance(same.cube.values, pin.input) < 1e-6)
    }

    @Test func testRefusalsCancellationAndProgress() async throws {
        let source = try cube(pin.inputShape, pin.input)
        await #expect(throws: CuboError.self) { _ = try await CubeEditing.resampleDetectorFourier(of: source, .scale(0)) }
        await #expect(throws: CuboError.self) { _ = try await CubeEditing.resampleDetectorFourier(of: source, .scale(0.05)) }
        await #expect(throws: CuboError.self) { _ = try await CubeEditing.resampleDetectorFourier(of: source, .outputSize(rows: 0, columns: 4)) }
        await #expect(throws: CuboError.self) {
            _ = try await CubeEditing.resampleDetectorFourier(of: source, .scale(2), options: FourierResampleOptions(bandlimitNyquist: 0))
        }
        await #expect(throws: CancellationError.self) {
            _ = try await CubeEditing.resampleDetectorFourier(of: source, .scale(2), isCancelled: { true })
        }
        #expect(throws: CuboError.self) { _ = try CubeEditing.fourierResample([1, 2, 3], rows: 2, columns: 2, outputRows: 4, outputColumns: 4) }
        let done = Progress()
        _ = try await CubeEditing.resampleDetectorFourier(of: source, .scale(2), progress: { done.set($0) })
        #expect(done.value == 1)
    }

    private final class Progress: @unchecked Sendable {
        private let lock = NSLock()
        private var v = 0.0
        func set(_ x: Double) { lock.lock(); v = x; lock.unlock() }
        var value: Double { lock.lock(); defer { lock.unlock() }; return v }
    }
}
