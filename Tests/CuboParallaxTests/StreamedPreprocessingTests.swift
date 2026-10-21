import Foundation
import Testing
import CuboCore
import CuboParallax

/// Streamed preprocessing against the resident Double `prepare`: Float32 rounding only, relative 2e-5 (D12, Float32).
@Suite struct StreamedPreprocessingTests {
    static let tolerance = 2e-5
    static let sampling = ParallaxShift(row: 0.05, column: 0.05)

    /// A 9 x 10 scan of 6 x 6 patterns: a bright disk (4 x 4 centre block) with varying intensity, dim elsewhere.
    static func cube() throws -> (MemoryCube, [Double]) {
        let shape = try CubeShape([9, 10, 6, 6])
        var state: UInt64 = 12345
        var values = [Float](); values.reserveCapacity(shape.elementCount)
        for p in 0..<shape.scanCount { for q in 0..<36 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let noise = Float(state >> 40) / Float(1 << 24)
            let r = q / 6, c = q % 6
            let bright = (1...4).contains(r) && (1...4).contains(c)
            values.append((bright ? 10 + 3 * noise + Float(p % 7) * 0.2 : 0.5 * noise))
        } }
        return (try MemoryCube(shape: shape, values: values), values.map(Double.init))
    }

    func close(_ a: [Float], _ b: [Double], _ label: String) {
        #expect(a.count == b.count, "\(label) count")
        var worst = 0.0
        for (x, y) in zip(a, b) { worst = max(worst, abs(Double(x) - y) / max(1, abs(y))) }
        #expect(worst <= Self.tolerance, "\(label) worst \(worst)")
    }

    @Test func matchesResidentPrepare() async throws {
        let (cube, doubles) = try Self.cube()
        for retention in [ParallaxStackRetention.windowed, .unwindowed] {
            let s = try await ParallaxPreprocessing.prepare(source: cube, reciprocalSampling: Self.sampling,
                energyEV: 200_000, threshold: 0.65, retention: retention)
            let r = try ParallaxPreprocessing.prepare(doubles, shape: cube.shape, reciprocalSampling: Self.sampling,
                energyEV: 200_000, threshold: 0.65, retention: retention)
            #expect(s.detectorIndices == r.detectorIndices)
            #expect(s.rows == r.rows && s.columns == r.columns)
            close(s.window, r.window, "window")
            close(s.stack, r.stack.flatMap { $0 }, "stack")
            close(s.unshiftedStack, r.unshiftedStack.flatMap { $0 }, "unshifted")
            close(s.incoherentImage, r.incoherentImage, "incoherent")
            close([s.initialError], [r.initialError], "error")
            close(s.reciprocalVectors.flatMap { [$0.row, $0.column] }.map(Float.init),
                  r.reciprocalVectors.flatMap { [$0.row, $0.column] }, "vectors")
            close(s.probeAnglesMrad.flatMap { [$0.row, $0.column] }.map(Float.init),
                  r.probeAngles.flatMap { [$0.row * 1000, $0.column * 1000] }, "angles")
            #expect(s.detectorMask.filter { $0 }.count == s.brightFieldCount)
        }
    }

    @Test func tileSizesGiveIdenticalResults() async throws {
        let (cube, _) = try Self.cube()
        let one = try await ParallaxPreprocessing.prepare(source: cube, reciprocalSampling: Self.sampling,
            energyEV: 200_000, threshold: 0.65, retention: .unwindowed, tiling: .rows(1))
        for tiling in [ScanTiling.rows(4), .rows(9), .automatic] {
            let other = try await ParallaxPreprocessing.prepare(source: cube, reciprocalSampling: Self.sampling,
                energyEV: 200_000, threshold: 0.65, retention: .unwindowed, tiling: tiling)
            #expect(other.stack == one.stack && other.unshiftedStack == one.unshiftedStack)
            #expect(other.incoherentImage == one.incoherentImage && other.initialError == one.initialError)
            #expect(other.detectorMask == one.detectorMask)
        }
    }

    @Test func explicitMaskSkipsPassOne() async throws {
        let (cube, doubles) = try Self.cube()
        var mask = [Bool](repeating: false, count: 36); mask[7] = true; mask[8] = true; mask[14] = true
        let fractions = Fractions()
        let s = try await ParallaxPreprocessing.prepare(source: cube, reciprocalSampling: Self.sampling,
            energyEV: 200_000, detectorMask: mask, tiling: .rows(2), progress: { fractions.add($0) })
        let r = try ParallaxPreprocessing.prepare(doubles, shape: cube.shape, reciprocalSampling: Self.sampling,
            energyEV: 200_000, detectorMask: mask)
        #expect(s.detectorMask == mask && s.detectorIndices == r.detectorIndices)
        close(s.stack, r.stack.flatMap { $0 }, "stack")
        let f = fractions.values
        #expect(f.last == 1 && f.sorted() == f && f.count == 5 + 1)
    }

    @Test func refusesLargeStacksAndInvalidInput() async throws {
        let (cube, _) = try Self.cube()
        let planes = (9 + 32) * (10 + 32)
        // windowed holds one stack, unwindowed two; bright field is the 16-pixel block at threshold 0.65
        let one = planes * 16 * 4
        await #expect(throws: ParallaxError.stackTooLarge(bytes: 2 * one, limit: 2 * one - 1)) {
            try await ParallaxPreprocessing.prepare(source: cube, reciprocalSampling: Self.sampling,
                energyEV: 200_000, threshold: 0.65, retention: .unwindowed, maximumStackBytes: 2 * one - 1)
        }
        _ = try await ParallaxPreprocessing.prepare(source: cube, reciprocalSampling: Self.sampling,
            energyEV: 200_000, threshold: 0.65, retention: .windowed, maximumStackBytes: one)
        await #expect(throws: ParallaxError.invalidInput) {
            try await ParallaxPreprocessing.prepare(source: cube, reciprocalSampling: Self.sampling,
                energyEV: 200_000, threshold: 0, maximumStackBytes: nil)
        }
        await #expect(throws: ParallaxError.invalidInput) {
            try await ParallaxPreprocessing.prepare(source: cube, reciprocalSampling: Self.sampling,
                energyEV: 200_000, detectorMask: [true])
        }
    }

    @Test func nonFiniteOnlyMattersInsideBrightField() async throws {
        let (cube, _) = try Self.cube()
        var values = cube.values
        values[3] = .nan  // pattern 0, detector pixel 3 (row 0): outside the bright field
        let outside = try MemoryCube(shape: cube.shape, values: values)
        let ok = try await ParallaxPreprocessing.prepare(source: outside, reciprocalSampling: Self.sampling,
            energyEV: 200_000, threshold: 0.65)
        #expect(ok.initialError.isFinite)
        values[7] = .infinity  // detector pixel 7 = row 1, column 1: inside the mask below
        // A non-finite mean drops the pixel from the threshold selection (as in the app), so force it in with a mask.
        var mask = [Bool](repeating: false, count: 36); mask[7] = true; mask[8] = true
        let inside = try MemoryCube(shape: cube.shape, values: values)
        await #expect(throws: ParallaxError.nonfiniteValues) {
            try await ParallaxPreprocessing.prepare(source: inside, reciprocalSampling: Self.sampling,
                energyEV: 200_000, detectorMask: mask)
        }
        let dropped = try await ParallaxPreprocessing.prepare(source: inside, reciprocalSampling: Self.sampling,
            energyEV: 200_000, threshold: 0.65)
        #expect(!dropped.detectorMask[7] && dropped.initialError.isFinite)
    }

    @Test func cancellation() async throws {
        let (cube, _) = try Self.cube()
        await #expect(throws: CancellationError.self) {
            try await ParallaxPreprocessing.prepare(source: cube, reciprocalSampling: Self.sampling,
                energyEV: 200_000, isCancelled: { true })
        }
        let polls = Fractions()
        await #expect(throws: CancellationError.self) {
            try await ParallaxPreprocessing.prepare(source: cube, reciprocalSampling: Self.sampling,
                energyEV: 200_000, threshold: 0.65, tiling: .rows(1), isCancelled: { polls.add(0); return polls.values.count > 12 })
        }
    }
}

final class Fractions: @unchecked Sendable {
    private let lock = NSLock(); private var stored = [Double]()
    func add(_ v: Double) { lock.lock(); stored.append(v); lock.unlock() }
    var values: [Double] { lock.lock(); defer { lock.unlock() }; return stored }
}
