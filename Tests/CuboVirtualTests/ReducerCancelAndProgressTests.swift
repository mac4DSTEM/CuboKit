import Foundation
import Testing
import CuboKitTestSupport
import CuboCore
import CuboVirtual

// 0.9.0: the whole-cube reducers take `isCancelled:` (asked before each tile; a true answer throws CancellationError and
// returns nothing) and report `progress:` (the fraction of scan rows done, never decreasing, ending at 1), in the style of
// RotationSolver.curve. Defaults keep every earlier call as it was.
private final class Log: @unchecked Sendable {
    private let lock = NSLock()
    private var asked = 0
    private var fractions: [Double] = []
    func ask() -> Int { lock.lock(); defer { lock.unlock() }; asked += 1; return asked }
    func report(_ f: Double) { lock.lock(); fractions.append(f); lock.unlock() }
    var questions: Int { lock.lock(); defer { lock.unlock() }; return asked }
    var seen: [Double] { lock.lock(); defer { lock.unlock() }; return fractions }
}

@Suite struct ReducerCancelAndProgressTests {
    static let rows = 24

    private func cube() throws -> MemoryCube {
        let shape = try CubeShape(scanRows: Self.rows, scanColumns: 4, detectorRows: 8, detectorColumns: 8)
        var state: UInt64 = 5
        let v = (0..<shape.elementCount).map { _ -> Float in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Double(state >> 11) / Double(1 << 53)) * 100
        }
        return try MemoryCube(shape: shape, values: v)
    }

    private func originMap() throws -> OriginMap {
        let n = Self.rows * 4
        let img = { (v: [Double]) in ScanImage(scanRows: Self.rows, scanColumns: 4, values: v, path: .cpu) }
        return try OriginMap(row: img([Double](repeating: 3.5, count: n)), column: img([Double](repeating: 3.5, count: n)))
    }

    /// Every reducer under test, as `(name, run)`; `run` gets the two closures.
    private func reducers(_ c: MemoryCube, path: ComputePath) throws -> [(String, @Sendable (_ progress: @escaping @Sendable (Double) -> Void,
                                                                                            _ cancel: @escaping @Sendable () -> Bool) async throws -> Void)] {
        let circle = DetectorGeometry.circle(centerRow: 4, centerColumn: 4, radius: 2)
        let mask = (0..<(Self.rows * 4)).map { $0 % 3 != 0 }
        let weights = (0..<(Self.rows * 4)).map { Double($0 % 5) + 0.5 }
        let map = try originMap()
        var out: [(String, @Sendable (@escaping @Sendable (Double) -> Void, @escaping @Sendable () -> Bool) async throws -> Void)] = [
            ("mean", { p, k in _ = try await VirtualDiffraction.mean(of: c, path: path, tiling: .rows(2), progress: p, isCancelled: k) }),
            ("max", { p, k in _ = try await VirtualDiffraction.max(of: c, path: path, tiling: .rows(2), progress: p, isCancelled: k) }),
            ("meanAndMax", { p, k in _ = try await VirtualDiffraction.meanAndMax(of: c, path: path, tiling: .rows(2), progress: p, isCancelled: k) }),
            ("image", { p, k in _ = try await VirtualImaging.image(of: c, geometry: circle, path: path, tiling: .rows(2), progress: p, isCancelled: k) }),
            ("centerOfMass", { p, k in _ = try await CenterOfMass.measure(of: c, mask: circle, path: path, tiling: .rows(2), progress: p, isCancelled: k) }),
            ("mask mean", { p, k in
                _ = try await VirtualDiffraction.pattern(of: c, statistic: .mean, scan: .mask(mask), path: path, tiling: .rows(2), progress: p, isCancelled: k) }),
            ("weights max", { p, k in
                _ = try await VirtualDiffraction.pattern(of: c, statistic: .max, scan: .weights(weights), path: path, tiling: .rows(2), progress: p, isCancelled: k) }),
        ]
        if path == .cpu {
            out.append(("median", { p, k in
                _ = try await VirtualDiffraction.pattern(of: c, statistic: .median, scan: .mask(mask), tiling: .rows(2), medianWorkingBytes: 8 * 40 * 20,
                                                         progress: p, isCancelled: k) }))
            out.append(("shifted mean", { p, k in
                _ = try await VirtualDiffraction.pattern(of: c, statistic: .mean, scan: .mask(mask), shift: .rounded(map), tiling: .rows(2), progress: p, isCancelled: k) }))
        }
        return out
    }

    @Test(arguments: [ComputePath.cpu, .metal])
    func progressIsMonotoneAndEndsAtOne(_ path: ComputePath) async throws {
        if path == .metal && !VirtualDiffraction.metalIsAvailable { return }
        let c = try cube()
        for (name, run) in try reducers(c, path: path) {
            let log = Log()
            try await run({ log.report($0) }, { false })
            let f = log.seen
            #expect(f.last == 1, "\(name) \(path): ends at 1 (\(f.suffix(2)))")
            #expect(zip(f, f.dropFirst()).allSatisfy { $0 <= $1 }, "\(name) \(path): monotone")
            #expect(f.allSatisfy { $0 > 0 && $0 <= 1 }, "\(name) \(path): in (0, 1]")
            #expect(f.count >= 2, "\(name) \(path): reports along the way")
        }
    }

    @Test(arguments: [ComputePath.cpu, .metal])
    func cancellingStopsEarlyAndReturnsNothing(_ path: ComputePath) async throws {
        if path == .metal && !VirtualDiffraction.metalIsAvailable { return }
        let c = try cube()
        for (name, run) in try reducers(c, path: path) {
            // Asked in a full run: at least once per tile of 2 rows (12 tiles).
            let full = Log()
            try await run({ _ in }, { _ = full.ask(); return false })
            #expect(full.questions >= 12, "\(name) \(path): asked per tile (\(full.questions))")
            // Cancelled from the third question on: throws, and the passes stop there.
            let log = Log(), progress = Log()
            var cancelled = false
            do { try await run({ progress.report($0) }, { log.ask() > 2 }) } catch is CancellationError { cancelled = true }
            #expect(cancelled, "\(name) \(path): throws CancellationError")
            #expect(log.questions <= 3, "\(name) \(path): stopped at the question that said so (\(log.questions) asked)")
            #expect((progress.seen.last ?? 0) < 0.5, "\(name) \(path): no progress past the stop")
            // Cancelled before the start: no tile read at all.
            let none = Log()
            var immediately = false
            do { try await run({ none.report($0) }, { true }) } catch is CancellationError { immediately = true }
            #expect(immediately && none.seen.isEmpty, "\(name) \(path): cancelled before the start")
        }
    }

    @Test func theDefaultsChangeNothing() async throws {
        let c = try cube()
        let a = try await VirtualDiffraction.mean(of: c, tiling: .rows(3)).values
        let b = try await VirtualDiffraction.mean(of: c, tiling: .rows(3), progress: { _ in }, isCancelled: { false }).values
        #expect(a.map(\.bitPattern) == b.map(\.bitPattern))
    }

    @Test func aCancelledTaskStillStopsTheReducers() async throws {
        let c = try cube()
        let task = Task { () -> Bool in
            withUnsafeCurrentTask { $0?.cancel() }
            do { _ = try await VirtualDiffraction.mean(of: c, tiling: .rows(1)); return false } catch is CancellationError { return true } catch { return false }
        }
        #expect(await task.value)
    }
}
