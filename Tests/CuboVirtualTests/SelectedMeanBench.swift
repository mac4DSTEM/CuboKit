import CuboCore
@testable import CuboVirtual
import Foundation
import Testing

/// The CPU selected-area mean over a mask or weights (CuboKit#54). Skipped unless `CUBOKIT_SELECTEDMEAN_BENCH` is set:
/// `CUBOKIT_SELECTEDMEAN_BENCH=1 swift test -c release --filter SelectedMeanBench`. On a resident 100 x 100 x 64 x 64 Float32
/// cube it prints the median and best of 15 calls of `VirtualDiffraction.pattern(statistic: .mean)` over a mask that picks
/// about half the positions (scattered), a mask that picks every position, and weights on every position.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["CUBOKIT_SELECTEDMEAN_BENCH"] != nil,
                             "set CUBOKIT_SELECTEDMEAN_BENCH"))
struct SelectedMeanBench {
    private func time(_ calls: Int = 15, _ body: () async throws -> Void) async rethrows -> (Double, Double) {
        try await body()
        var ts = [Double]()
        for _ in 0..<calls {
            let t0 = DispatchTime.now().uptimeNanoseconds
            try await body()
            ts.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
        }
        ts.sort()
        return (ts[ts.count / 2], ts[0])
    }

    private func report(_ name: String, _ t: (Double, Double)) {
        print(String(format: "SELECTEDMEANBENCH %-20@ median %6.2f ms  best %6.2f ms", name as NSString, t.0, t.1))
    }

    @Test func typicalCube() async throws {
        let shape = try CubeShape(scanRows: 100, scanColumns: 100, detectorRows: 64, detectorColumns: 64)
        var state: UInt64 = 7
        func next() -> UInt64 { state = state &* 6364136223846793005 &+ 1442695040888963407; return state >> 40 }
        let values = (0..<shape.elementCount).map { _ in Float(next()) / 1024 }
        let cube = try MemoryCube(shape: shape, values: values)
        let half = (0..<shape.scanCount).map { _ in next() & 1 == 0 }
        let every = [Bool](repeating: true, count: shape.scanCount)
        let weights = (0..<shape.scanCount).map { _ in Double(next()) / Double(1 << 24) + 0.01 }

        report("mask half", try await time {
            _ = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .mask(half))
        })
        let firstHalf = (0..<shape.scanCount).map { $0 < shape.scanCount / 2 }
        report("mask first half", try await time {
            _ = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .mask(firstHalf))
        })
        report("mask every", try await time {
            _ = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .mask(every))
        })
        report("weights", try await time {
            _ = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .weights(weights))
        })
    }
}
