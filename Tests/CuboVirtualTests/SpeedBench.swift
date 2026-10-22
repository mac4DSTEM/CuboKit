import Foundation
import Testing
import CuboKitTestSupport
import CuboCore
import CuboVirtual

// Alternating before/after timing for docs/benchmarks.md: calls of the CPU mean pattern, the selected-area mask, binScan and
// medianFilterMaskedPixels; prints the median and best in ms and a hash of each result's bits, so two builds can be compared
// bit for bit. Skipped unless CUBOKIT_SPEED_DIR names a folder with `bench.py make` output (CUBOKIT_SPEED_CALLS, default 15).
@Suite(.enabled(if: ProcessInfo.processInfo.environment["CUBOKIT_SPEED_DIR"] != nil, "set CUBOKIT_SPEED_DIR"))
struct SpeedBench {
    private func hash(_ v: [Double]) -> String {
        var h: UInt64 = 1469598103934665603
        for x in v { h = (h ^ x.bitPattern) &* 1099511628211 }
        return String(h, radix: 16)
    }
    private func hash(_ v: [Float]) -> String { hash(v.map { Double($0) }) }

    private func time(_ calls: Int, _ body: () async throws -> Void) async throws -> (Double, Double) {
        try await body()
        var ts = [Double]()
        for _ in 0..<calls {
            let t0 = ContinuousClock.now
            try await body()
            let d = ContinuousClock.now - t0
            ts.append((Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18) * 1e3)
        }
        ts.sort()
        return (ts[ts.count / 2], ts[0])
    }

    @Test func run() async throws {
        let env = ProcessInfo.processInfo.environment
        let dir = URL(fileURLWithPath: env["CUBOKIT_SPEED_DIR"]!)
        let calls = Int(env["CUBOKIT_SPEED_CALLS"] ?? "") ?? 15
        func report(_ name: String, _ t: (Double, Double), _ h: String) {
            print("SPEED \(name) median_ms=\(String(format: "%.2f", t.0)) best_ms=\(String(format: "%.2f", t.1)) hash=\(h)")
        }
        // CUBOKIT_SPEED_DUMP=<folder>: also write the results (and the mask and weights used) raw, for `bench.py compare`.
        let dump = env["CUBOKIT_SPEED_DUMP"].map { URL(fileURLWithPath: $0) }
        func save<T>(_ name: String, _ v: [T]) throws {
            guard let dump else { return }
            try FileManager.default.createDirectory(at: dump, withIntermediateDirectories: true)
            try v.withUnsafeBytes { try Data($0).write(to: dump.appendingPathComponent(name)) }
        }
        for size in ["typical", "mid"] {
            let cube = try await MemoryCube(loading: try MappedCube(npy: dir.appendingPathComponent("\(size).npy")))
            let shape = cube.shape
            var state: UInt64 = 3
            func rnd() -> Double { state = state &* 6364136223846793005 &+ 1442695040888963407; return Double(state >> 11) / Double(1 << 53) }
            let mask = (0..<shape.scanCount).map { _ in rnd() > 0.2 }
            if size == "typical" {
                var rd = [Double]()
                var t = try await time(calls) { rd = try await VirtualDiffraction.mean(of: cube, path: .cpu).values }
                report("dp_mean|\(size)", t, hash(rd))
                try save("dp_mean.f64", rd)
                try save("mask.u8", mask.map { $0 ? UInt8(1) : 0 })
                t = try await time(calls) {
                    rd = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .mask(mask), path: .cpu).values }
                report("mask_mean|\(size)", t, hash(rd))
                try save("mask_mean.f64", rd)
                let weights = mask.map { $0 ? rnd() : 0 }
                t = try await time(calls) {
                    rd = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .weights(weights), path: .cpu).values }
                report("weights_mean|\(size)", t, hash(rd))
                try save("weights_mean.f64", rd)
                try save("weights.f64", weights)
                var r = [Float]()
                t = try await time(calls) { r = try await CubeEditing.binScan(of: cube, by: 2).cube.values }
                report("bin_R_2|\(size)", t, hash(r))
                try save("bin_R_2_typical.f32", r)
            }
            var pm = [Bool](repeating: false, count: shape.detectorCount)
            pm[5 * shape.detectorColumns + 7] = true; pm[20 * shape.detectorColumns + 30] = true
            if size == "mid" {
                var r = [Float]()
                var t = try await time(calls) { r = try await CubeEditing.medianFilterMaskedPixels(of: cube, mask: pm).cube.values }
                report("median_masked_pixels|\(size)", t, hash(r))
                try save("median_masked_pixels_mid.f32", r)
                t = try await time(calls) { r = try await CubeEditing.binScan(of: cube, by: 2).cube.values }
                report("bin_R_2|\(size)", t, hash(r))
            }
        }
    }
}
