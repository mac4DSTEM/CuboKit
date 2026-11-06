import Foundation
import Testing
import CuboCore
import CuboVirtual

// The kit side of docs/benchmarks.md (tools/benchmarks/bench.py has the py4DSTEM side and the merge). Skipped unless
// CUBOKIT_BENCH_DIR names a folder that `bench.py make` filled. One benchmark at a time machine-wide.
@Suite(.enabled(if: ProcessInfo.processInfo.environment["CUBOKIT_BENCH_DIR"] != nil, "set CUBOKIT_BENCH_DIR"))
struct BenchmarkHarness {
    private func time(_ body: () async throws -> Void) async throws -> [String: Double] {
        try await body()
        var ts = [Double]()
        for _ in 0..<5 {
            let t0 = ContinuousClock.now
            try await body()
            let d = ContinuousClock.now - t0
            ts.append(Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18)
        }
        ts.sort()
        return ["best": ts[0], "median": ts[2], "spread": ts[4] - ts[0]]
    }

    @Test func run() async throws {
        let dir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["CUBOKIT_BENCH_DIR"]!)
        var results = [String: [String: Double]]()
        for size in ["small", "mid", "typical"] {
            let mapped = try MappedCube(npy: dir.appendingPathComponent("\(size).npy"))
            let cube = try await MemoryCube(loading: mapped)
            let shape = cube.shape
            let resident: ResidentCube? = VirtualDiffraction.metalIsAvailable ? try await ResidentCube(loading: cube) : nil
            var state: UInt64 = 3
            func rnd() -> Double { state = state &* 6364136223846793005 &+ 1442695040888963407; return Double(state >> 11) / Double(1 << 53) }
            let mask = (0..<shape.scanCount).map { _ in rnd() > 0.2 }
            let weights = mask.map { $0 ? rnd() : 0 }
            let qr = Double(shape.detectorRows), qc = Double(shape.detectorColumns)
            let circle = DetectorGeometry.circle(centerRow: qr / 2, centerColumn: qc / 2, radius: qr / 4)
            let paths: [(String, ComputePath)] = VirtualDiffraction.metalIsAvailable ? [("cpu", .cpu), ("metal", .metal)] : [("cpu", .cpu)]
            for (label, path) in paths {
                results["dp_mean|\(size)|\(label)"] = try await time { _ = try await VirtualDiffraction.mean(of: cube, path: path) }
                results["dp_max|\(size)|\(label)"] = try await time { _ = try await VirtualDiffraction.max(of: cube, path: path) }
                results["virtual_image_circle|\(size)|\(label)"] = try await time { _ = try await VirtualImaging.image(of: cube, geometry: circle, path: path) }
                results["selected_area_mean_mask|\(size)|\(label)"] = try await time {
                    _ = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .mask(mask), path: path) }
                results["selected_area_mean_weights|\(size)|\(label)"] = try await time {
                    _ = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .weights(weights), path: path) }
                results["center_of_mass|\(size)|\(label)"] = try await time { _ = try await CenterOfMass.measure(of: cube, mask: circle, path: path) }
            }
            if let resident {
                results["selected_area_mean_mask_resident|\(size)|metal"] = try await time {
                    _ = try await VirtualDiffraction.pattern(of: resident, statistic: .mean, scan: .mask(mask), path: .metal) }
                results["selected_area_mean_weights_resident|\(size)|metal"] = try await time {
                    _ = try await VirtualDiffraction.pattern(of: resident, statistic: .mean, scan: .weights(weights), path: .metal) }
                results["dp_mean_resident|\(size)|metal"] = try await time { _ = try await VirtualDiffraction.mean(of: resident, path: .metal) }
            }
            results["dp_median|\(size)|cpu"] = try await time { _ = try await VirtualDiffraction.median(of: cube) }
            results["bin_Q_2|\(size)|cpu"] = try await time {
                _ = try await MemoryCube(loading: ViewedCube(base: cube, specification: CubeViewSpecification(detectorBin: 2))) }
            results["bin_R_2|\(size)|cpu"] = try await time { _ = try await CubeEditing.binScan(of: cube, by: 2) }
            results["thin_R_2|\(size)|cpu"] = try await time { _ = try await CubeEditing.thinScan(of: cube, by: 2) }
            results["pad_Q_2|\(size)|cpu"] = try await time { _ = try await CubeEditing.padDetector(of: cube, .factor(rows: 2, columns: 2)) }
            if size != "typical" {
                results["resample_Q_half|\(size)|cpu"] = try await time {
                    _ = try await CubeEditing.resampleDetector(of: cube, .factor(rows: 0.5, columns: 0.5)) }
                var pm = [Bool](repeating: false, count: shape.detectorCount); pm[5 * shape.detectorColumns + 7] = true; pm[20 * shape.detectorColumns + 30] = true
                results["median_masked_pixels|\(size)|cpu"] = try await time { _ = try await CubeEditing.medianFilterMaskedPixels(of: cube, mask: pm) }
            }
            print("BENCH finished \(size)")
        }
        let toolchain = "Swift \(ProcessInfo.processInfo.environment["CUBOKIT_BENCH_TOOLCHAIN"] ?? "6.4")"
        let json = try JSONSerialization.data(withJSONObject: ["toolchain": toolchain, "results": results], options: [.prettyPrinted, .sortedKeys])
        try json.write(to: dir.appendingPathComponent("swift.json"))
    }
}
