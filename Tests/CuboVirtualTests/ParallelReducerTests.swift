import Foundation
import Testing
import CuboCore
import CuboVirtual

// The CPU mean and max run on pixel blocks across cores (0.8.0). Each pixel is still added in scan order with the same
// Neumaier step, so the result is bit for bit the single-thread loop's: this transcribes that loop and compares bits.
@Suite struct ParallelReducerTests {
    private func serial(_ values: [Float], scanCount: Int, n: Int) -> (mean: [Double], max: [Double]) {
        var sum = [Double](repeating: -0.0, count: n), corr = sum
        var mx = [Float](repeating: -.infinity, count: n)
        for k in 0..<scanCount { for i in 0..<n {
            let x = Double(values[k * n + i]), a = sum[i], t = a + x
            if t.isFinite { let e = abs(a) >= abs(x) ? (a - t) + x : (x - t) + a; if e != 0 { corr[i] += e } }
            sum[i] = t
            mx[i] = (mx[i].isNaN || values[k * n + i].isNaN) ? .nan : Swift.max(mx[i], values[k * n + i])
        } }
        return ((0..<n).map { (sum[$0] + corr[$0]) / Double(scanCount) }, mx.map(Double.init))
    }

    @Test func blockParallelMeanAndMaxAreBitIdenticalToTheSerialLoop() async throws {
        var state: UInt64 = 99
        func next() -> Float { state = state &* 6364136223846793005 &+ 1442695040888963407; return Float(Double(state >> 11) / Double(1 << 53) - 0.4) * 1e4 }
        // 100 x 96 = 9600 detector pixels: several 2048-pixel blocks, a short last block; some -0 and NaN-free values
        let shape = try CubeShape(scanRows: 7, scanColumns: 5, detectorRows: 100, detectorColumns: 96)
        var v = (0..<shape.elementCount).map { _ in next() }
        for k in 0..<shape.scanCount { v[k * shape.detectorCount + 4000] = -0.0 }   // a column of -0 keeps its sign
        let cube = try MemoryCube(shape: shape, values: v)
        let want = serial(v, scanCount: shape.scanCount, n: shape.detectorCount)
        for tiling in [ScanTiling.rows(1), .rows(3), .automatic] {
            let both = try await VirtualDiffraction.meanAndMax(of: cube, tiling: tiling)
            let mean = try await VirtualDiffraction.mean(of: cube, tiling: tiling), max = try await VirtualDiffraction.max(of: cube, tiling: tiling)
            #expect(both.mean.values.map(\.bitPattern) == want.mean.map(\.bitPattern), "mean bits \(tiling)")
            #expect(mean.values.map(\.bitPattern) == want.mean.map(\.bitPattern), "mean-only bits \(tiling)")
            #expect(both.max.values == want.max && max.values == want.max, "max \(tiling)")
        }
    }
}
