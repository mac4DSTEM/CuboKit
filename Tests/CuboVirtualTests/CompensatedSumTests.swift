import Foundation
import Testing
import CuboKitTestSupport
import CuboCore
@testable import CuboVirtual

// 0.9.0: the CPU mean pattern (and the selected-area mean over a mask or weights) adds with a branch-free SIMD TwoSum and no
// per-element finite check. This transcribes the scalar Neumaier loop it replaced (finite check, magnitude branch, skipped zero
// corrections) and compares bits, on random values, on columns with NaN, +Inf, -Inf and +Inf with -Inf, and on a -0 column.
@Suite struct CompensatedSumTests {
    private struct OldNeumaier {
        var sum: Double, correction: Double
        init(start: Double) { sum = start; correction = start }
        mutating func add(_ x: Double) {
            let a = sum, t = a + x
            if t.isFinite {
                let e = abs(a) >= abs(x) ? (a - t) + x : (x - t) + a
                if e != 0 { correction += e }
            }
            sum = t
        }
        var value: Double { sum + correction }
    }

    private func sameBits(_ a: [Double], _ b: [Double]) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern || ($0.isNaN && $1.isNaN) }
    }

    private func values(_ shape: CubeShape) -> [Float] {
        var state: UInt64 = 2026
        func next() -> Float { state = state &* 6364136223846793005 &+ 1442695040888963407; return Float(Double(state >> 11) / Double(1 << 53) - 0.35) * 3e3 }
        var v = (0..<shape.elementCount).map { _ in next() }
        // wide dynamic range so the corrections are not all zero
        for k in stride(from: 0, to: v.count, by: 7) { v[k] *= 1e-6 }
        for k in stride(from: 3, to: v.count, by: 11) { v[k] *= 1e6 }
        return v
    }

    private func specials(_ v: inout [Float], dc: Int, scanCount: Int) {
        v[2 * dc + 0] = .nan                                   // NaN in pixel 0
        v[1 * dc + 1] = .infinity                              // +Inf in pixel 1
        v[0 * dc + 2] = -.infinity                             // -Inf in pixel 2
        v[3 * dc + 3] = .infinity; v[5 * dc + 3] = -.infinity  // +Inf then -Inf: NaN in pixel 3
        v[4 * dc + 4] = .infinity; v[6 * dc + 4] = 5           // +Inf stays +Inf
        for k in 0..<scanCount { v[k * dc + 5] = -0.0 }        // a column of -0 keeps its sign
        for k in 0..<scanCount { v[k * dc + 6] = 0.0 }
        v[3 * dc + 7] = .greatestFiniteMagnitude; v[4 * dc + 7] = .greatestFiniteMagnitude  // large, still finite in Double
    }

    @Test func meanEqualsTheScalarLoopOnRandomNaNAndInfInputs() async throws {
        // 13 x 11 = 143 detector pixels: tails of every SIMD width; 3 x 4 = 12 positions
        let shape = try CubeShape(scanRows: 3, scanColumns: 4, detectorRows: 13, detectorColumns: 11)
        var v = values(shape)
        specials(&v, dc: 143, scanCount: 12)
        let c = try MemoryCube(shape: shape, values: v)
        var old = [OldNeumaier](repeating: OldNeumaier(start: -0.0), count: 143)
        for k in 0..<12 { for i in 0..<143 { old[i].add(Double(v[k * 143 + i])) } }
        let want = old.map { $0.value / 12 }
        for tiling in [ScanTiling.rows(1), .rows(2), .automatic] {
            let got = try await VirtualDiffraction.mean(of: c, tiling: tiling).values
            #expect(sameBits(got, want), "mean \(tiling)")
        }
        let v0 = try await VirtualDiffraction.mean(of: c).values
        #expect(v0[0].isNaN && v0[1] == .infinity && v0[2] == -.infinity && v0[3].isNaN && v0[4] == .infinity)
        #expect(v0[5].bitPattern == (-0.0).bitPattern && v0[6].bitPattern == (0.0).bitPattern)
    }

    @Test func meanEqualsTheScalarLoopOnRandomFiniteInput() async throws {
        let shape = try CubeShape(scanRows: 7, scanColumns: 5, detectorRows: 100, detectorColumns: 96)   // 9600 pixels: several blocks
        let v = values(shape)
        let c = try MemoryCube(shape: shape, values: v)
        var old = [OldNeumaier](repeating: OldNeumaier(start: -0.0), count: 9600)
        for k in 0..<35 { for i in 0..<9600 { old[i].add(Double(v[k * 9600 + i])) } }
        let got = try await VirtualDiffraction.mean(of: c, tiling: .rows(2)).values
        #expect(sameBits(got, old.map { $0.value / 35 }))
    }

    /// CuboKit#7: `addRows` (four rows per pass, the sum in registers) gives the bits of one `add` per row, for every row count
    /// around the pass width, pixel counts with SIMD tails, and NaN, +-Inf, -0 and huge values in the rows.
    @Test func addRowsEqualsOneAddPerRow() throws {
        for (rows, count) in [(1, 7), (3, 4), (4, 9), (5, 143), (8, 16), (9, 3), (13, 130)] {
            let stride = count + 5   // rows not packed: the stride is honoured
            let shape = try CubeShape(scanRows: 1, scanColumns: rows, detectorRows: 1, detectorColumns: stride)
            var v = values(shape)
            if rows >= 7, count > 7 {
                specials(&v, dc: stride, scanCount: rows)
                // and in the scalar tail (the last pixels): +Inf then -Inf, NaN, +Inf kept
                v[1 * stride + count - 1] = .infinity; v[5 * stride + count - 1] = -.infinity
                v[2 * stride + count - 2] = .nan
                v[6 * stride + count - 3] = .infinity
            }
            var s1 = [Double](repeating: -0.0, count: count), c1 = s1, s2 = s1, c2 = s1
            v.withUnsafeBufferPointer { p in
                for k in 0..<rows { CompensatedSum.add(p.baseAddress! + k * stride, weight: nil, sum: &s1, correction: &c1, count: count) }
                CompensatedSum.addRows(p.baseAddress!, stride: stride, rows: rows, sum: &s2, correction: &c2, count: count)
            }
            let a = zip(s1, c1).map { CompensatedSum.value(sum: $0, correction: $1) }
            let b = zip(s2, c2).map { CompensatedSum.value(sum: $0, correction: $1) }
            #expect(sameBits(a, b), "rows \(rows) count \(count)")
            #expect(sameBits(s1, s2), "sums rows \(rows) count \(count)")
        }
    }

    @Test func maskAndWeightedMeansEqualTheScalarNeumaier() async throws {
        let shape = try CubeShape(scanRows: 4, scanColumns: 5, detectorRows: 13, detectorColumns: 11)
        var v = values(shape)
        specials(&v, dc: 143, scanCount: 20)
        let c = try MemoryCube(shape: shape, values: v)
        let mask = (0..<20).map { $0 % 3 != 1 }
        let weights = (0..<20).map { Double($0 % 7) * 0.37 + 0.1 }
        // the selected-area reference starts at +0 (the Neumaier struct of the reductions)
        var plain = [OldNeumaier](repeating: OldNeumaier(start: 0), count: 143), weighted = plain
        var count = 0.0, weightSum = OldNeumaier(start: 0)
        for k in 0..<20 {
            for i in 0..<143 { weighted[i].add(Double(v[k * 143 + i]) * weights[k]) }
            weightSum.add(weights[k])
            guard mask[k] else { continue }
            count += 1
            for i in 0..<143 { plain[i].add(Double(v[k * 143 + i])) }
        }
        for tiling in [ScanTiling.rows(1), .rows(3), .automatic] {
            let m = try await VirtualDiffraction.pattern(of: c, statistic: .mean, scan: .mask(mask), tiling: tiling).values
            #expect(sameBits(m, plain.map { $0.value / count }), "mask \(tiling)")
            let w = try await VirtualDiffraction.pattern(of: c, statistic: .mean, scan: .weights(weights), tiling: tiling).values
            #expect(sameBits(w, weighted.map { $0.value / weightSum.value }), "weights \(tiling)")
        }
    }
}
