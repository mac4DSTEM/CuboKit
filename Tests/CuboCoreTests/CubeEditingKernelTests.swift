import Foundation
import Testing
import CuboCore

// 0.9.0: `binScan` and `medianFilterMaskedPixels` run vectorised and across cores. These transcribe the scalar loops they
// replaced (Double sums in block order rounded once; a sorted scratch array per masked pixel and pattern) and compare bits.
@Suite struct CubeEditingKernelTests {
    private func values(_ count: Int, nan: Bool = false) -> [Float] {
        var state: UInt64 = 77
        var v = (0..<count).map { _ -> Float in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Double(state >> 11) / Double(1 << 53) - 0.4) * 500
        }
        if nan { for k in stride(from: 13, to: count, by: 97) { v[k] = .nan } }
        return v
    }

    private func sameBits(_ a: [Float], _ b: [Float]) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern || ($0.isNaN && $1.isNaN) }
    }

    @Test(arguments: [(2, 7, 9, 5, 3), (3, 10, 8, 4, 4), (1, 4, 5, 3, 3), (2, 9, 6, 8, 8)])
    func binScanEqualsTheScalarSums(_ c: (Int, Int, Int, Int, Int)) async throws {
        let (factor, rows, columns, qr, qc) = c
        let shape = try CubeShape(scanRows: rows, scanColumns: columns, detectorRows: qr, detectorColumns: qc)
        let v = values(shape.elementCount)
        let cube = try MemoryCube(shape: shape, values: v)
        let dc = qr * qc
        var want = [Float]()
        for r in 0..<(rows / factor) { for oc in 0..<(columns / factor) { for q in 0..<dc {
            var acc = 0.0
            for dr in 0..<factor { for dcol in 0..<factor { acc += Double(v[(((r * factor + dr) * columns) + oc * factor + dcol) * dc + q]) } }
            want.append(Float(acc))
        } } }
        let got = try await CubeEditing.binScan(of: cube, by: factor).cube.values
        #expect(sameBits(got, want))
    }

    @Test(arguments: [3, 4, 5, 8])
    func medianFilterEqualsTheSortedScratchLoop(kernel: Int) async throws {
        let shape = try CubeShape(scanRows: 5, scanColumns: 6, detectorRows: 12, detectorColumns: 10)
        let v = values(shape.elementCount, nan: true)
        let cube = try MemoryCube(shape: shape, values: v)
        var mask = [Bool](repeating: false, count: 120)
        for k in [0, 9, 11, 12, 55, 56, 57, 119] { mask[k] = true }
        let widthMax = kernel % 2 == 0 ? kernel / 2 : (kernel + 1) / 2, widthMin = kernel / 2
        var want = v
        for pattern in 0..<30 {
            var p = Array(v[(pattern * 120)..<((pattern + 1) * 120)])
            for r in 0..<12 { for c in 0..<10 where mask[r * 10 + c] {
                var scratch = [Float]()
                for rr in max(0, r - widthMin)..<min(12, r + widthMax) { for cc in max(0, c - widthMin)..<min(10, c + widthMax) { scratch.append(p[rr * 10 + cc]) } }
                let n = scratch.count
                if scratch.contains(where: \.isNaN) { p[r * 10 + c] = .nan; continue }
                scratch.sort()
                p[r * 10 + c] = n % 2 == 1 ? scratch[n / 2] : (scratch[n / 2 - 1] + scratch[n / 2]) / 2
            } }
            want.replaceSubrange((pattern * 120)..<((pattern + 1) * 120), with: p)
        }
        let got = try await CubeEditing.medianFilterMaskedPixels(of: cube, mask: mask, kernelWidth: kernel).cube.values
        #expect(sameBits(got, want))
    }

    @Test func medianFilterWithAWideKernelUsesTheSortPath() async throws {
        let shape = try CubeShape(scanRows: 2, scanColumns: 2, detectorRows: 9, detectorColumns: 9)
        let v = values(shape.elementCount)
        let cube = try MemoryCube(shape: shape, values: v)
        var mask = [Bool](repeating: false, count: 81); mask[40] = true
        let got = try await CubeEditing.medianFilterMaskedPixels(of: cube, mask: mask, kernelWidth: 7).cube.values   // 49 values
        for pattern in 0..<4 {
            var window = [Float]()
            for rr in 1..<8 { for cc in 1..<8 { window.append(v[pattern * 81 + rr * 9 + cc]) } }
            window.sort()
            #expect(got[pattern * 81 + 40] == window[24])
        }
    }
}
