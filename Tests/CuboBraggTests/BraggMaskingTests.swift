import Foundation
import Testing
import CuboCore
import CuboBragg

private struct MaskCase: Decodable { let r: Int, c: Int, radius: Double, peaks: [[Double]], mask: [Int] }
private struct AveCase: Decodable { let r: Int, c: Int, radius: Double?, plain: [Double]?, masked: [Double]?, valid: [Int]? }
private struct LatticeCase: Decodable { let g1: [Double], g2: [Double], radius: Double, origin: [Double], maxQ: Double, includeOrigin: Bool, rotation: Double, mask: [Int] }
private struct Pin: Decodable { let shape: [Int], data: [Double], peaks: [String: [[Double]]], masks: [MaskCase], ave: [AveCase], lattice: [LatticeCase] }
private func pin() throws -> Pin {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/braggmask.json")
    return try JSONDecoder().decode(Pin.self, from: Data(contentsOf: url))
}
private func vectors(_ p: Pin) -> BraggVectors {
    let s = p.shape
    return BraggVectors(scanRows: s[0], scanColumns: s[1], peaks: (0..<(s[0] * s[1])).map { k in
        (p.peaks["\(k / s[1]),\(k % s[1])"] ?? []).map { BraggPeak(row: $0[0], column: $0[1], intensity: 1) } })
}

@Suite struct BraggMaskingTests {
    // The mask is a comparison of hypot distances with the radius: exact. The average is Double arithmetic on Float32 data:
    // np.allclose defaults (rtol 1e-5, atol 1e-8; measured far below).
    @Test func maskAndLocalAverageAgainstPy4DSTEM() async throws {
        let p = try pin(), v = vectors(p)
        let shape = try CubeShape(p.shape)
        for m in p.masks {
            let got = try BraggMasking.mask(vectors: v, scanRow: m.r, scanColumn: m.c, detectorRows: shape.detectorRows,
                                            detectorColumns: shape.detectorColumns, radius: m.radius)
            #expect(got.map { $0 ? 1 : 0 } == m.mask, "mask at (\(m.r), \(m.c)) radius \(m.radius)")
        }
        let cube = try MemoryCube(shape: shape, values: p.data.map { Float($0) })
        for a in p.ave {
            let got = try await BraggMasking.localAverage(of: cube, scanRow: a.r, scanColumn: a.c,
                                                          braggMask: a.radius.map { (v, $0) })
            if let plain = a.plain {
                #expect(zip(got.values, plain).allSatisfy { abs($0 - $1) <= 1e-8 + 1e-5 * abs($1) }, "plain (\(a.r), \(a.c))")
            } else if let masked = a.masked, let valid = a.valid {
                #expect(got.valid.map { $0 ? 1 : 0 } == valid, "validity (\(a.r), \(a.c))")
                for i in valid.indices {
                    if valid[i] == 0 { #expect(got.values[i].isNaN) } else { #expect(abs(got.values[i] - masked[i]) <= 1e-8 + 1e-5 * abs(masked[i])) }
                }
            }
        }
        #expect(p.ave.contains { $0.valid?.contains(0) == true }, "the pin holds a pixel every neighbour masked")
    }

    // make_bragg_mask: disks on a rotated lattice summed; counts are exact.
    @Test func latticeMaskAgainstPy4DSTEM() throws {
        let p = try pin()
        for l in p.lattice {
            let got = try BraggMasking.latticeMask(detectorRows: p.shape[2], detectorColumns: p.shape[3], g1: (l.g1[0], l.g1[1]), g2: (l.g2[0], l.g2[1]),
                                                   radius: l.radius, origin: (l.origin[0], l.origin[1]), maximumDistance: l.maxQ,
                                                   includeOrigin: l.includeOrigin, rotationDegrees: l.rotation)
            #expect(got == l.mask, "lattice mask rotation \(l.rotation) includeOrigin \(l.includeOrigin)")
        }
        #expect(throws: CuboBraggError.self) {
            _ = try BraggMasking.latticeMask(detectorRows: 4, detectorColumns: 4, g1: (0, 0), g2: (1, 1), radius: 1, origin: (0, 0), maximumDistance: 3)
        }
    }

    @Test func refusals() async throws {
        let p = try pin(), v = vectors(p)
        #expect(throws: CuboBraggError.self) { _ = try BraggMasking.mask(vectors: v, scanRow: 9, scanColumn: 0, detectorRows: 4, detectorColumns: 4, radius: 1) }
        #expect(throws: CuboBraggError.self) { _ = try BraggMasking.mask(vectors: v, scanRow: 0, scanColumn: 0, detectorRows: 4, detectorColumns: 4, radius: .nan) }
        let cube = try MemoryCube(shape: CubeShape(p.shape), values: p.data.map { Float($0) })
        await #expect(throws: CuboBraggError.self) { _ = try await BraggMasking.localAverage(of: cube, scanRow: 4, scanColumn: 0) }
    }
}

@Suite struct BraggMaskingHardeningTests {
    @Test func tinyLatticeVectorsAreRefusedAndNaNPeakMasksEverything() throws {
        #expect(throws: CuboBraggError.self) {
            _ = try BraggMasking.latticeMask(detectorRows: 4, detectorColumns: 4, g1: (1e-300, 0), g2: (0, 1), radius: 1, origin: (0, 0), maximumDistance: 4)
        }
        let v = BraggVectors(scanRows: 1, scanColumns: 1, peaks: [[BraggPeak(row: .nan, column: 1, intensity: 1)]])
        let m = try BraggMasking.mask(vectors: v, scanRow: 0, scanColumn: 0, detectorRows: 3, detectorColumns: 3, radius: 1)
        #expect(m.allSatisfy { !$0 })
    }
}
