import Foundation
import Testing
import CuboCore
import CuboVirtual

/// A deterministic cube: Gaussian blob plus noise minus an offset, so sums cancel (the case plain Float32 sums miss).
private func testCube(rows: Int, columns: Int, qr: Int, qc: Int, seed: UInt64) throws -> MemoryCube {
    var state = seed
    func next() -> Double { state = state &* 6364136223846793005 &+ 1442695040888963407; return Double(state >> 11) / Double(1 << 53) }
    let shape = try CubeShape(scanRows: rows, scanColumns: columns, detectorRows: qr, detectorColumns: qc)
    var v = [Float](); v.reserveCapacity(shape.elementCount)
    for p in 0..<shape.scanCount { for r in 0..<qr { for c in 0..<qc {
        let d = Double((r - qr / 2) * (r - qr / 2) + (c - qc / 2) * (c - qc / 2))
        v.append(Float(400 * exp(-d / 30) * (1 + 0.3 * sin(Double(p))) + 20 * (next() - 0.5) - 30))
    } } }
    return try MemoryCube(shape: shape, values: v)
}

// Metal against CPU: relative 1e-5 (the registered Metal tolerance; py4DSTEM states none for its float32 paths).
@Suite(.enabled(if: VirtualDiffraction.metalIsAvailable, "no Metal device on this machine"))
struct SelectedAreaMetalTests {
    private func selection(_ n: Int) -> (mask: [Bool], weights: [Double]) {
        let mask = (0..<n).map { ($0 * 7 + 3) % 5 != 0 }
        let weights = (0..<n).map { k in mask[k] ? 0.25 + Double((k * 13) % 17) / 8 - (k % 6 == 0 ? 1.5 : 0) : 0 }
        return (mask, weights)
    }

    @Test func weightedAndMaskedPatternsAgreeWithTheCPU() async throws {
        let cube = try await testCube(rows: 11, columns: 9, qr: 20, qc: 24, seed: 11)
        let (mask, weights) = selection(cube.shape.scanCount)
        for tiling in [ScanTiling.rows(1), .rows(4), .automatic] {
            for statistic in [DiffractionStatistic.mean, .max] {
                for (name, scan) in [("mask", ScanSelection.mask(mask)), ("weights", .weights(weights))] {
                    let cpu = try await VirtualDiffraction.pattern(of: cube, statistic: statistic, scan: scan, tiling: tiling)
                    let gpu = try await VirtualDiffraction.pattern(of: cube, statistic: statistic, scan: scan, path: .metal, tiling: tiling)
                    #expect(gpu.path == .metal && cpu.path == .cpu)
                    let worst = assertRelative(gpu.values, cpu.values, 1e-5, "\(statistic) \(name) \(tiling)")
                    let abs = zip(gpu.values, cpu.values).map { Swift.abs($0 - $1) }.max() ?? 0
                    print("SELECTEDAREA absolute gap \(abs)")
                    print("SELECTEDAREA \(statistic) \(name) \(tiling) worst \(worst)")
                }
            }
        }
    }

    // Negative controls: the comparison has teeth. A selection one position off, and a mask where weights were meant, each
    // move the result by far more than the tolerance.
    @Test func theComparisonCatchesAWrongSelection() async throws {
        let cube = try testCube(rows: 11, columns: 9, qr: 20, qc: 24, seed: 11)
        var (mask, weights) = selection(cube.shape.scanCount)
        _ = mask
        let cpu = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .mask(mask))
        var off = mask; off[mask.firstIndex(of: true)!] = false   // one selected position dropped
        let gpuOff = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .mask(off), path: .metal)
        print("SELECTEDAREA control: one position dropped moves the mean by \(distance(gpuOff.values, cpu.values))")
        #expect(distance(gpuOff.values, cpu.values) > 1e-5)   // the tolerance
        let cpuW = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .weights(weights))
        let gpuMask = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .mask(weights.map { $0 != 0 }), path: .metal)
        #expect(distance(gpuMask.values, cpuW.values) > 1e-3)
        weights = weights.map { $0 * 1.01 }
        let gpuW = try await VirtualDiffraction.pattern(of: cube, statistic: .max, scan: .weights(weights), path: .metal)
        let cpuMax = try await VirtualDiffraction.pattern(of: cube, statistic: .max, scan: .weights(selection(cube.shape.scanCount).weights))
        #expect(distance(gpuW.values, cpuMax.values) > 1e-3)   // a 1% error in every weight is seen
    }

    @Test func refusalsAndEdges() async throws {
        let cube = try testCube(rows: 4, columns: 4, qr: 6, qc: 6, seed: 3)
        let all = [Bool](repeating: true, count: 16)
        await #expect(throws: CuboVirtualError.self) { _ = try await VirtualDiffraction.pattern(of: cube, statistic: .median, scan: .mask(all), path: .metal) }
        await #expect(throws: CuboVirtualError.self) {
            _ = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .mask(all), shift: .rounded(try constantMap(cube.shape)), path: .metal)
        }
        await #expect(throws: CuboVirtualError.self) { _ = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .mask([true]), path: .metal) }
        await #expect(throws: CuboVirtualError.self) { _ = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .mask([Bool](repeating: false, count: 16)), path: .metal) }
        await #expect(throws: CuboVirtualError.self) { _ = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .weights([Double](repeating: 0, count: 16)), path: .metal) }
        // the .all selection on Metal is the existing mean and max path
        let m = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: .all, path: .metal)
        #expect(m.path == .metal)
        // a negative-only cube: the maximum is negative (seeded at -infinity), as on the CPU
        let neg = try MemoryCube(shape: cube.shape, values: cube.values.map { -abs($0) - 1 })
        let gpu = try await VirtualDiffraction.pattern(of: neg, statistic: .max, scan: .mask(all), path: .metal)
        #expect(gpu.values.allSatisfy { $0 < 0 })
    }

    @Test func aResidentCubeIsTheSameCubeReadInPlace() async throws {
        let cube = try testCube(rows: 9, columns: 7, qr: 16, qc: 18, seed: 5)
        let resident = try await ResidentCube(loading: cube)
        #expect(resident.shape == cube.shape && resident.byteCount == cube.shape.elementCount * 4)
        #expect(try await resident.readScanTile(rows: 2..<5).values == cube.readScanTile(rows: 2..<5).values)
        #expect(try await resident.readPattern(row: 8, column: 6).values == cube.readPattern(row: 8, column: 6).values)
        await #expect(throws: CuboError.self) { _ = try await resident.readScanTile(rows: 8..<10) }
        let (mask, weights) = selection(cube.shape.scanCount)
        for tiling in [ScanTiling.rows(2), .automatic] {
            // same kernel, same tiles: bit for bit the copying Metal path
            for scan in [ScanSelection.mask(mask), .weights(weights)] {
                let a = try await VirtualDiffraction.pattern(of: cube, statistic: .mean, scan: scan, path: .metal, tiling: tiling)
                let b = try await VirtualDiffraction.pattern(of: resident, statistic: .mean, scan: scan, path: .metal, tiling: tiling)
                #expect(a.values == b.values)
            }
            let mm = try await VirtualDiffraction.meanAndMax(of: cube, path: .metal, tiling: tiling)
            let rm = try await VirtualDiffraction.meanAndMax(of: resident, path: .metal, tiling: tiling)
            #expect(mm.mean.values == rm.mean.values && mm.max.values == rm.max.values)
            let geometry = DetectorGeometry.circle(centerRow: 8, centerColumn: 9, radius: 5)
            let vi = try await VirtualImaging.image(of: cube, geometry: geometry, path: .metal, tiling: tiling)
            let vr = try await VirtualImaging.image(of: resident, geometry: geometry, path: .metal, tiling: tiling)
            #expect(vi.values == vr.values)
            let cpu = try await VirtualImaging.image(of: resident, geometry: geometry, path: .cpu, tiling: tiling)
            _ = assertRelative(vr.values, cpu.values, 1e-5, "resident virtual image against the CPU")
        }
        // a cube too large for one buffer is refused up front, not on the GPU
        #expect(ResidentCube.fits(cube.shape))
    }
}

private func distance(_ a: [Double], _ b: [Double]) -> Double {
    let scale = max(1e-12, b.map(abs).max() ?? 1)
    return zip(a, b).map { abs($0 - $1) / scale }.max() ?? 0
}

private func constantMap(_ shape: CubeShape) throws -> OriginMap {
    let v = [Double](repeating: 3, count: shape.scanCount)
    return try OriginMap(row: ScanImage(scanRows: shape.scanRows, scanColumns: shape.scanColumns, values: v, path: .cpu),
                         column: ScanImage(scanRows: shape.scanRows, scanColumns: shape.scanColumns, values: v, path: .cpu))
}
