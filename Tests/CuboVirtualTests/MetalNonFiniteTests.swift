import Foundation
import Testing
import CuboKitTestSupport
import CuboCore
import CuboVirtual

/// Hot-pixel data with infinities: a +∞ (and a −∞, and a pixel holding both) in selected patterns. The CPU float64 path is
/// the reference: IEEE sums give ±∞ (NumPy and py4DSTEM do the same), ∞ − ∞ gives NaN. Metal must give the same special
/// value at those pixels and agree within relative 1e-5 everywhere else.
@Suite(.enabled(if: VirtualDiffraction.metalIsAvailable, "no Metal device on this machine"))
struct MetalNonFiniteTests {
    static let rows = 6, columns = 5, qr = 8, qc = 10
    static let plusPixel = 3 * qc + 4, minusPixel = 5 * qc + 1, bothPixel = 6 * qc + 7

    private func hotCube() throws -> MemoryCube {
        let shape = try CubeShape(scanRows: Self.rows, scanColumns: Self.columns, detectorRows: Self.qr, detectorColumns: Self.qc)
        var v = [Float](); v.reserveCapacity(shape.elementCount)
        for p in 0..<shape.scanCount { for i in 0..<shape.detectorCount {
            v.append(Float(sin(Double(p * 31 + i * 7)) * 50 + Double(i % 9) - 3))
        } }
        let n = shape.detectorCount
        v[7 * n + Self.plusPixel] = .infinity           // one hot pixel at one selected position (tile 2 with .rows(1))
        v[22 * n + Self.minusPixel] = -.infinity
        v[3 * n + Self.bothPixel] = .infinity           // +∞ then −∞ at the same pixel: NaN on both paths
        v[18 * n + Self.bothPixel] = -.infinity
        return try MemoryCube(shape: shape, values: v)
    }

    private func expectSame(_ gpu: [Double], _ cpu: [Double], _ label: String) {
        #expect(gpu.count == cpu.count)
        var finiteG = [Double](), finiteC = [Double]()
        for k in cpu.indices {
            if cpu[k].isFinite {
                finiteG.append(gpu[k]); finiteC.append(cpu[k])
            } else if cpu[k].isNaN {
                #expect(gpu[k].isNaN, "\(label): pixel \(k) is NaN on the CPU, \(gpu[k]) on Metal")
            } else {
                #expect(gpu[k] == cpu[k], "\(label): pixel \(k) is \(cpu[k]) on the CPU, \(gpu[k]) on Metal")
            }
        }
        _ = assertRelative(finiteG, finiteC, 1e-5, "\(label): finite pixels")
    }

    @Test func infinitiesInSelectedPatternsMatchTheCPU() async throws {
        let cube = try hotCube()
        let count = cube.shape.scanCount
        let mask = (0..<count).map { $0 % 4 != 1 }               // keeps positions 3, 7, 18 and 22
        let weights = (0..<count).map { 0.5 + Double($0 % 5) * 0.25 }
        for tiling in [ScanTiling.rows(1), .rows(4), .automatic] {
            for statistic in [DiffractionStatistic.mean, .max] {
                for (name, scan) in [("all", ScanSelection.all), ("mask", .mask(mask)), ("weights", .weights(weights))] {
                    let cpu = try await VirtualDiffraction.pattern(of: cube, statistic: statistic, scan: scan, tiling: tiling)
                    let gpu = try await VirtualDiffraction.pattern(of: cube, statistic: statistic, scan: scan, path: .metal,
                                                                   tiling: tiling)
                    let label = "\(statistic) \(name) \(tiling)"
                    if statistic == .mean {
                        #expect(cpu.values[Self.plusPixel] == .infinity, "\(label): the CPU reference")
                        #expect(cpu.values[Self.minusPixel] == -.infinity, "\(label): the CPU reference")
                    }
                    #expect(cpu.values[Self.bothPixel].isNaN || statistic == .max, "\(label): the CPU reference")
                    expectSame(gpu.values, cpu.values, label)
                }
            }
        }
    }

    @Test func centreOfMassSumsWithInfinitiesMatchTheCPU() async throws {
        let cube = try hotCube()
        let geometry = DetectorGeometry.circle(centerRow: 4, centerColumn: 4, radius: 3)
        for mask in [nil, geometry] {
            for tiling in [ScanTiling.rows(1), .automatic] {
                let cpu = try await CenterOfMass.measure(of: cube, mask: mask, path: .cpu, tiling: tiling)
                let gpu = try await CenterOfMass.measure(of: cube, mask: mask, path: .metal, tiling: tiling)
                let label = "centre of mass, mask \(mask == nil ? "none" : "disk") \(tiling)"
                #expect(cpu.intensity.values[7] == .infinity, "\(label): the CPU reference")
                expectSame(gpu.intensity.values, cpu.intensity.values, "\(label) intensity")
                expectSame(gpu.momentRow.values, cpu.momentRow.values, "\(label) row moment")
                expectSame(gpu.momentColumn.values, cpu.momentColumn.values, "\(label) column moment")
                expectSame(gpu.row.values, cpu.row.values, "\(label) row")
                expectSame(gpu.column.values, cpu.column.values, "\(label) column")
            }
        }
    }

    @Test func anInfinityInsideADetectorMaskMatchesTheCPU() async throws {
        let cube = try hotCube()
        let geometry = DetectorGeometry.circle(centerRow: 4, centerColumn: 4, radius: 3)
        for tiling in [ScanTiling.rows(1), .automatic] {
            let cpu = try await VirtualImaging.image(of: cube, geometry: geometry, path: .cpu, tiling: tiling)
            let gpu = try await VirtualImaging.image(of: cube, geometry: geometry, path: .metal, tiling: tiling)
            #expect(cpu.values[7] == .infinity, "the CPU reference: the hot pixel lies inside the disk")
            expectSame(gpu.values, cpu.values, "virtual image \(tiling)")
        }
    }
}
