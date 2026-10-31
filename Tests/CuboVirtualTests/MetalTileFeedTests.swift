import CuboCore
import CuboKitTestSupport
@testable import CuboVirtual
import Foundation
import Metal
import Testing

/// A source that is neither a ``MemoryCube`` nor a ``ResidentCube``: the `.metal` path stages its tiles.
private actor ForwardingCube: CubeSource {
    nonisolated let shape: CubeShape
    private let inner: MemoryCube
    init(_ inner: MemoryCube) { self.inner = inner; self.shape = inner.shape }
    func readPattern(row: Int, column: Int) async throws -> DiffractionPattern { try await inner.readPattern(row: row, column: column) }
    func readScanTile(rows: Range<Int>) async throws -> ScanTile { try await inner.readScanTile(rows: rows) }
}

/// 13 x 11 x 48 x 40 Float32 is 1.1 MB, above the no-copy minimum; one infinite and one NaN pixel (the #51 rules).
private func largeCube(nonFinite: Bool = true) throws -> MemoryCube {
    let shape = try CubeShape(scanRows: 13, scanColumns: 11, detectorRows: 48, detectorColumns: 40)
    var state: UInt64 = 3
    var v = (0..<shape.elementCount).map { k -> Float in
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Float(state >> 40) / 4096 - 1500 + Float(k % 7)
    }
    if nonFinite { v[5 * shape.detectorCount + 17] = .infinity; v[40 * shape.detectorCount + 3] = .nan }
    return try MemoryCube(shape: shape, values: v)
}

private func bits(_ v: [Double]) -> [UInt64] { v.map(\.bitPattern) }

@Suite(.serialized, .enabled(if: VirtualDiffraction.metalIsAvailable, "no Metal device on this machine"))
struct MetalTileFeedTests {
    @Test func aLargeMemoryCubeReachesTheGPUWithoutACopy() async throws {
        let metal = try MetalReducers.context()
        let cube = try largeCube()
        #expect(cube.values.count * 4 >= TileFeed.noCopyMinimumBytes)
        let feed = try TileFeed(metal, cube, tiling: .rows(4))
        #expect(feed.kind == .noCopy)
        let (buffer, offset, positions) = try await feed.buffer(cube, 4..<8)
        #expect(positions == 4 * cube.shape.scanColumns)
        cube.values.withUnsafeBufferPointer { v in
            #expect(buffer.contents() + offset == UnsafeMutableRawPointer(mutating: v.baseAddress! + 4 * cube.shape.elementsPerScanRow))
        }
    }

    @Test func otherSourcesAreStagedOrReadInPlace() async throws {
        let metal = try MetalReducers.context()
        let large = try largeCube()
        let small = try MemoryCube(shape: try CubeShape(scanRows: 3, scanColumns: 2, detectorRows: 4, detectorColumns: 5),
                                   values: (0..<120).map(Float.init))
        #expect(try TileFeed(metal, small, tiling: .automatic).kind == .staging)
        #expect(try TileFeed(metal, ForwardingCube(large), tiling: .automatic).kind == .staging)
        #expect(try TileFeed(metal, await ResidentCube(loading: large), tiling: .automatic).kind == .resident)
        // the staging buffer is reused across tiles and holds each tile in turn
        let forwarding = ForwardingCube(large)
        let feed = try TileFeed(metal, forwarding, tiling: .rows(5))
        let a = try await feed.buffer(forwarding, 0..<5), b = try await feed.buffer(forwarding, 10..<13)
        #expect(a.buffer === b.buffer && b.positions == 3 * large.shape.scanColumns)
        #expect(MetalContext.floats(b.buffer, count: 3 * large.shape.elementsPerScanRow).map(\.bitPattern)
                == large.values[(10 * large.shape.elementsPerScanRow)...].map(\.bitPattern))
    }

    /// The source changes where the kernels read, never what they compute: the same bits from a no-copy memory cube, a
    /// staged source and a resident cube, for every reducer and tiling (a short last tile included).
    @Test func everySourceGivesTheSameBits() async throws {
        let memory = try largeCube()
        let staged = ForwardingCube(memory)
        let resident = try await ResidentCube(loading: memory)
        let shape = memory.shape
        let disk = DetectorGeometry.circle(centerRow: 20, centerColumn: 18, radius: 9)
        // scan rows 4 to 7 weigh nothing, so with .rows(4) one tile is skipped
        let weights = (0..<shape.scanCount).map { (4..<8).contains($0 / shape.scanColumns) ? 0 : Double($0 % 5) - 1.5 }
        for tiling in [ScanTiling.rows(1), .rows(4), .automatic] {
            func run(_ s: some CubeSource) async throws -> [[UInt64]] {
                let (mean, max) = try await VirtualDiffraction.meanAndMax(of: s, path: .metal, tiling: tiling)
                let image = try await VirtualImaging.image(of: s, geometry: disk, path: .metal, tiling: tiling)
                let com = try await CenterOfMass.measure(of: s, path: .metal, tiling: tiling)
                let area = try await VirtualDiffraction.pattern(of: s, statistic: .mean, scan: .weights(weights), path: .metal,
                                                                tiling: tiling)
                return [bits(mean.values), bits(max.values), bits(image.values), bits(com.intensity.values),
                        bits(com.momentRow.values), bits(area.values)]
            }
            let fromMemory = try await run(memory)
            #expect(fromMemory == (try await run(staged)), "\(tiling) staged")
            #expect(fromMemory == (try await run(resident)), "\(tiling) resident")
        }
        // and the no-copy path still agrees with the CPU reference (relative 1e-5, the Metal tolerance)
        let finite = try largeCube(nonFinite: false)
        let cpu = try await VirtualImaging.image(of: finite, geometry: disk, path: .cpu)
        let gpu = try await VirtualImaging.image(of: finite, geometry: disk, path: .metal)
        _ = assertRelative(gpu.values, cpu.values, 1e-5, "no-copy virtual image")
    }
}
