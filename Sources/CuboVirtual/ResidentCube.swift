//
//  ResidentCube.swift
//  A cube held in one Metal buffer (Float32, storage order) so that repeated `.metal` reductions read it in place instead of
//  copying every tile to the GPU on each call. On Apple silicon the buffer is shared memory, so the CPU reads it too: a
//  `ResidentCube` is an ordinary ``CubeSource`` and every CPU function accepts it.
//

import CuboCore
import Foundation
import Metal

/// The Metal buffer behind a ``ResidentCube``; immutable after construction.
final class ResidentStorage: @unchecked Sendable {
    let buffer: MTLBuffer
    init(buffer: MTLBuffer) { self.buffer = buffer }
}

/// A cube resident in GPU-accessible memory. `.metal` virtual imaging, mean and max, centre of mass and the selected-area
/// pattern run over its buffer without a copy; reads (`readPattern`, `readScanTile`) copy out of it.
public actor ResidentCube: CubeSource {
    public nonisolated let shape: CubeShape
    nonisolated let storage: ResidentStorage

    /// Loads every pattern of `source`, tile by tile, into one buffer. Throws `CuboVirtualError.noMetalDevice` without a
    /// device and `CuboVirtualError.metal` when the cube is larger than the device's largest buffer
    /// (`maxBufferLength`; check ``fits(_:)`` first).
    public init(loading source: some CubeSource, tiling: ScanTiling = .automatic) async throws {
        let metal = try MetalReducers.context()
        let shape = source.shape
        guard Self.fits(shape) else {
            throw CuboVirtualError.metal("the cube (\(shape)) is \(shape.elementCount * 4) bytes; the largest Metal buffer here is \(metal.device.maxBufferLength)")
        }
        let buffer = try metal.buffer(floats: shape.elementCount, label: "CuboKit resident cube \(shape)")
        let base = buffer.contents().bindMemory(to: Float.self, capacity: shape.elementCount)
        for range in try tiling.ranges(for: shape) {
            try checkInterrupted()
            let tile = try await source.readScanTile(rows: range)
            tile.values.withUnsafeBufferPointer { v in
                (base + range.lowerBound * shape.elementsPerScanRow).update(from: v.baseAddress!, count: v.count)
            }
        }
        self.shape = shape
        self.storage = ResidentStorage(buffer: buffer)
    }

    /// Whether a cube of this shape fits one Metal buffer on this machine (`false` without a Metal device).
    public static func fits(_ shape: CubeShape) -> Bool {
        guard case .success(let metal) = MetalContext.shared else { return false }
        return shape.elementCount * MemoryLayout<Float>.stride <= metal.device.maxBufferLength
    }

    /// Bytes held on the GPU side.
    public nonisolated var byteCount: Int { shape.elementCount * MemoryLayout<Float>.stride }

    public func readPattern(row: Int, column: Int) throws -> DiffractionPattern {
        try shape.requirePosition(row: row, column: column)
        let start = shape.scanIndex(row: row, column: column) * shape.detectorCount
        return try DiffractionPattern(detectorRows: shape.detectorRows, detectorColumns: shape.detectorColumns,
                                      values: values(start, shape.detectorCount))
    }

    public func readScanTile(rows: Range<Int>) throws -> ScanTile {
        try shape.requireRows(rows)
        return try ScanTile(shape: shape, rows: rows,
                            values: values(rows.lowerBound * shape.elementsPerScanRow, rows.count * shape.elementsPerScanRow))
    }

    private func values(_ start: Int, _ count: Int) -> [Float] {
        let base = storage.buffer.contents().bindMemory(to: Float.self, capacity: shape.elementCount)
        return Array(UnsafeBufferPointer(start: base + start, count: count))
    }
}
