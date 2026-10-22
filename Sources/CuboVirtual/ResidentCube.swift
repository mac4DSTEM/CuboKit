//
//  ResidentCube.swift
//  A cube held in one Metal buffer (Float32, storage order) so that repeated `.metal` reductions read it in place instead of
//  copying every tile to the GPU on each call. On Apple silicon the buffer is shared memory, so the CPU reads it too: a
//  `ResidentCube` is an ordinary ``CubeSource`` and every CPU function accepts it.
//

import CuboCore
import Foundation
import Metal

/// Thrown by ``ResidentCube/init(loading:tiling:)`` for a cube that cannot be held resident (but a machine without a Metal device
/// throws `CuboVirtualError.noMetalDevice`, as before).
public enum ResidentCubeError: Error, Equatable, Sendable, LocalizedError {
    case unavailable(ResidentCube.UnavailableReason)

    public var errorDescription: String? {
        if case .unavailable(let reason) = self { return reason.message }
        return nil
    }
}

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
    /// device and ``ResidentCubeError/unavailable(_:)`` when the cube is larger than the memory cap
    /// (``memoryCapFraction`` of the physical memory) or than the device's largest buffer (`maxBufferLength`); ask
    /// ``availability(of:)`` first to get the reason as a value and its words as ``UnavailableReason/message``.
    public init(loading source: some CubeSource, tiling: ScanTiling = .automatic) async throws {
        let metal = try MetalReducers.context()
        let shape = source.shape
        if case .unavailable(let reason) = Self.availability(of: shape) { throw ResidentCubeError.unavailable(reason) }
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

    /// The share of the machine's physical memory one resident cube may take (half: the Metal buffer is shared memory on Apple
    /// silicon, and the rest is the app's, the system's and the other cubes').
    public static let memoryCapFraction = 0.5

    /// Why a cube cannot be held resident here.
    public enum UnavailableReason: Equatable, Sendable {
        /// This machine has no Metal device.
        case noMetalDevice
        /// The cube (`bytes`, Float32) is larger than the memory cap (`cap` bytes, ``memoryCapFraction`` of `physicalMemory`).
        case exceedsMemoryCap(bytes: Int, cap: Int, physicalMemory: Int)
        /// The cube is larger than the device's largest single buffer.
        case exceedsMetalBufferLimit(bytes: Int, limit: Int)

        /// The reason in a sentence, for a status line or a refusal.
        public var message: String {
            func gb(_ n: Int) -> String { String(format: "%.1f GB", Double(n) / 1e9) }
            switch self {
            case .noMetalDevice: return "This Mac has no Metal device, so the cube cannot be held resident."
            case .exceedsMemoryCap(let bytes, let cap, let physical):
                return "The cube is \(gb(bytes)) in memory; a resident cube may use at most half of this Mac's memory (\(gb(cap)) of \(gb(physical)))."
            case .exceedsMetalBufferLimit(let bytes, let limit):
                return "The cube is \(gb(bytes)); the largest single Metal buffer on this Mac is \(gb(limit))."
            }
        }
    }

    public enum Availability: Equatable, Sendable {
        case available
        case unavailable(UnavailableReason)
    }

    /// Whether a cube of this shape can be held resident on this machine, or why not. Checks, in order, the memory cap
    /// (``memoryCapFraction`` of `physicalMemory`, this machine's by default) and then the Metal device and its largest buffer.
    public static func availability(of shape: CubeShape, physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) -> Availability {
        let bytes = shape.elementCount * MemoryLayout<Float>.stride   // `CubeShape` guarantees elementCount * 8 fits an Int
        let physical = Int(clamping: physicalMemory)
        let cap = Int(Double(physical) * memoryCapFraction)
        guard bytes <= cap else { return .unavailable(.exceedsMemoryCap(bytes: bytes, cap: cap, physicalMemory: physical)) }
        guard case .success(let metal) = MetalContext.shared else { return .unavailable(.noMetalDevice) }
        guard bytes <= metal.device.maxBufferLength else {
            return .unavailable(.exceedsMetalBufferLimit(bytes: bytes, limit: metal.device.maxBufferLength))
        }
        return .available
    }

    /// Whether ``availability(of:)`` is `.available` (`false` without a Metal device, above the memory cap or the buffer limit).
    public static func fits(_ shape: CubeShape) -> Bool { availability(of: shape) == .available }

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
