//
//  MappedCube.swift
//  The lazy cube: a file region mapped read-only and private; a read converts only the requested elements to Float32.
//  Nothing is read until a pattern or a tile is asked for.
//
//  Conversions (all native little-endian): float32 is copied; float64 rounds to nearest even (vDSP_vdpsp, as NumPy's
//  astype(np.float32)); the 8-, 16- and 32-bit integers convert with vDSP (exact up to 2^24); the 64-bit integers with a
//  scalar loop (Swift's Float(_:) rounds to nearest even). Big-endian data is refused by name.
//

import Accelerate
import Darwin
import Foundation

/// A stored element type the lazy reader converts to Float32.
public enum ElementType: String, Sendable, Hashable, CaseIterable, Codable {
    case float32, float64, uint8, int8, uint16, int16, uint32, int32, uint64, int64

    /// Bytes per element.
    public var byteSize: Int {
        switch self {
        case .uint8, .int8: 1
        case .uint16, .int16: 2
        case .float32, .uint32, .int32: 4
        case .float64, .uint64, .int64: 8
        }
    }

    /// Whether every value of this type converts to Float32 exactly (the integers up to 2^24 do; this answers for the
    /// whole type, not for a particular cube).
    public var convertsExactlyToFloat32: Bool {
        switch self {
        case .float32, .uint8, .int8, .uint16, .int16: true
        case .float64, .uint32, .int32, .uint64, .int64: false
        }
    }

    /// The NumPy `descr` of this type, little-endian.
    public var npyDescriptor: String {
        switch self {
        case .float32: "<f4"
        case .float64: "<f8"
        case .uint8: "|u1"
        case .int8: "|i1"
        case .uint16: "<u2"
        case .int16: "<i2"
        case .uint32: "<u4"
        case .int32: "<i4"
        case .uint64: "<u8"
        case .int64: "<i8"
        }
    }

    /// Converts `count` elements at `source` (aligned to `byteSize`) to Float32 into `destination`.
    package func convert(_ source: UnsafeRawPointer, count: Int, into destination: UnsafeMutablePointer<Float>) {
        guard count > 0 else { return }
        let n = vDSP_Length(count)
        switch self {
        case .float32:
            destination.update(from: source.assumingMemoryBound(to: Float.self), count: count)
        case .float64:
            vDSP_vdpsp(source.assumingMemoryBound(to: Double.self), 1, destination, 1, n)
        case .uint8:
            vDSP_vfltu8(source.assumingMemoryBound(to: UInt8.self), 1, destination, 1, n)
        case .int8:
            vDSP_vflt8(source.assumingMemoryBound(to: Int8.self), 1, destination, 1, n)
        case .uint16:
            vDSP_vfltu16(source.assumingMemoryBound(to: UInt16.self), 1, destination, 1, n)
        case .int16:
            vDSP_vflt16(source.assumingMemoryBound(to: Int16.self), 1, destination, 1, n)
        case .uint32:
            vDSP_vfltu32(source.assumingMemoryBound(to: UInt32.self), 1, destination, 1, n)
        case .int32:
            vDSP_vflt32(source.assumingMemoryBound(to: Int32.self), 1, destination, 1, n)
        case .uint64:
            let s = source.assumingMemoryBound(to: UInt64.self)
            for i in 0..<count { destination[i] = Float(s[i]) }
        case .int64:
            let s = source.assumingMemoryBound(to: Int64.self)
            for i in 0..<count { destination[i] = Float(s[i]) }
        }
    }
}

/// A cube read lazily from a memory-mapped file.
public actor MappedCube: CubeSource {
    public nonisolated let shape: CubeShape
    public nonisolated let elementType: ElementType
    /// The file and the byte offset of the first element.
    public nonisolated let url: URL
    public nonisolated let dataOffset: Int

    /// The read-only mapping, unmapped when the last reference goes.
    private let mapping: Mapping

    /// Maps `url` read-only from `dataOffset`, holding a cube of `shape` stored as `elementType`.
    public init(url: URL, dataOffset: Int, elementType: ElementType, shape: CubeShape) throws {
        let path = url.path
        guard dataOffset >= 0 else { throw CuboError.misalignedOffset(dataOffset, elementSize: elementType.byteSize) }
        guard dataOffset % elementType.byteSize == 0 else {
            throw CuboError.misalignedOffset(dataOffset, elementSize: elementType.byteSize)
        }
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw CuboError.cannotOpen(path: path, reason: String(cString: strerror(errno))) }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0 else {
            throw CuboError.cannotOpen(path: path, reason: String(cString: strerror(errno)))
        }
        let fileSize = Int(info.st_size)
        let (cubeBytes, overflow) = shape.elementCount.multipliedReportingOverflow(by: elementType.byteSize)
        guard !overflow else { throw CuboError.invalidShape(shape.extents) }
        let (needed, overflow2) = dataOffset.addingReportingOverflow(cubeBytes)
        guard !overflow2, needed <= fileSize else {
            throw CuboError.fileTooShort(path: path, needed: overflow2 ? Int.max : needed, available: fileSize)
        }
        let page = Int(getpagesize())
        let mappedOffset = (dataOffset / page) * page
        let length = needed - mappedOffset
        guard let base = mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, off_t(mappedOffset)),
              base != MAP_FAILED else {
            throw CuboError.cannotOpen(path: path, reason: "mmap: \(String(cString: strerror(errno)))")
        }
        _ = madvise(base, length, MADV_SEQUENTIAL)
        self.mapping = Mapping(base: base, length: length, cubeStart: UnsafeRawPointer(base) + (dataOffset - mappedOffset))
        self.shape = shape
        self.elementType = elementType
        self.url = url
        self.dataOffset = dataOffset
    }

    /// The rank-4 array in a NumPy `.npy` file: C order, little-endian, a supported element type.
    public init(npy url: URL) throws {
        let header = try NPYFile.header(of: url)
        guard header.shape.count == 4 else { throw CuboError.wrongRank(expected: 4, received: header.shape.count) }
        try self.init(url: url, dataOffset: header.dataOffset, elementType: header.elementType,
                      shape: try CubeShape(header.shape))
    }

    public func readPattern(row: Int, column: Int) throws -> DiffractionPattern {
        try shape.requirePosition(row: row, column: column)
        let count = shape.detectorCount
        let first = shape.scanIndex(row: row, column: column) * count
        let values = convert(firstElement: first, count: count)
        return try DiffractionPattern(detectorRows: shape.detectorRows, detectorColumns: shape.detectorColumns,
                                      values: values)
    }

    public func readScanTile(rows: Range<Int>) throws -> ScanTile {
        try shape.requireRows(rows)
        let stride = shape.elementsPerScanRow
        let values = convert(firstElement: rows.lowerBound * stride, count: rows.count * stride)
        return try ScanTile(shape: shape, rows: rows, values: values)
    }

    private func convert(firstElement: Int, count: Int) -> [Float] {
        let source = mapping.cubeStart + firstElement * elementType.byteSize
        let type = elementType
        return [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
            if let base = buffer.baseAddress { type.convert(source, count: count, into: base) }
            initialized = count
        }
    }
}

/// A read-only, private file mapping: `base` is page-aligned, the cube starts at `cubeStart`. Never written through after
/// construction, which is what the unchecked conformance states; unmapped in `deinit`.
final class Mapping: @unchecked Sendable {
    let base: UnsafeMutableRawPointer
    let length: Int
    let cubeStart: UnsafeRawPointer

    init(base: UnsafeMutableRawPointer, length: Int, cubeStart: UnsafeRawPointer) {
        self.base = base
        self.length = length
        self.cubeStart = cubeStart
    }

    deinit { munmap(base, length) }
}
