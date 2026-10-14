//
//  NPYFile.swift
//  NumPy's .npy container, versions 1.0, 2.0 and 3.0 (registration 1.1): the magic "\x93NUMPY", the version, the
//  header length (little-endian UInt16 for 1.0, UInt32 for 2.0 and 3.0), then a Python dict literal with 'descr',
//  'fortran_order' and 'shape'. The kit's exchange format with py4DSTEM in slice 1 (np.save(path, datacube.data));
//  HDF5, DM4 and the vendor formats are FormatoKit's.
//

import Foundation

/// What an `.npy` header says.
public struct NPYHeader: Sendable, Equatable {
    public let majorVersion: Int
    public let minorVersion: Int
    public let elementType: ElementType
    public let shape: [Int]
    /// Byte offset of the first element.
    public let dataOffset: Int

    /// Elements in the array (1 for a scalar).
    public var elementCount: Int { shape.reduce(1, *) }
}

public enum NPYFile {
    static let magic: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]   // \x93NUMPY

    /// Parses the header of the `.npy` file at `url`; reads only the header bytes.
    public static func header(of url: URL) throws -> NPYHeader {
        let handle: FileHandle
        do { handle = try FileHandle(forReadingFrom: url) } catch {
            throw CuboError.cannotOpen(path: url.path, reason: error.localizedDescription)
        }
        defer { try? handle.close() }
        let prefix = try handle.read(upToCount: 12) ?? Data()
        guard prefix.count >= 10, Array(prefix.prefix(6)) == magic else { throw CuboError.notNPY("no NUMPY magic") }
        let major = Int(prefix[6]), minor = Int(prefix[7])
        let headerLength: Int
        let headerStart: Int
        switch major {
        case 1:
            headerLength = Int(prefix[8]) | Int(prefix[9]) << 8
            headerStart = 10
        case 2, 3:
            guard prefix.count >= 12 else { throw CuboError.notNPY("truncated version \(major) preamble") }
            headerLength = Int(prefix[8]) | Int(prefix[9]) << 8 | Int(prefix[10]) << 16 | Int(prefix[11]) << 24
            headerStart = 12
        default:
            throw CuboError.notNPY("version \(major).\(minor)")
        }
        guard headerLength > 0, headerLength < 1 << 26 else { throw CuboError.notNPY("header length \(headerLength)") }
        try handle.seek(toOffset: UInt64(headerStart))
        let headerData = try handle.read(upToCount: headerLength) ?? Data()
        guard headerData.count == headerLength,
              let text = String(data: headerData, encoding: major == 3 ? .utf8 : .isoLatin1) else {
            throw CuboError.notNPY("truncated header")
        }
        let fields = try parseDictionary(text)
        guard let descr = fields.descr else { throw CuboError.notNPY("no 'descr'") }
        guard let fortran = fields.fortranOrder else { throw CuboError.notNPY("no 'fortran_order'") }
        guard let shape = fields.shape else { throw CuboError.notNPY("no 'shape'") }
        guard !fortran else { throw CuboError.fortranOrderNotSupported }
        let type = try elementType(descr)
        guard shape.allSatisfy({ $0 >= 0 }) else { throw CuboError.notNPY("negative extent in 'shape'") }
        // The element and byte counts must fit an Int (Gate B, 2026-10-15: an unchecked product trapped).
        var bytes = type.byteSize
        for e in shape {
            let (product, overflow) = bytes.multipliedReportingOverflow(by: e)
            guard !overflow else { throw CuboError.notNPY("'shape' \(shape) overflows the byte count") }
            bytes = product
        }
        return NPYHeader(majorVersion: major, minorVersion: minor, elementType: type, shape: shape,
                         dataOffset: headerStart + headerLength)
    }

    /// The element type of a little-endian (or byte-order-free) `descr`; big-endian is refused by name.
    public static func elementType(_ descr: String) throws -> ElementType {
        guard let order = descr.first, descr.count >= 3 else { throw CuboError.unsupportedElementType(descr) }
        let code = String(descr.dropFirst())
        if order == ">" {
            if code == "u1" || code == "i1" { return try elementType("|" + code) }
            throw CuboError.bigEndianNotSupported(descr)
        }
        guard order == "<" || order == "|" || order == "=" else { throw CuboError.unsupportedElementType(descr) }
        switch code {
        case "f4": return .float32
        case "f8": return .float64
        case "u1": return .uint8
        case "i1": return .int8
        case "u2": return .uint16
        case "i2": return .int16
        case "u4": return .uint32
        case "i4": return .int32
        case "u8": return .uint64
        case "i8": return .int64
        default: throw CuboError.unsupportedElementType(descr)
        }
    }

    /// Reads a whole `.npy` array of any rank and supported type as Double (exact for every supported type).
    public static func readDoubles(_ url: URL) throws -> (shape: [Int], values: [Double]) {
        let header = try header(of: url)
        let count = header.elementCount
        let data: Data
        do { data = try Data(contentsOf: url, options: .alwaysMapped) } catch {
            throw CuboError.cannotOpen(path: url.path, reason: error.localizedDescription)
        }
        let (needed, overflow) = header.dataOffset.addingReportingOverflow(count * header.elementType.byteSize)
        guard !overflow, data.count >= needed else {
            throw CuboError.fileTooShort(path: url.path, needed: needed, available: data.count)
        }
        let values: [Double] = data.withUnsafeBytes { raw in
            let base = raw.baseAddress! + header.dataOffset
            func load<T>(_: T.Type, _ f: (T) -> Double) -> [Double] {
                (0..<count).map { f(base.loadUnaligned(fromByteOffset: $0 * MemoryLayout<T>.size, as: T.self)) }
            }
            switch header.elementType {
            case .float32: return load(Float.self) { Double($0) }
            case .float64: return load(Double.self) { $0 }
            case .uint8: return load(UInt8.self) { Double($0) }
            case .int8: return load(Int8.self) { Double($0) }
            case .uint16: return load(UInt16.self) { Double($0) }
            case .int16: return load(Int16.self) { Double($0) }
            case .uint32: return load(UInt32.self) { Double($0) }
            case .int32: return load(Int32.self) { Double($0) }
            case .uint64: return load(UInt64.self) { Double($0) }
            case .int64: return load(Int64.self) { Double($0) }
            }
        }
        return (header.shape, values)
    }

    /// Writes `values` (C order) as a version 1.0 `.npy` file of `shape`, as `np.save` does (header padded with spaces
    /// to a 64-byte boundary, ending in a newline). Float32 only; the kit's own fixtures and exports.
    public static func write(_ values: [Float], shape: [Int], to url: URL) throws {
        guard shape.reduce(1, *) == values.count else {
            throw CuboError.valueCountMismatch(expected: shape.reduce(1, *), received: values.count)
        }
        let shapeText = shape.count == 1 ? "(\(shape[0]),)" : "(" + shape.map(String.init).joined(separator: ", ") + ")"
        var dict = "{'descr': '<f4', 'fortran_order': False, 'shape': \(shapeText), }"
        let unpadded = 10 + dict.utf8.count + 1
        dict += String(repeating: " ", count: (64 - unpadded % 64) % 64) + "\n"
        var data = Data(magic + [1, 0])
        let length = UInt16(dict.utf8.count)
        data.append(contentsOf: [UInt8(length & 0xFF), UInt8(length >> 8)])
        data.append(contentsOf: Array(dict.utf8))
        values.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        do { try data.write(to: url) } catch {
            throw CuboError.cannotOpen(path: url.path, reason: error.localizedDescription)
        }
    }

    // MARK: - The header dictionary

    struct Fields { var descr: String?; var fortranOrder: Bool?; var shape: [Int]? }

    /// Parses `{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3, 5, 7), }` and the spacing variants np.save
    /// and older writers produce. Unknown keys are refused: a header this reader half-understands is not read.
    static func parseDictionary(_ text: String) throws -> Fields {
        var s = Substring(text.trimmingCharacters(in: .whitespacesAndNewlines))
        guard s.first == "{", s.last == "}" else { throw CuboError.notNPY("header is not a dict literal") }
        s = s.dropFirst().dropLast()
        var fields = Fields()
        func skipSpace() { while let c = s.first, c == " " || c == "\t" || c == "\n" { s = s.dropFirst() } }
        func quoted() throws -> String {
            skipSpace()
            guard let q = s.first, q == "'" || q == "\"" else { throw CuboError.notNPY("expected a quoted string") }
            s = s.dropFirst()
            guard let end = s.firstIndex(of: q) else { throw CuboError.notNPY("unterminated string") }
            let value = String(s[..<end])
            s = s[s.index(after: end)...]
            return value
        }
        func expect(_ c: Character) throws {
            skipSpace()
            guard s.first == c else { throw CuboError.notNPY("expected '\(c)'") }
            s = s.dropFirst()
        }
        while true {
            skipSpace()
            if s.isEmpty { break }
            let key = try quoted()
            try expect(":")
            skipSpace()
            switch key {
            case "descr":
                guard fields.descr == nil else { throw CuboError.notNPY("'descr' twice") }
                fields.descr = try quoted()
            case "fortran_order":
                guard fields.fortranOrder == nil else { throw CuboError.notNPY("'fortran_order' twice") }
                if s.hasPrefix("True") { fields.fortranOrder = true; s = s.dropFirst(4) }
                else if s.hasPrefix("False") { fields.fortranOrder = false; s = s.dropFirst(5) }
                else { throw CuboError.notNPY("'fortran_order' is not True or False") }
            case "shape":
                guard fields.shape == nil else { throw CuboError.notNPY("'shape' twice") }
                try expect("(")
                guard let close = s.firstIndex(of: ")") else { throw CuboError.notNPY("unterminated 'shape'") }
                let inner = s[..<close]
                s = s[s.index(after: close)...]
                let parts = inner.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty }
                var shape: [Int] = []
                for p in parts {
                    guard let v = Int(p.hasSuffix("L") ? String(p.dropLast()) : p) else {
                        throw CuboError.notNPY("'shape' holds \(p)")
                    }
                    shape.append(v)
                }
                fields.shape = shape
            default:
                throw CuboError.notNPY("unknown key '\(key)'")
            }
            skipSpace()
            if s.first == "," { s = s.dropFirst(); continue }
            skipSpace()
            guard s.isEmpty else { throw CuboError.notNPY("expected ',' after '\(key)'") }
        }
        return fields
    }
}
