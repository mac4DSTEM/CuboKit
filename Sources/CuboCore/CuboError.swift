//
//  CuboError.swift
//  Every refusal of CuboCore. A refusal is never a fallback: a read that cannot be honoured stops (registration 1.1).
//

import Foundation

public enum CuboError: Error, Equatable, Sendable, LocalizedError {
    /// A shape with a non-positive extent, a rank other than 4, or an element count that overflows.
    case invalidShape([Int])
    /// The number of values does not equal the shape's element count.
    case valueCountMismatch(expected: Int, received: Int)
    /// A scan position or a row range outside the cube.
    case outOfRange(String)
    /// The file could not be opened or mapped.
    case cannotOpen(path: String, reason: String)
    /// The file is shorter than the cube it is said to hold.
    case fileTooShort(path: String, needed: Int, available: Int)
    /// The data offset is not a multiple of the element size.
    case misalignedOffset(Int, elementSize: Int)
    /// Not a NumPy `.npy` file, or a header this reader does not parse.
    case notNPY(String)
    /// An `.npy` element type outside the supported list.
    case unsupportedElementType(String)
    /// Big-endian data (`>f4` and the like); slice 2 at the earliest.
    case bigEndianNotSupported(String)
    /// `fortran_order: True`.
    case fortranOrderNotSupported
    /// An `.npy` array whose rank is not the one asked for.
    case wrongRank(expected: Int, received: Int)
    /// A tiling with no rows.
    case invalidTiling(Int)
    /// A crop-and-bin view that cannot be applied to its source.
    case invalidView(String)

    public var errorDescription: String? {
        switch self {
        case .invalidShape(let e):
            return "A cube needs four positive extents; got \(e.map(String.init).joined(separator: " × "))."
        case .valueCountMismatch(let expected, let received):
            return "The cube's shape holds \(expected) values, but \(received) were given."
        case .outOfRange(let what):
            return "\(what) is outside the cube."
        case .cannotOpen(let path, let reason):
            return "Could not open \((path as NSString).lastPathComponent): \(reason)."
        case .fileTooShort(let path, let needed, let available):
            return "\((path as NSString).lastPathComponent) holds \(available) bytes, but its cube needs \(needed)."
        case .misalignedOffset(let offset, let size):
            return "The data offset \(offset) is not a multiple of the element size \(size)."
        case .notNPY(let detail):
            return "Not a NumPy .npy file this reader understands: \(detail)."
        case .unsupportedElementType(let d):
            return "The element type \(d) is not supported."
        case .bigEndianNotSupported(let d):
            return "Big-endian data (\(d)) is not supported."
        case .fortranOrderNotSupported:
            return "Fortran-ordered .npy files are not supported; save the array in C order."
        case .wrongRank(let expected, let received):
            return "Expected an array of rank \(expected), got rank \(received)."
        case .invalidTiling(let rows):
            return "A tiling needs at least one scan row per tile; got \(rows)."
        case .invalidView(let detail):
            return "The crop-and-bin view cannot be applied: \(detail)."
        }
    }
}
