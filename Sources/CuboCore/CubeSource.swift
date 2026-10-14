//
//  CubeSource.swift
//  How a computation gets its pixels: one pattern, or contiguous scan rows (registration 1.1). Every read returns
//  Float32 whatever the stored element type.
//
//  Extracted from mac4DSTEM 7448401b, Core/Data/FourDDataSource.swift (`FourDDataSource`, `FourDScanTile`), with the
//  app's load views, calibration, probes and file attributes left in the app: a source here is only a shape and reads.
//

/// One diffraction pattern, `detectorRows × detectorColumns` values, row-major.
public struct DiffractionPattern: Sendable, Equatable {
    public let detectorRows: Int
    public let detectorColumns: Int
    public let values: [Float]

    public init(detectorRows: Int, detectorColumns: Int, values: [Float]) throws {
        guard detectorRows > 0, detectorColumns > 0 else {
            throw CuboError.invalidShape([detectorRows, detectorColumns])
        }
        guard values.count == detectorRows * detectorColumns else {
            throw CuboError.valueCountMismatch(expected: detectorRows * detectorColumns, received: values.count)
        }
        self.detectorRows = detectorRows
        self.detectorColumns = detectorColumns
        self.values = values
    }

    /// The value at detector row `row`, column `column`.
    public subscript(row: Int, column: Int) -> Float { values[row * detectorColumns + column] }
}

/// Contiguous scan rows of a cube: `[rows.count][scanColumns][detectorRows][detectorColumns]` values.
public struct ScanTile: Sendable, Equatable {
    /// The cube's shape (not the tile's).
    public let shape: CubeShape
    /// Which scan rows these are.
    public let rows: Range<Int>
    public let values: [Float]

    public init(shape: CubeShape, rows: Range<Int>, values: [Float]) throws {
        guard rows.lowerBound >= 0, rows.upperBound <= shape.scanRows, !rows.isEmpty else {
            throw CuboError.outOfRange("Scan rows \(rows.lowerBound)..<\(rows.upperBound)")
        }
        let expected = rows.count * shape.elementsPerScanRow
        guard values.count == expected else {
            throw CuboError.valueCountMismatch(expected: expected, received: values.count)
        }
        self.shape = shape
        self.rows = rows
        self.values = values
    }

    /// Scan positions in the tile, `rows.count · scanColumns`.
    public var positionCount: Int { rows.count * shape.scanColumns }
}

/// What every cube conforms to. An actor: reads are serialised per source.
public protocol CubeSource: Actor {
    /// The cube's shape; fixed for the source's lifetime.
    nonisolated var shape: CubeShape { get }
    /// One pattern at scan position (`row`, `column`).
    func readPattern(row: Int, column: Int) throws -> DiffractionPattern
    /// Contiguous scan rows; `rows` must lie inside the cube and must not be empty.
    func readScanTile(rows: Range<Int>) throws -> ScanTile
}

extension CubeShape {
    /// Throws unless (`row`, `column`) is a scan position of this shape.
    package func requirePosition(row: Int, column: Int) throws {
        guard (0..<scanRows).contains(row), (0..<scanColumns).contains(column) else {
            throw CuboError.outOfRange("Scan position (\(row), \(column))")
        }
    }

    /// Throws unless `rows` is a non-empty range of this shape's scan rows.
    package func requireRows(_ rows: Range<Int>) throws {
        guard rows.lowerBound >= 0, rows.upperBound <= scanRows, !rows.isEmpty else {
            throw CuboError.outOfRange("Scan rows \(rows.lowerBound)..<\(rows.upperBound)")
        }
    }
}

/// How a whole-cube pass walks the scan axis (registration 1.1).
public enum ScanTiling: Sendable, Hashable {
    /// The largest row count whose Float32 tile is at most 64 MiB, at least 1: a function of the shape alone, never of
    /// the machine, so a tiled result is the same on every Mac.
    case automatic
    /// A fixed number of scan rows per tile (the last tile may be shorter).
    case rows(Int)

    /// The tile budget of `.automatic`, in bytes.
    public static let automaticTileBytes = 64 * 1024 * 1024

    /// Rows per tile for `shape`, between 1 and `shape.scanRows`.
    public func rowsPerTile(for shape: CubeShape) throws -> Int {
        switch self {
        case .automatic:
            let rowBytes = shape.elementsPerScanRow * MemoryLayout<Float>.stride
            return max(1, min(shape.scanRows, Self.automaticTileBytes / max(1, rowBytes)))
        case .rows(let n):
            guard n > 0 else { throw CuboError.invalidTiling(n) }
            return min(n, shape.scanRows)
        }
    }

    /// The row ranges of a pass over `shape`, in scan order.
    public func ranges(for shape: CubeShape) throws -> [Range<Int>] {
        let step = try rowsPerTile(for: shape)
        return stride(from: 0, to: shape.scanRows, by: step).map { $0..<min(shape.scanRows, $0 + step) }
    }
}
