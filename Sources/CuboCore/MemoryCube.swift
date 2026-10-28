//
//  MemoryCube.swift
//  The eager cube: the whole array as one [Float] in storage order. Equal to the lazy
//  MappedCube of the same bytes, bit for bit (tested in CuboCoreTests).
//

/// A cube held in memory as Float32, `[ry][rx][qy][qx]`.
public actor MemoryCube: CubeSource {
    public nonisolated let shape: CubeShape
    /// Every value, in storage order.
    public nonisolated let values: [Float]

    /// A cube from its values in storage order.
    public init(shape: CubeShape, values: [Float]) throws {
        guard values.count == shape.elementCount else {
            throw CuboError.valueCountMismatch(expected: shape.elementCount, received: values.count)
        }
        self.shape = shape
        self.values = values
    }

    /// Every value of `source`, read tile by tile (the eager path of a lazy source).
    public init(loading source: some CubeSource, tiling: ScanTiling = .automatic) async throws {
        let shape = source.shape
        var all = [Float]()
        all.reserveCapacity(shape.elementCount)
        for range in try tiling.ranges(for: shape) {
            try Task.checkCancellation()
            let tile = try await source.readScanTile(rows: range)
            all.append(contentsOf: tile.values)
        }
        try self.init(shape: shape, values: all)
    }

    public func readPattern(row: Int, column: Int) throws -> DiffractionPattern {
        try shape.requirePosition(row: row, column: column)
        let start = shape.scanIndex(row: row, column: column) * shape.detectorCount
        return try DiffractionPattern(detectorRows: shape.detectorRows, detectorColumns: shape.detectorColumns,
                                      values: Array(values[start..<(start + shape.detectorCount)]))
    }

    public func readScanTile(rows: Range<Int>) throws -> ScanTile {
        try shape.requireRows(rows)
        let stride = shape.elementsPerScanRow
        return try ScanTile(shape: shape, rows: rows,
                            values: Array(values[(rows.lowerBound * stride)..<(rows.upperBound * stride)]))
    }
}
