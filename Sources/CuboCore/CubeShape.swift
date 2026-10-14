//
//  CubeShape.swift
//  The shape of a 4D-STEM cube and the one index map the kit uses (registration 1.1).
//
//  Storage order is C order [scanRow][scanColumn][detectorRow][detectorColumn], py4DSTEM's (R_Nx, R_Ny, Q_Nx, Q_Ny).
//  py4DSTEM's x (Rx, Qx: the first axis of a pair) is this kit's ROW; its y (Ry, Qy) is this kit's COLUMN. The kit says
//  row and column, never x and y, so the swap mac4DSTEM documents at every call site cannot be silent here.
//

/// The four extents of a cube, scan first, in storage order.
public struct CubeShape: Hashable, Codable, Sendable, CustomStringConvertible {
    /// Scan rows (py4DSTEM `R_Nx`).
    public let scanRows: Int
    /// Scan columns (py4DSTEM `R_Ny`).
    public let scanColumns: Int
    /// Detector rows (py4DSTEM `Q_Nx`).
    public let detectorRows: Int
    /// Detector columns (py4DSTEM `Q_Ny`).
    public let detectorColumns: Int

    /// Every extent must be positive, and the element count must fit an `Int`.
    public init(scanRows: Int, scanColumns: Int, detectorRows: Int, detectorColumns: Int) throws {
        let extents = [scanRows, scanColumns, detectorRows, detectorColumns]
        guard extents.allSatisfy({ $0 > 0 }) else { throw CuboError.invalidShape(extents) }
        var total = 1
        for e in extents {
            let (product, overflow) = total.multipliedReportingOverflow(by: e)
            guard !overflow else { throw CuboError.invalidShape(extents) }
            total = product
        }
        let (bytes, overflow) = total.multipliedReportingOverflow(by: 8)
        guard !overflow, bytes > 0 else { throw CuboError.invalidShape(extents) }
        self.scanRows = scanRows
        self.scanColumns = scanColumns
        self.detectorRows = detectorRows
        self.detectorColumns = detectorColumns
    }

    /// The shape from four extents in storage order, `[ry, rx, qy, qx]` (NumPy's `data.shape`).
    public init(_ extents: [Int]) throws {
        guard extents.count == 4 else { throw CuboError.invalidShape(extents) }
        try self.init(scanRows: extents[0], scanColumns: extents[1], detectorRows: extents[2],
                      detectorColumns: extents[3])
    }

    /// `[ry, rx, qy, qx]`.
    public var extents: [Int] { [scanRows, scanColumns, detectorRows, detectorColumns] }
    /// Scan positions, `ry · rx`.
    public var scanCount: Int { scanRows * scanColumns }
    /// Detector pixels per pattern, `qy · qx`.
    public var detectorCount: Int { detectorRows * detectorColumns }
    /// Elements in the cube.
    public var elementCount: Int { scanCount * detectorCount }
    /// Elements in one scan row, `rx · qy · qx`.
    public var elementsPerScanRow: Int { scanColumns * detectorCount }

    /// The flat index of a scan position, row-major.
    public func scanIndex(row: Int, column: Int) -> Int { row * scanColumns + column }

    public var description: String { extents.map(String.init).joined(separator: " × ") }

    // Decoded through the validating initializer, so a decoded shape is as valid as a constructed one.
    private enum CodingKeys: String, CodingKey { case scanRows, scanColumns, detectorRows, detectorColumns }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(scanRows: try c.decode(Int.self, forKey: .scanRows),
                      scanColumns: try c.decode(Int.self, forKey: .scanColumns),
                      detectorRows: try c.decode(Int.self, forKey: .detectorRows),
                      detectorColumns: try c.decode(Int.self, forKey: .detectorColumns))
    }
}
