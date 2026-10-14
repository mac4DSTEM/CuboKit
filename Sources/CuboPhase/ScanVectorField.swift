//
//  ScanVectorField.swift
//  A 2-vector per scan position (registration slice 2, section 1): py4DSTEM's (x, y) component pairs, which the kit names
//  by the detector axis they lie along: `row` is py4DSTEM's x (the first detector axis), `column` its y.
//

import CuboCore
import Foundation

public struct ScanVectorField: Sendable, Equatable {
    public let scanRows: Int
    public let scanColumns: Int
    /// The component along the detector rows (py4DSTEM x), row-major over the scan.
    public let row: [Double]
    /// The component along the detector columns (py4DSTEM y).
    public let column: [Double]

    public init(scanRows: Int, scanColumns: Int, row: [Double], column: [Double]) throws {
        guard scanRows > 0, scanColumns > 0 else { throw CuboPhaseError.invalidField("a field needs a positive shape") }
        guard row.count == scanRows * scanColumns, column.count == row.count else {
            throw CuboPhaseError.invalidField("a \(scanRows) × \(scanColumns) field needs \(scanRows * scanColumns) values per component")
        }
        self.scanRows = scanRows
        self.scanColumns = scanColumns
        self.row = row
        self.column = column
    }

    var count: Int { scanRows * scanColumns }
}

public enum CuboPhaseError: Error, Equatable, Sendable, LocalizedError {
    case invalidField(String)
    /// The plane fit needs at least three finite positions not on one line.
    case planeFitDegenerate
    /// The rotation's curl and divergence need an interior: at least 3 × 3 positions.
    case scanTooSmallForRotation(rows: Int, columns: Int)
    case invalidAngles
    /// DPC's error is normalized by the field's mean square; an all-zero field has none.
    case zeroField
    case invalidParameter(String)

    public var errorDescription: String? {
        switch self {
        case .invalidField(let why): "Invalid vector field: \(why)."
        case .planeFitDegenerate: "The plane fit needs at least three finite positions that are not on one line."
        case .scanTooSmallForRotation(let r, let c): "The rotation search needs at least 3 × 3 scan positions; got \(r) × \(c)."
        case .invalidAngles: "The rotation search needs at least one finite angle."
        case .zeroField: "The field is zero everywhere, so there is nothing to integrate."
        case .invalidParameter(let why): "Invalid parameter: \(why)."
        }
    }
}
