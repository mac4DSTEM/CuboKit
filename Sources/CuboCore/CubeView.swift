//
//  CubeView.swift
//  A crop and a diffraction bin applied at read time: a view of a cube, never a new dataset.
//
//  Reference: py4DSTEM `preprocess.crop_data_real`, `crop_data_diffraction` and `bin_data_diffraction`. Binning sums
//  `bin x bin` blocks (it does not average) after trimming the detector to a multiple of the factor off the END of each
//  axis. The crop is applied first, so the trimmed extent is that of the cropped detector. Row and column are py4DSTEM's
//  first and second axis of each pair.
//

import Foundation

/// A half-open rectangle in one plane, in source pixels.
public struct AxisCrop: Sendable, Equatable {
    /// First row and first column of the rectangle.
    public var rowOffset: Int, columnOffset: Int
    /// Extent in rows and columns.
    public var rows: Int, columns: Int
    /// Creates a crop; validated against a source by ``CubeViewGeometry``.
    public init(rowOffset: Int, columnOffset: Int, rows: Int, columns: Int) {
        self.rowOffset = rowOffset; self.columnOffset = columnOffset; self.rows = rows; self.columns = columns
    }
    /// The rows covered. The end saturates at `Int.max` instead of overflowing (the initialiser is public and takes any
    /// integers; such a crop does not fit any plane and `fits` and ``CubeViewGeometry`` refuse it).
    public var rowRange: Range<Int> { Self.range(from: rowOffset, count: rows) }
    /// The columns covered (saturating like ``rowRange``).
    public var columnRange: Range<Int> { Self.range(from: columnOffset, count: columns) }
    private static func range(from offset: Int, count: Int) -> Range<Int> {
        let (end, overflow) = offset.addingReportingOverflow(max(0, count))
        return offset..<(overflow ? Int.max : end)
    }
    /// Whether the rectangle is non-empty and lies inside a `rows x columns` plane.
    public func fits(rows planeRows: Int, columns planeColumns: Int) -> Bool {
        guard rowOffset >= 0, columnOffset >= 0, rows > 0, columns > 0 else { return false }
        let (lastRowStart, rowOverflow) = planeRows.subtractingReportingOverflow(rows)
        let (lastColumnStart, columnOverflow) = planeColumns.subtractingReportingOverflow(columns)
        return !rowOverflow && !columnOverflow && rowOffset <= lastRowStart && columnOffset <= lastColumnStart
    }

    private static func scaled(_ value: Int, by stride: Int) -> Int? {
        let (product, overflow) = value.multipliedReportingOverflow(by: stride)
        return overflow ? nil : product
    }

    /// The smallest whole-pixel crop enclosing a pointer rectangle, in pixels of a plane scaled by `strideRows` and
    /// `strideColumns` (a decimated overview), clamped to the plane. The rectangle's corners are floored; the far corner
    /// is included (`floor(max) + 1`); at least one pixel survives. `nil` when the rectangle is not finite, the plane is
    /// empty, a stride product overflows, or the crop is the whole plane (no crop).
    public static func enclosing(firstRow: Double, firstColumn: Double, lastRow: Double, lastColumn: Double,
                                 strideRows: Int = 1, strideColumns: Int = 1,
                                 planeRows: Int, planeColumns: Int) -> AxisCrop? {
        guard planeRows > 0, planeColumns > 0,
              [firstRow, firstColumn, lastRow, lastColumn].allSatisfy({ $0.isFinite && abs($0) < Double(Int32.max) })
        else { return nil }
        let sr = max(1, strideRows), sc = max(1, strideColumns)
        // the strides are caller integers: a product that does not fit an Int is refused (nil), never a trap
        guard let r0 = scaled(Int(min(firstRow, lastRow).rounded(.down)), by: sr),
              let c0 = scaled(Int(min(firstColumn, lastColumn).rounded(.down)), by: sc),
              let r1 = scaled(Int(max(firstRow, lastRow).rounded(.down)) + 1, by: sr),
              let c1 = scaled(Int(max(firstColumn, lastColumn).rounded(.down)) + 1, by: sc) else { return nil }
        let cr0 = min(max(0, r0), planeRows - 1), cc0 = min(max(0, c0), planeColumns - 1)
        let cr1 = min(max(cr0 + 1, r1), planeRows), cc1 = min(max(cc0 + 1, c1), planeColumns)
        let h = cr1 - cr0, w = cc1 - cc0
        guard h > 0, w > 0 else { return nil }
        if cr0 == 0, cc0 == 0, h == planeRows, w == planeColumns { return nil }
        return AxisCrop(rowOffset: cr0, columnOffset: cc0, rows: h, columns: w)
    }
}

/// Which part of a cube is read: a scan crop, a detector crop and a detector bin factor.
public struct CubeViewSpecification: Sendable, Equatable {
    /// Scan crop; `nil` is the whole scan.
    public var scanCrop: AxisCrop?
    /// Detector crop; `nil` is the whole detector.
    public var detectorCrop: AxisCrop?
    /// Detector bin factor; 1 is none. py4DSTEM accepts any positive integer.
    public var detectorBin: Int
    /// Creates a specification.
    public init(scanCrop: AxisCrop? = nil, detectorCrop: AxisCrop? = nil, detectorBin: Int = 1) {
        self.scanCrop = scanCrop; self.detectorCrop = detectorCrop; self.detectorBin = detectorBin
    }
    /// Reads the cube exactly as stored.
    public static let fullExtent = CubeViewSpecification()
    /// Whether this is ``fullExtent``.
    public var isFullExtent: Bool { scanCrop == nil && detectorCrop == nil && detectorBin == 1 }
}

/// A specification validated against a source shape, with the shape it produces.
public struct CubeViewGeometry: Sendable, Equatable {
    /// The source and the specification.
    public let source: CubeShape, specification: CubeViewSpecification
    /// The shape of the view: the scan crop, the detector crop trimmed to whole bins, divided by the bin factor.
    public let shape: CubeShape
    /// The detector rectangle read from each source pattern: the crop (or the whole detector) trimmed to a multiple of the bin factor
    /// off the end of each axis. `nil` reads the whole detector.
    public let readDetectorCrop: AxisCrop?
    /// Detector rows and columns dropped as the edge remainder.
    public let discardedDetectorRows: Int, discardedDetectorColumns: Int

    /// Validates `specification` against `source`; a crop that does not fit, a bin factor below 1 or a binned detector with no
    /// pixel left throws `CuboError.invalidView` (it is never clamped).
    public init(source: CubeShape, specification: CubeViewSpecification) throws {
        let bin = specification.detectorBin
        guard bin >= 1 else { throw CuboError.invalidView("the bin factor \(bin) is below 1") }
        if let c = specification.scanCrop, !c.fits(rows: source.scanRows, columns: source.scanColumns) {
            throw CuboError.invalidView("the scan crop \(c.rows) x \(c.columns) at (\(c.rowOffset), \(c.columnOffset)) does not fit the \(source.scanRows) x \(source.scanColumns) scan")
        }
        if let c = specification.detectorCrop, !c.fits(rows: source.detectorRows, columns: source.detectorColumns) {
            throw CuboError.invalidView("the detector crop \(c.rows) x \(c.columns) at (\(c.rowOffset), \(c.columnOffset)) does not fit the \(source.detectorRows) x \(source.detectorColumns) detector")
        }
        let wantedRows = specification.detectorCrop?.rows ?? source.detectorRows
        let wantedColumns = specification.detectorCrop?.columns ?? source.detectorColumns
        let dropRows = wantedRows % bin, dropColumns = wantedColumns % bin
        let readRows = wantedRows - dropRows, readColumns = wantedColumns - dropColumns
        guard readRows > 0, readColumns > 0 else {
            throw CuboError.invalidView("binning \(wantedRows) x \(wantedColumns) detector pixels by \(bin) leaves nothing")
        }
        self.source = source; self.specification = specification
        discardedDetectorRows = dropRows; discardedDetectorColumns = dropColumns
        readDetectorCrop = (bin == 1 && specification.detectorCrop == nil) ? nil
            : AxisCrop(rowOffset: specification.detectorCrop?.rowOffset ?? 0, columnOffset: specification.detectorCrop?.columnOffset ?? 0,
                       rows: readRows, columns: readColumns)
        shape = try CubeShape(scanRows: specification.scanCrop?.rows ?? source.scanRows,
                              scanColumns: specification.scanCrop?.columns ?? source.scanColumns,
                              detectorRows: readRows / bin, detectorColumns: readColumns / bin)
    }

    /// The view of one full source pattern (`source.detectorRows x source.detectorColumns`, row-major): the crop, then the
    /// bin sums, each block accumulated in Double and rounded once to Float32.
    public func viewOfPattern(_ pattern: [Float]) -> [Float] {
        precondition(pattern.count == source.detectorCount, "a source pattern has \(source.detectorCount) values")
        let bin = specification.detectorBin
        let crop = readDetectorCrop
        let rows = crop?.rows ?? source.detectorRows, columns = crop?.columns ?? source.detectorColumns
        let r0 = crop?.rowOffset ?? 0, c0 = crop?.columnOffset ?? 0
        if bin == 1 {
            if crop == nil { return pattern }
            var out = [Float](); out.reserveCapacity(rows * columns)
            for r in 0..<rows {
                let start = (r0 + r) * source.detectorColumns + c0
                out.append(contentsOf: pattern[start..<(start + columns)])
            }
            return out
        }
        let outRows = rows / bin, outColumns = columns / bin
        var out = [Float](repeating: 0, count: outRows * outColumns)
        for r in 0..<outRows { for c in 0..<outColumns {
            var sum = 0.0
            for dr in 0..<bin {
                let base = (r0 + r * bin + dr) * source.detectorColumns + c0 + c * bin
                for dc in 0..<bin { sum += Double(pattern[base + dc]) }
            }
            out[r * outColumns + c] = Float(sum)
        } }
        return out
    }
}

/// A view of another cube: scan crop, detector crop and bin applied as the patterns are read.
public actor ViewedCube<Base: CubeSource>: CubeSource {
    public nonisolated let shape: CubeShape
    /// The validated specification and the shape it makes.
    public nonisolated let geometry: CubeViewGeometry
    private let base: Base

    /// Applies `specification` to `base`; throws `CuboError.invalidView` when it cannot.
    public init(base: Base, specification: CubeViewSpecification) throws {
        let g = try CubeViewGeometry(source: base.shape, specification: specification)
        self.base = base; geometry = g; shape = g.shape
    }

    public func readPattern(row: Int, column: Int) async throws -> DiffractionPattern {
        try shape.requirePosition(row: row, column: column)
        let source = try await base.readPattern(row: row + (geometry.specification.scanCrop?.rowOffset ?? 0),
                                                column: column + (geometry.specification.scanCrop?.columnOffset ?? 0))
        return try DiffractionPattern(detectorRows: shape.detectorRows, detectorColumns: shape.detectorColumns,
                                      values: geometry.viewOfPattern(source.values))
    }

    public func readScanTile(rows: Range<Int>) async throws -> ScanTile {
        try shape.requireRows(rows)
        let crop = geometry.specification.scanCrop
        let rowShift = crop?.rowOffset ?? 0, columnShift = crop?.columnOffset ?? 0
        let tile = try await base.readScanTile(rows: (rows.lowerBound + rowShift)..<(rows.upperBound + rowShift))
        let sourceDetector = geometry.source.detectorCount
        var out = [Float](); out.reserveCapacity(rows.count * shape.elementsPerScanRow)
        for r in 0..<rows.count { for c in 0..<shape.scanColumns {
            let start = (r * geometry.source.scanColumns + c + columnShift) * sourceDetector
            out.append(contentsOf: geometry.viewOfPattern(Array(tile.values[start..<(start + sourceDetector)])))
        } }
        return try ScanTile(shape: shape, rows: rows, values: out)
    }
}
