//
//  FrameTransform.swift
//  Re-referencing detector positions and per-position origin maps under a scan crop, a detector crop and a detector bin.
//
//  py4DSTEM 0.14.19 (f050d207) crops with plain slicing and bins by summing blocks and multiplying the Q pixel size by the
//  bin factor (preprocess.py:208); it moves no origin. The frame rules here are the mac4DSTEM app's
//  (CalibrationReReference.swift at c809679c), pinned to that source: a position moves with the crop, then maps
//  through the pixel-centre bin transform. Arithmetic is Float32 as in the app.
//

/// Detector-frame rules for crop and bin. "Column" is the detector's fast axis (the app's x), "row" the slow one (y).
public enum FrameTransform {
    /// Maps a coordinate from the cropped frame into the binned frame: `(v + 0.5) / bin - 0.5`, which sends the centre
    /// of a bin block to the centre of the binned pixel. Identity for `bin <= 1`.
    public static func binnedCoordinate(_ value: Float, bin: Int) -> Float {
        bin > 1 ? (value + 0.5) / Float(bin) - 0.5 : value
    }

    /// Exact inverse of ``binnedCoordinate(_:bin:)``: `(v + 0.5) * bin - 0.5`.
    public static func sourceCoordinate(_ value: Float, bin: Int) -> Float {
        bin > 1 ? (value + 0.5) * Float(bin) - 0.5 : value
    }

    /// True when a position lies inside the detector's physical extent `[-0.5, width-0.5) × [-0.5, height-0.5)`
    /// in pixel-centre coordinates. Non-finite positions are outside.
    public static func isInside(column: Float, row: Float, width: Int, height: Int) -> Bool {
        column.isFinite && row.isFinite
            && column >= -0.5 && column < Float(width) - 0.5
            && row >= -0.5 && row < Float(height) - 0.5
    }

    /// Moves a source-frame detector position into the loaded view: subtract the crop offset, then bin. Returns nil
    /// when the result is outside `width × height` (the loaded, binned detector).
    public static func movePosition(column: Float, row: Float, columnOffset: Int, rowOffset: Int, bin: Int,
                                    width: Int, height: Int) -> (column: Float, row: Float)? {
        let c = binnedCoordinate(column - Float(columnOffset), bin: bin)
        let r = binnedCoordinate(row - Float(rowOffset), bin: bin)
        return isInside(column: c, row: r, width: width, height: height) ? (c, r) : nil
    }

    /// Takes a scan sub-rectangle of a row-major map (`mapWidth` columns). Exact selection.
    /// Returns nil when the rectangle does not fit the map.
    public static func cropMap(_ values: [Float], mapWidth: Int, mapHeight: Int, columnOffset: Int, rowOffset: Int,
                               width: Int, height: Int) -> [Float]? {
        guard values.count == mapWidth * mapHeight, columnOffset >= 0, rowOffset >= 0, width > 0, height > 0,
              columnOffset + width <= mapWidth, rowOffset + height <= mapHeight else { return nil }
        var out = [Float](); out.reserveCapacity(width * height)
        for y in 0..<height {
            let start = (rowOffset + y) * mapWidth + columnOffset
            out.append(contentsOf: values[start..<(start + width)])
        }
        return out
    }

    /// Result of moving a per-position origin map into a detector view.
    public struct MovedOriginMap: Sendable, Equatable {
        /// The moved columns and rows, same length as the input.
        public var columns: [Float]
        public var rows: [Float]
        /// Index of the first position whose moved origin is outside the detector, nil when all are inside.
        public var firstOutside: Int?
    }

    /// Moves per-position origins (detector crop offset, then bin) and reports the first one outside the loaded
    /// `width × height` detector. The caller decides what an outside origin invalidates.
    public static func moveOriginMap(columns: [Float], rows: [Float], columnOffset: Int, rowOffset: Int, bin: Int,
                                     width: Int, height: Int) -> MovedOriginMap? {
        guard columns.count == rows.count else { return nil }
        let c = columns.map { binnedCoordinate($0 - Float(columnOffset), bin: bin) }
        let r = rows.map { binnedCoordinate($0 - Float(rowOffset), bin: bin) }
        var first: Int?
        for i in 0..<c.count where !isInside(column: c[i], row: r[i], width: width, height: height) { first = i; break }
        return MovedOriginMap(columns: c, rows: r, firstOutside: first)
    }

    /// Calibration scalars under detector binning: lengths in detector pixels divide by `bin`; the reciprocal pixel
    /// size multiplies by `bin` (the one rescaling py4DSTEM also performs).
    public static func binnedLength(_ pixels: Float, bin: Int) -> Float { bin > 1 ? pixels / Float(bin) : pixels }
    public static func binnedPixelSize(_ size: Double, bin: Int) -> Double { bin > 1 ? size * Double(bin) : size }
}
