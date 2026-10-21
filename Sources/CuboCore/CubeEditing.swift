//
//  CubeEditing.swift
//  Eager edits of a cube that py4DSTEM's `DataCube` methods do: set_scan_shape, swap_RQ, swap_Rxy, swap_Qxy, bin_R, thin_R,
//  pad_Q, resample_Q (bilinear) and median_filter_masked_pixels. `crop_*` and `bin_Q` are ``ViewedCube``.
//
//  Reference: py4DSTEM `preprocess/preprocess.py` at f050d207. Each function reads its source tile by tile (set_scan_shape and
//  the two scan swaps load the whole cube and need twice its size in memory) and returns a
//  ``MemoryCube`` with the ``Calibration`` the py4DSTEM function leaves behind (pixel sizes updated; see ``Calibration``).
//  Where py4DSTEM prints a message and returns the cube unchanged (an impossible `set_scan_shape`), the kit throws.
//  Float32 in and out; sums run in Double and are rounded once (NumPy accumulates in Float32: relative 1e-6 of the largest
//  value, the tolerance of ``ViewedCube``'s bin).
//

import Foundation

/// A cube made by an edit, and the calibration that goes with it.
public struct EditedCube: Sendable {
    public let cube: MemoryCube
    public let calibration: Calibration
}

/// py4DSTEM's reshaping and pre-processing of a `DataCube`, on any ``CubeSource``.
public enum CubeEditing {

    // MARK: shape

    /// `set_scan_shape(rows, columns)`: the same patterns in the same order under a new scan shape. Throws
    /// `CuboError.invalidShape` unless `rows * columns` equals the scan count.
    public static func setScanShape(of source: some CubeSource, rows: Int, columns: Int,
                                    calibration: Calibration = Calibration()) async throws -> EditedCube {
        let s = source.shape
        guard rows > 0, columns > 0, rows * columns == s.scanCount else { throw CuboError.invalidShape([rows, columns]) }
        let all = try await MemoryCube(loading: source).values
        let shape = try CubeShape(scanRows: rows, scanColumns: columns, detectorRows: s.detectorRows, detectorColumns: s.detectorColumns)
        return EditedCube(cube: try MemoryCube(shape: shape, values: all), calibration: calibration)
    }

    /// `swap_RQ`: the first two axes and the last two trade places, so a `(R, R, Q, Q)` cube becomes `(Q, Q, R, R)`.
    public static func swapScanAndDetector(of source: some CubeSource,
                                           calibration: Calibration = Calibration()) async throws -> EditedCube {
        let s = source.shape
        let v = try await MemoryCube(loading: source).values
        let out = try CubeShape(scanRows: s.detectorRows, scanColumns: s.detectorColumns, detectorRows: s.scanRows, detectorColumns: s.scanColumns)
        var o = [Float](repeating: 0, count: v.count)
        let dc = s.detectorCount
        for r in 0..<s.scanRows { for c in 0..<s.scanColumns { for q in 0..<dc {
            o[q * s.scanCount + (r * s.scanColumns + c)] = v[(r * s.scanColumns + c) * dc + q]
        } } }
        return EditedCube(cube: try MemoryCube(shape: out, values: o), calibration: calibration)
    }

    /// `swap_Rxy`: scan rows and scan columns trade places.
    public static func swapScanAxes(of source: some CubeSource,
                                    calibration: Calibration = Calibration()) async throws -> EditedCube {
        let s = source.shape
        let v = try await MemoryCube(loading: source).values
        let out = try CubeShape(scanRows: s.scanColumns, scanColumns: s.scanRows, detectorRows: s.detectorRows, detectorColumns: s.detectorColumns)
        var o = [Float](repeating: 0, count: v.count)
        let dc = s.detectorCount
        for r in 0..<s.scanRows { for c in 0..<s.scanColumns {
            let from = (r * s.scanColumns + c) * dc, to = (c * s.scanRows + r) * dc
            o.replaceSubrange(to..<(to + dc), with: v[from..<(from + dc)])
        } }
        return EditedCube(cube: try MemoryCube(shape: out, values: o), calibration: calibration)
    }

    /// `swap_Qxy`: detector rows and detector columns trade places in every pattern.
    public static func swapDetectorAxes(of source: some CubeSource,
                                        calibration: Calibration = Calibration()) async throws -> EditedCube {
        let s = source.shape
        let out = try CubeShape(scanRows: s.scanRows, scanColumns: s.scanColumns, detectorRows: s.detectorColumns, detectorColumns: s.detectorRows)
        return try await perPattern(source, outShape: out, calibration: calibration) { p in
            var o = [Float](repeating: 0, count: p.count)
            for r in 0..<s.detectorRows { for c in 0..<s.detectorColumns { o[c * s.detectorRows + r] = p[r * s.detectorColumns + c] } }
            return o
        }
    }

    // MARK: scan

    /// `bin_R(N)`: sums of `N x N` blocks of scan positions after trimming the scan to a multiple of `N` off the END of each
    /// axis; the R pixel size multiplies by `N`. `N = 1` returns the cube. Throws when `N` leaves no scan position.
    public static func binScan(of source: some CubeSource, by factor: Int,
                               calibration: Calibration = Calibration()) async throws -> EditedCube {
        let s = source.shape
        guard factor >= 1 else { throw CuboError.invalidView("the bin factor \(factor) is below 1") }
        let outRows = s.scanRows / factor, outColumns = s.scanColumns / factor
        guard outRows > 0, outColumns > 0 else { throw CuboError.invalidView("binning a \(s.scanRows) x \(s.scanColumns) scan by \(factor) leaves nothing") }
        let out = try CubeShape(scanRows: outRows, scanColumns: outColumns, detectorRows: s.detectorRows, detectorColumns: s.detectorColumns)
        let dc = s.detectorCount
        var o = [Float](); o.reserveCapacity(out.elementCount)
        for r in 0..<outRows {
            try Task.checkCancellation()
            let tile = try await source.readScanTile(rows: (r * factor)..<(r * factor + factor)).values
            var sums = [Double](repeating: 0, count: outColumns * dc)
            for dr in 0..<factor { for c in 0..<(outColumns * factor) {
                let base = (dr * s.scanColumns + c) * dc, target = (c / factor) * dc
                for q in 0..<dc { sums[target + q] += Double(tile[base + q]) }
            } }
            o.append(contentsOf: sums.map { Float($0) })
        }
        return EditedCube(cube: try MemoryCube(shape: out, values: o), calibration: try calibration.afterBinningScan(by: factor))
    }

    /// `thin_R(N)`: every `N`-th scan position in both axes, `floor(R / N)` of them; the R pixel size multiplies by `N`.
    public static func thinScan(of source: some CubeSource, by factor: Int,
                                calibration: Calibration = Calibration()) async throws -> EditedCube {
        let s = source.shape
        guard factor >= 1 else { throw CuboError.invalidView("the thinning factor \(factor) is below 1") }
        let outRows = s.scanRows / factor, outColumns = s.scanColumns / factor
        guard outRows > 0, outColumns > 0 else { throw CuboError.invalidView("thinning a \(s.scanRows) x \(s.scanColumns) scan by \(factor) leaves nothing") }
        let out = try CubeShape(scanRows: outRows, scanColumns: outColumns, detectorRows: s.detectorRows, detectorColumns: s.detectorColumns)
        let dc = s.detectorCount
        var o = [Float](); o.reserveCapacity(out.elementCount)
        for r in 0..<outRows {
            try Task.checkCancellation()
            let tile = try await source.readScanTile(rows: (r * factor)..<(r * factor + 1)).values
            for c in 0..<outColumns { o.append(contentsOf: tile[(c * factor * dc)..<((c * factor + 1) * dc)]) }
        }
        return EditedCube(cube: try MemoryCube(shape: out, values: o), calibration: try calibration.afterThinningScan(by: factor))
    }

    // MARK: detector

    /// How ``padDetector(of:_:calibration:)`` is told its size.
    public enum PadSpecification: Sendable, Equatable {
        /// py4DSTEM `pad_factor`: each side gets `round(extent * (factor - 1) / 2)` pixels (NumPy rounds half to even). A factor
        /// below 1 throws.
        case factor(rows: Double, columns: Double)
        /// py4DSTEM `output_size`: the extra pixels split with the odd pixel at the END. Smaller than the detector throws.
        case outputSize(rows: Int, columns: Int)
    }

    /// `pad_Q`: zero padding of the detector; the pixel sizes are unchanged (and so is the origin, in pixels, which now sits
    /// further from the first pixel: the caller that stores the origin shifts it by the leading pad, see
    /// ``padLeading(rows:columns:specification:)``).
    public static func padDetector(of source: some CubeSource, _ spec: PadSpecification,
                                   calibration: Calibration = Calibration()) async throws -> EditedCube {
        let s = source.shape
        let (lr, tr, lc, tc) = try padWidths(rows: s.detectorRows, columns: s.detectorColumns, spec)
        let outRows = s.detectorRows + lr + tr, outColumns = s.detectorColumns + lc + tc
        let out = try CubeShape(scanRows: s.scanRows, scanColumns: s.scanColumns, detectorRows: outRows, detectorColumns: outColumns)
        return try await perPattern(source, outShape: out, calibration: calibration) { p in
            var o = [Float](repeating: 0, count: outRows * outColumns)
            for r in 0..<s.detectorRows {
                let from = r * s.detectorColumns, to = (r + lr) * outColumns + lc
                o.replaceSubrange(to..<(to + s.detectorColumns), with: p[from..<(from + s.detectorColumns)])
            }
            return o
        }
    }

    /// The pixels `pad_Q` adds before the first row and the first column.
    public static func padLeading(rows: Int, columns: Int, specification: PadSpecification) throws -> (rows: Int, columns: Int) {
        let w = try padWidths(rows: rows, columns: columns, specification)
        return (w.0, w.2)
    }

    private static func padWidths(rows: Int, columns: Int, _ spec: PadSpecification) throws -> (Int, Int, Int, Int) {
        switch spec {
        case .factor(let fr, let fc):
            guard fr.isFinite, fc.isFinite, fr >= 1, fc >= 1 else { throw CuboError.invalidView("the pad factor must be at least 1") }
            let pr = Int((Double(rows) * (fr - 1) / 2).rounded(.toNearestOrEven))
            let pc = Int((Double(columns) * (fc - 1) / 2).rounded(.toNearestOrEven))
            return (pr, pr, pc, pc)
        case .outputSize(let r, let c):
            guard r >= rows, c >= columns else { throw CuboError.invalidView("the padded size \(r) x \(c) is smaller than the \(rows) x \(columns) detector") }
            let dr = r - rows, dc = c - columns
            return (dr / 2, dr / 2 + dr % 2, dc / 2, dc / 2 + dc % 2)
        }
    }

    /// How ``resampleDetector(of:_:conserveSums:calibration:)`` is told its size.
    public enum ResampleSpecification: Sendable, Equatable {
        /// py4DSTEM `resampling_factor`; the output is `round(extent * factor)` (half to even).
        case factor(rows: Double, columns: Double)
        case outputSize(rows: Int, columns: Int)
    }

    /// `resample_Q(method="bilinear")`: scipy's `ndimage.zoom(order=1, mode="nearest", grid_mode=True)` of every pattern, so
    /// output pixel `o` samples the input at `(o + 0.5) * in / out - 0.5`, clamped to the detector, linearly. With
    /// `conserveSums` the result is divided by the product of the two factors as py4DSTEM does. The Q pixel size divides by the
    /// row factor.
    public static func resampleDetector(of source: some CubeSource, _ spec: ResampleSpecification, conserveSums: Bool = false,
                                        calibration: Calibration = Calibration()) async throws -> EditedCube {
        let s = source.shape
        let outRows: Int, outColumns: Int, fr: Double, fc: Double
        switch spec {
        case .factor(let a, let b):
            guard a.isFinite, b.isFinite, a > 0, b > 0 else { throw CuboError.invalidView("the resampling factor must be positive") }
            outRows = Int((Double(s.detectorRows) * a).rounded(.toNearestOrEven)); outColumns = Int((Double(s.detectorColumns) * b).rounded(.toNearestOrEven))
            fr = a; fc = b
        case .outputSize(let r, let c):
            outRows = r; outColumns = c
            fr = Double(r) / Double(s.detectorRows); fc = Double(c) / Double(s.detectorColumns)
        }
        guard outRows > 0, outColumns > 0 else { throw CuboError.invalidView("resampling leaves no detector pixel") }
        let out = try CubeShape(scanRows: s.scanRows, scanColumns: s.scanColumns, detectorRows: outRows, detectorColumns: outColumns)
        let rowPlan = interpolation(inCount: s.detectorRows, outCount: outRows)
        let columnPlan = interpolation(inCount: s.detectorColumns, outCount: outColumns)
        let divisor = conserveSums ? fr * fc : 1
        let edited = try await perPattern(source, outShape: out, calibration: try calibration.afterResamplingDetector(rowFactor: fr)) { p in
            var o = [Float](repeating: 0, count: outRows * outColumns)
            for r in 0..<outRows {
                let (r0, r1, wr) = rowPlan[r]
                for c in 0..<outColumns {
                    let (c0, c1, wc) = columnPlan[c]
                    let top = Double(p[r0 * s.detectorColumns + c0]) * (1 - wc) + Double(p[r0 * s.detectorColumns + c1]) * wc
                    let bottom = Double(p[r1 * s.detectorColumns + c0]) * (1 - wc) + Double(p[r1 * s.detectorColumns + c1]) * wc
                    // scipy stores the Float32 result; py4DSTEM then divides in Float64
                    o[r * outColumns + c] = Float(Float(top * (1 - wr) + bottom * wr).asDouble / divisor)
                }
            }
            return o
        }
        return edited
    }

    private static func interpolation(inCount: Int, outCount: Int) -> [(Int, Int, Double)] {
        let zoom = Double(inCount) / Double(outCount)
        return (0..<outCount).map { o in
            let x = min(max((Double(o) + 0.5) * zoom - 0.5, 0), Double(inCount - 1))
            let i0 = Int(x.rounded(.down)), i1 = min(i0 + 1, inCount - 1)
            return (i0, i1, x - Double(i0))
        }
    }

    /// `median_filter_masked_pixels(mask, kernel_width)`: every masked detector pixel (row-major order) is replaced, in every
    /// pattern, by the median of the window `[i - floor(k/2), i + ceil(k/2))` for odd `k` or `[i - k/2, i + k/2)` for even `k`,
    /// clipped to the detector and INCLUDING the pixel itself and the replacements made before it, as upstream. An even count
    /// takes the mean of the two middle values in Float32.
    public static func medianFilterMaskedPixels(of source: some CubeSource, mask: [Bool], kernelWidth: Int = 3,
                                                calibration: Calibration = Calibration()) async throws -> EditedCube {
        let s = source.shape
        guard mask.count == s.detectorCount else { throw CuboError.valueCountMismatch(expected: s.detectorCount, received: mask.count) }
        guard kernelWidth >= 1 else { throw CuboError.invalidView("the median kernel width \(kernelWidth) is below 1") }
        let widthMax = kernelWidth % 2 == 0 ? kernelWidth / 2 : (kernelWidth + 1) / 2
        let widthMin = kernelWidth / 2
        var windows = [(Int, Int, Int, Int, Int, Int)]()   // pixel row, column, then the clipped window
        for r in 0..<s.detectorRows { for c in 0..<s.detectorColumns where mask[r * s.detectorColumns + c] {
            windows.append((r, c, max(0, r - widthMin), min(s.detectorRows, r + widthMax), max(0, c - widthMin), min(s.detectorColumns, c + widthMax)))
        } }
        return try await perPattern(source, outShape: s, calibration: calibration) { pattern in
            var p = pattern
            var scratch = [Float]()
            for (r, c, r0, r1, c0, c1) in windows {
                scratch.removeAll(keepingCapacity: true)
                for rr in r0..<r1 { for cc in c0..<c1 { scratch.append(p[rr * s.detectorColumns + cc]) } }
                let n = scratch.count
                if scratch.contains(where: \.isNaN) { p[r * s.detectorColumns + c] = .nan; continue }   // np.median
                scratch.sort()
                p[r * s.detectorColumns + c] = n % 2 == 1 ? scratch[n / 2] : (scratch[n / 2 - 1] + scratch[n / 2]) / 2
            }
            return p
        }
    }

    // MARK: plumbing

    /// Applies `transform` to every pattern, tile by tile, into a cube of `outShape` (same scan).
    private static func perPattern(_ source: some CubeSource, outShape: CubeShape, calibration: Calibration,
                                   _ transform: ([Float]) -> [Float]) async throws -> EditedCube {
        let s = source.shape
        let dc = s.detectorCount
        var all = [Float](); all.reserveCapacity(outShape.elementCount)
        for range in try ScanTiling.automatic.ranges(for: s) {
            try Task.checkCancellation()
            let v = try await source.readScanTile(rows: range).values
            for p in 0..<(range.count * s.scanColumns) {
                all.append(contentsOf: transform(Array(v[(p * dc)..<((p + 1) * dc)])))
            }
        }
        return EditedCube(cube: try MemoryCube(shape: outShape, values: all), calibration: calibration)
    }
}

private extension Float {
    var asDouble: Double { Double(self) }
}
