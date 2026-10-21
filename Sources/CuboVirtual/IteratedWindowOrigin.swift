//
//  IteratedWindowOrigin.swift
//  The origin estimator of mac4DSTEM's calibration (named, so a consumer can ask for it): a coarse argmax over bin-summed
//  blocks, then up to a few centre-of-mass refinements of a window re-centred on its own estimate. Float32 throughout, one
//  algorithm in two implementations (a Swift reference here, a Metal kernel in MetalReducers.swift) that agree within
//  relative 1e-5. py4DSTEM's estimator (OriginMeasurement.swift) stays the default; this is a deliberate variant:
//  a block-sum argmax in place of the Gaussian-filtered one, a window never smaller than r + margin pixels, boundary pixels
//  included, negative intensities clamped at 0, and non-finite pixels read as the median of their finite 8 neighbours.
//

import CuboCore
import Foundation

/// Which algorithm `OriginMeasurement.measure(of:radius:radiusScale:estimator:path:tiling:progress:)` runs.
public enum OriginEstimator: Sendable, Hashable {
    /// py4DSTEM's `get_origin` as `OriginMeasurement.measure(of:radius:radiusScale:tiling:)` runs it: Double, the Gaussian
    /// argmax and one centre of mass. CPU only.
    case py4DSTEM
    /// The iterated-window estimator, in Float32. Coarse step: the brightest block of the pattern cut into `bin × bin`
    /// blocks, `bin = max(1, round(radius))` (`binnedCoarse: false` uses `bin = 1`, the brightest pixel), its centre
    /// `(first + last) / 2`. Refinement: up to `passes` centres of mass on the previous centre, over the pixels within
    /// `max(radius · radiusScale, radius + minimumMargin)` of it (boundary included), intensities clamped at 0; a pass with
    /// no mass keeps the last centre and stops, a pass that moves the centre by under 1e-4 pixel stops after taking it.
    /// Non-finite pixels read as the median of their finite 8 neighbours (0 when none).
    case iteratedWindow(passes: Int = 4, minimumMargin: Double = 1.5, binnedCoarse: Bool = true)
}

/// The Float32 arithmetic of the iterated-window estimator, shared by the CPU path and the host side of the Metal path.
enum IteratedWindow {
    /// `value` unless non-finite, then the median of the finite 8 neighbours read from the original pattern.
    @inline(__always)
    static func filled(_ pat: UnsafeBufferPointer<Float>, _ row: Int, _ column: Int, _ rows: Int, _ columns: Int) -> Float {
        let v = pat[row * columns + column]
        if v.isFinite { return v }
        var nb: [Float] = []
        nb.reserveCapacity(8)
        for dr in -1...1 { for dc in -1...1 {
            if dr == 0 && dc == 0 { continue }
            let rr = row + dr, cc = column + dc
            if rr < 0 || rr >= rows || cc < 0 || cc >= columns { continue }
            let w = pat[rr * columns + cc]
            if !w.isFinite { continue }
            var i = nb.count
            nb.append(w)
            while i > 0 && nb[i - 1] > w { nb[i] = nb[i - 1]; i -= 1 }
            nb[i] = w
        } }
        let k = nb.count
        if k == 0 { return 0 }
        return k % 2 == 1 ? nb[k / 2] : 0.5 * (nb[k / 2 - 1] + nb[k / 2])
    }

    /// One pattern: the centre (row, column) in detector pixels.
    static func origin(_ pat: UnsafeBufferPointer<Float>, rows qy: Int, columns qx: Int, radius r: Float, radiusScale: Float,
                       passes: Int, margin: Float, binned: Bool) -> (row: Float, column: Float) {
        let bin = binned ? max(1, Int(r.rounded())) : 1   // rounded(): half away from zero, as Metal's round
        var bestSum = -Float.greatestFiniteMagnitude
        var coarseR = 0.5 * Float(qy - 1), coarseC = 0.5 * Float(qx - 1)
        var by = 0
        while by < qy {
            let yEnd = min(by + bin, qy)
            var bx = 0
            while bx < qx {
                let xEnd = min(bx + bin, qx)
                var s: Float = 0
                for y in by..<yEnd { for x in bx..<xEnd { s += filled(pat, y, x, qy, qx) } }
                if s > bestSum {
                    bestSum = s
                    coarseC = 0.5 * Float(bx + xEnd - 1)
                    coarseR = 0.5 * Float(by + yEnd - 1)
                }
                bx += bin
            }
            by += bin
        }
        let win = max(r * radiusScale, r + margin)
        let win2 = win * win
        var cr = coarseR, cc = coarseC
        var pass = 0
        while pass < passes {
            pass += 1
            let y0 = max(0, Int((cr - win).rounded(.down)))
            let y1 = min(qy - 1, Int((cr + win).rounded(.up)))
            let x0 = max(0, Int((cc - win).rounded(.down)))
            let x1 = min(qx - 1, Int((cc + win).rounded(.up)))
            var sumI: Float = 0, sumIX: Float = 0, sumIY: Float = 0
            if y0 <= y1 && x0 <= x1 {
                for y in y0...y1 {
                    let dy = Float(y) - cr
                    let dy2 = dy * dy
                    if dy2 > win2 { continue }
                    for x in x0...x1 {
                        let dx = Float(x) - cc
                        if dy2.addingProduct(dx, dx) > win2 { continue }
                        let v = max(filled(pat, y, x, qy, qx), 0)
                        sumI += v
                        sumIX = sumIX.addingProduct(v, Float(x))
                        sumIY = sumIY.addingProduct(v, Float(y))
                    }
                }
            }
            if !(sumI > 0) { break }
            let nc = sumIX / sumI, nr = sumIY / sumI
            let settled = abs(nc - cc) < 1e-4 && abs(nr - cr) < 1e-4
            cc = nc; cr = nr
            if settled { break }
        }
        return (cr, cc)
    }
}

extension OriginMeasurement {
    /// The origin of every scan position by a named `estimator`, as an `OriginMap` (row and column in detector pixels, each
    /// map's `path` the one that computed it). `radius` is the probe radius in pixels (for example `ProbeSize`'s).
    ///
    /// `.py4DSTEM` runs the measurement of `measure(of:radius:radiusScale:tiling:)` (CPU only; `.metal` throws
    /// `unsupportedOnMetal`). `.iteratedWindow` runs the Float32 algorithm on the CPU or, with `path: .metal`, in the
    /// `cubo_measure_origin` kernel, one thread per position; the two agree within relative 1e-5 (their arithmetic is
    /// identical operation by operation; the Float32 results are widened to Double). The tiles are walked in scan order and
    /// cancellation is checked between them.
    public static func measure(of source: some CubeSource, radius: Double, radiusScale: Double = 1.2,
                               estimator: OriginEstimator, path: ComputePath = .cpu, tiling: ScanTiling = .automatic,
                               progress: (@Sendable (Double) -> Void)? = nil) async throws -> OriginMap {
        guard radius > 0, radius < 1e6 else { throw CuboVirtualError.invalidRadius(radius) }
        guard radiusScale > 0, radiusScale < 1e6, radius * radiusScale < 1e6 else { throw CuboVirtualError.invalidRadius(radiusScale) }
        switch estimator {
        case .py4DSTEM:
            guard path == .cpu else { throw CuboVirtualError.unsupportedOnMetal("The py4DSTEM origin estimator") }
            let (map, _) = try await measure(of: source, radius: radius, radiusScale: radiusScale, tiling: tiling)
            progress?(1)
            return map
        case .iteratedWindow(let passes, let minimumMargin, let binnedCoarse):
            guard passes >= 0, passes <= 1000 else { throw CuboVirtualError.invalidInput("the pass count must be from 0 to 1000 (got \(passes))") }
            guard minimumMargin >= 0, minimumMargin.isFinite else {
                throw CuboVirtualError.invalidInput("the minimum margin must be finite and not negative (got \(minimumMargin))")
            }
            let shape = source.shape
            let qy = shape.detectorRows, qx = shape.detectorColumns, n = shape.detectorCount
            let r = Float(radius), scale = Float(radiusScale), margin = Float(minimumMargin)
            var rows = [Double](repeating: 0, count: shape.scanCount), cols = rows
            let metal = path == .metal ? try MetalReducers.context() : nil
            for range in try tiling.ranges(for: shape) {
                try Task.checkCancellation()
                let tile = try await source.readScanTile(rows: range)
                let first = range.lowerBound * shape.scanColumns
                let found: [(Float, Float)]
                if let metal {
                    found = try await MetalReducers.iteratedWindowOrigins(
                        metal, tile: tile.values, positions: tile.positionCount, shape: shape, radius: r, radiusScale: scale,
                        passes: passes, margin: margin, binned: binnedCoarse)
                } else {
                    let values = tile.values
                    found = parallelMap(tile.positionCount) { k in
                        values.withUnsafeBufferPointer { all in
                            let o = IteratedWindow.origin(UnsafeBufferPointer(rebasing: all[k * n..<(k + 1) * n]), rows: qy,
                                                          columns: qx, radius: r, radiusScale: scale, passes: passes,
                                                          margin: margin, binned: binnedCoarse)
                            return (o.row, o.column)
                        }
                    }
                }
                for k in found.indices { rows[first + k] = Double(found[k].0); cols[first + k] = Double(found[k].1) }
                progress?(Double(range.upperBound) / Double(shape.scanRows))
            }
            let img = { (v: [Double]) in ScanImage(scanRows: shape.scanRows, scanColumns: shape.scanColumns, values: v, path: path) }
            return try OriginMap(row: img(rows), column: img(cols))
        }
    }

    /// The scan indices of a strided sample: every n-th position, n the smallest step that keeps the count at or under
    /// `limit` and, when `patternBytes` is positive, under `budgetBytes / patternBytes` (at least one position).
    static func sensitivitySampleIndices(total: Int, limit: Int, patternBytes: Int, budgetBytes: Int) -> [Int] {
        guard total > 0, limit > 0 else { return [] }
        let byBytes = patternBytes > 0 ? max(1, budgetBytes / patternBytes) : limit
        let bound = Swift.min(limit, byBytes)
        let step = (total + bound - 1) / bound
        return Array(stride(from: 0, to: total, by: step))
    }

    /// How much the iterated-window estimate depends on the window: the median, over a strided sample of at most
    /// `sampleLimit` scan positions (and at most `sampleBudgetBytes` of gathered Float32 patterns, at least one position),
    /// of the distance in detector pixels between the origin at `radiusScale = shippedScale` and at `wideScale`, both
    /// `.iteratedWindow` with its defaults. A quantity, not a verdict: a compact probe reads near 0, a probe with a ring
    /// outside its core moves with the window. Computed in Float32 as the estimator is (distance and median included);
    /// nil when `radius` or a scale is not positive and finite, or no distance is finite. `path` as in `measure`.
    public static func windowSensitivity(of source: some CubeSource, radius: Double, shippedScale: Double = 1.2,
                                         wideScale: Double = 2.5, sampleLimit: Int = 200,
                                         sampleBudgetBytes: Int = 256 << 20, path: ComputePath = .cpu) async throws -> Double? {
        let shape = source.shape
        let indices = sensitivitySampleIndices(total: shape.scanCount, limit: sampleLimit,
                                               patternBytes: shape.detectorCount * MemoryLayout<Float>.stride,
                                               budgetBytes: sampleBudgetBytes)
        guard !indices.isEmpty, radius.isFinite, radius > 0, shippedScale.isFinite, shippedScale > 0,
              wideScale.isFinite, wideScale > 0 else { return nil }
        var gathered = [Float]()
        gathered.reserveCapacity(indices.count * shape.detectorCount)
        for index in indices {
            try Task.checkCancellation()
            gathered.append(contentsOf: try await source.readPattern(row: index / shape.scanColumns,
                                                                     column: index % shape.scanColumns).values)
        }
        let sample = try MemoryCube(shape: try CubeShape([1, indices.count, shape.detectorRows, shape.detectorColumns]),
                                    values: gathered)
        func measured(_ scale: Double) async throws -> OriginMap {
            try await measure(of: sample, radius: radius, radiusScale: scale, estimator: .iteratedWindow(), path: path)
        }
        let shipped = try await measured(shippedScale), wide = try await measured(wideScale)
        var distances: [Float] = []
        for i in 0..<indices.count {
            let dr = Float(wide.row.values[i]) - Float(shipped.row.values[i])
            let dc = Float(wide.column.values[i]) - Float(shipped.column.values[i])
            let d = (dr * dr + dc * dc).squareRoot()
            if d.isFinite { distances.append(d) }
        }
        guard !distances.isEmpty else { return nil }
        distances.sort()
        let m = distances.count / 2
        return Double(distances.count.isMultiple(of: 2) ? (distances[m - 1] + distances[m]) / 2 : distances[m])
    }
}
