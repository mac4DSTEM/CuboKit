//
//  CubeEditingParallel.swift
//  The multi-core kernels of ``CubeEditing/binScan(of:by:calibration:)`` and
//  ``CubeEditing/medianFilterMaskedPixels(of:mask:kernelWidth:calibration:)``: results bit for bit those of the scalar loops
//  they replaced (the same additions in the same order; the same comparisons), only vectorised and split across cores.
//

import Dispatch
import Foundation

/// A pointer handed to concurrent workers that each touch only their own elements.
struct DisjointBuffer<T>: @unchecked Sendable {
    let base: UnsafeMutablePointer<T>
}

extension CubeEditing {
    /// The values of scan rows `rows` of `source` as a pointer valid inside `body`: read in place for a ``MemoryCube`` (no tile
    /// copy), else from `readScanTile`.
    static func withScanRows<R>(of source: some CubeSource, rows: Range<Int>,
                                _ body: (UnsafePointer<Float>) -> R) async throws -> R {
        if let memory = source as? MemoryCube {
            try source.shape.requireRows(rows)
            let offset = rows.lowerBound * source.shape.elementsPerScanRow
            return memory.values.withUnsafeBufferPointer { body($0.baseAddress! + offset) }
        }
        let tile = try await source.readScanTile(rows: rows)
        return tile.values.withUnsafeBufferPointer { body($0.baseAddress!) }
    }

    /// `bin_R` for `outRowCount` output rows starting at scan row `firstRow * factor` of `rows`: each output pixel is the sum of
    /// its `factor x factor` block in Double, in the order rows then columns, rounded once to Float32, four detector pixels per
    /// step; the output positions run on every core.
    static func binScanRows(_ input: UnsafePointer<Float>, scanColumns: Int, detectorCount dc: Int, factor: Int,
                            outRowCount: Int, outColumns: Int, into output: UnsafeMutablePointer<Float>) {
        let out = DisjointBuffer(base: output)
        let source = DisjointBuffer(base: UnsafeMutablePointer(mutating: input))
        let items = outRowCount * outColumns
        DispatchQueue.concurrentPerform(iterations: items) { item in
            let ro = item / outColumns, oc = item % outColumns
            let target = out.base + item * dc
            var q = 0
            while q + 4 <= dc {
                var acc = SIMD4<Double>(repeating: 0)
                for dr in 0..<factor { for dcol in 0..<factor {
                    let p = UnsafeRawPointer(source.base + (((ro * factor + dr) * scanColumns + oc * factor + dcol) * dc + q))
                    acc += SIMD4<Double>(p.loadUnaligned(as: SIMD4<Float>.self))
                } }
                UnsafeMutableRawPointer(target + q).storeBytes(of: SIMD4<Float>(acc), as: SIMD4<Float>.self)
                q += 4
            }
            while q < dc {
                var acc = 0.0
                for dr in 0..<factor { for dcol in 0..<factor {
                    acc += Double(source.base[((ro * factor + dr) * scanColumns + oc * factor + dcol) * dc + q])
                } }
                target[q] = Float(acc)
                q += 1
            }
        }
    }

    /// One masked-pixel window of `median_filter_masked_pixels`: pixel row, column, then the clipped window rows and columns.
    struct MedianWindow: Sendable {
        let row: Int, column: Int, rowStart: Int, rowEnd: Int, columnStart: Int, columnEnd: Int
    }

    /// Applies the median filter to `positionCount` patterns from `input` into `output` (same layout), patterns on every core.
    /// Per pattern: copy, then each window in order (earlier replacements count), its values gathered into a fixed buffer and
    /// ordered by insertion (a window is small); NaN anywhere in the window gives NaN, an even count the Float32 mean of the
    /// two middle values.
    static func medianFilterPatterns(_ input: UnsafePointer<Float>, positionCount: Int, detectorRows: Int, detectorColumns: Int,
                                     windows: [MedianWindow], into output: UnsafeMutablePointer<Float>) {
        let dc = detectorRows * detectorColumns
        let out = DisjointBuffer(base: output)
        let source = DisjointBuffer(base: UnsafeMutablePointer(mutating: input))
        let maxWindow = windows.map { ($0.rowEnd - $0.rowStart) * ($0.columnEnd - $0.columnStart) }.max() ?? 0
        let chunk = Swift.max(1, positionCount / (4 * ProcessInfo.processInfo.activeProcessorCount))
        let chunks = (positionCount + chunk - 1) / chunk
        DispatchQueue.concurrentPerform(iterations: chunks) { c in
            withUnsafeTemporaryAllocation(of: Float.self, capacity: Swift.max(1, maxWindow)) { scratch in
                for k in (c * chunk)..<Swift.min(positionCount, (c + 1) * chunk) {
                    let src = source.base + k * dc, p = out.base + k * dc
                    p.update(from: src, count: dc)
                    for w in windows {
                        var n = 0, hasNaN = false
                        for rr in w.rowStart..<w.rowEnd { for cc in w.columnStart..<w.columnEnd {
                            let v = p[rr * detectorColumns + cc]
                            if v.isNaN { hasNaN = true }
                            scratch[n] = v; n += 1
                        } }
                        let target = w.row * detectorColumns + w.column
                        if hasNaN { p[target] = .nan; continue }   // np.median
                        if n <= 32 {
                            var i = 1
                            while i < n {
                                let v = scratch[i]
                                var j = i - 1
                                while j >= 0 && scratch[j] > v { scratch[j + 1] = scratch[j]; j -= 1 }
                                scratch[j + 1] = v
                                i += 1
                            }
                        } else {
                            var sorted = UnsafeMutableBufferPointer(rebasing: scratch[0..<n])
                            sorted.sort()
                        }
                        p[target] = n % 2 == 1 ? scratch[n / 2] : (scratch[n / 2 - 1] + scratch[n / 2]) / 2
                    }
                }
            }
        }
    }
}
