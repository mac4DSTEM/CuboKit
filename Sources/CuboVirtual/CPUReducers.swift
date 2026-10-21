//
//  CPUReducers.swift
//  The reference path (registration 3.2, 3.3): Accelerate, Double accumulation, the same arithmetic whatever the
//  tiling, so the eager and lazy cubes and every tiling give the same bits.
//

import Accelerate
import CuboCore

enum CPUReducers {
    /// Virtual image: per position, the pattern widened to Double (vDSP_vspdp) and its dot product with the Double
    /// mask (vDSP_dotprD). Tiles partition the scan axis, so no tile boundary touches a position's sum.
    static func virtualImage(of source: some CubeSource, mask: [Float], tiling: ScanTiling,
                             progress: (@Sendable (Double) -> Void)?) async throws -> ScanImage {
        let shape = source.shape
        let n = shape.detectorCount
        let maskD = mask.map(Double.init)
        var out = [Double](repeating: 0, count: shape.scanCount)
        var pattern = [Double](repeating: 0, count: n)
        for range in try tiling.ranges(for: shape) {
            try Task.checkCancellation()
            let tile = try await source.readScanTile(rows: range)
            let first = range.lowerBound * shape.scanColumns
            tile.values.withUnsafeBufferPointer { values in
                maskD.withUnsafeBufferPointer { m in
                    pattern.withUnsafeMutableBufferPointer { p in
                        for k in 0..<tile.positionCount {
                            vDSP_vspdp(values.baseAddress! + k * n, 1, p.baseAddress!, 1, vDSP_Length(n))
                            var sum = 0.0
                            vDSP_dotprD(p.baseAddress!, 1, m.baseAddress!, 1, &sum, vDSP_Length(n))
                            out[first + k] = sum
                        }
                    }
                }
            }
            progress?(Double(range.upperBound) / Double(shape.scanRows))
        }
        return ScanImage(scanRows: shape.scanRows, scanColumns: shape.scanColumns, values: out, path: .cpu)
    }

    /// Mean and maximum pattern: per position in scan order, each pixel widened to Double and added with Neumaier
    /// compensation; the maximum in Float32 (exact), seeded at −∞. The detector pixels are split into blocks that run on
    /// every core; each pixel is still added in scan order, so the result is bit for bit the single-thread one. `wantMean`
    /// and `wantMax` skip the statistic that is not asked for (its result is then empty values of the right length).
    static func meanAndMax(of source: some CubeSource, tiling: ScanTiling, wantMean: Bool = true, wantMax: Bool = true,
                           progress: (@Sendable (Double) -> Void)?) async throws -> (DetectorImage, DetectorImage) {
        let shape = source.shape
        let n = shape.detectorCount
        // −0 is the identity of IEEE addition (−0 + −0 = −0, −0 + x = x), so an all-(−0) column keeps its sign, as NumPy's.
        var sum = [Double](repeating: -0.0, count: n)
        var correction = [Double](repeating: -0.0, count: n)
        var maximum = [Float](repeating: -.infinity, count: n)
        let blockSize = Swift.max(2048, (n + 4 * ProcessInfo.processInfo.activeProcessorCount - 1) / (4 * ProcessInfo.processInfo.activeProcessorCount))
        let blocks = (n + blockSize - 1) / blockSize
        for range in try tiling.ranges(for: shape) {
            try Task.checkCancellation()
            let tile = try await source.readScanTile(rows: range)
            let positions = tile.positionCount
            tile.values.withUnsafeBufferPointer { values in
                sum.withUnsafeMutableBufferPointer { s in
                    correction.withUnsafeMutableBufferPointer { c in
                        maximum.withUnsafeMutableBufferPointer { mx in
                            let vp = SharedBuffer(base: UnsafeMutablePointer(mutating: values.baseAddress!))
                            let sp = SharedBuffer(base: s.baseAddress!), cp = SharedBuffer(base: c.baseAddress!)
                            let mp = SharedBuffer(base: mx.baseAddress!)
                            DispatchQueue.concurrentPerform(iterations: blocks) { b in
                                let lo = b * blockSize, hi = Swift.min(n, lo + blockSize)
                                for k in 0..<positions {
                                    let p = vp.base + k * n
                                    if wantMean {
                                        for i in lo..<hi {
                                            let x = Double(p[i])
                                            let a = sp.base[i]
                                            let t = a + x
                                            // Only a finite step carries a correction (∞ − ∞ is NaN), and only a non-zero one is
                                            // added, so a correction of +0 never turns a −0 column positive.
                                            if t.isFinite {
                                                let e = abs(a) >= abs(x) ? (a - t) + x : (x - t) + a
                                                if e != 0 { cp.base[i] += e }
                                            }
                                            sp.base[i] = t
                                        }
                                    }
                                    if wantMax {
                                        vDSP_vmax(mp.base + lo, 1, p + lo, 1, mp.base + lo, 1, vDSP_Length(hi - lo))
                                    }
                                }
                            }
                        }
                    }
                }
            }
            progress?(Double(range.upperBound) / Double(shape.scanRows))
        }
        let count = Double(shape.scanCount)
        let mean = (0..<n).map { (sum[$0] + correction[$0]) / count }
        return (DetectorImage(detectorRows: shape.detectorRows, detectorColumns: shape.detectorColumns,
                              values: mean, path: .cpu),
                DetectorImage(detectorRows: shape.detectorRows, detectorColumns: shape.detectorColumns,
                              values: maximum.map(Double.init), path: .cpu))
    }
}
