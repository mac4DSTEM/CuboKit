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
    /// compensation; the maximum in Float32 (exact), seeded at −∞.
    static func meanAndMax(of source: some CubeSource, tiling: ScanTiling,
                           progress: (@Sendable (Double) -> Void)?) async throws -> (DetectorImage, DetectorImage) {
        let shape = source.shape
        let n = shape.detectorCount
        var sum = [Double](repeating: 0, count: n)
        var correction = [Double](repeating: 0, count: n)
        var maximum = [Float](repeating: -.infinity, count: n)
        for range in try tiling.ranges(for: shape) {
            try Task.checkCancellation()
            let tile = try await source.readScanTile(rows: range)
            tile.values.withUnsafeBufferPointer { values in
                sum.withUnsafeMutableBufferPointer { s in
                    correction.withUnsafeMutableBufferPointer { c in
                        maximum.withUnsafeMutableBufferPointer { mx in
                            for k in 0..<tile.positionCount {
                                let p = values.baseAddress! + k * n
                                for i in 0..<n {
                                    let x = Double(p[i])
                                    let a = s[i]
                                    let t = a + x
                                    c[i] += abs(a) >= abs(x) ? (a - t) + x : (x - t) + a
                                    s[i] = t
                                }
                                vDSP_vmax(mx.baseAddress!, 1, p, 1, mx.baseAddress!, 1, vDSP_Length(n))
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
