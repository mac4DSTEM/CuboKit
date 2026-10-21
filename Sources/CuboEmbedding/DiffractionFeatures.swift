import Accelerate
import Foundation

/// The per-pattern feature vector mac4DSTEM feeds to PCA. There is no scikit-learn or py4DSTEM counterpart; this is
/// pinned to the mac4DSTEM source only.
public enum DiffractionFeatures {
    /// log1p of the non-negative intensities, divided by the pattern maximum, then summed over `box × box`
    /// cells (`box = size / bins`, `bins = min(binnedSize, size)`, the used region centred with the remainder
    /// split by floor on each side). The result has `binnedSize²` entries, row-major; cells beyond `bins`
    /// (when a detector axis is smaller than `binnedSize`) stay zero. Float32 arithmetic; NaN is kept, negative
    /// values and −∞ count as zero.
    ///
    /// Pinned to mac4DSTEM's vDSP/`vvlog1pf` path for finite patterns; the scalar summation order here differs, so
    /// agreement is to Float32 rounding, not bit for bit.
    ///
    /// `implementation` chooses how it is computed; the default `.scalar` is the loop described above.
    /// `.accelerate` runs the same steps on vDSP (`vDSP_vthr`, `vvlog1pf`, `vDSP_maxv`, `vDSP_vsmul`, a `vDSP_sve`
    /// per cell row), mac4DSTEM's former fast path. A pattern with NaN or ±∞ is computed by the scalar loop on either
    /// setting, because `vDSP_vthr` would turn NaN into 0. The two agree to Float32 rounding (`vvlog1pf` against
    /// `log1pf`, the order of the cell sums; measured in `docs/embedding.md`), and exactly on a pattern of zeros.
    public static func binnedVector(pattern: [Float], rows qy: Int, columns qx: Int, binnedSize: Int,
                                    implementation: Implementation = .scalar) throws -> [Float] {
        guard qy > 0, qx > 0, binnedSize > 0, pattern.count == qy * qx else {
            throw EmbeddingError.invalidShape("pattern.count \(pattern.count) is not rows \(qy) × columns \(qx), or binnedSize is not positive")
        }
        return pattern.withUnsafeBufferPointer {
            binned($0, qy, qx, binnedSize, implementation)
        }
    }

    /// The same, for one pattern inside a larger buffer (`pattern` is the `rows × columns` slice, no copy needed).
    public static func binnedVector(pattern: UnsafeBufferPointer<Float>, rows qy: Int, columns qx: Int,
                                    binnedSize: Int, implementation: Implementation = .scalar) throws -> [Float] {
        guard qy > 0, qx > 0, binnedSize > 0, pattern.count == qy * qx else {
            throw EmbeddingError.invalidShape("pattern.count \(pattern.count) is not rows \(qy) × columns \(qx), or binnedSize is not positive")
        }
        return binned(pattern, qy, qx, binnedSize, implementation)
    }

    /// How ``binnedVector(pattern:rows:columns:binnedSize:implementation:)`` computes.
    public enum Implementation: Sendable, Equatable {
        /// The Float32 scalar loop (the kit's original behaviour).
        case scalar
        /// vDSP and `vvlog1pf`; see the method's discussion.
        case accelerate
    }

    private static func axis(_ size: Int, _ binnedSize: Int) -> (bins: Int, box: Int, offset: Int) {
        let bins = min(binnedSize, size)
        let box = max(1, size / bins)
        return (bins, box, (size - box * bins) / 2)
    }

    private static func binned(_ pattern: UnsafeBufferPointer<Float>, _ qy: Int, _ qx: Int, _ binnedSize: Int,
                               _ implementation: Implementation) -> [Float] {
        if implementation == .accelerate {
            var total: Float = 0
            vDSP_sve(pattern.baseAddress!, 1, &total, vDSP_Length(qy * qx))
            if total.isFinite { return binnedAccelerate(pattern, qy, qx, binnedSize) }
        }
        return binnedScalar(pattern, qy, qx, binnedSize)
    }

    private static func binnedAccelerate(_ pattern: UnsafeBufferPointer<Float>, _ qy: Int, _ qx: Int,
                                         _ binnedSize: Int) -> [Float] {
        let y = axis(qy, binnedSize), x = axis(qx, binnedSize)
        let count = qy * qx
        var logged = [Float](repeating: 0, count: count)
        var scaled = [Float](repeating: 0, count: count)
        var maxVal: Float = 0
        var zero: Float = 0
        vDSP_vthr(pattern.baseAddress!, 1, &zero, &scaled, 1, vDSP_Length(count))
        var n = Int32(count)
        vvlog1pf(&logged, scaled, &n)
        vDSP_maxv(logged, 1, &maxVal, vDSP_Length(count))
        if maxVal > 0 {
            var inv = 1 / maxVal
            vDSP_vsmul(logged, 1, &inv, &logged, 1, vDSP_Length(count))
        }
        var out = [Float](repeating: 0, count: binnedSize * binnedSize)
        logged.withUnsafeBufferPointer { lg in
            for by in 0..<y.bins {
                let y0 = y.offset + by * y.box
                for bx in 0..<x.bins {
                    let x0 = x.offset + bx * x.box
                    var sum: Float = 0
                    for dy in 0..<y.box {
                        var row: Float = 0
                        vDSP_sve(lg.baseAddress! + (y0 + dy) * qx + x0, 1, &row, vDSP_Length(x.box))
                        sum += row
                    }
                    out[by * binnedSize + bx] = sum
                }
            }
        }
        return out
    }

    private static func binnedScalar(_ pattern: UnsafeBufferPointer<Float>, _ qy: Int, _ qx: Int,
                                     _ binnedSize: Int) -> [Float] {
        let y = axis(qy, binnedSize), x = axis(qx, binnedSize)
        var scaled = [Float](repeating: 0, count: qy * qx)
        var maxVal: Float = 0
        for i in 0..<(qy * qx) {
            let v = log1p(max(pattern[i], 0))
            scaled[i] = v
            if v > maxVal { maxVal = v }
        }
        if maxVal > 0 {
            let inv = 1 / maxVal
            for i in 0..<scaled.count { scaled[i] *= inv }
        }
        var out = [Float](repeating: 0, count: binnedSize * binnedSize)
        for by in 0..<y.bins {
            for bx in 0..<x.bins {
                var sum: Float = 0
                for dy in 0..<y.box {
                    var row: Float = 0
                    for dx in 0..<x.box { row += scaled[(y.offset + by * y.box + dy) * qx + x.offset + bx * x.box + dx] }
                    sum += row
                }
                out[by * binnedSize + bx] = sum
            }
        }
        return out
    }
}
