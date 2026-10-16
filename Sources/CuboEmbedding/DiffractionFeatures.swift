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
    public static func binnedVector(pattern: [Float], rows qy: Int, columns qx: Int, binnedSize: Int) throws -> [Float] {
        guard qy > 0, qx > 0, binnedSize > 0, pattern.count == qy * qx else {
            throw EmbeddingError.invalidShape("pattern.count \(pattern.count) is not rows \(qy) × columns \(qx), or binnedSize is not positive")
        }
        func axis(_ size: Int) -> (bins: Int, box: Int, offset: Int) {
            let bins = min(binnedSize, size)
            let box = max(1, size / bins)
            return (bins, box, (size - box * bins) / 2)
        }
        let y = axis(qy), x = axis(qx)
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
