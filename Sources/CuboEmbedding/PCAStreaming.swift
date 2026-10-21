import CEmbeddingLAPACK
import Foundation

/// A streaming mean and covariance of `columns`-dimensional rows, for a PCA of data too large to hold at once.
///
/// Rows are added in blocks (a scan tile at a time). Each block is reduced to its own mean and centred scatter
/// (`dsyrk` on the centred block), and merged into the running state with the pairwise update of Chan, Golub and
/// LeVeque (the parallel form of Welford's algorithm), so no sum of squares of large numbers is ever cancelled. The
/// state is Double throughout. Two accumulators merge into one, so tiles can be reduced on separate tasks. The
/// result does not depend on the block sizes or the order, except for rounding (measured in `docs/embedding.md`).
public struct PCACovarianceAccumulator: Sendable, Equatable {
    /// Number of features per row.
    public let columns: Int
    /// Rows added so far.
    public private(set) var count: Int = 0
    /// Running per-feature mean, length `columns` (zeros while `count` is 0).
    public private(set) var mean: [Double]
    /// Centred scatter Σ (x − mean)(x − mean)ᵀ, `columns × columns` row-major; only the upper triangle is kept.
    var scatter: [Double]

    /// An empty accumulator for rows of `columns` features.
    public init(columns: Int) throws {
        guard columns > 0 else { throw EmbeddingError.invalidShape("columns must be positive") }
        self.columns = columns
        mean = [Double](repeating: 0, count: columns)
        scatter = [Double](repeating: 0, count: columns * columns)
    }

    /// Adds `rows` rows (`rows × columns`, row-major). An empty block is accepted and changes nothing.
    public mutating func add(_ block: [Double], rows: Int) throws {
        let d = columns
        guard rows >= 0, block.count == rows * d else {
            throw EmbeddingError.invalidShape("block.count \(block.count) is not rows \(rows) × columns \(d)")
        }
        guard block.allSatisfy(\.isFinite) else { throw EmbeddingError.nonFinite }
        if rows == 0 { return }
        // Block mean by compensated (Neumaier) summation, then the centred block's scatter.
        var sum = [Double](repeating: 0, count: d), comp = [Double](repeating: 0, count: d)
        for i in 0..<rows {
            for j in 0..<d {
                let x = block[i * d + j], t = sum[j] + x
                comp[j] += abs(sum[j]) >= abs(x) ? (sum[j] - t) + x : (x - t) + sum[j]
                sum[j] = t
            }
        }
        let m = Double(rows)
        let blockMean = (0..<d).map { (sum[$0] + comp[$0]) / m }
        var centred = block
        for i in 0..<rows { for j in 0..<d { centred[i * d + j] -= blockMean[j] } }
        var blockScatter = [Double](repeating: 0, count: d * d)
        blockScatter.withUnsafeMutableBufferPointer { c in
            centred.withUnsafeBufferPointer { x in
                cubo_embedding_syrk_upper(Int32(d), Int32(rows), 1, x.baseAddress!, 0, c.baseAddress!)
            }
        }
        combine(count: rows, mean: blockMean, scatter: blockScatter)
    }

    /// Adds `rows` Float32 rows (`rows × columns`, row-major), converted to Double in chunks of 1024 rows.
    public mutating func add(_ block: [Float], rows: Int) throws {
        let d = columns
        guard rows >= 0, block.count == rows * d else {
            throw EmbeddingError.invalidShape("block.count \(block.count) is not rows \(rows) × columns \(d)")
        }
        var start = 0
        while start < rows {
            let m = min(1024, rows - start)
            try add(block[(start * d)..<((start + m) * d)].map(Double.init), rows: m)
            start += m
        }
    }

    /// Merges another accumulator of the same `columns` into this one.
    public mutating func merge(_ other: PCACovarianceAccumulator) throws {
        guard other.columns == columns else {
            throw EmbeddingError.invalidShape("cannot merge \(other.columns) columns into \(columns)")
        }
        if other.count == 0 { return }
        combine(count: other.count, mean: other.mean, scatter: other.scatter)
    }

    /// The sample covariance (divisor `count − 1`, as scikit-learn), `columns × columns` row-major and symmetric.
    /// Needs at least two rows.
    public func covariance() throws -> [Double] {
        guard count >= 2 else { throw EmbeddingError.invalidShape("at least two rows are needed") }
        let d = columns, f = 1 / Double(count - 1)
        var cov = scatter.map { $0 * f }
        PCA.mirrorUpper(&cov, d)
        return cov
    }

    private mutating func combine(count nb: Int, mean meanB: [Double], scatter scatterB: [Double]) {
        let d = columns
        if count == 0 {
            count = nb; mean = meanB; scatter = scatterB
            return
        }
        let na = Double(count), nbD = Double(nb), n = na + nbD
        let w = na * nbD / n
        let delta = (0..<d).map { meanB[$0] - mean[$0] }
        for i in 0..<d {
            for j in i..<d { scatter[i * d + j] += scatterB[i * d + j] + w * delta[i] * delta[j] }
        }
        for j in 0..<d { mean[j] += delta[j] * nbD / n }
        count += nb
    }
}

extension PCA {
    /// Fits a PCA to a covariance that was accumulated elsewhere (scikit-learn 1.4.2 conventions, Double).
    ///
    /// - Parameters:
    ///   - covariance: the `d × d` sample covariance, divisor `count − 1`, row-major (the upper triangle is read);
    ///     ``PCACovarianceAccumulator/covariance()`` returns it.
    ///   - count: number of rows behind it, at least 2.
    ///   - mean: the per-feature mean, length `d`.
    ///   - components: in `1...min(count, d)`.
    ///
    /// Eigenvalues, ratios and singular values are those of ``fit(_:rows:columns:components:variant:)``, to rounding.
    /// The data are not available here, so two fields differ from the in-memory fit: `scores` is empty (project
    /// with ``transform(_:rows:result:)``), and the sign of a component is not scikit-learn's `svd_flip`, which needs
    /// the scores. Each axis is given the provisional sign that makes its largest-magnitude entry positive (the first
    /// index wins a tie). To get `svd_flip`'s sign, stream the data once more through ``PCASignResolver``.
    public static func fit(covariance: [Double], count: Int, mean: [Double], components: Int) throws -> PCAResult {
        let d = mean.count
        guard d > 0, covariance.count == d * d else {
            throw EmbeddingError.invalidShape("covariance.count \(covariance.count) is not mean.count \(d) squared")
        }
        guard count >= 2 else { throw EmbeddingError.invalidShape("at least two rows are needed") }
        guard components >= 1, components <= min(count, d) else {
            throw EmbeddingError.invalidCount("components \(components) must be in 1...\(min(count, d))")
        }
        guard covariance.allSatisfy(\.isFinite), mean.allSatisfy(\.isFinite) else { throw EmbeddingError.nonFinite }
        var cov = covariance
        mirrorUpper(&cov, d)
        let (values, vectors) = try topEigen(cov, d, d)
        let variances = values.map { max(0, $0) }
        let total = variances.reduce(0, +)
        let k = components
        var comps = Array(vectors[0..<(k * d)])
        for c in 0..<k {
            var best = 0
            for j in 1..<d where abs(comps[c * d + j]) > abs(comps[c * d + best]) { best = j }
            if comps[c * d + best] < 0 { for j in 0..<d { comps[c * d + j] = -comps[c * d + j] } }
        }
        let ev = Array(variances[0..<k])
        return PCAResult(rows: count, columns: d, componentCount: k, mean: mean, components: comps,
                         explainedVariance: ev,
                         explainedVarianceRatio: ev.map { total > 0 ? $0 / total : 0 },
                         singularValues: ev.map { ($0 * Double(count - 1)).squareRoot() }, scores: [])
    }

    /// Fits a PCA to the covariance and mean an accumulator holds. See ``fit(covariance:count:mean:components:)``.
    public static func fit(_ accumulator: PCACovarianceAccumulator, components: Int) throws -> PCAResult {
        try fit(covariance: try accumulator.covariance(), count: accumulator.count, mean: accumulator.mean,
                components: components)
    }
}

/// Gives a streamed PCA scikit-learn's component signs.
///
/// scikit-learn 1.4.2's `svd_flip` makes the largest-magnitude score of each component positive. A fit from a
/// covariance has no scores, so this pass finds that score by streaming the rows once more: add every block with the
/// global index of its first row, then call ``resolved()``. The first index wins an exact tie, whatever the order the
/// blocks arrive in or the accumulator they were added to, so blocks may be processed on separate tasks and merged.
/// Scores are computed as `(x − mean) · component` in Double, as the in-memory fit does.
public struct PCASignResolver: Sendable, Equatable {
    private let result: PCAResult
    private var bestMagnitude: [Double]
    private var bestIndex: [Int]
    private var bestScore: [Double]

    /// A resolver for the components of `result` (from a covariance fit).
    public init(result: PCAResult) {
        self.result = result
        let k = result.componentCount
        bestMagnitude = [Double](repeating: -1, count: k)
        bestIndex = [Int](repeating: Int.max, count: k)
        bestScore = [Double](repeating: 0, count: k)
    }

    /// Scores `rows` rows (`rows × columns`) whose first row has global index `firstRow`.
    public mutating func add(_ block: [Double], rows: Int, firstRow: Int) throws {
        let d = result.columns, k = result.componentCount
        guard rows >= 0, firstRow >= 0, block.count == rows * d else {
            throw EmbeddingError.invalidShape("block.count \(block.count) is not rows \(rows) × columns \(d)")
        }
        guard block.allSatisfy(\.isFinite) else { throw EmbeddingError.nonFinite }
        if rows == 0 { return }
        var centred = block
        for i in 0..<rows { for j in 0..<d { centred[i * d + j] -= result.mean[j] } }
        var scores = [Double](repeating: 0, count: rows * k)
        centred.withUnsafeBufferPointer { x in
            result.components.withUnsafeBufferPointer { w in
                scores.withUnsafeMutableBufferPointer { s in
                    cubo_embedding_gemm_nt(Int32(rows), Int32(k), Int32(d), x.baseAddress!, w.baseAddress!,
                                           s.baseAddress!)
                }
            }
        }
        for i in 0..<rows {
            for c in 0..<k {
                consider(c, magnitude: abs(scores[i * k + c]), score: scores[i * k + c], index: firstRow + i)
            }
        }
    }

    /// Float32 rows, converted to Double in chunks of 1024 rows.
    public mutating func add(_ block: [Float], rows: Int, firstRow: Int) throws {
        let d = result.columns
        guard rows >= 0, block.count == rows * d else {
            throw EmbeddingError.invalidShape("block.count \(block.count) is not rows \(rows) × columns \(d)")
        }
        var start = 0
        while start < rows {
            let m = min(1024, rows - start)
            try add(block[(start * d)..<((start + m) * d)].map(Double.init), rows: m, firstRow: firstRow + start)
            start += m
        }
    }

    /// Merges a resolver built from the same result over other blocks.
    public mutating func merge(_ other: PCASignResolver) throws {
        guard other.result == result else {
            throw EmbeddingError.invalidShape("the resolvers were built from different results")
        }
        for c in 0..<result.componentCount where other.bestIndex[c] != Int.max {
            consider(c, magnitude: other.bestMagnitude[c], score: other.bestScore[c], index: other.bestIndex[c])
        }
    }

    /// The result with each component flipped so that its largest-magnitude score is positive (`scores`, when the
    /// result carries any, are flipped with it). A component no row was added for keeps its sign.
    public func resolved() -> PCAResult {
        let d = result.columns, k = result.componentCount
        var comps = result.components
        var scores = result.scores
        let n = scores.isEmpty ? 0 : scores.count / k
        for c in 0..<k where bestScore[c] < 0 {
            for j in 0..<d { comps[c * d + j] = -comps[c * d + j] }
            for i in 0..<n { scores[i * k + c] = -scores[i * k + c] }
        }
        return PCAResult(rows: result.rows, columns: d, componentCount: k, mean: result.mean, components: comps,
                         explainedVariance: result.explainedVariance,
                         explainedVarianceRatio: result.explainedVarianceRatio,
                         singularValues: result.singularValues, scores: scores)
    }

    private mutating func consider(_ c: Int, magnitude: Double, score: Double, index: Int) {
        if magnitude > bestMagnitude[c] || (magnitude == bestMagnitude[c] && index < bestIndex[c]) {
            bestMagnitude[c] = magnitude; bestScore[c] = score; bestIndex[c] = index
        }
    }
}
