import Accelerate
import CEmbeddingLAPACK
import Foundation

/// The outcome of a principal component analysis. Matrices are row-major.
public struct PCAResult: Sendable, Equatable {
    /// Number of samples.
    public let rows: Int
    /// Number of features.
    public let columns: Int
    /// Components retained.
    public let componentCount: Int
    /// Per-feature mean, length `columns`.
    public let mean: [Double]
    /// `componentCount × columns`, one principal axis per row, descending by variance.
    public let components: [Double]
    /// Variance along each component. scikit-learn variant: divisor rows − 1. mac4DSTEM variant: the covariance
    /// eigenvalues with divisor rows.
    public let explainedVariance: [Double]
    /// Variance of each component over the total variance of all features (the covariance trace). Divisor-free.
    public let explainedVarianceRatio: [Double]
    /// scikit-learn's `singular_values_`, `sqrt(explainedVariance · (rows − 1))`. Empty for the mac4DSTEM variant.
    public let singularValues: [Double]
    /// `rows × componentCount`: the centred data projected on the components.
    public let scores: [Double]
}

/// Principal component analysis through the covariance eigendecomposition (LAPACK `dsyevd`).
public enum PCA {
    /// Which implementation's conventions to follow.
    public enum Variant: Sendable, Equatable {
        /// scikit-learn 1.4.2 `PCA(svd_solver="full")`: Double throughout, covariance divisor rows − 1, component
        /// signs fixed by its `svd_flip` (the largest-magnitude score of each component is made positive; the first
        /// index wins a tie), `components` in `1...min(rows, columns)`.
        case scikitLearn
        /// mac4DSTEM `DiffractionEmbedding.compute` (c809679c): data, mean, components and ratios rounded to Float32,
        /// covariance divisor rows, the sum of outer products accumulated by `dsyrk` in chunks of 1024, ratios
        /// clamped at zero, the raw LAPACK sign (no sign convention), the component count clamped silently to
        /// `1...min(columns, rows − 1)`, scores computed in Float32 with `vDSP_dotpr`.
        case diffractionEmbeddingApp
    }

    /// Fits a PCA to `rows × columns` row-major `data` and returns the top `components` axes.
    ///
    /// The covariance is formed explicitly, so accuracy of a component degrades with the ratio of the largest
    /// eigenvalue to its gap (relative error about 1e-16 · λ₁ / gap); that is not scikit-learn's SVD path.
    public static func fit(_ data: [Double], rows: Int, columns: Int, components: Int,
                           variant: Variant = .scikitLearn) throws -> PCAResult {
        guard rows > 0, columns > 0, data.count == rows * columns else {
            throw EmbeddingError.invalidShape("data.count \(data.count) is not rows \(rows) × columns \(columns)")
        }
        guard data.allSatisfy(\.isFinite) else { throw EmbeddingError.nonFinite }
        switch variant {
        case .scikitLearn: return try fitScikit(data, rows, columns, components)
        case .diffractionEmbeddingApp: return try fitApp(data, rows, columns, components)
        }
    }

    /// Projects `data` (`rows × columns`) on `result.components` after subtracting `result.mean` (Double arithmetic).
    public static func transform(_ data: [Double], rows: Int, result: PCAResult) throws -> [Double] {
        let d = result.columns, k = result.componentCount
        guard rows > 0, data.count == rows * d else {
            throw EmbeddingError.invalidShape("data.count \(data.count) is not rows \(rows) × columns \(d)")
        }
        var out = [Double](repeating: 0, count: rows * k)
        for i in 0..<rows {
            for c in 0..<k {
                var s = 0.0
                for j in 0..<d { s += (data[i * d + j] - result.mean[j]) * result.components[c * d + j] }
                out[i * k + c] = s
            }
        }
        return out
    }

    /// One Float32 vector's coordinates on the principal components: `out[c] = Σᵢ (vector[i] − mean[i]) · basis[c·dims + i]`.
    ///
    /// The mac4DSTEM projection: `vDSP_vsub` centres exactly, `vDSP_dotpr` accumulates in Float32 in vDSP's own order (not
    /// left to right), so it differs from the scalar sum by Float32 rounding, about `dims · 6e-8` relative to
    /// `Σ|centred · basis|`; pinned in `ProjectFloatTests`. `basis` is `components × dims` row-major, `centred` is `dims`
    /// scratch, `out` takes `components` values. The same call `PCA.fit(variant: .diffractionEmbeddingApp)` makes for its scores.
    public static func project(_ vector: UnsafePointer<Float>, mean: UnsafePointer<Float>, basis: UnsafePointer<Float>,
                               dims: Int, components: Int, centred: UnsafeMutablePointer<Float>,
                               into out: UnsafeMutablePointer<Float>) {
        vDSP_vsub(mean, 1, vector, 1, centred, 1, vDSP_Length(dims))
        for c in 0..<components {
            vDSP_dotpr(centred, 1, basis + c * dims, 1, out + c, vDSP_Length(dims))
        }
    }

    // MARK: scikit-learn

    private static func fitScikit(_ data: [Double], _ n: Int, _ d: Int, _ k: Int) throws -> PCAResult {
        guard n >= 2 else { throw EmbeddingError.invalidShape("at least two rows are needed") }
        guard k >= 1, k <= min(n, d) else {
            throw EmbeddingError.invalidCount("components \(k) must be in 1...\(min(n, d))")
        }
        var mean = [Double](repeating: 0, count: d)
        for i in 0..<n { for j in 0..<d { mean[j] += data[i * d + j] } }
        for j in 0..<d { mean[j] /= Double(n) }
        var centred = data
        for i in 0..<n { for j in 0..<d { centred[i * d + j] -= mean[j] } }
        var cov = [Double](repeating: 0, count: d * d)
        cov.withUnsafeMutableBufferPointer { c in
            centred.withUnsafeBufferPointer { x in
                cubo_embedding_syrk_upper(Int32(d), Int32(n), 1 / Double(n - 1), x.baseAddress!, 0, c.baseAddress!)
            }
        }
        mirrorUpper(&cov, d)
        let (values, vectors) = try topEigen(cov, d, d)
        let variances = values.map { max(0, $0) }
        let total = variances.reduce(0, +)
        var comps = Array(vectors[0..<(k * d)])
        var scores = [Double](repeating: 0, count: n * k)
        for c in 0..<k {
            for i in 0..<n {
                var s = 0.0
                for j in 0..<d { s += centred[i * d + j] * comps[c * d + j] }
                scores[i * k + c] = s
            }
            // scikit-learn's svd_flip, u-based: u = score / singular value, so the sign of the score decides.
            var best = 0
            for i in 1..<n where abs(scores[i * k + c]) > abs(scores[best * k + c]) { best = i }
            if scores[best * k + c] < 0 {
                for j in 0..<d { comps[c * d + j] = -comps[c * d + j] }
                for i in 0..<n { scores[i * k + c] = -scores[i * k + c] }
            }
        }
        let ev = Array(variances[0..<k])
        return PCAResult(rows: n, columns: d, componentCount: k, mean: mean, components: comps,
                         explainedVariance: ev,
                         explainedVarianceRatio: ev.map { total > 0 ? $0 / total : 0 },
                         singularValues: ev.map { ($0 * Double(n - 1)).squareRoot() }, scores: scores)
    }

    // MARK: mac4DSTEM diffraction embedding

    private static func fitApp(_ data: [Double], _ n: Int, _ d: Int, _ requested: Int) throws -> PCAResult {
        let k = max(1, min(requested, d, max(1, n - 1)))
        let x = data.map { Float($0) }
        guard x.allSatisfy(\.isFinite) else { throw EmbeddingError.nonFinite }
        var sumVec = [Double](repeating: 0, count: d)
        var sumOuter = [Double](repeating: 0, count: d * d)
        let chunk = 1024
        var batch = [Double](repeating: 0, count: min(chunk, n) * d)
        var start = 0
        while start < n {
            let m = min(chunk, n - start)
            for r in 0..<m {
                for j in 0..<d {
                    let v = Double(x[(start + r) * d + j])
                    batch[r * d + j] = v
                    sumVec[j] += v
                }
            }
            batch.withUnsafeBufferPointer { b in
                sumOuter.withUnsafeMutableBufferPointer { so in
                    cubo_embedding_syrk_upper(Int32(d), Int32(m), 1, b.baseAddress!, 1, so.baseAddress!)
                }
            }
            start += m
        }
        mirrorUpper(&sumOuter, d)
        let nd = Double(n)
        let mean = sumVec.map { $0 / nd }
        var cov = [Double](repeating: 0, count: d * d)
        for i in 0..<d { for j in 0..<d { cov[i * d + j] = sumOuter[i * d + j] / nd - mean[i] * mean[j] } }
        var total = 0.0
        for i in 0..<d { total += cov[i * d + i] }
        guard total.isFinite, cov.allSatisfy(\.isFinite) else { throw EmbeddingError.nonFinite }
        let (values, vectors) = try topEigen(cov, d, k)
        let basis = vectors.map { Float($0) }
        let ratio = (0..<k).map { Float(total > 0 ? max(0, values[$0] / total) : 0) }
        let meanF = mean.map { Float($0) }
        var coords = [Float](repeating: 0, count: n * k)
        var centred = [Float](repeating: 0, count: d)
        x.withUnsafeBufferPointer { xb in
            meanF.withUnsafeBufferPointer { mb in
                basis.withUnsafeBufferPointer { bb in
                    coords.withUnsafeMutableBufferPointer { cb in
                        for i in 0..<n {
                            project(xb.baseAddress! + i * d, mean: mb.baseAddress!, basis: bb.baseAddress!, dims: d,
                                    components: k, centred: &centred, into: cb.baseAddress! + i * k)
                        }
                    }
                }
            }
        }
        return PCAResult(rows: n, columns: d, componentCount: k, mean: meanF.map(Double.init),
                         components: basis.map(Double.init), explainedVariance: Array(values[0..<k]),
                         explainedVarianceRatio: ratio.map(Double.init), singularValues: [],
                         scores: coords.map(Double.init))
    }

    // MARK: shared

    static func mirrorUpper(_ m: inout [Double], _ d: Int) {
        for i in 1..<max(1, d) { for j in 0..<i { m[i * d + j] = m[j * d + i] } }
    }

    /// Top `k` eigenpairs of a symmetric row-major matrix, descending; vectors row-major `k × d`.
    static func topEigen(_ matrix: [Double], _ d: Int, _ k: Int) throws -> ([Double], [Double]) {
        var a = matrix
        var w = [Double](repeating: 0, count: d)
        let info = cubo_embedding_symmetric_eigen(Int32(d), &a, &w)
        guard info == 0 else { throw EmbeddingError.solverFailed(Int(info)) }
        var values = [Double](repeating: 0, count: k)
        var vectors = [Double](repeating: 0, count: k * d)
        for c in 0..<k {
            let col = d - 1 - c
            values[c] = w[col]
            for j in 0..<d { vectors[c * d + j] = a[col * d + j] }
        }
        return (values, vectors)
    }
}
