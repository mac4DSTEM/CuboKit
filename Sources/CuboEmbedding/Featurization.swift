//
//  Featurization.swift
//  py4DSTEM's `Featurization` (process/classification/featurization.py) on a feature matrix with one row per scan position:
//  the preprocessing methods (scalers, mean/median/max of feature columns), `PCA`, the iterative `NMF` with the correlation merge
//  of `_nmf_single`, `GMM` (`_gmm_single`) and the class patterns of `get_class_DPs`. The feature matrix itself, for example
//  binned diffraction patterns (`DiffractionFeatures`), is the caller's; `from_braggvectors` (peaks binned on a grid) and the
//  spatial-separation and consensus steps are not ported. NMF, PCA and GMM follow scikit-learn 1.9.1, the version py4DSTEM
//  0.14.19's reference environment carries.
//

import Foundation

/// A dense row-major matrix of Doubles.
public struct FeatureMatrix: Sendable, Equatable {
    public let rows: Int
    public let columns: Int
    public let values: [Double]

    public init(rows: Int, columns: Int, values: [Double]) throws {
        guard rows >= 0, columns >= 0, values.count == rows * columns else {
            throw FeaturizationError.invalidInput("values.count \(values.count) is not rows \(rows) × columns \(columns)")
        }
        self.rows = rows
        self.columns = columns
        self.values = values
    }

    public subscript(row: Int, column: Int) -> Double { values[row * columns + column] }

    /// One column (a class image when the rows are scan positions).
    public func column(_ c: Int) -> [Double] { (0..<rows).map { values[$0 * columns + c] } }
}

/// The result of py4DSTEM's iterative NMF over several independent models.
public struct FeaturizationNMFResult: Sendable, Equatable {
    /// With `saveAllModels` one `positions × classes` weight matrix per model, in model order; otherwise only the model with the
    /// lowest mean reconstruction error.
    public let models: [FeatureMatrix]
}

/// How `Featurization.reduceFeatures` combines the chosen columns.
public enum FeatureReduction: Sendable, Equatable {
    case mean, median, maximum
}

/// A feature matrix over a scan (`scanRows × scanColumns` rows, one row per position, row-major scan order).
public struct Featurization: Sendable, Equatable {
    public let scanRows: Int
    public let scanColumns: Int
    public let name: String
    /// `scanRows·scanColumns` rows by `featureCount` columns.
    public private(set) var features: FeatureMatrix

    public var featureCount: Int { features.columns }

    /// `Featurization(features, R_Nx, R_Ny, name)` for a `positions × featureCount` matrix.
    public init(features: FeatureMatrix, scanRows: Int, scanColumns: Int, name: String = "") throws {
        guard scanRows > 0, scanColumns > 0, features.rows == scanRows * scanColumns else {
            throw FeaturizationError.invalidInput("features have \(features.rows) rows, the scan has \(scanRows) × \(scanColumns) positions")
        }
        self.features = features
        self.scanRows = scanRows
        self.scanColumns = scanColumns
        self.name = name
    }

    /// `concatenate_features`: the feature columns of several instances side by side; the scans must agree.
    public static func concatenate(_ parts: [Featurization], name: String = "") throws -> Featurization {
        guard let first = parts.first else { throw FeaturizationError.invalidInput("nothing to concatenate") }
        guard parts.allSatisfy({ $0.scanRows == first.scanRows && $0.scanColumns == first.scanColumns }) else {
            throw FeaturizationError.invalidInput("Can only concatenate Featurization instances with the same scan")
        }
        let n = first.features.rows
        let total = parts.reduce(0) { $0 + $1.featureCount }
        var values = [Double](repeating: 0, count: n * total)
        var offset = 0
        for part in parts {
            for i in 0..<n { for j in 0..<part.featureCount { values[i * total + offset + j] = part.features[i, j] } }
            offset += part.featureCount
        }
        return try Featurization(features: FeatureMatrix(rows: n, columns: total, values: values), scanRows: first.scanRows,
                                 scanColumns: first.scanColumns, name: name)
    }

    // MARK: preprocessing

    /// `mean_feature`, `median_feature` and `max_feature`: the chosen columns are removed and their row-wise reduction is
    /// appended as the last column.
    public func reduceFeatures(_ indices: [Int], with reduction: FeatureReduction) throws -> Featurization {
        guard !indices.isEmpty, indices.allSatisfy({ (0..<featureCount).contains($0) }) else {
            throw FeaturizationError.invalidInput("feature indices must be non-empty and inside 0..<\(featureCount)")
        }
        let n = features.rows
        let kept = (0..<featureCount).filter { !indices.contains($0) }
        var values = [Double](repeating: 0, count: n * (kept.count + 1))
        for i in 0..<n {
            for (t, j) in kept.enumerated() { values[i * (kept.count + 1) + t] = features[i, j] }
            let picked = indices.map { features[i, $0] }
            let reduced: Double
            switch reduction {
            case .mean: reduced = picked.reduce(0, +) / Double(picked.count)
            case .maximum: reduced = picked.contains(where: \.isNaN) ? .nan : picked.max()!
            case .median:
                if picked.contains(where: \.isNaN) { reduced = .nan } else {
                    let s = picked.sorted(), m = s.count
                    reduced = m % 2 == 1 ? s[m / 2] : (s[m / 2 - 1] + s[m / 2]) / 2
                }
            }
            values[i * (kept.count + 1) + kept.count] = reduced
        }
        return try Featurization(features: FeatureMatrix(rows: n, columns: kept.count + 1, values: values),
                                 scanRows: scanRows, scanColumns: scanColumns, name: name)
    }

    /// `MinMaxScaler()`: each column mapped to `[0, 1]` by `x·scale + min` with `scale = 1/range` (1 for a constant column).
    public func minMaxScaled() -> Featurization {
        let n = features.rows, d = featureCount
        var values = features.values
        for j in 0..<d {
            var lo = Double.infinity, hi = -Double.infinity
            for i in 0..<n { lo = min(lo, values[i * d + j]); hi = max(hi, values[i * d + j]) }
            var range = hi - lo
            if range < 10 * Double.ulpOfOne { range = 1 }
            let scale = 1.0 / range
            let offset = 0 - lo * scale
            for i in 0..<n { values[i * d + j] = values[i * d + j] * scale + offset }
        }
        var out = self
        out.features = try! FeatureMatrix(rows: n, columns: d, values: values)
        return out
    }

    /// `shift_positive()`: adds `|min|` of the whole matrix to every entry.
    public func shiftedPositive() -> Featurization {
        let shift = abs(features.values.min() ?? 0)
        var out = self
        out.features = try! FeatureMatrix(rows: features.rows, columns: features.columns, values: features.values.map { $0 + shift })
        return out
    }

    // MARK: PCA

    /// `PCA(n_components=components).fit_transform(features)` with scikit-learn 1.9.1's sign convention (`svd_flip` on the
    /// components: the largest-magnitude entry of each component is positive, the first index winning a tie). The projection is
    /// `PCAResult.scores`. 1.9.1 picks the full SVD or the covariance eigendecomposition by shape; both are computed here through
    /// the covariance eigendecomposition of `PCA.fit`, which agrees with them to rounding when the eigenvalues are separated.
    public func pca(components: Int) throws -> PCAResult {
        let base: PCAResult
        do {
            base = try PCA.fit(features.values, rows: features.rows, columns: featureCount, components: components,
                               variant: .scikitLearn)
        } catch let error as EmbeddingError {
            throw FeaturizationError.invalidInput(error.description)
        }
        var comps = base.components
        var scores = base.scores
        let d = base.columns, k = base.componentCount, n = base.rows
        for c in 0..<k {
            var best = 0
            for j in 1..<d where abs(comps[c * d + j]) > abs(comps[c * d + best]) { best = j }
            if comps[c * d + best] < 0 {
                for j in 0..<d { comps[c * d + j] = -comps[c * d + j] }
                for i in 0..<n { scores[i * k + c] = -scores[i * k + c] }
            }
        }
        return PCAResult(rows: n, columns: d, componentCount: k, mean: base.mean, components: comps,
                         explainedVariance: base.explainedVariance, explainedVarianceRatio: base.explainedVarianceRatio,
                         singularValues: base.singularValues, scores: scores)
    }

    // MARK: NMF

    /// `Featurization.NMF`: `numModels` independent scikit-learn NMFs of the (non-negative) features, each started from
    /// `maxComponents` components and, when `maxIterations > 1`, repeated on its own weights with the columns whose Pearson
    /// correlation reaches `mergeThreshold` added together (`_nmf_single`).
    ///
    /// Without `randomSeed` the seeds are drawn from `RandomState(42).randint(5000)`, one per model; with it every model uses
    /// that seed. As in py4DSTEM only the first of the columns merged in a round is deleted, and the weights kept for a model
    /// are those of the last merge, not of the last fit. `saveAllModels == false` keeps the model with the lowest mean
    /// reconstruction error.
    ///
    /// `isCancelled` is asked before every NMF fit; when it answers true the call throws `CancellationError`.
    public func nmf(maxComponents: Int, numModels: Int, mergeThreshold: Double = 1, maxIterations: Int = 1,
                    randomSeed: UInt32? = nil, saveAllModels: Bool = true,
                    isCancelled: (@Sendable () -> Bool)? = nil) throws -> FeaturizationNMFResult {
        guard maxComponents >= 1, numModels >= 1, maxIterations >= 1 else {
            throw FeaturizationError.invalidInput("maxComponents, numModels and maxIterations must be at least 1")
        }
        let n = features.rows
        var rng = NumPyRandomState(seed: 42)
        var seed = randomSeed ?? 0
        var best = Double.infinity
        var saved = [FeatureMatrix]()
        var bestMatrix: FeatureMatrix?
        for _ in 0..<numModels {
            if randomSeed == nil { seed = UInt32(rng.randint(5000)) }
            var components = maxComponents
            var reconstruction = 0.0
            var counter = 0
            var current = [Double](), currentColumns = 0
            for _ in 0..<maxIterations {
                if isCancelled?() == true { throw CancellationError() }
                var second: [Double]?
                var fit: NMF.Result
                if counter == 0 {
                    fit = try NMF.fit(features.values, rows: n, columns: featureCount, components: components,
                                      initialization: .automatic(seed: seed), isCancelled: isCancelled)
                    current = fit.w; currentColumns = fit.components
                } else {
                    do {
                        fit = try NMF.fit(current, rows: n, columns: currentColumns, components: components,
                                          initialization: .automatic(seed: seed), isCancelled: isCancelled)
                    } catch FeaturizationError.degenerateInitialisation {
                        break       // py4DSTEM: "Warning encountered in NMF: Returning last result"
                    }
                    second = fit.w
                }
                reconstruction += fit.reconstructionError
                counter += 1
                if counter >= maxIterations { break }
                // the correlation of the weight columns
                let source: [Double], sourceColumns: Int
                if counter > 1 {
                    source = second!; sourceColumns = fit.components
                } else {
                    source = current; sourceColumns = currentColumns
                }
                guard sourceColumns >= 2 else {
                    throw FeaturizationError.invalidInput("the correlation of fewer than two columns is undefined (numpy raises)")
                }
                guard let lower = Featurization.lowerCorrelation(source, n, sourceColumns, strict: counter > 1) else {
                    break           // py4DSTEM: "Warning encountered in correlation: Returning last result"
                }
                if counter > 1 { current = source; currentColumns = sourceColumns }
                // merge the correlated columns
                var topValue = -Double.infinity
                for v in lower where !v.isNaN && v > topValue { topValue = v }
                if topValue >= mergeThreshold {
                    var pairs = [(Int, Int)]()
                    for a in 0..<sourceColumns { for b in 0..<sourceColumns where lower[a * sourceColumns + b] >= mergeThreshold {
                        pairs.append((a, b))
                    } }
                    for (a, b) in pairs { for i in 0..<n { current[i * currentColumns + a] += current[i * currentColumns + b] } }
                    let drop = pairs[pairs.count - 1].1
                    var next = [Double](repeating: 0, count: n * (currentColumns - 1))
                    for i in 0..<n {
                        var t = 0
                        for j in 0..<currentColumns where j != drop { next[i * (currentColumns - 1) + t] = current[i * currentColumns + j]; t += 1 }
                    }
                    current = next
                    currentColumns -= 1
                } else {
                    break
                }
                components = currentColumns - 1
                if components <= 2 { break }
            }
            let matrix = try FeatureMatrix(rows: n, columns: currentColumns, values: current)
            if saveAllModels {
                saved.append(matrix)
            } else if counter > 0, reconstruction / Double(counter) < best {
                best = reconstruction / Double(counter)
                bestMatrix = matrix
            }
        }
        if saveAllModels { return FeaturizationNMFResult(models: saved) }
        return FeaturizationNMFResult(models: bestMatrix.map { [$0] } ?? [])
    }

    /// `np.tril(np.corrcoef(m, rowvar=False), k=-1)` of the `n × k` matrix, row-major `k × k`. With `strict` (numpy's
    /// `errstate(invalid, divide="raise")`) nil is returned when a column has zero variance; otherwise NaN fills those entries.
    static func lowerCorrelation(_ m: [Double], _ n: Int, _ k: Int, strict: Bool) -> [Double]? {
        var mean = [Double](repeating: 0, count: k)
        for i in 0..<n { for j in 0..<k { mean[j] += m[i * k + j] } }
        for j in 0..<k { mean[j] /= Double(n) }
        var c = [Double](repeating: 0, count: k * k)
        for a in 0..<k { for b in 0..<k {
            var s = 0.0
            for i in 0..<n { s += (m[i * k + a] - mean[a]) * (m[i * k + b] - mean[b]) }
            c[a * k + b] = s * (1.0 / Double(n - 1))
        } }
        let std = (0..<k).map { c[$0 * k + $0].squareRoot() }
        if strict && std.contains(where: { $0 == 0 || $0.isNaN }) { return nil }
        var out = [Double](repeating: 0, count: k * k)
        for a in 0..<k { for b in 0..<a {
            let v = (c[a * k + b] / std[a]) / std[b]
            out[a * k + b] = v.isNaN ? v : min(1, max(-1, v))
        } }
        return out
    }

    // MARK: GMM

    /// `Featurization.GMM`: for each of `numModels` seeds, each entry of `components` and each covariance type, a
    /// `GaussianMixture` fit (`_gmm_single` with `return_all=True`), in that loop order (models outermost). Without
    /// `randomSeed` the seeds are drawn from `RandomState(42).randint(5000)`, one per model.
    ///
    /// `isCancelled` is asked before each fit and every EM iteration; when it answers true the call throws `CancellationError`.
    public func gmm(covarianceTypes: [GaussianMixture.CovarianceType], components: [Int], numModels: Int,
                    randomSeed: UInt32? = nil, isCancelled: (@Sendable () -> Bool)? = nil) throws -> [GaussianMixture.Fit] {
        guard numModels >= 1, !components.isEmpty, !covarianceTypes.isEmpty else {
            throw FeaturizationError.invalidInput("numModels, components and covarianceTypes must not be empty")
        }
        var rng = NumPyRandomState(seed: 42)
        var seed = randomSeed ?? 0
        var fits = [GaussianMixture.Fit]()
        for _ in 0..<numModels {
            if randomSeed == nil { seed = UInt32(rng.randint(5000)) }
            for count in components {
                for type in covarianceTypes {
                    if isCancelled?() == true { throw CancellationError() }
                    fits.append(try GaussianMixture.fit(features.values, rows: features.rows, columns: featureCount,
                                                        components: count, covarianceType: type, seed: seed,
                                                        isCancelled: isCancelled))
                }
            }
        }
        return fits
    }

    // MARK: class patterns

    /// The weighted class patterns of `get_class_DPs` for one weight matrix (NMF weights, GMM responsibilities, PCA scores):
    /// for each class `l`, the patterns of the positions whose weight exceeds `threshold`, weighted and divided by the sum of
    /// those weights. A class with no such position yields NaN (0/0, as in NumPy). `patterns` is `positions × pixels`,
    /// row-major, in scan order.
    ///
    /// py4DSTEM's own `"gmm"` branch loops `range(np.max(labels))`, which leaves out the last class; this takes whatever
    /// columns `weights` has.
    public static func classPatterns(weights: FeatureMatrix, patterns: [Float], pixelCount: Int,
                                     threshold: Double) throws -> [[Double]] {
        guard pixelCount > 0, patterns.count == weights.rows * pixelCount else {
            throw FeaturizationError.invalidInput("patterns.count \(patterns.count) is not positions \(weights.rows) × pixels \(pixelCount)")
        }
        return (0..<weights.columns).map { l in
            var pattern = [Double](repeating: 0, count: pixelCount)
            var total = 0.0
            for i in 0..<weights.rows {
                let w = weights[i, l]
                guard w > threshold else { continue }
                for p in 0..<pixelCount { pattern[p] += Double(patterns[i * pixelCount + p]) * w }
                total += w
            }
            return pattern.map { $0 / total }
        }
    }
}
