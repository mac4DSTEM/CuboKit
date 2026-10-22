//
//  GaussianMixture.swift
//  scikit-learn's `GaussianMixture` as py4DSTEM's `Featurization.GMM` runs it (`random_state=<int>`, `init_params="kmeans"`,
//  `n_init=1`, `max_iter=100`, `tol=1e-3`, `reg_covar=1e-6`): the k-means initialisation with scikit-learn's k-means++ seeding
//  drawn from NumPy's `RandomState`, the four covariance types, the EM loop and `bic`. Double throughout, CPU.
//

import Foundation

/// A Gaussian mixture fitted by expectation maximisation, after scikit-learn.
public enum GaussianMixture {
    /// scikit-learn's `covariance_type`.
    public enum CovarianceType: String, Sendable, Hashable, CaseIterable {
        case full, tied, diag, spherical
    }

    /// The fitted parameters (scikit-learn's `weights_`, `means_`, `covariances_` and `precisions_cholesky_`).
    public struct Model: Sendable, Equatable {
        /// Samples the model was fitted on, features, components.
        public let rows: Int
        public let columns: Int
        public let components: Int
        public let covarianceType: CovarianceType
        /// Length `components`.
        public let weights: [Double]
        /// `components × columns`, row-major.
        public let means: [Double]
        /// `full`: `components × columns × columns`; `tied`: `columns × columns`; `diag`: `components × columns`;
        /// `spherical`: `components`. Row-major.
        public let covariances: [Double]
        /// Same shapes as `covariances`. `full` and `tied` hold the transposed inverse of the lower Cholesky factor (upper
        /// triangular); `diag` and `spherical` hold `1/sqrt(variance)`.
        public let precisionsCholesky: [Double]
        /// The mean log-likelihood per sample at the last E-step (`lower_bound_`).
        public let lowerBound: Double
        /// EM iterations run (`n_iter_`).
        public let iterations: Int
        /// False when `maxIterations` ended the run (scikit-learn warns `ConvergenceWarning`).
        public let converged: Bool

        /// The `components`-column responsibilities of `x` (`rows × columns`) under this model (`predict_proba`).
        public func probabilities(of x: [Double], rows n: Int) throws -> [Double] {
            guard n > 0, x.count == n * columns else {
                throw FeaturizationError.invalidInput("x.count \(x.count) is not rows \(n) × columns \(columns)")
            }
            return GaussianMixture.responsibilities(self, x, n).logResp.map { exp($0) }
        }

        /// The number of free parameters (`_n_parameters`).
        public var parameterCount: Int {
            let d = Double(columns), k = Double(components)
            let cov: Double
            switch covarianceType {
            case .full: cov = k * d * (d + 1) / 2
            case .diag: cov = k * d
            case .tied: cov = d * (d + 1) / 2
            case .spherical: cov = k
            }
            return Int(cov + k * d + k - 1)
        }

        /// The Bayesian information criterion of `x` under this model (`bic`).
        public func bic(of x: [Double], rows n: Int) throws -> Double {
            guard n > 0, x.count == n * columns else {
                throw FeaturizationError.invalidInput("x.count \(x.count) is not rows \(n) × columns \(columns)")
            }
            let score = GaussianMixture.responsibilities(self, x, n).logProbNorm.reduce(0, +) / Double(n)
            return -2 * score * Double(n) + Double(parameterCount) * log(Double(n))
        }
    }

    /// A fitted model with the labels of the training samples and their responsibilities.
    public struct Fit: Sendable, Equatable {
        public let model: Model
        /// The most probable component of each sample (`fit_predict`).
        public let labels: [Int]
        /// `rows × components` (`predict_proba`).
        public let probabilities: [Double]
        /// `model.bic` on the training data.
        public let bic: Double
    }

    /// `GaussianMixture(n_components, covariance_type, random_state=seed).fit_predict(x)` with scikit-learn's defaults
    /// (`n_init=1`, `init_params="kmeans"`). `x` is `rows × columns`, row-major.
    ///
    /// The k-means initialisation seeds with k-means++ drawn from `RandomState(seed)` (the choice of the first centre, then
    /// `2 + int(ln k)` greedy trials per further centre), then runs Lloyd's algorithm (`KMeans.fit`, tolerance 1e-4, 300
    /// iterations). Distances use scikit-learn's expanded form, so the draws agree except where two candidate distances tie to
    /// rounding.
    ///
    /// `isCancelled` is asked at the start of every EM iteration; when it answers true the call throws `CancellationError`.
    public static func fit(_ x: [Double], rows n: Int, columns d: Int, components k: Int,
                           covarianceType: CovarianceType = .full, seed: UInt32, regCovar: Double = 1e-6,
                           tolerance: Double = 1e-3, maxIterations: Int = 100,
                           isCancelled: (@Sendable () -> Bool)? = nil) throws -> Fit {
        guard n >= 2, d > 0, x.count == n * d else {
            throw FeaturizationError.invalidInput("x.count \(x.count) is not rows \(n) × columns \(d), or fewer than two rows")
        }
        guard k >= 1, k <= n else { throw FeaturizationError.invalidInput("components \(k) must be in 1...\(n)") }
        guard regCovar >= 0, tolerance >= 0, maxIterations >= 1 else {
            throw FeaturizationError.invalidInput("regCovar and tolerance must not be negative and maxIterations at least 1")
        }
        guard x.allSatisfy(\.isFinite) else { throw FeaturizationError.nonFinite }
        var rng = NumPyRandomState(seed: seed)
        let labels0 = try kmeansLabels(x, n, d, k, &rng)
        var resp = [Double](repeating: 0, count: n * k)
        for i in 0..<n { resp[i * k + labels0[i]] = 1 }
        var parameters = try estimate(x, n, d, k, resp, regCovar, covarianceType)
        parameters.weights = parameters.weights.map { $0 / Double(n) }
        var model = try makeModel(parameters, n, d, k, covarianceType, regCovar: regCovar, lowerBound: -.infinity,
                                  iterations: 0, converged: false)
        var lowerBound = -Double.infinity
        var converged = false
        var iterations = 0
        for iteration in 1...maxIterations {
            if isCancelled?() == true { throw CancellationError() }
            iterations = iteration
            let previous = lowerBound
            let e = responsibilities(model, x, n)
            let respNew = e.logResp.map { exp($0) }
            var p = try estimate(x, n, d, k, respNew, regCovar, covarianceType)
            let total = p.weights.reduce(0, +)
            p.weights = p.weights.map { $0 / total }
            lowerBound = e.logProbNorm.reduce(0, +) / Double(n)
            model = try makeModel(p, n, d, k, covarianceType, regCovar: regCovar, lowerBound: lowerBound,
                                  iterations: iteration, converged: false)
            if abs(lowerBound - previous) < tolerance { converged = true; break }
        }
        model = Model(rows: n, columns: d, components: k, covarianceType: covarianceType, weights: model.weights,
                      means: model.means, covariances: model.covariances, precisionsCholesky: model.precisionsCholesky,
                      lowerBound: lowerBound, iterations: iterations, converged: converged)
        let e = responsibilities(model, x, n)
        var labels = [Int](repeating: 0, count: n)
        for i in 0..<n {
            var best = 0
            for c in 1..<k where e.logResp[i * k + c] > e.logResp[i * k + best] { best = c }
            labels[i] = best
        }
        let bic = -2 * (e.logProbNorm.reduce(0, +) / Double(n)) * Double(n) + Double(model.parameterCount) * log(Double(n))
        return Fit(model: model, labels: labels, probabilities: e.logResp.map { exp($0) }, bic: bic)
    }

    // MARK: parameters

    private struct Parameters {
        var weights: [Double]
        var means: [Double]
        var covariances: [Double]
    }

    private static func estimate(_ x: [Double], _ n: Int, _ d: Int, _ k: Int, _ resp: [Double], _ reg: Double,
                                  _ type: CovarianceType) throws -> Parameters {
        let eps = 10 * Double.ulpOfOne
        var nk = [Double](repeating: 0, count: k)
        for i in 0..<n { for c in 0..<k { nk[c] += resp[i * k + c] } }
        for c in 0..<k { nk[c] += eps }
        var means = [Double](repeating: 0, count: k * d)
        for i in 0..<n { for c in 0..<k {
            let r = resp[i * k + c]
            if r == 0 { continue }
            for j in 0..<d { means[c * d + j] += r * x[i * d + j] }
        } }
        for c in 0..<k { for j in 0..<d { means[c * d + j] /= nk[c] } }
        var cov: [Double]
        switch type {
        case .full:
            cov = [Double](repeating: 0, count: k * d * d)
            var diff = [Double](repeating: 0, count: n * d)
            for c in 0..<k {
                for i in 0..<n { for a in 0..<d { diff[i * d + a] = x[i * d + a] - means[c * d + a] } }
                for a in 0..<d { for b in 0..<d {
                    var s = 0.0
                    for i in 0..<n { s += (resp[i * k + c] * diff[i * d + a]) * diff[i * d + b] }
                    cov[(c * d + a) * d + b] = s / nk[c]
                } }
                for a in 0..<d { cov[(c * d + a) * d + a] += reg }
            }
        case .tied:
            cov = [Double](repeating: 0, count: d * d)
            var total = 0.0
            for c in 0..<k { total += nk[c] }
            for a in 0..<d { for b in 0..<d {
                var avgX2 = 0.0
                for i in 0..<n { avgX2 += x[i * d + a] * x[i * d + b] }
                var avgMeans2 = 0.0
                for c in 0..<k { avgMeans2 += (nk[c] * means[c * d + a]) * means[c * d + b] }
                cov[a * d + b] = (avgX2 - avgMeans2) / total
            } }
            for a in 0..<d { cov[a * d + a] += reg }
        case .diag, .spherical:
            var diag = [Double](repeating: 0, count: k * d)
            for c in 0..<k { for j in 0..<d {
                var s = 0.0
                for i in 0..<n { s += resp[i * k + c] * (x[i * d + j] * x[i * d + j]) }
                diag[c * d + j] = s / nk[c] - means[c * d + j] * means[c * d + j] + reg
            } }
            if type == .diag {
                cov = diag
            } else {
                cov = (0..<k).map { c in
                    var s = 0.0
                    for j in 0..<d { s += diag[c * d + j] }
                    return s / Double(d)
                }
            }
        }
        return Parameters(weights: nk, means: means, covariances: cov)
    }

    private static func makeModel(_ p: Parameters, _ n: Int, _ d: Int, _ k: Int, _ type: CovarianceType, regCovar: Double,
                                  lowerBound: Double, iterations: Int, converged: Bool) throws -> Model {
        var chol: [Double]
        switch type {
        case .full:
            chol = []
            for c in 0..<k {
                chol += try precisionCholesky(Array(p.covariances[(c * d * d)..<((c + 1) * d * d)]), d)
            }
        case .tied:
            chol = try precisionCholesky(p.covariances, d)
        case .diag, .spherical:
            guard !p.covariances.contains(where: { !($0 > 0) }) else { throw FeaturizationError.illDefinedCovariance }
            chol = p.covariances.map { 1 / $0.squareRoot() }
        }
        return Model(rows: n, columns: d, components: k, covarianceType: type, weights: p.weights, means: p.means,
                     covariances: p.covariances, precisionsCholesky: chol, lowerBound: lowerBound, iterations: iterations,
                     converged: converged)
    }

    /// The transposed inverse of the lower Cholesky factor of the symmetric `d × d` matrix (its lower triangle is read).
    private static func precisionCholesky(_ a: [Double], _ d: Int) throws -> [Double] {
        var l = [Double](repeating: 0, count: d * d)
        for j in 0..<d {
            var s = a[j * d + j]
            for t in 0..<j { s -= l[j * d + t] * l[j * d + t] }
            guard s > 0, s.isFinite else { throw FeaturizationError.illDefinedCovariance }
            let ljj = s.squareRoot()
            l[j * d + j] = ljj
            for i in (j + 1)..<max(j + 1, d) {
                var v = a[i * d + j]
                for t in 0..<j { v -= l[i * d + t] * l[j * d + t] }
                l[i * d + j] = v / ljj
            }
        }
        var inverse = [Double](repeating: 0, count: d * d)       // inverse of L, lower triangular
        for c in 0..<d {
            for i in c..<d {
                var s = i == c ? 1.0 : 0.0
                for t in c..<i { s -= l[i * d + t] * inverse[t * d + c] }
                inverse[i * d + c] = s / l[i * d + i]
            }
        }
        var out = [Double](repeating: 0, count: d * d)
        for i in 0..<d { for j in 0..<d { out[i * d + j] = inverse[j * d + i] } }
        return out
    }

    // MARK: E-step

    fileprivate static func responsibilities(_ m: Model, _ x: [Double], _ n: Int) -> (logResp: [Double], logProbNorm: [Double]) {
        let d = m.columns, k = m.components
        var logProb = [Double](repeating: 0, count: n * k)
        let logDet: [Double]
        switch m.covarianceType {
        case .full:
            logDet = (0..<k).map { c in
                var s = 0.0
                for j in 0..<d { s += log(m.precisionsCholesky[(c * d + j) * d + j]) }
                return s
            }
            for c in 0..<k {
                let p = m.precisionsCholesky
                let base = c * d * d
                var mup = [Double](repeating: 0, count: d)
                for j in 0..<d { var s = 0.0; for i in 0..<d { s += m.means[c * d + i] * p[base + i * d + j] }; mup[j] = s }
                var xp = [Double](repeating: 0, count: d)
                for r in 0..<n {
                    for j in 0..<d { var s = 0.0; for i in 0..<d { s += x[r * d + i] * p[base + i * d + j] }; xp[j] = s }
                    var sq = 0.0
                    for j in 0..<d { let y = xp[j] - mup[j]; sq += y * y }
                    logProb[r * k + c] = sq
                }
            }
        case .tied:
            var s0 = 0.0
            for j in 0..<d { s0 += log(m.precisionsCholesky[j * d + j]) }
            logDet = [Double](repeating: s0, count: k)
            let p = m.precisionsCholesky
            for c in 0..<k {
                var mup = [Double](repeating: 0, count: d)
                for j in 0..<d { var s = 0.0; for i in 0..<d { s += m.means[c * d + i] * p[i * d + j] }; mup[j] = s }
                var xp = [Double](repeating: 0, count: d)
                for r in 0..<n {
                    for j in 0..<d { var s = 0.0; for i in 0..<d { s += x[r * d + i] * p[i * d + j] }; xp[j] = s }
                    var sq = 0.0
                    for j in 0..<d { let y = xp[j] - mup[j]; sq += y * y }
                    logProb[r * k + c] = sq
                }
            }
        case .diag:
            logDet = (0..<k).map { c in
                var s = 0.0
                for j in 0..<d { s += log(m.precisionsCholesky[c * d + j]) }
                return s
            }
            for c in 0..<k {
                var meanTerm = 0.0
                for j in 0..<d { let pr = m.precisionsCholesky[c * d + j] * m.precisionsCholesky[c * d + j]; meanTerm += m.means[c * d + j] * m.means[c * d + j] * pr }
                for r in 0..<n {
                    var cross = 0.0, quad = 0.0
                    for j in 0..<d {
                        let pr = m.precisionsCholesky[c * d + j] * m.precisionsCholesky[c * d + j]
                        cross += x[r * d + j] * (m.means[c * d + j] * pr)
                        quad += (x[r * d + j] * x[r * d + j]) * pr
                    }
                    logProb[r * k + c] = meanTerm - 2.0 * cross + quad
                }
            }
        case .spherical:
            logDet = (0..<k).map { Double(d) * log(m.precisionsCholesky[$0]) }
            for c in 0..<k {
                let pr = m.precisionsCholesky[c] * m.precisionsCholesky[c]
                var meanSq = 0.0
                for j in 0..<d { meanSq += m.means[c * d + j] * m.means[c * d + j] }
                for r in 0..<n {
                    var cross = 0.0, norm2 = 0.0
                    for j in 0..<d { cross += x[r * d + j] * m.means[c * d + j]; norm2 += x[r * d + j] * x[r * d + j] }
                    logProb[r * k + c] = meanSq * pr - 2 * (cross * pr) + norm2 * pr
                }
            }
        }
        let logWeights = m.weights.map { log($0) }
        var logResp = [Double](repeating: 0, count: n * k)
        var norm = [Double](repeating: 0, count: n)
        let constant = Double(d) * log(2 * Double.pi)
        for r in 0..<n {
            var weighted = [Double](repeating: 0, count: k)
            for c in 0..<k { weighted[c] = (-0.5 * (constant + logProb[r * k + c]) + logDet[c]) + logWeights[c] }
            let top = weighted.max() ?? 0
            var s = 0.0
            for c in 0..<k { s += exp(weighted[c] - top) }
            let ln = log(s) + top
            norm[r] = ln
            for c in 0..<k { logResp[r * k + c] = weighted[c] - ln }
        }
        return (logResp, norm)
    }

    // MARK: k-means initialisation

    /// `KMeans(n_clusters=k, n_init=1, random_state=rng).fit(x).labels_`: the data are centred on their mean, seeded by
    /// `_kmeans_plusplus` and run through Lloyd's algorithm.
    private static func kmeansLabels(_ x: [Double], _ n: Int, _ d: Int, _ k: Int, _ rng: inout NumPyRandomState) throws -> [Int] {
        var mean = [Double](repeating: 0, count: d)
        for i in 0..<n { for j in 0..<d { mean[j] += x[i * d + j] } }
        for j in 0..<d { mean[j] /= Double(n) }
        var xc = x
        for i in 0..<n { for j in 0..<d { xc[i * d + j] -= mean[j] } }
        var norms = [Double](repeating: 0, count: n)
        for i in 0..<n { var s = 0.0; for j in 0..<d { s += xc[i * d + j] * xc[i * d + j] }; norms[i] = s }
        func distances(from c: Int) -> [Double] {
            (0..<n).map { j in
                var dot = 0.0
                for t in 0..<d { dot += xc[c * d + t] * xc[j * d + t] }
                return max(0, ((-2 * dot) + norms[c]) + norms[j])
            }
        }
        let trials = 2 + Int(log(Double(k)))
        var centres = [Int]()
        centres.append(rng.choice(probabilities: [Double](repeating: 1.0 / Double(n), count: n)))
        var closest = distances(from: centres[0])
        var potential = closest.reduce(0, +)
        for _ in 1..<max(1, k) {
            var cumulative = [Double](repeating: 0, count: n)
            var running = 0.0
            for i in 0..<n { running += closest[i]; cumulative[i] = running }
            var candidates = [Int]()
            for _ in 0..<trials {
                let value = rng.randomSample() * potential
                var low = 0, high = n
                while low < high {
                    let mid = (low + high) / 2
                    if cumulative[mid] < value { low = mid + 1 } else { high = mid }
                }
                candidates.append(min(low, n - 1))
            }
            var bestPotential = Double.infinity
            var bestDistance = closest
            var bestId = candidates[0]
            for id in candidates {
                let dist = distances(from: id)
                let merged = zip(closest, dist).map { min($0, $1) }
                let pot = merged.reduce(0, +)
                if pot < bestPotential { bestPotential = pot; bestDistance = merged; bestId = id }
            }
            potential = bestPotential
            closest = bestDistance
            centres.append(bestId)
        }
        var init0 = [Double]()
        for id in centres { init0 += xc[(id * d)..<((id + 1) * d)] }
        return try KMeans.fit(xc, rows: n, columns: d, initialCentroids: init0).labels
    }
}
