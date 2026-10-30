//
//  NMF.swift
//  scikit-learn's `NMF` (solver "cd", beta_loss "frobenius", no regularisation, no shuffle) as py4DSTEM's `Featurization.NMF` and
//  `BraggVectorClassification.nmf` call it: the NNDSVD family of initialisations, the cyclic coordinate descent of
//  `_cdnmf_fast.pyx` and the stopping rule of `_fit_coordinate_descent`. Double throughout, CPU.
//

import CEmbeddingLAPACK
import Foundation

/// What went wrong in the NMF, Gaussian mixture and featurization routines.
public enum FeaturizationError: Error, Sendable, Equatable, CustomStringConvertible {
    /// A shape, count or option is outside what scikit-learn accepts (the message says which).
    case invalidInput(String)
    /// The data contain NaN or infinity.
    case nonFinite
    /// NMF input must be non-negative (scikit-learn: "Negative values in data passed to NMF").
    case negativeValues
    /// NNDSVD met a zero norm where scikit-learn divides by it (NaN there); an ill-defined initialisation.
    case degenerateInitialisation
    /// A covariance is not positive definite (scikit-learn: "ill-defined empirical covariance").
    case illDefinedCovariance
    /// LAPACK reported a failure (its `info` value).
    case solverFailed(Int)

    public var description: String {
        switch self {
        case .invalidInput(let m): "Invalid input: \(m)"
        case .nonFinite: "The data contain non-finite values."
        case .negativeValues: "NMF needs non-negative data."
        case .degenerateInitialisation: "NNDSVD initialisation met a zero-norm singular vector."
        case .illDefinedCovariance: "A component has an ill-defined covariance (singleton or collapsed samples); reduce the components or scale the data."
        case .solverFailed(let info): "The SVD failed (LAPACK info \(info))."
        }
    }
}

/// Non-negative matrix factorisation `X ≈ W·H` after scikit-learn's `NMF(solver="cd", beta_loss="frobenius")`.
public enum NMF {
    /// scikit-learn's `init`.
    public enum Initialization: Sendable, Equatable {
        /// `init=None`: `.nndsvda` when `components <= min(rows, columns)`, otherwise `.random(seed)`.
        case automatic(seed: UInt32)
        /// `"nndsvd"`: exact singular vectors (scikit-learn takes a randomised SVD that converges to them); zeros stay zero.
        case nndsvd
        /// `"nndsvda"`: `.nndsvd` with the zeros filled by the mean of `X`.
        case nndsvda
        /// `"nndsvdar"`: `.nndsvd` with the zeros filled by small values drawn from NumPy's `RandomState(seed)`.
        case nndsvdar(seed: UInt32)
        /// `"random"`: `sqrt(mean/components)` times the absolute value of standard normals from `RandomState(seed)` (`H` first).
        case random(seed: UInt32)
        /// `"custom"`: the given `W` (`rows × components`) and `H` (`components × columns`), row-major.
        case custom(w: [Double], h: [Double])
    }

    /// The factors and what the solver did.
    public struct Result: Sendable, Equatable {
        public let rows: Int
        public let columns: Int
        public let components: Int
        /// `rows × components`, row-major (scikit-learn's `fit_transform` result).
        public let w: [Double]
        /// `components × columns`, row-major (scikit-learn's `components_`).
        public let h: [Double]
        /// `‖X − WH‖_F` (scikit-learn's `reconstruction_err_`).
        public let reconstructionError: Double
        /// Sweeps done (scikit-learn's `n_iter_`).
        public let iterations: Int
        /// False when the iteration cap ended the run (scikit-learn warns `ConvergenceWarning`).
        public let converged: Bool
    }

    /// Factors the non-negative `rows × columns` row-major matrix `x`. `tolerance` and `maxIterations` are scikit-learn's `tol`
    /// (default 1e-4) and `max_iter` (200). Each sweep updates `W` then `H`, one component at a time in order, and stops when
    /// the summed projected-gradient violation has fallen to `tolerance` times its first-sweep value.
    ///
    /// `isCancelled` is asked at the start of every sweep; when it answers true the call throws `CancellationError`.
    public static func fit(_ x: [Double], rows n: Int, columns d: Int, components k: Int,
                           initialization: Initialization = .automatic(seed: 0), tolerance: Double = 1e-4,
                           maxIterations: Int = 200, isCancelled: (@Sendable () -> Bool)? = nil) throws -> Result {
        guard n > 0, d > 0, x.count == n * d else {
            throw FeaturizationError.invalidInput("x.count \(x.count) is not rows \(n) × columns \(d)")
        }
        guard k >= 1, maxIterations >= 1, tolerance >= 0, tolerance.isFinite else {
            throw FeaturizationError.invalidInput("components and maxIterations must be at least 1 and the tolerance not negative")
        }
        guard x.allSatisfy(\.isFinite) else { throw FeaturizationError.nonFinite }
        guard !x.contains(where: { $0 < 0 }) else { throw FeaturizationError.negativeValues }
        let (w0, h0) = try initialFactors(x, n, d, k, initialization)
        let (w, h, iterations, converged) = try coordinateDescent(x, n, d, k, w: w0, h: h0, tolerance: tolerance,
                                                                  maxIterations: maxIterations, isCancelled: isCancelled)
        var squared = 0.0
        for i in 0..<n {
            for j in 0..<d {
                var wh = 0.0
                for c in 0..<k { wh += w[i * k + c] * h[c * d + j] }
                let r = x[i * d + j] - wh
                squared += r * r
            }
        }
        return Result(rows: n, columns: d, components: k, w: w, h: h, reconstructionError: squared.squareRoot(),
                      iterations: iterations, converged: converged)
    }

    // MARK: coordinate descent

    /// The sweeps of `_fit_coordinate_descent` from the start `w` (`n × k`) and `h` (`k × d`), unchecked: each sweep updates `W`
    /// then `H` and the loop stops when the summed projected-gradient violation has fallen to `tolerance` times its first-sweep
    /// value, or after `maxIterations` (at least 1) sweeps. `NMF.fit` and CuboBragg's classification NMF both run this.
    package static func coordinateDescent(_ x: [Double], _ n: Int, _ d: Int, _ k: Int, w w0: [Double], h h0: [Double],
                                          tolerance: Double, maxIterations: Int, isCancelled: (@Sendable () -> Bool)?) throws
        -> (w: [Double], h: [Double], iterations: Int, converged: Bool) {
        var w = w0
        var ht = transposed(h0, k, d)                // d × k
        let xt = transposed(x, n, d)                 // d × n
        var violationInit = 0.0
        var iterations = 0
        var converged = false
        for sweep in 1...max(1, maxIterations) {
            if isCancelled?() == true { throw CancellationError() }
            iterations = sweep
            var violation = update(&w, n, k, other: ht, rows: d, data: x, dataColumns: d)
            violation += update(&ht, d, k, other: w, rows: n, data: xt, dataColumns: n)
            if sweep == 1 { violationInit = violation }
            if violationInit == 0 { converged = true; break }
            if violation / violationInit <= tolerance { converged = true; break }
        }
        return (w, transposed(ht, d, k), iterations, converged)
    }

    /// One `_update_coordinate_descent`: `target` (`m × k`) is moved one component at a time against
    /// `HHt = otherᵀ·other` and `XHt = data · other`, where `data` is `m × dataColumns` and `other` is `dataColumns × k`.
    private static func update(_ target: inout [Double], _ m: Int, _ k: Int, other: [Double], rows p: Int,
                               data: [Double], dataColumns: Int) -> Double {
        var hht = [Double](repeating: 0, count: k * k)
        for r in 0..<p {
            for a in 0..<k {
                let v = other[r * k + a]
                if v == 0 { continue }
                for b in 0..<k { hht[a * k + b] += v * other[r * k + b] }
            }
        }
        var xht = [Double](repeating: 0, count: m * k)
        for i in 0..<m {
            for j in 0..<dataColumns {
                let v = data[i * dataColumns + j]
                if v == 0 { continue }
                for c in 0..<k { xht[i * k + c] += v * other[j * k + c] }
            }
        }
        var violation = 0.0
        for t in 0..<k {
            let hess = hht[t * k + t]
            for i in 0..<m {
                var grad = -xht[i * k + t]
                for r in 0..<k { grad += hht[t * k + r] * target[i * k + r] }
                let pg = target[i * k + t] == 0 ? min(0, grad) : grad
                violation += abs(pg)
                if hess != 0 { target[i * k + t] = max(target[i * k + t] - grad / hess, 0) }
            }
        }
        return violation
    }

    private static func transposed(_ a: [Double], _ rows: Int, _ columns: Int) -> [Double] {
        var out = [Double](repeating: 0, count: a.count)
        for i in 0..<rows { for j in 0..<columns { out[j * rows + i] = a[i * columns + j] } }
        return out
    }

    // MARK: initialisation

    private static func initialFactors(_ x: [Double], _ n: Int, _ d: Int, _ k: Int,
                                       _ initialization: Initialization) throws -> ([Double], [Double]) {
        var mean = 0.0
        for v in x { mean += v }
        mean /= Double(x.count)
        func random(_ seed: UInt32) -> ([Double], [Double]) {
            var rng = NumPyRandomState(seed: seed)
            let avg = (mean / Double(k)).squareRoot()
            let h = (0..<(k * d)).map { _ in abs(avg * rng.standardNormal()) }
            let w = (0..<(n * k)).map { _ in abs(avg * rng.standardNormal()) }
            return (w, h)
        }
        func svdBased(fill: (inout [Double], inout [Double]) -> Void) throws -> ([Double], [Double]) {
            guard k <= min(n, d) else {
                throw FeaturizationError.invalidInput("NNDSVD needs components <= min(rows, columns)")
            }
            let (u, s, vt) = try truncatedSVD(x, n, d, k)       // u: n×k, vt: k×d
            var w = [Double](repeating: 0, count: n * k)
            var h = [Double](repeating: 0, count: k * d)
            for i in 0..<n { w[i * k] = s[0].squareRoot() * abs(u[i * k]) }
            for j in 0..<d { h[j] = s[0].squareRoot() * abs(vt[j]) }
            for c in 1..<max(1, k) {
                let xv = (0..<n).map { u[$0 * k + c] }, yv = (0..<d).map { vt[c * d + $0] }
                let xp = xv.map { max($0, 0) }, yp = yv.map { max($0, 0) }
                let xn = xv.map { abs(min($0, 0)) }, yn = yv.map { abs(min($0, 0)) }
                let xpN = norm(xp), ypN = norm(yp), xnN = norm(xn), ynN = norm(yn)
                let mp = xpN * ypN, mn = xnN * ynN
                let uu: [Double], vv: [Double], sigma: Double
                if mp > mn {
                    guard xpN > 0, ypN > 0 else { throw FeaturizationError.degenerateInitialisation }
                    uu = xp.map { $0 / xpN }; vv = yp.map { $0 / ypN }; sigma = mp
                } else {
                    guard xnN > 0, ynN > 0 else { throw FeaturizationError.degenerateInitialisation }
                    uu = xn.map { $0 / xnN }; vv = yn.map { $0 / ynN }; sigma = mn
                }
                let lbd = (s[c] * sigma).squareRoot()
                for i in 0..<n { w[i * k + c] = lbd * uu[i] }
                for j in 0..<d { h[c * d + j] = lbd * vv[j] }
            }
            for i in w.indices where w[i] < 1e-6 { w[i] = 0 }
            for i in h.indices where h[i] < 1e-6 { h[i] = 0 }
            fill(&w, &h)
            return (w, h)
        }
        switch initialization {
        case .custom(let w, let h):
            guard w.count == n * k, h.count == k * d else {
                throw FeaturizationError.invalidInput("custom W must be rows × components and H components × columns")
            }
            guard w.allSatisfy(\.isFinite), h.allSatisfy(\.isFinite) else { throw FeaturizationError.nonFinite }
            guard !w.contains(where: { $0 < 0 }), !h.contains(where: { $0 < 0 }) else { throw FeaturizationError.negativeValues }
            return (w, h)
        case .random(let seed):
            return random(seed)
        case .automatic(let seed):
            if k <= min(n, d) { return try initialFactors(x, n, d, k, .nndsvda) }
            return random(seed)
        case .nndsvd:
            return try svdBased { _, _ in }
        case .nndsvda:
            return try svdBased { w, h in
                for i in w.indices where w[i] == 0 { w[i] = mean }
                for i in h.indices where h[i] == 0 { h[i] = mean }
            }
        case .nndsvdar(let seed):
            return try svdBased { w, h in
                var rng = NumPyRandomState(seed: seed)
                for i in w.indices where w[i] == 0 { w[i] = abs(mean * rng.standardNormal() / 100) }
                for i in h.indices where h[i] == 0 { h[i] = abs(mean * rng.standardNormal() / 100) }
            }
        }
    }

    private static func norm(_ v: [Double]) -> Double {
        var s = 0.0
        for a in v { s += a * a }
        return s.squareRoot()
    }

    /// The leading `k` singular triplets of the `n × d` row-major matrix: `u` is `n × k`, `vt` is `k × d`, row-major.
    package static func truncatedSVD(_ x: [Double], _ n: Int, _ d: Int, _ k: Int) throws -> ([Double], [Double], [Double]) {
        let r = min(n, d)
        var a = [Double](repeating: 0, count: n * d)           // column-major copy
        for i in 0..<n { for j in 0..<d { a[j * n + i] = x[i * d + j] } }
        var s = [Double](repeating: 0, count: r)
        var uCol = [Double](repeating: 0, count: n * r)
        var vtCol = [Double](repeating: 0, count: r * d)
        let info = a.withUnsafeMutableBufferPointer { ap in
            uCol.withUnsafeMutableBufferPointer { up in
                vtCol.withUnsafeMutableBufferPointer { vp in
                    s.withUnsafeMutableBufferPointer { sp in
                        cubo_embedding_svd(Int32(n), Int32(d), ap.baseAddress!, sp.baseAddress!, up.baseAddress!, vp.baseAddress!)
                    }
                }
            }
        }
        guard info == 0 else { throw FeaturizationError.solverFailed(Int(info)) }
        var u = [Double](repeating: 0, count: n * k)
        var vt = [Double](repeating: 0, count: k * d)
        for c in 0..<k {
            for i in 0..<n { u[i * k + c] = uCol[c * n + i] }
            for j in 0..<d { vt[c * d + j] = vtCol[j * r + c] }
        }
        return (u, Array(s[0..<k]), vt)
    }
}
