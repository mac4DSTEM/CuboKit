//
//  ClassificationNumerics.swift
//  The numerical pieces `BraggVectorClassification` takes from NumPy, SciPy and scikit-learn: `np.linalg.lstsq` (minimum-norm
//  solution by SVD, here one-sided Jacobi), scikit-learn's coordinate-descent NMF from a custom start, `scipy.ndimage`
//  `gaussian_filter` and the binary morphology on a cross, `skimage.measure.label` with 8-connectivity, and `np.corrcoef`.
//  All Double, CPU, package access.
//

import CuboCore
import Foundation

enum ClassificationNumerics {
    // MARK: least squares

    /// `np.linalg.lstsq(a, b, rcond=None)[0]` for the `m × n` matrix `a` and `m × p` matrix `b`, row-major: the minimum-norm
    /// solution `n × p`. Singular values at or below `eps·max(m, n)` times the largest are treated as zero.
    static func leastSquares(_ a: [Double], _ m: Int, _ n: Int, _ b: [Double], _ p: Int) -> [Double] {
        var x = [Double](repeating: 0, count: n * p)
        guard m > 0, n > 0, p > 0 else { return x }
        // thin SVD by one-sided Jacobi: columns of `left` are σ·u, columns of `right` are v, for the narrow side
        let tall = m >= n
        let rows = tall ? m : n, cols = tall ? n : m
        var work = [Double](repeating: 0, count: rows * cols)          // column-major copy: work[c * rows + r]
        for i in 0..<m { for j in 0..<n {
            if tall { work[j * rows + i] = a[i * n + j] } else { work[i * rows + j] = a[i * n + j] }
        } }
        var v = [Double](repeating: 0, count: cols * cols)
        for c in 0..<cols { v[c * cols + c] = 1 }
        let epsilon = Double.ulpOfOne
        for _ in 0..<60 {
            var rotated = false
            for pcol in 0..<(cols - 1) where cols > 1 {
                for q in (pcol + 1)..<cols {
                    var alpha = 0.0, beta = 0.0, gamma = 0.0
                    for r in 0..<rows {
                        let x1 = work[pcol * rows + r], x2 = work[q * rows + r]
                        alpha += x1 * x1; beta += x2 * x2; gamma += x1 * x2
                    }
                    if gamma == 0 || abs(gamma) <= epsilon * (alpha * beta).squareRoot() { continue }
                    rotated = true
                    let zeta = (beta - alpha) / (2 * gamma)
                    let t = (zeta >= 0 ? 1.0 : -1.0) / (abs(zeta) + (1 + zeta * zeta).squareRoot())
                    let c = 1 / (1 + t * t).squareRoot(), s = c * t
                    for r in 0..<rows {
                        let x1 = work[pcol * rows + r], x2 = work[q * rows + r]
                        work[pcol * rows + r] = c * x1 - s * x2
                        work[q * rows + r] = s * x1 + c * x2
                    }
                    for r in 0..<cols {
                        let v1 = v[pcol * cols + r], v2 = v[q * cols + r]
                        v[pcol * cols + r] = c * v1 - s * v2
                        v[q * cols + r] = s * v1 + c * v2
                    }
                }
            }
            if !rotated { break }
        }
        var sigma = [Double](repeating: 0, count: cols)
        for c in 0..<cols {
            var s = 0.0
            for r in 0..<rows { s += work[c * rows + r] * work[c * rows + r] }
            sigma[c] = s.squareRoot()
        }
        let cutoff = epsilon * Double(max(m, n)) * (sigma.max() ?? 0)
        for c in 0..<cols where sigma[c] > cutoff {
            // tall: A = (work·Vᵀ) with work = U·Σ, so x += v_c · (work_c · b) / σ²
            // wide: Aᵀ = work·Vᵀ, so A = V·workᵀ with V's columns the left vectors: x += (work_c/σ) (v_c · b)/σ
            for col in 0..<p {
                if tall {
                    var dot = 0.0
                    for r in 0..<rows { dot += work[c * rows + r] * b[r * p + col] }
                    let coefficient = dot / (sigma[c] * sigma[c])
                    for j in 0..<n { x[j * p + col] += v[c * cols + j] * coefficient }
                } else {
                    var dot = 0.0
                    for r in 0..<m { dot += v[c * cols + r] * b[r * p + col] }
                    let coefficient = dot / (sigma[c] * sigma[c])
                    for j in 0..<n { x[j * p + col] += work[c * rows + j] * coefficient }
                }
            }
        }
        return x
    }

    // MARK: NMF

    /// `NMF(n_components, init="custom", max_iter=maxIterations).fit_transform(X, W=w, H=h)` (solver "cd", tol 1e-4): returns the
    /// factors. `x` is `n × d`, `w` `n × k`, `h` `k × d`.
    static func coordinateDescentNMF(_ x: [Double], _ n: Int, _ d: Int, _ k: Int, _ w0: [Double], _ h0: [Double],
                                     maxIterations: Int, tolerance: Double = 1e-4,
                                     isCancelled: (@Sendable () -> Bool)?) throws -> (w: [Double], h: [Double]) {
        var w = w0
        var ht = [Double](repeating: 0, count: d * k)
        for c in 0..<k { for j in 0..<d { ht[j * k + c] = h0[c * d + j] } }
        var xt = [Double](repeating: 0, count: d * n)
        for i in 0..<n { for j in 0..<d { xt[j * n + i] = x[i * d + j] } }
        var violationInit = 0.0
        for sweep in 1...max(1, maxIterations) {
            if isCancelled?() == true { throw CancellationError() }
            var violation = update(&w, n, k, other: ht, otherRows: d, data: x)
            violation += update(&ht, d, k, other: w, otherRows: n, data: xt)
            if sweep == 1 { violationInit = violation }
            if violationInit == 0 || violation / violationInit <= tolerance { break }
        }
        var h = [Double](repeating: 0, count: k * d)
        for c in 0..<k { for j in 0..<d { h[c * d + j] = ht[j * k + c] } }
        return (w, h)
    }

    private static func update(_ target: inout [Double], _ m: Int, _ k: Int, other: [Double], otherRows p: Int,
                               data: [Double]) -> Double {
        var hht = [Double](repeating: 0, count: k * k)
        for r in 0..<p { for a in 0..<k {
            let value = other[r * k + a]
            if value == 0 { continue }
            for b in 0..<k { hht[a * k + b] += value * other[r * k + b] }
        } }
        var xht = [Double](repeating: 0, count: m * k)
        for i in 0..<m { for j in 0..<p {
            let value = data[i * p + j]
            if value == 0 { continue }
            for c in 0..<k { xht[i * k + c] += value * other[j * k + c] }
        } }
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

    // MARK: scipy.ndimage

    /// `scipy.ndimage.gaussian_filter(image, sigma)` (mode "reflect", truncate 4.0) on a `rows × columns` image: a 1-D
    /// correlation along axis 0 then axis 1, the symmetric kernel summed outwards from the centre as SciPy's C loop does.
    static func gaussianFilter(_ image: [Double], _ rows: Int, _ columns: Int, sigma: Double) -> [Double] {
        guard sigma > 1e-15 else { return image }
        let radius = Int(4.0 * sigma + 0.5)
        let sigma2 = sigma * sigma
        var kernel = (-radius...radius).map { exp(-0.5 / sigma2 * Double($0 * $0)) }
        let total = NumPySum.pairwise(kernel)
        kernel = kernel.map { $0 / total }
        func reflect(_ i: Int, _ n: Int) -> Int {
            if n == 1 { return 0 }
            var j = i % (2 * n)
            if j < 0 { j += 2 * n }
            return j >= n ? 2 * n - 1 - j : j
        }
        func filter(_ line: [Double]) -> [Double] {
            let n = line.count
            return (0..<n).map { i in
                var tmp = line[i] * kernel[radius]
                if radius >= 1 {
                    for ll in 1...radius { tmp += (line[reflect(i - ll, n)] + line[reflect(i + ll, n)]) * kernel[radius + ll] }
                }
                return tmp
            }
        }
        var out = image
        for c in 0..<columns {                                  // axis 0
            let filtered = filter((0..<rows).map { out[$0 * columns + c] })
            for r in 0..<rows { out[r * columns + c] = filtered[r] }
        }
        for r in 0..<rows {                                     // axis 1
            let filtered = filter(Array(out[(r * columns)..<((r + 1) * columns)]))
            for c in 0..<columns { out[r * columns + c] = filtered[c] }
        }
        return out
    }

    /// One step of the cross-structured dilation (`erode == false`) or erosion of a binary image, the outside being false.
    private static func step(_ mask: [Bool], _ rows: Int, _ columns: Int, erode: Bool) -> [Bool] {
        var out = mask
        for r in 0..<rows { for c in 0..<columns {
            var values = [mask[r * columns + c]]
            for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let rr = r + dr, cc = c + dc
                values.append((0..<rows).contains(rr) && (0..<columns).contains(cc) ? mask[rr * columns + cc] : false)
            }
            out[r * columns + c] = erode ? !values.contains(false) : values.contains(true)
        } }
        return out
    }

    /// `binary_dilation` / `binary_erosion` with the default cross structure and `border_value=0`; `iterations < 1` repeats
    /// until the image no longer changes (SciPy's rule).
    static func morph(_ mask: [Bool], _ rows: Int, _ columns: Int, erode: Bool, iterations: Int) -> [Bool] {
        var current = mask
        if iterations < 1 {
            while true {
                let next = step(current, rows, columns, erode: erode)
                if next == current { return current }
                current = next
            }
        }
        for _ in 0..<iterations { current = step(current, rows, columns, erode: erode) }
        return current
    }

    /// `skimage.measure.label(mask, background=0, connectivity=2)`: 8-connected components numbered 1, 2, ... in the raster order
    /// of their first pixel; the count is returned with the labels (0 is background).
    static func label(_ mask: [Bool], _ rows: Int, _ columns: Int) -> (labels: [Int], count: Int) {
        var labels = [Int](repeating: 0, count: mask.count)
        var count = 0
        for start in 0..<mask.count where mask[start] && labels[start] == 0 {
            count += 1
            labels[start] = count
            var stack = [start]
            while let pixel = stack.popLast() {
                let r = pixel / columns, c = pixel % columns
                for dr in -1...1 { for dc in -1...1 where dr != 0 || dc != 0 {
                    let rr = r + dr, cc = c + dc
                    guard (0..<rows).contains(rr), (0..<columns).contains(cc) else { continue }
                    let q = rr * columns + cc
                    if mask[q], labels[q] == 0 { labels[q] = count; stack.append(q) }
                } }
            }
        }
        return (labels, count)
    }

    // MARK: statistics

    /// `np.corrcoef` of `variables` variables (the rows of `values`, `variables × observations`): the `variables × variables`
    /// correlation, NaN where a variable is constant, clipped to `[-1, 1]`.
    static func correlation(_ values: [Double], variables: Int, observations: Int) -> [Double] {
        var centred = values
        for v in 0..<variables {
            var mean = 0.0
            for o in 0..<observations { mean += values[v * observations + o] }
            mean /= Double(observations)
            for o in 0..<observations { centred[v * observations + o] -= mean }
        }
        let scale = 1.0 / Double(observations - 1)
        var c = [Double](repeating: 0, count: variables * variables)
        for a in 0..<variables { for b in 0..<variables {
            var s = 0.0
            for o in 0..<observations { s += centred[a * observations + o] * centred[b * observations + o] }
            c[a * variables + b] = s * scale
        } }
        let std = (0..<variables).map { c[$0 * variables + $0].squareRoot() }
        for a in 0..<variables { for b in 0..<variables {
            let value = (c[a * variables + b] / std[a]) / std[b]
            c[a * variables + b] = value.isNaN ? value : min(1, max(-1, value))
        } }
        return c
    }
}
