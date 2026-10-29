//
//  ClassificationNumerics.swift
//  The numerical pieces `BraggVectorClassification` takes from NumPy, SciPy and scikit-learn: `np.linalg.lstsq` (minimum-norm
//  solution by SVD, here one-sided Jacobi), `scipy.ndimage`'s binary morphology on a cross, `skimage.measure.label` with
//  8-connectivity, and `np.corrcoef`. All Double, CPU, package access. The coordinate-descent NMF is CuboEmbedding's
//  `NMF.coordinateDescent` and `gaussian_filter` is CuboCore's `SciPyKernels.gaussianFilter2D`.
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

    // MARK: scipy.ndimage

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
