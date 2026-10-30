//
//  ClassificationNumerics.swift
//  The numerical pieces `BraggVectorClassification` takes from NumPy, SciPy and scikit-learn: `np.linalg.lstsq` (minimum-norm
//  solution by CuboEmbedding's LAPACK SVD), `scipy.ndimage`'s binary morphology on a cross, `skimage.measure.label` with
//  8-connectivity, and `np.corrcoef`. All Double, CPU, package access. The coordinate-descent NMF is CuboEmbedding's
//  `NMF.coordinateDescent` and `gaussian_filter` is CuboCore's `SciPyKernels.gaussianFilter2D`.
//

import CuboCore
internal import CuboEmbedding
import Foundation

enum ClassificationNumerics {
    // MARK: least squares

    /// `np.linalg.lstsq(a, b, rcond=None)[0]` for the `m × n` matrix `a` and `m × p` matrix `b`, row-major: the minimum-norm
    /// solution `n × p`, through CuboEmbedding's LAPACK SVD (`dgesdd`). Singular values at or below `eps·max(m, n)` times the
    /// largest are treated as zero, as NumPy's `rcond=None` does. A non-finite value in `a` throws (NumPy raises "SVD did not converge");
    /// one in `b` carries through to the solution, as in NumPy.
    static func leastSquares(_ a: [Double], _ m: Int, _ n: Int, _ b: [Double], _ p: Int) throws -> [Double] {
        var x = [Double](repeating: 0, count: n * p)
        guard m > 0, n > 0, p > 0 else { return x }
        guard a.allSatisfy(\.isFinite) else {
            throw CuboBraggError.invalidOption("least squares: the matrix holds a non-finite value")
        }
        let r = min(m, n)
        let u: [Double], sigma: [Double], vt: [Double]
        do {
            (u, sigma, vt) = try NMF.truncatedSVD(a, m, n, r)     // u: m × r, vt: r × n, sigma descending
        } catch {
            throw CuboBraggError.invalidOption("least squares: the SVD failed (\(error))")
        }
        let cutoff = Double.ulpOfOne * Double(max(m, n)) * (sigma.first ?? 0)
        for c in 0..<r where sigma[c] > cutoff {
            for col in 0..<p {
                var dot = 0.0
                for i in 0..<m { dot += u[i * r + c] * b[i * p + col] }
                let coefficient = dot / sigma[c]
                for j in 0..<n { x[j * p + col] += vt[c * n + j] * coefficient }
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
