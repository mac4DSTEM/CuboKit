//
//  SciPyKernels.swift
//  The few NumPy and SciPy primitives the polar and ellipse ports reproduce operation by operation, so that a port's rounding
//  follows the reference's: `np.arange` with a float step, Python's float modulo, `scipy.ndimage.gaussian_filter1d` (the
//  kernel, the symmetric correlation order, the `wrap`, `reflect` and `nearest` boundaries), `gaussian_filter` on a plane, and
//  `scipy.interpolate.interp1d` (linear, extrapolating).
//
//  Reference: NumPy 2.5.4 `arange`, `mod`; SciPy 1.18.1 `ndimage.gaussian_filter1d`/`correlate1d` (`NI_Correlate1D`, symmetric
//  weights: the centre term first, then the pairs from the outermost inwards), `interpolate.interp1d._call_linear`.
//

import Foundation

package enum SciPyKernels {
    /// `np.arange(start, stop, step)` for a positive float step: `ceil((stop - start) / step)` values, the first two exact
    /// (`start`, `start + step`) and then `start + i * delta` with `delta = (start + step) - start`, as NumPy fills them.
    ///
    /// ``arangeLength(_:_:_:limit:)`` is the count, `nil` for a request that is not finite or longer than ``maximumArangeLength``;
    /// `arange` returns no values for that (it never traps), so a caller that needs the reason checks the length first.
    package static func arange(_ start: Double, _ stop: Double, _ step: Double) -> [Double] {
        guard let length = arangeLength(start, stop, step), length > 0 else { return [] }
        if length == 1 { return [start] }
        let second = start + step
        let delta = second - start
        var out = [Double](repeating: 0, count: length)
        out[0] = start; out[1] = second
        if length > 2 { for i in 2..<length { out[i] = start + Double(i) * delta } }
        return out
    }

    /// The most values a caller of ``arange(_:_:_:)`` is offered.
    package static let maximumArangeLength = 10_000_000

    /// `ceil((stop - start) / step)` as an `Int` (negative counts become 0), or `nil` when that is NaN, infinite or above `limit`.
    package static func arangeLength(_ start: Double, _ stop: Double, _ step: Double, limit: Int = maximumArangeLength) -> Int? {
        let n = ((stop - start) / step).rounded(.up)
        guard n.isFinite, n <= Double(limit) else { return nil }
        return n > 0 ? Int(n) : 0
    }

    /// The half-width, in samples, of the Gaussian kernel of `sigma` (`int(truncate * sigma + 0.5)`), or `nil` when `sigma` is not
    /// finite or the half-width exceeds ``maximumKernelRadius``.
    package static func gaussianRadius(sigma: Double, truncate: Double = 4.0) -> Int? {
        let r = truncate * sigma + 0.5
        guard r.isFinite, r < Double(maximumKernelRadius) else { return nil }
        return r > 0 ? Int(r) : 0
    }

    /// The largest Gaussian kernel half-width accepted.
    package static let maximumKernelRadius = 1 << 20

    /// Python and `np.mod` on floats: the result takes the sign of the divisor.
    package static func mod(_ x: Double, _ m: Double) -> Double {
        var r = fmod(x, m)
        if r != 0 { if (r < 0) != (m < 0) { r += m } } else { r = copysign(0, m) }
        return r
    }

    /// How a one-dimensional filter reads beyond the ends: `wrap` is periodic, `reflect` is `d c b a | a b c d | d c b a`.
    /// SciPy's `mode`: `wrap` (a b c d | a b c d), `reflect` (d c b a | a b c d | d c b a, folded again for a radius beyond the
    /// extent) and `nearest` (a b c d | d d d).
    package enum Boundary: Sendable { case wrap, reflect, nearest }

    /// The weights of `gaussian_filter1d(..., order=0, truncate=4)`: radius `int(4 sigma + 0.5)`, `exp(-0.5/sigma^2 x^2)`
    /// normalised by NumPy's pairwise sum.
    package static func gaussianWeights(sigma: Double, truncate: Double = 4.0) -> [Double] {
        // callers validate; a bad sigma is the identity, not a trap
        gaussianWeights(sigma: sigma, radius: gaussianRadius(sigma: sigma, truncate: truncate) ?? 0)
    }

    /// The `2 radius + 1` weights `exp(-0.5/sigma^2 x^2)`, `x = -radius ... radius`, normalised by NumPy's pairwise sum.
    package static func gaussianWeights(sigma: Double, radius: Int) -> [Double] {
        let sigma2 = sigma * sigma
        var w = (-radius...radius).map { exp(-0.5 / sigma2 * Double($0 * $0)) }
        let s = NumPySum.pairwise(w)
        for i in w.indices { w[i] = w[i] / s }
        return w
    }

    private static func source(_ i: Int, _ n: Int, _ boundary: Boundary) -> Int {
        if i >= 0 && i < n { return i }
        switch boundary {
        case .wrap:
            let m = i % n
            return m < 0 ? m + n : m
        case .reflect:
            let period = 2 * n
            var m = i % period
            if m < 0 { m += period }
            return m < n ? m : period - 1 - m
        case .nearest:
            return i < 0 ? 0 : n - 1
        }
    }

    /// `scipy.ndimage.gaussian_filter1d(values, sigma, mode=...)` of a vector (`sigma > 0`).
    package static func gaussianFilter1D(_ values: [Double], sigma: Double, boundary: Boundary, truncate: Double = 4.0) -> [Double] {
        let w = gaussianWeights(sigma: sigma, truncate: truncate)
        return correlateSymmetric(values, weights: w, boundary: boundary)
    }

    /// The symmetric branch of `NI_Correlate1D`: `in[i] * w[0]`, then `(in[i - j] + in[i + j]) * w[j]` for `j = r ... 1`.
    package static func correlateSymmetric(_ values: [Double], weights w: [Double], boundary: Boundary) -> [Double] {
        let n = values.count, r = w.count / 2
        guard n > 0 else { return [] }
        var out = [Double](repeating: 0, count: n)
        for i in 0..<n {
            var acc = values[i] * w[r]
            if r > 0 {
                for j in stride(from: r, through: 1, by: -1) {
                    acc += (values[source(i - j, n, boundary)] + values[source(i + j, n, boundary)]) * w[r - j]
                }
            }
            out[i] = acc
        }
        return out
    }

    /// `scipy.ndimage.gaussian_filter(plane, sigma)` with mode `reflect`: axis 0, then axis 1.
    package static func gaussianFilter2D(_ values: [Double], rows: Int, columns: Int, sigma: Double) -> [Double] {
        guard sigma > 1e-15 else { return values }
        return correlateSymmetric2D(values, rows: rows, columns: columns, weights: gaussianWeights(sigma: sigma), boundary: .reflect)
    }

    /// The separable `correlate1d` of a `rows × columns` row-major plane with the symmetric `weights` along axis 0, then
    /// axis 1, each output in `NI_Correlate1D`'s order (``correlateSymmetric(_:weights:boundary:)``). The one 2-D Gaussian
    /// filter of the kit: ``gaussianFilter2D(_:rows:columns:sigma:)`` and CuboVirtual's `GaussianFilter.apply` run it.
    package static func correlateSymmetric2D(_ values: [Double], rows: Int, columns: Int, weights w: [Double],
                                             boundary: Boundary) -> [Double] {
        let r = w.count / 2, centre = w[r]
        var a = [Double](repeating: 0, count: values.count)
        for i in 0..<rows { for j in 0..<columns {
            var acc = values[i * columns + j] * centre
            for t in stride(from: r, through: 1, by: -1) {
                acc += (values[source(i - t, rows, boundary) * columns + j] + values[source(i + t, rows, boundary) * columns + j])
                    * w[r - t]
            }
            a[i * columns + j] = acc
        } }
        var b = [Double](repeating: 0, count: values.count)
        for i in 0..<rows { for j in 0..<columns {
            var acc = a[i * columns + j] * centre
            for t in stride(from: r, through: 1, by: -1) {
                acc += (a[i * columns + source(j - t, columns, boundary)] + a[i * columns + source(j + t, columns, boundary)])
                    * w[r - t]
            }
            b[i * columns + j] = acc
        } }
        return b
    }

    /// `interp1d(x, y, fill_value="extrapolate")(xNew)` for increasing `x` with at least two points: the segment is found as
    /// `searchsorted(x, v)` clipped to `1 ... n - 1`, `y = slope * (v - x_lo) + y_lo`, `slope = (y_hi - y_lo) / (x_hi - x_lo)`.
    package static func interpolateLinear(x: [Double], y: [Double], at xNew: Double) -> Double {
        let n = x.count
        var lo = 0, hi = n   // leftmost index with x[i] >= v (NumPy searchsorted, side left); a NaN sorts last
        if xNew.isNaN { hi = n } else {
            while lo < hi {
                let mid = (lo + hi) / 2
                if x[mid] < xNew { lo = mid + 1 } else { hi = mid }
            }
        }
        let idx = min(max(lo, 1), n - 1)
        let x0 = x[idx - 1], x1 = x[idx], y0 = y[idx - 1], y1 = y[idx]
        let slope = (y1 - y0) / (x1 - x0)
        return slope * (xNew - x0) + y0
    }
}
