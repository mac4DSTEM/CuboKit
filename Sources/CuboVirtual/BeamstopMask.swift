//
//  BeamstopMask.swift
//  The beamstop mask of a pattern: py4DSTEM's DataCube.get_beamstop_mask
//  (datacube/datacube.py 1013 at f050d207) on a pattern the caller passes (the mean pattern, or the maximum for
//  `use_max_dp`), in Double, with scipy's binary_fill_holes (4-connected) and the exact Euclidean distance transform.
//
//  Unlike a Float32 reference implementation that clips the threshold index to N − 1, this follows py4DSTEM in Double.
//  DEVIATIONS, in the refusing direction: a threshold whose index is N (py4DSTEM raises IndexError), a pattern with NaN,
//  and a cleaned mask with no false pixel (scipy's transform then measures to a virtual pixel outside the array).
//

import CuboCore
import Foundation

public enum BeamstopMask {
    /// `true` under the beamstop (and within `distanceEdge` pixels of it, and of the border with `includeEdges`), `false`
    /// in the pattern, row-major. `threshold` is the fraction of the sorted intensities below the cut; `sigma > 0` smooths
    /// first (`GaussianFilter`); `scaleRadial` multiplies by 1 + s · (distance from the centre in half-widths).
    public static func make(from pattern: DetectorImage, threshold: Double = 0.25, distanceEdge: Double = 2.0,
                            includeEdges: Bool = true, sigma: Double = 0, scaleRadial: Double? = nil) throws -> [Bool] {
        let qy = pattern.detectorRows, qx = pattern.detectorColumns, n = qy * qx
        var im = pattern.values
        guard !im.contains(where: \.isNaN) else { throw CuboVirtualError.invalidInput("the pattern holds NaN") }
        guard threshold.isFinite, distanceEdge.isFinite, sigma.isFinite, sigma >= 0 else {
            throw CuboVirtualError.invalidInput("threshold, distance and sigma must be finite, sigma not negative")
        }
        if sigma > 0 { im = try GaussianFilter(sigma: sigma).apply(im, rows: qy, columns: qx) }
        if let s = scaleRadial {
            guard s.isFinite else { throw CuboVirtualError.invalidInput("the radial scale must be finite") }
            let x = (0..<qy).map { Double($0) * 2.0 / Double(qy) }, y = (0..<qx).map { Double($0) * 2.0 / Double(qx) }
            let mx = NumPySum.mean(x), my = NumPySum.mean(y)
            for i in 0..<qy { for j in 0..<qx {
                let xa = x[i] - mx, ya = y[j] - my
                im[i * qx + j] *= 1.0 + (xa * xa + ya * ya).squareRoot() * s
            } }
        }
        let sorted = im.sorted()
        let index = Int((Swift.min(Swift.max(Double(n) * threshold, 0), Double(n))).rounded(.toNearestOrEven))
        guard index < n else { throw CuboVirtualError.invalidInput("a threshold of \(threshold) leaves no intensity to cut at") }
        let cut = sorted[index]
        var keep = im.map { $0 >= cut }
        var dark = keep.map { !$0 }
        fillHoles(&dark, qy, qx)
        keep = dark.map { !$0 }
        fillHoles(&keep, qy, qx)
        if includeEdges {
            for j in 0..<qx { keep[j] = false; keep[(qy - 1) * qx + j] = false }
            for i in 0..<qy { keep[i * qx] = false; keep[i * qx + qx - 1] = false }
        }
        guard keep.contains(false) else {
            throw CuboVirtualError.invalidInput("the cleaned mask has no pixel outside the pattern")
        }
        let d2 = squaredDistances(keep, qy, qx)
        return d2.map { $0.squareRoot() < distanceEdge }
    }

    /// scipy's binary_fill_holes with its default (4-connected) structure: a false pixel not 4-connected to the border
    /// becomes true.
    static func fillHoles(_ mask: inout [Bool], _ qy: Int, _ qx: Int) {
        var reached = [Bool](repeating: false, count: qy * qx)
        var stack: [Int] = []
        func push(_ i: Int, _ j: Int) {
            guard i >= 0, i < qy, j >= 0, j < qx else { return }
            let k = i * qx + j
            if !mask[k] && !reached[k] { reached[k] = true; stack.append(k) }
        }
        for j in 0..<qx { push(0, j); push(qy - 1, j) }
        for i in 0..<qy { push(i, 0); push(i, qx - 1) }
        while let k = stack.popLast() {
            let i = k / qx, j = k % qx
            push(i - 1, j); push(i + 1, j); push(i, j - 1); push(i, j + 1)
        }
        for k in 0..<(qy * qx) where !mask[k] && !reached[k] { mask[k] = true }
    }

    /// The exact squared Euclidean distance of every pixel to the nearest false pixel (0 at a false pixel), by the
    /// lower envelope of parabolas along the columns and then the rows (Felzenszwalb and Huttenlocher). Integers held
    /// exactly in Double.
    static func squaredDistances(_ keep: [Bool], _ qy: Int, _ qx: Int) -> [Double] {
        var g = [Double](repeating: .infinity, count: qy * qx)
        var f = [Double](repeating: 0, count: Swift.max(qy, qx))
        for j in 0..<qx {
            for i in 0..<qy { f[i] = keep[i * qx + j] ? .infinity : 0 }
            let d = envelope(f, qy)
            for i in 0..<qy { g[i * qx + j] = d[i] }
        }
        var out = [Double](repeating: 0, count: qy * qx)
        for i in 0..<qy {
            for j in 0..<qx { f[j] = g[i * qx + j] }
            let d = envelope(f, qx)
            for j in 0..<qx { out[i * qx + j] = d[j] }
        }
        return out
    }

    /// d[q] = min over p of (q − p)² + f[p], the infinite f skipped.
    static func envelope(_ f: [Double], _ n: Int) -> [Double] {
        var v = [Int](repeating: 0, count: n), z = [Double](repeating: 0, count: n + 1)
        var k = -1
        for q in 0..<n where f[q].isFinite {
            let fq = f[q] + Double(q * q)
            while k >= 0 {
                let p = v[k]
                let s = (fq - (f[p] + Double(p * p))) / Double(2 * q - 2 * p)
                if s <= z[k] { k -= 1 } else { break }
            }
            k += 1
            v[k] = q
            z[k] = k == 0 ? -.infinity : (fq - (f[v[k - 1]] + Double(v[k - 1] * v[k - 1]))) / Double(2 * q - 2 * v[k - 1])
            z[k + 1] = .infinity
        }
        var d = [Double](repeating: .infinity, count: n)
        guard k >= 0 else { return d }
        var j = 0
        for q in 0..<n {
            while z[j + 1] < Double(q) { j += 1 }
            let p = v[j]
            d[q] = Double((q - p) * (q - p)) + f[p]
        }
        return d
    }
}
