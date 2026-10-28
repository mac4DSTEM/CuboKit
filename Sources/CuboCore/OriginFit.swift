//
//  OriginFit.swift
//  py4DSTEM's `fit_origin` for a scan map: process/calibration/origin.py 76 and fit_2D
//  (process/fit/fit.py 38) at f050d207. The four models are linear in their parameters, so the fit is the least-squares
//  solution, here by Householder QR in Double on the design matrix with py4DSTEM's [0, 1] coordinates; py4DSTEM reaches
//  it with curve_fit within that solver's convergence. The robust option follows fit_2D's loop exactly.
//

import Foundation

public enum OriginFitModel: String, Sendable, Hashable, CaseIterable {
    case constant, plane, parabola
    /// py4DSTEM's `bezier_two`: the tensor product of quadratic Bernstein polynomials in each coordinate (9 terms).
    case bezierTwo
}

public enum OriginFitError: Error, Equatable, Sendable {
    case invalidMap(String)
    /// Fewer finite positions than the model has parameters, or a rank-deficient design.
    case underdetermined(OriginFitModel)
}

public enum OriginFit {
    /// The fitted surface at every position of a `rows × columns` map (row-major), over its finite values.
    /// `rankTolerance`: a design column whose remaining norm is at most this fraction of the largest column norm is left
    /// out (the rank-deficient projection, as NumPy's lstsq); 1e-12 by default.
    public static func fit(_ values: [Double], rows: Int, columns: Int, model: OriginFitModel, robust: Bool = false,
                           robustSteps: Int = 3, robustThreshold: Double = 2,
                           rankTolerance: Double = 1e-12) throws -> [Double] {
        guard rankTolerance >= 0, rankTolerance.isFinite else {
            throw OriginFitError.invalidMap("the rank tolerance must be finite and not negative (\(rankTolerance))")
        }
        guard rows > 0, columns > 0, values.count == rows * columns else {
            throw OriginFitError.invalidMap("\(values.count) values for \(rows) × \(columns)")
        }
        if model == .constant { return Array(repeating: NumPySum.mean(values), count: values.count) }
        let design = Self.design(rows: rows, columns: columns, model: model)
        let p = design.count / values.count
        var mask = values.map(\.isFinite)
        var coefficients = try solve(design, values, mask, p, model, rankTolerance)
        if robust {
            for _ in 0..<max(0, robustSteps) {
                var squared = [Double](repeating: 0, count: values.count)
                for k in values.indices {
                    var f = 0.0
                    for j in 0..<p { f += design[k * p + j] * coefficients[j] }
                    squared[k] = (f - values[k]) * (f - values[k])
                }
                let limit = NumPySum.mean(squared) * robustThreshold * robustThreshold
                for k in values.indices where squared[k] > limit { mask[k] = false }
                coefficients = try solve(design, values, mask, p, model, rankTolerance)
            }
        }
        return (0..<values.count).map { k in (0..<p).reduce(0.0) { $0 + design[k * p + $1] * coefficients[$1] } }
    }

    /// Row-major design matrix (positions × parameters) on NumPy's linspace(0, 1, n) coordinates, in fit.py's order.
    static func design(rows: Int, columns: Int, model: OriginFitModel) -> [Double] {
        func linspace(_ n: Int) -> [Double] {
            guard n > 1 else { return [0] }
            let step = 1.0 / Double(n - 1)
            var v = (0..<n).map { Double($0) * step }
            v[n - 1] = 1
            return v
        }
        let xs = linspace(rows), ys = linspace(columns)
        var a = [Double]()
        for r in 0..<rows { for c in 0..<columns {
            let x = xs[r], y = ys[c]
            switch model {
            case .plane: a += [x, y, 1]
            case .parabola: a += [1, x, y, x * x, y * y, x * y]
            case .bezierTwo:
                let bx = [(1 - x) * (1 - x), 2 * (1 - x) * x, x * x], by = [(1 - y) * (1 - y), 2 * (1 - y) * y, y * y]
                for i in 0..<3 { for j in 0..<3 { a.append(bx[i] * by[j]) } }
            case .constant: a.append(1)
            }
        } }
        return a
    }

    /// Least squares over the masked rows by Householder QR. A column whose remaining norm is negligible (≤ `tolerance` of
    /// the largest column norm, 1e-12 by default) is left out with a zero coefficient: the fitted surface is then the projection onto the
    /// design's column space, the surface NumPy's lstsq returns for a rank-deficient design (a 1 × N scan, say).
    static func solve(_ design: [Double], _ values: [Double], _ mask: [Bool], _ p: Int, _ model: OriginFitModel,
                      _ tolerance: Double = 1e-12) throws -> [Double] {
        let rowsUsed = values.indices.filter { mask[$0] && values[$0].isFinite }
        let m = rowsUsed.count
        guard m >= 1 else { throw OriginFitError.underdetermined(model) }
        var a = [Double](repeating: 0, count: m * p)   // column-major m × p
        var b = rowsUsed.map { values[$0] }
        for (i, k) in rowsUsed.enumerated() { for j in 0..<p { a[j * m + i] = design[k * p + j] } }
        var largest = 0.0
        for j in 0..<p { largest = max(largest, (0..<m).reduce(0.0) { $0 + a[j * m + $1] * a[j * m + $1] }.squareRoot()) }
        var kept: [(column: Int, row: Int)] = []
        var pos = 0
        for j in 0..<p where pos < m {
            var norm = 0.0
            for i in pos..<m { norm += a[j * m + i] * a[j * m + i] }
            norm = norm.squareRoot()
            guard norm > tolerance * largest else { continue }
            let alpha = a[j * m + pos] > 0 ? -norm : norm
            var v = [Double](repeating: 0, count: m - pos)
            for i in pos..<m { v[i - pos] = a[j * m + i] }
            v[0] -= alpha
            let vnorm2 = v.reduce(0) { $0 + $1 * $1 }
            if vnorm2 > 0 {
                for col in 0..<p {
                    var dot = 0.0
                    for i in pos..<m { dot += v[i - pos] * a[col * m + i] }
                    let f = 2 * dot / vnorm2
                    for i in pos..<m { a[col * m + i] -= f * v[i - pos] }
                }
                var dot = 0.0
                for i in pos..<m { dot += v[i - pos] * b[i] }
                let f = 2 * dot / vnorm2
                for i in pos..<m { b[i] -= f * v[i - pos] }
            }
            kept.append((j, pos))
            pos += 1
        }
        guard !kept.isEmpty else { throw OriginFitError.underdetermined(model) }
        var x = [Double](repeating: 0, count: p)
        for (idx, entry) in kept.enumerated().reversed() {
            var s = b[entry.row]
            for later in kept[(idx + 1)...] { s -= a[later.column * m + entry.row] * x[later.column] }
            x[entry.column] = s / a[entry.column * m + entry.row]
        }
        return x
    }
}

// MARK: - Trimmed (MAD) fit, mac4DSTEM's variant

/// The result of ``OriginFit/fitTrimmed(originRow:originColumn:rows:columns:model:trimSigma:madScale:rounds:)``.
public struct TrimmedOriginFit: Sendable, Equatable {
    /// The fitted surfaces of the two maps, evaluated at every position (the excluded ones included), row-major.
    public let fittedRow: [Double], fittedColumn: [Double]
    /// The positions the final fit used. `false` means the measurement was left out of the fit, not that the position has no
    /// fitted origin.
    public let kept: [Bool]
    /// RMS of the joint residual `hypot(row - fittedRow, column - fittedColumn)` over the kept positions, and over every position
    /// with finite measurements. Trimming removes the largest residuals, so the first falls by construction and must not be
    /// compared with a full-scan threshold.
    public let keptResidual: Double, fullScanResidual: Double
    /// The share of positions left out, `nil` for an empty map.
    public var excludedFraction: Double? {
        kept.isEmpty ? nil : 1 - Double(kept.filter { $0 }.count) / Double(kept.count)
    }
}

extension OriginFit {
    /// An iteratively trimmed fit of two origin maps, the variant mac4DSTEM's calibration uses where py4DSTEM's `fit_origin` is
    /// ordinary least squares over every position (or `fit_2D`'s root-mean-square rule, ``fit(_:rows:columns:model:robust:robustSteps:robustThreshold:rankTolerance:)``).
    /// A minority of positions whose origin measurement failed outright drags a plain fit; this one drops them.
    ///
    /// Each round fits both maps on the kept positions, measures the joint residual `hypot(dRow, dColumn)` at every position, takes
    /// its median `m` and `madScale` times the median absolute deviation `s` (1.4826, the normal-distribution consistency factor), and
    /// keeps the positions with residual `<= m + trimSigma * s`. The loop ends after `rounds` rounds (at least one), or earlier when `s`
    /// is zero or not finite (degenerate residuals; cutting at the median would drop half the scan for no reason), when no position would
    /// remain, or when the mask stops changing. The reported surfaces are a final fit on the reported mask. It works for a contamination
    /// below about half the positions that is scattered over the scan; a spatially clustered failure or one at or above the breakdown
    /// point of a median leaves it at the plain fit.
    ///
    /// A fit that cannot be made on the kept positions (fewer than the model has terms, or a scan one position wide) is the mean of
    /// the kept positions, not the projection onto the design that ``fit(_:rows:columns:model:robust:robustSteps:robustThreshold:rankTolerance:)``
    /// returns, so the positions just excluded cannot come back through an average. Non-finite measurements are never kept and do not
    /// enter the median or the deviation. Unlike the app, empty or mismatched maps, a non-positive `trimSigma` or `madScale` throw
    /// `OriginFitError.invalidMap`, and a map without a finite position throws `underdetermined`.
    public static func fitTrimmed(originRow: [Double], originColumn: [Double], rows: Int, columns: Int, model: OriginFitModel,
                                  trimSigma: Double = 3, madScale: Double = 1.4826, rounds: Int = 3) throws -> TrimmedOriginFit {
        guard rows > 0, columns > 0, rows <= Int.max / columns, originRow.count == rows * columns, originColumn.count == originRow.count else {
            throw OriginFitError.invalidMap("\(originRow.count) and \(originColumn.count) values for \(rows) × \(columns)")
        }
        guard trimSigma.isFinite, trimSigma > 0, madScale.isFinite, madScale > 0 else {
            throw OriginFitError.invalidMap("the trim width \(trimSigma) and the deviation scale \(madScale) must be finite and positive")
        }
        let count = originRow.count
        let usable = (0..<count).map { originRow[$0].isFinite && originColumn[$0].isFinite }
        guard usable.contains(true) else { throw OriginFitError.underdetermined(model) }
        var kept = usable
        var fittedRow = originRow, fittedColumn = originColumn

        func masked(_ z: [Double], _ included: [Bool]) throws -> [Double] {
            let used = z.indices.filter { included[$0] && z[$0].isFinite }
            let source = used.isEmpty ? Array(z.indices) : used
            func meanSurface() throws -> [Double] {
                [Double](repeating: try OriginFit.fit(source.map { z[$0] }, rows: source.count, columns: 1, model: .constant)[0], count: z.count)
            }
            let terms: Int
            switch model {
            case .constant: return try meanSurface()
            case .plane: terms = 3
            case .parabola: terms = 6
            case .bezierTwo: terms = 9
            }
            guard rows > 1, columns > 1, used.count >= terms else { return try meanSurface() }
            var values = [Double](repeating: .nan, count: z.count)
            for k in used { values[k] = z[k] }
            if let surface = try? OriginFit.fit(values, rows: rows, columns: columns, model: model) { return surface }
            return try meanSurface()
        }
        func residuals(_ a: [Double], _ b: [Double]) -> [Double] {
            (0..<count).map { usable[$0] ? hypot(originRow[$0] - a[$0], originColumn[$0] - b[$0]) : .nan }
        }
        func median(_ v: [Double]) -> Double {
            let s = v.sorted(), middle = s.count / 2
            return s.count.isMultiple(of: 2) ? (s[middle - 1] + s[middle]) / 2 : s[middle]
        }
        for _ in 0..<max(rounds, 1) {
            fittedRow = try masked(originRow, kept); fittedColumn = try masked(originColumn, kept)
            let residual = residuals(fittedRow, fittedColumn)
            let finite = residual.filter(\.isFinite)
            guard !finite.isEmpty else { break }       // every residual overflowed: nothing to take a median of
            let centre = median(finite)
            let deviation = madScale * median(finite.map { abs($0 - centre) })
            guard deviation.isFinite, deviation > 0 else { break }
            let cutoff = centre + trimSigma * deviation
            let next = residual.map { $0 <= cutoff }       // a non-finite residual compares false
            if !next.contains(true) { break }
            if next == kept { kept = next; break }
            kept = next
        }
        if kept.contains(true) {
            fittedRow = try masked(originRow, kept); fittedColumn = try masked(originColumn, kept)
        }
        let residual = residuals(fittedRow, fittedColumn)
        func rms(_ included: [Int]) -> Double {
            included.isEmpty ? .nan : (included.reduce(0) { $0 + residual[$1] * residual[$1] } / Double(included.count)).squareRoot()
        }
        return TrimmedOriginFit(fittedRow: fittedRow, fittedColumn: fittedColumn, kept: kept,
                                keptResidual: rms((0..<count).filter { kept[$0] }),
                                fullScanResidual: rms((0..<count).filter { usable[$0] }))
    }
}
