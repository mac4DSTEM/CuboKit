//
//  OriginFit.swift
//  py4DSTEM's `fit_origin` for a scan map (registration slice 3, 3.2): process/calibration/origin.py 76 and fit_2D
//  (process/fit/fit.py 38) at f050d207. The four models are linear in their parameters, so the fit is the least-squares
//  solution, here by Householder QR in Double on the design matrix with py4DSTEM's [0, 1] coordinates; py4DSTEM reaches
//  it with curve_fit within that solver's convergence. The robust option follows fit_2D's loop exactly.
//

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
    public static func fit(_ values: [Double], rows: Int, columns: Int, model: OriginFitModel, robust: Bool = false,
                           robustSteps: Int = 3, robustThreshold: Double = 2) throws -> [Double] {
        guard rows > 0, columns > 0, values.count == rows * columns else {
            throw OriginFitError.invalidMap("\(values.count) values for \(rows) × \(columns)")
        }
        if model == .constant { return Array(repeating: NumPySum.mean(values), count: values.count) }
        let design = Self.design(rows: rows, columns: columns, model: model)
        let p = design.count / values.count
        var mask = values.map(\.isFinite)
        var coefficients = try solve(design, values, mask, p, model)
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
                coefficients = try solve(design, values, mask, p, model)
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

    /// Least squares over the masked rows by Householder QR. A column whose remaining norm is negligible (≤ 1e-12 of the
    /// largest column norm) is left out with a zero coefficient: the fitted surface is then the projection onto the
    /// design's column space, the surface NumPy's lstsq returns for a rank-deficient design (a 1 × N scan, say).
    static func solve(_ design: [Double], _ values: [Double], _ mask: [Bool], _ p: Int, _ model: OriginFitModel) throws -> [Double] {
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
            guard norm > 1e-12 * largest else { continue }
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
