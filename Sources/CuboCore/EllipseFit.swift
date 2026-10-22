//
//  EllipseFit.swift
//  Fitting the elliptical distortion of a diffraction pattern (task 18): py4DSTEM's `process/calibration/ellipse.py` at f050d207
//  (`fit_ellipse_1D`, `ellipse_err`, `fit_ellipse_amorphous_ring`, `double_sided_gaussian`, `double_sided_gaussian_fiterr`,
//  `constrain_degenerate_ellipse`) and the two conversions of `process/utils/elliptical_coords.py` (`convert_ellipse_params`,
//  `convert_ellipse_params_r`), in Double. The fits run MINPACK's Levenberg-Marquardt (``MINPACKLeastSquares``) from the same
//  starting point as `scipy.optimize.leastsq`, so they stop where py4DSTEM's stop.
//
//  The ellipse is (row, column, a, b, theta): py4DSTEM's (x0, y0, a, b, theta) with x the ROW and y the COLUMN, a the semi-major
//  and b the semi-minor axis, theta the tilt of the semi-major axis from the +row axis towards the +column axis, in radians. These
//  are the numbers ``Calibration/ellipse`` stores (``EllipseParameters/calibration``), so a fit feeds the calibration without
//  conversion. The canonical form is `A (x - x0)^2 + B (x - x0)(y - y0) + C (y - y0)^2 = 1`.
//
//  Differences from py4DSTEM:
//    - `fit_ellipse_1D(fitradii=None)` raises a `TypeError` upstream (`np.minimum(ar.shape)` with one argument); the default
//      here is its evident intent, `(0, min(shape) / 2)`.
//    - A fit whose result is not finite throws ``EllipseFitError/notFinite``; py4DSTEM returns the NaNs. A result that did not
//      converge (MINPACK `info` 5 to 8, which `leastsq` warns about) is returned with `converged == false`.
//    - Pixels are `Double`; py4DSTEM works on the dtype it is given. For a Float32 Bragg-vector map the two Gaussian-filtered
//      planes in `constrainDegenerate` differ in the last bit (py4DSTEM rounds each pass to Float32), which can move the
//      position of a flat maximum, not the CoM that refines it beyond `fitRadius`.
//    - Typed errors where py4DSTEM asserts or raises; `isCancelled` is asked before every function evaluation.
//

import Foundation

/// A real plane the ellipse fits read: `rows x columns`, row-major.
public struct EllipseFitPlane: Sendable, Equatable {
    public let rows: Int, columns: Int
    public let values: [Double]
    public init(rows: Int, columns: Int, values: [Double]) throws {
        guard rows > 0, columns > 0, values.count == rows * columns else {
            throw EllipseFitError.invalidInput("\(values.count) values for a \(rows) x \(columns) plane")
        }
        self.rows = rows; self.columns = columns; self.values = values
    }
    public subscript(row: Int, column: Int) -> Double { values[row * columns + column] }
}

/// The five numbers of an ellipse: centre (row, column), semi-axes `a >= b`, and the tilt of `a` from the +row axis towards the
/// +column axis. An ellipse is the same after adding pi to theta.
public struct EllipseParameters: Sendable, Equatable {
    public var centerRow: Double, centerColumn: Double
    public var a: Double, b: Double, theta: Double
    public init(centerRow: Double, centerColumn: Double, a: Double, b: Double, theta: Double) {
        self.centerRow = centerRow; self.centerColumn = centerColumn; self.a = a; self.b = b; self.theta = theta
    }
    /// What ``Calibration/ellipse`` stores: a, b, theta.
    public var calibration: EllipseCalibration { EllipseCalibration(a: a, b: b, theta: theta) }
    /// The canonical form `(A, B, C)`.
    public var canonical: (A: Double, B: Double, C: Double) { EllipseFit.canonical(a: a, b: b, theta: theta) }
}

/// What a fit returns: the ellipse, the parameters the optimiser ended on, and how it ended.
public struct EllipseFitResult: Sendable, Equatable {
    public let ellipse: EllipseParameters
    /// `fit_ellipse_1D`: `(x0, y0, A, B, C)`. `fit_ellipse_amorphous_ring`: the 11 parameters
    /// `(I0, I1, sigma0, sigma1, sigma2, c_bkgd, x0, y0, A, B, C)` of the double-sided Gaussian.
    public let parameters: [Double]
    /// MINPACK's `info` (1 to 4 converged; 5 evaluations exhausted; 6 to 8 tolerances too small).
    public let info: Int
    public let evaluations: Int
    public var converged: Bool { (1...4).contains(info) }
}

public enum EllipseFitError: Error, Equatable, Sendable, LocalizedError {
    case invalidInput(String)
    /// Fewer pixels in the fitting region than parameters to fit.
    case tooFewPixels(found: Int, needed: Int)
    /// No radial bin lies strictly between the two fitting radii (the amorphous-ring guess of the ring radius).
    case noRadialBin(inner: Double, outer: Double)
    /// The optimiser produced NaN or an infinity.
    case notFinite

    public var errorDescription: String? {
        switch self {
        case .invalidInput(let d): "Invalid ellipse-fit input: \(d)."
        case .tooFewPixels(let f, let n): "The fitting region holds \(f) pixels, the fit needs at least \(n)."
        case .noRadialBin(let i, let o): "No radial bin of the pattern lies between the radii \(i) and \(o)."
        case .notFinite: "The ellipse fit did not end on finite numbers."
        }
    }
}

public enum EllipseFit {
    // MARK: conversions (process/utils/elliptical_coords.py)

    /// `convert_ellipse_params(A, B, C)`: the semi-axes (`a >= b`) and the tilt of `a` from the +row axis, from the canonical form.
    public static func semiAxes(A: Double, B: Double, C: Double) -> (a: Double, b: Double, theta: Double) {
        let val = ((A - C) * (A - C) + B * B).squareRoot()
        let b4a = B * B - 4 * A * C
        let theta: Double
        if B == 0 { theta = A < C ? 0 : Double.pi / 2 } else { theta = atan2(C - A - val, B) }
        let a = -(-2 * b4a * (A + C + val)).squareRoot() / b4a
        let b = -(-2 * b4a * (A + C - val)).squareRoot() / b4a
        // Python's max(a, b), min(a, b): the first argument unless the second is strictly larger (smaller)
        return (b > a ? b : a, b < a ? b : a, theta)
    }

    /// `convert_ellipse_params_r(a, b, theta)`: the canonical form `(A, B, C)`.
    public static func canonical(a: Double, b: Double, theta: Double) -> (A: Double, B: Double, C: Double) {
        let sin2 = sin(theta) * sin(theta), cos2 = cos(theta) * cos(theta)
        let a2 = a * a, b2 = b * b
        let A = sin2 / b2 + cos2 / a2
        let C = cos2 / b2 + sin2 / a2
        let B = 2 * (b2 - a2) * sin(theta) * cos(theta) / (a2 * b2)
        return (A, B, C)
    }

    // MARK: fit_ellipse_1D

    /// `ellipse_err(p, x, y, val)`: for points (`rows`, `columns`) of weight `values` and the canonical ellipse
    /// `p = (x0, y0, A, B, C)`, `(A x^2 + B x y + C y^2 - 1) * val` with `x`, `y` relative to the centre.
    public static func ellipseError(parameters p: [Double], rows x: [Double], columns y: [Double], values val: [Double]) -> [Double] {
        precondition(p.count == 5 && x.count == y.count && x.count == val.count)
        var out = [Double](repeating: 0, count: x.count)
        for i in x.indices {
            let dx = x[i] - p[0], dy = y[i] - p[1]
            out[i] = (p[2] * (dx * dx) + p[3] * dx * dy + p[4] * (dy * dy) - 1) * val[i]
        }
        return out
    }

    /// `fit_ellipse_1D(ar, center, fitradii, mask)`: fits a one-dimensional elliptical curve to the data inside the annulus
    /// `inner < r <= outer` about `center`, each pixel weighted by its value. `center` defaults to the middle of the plane,
    /// `fitRadii` to `(0, min(extents) / 2)`. `ignoring` marks pixels to leave out (`true` = ignore, as py4DSTEM's `mask`).
    public static func fit1D(_ plane: EllipseFitPlane, center: (row: Double, column: Double)? = nil,
                             fitRadii: (inner: Double, outer: Double)? = nil, ignoring mask: [Bool]? = nil,
                             isCancelled: (@Sendable () -> Bool)? = nil) throws -> EllipseFitResult {
        let c = center ?? (Double(plane.rows) / 2, Double(plane.columns) / 2)
        let radii = fitRadii ?? (0, Double(min(plane.rows, plane.columns)) / 2)
        try require(finite: [c.row, c.column, radii.inner, radii.outer])
        if let mask, mask.count != plane.values.count {
            throw EllipseFitError.invalidInput("a mask of \(mask.count) values for a \(plane.rows) x \(plane.columns) plane")
        }
        var xs = [Double](), ys = [Double](), vals = [Double]()
        for r in 0..<plane.rows { for col in 0..<plane.columns {
            let dr = Double(r) - c.row, dc = Double(col) - c.column
            let rr = (dr * dr + dc * dc).squareRoot()
            guard rr > radii.inner, rr <= radii.outer else { continue }
            if let mask, mask[r * plane.columns + col] { continue }
            xs.append(Double(r)); ys.append(Double(col)); vals.append(plane[r, col])
        } }
        guard xs.count >= 5 else { throw EllipseFitError.tooFewPixels(found: xs.count, needed: 5) }
        let guess = (2 / (radii.inner + radii.outer)) * (2 / (radii.inner + radii.outer))
        let fit = try MINPACKLeastSquares.leastSquares([c.row, c.column, guess, 0, guess], shouldStop: isCancelled.map { f in { f() } }) {
            ellipseError(parameters: $0, rows: xs, columns: ys, values: vals)
        }
        guard fit.x.allSatisfy(\.isFinite) else { throw EllipseFitError.notFinite }
        let s = semiAxes(A: fit.x[2], B: fit.x[3], C: fit.x[4])
        return EllipseFitResult(ellipse: EllipseParameters(centerRow: fit.x[0], centerColumn: fit.x[1], a: s.a, b: s.b, theta: s.theta),
                                parameters: fit.x, info: fit.info, evaluations: fit.evaluations)
    }

    // MARK: fit_ellipse_amorphous_ring

    /// `double_sided_gaussian(p, x, y)` at points (`rows`, `columns`): a Gaussian at the centre plus a Gaussian ring of two
    /// widths (inner and outer) at the mean of the semi-axes, on an elliptical radius, plus a constant.
    /// `p = (I0, I1, sigma0, sigma1, sigma2, c_bkgd, x0, y0, A, B, C)`.
    public static func doubleSidedGaussian(parameters p: [Double], rows x: [Double], columns y: [Double]) -> [Double] {
        precondition(p.count == 11 && x.count == y.count)
        let I0 = p[0], I1 = p[1], sigma0 = p[2], sigma1 = p[3], sigma2 = p[4], cBkgd = p[5], x0 = p[6], y0 = p[7]
        let s = semiAxes(A: p[8], B: p[9], C: p[10])
        let R = (s.a + s.b) / 2
        let R2 = R * R
        let A = p[8] * R2, B = p[9] * R2, C = p[10] * R2
        var out = [Double](repeating: 0, count: x.count)
        func heaviside(_ v: Double) -> Double { v.isNaN ? .nan : (v < 0 ? 0 : (v == 0 ? 0.5 : 1)) }
        for i in x.indices {
            let dx = x[i] - x0, dy = y[i] - y0
            let r2 = A * (dx * dx) + B * dx * dy + C * (dy * dy)
            let r = r2.squareRoot() - R
            let t0 = I0 * exp(-r2 / (2 * (sigma0 * sigma0)))
            let t1 = I1 * exp(-(r * r) / (2 * (sigma1 * sigma1))) * heaviside(-r)
            let t2 = I1 * exp(-(r * r) / (2 * (sigma2 * sigma2))) * heaviside(r)
            out[i] = t0 + t1 + t2 + cBkgd
        }
        return out
    }

    /// `double_sided_gaussian_fiterr(p, x, y, val)`: the model minus the data.
    public static func doubleSidedGaussianFitError(parameters p: [Double], rows x: [Double], columns y: [Double],
                                                   values val: [Double]) -> [Double] {
        let model = doubleSidedGaussian(parameters: p, rows: x, columns: y)
        return model.indices.map { model[$0] - val[$0] }
    }

    /// `radial_integral(ar, x0, y0, dr=1)`: the sum over the angle of the polar-elliptical transform with `a = b = 1`,
    /// `dphi = 2 degrees`, bins of width `dr` out to `rmax` (default: the farthest corner, truncated); `nil` where every angle of
    /// a radius falls outside the plane (a masked entry). Returns the bin centres and the sums.
    static func radialIntegral(_ plane: EllipseFitPlane, centerRow x0: Double, centerColumn y0: Double, dr: Double = 1)
        -> (centers: [Double], sums: [Double?]) {
        let nx = Double(plane.rows), ny = Double(plane.columns)
        let rmax = Int(max(max(hypot(x0, y0), hypot(x0, ny - y0)), max(hypot(nx - x0, y0), hypot(nx - x0, ny - y0))))
        let rBins = SciPyKernels.arange(0 + dr / 2.0, Double(rmax) + dr / 2.0, dr)
        let dphi = 2.0 * (Double.pi / 180)      // np.radians(2) = 2 * (pi / 180)
        let pBins = SciPyKernels.arange(-Double.pi + dphi / 2.0, Double.pi + dphi / 2.0, dphi)
        let theta = 0.0, ratio = 1.0 / 1.0      // (b / a) with a = b = 1
        var sums = [Double](repeating: 0, count: rBins.count)
        var any = [Bool](repeating: false, count: rBins.count)
        for p in pBins {
            for (k, r) in rBins.enumerated() {
                let xr = r * cos(p), yr = r * sin(p)
                let qx = x0 + xr * cos(theta) - yr * ratio * sin(theta)
                let qy = y0 + xr * sin(theta) + yr * ratio * cos(theta)
                guard qx > 0, qy > 0, qx < nx - 1, qy < ny - 1 else { continue }
                let xF = qx.rounded(.down), yF = qy.rounded(.down)
                let dx = qx - xF, dy = qy - yF
                let i = Int(xF), j = Int(yF)
                let w00 = (1 - dx) * (1 - dy), w10 = dx * (1 - dy), w01 = (1 - dx) * dy, w11 = dx * dy
                let v = plane[i, j] * w00 + plane[i + 1, j] * w10 + plane[i, j + 1] * w01 + plane[i + 1, j + 1] * w11
                // the masked-array sum keeps a cell only where the transformed mask (the weights, summed) is >= 0.99
                let maskValue = w00 + w10 + w01 + w11
                guard !(maskValue < 0.99) else { continue }
                sums[k] += v; any[k] = true
            }
        }
        return (rBins, (0..<rBins.count).map { any[$0] ? sums[$0] : nil })
    }

    /// `fit_ellipse_amorphous_ring(data, center, fitradii, p0, mask)`: fits the double-sided Gaussian to the pixels of the
    /// annulus `inner < r < outer` about `center` where `mask` is `true` (all of them without a mask), starting from py4DSTEM's
    /// guesses. `initial` holds the 11 starting values `(I0, I1, sigma0, sigma1, sigma2, c_bkgd, x0, y0, A, B, C)`, each `nil` to
    /// take the guess. Returns the ellipse and the 11 fitted parameters.
    public static func fitAmorphousRing(_ plane: EllipseFitPlane, center: (row: Double, column: Double),
                                        fitRadii: (inner: Double, outer: Double), initial: [Double?]? = nil,
                                        keeping mask: [Bool]? = nil,
                                        isCancelled: (@Sendable () -> Bool)? = nil) throws -> EllipseFitResult {
        try require(finite: [center.row, center.column, fitRadii.inner, fitRadii.outer])
        let keep = mask ?? [Bool](repeating: true, count: plane.values.count)
        guard keep.count == plane.values.count else {
            throw EllipseFitError.invalidInput("data and mask must have the same shape")
        }
        if let initial, initial.count != 11 { throw EllipseFitError.invalidInput("p0 must hold 11 values, not \(initial.count)") }
        let ri = fitRadii.inner, ro = fitRadii.outer
        var xs = [Double](), ys = [Double](), vals = [Double]()
        for r in 0..<plane.rows { for c in 0..<plane.columns {
            let rr = hypot(Double(r) - center.row, Double(c) - center.column)
            guard rr > ri, rr < ro, keep[r * plane.columns + c] else { continue }
            xs.append(Double(r)); ys.append(Double(c)); vals.append(plane[r, c])
        } }
        guard xs.count >= 11 else { throw EllipseFitError.tooFewPixels(found: xs.count, needed: 11) }
        // the guesses
        let I0 = plane.values.max() ?? 0
        var I1 = -Double.infinity
        for k in plane.values.indices { I1 = max(I1, plane.values[k] * (keep[k] ? 1 : 0)) }
        let sigma0 = ri / 2.0, sigma1 = (ro - ri) / 4.0, sigma2 = (ro - ri) / 4.0
        let cBkgd = plane.values.min() ?? 0
        let profile = radialIntegral(plane, centerRow: center.row, centerColumn: center.column, dr: 1)
        let candidates = profile.centers.indices.filter { profile.centers[$0] > ri && profile.centers[$0] < ro }
        guard !candidates.isEmpty else { throw EllipseFitError.noRadialBin(inner: ri, outer: ro) }
        var best = candidates[0]
        for k in candidates {   // np.argmax of the masked profile: masked entries are -inf, the first maximum wins
            let v = profile.sums[k] ?? -Double.infinity, b = profile.sums[best] ?? -Double.infinity
            if v > b { best = k }
        }
        let R = profile.centers[best]
        let abc = canonical(a: R, b: R, theta: 0)
        let guess: [Double] = [I0, I1, sigma0, sigma1, sigma2, cBkgd, center.row, center.column, abc.A, abc.B, abc.C]
        let start = (0..<11).map { initial?[$0] ?? guess[$0] }
        let fit = try MINPACKLeastSquares.leastSquares(start, shouldStop: isCancelled.map { f in { f() } }) {
            doubleSidedGaussianFitError(parameters: $0, rows: xs, columns: ys, values: vals)
        }
        guard fit.x.allSatisfy(\.isFinite) else { throw EllipseFitError.notFinite }
        let s = semiAxes(A: fit.x[8], B: fit.x[9], C: fit.x[10])
        return EllipseFitResult(ellipse: EllipseParameters(centerRow: fit.x[6], centerColumn: fit.x[7], a: s.a, b: s.b, theta: s.theta),
                                parameters: fit.x, info: fit.info, evaluations: fit.evaluations)
    }

    // MARK: constrain_degenerate_ellipse

    /// `constrain_degenerate_ellipse(data, p_ellipse, r_inner, r_outer, phi_known, fitrad)`: from the four brightest spots of a
    /// Bragg-vector map in the annulus `r_inner < r <= r_outer` (each refined to the centre of mass inside `fitRadius`), the
    /// semi-axes `(a, b)` of the ellipse with the given tilt `theta` that makes the angle between a pair of spots `phiKnown`
    /// (the smaller of the two complementary angles, radians).
    public static func constrainDegenerate(_ plane: EllipseFitPlane, ellipse p: EllipseParameters, innerRadius rInner: Double,
                                           outerRadius rOuter: Double, phiKnown: Double, fitRadius: Double = 6,
                                           isCancelled: (@Sendable () -> Bool)? = nil) throws -> (a: Double, b: Double) {
        try require(finite: [p.centerRow, p.centerColumn, p.a, p.b, p.theta, rInner, rOuter, phiKnown, fitRadius])
        guard p.a != 0, cos(phiKnown / 2.0) != 0 else { throw EllipseFitError.invalidInput("a degenerate ellipse or angle") }
        let x = p.centerRow, y = p.centerColumn
        let rows = plane.rows, columns = plane.columns
        var temp = [Double](repeating: 0, count: rows * columns)
        for r in 0..<rows { for c in 0..<columns {
            let rr = hypot(Double(r) - x, Double(c) - y)
            if rr > rInner && rr <= rOuter { temp[r * columns + c] = plane[r, c] }
        } }
        var xs = [Double](repeating: 0, count: 4), ys = xs
        for i in 0..<4 {
            if isCancelled?() == true { throw CancellationError() }
            let filtered = SciPyKernels.gaussianFilter2D(temp, rows: rows, columns: columns, sigma: 2)
            var arg = 0
            for k in filtered.indices where filtered[k] > filtered[arg] { arg = k }
            let xc = Double(arg / columns), yc = Double(arg % columns)
            var weighted = [Double](repeating: 0, count: rows * columns)
            var inside = [Bool](repeating: false, count: rows * columns)
            for r in 0..<rows { for c in 0..<columns {
                let rr = ((Double(r) - xc) * (Double(r) - xc) + (Double(c) - yc) * (Double(c) - yc)).squareRoot()
                if rr < fitRadius { inside[r * columns + c] = true; weighted[r * columns + c] = plane[r, c] }
            } }
            // get_CoM(data * mask): sums in NumPy's pairwise order
            let total = NumPySum.pairwise(weighted)
            let wr = (0..<weighted.count).map { Double($0 / columns) * weighted[$0] }
            let wc = (0..<weighted.count).map { Double($0 % columns) * weighted[$0] }
            xs[i] = NumPySum.pairwise(wr) / total
            ys[i] = NumPySum.pairwise(wc) / total
            for k in inside.indices where inside[k] { temp[k] = 0 }
        }
        for i in 0..<4 { xs[i] -= x; ys[i] -= y }
        let ct = cos(p.theta), st = sin(p.theta)
        var tx = [Double](repeating: 0, count: 4), ty = tx
        for i in 0..<4 { tx[i] = ct * xs[i] + st * ys[i]; ty[i] = -st * xs[i] + ct * ys[i] }
        var angleSum = 0.0, distSum = 0.0
        for i in 0..<4 {
            let ang = atan2(ty[i], tx[i])
            angleSum += min(abs(ang), Double.pi - abs(ang))
            distSum += hypot(tx[i], ty[i])
        }
        let angle = angleSum / 4, distance = distSum / 4
        let xFixed = distance * cos(angle), yFixed = distance * sin(angle)
        let t = xFixed / (p.a * cos(phiKnown / 2.0))
        let aConstrained = p.a * t
        let bConstrained = (yFixed * yFixed / (1 - (xFixed / aConstrained) * (xFixed / aConstrained))).squareRoot()
        return (aConstrained, bConstrained)
    }

    private static func require(finite values: [Double]) throws {
        guard values.allSatisfy(\.isFinite) else { throw EllipseFitError.invalidInput("the centre, radii and angles must be finite") }
    }
}
