import Foundation

/// The robust local fit and the basis helpers that mac4DSTEM's strain map is built from.
extension LatticeStrain {
    /// Robust intensity-once fit of [origin, g1, g2] to indexed peaks: ``fitIntensityOnce(_:)`` on the peaks, then one
    /// reweighted pass that drops peaks whose distance to the fitted lattice exceeds
    /// `max(0.35, median + 3 * max(1.4826 * MAD, 0.05))` (median and MAD of the distances, detector units) and refits the
    /// rest. The first fit is kept when fewer than `minimumPeaks` peaks would remain or when nothing is dropped.
    ///
    /// Weights are the intensities floored at 1e-6 and divided by the largest weight when it exceeds 1, before every
    /// fit, so a very bright peak cannot trip the representability guard (the ratios are all the fit depends on).
    /// Intensities of 1e10 and above are first divided by the largest intensity in Double (before the Float32 conversion
    /// that would overflow or lose the small weights), so any finite intensity is accepted; below 1e10 the weights are
    /// exactly those of earlier versions.
    /// The peak coordinates are relative to the diffraction origin; the returned origin is the fitted intercept in the
    /// same frame. Arithmetic follows the Float32 source (as ``fitIntensityOnce(_:)``): agreement with the Double
    /// formulas is about 1e-5 relative. Returns nil for fewer than `minimumPeaks` peaks or a singular design.
    /// Throws `invalidInput` for a non-finite coordinate, index or intensity, a coordinate or index of 1e10 or beyond
    /// (unrepresentable in the Float32 fit), or a negative intensity.
    public static func fitRobust(_ peaks: [LatticePeak], minimumPeaks: Int = 5) throws -> LatticeFit? {
        guard minimumPeaks >= 1,
              peaks.allSatisfy({ $0.vector.row.isFinite && $0.vector.column.isFinite && $0.intensity.isFinite && $0.intensity >= 0
                                 && $0.h.isFinite && $0.k.isFinite }),
              peaks.allSatisfy({ [$0.vector.row, $0.vector.column, $0.h, $0.k].allSatisfy { Float($0).isFinite && abs($0) < 1e10 } })
        else { throw LatticeStrainError.invalidInput }
        guard peaks.count >= minimumPeaks else { return nil }
        typealias Row = (h: Float, k: Float, x: Float, y: Float, wgt: Float)
        // Intensities of 1e10 and above: ratios in Double first (the largest becomes 1; a weight below 1e-6 is floored later, as
        // any weight is after the division by the largest).
        let brightest = peaks.reduce(0.0) { max($0, $1.intensity) }
        let divisor = brightest >= 1e10 ? brightest : 1.0
        let rows: [Row] = peaks.map {
            (Float($0.h), Float($0.k), Float($0.vector.row), Float($0.vector.column), max(Float($0.intensity / divisor), 1e-6))
        }
        func fitScaled(_ r: [Row]) -> (originX: Float, originY: Float, g1x: Float, g1y: Float, g2x: Float, g2y: Float, residual: Float)? {
            let top = r.reduce(Float(0)) { max($0, $1.wgt) }
            let scale = top > 1 ? 1 / Double(top) : 1
            return ConsensusNumerics.fitLattice(r.map { ($0.h, $0.k, $0.x, $0.y, max(Float(Double($0.wgt) * scale), 1e-6)) })
        }
        func lattice(_ f: (originX: Float, originY: Float, g1x: Float, g1y: Float, g2x: Float, g2y: Float, residual: Float)) -> LatticeFit {
            .init(origin: .init(row: Double(f.originX), column: Double(f.originY)),
                  basis: .init(g1: .init(row: Double(f.g1x), column: Double(f.g1y)), g2: .init(row: Double(f.g2x), column: Double(f.g2y))),
                  residual: Double(f.residual))
        }
        guard let initial = fitScaled(rows) else { return nil }
        let residuals = rows.map { p in
            hypot(p.x - initial.originX - p.h * initial.g1x - p.k * initial.g2x,
                  p.y - initial.originY - p.h * initial.g1y - p.k * initial.g2y)
        }
        guard let center = ConsensusNumerics.median(residuals) else { return lattice(initial) }
        let mad = ConsensusNumerics.median(residuals.map { abs($0 - center) }) ?? 0
        let threshold = max(0.35, center + 3 * max(1.4826 * mad, 0.05))
        let inliers = zip(rows, residuals).compactMap { $1 <= threshold ? $0 : nil }
        guard inliers.count >= minimumPeaks, inliers.count < rows.count else { return lattice(initial) }
        return lattice(fitScaled(inliers) ?? initial)
    }

    /// Indexing tolerance for a basis, detector units: `max(0.5, 0.18 * min(|g1|, |g2|))`. Double; the Float32 value in
    /// ``ConsensusBasis/indexingTolerance`` agrees to about 1e-6 relative.
    public static func indexingTolerance(basis: LatticeBasis) -> Double {
        max(0.5, 0.18 * min(hypot(basis.g1.row, basis.g1.column), hypot(basis.g2.row, basis.g2.column)))
    }

    /// Condition number sqrt(λmax/λmin) of the 2 × 2 matrix whose rows are g1 and g2, from its Gram matrix; nil when the
    /// basis is singular (λmin <= 1e-10) or not finite. Double, relative agreement with the Float32 source about 1e-6.
    public static func conditionNumber(basis: LatticeBasis) -> Double? {
        let (a, b, c, d) = (basis.g1.row, basis.g1.column, basis.g2.row, basis.g2.column)
        let trace = a * a + b * b + c * c + d * d
        let determinant = a * d - c * b
        let root = max(0, trace * trace - 4 * determinant * determinant).squareRoot()
        let maximum = (trace + root) / 2, minimum = (trace - root) / 2
        guard minimum > 1e-10, maximum.isFinite else { return nil }
        return (maximum / minimum).squareRoot()
    }

    /// Consensus score of a basis against centred observations (peak vectors relative to the diffraction origin): each is
    /// indexed by rounding (ties away from zero) the coefficients of `basis⁻¹ · v`; it supports the basis when
    /// |h|, |k| <= 8, (h, k) is not (0, 0) and its distance to the lattice node is at most `tolerance`. Returns the
    /// support count and the RMS distance of the supporting observations (`.infinity` without support, or for a basis
    /// with |determinant| <= 1e-12). Double; agrees with the Float32 source to about 1e-6 relative.
    public static func scoreBasis(basis: LatticeBasis, observations: [LatticeVector], tolerance: Double) -> (support: Int, residual: Double) {
        guard let inverse = inverse(basis) else { return (0, .infinity) }
        var support = 0, squared = 0.0
        for o in observations {
            let h = (inverse.a * o.row + inverse.b * o.column).rounded(), k = (inverse.c * o.row + inverse.d * o.column).rounded()
            guard abs(h) <= 8, abs(k) <= 8, h != 0 || k != 0 else { continue }
            let r = hypot(o.row - h * basis.g1.row - k * basis.g2.row, o.column - h * basis.g1.column - k * basis.g2.column)
            if r <= tolerance { support += 1; squared += r * r }
        }
        return (support, support > 0 ? (squared / Double(support)).squareRoot() : .infinity)
    }

    /// Indexes measured peaks by rounding: the peak vector minus `origin` is expressed in `basis` (`basis⁻¹ · v`), the
    /// coefficients rounded to the nearest integer (ties away from zero, unlike ``index(directions:basis:)``), and the
    /// peak kept when |h|, |k| <= `maximumOrder` and its distance to the lattice node h·g1 + k·g2 is at most `tolerance`
    /// (detector units, inclusive; see ``indexingTolerance(basis:)``). (h, k) = (0, 0) is kept. The returned peaks are in
    /// the centred frame (origin subtracted) with their indices set, in input order. Double; the Float32 source can
    /// round a coefficient within about 1e-6 of a half-integer differently.
    /// Throws `singularBasis` for |determinant| <= 1e-12 and `invalidInput` for non-finite input.
    public static func indexByRounding(peaks: [LatticePeak], origin: LatticeVector, basis: LatticeBasis,
                                       tolerance: Double, maximumOrder: Int = 8) throws -> [LatticePeak] {
        guard tolerance.isFinite, maximumOrder >= 0, origin.row.isFinite, origin.column.isFinite,
              [basis.g1.row, basis.g1.column, basis.g2.row, basis.g2.column].allSatisfy(\.isFinite),
              peaks.allSatisfy({ $0.vector.row.isFinite && $0.vector.column.isFinite && $0.intensity.isFinite })
        else { throw LatticeStrainError.invalidInput }
        guard let inverse = inverse(basis) else { throw LatticeStrainError.singularBasis }
        var out: [LatticePeak] = []
        for p in peaks {
            let vr = p.vector.row - origin.row, vc = p.vector.column - origin.column
            let h = (inverse.a * vr + inverse.b * vc).rounded(), k = (inverse.c * vr + inverse.d * vc).rounded()
            let r = hypot(vr - (h * basis.g1.row + k * basis.g2.row), vc - (h * basis.g1.column + k * basis.g2.column))
            if r <= tolerance, abs(h) <= Double(maximumOrder), abs(k) <= Double(maximumOrder) {
                out.append(.init(row: vr, column: vc, intensity: p.intensity, h: h, k: k))
            }
        }
        return out
    }

    /// Inverse of [[g1.row, g2.row], [g1.column, g2.column]], nil when |determinant| <= 1e-12.
    private static func inverse(_ b: LatticeBasis) -> (a: Double, b: Double, c: Double, d: Double)? {
        let (a, bb, c, d) = (b.g1.row, b.g2.row, b.g1.column, b.g2.column)
        let det = a * d - bb * c
        guard abs(det) > 1e-12 else { return nil }
        let inv = 1 / det
        return (d * inv, -bb * inv, -c * inv, a * inv)
    }
}
