import Foundation
import Testing
import CuboStrain

/// The robust local fit, the basis helpers, basis-rounding indexing and the frame rotation that mac4DSTEM's strain map is
/// built from, against hand transcriptions of the app's formulas on small synthetic lattices.
/// Tolerances (D12): the Double helpers (score, tolerance, condition, rounding, frame rotation) 1e-9 relative; the robust
/// fit, whose arithmetic is Float32 like `fitIntensityOnce`, 1e-6 relative to the vector length (limit 1e-5);
/// planted strain recovered through the Float32 fit to 1e-5 absolute.
@Suite struct StrainRobustTests {
    let g1 = LatticeVector(row: 10.2, column: 1.3)
    let g2 = LatticeVector(row: -1.1, column: 9.6)

    func close(_ a: Double, _ b: Double, _ rel: Double, scale: Double? = nil, _ loc: SourceLocation = #_sourceLocation) {
        let s = scale ?? max(abs(a), abs(b), 1e-300)
        #expect(abs(a - b) <= rel * s, "\(a) vs \(b)", sourceLocation: loc)
    }

    /// The nodes h·g1 + k·g2, |h|, |k| <= 2, (0, 0) excluded, with the row axis scaled by `rowScale`.
    func lattice(rowScale: Double = 1, intensity: (Int, Int) -> Double = { h, k in 1 + 0.3 * Double(abs(h) + abs(k)) }) -> [LatticePeak] {
        var out: [LatticePeak] = []
        for h in -2...2 { for k in -2...2 where h != 0 || k != 0 {
            out.append(.init(row: rowScale * (Double(h) * g1.row + Double(k) * g2.row), column: Double(h) * g1.column + Double(k) * g2.column,
                             intensity: intensity(h, k), h: Double(h), k: Double(k)))
        } }
        return out
    }

    /// Hand transcription of the app's robust fit in Double: weights floored at 1e-6 and divided by the largest above 1,
    /// weighted normal equations by Cramer's rule, one MAD pass.
    func transcribedFit(_ peaks: [LatticePeak], minimumPeaks: Int = 5) -> (g1: LatticeVector, g2: LatticeVector, origin: LatticeVector) {
        func fit(_ p: [LatticePeak]) -> (g1: LatticeVector, g2: LatticeVector, origin: LatticeVector) {
            let top = p.map { max($0.intensity, 1e-6) }.max()!, scale = top > 1 ? 1 / top : 1
            var a = [[Double]](repeating: [0, 0, 0], count: 3), bx = [0.0, 0, 0], by = [0.0, 0, 0]
            for q in p {
                let w = max(max(q.intensity, 1e-6) * scale, 1e-6), b = [1.0, q.h, q.k]
                for i in 0..<3 { bx[i] += w * b[i] * q.vector.row; by[i] += w * b[i] * q.vector.column
                    for j in 0..<3 { a[i][j] += w * b[i] * b[j] } }
            }
            func det(_ m: [[Double]]) -> Double {
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            }
            func solve(_ r: [Double]) -> [Double] {
                (0..<3).map { c in var m = a; for i in 0..<3 { m[i][c] = r[i] }; return det(m) / det(a) }
            }
            let x = solve(bx), y = solve(by)
            return (.init(row: x[1], column: y[1]), .init(row: x[2], column: y[2]), .init(row: x[0], column: y[0]))
        }
        func median(_ v: [Double]) -> Double { let s = v.sorted(), i = s.count / 2; return s.count % 2 == 1 ? s[i] : (s[i - 1] + s[i]) / 2 }
        let first = fit(peaks)
        let res = peaks.map { q in hypot(q.vector.row - first.origin.row - q.h * first.g1.row - q.k * first.g2.row,
                                         q.vector.column - first.origin.column - q.h * first.g1.column - q.k * first.g2.column) }
        let c = median(res), mad = median(res.map { abs($0 - c) })
        let threshold = max(0.35, c + 3 * max(1.4826 * mad, 0.05))
        let inliers = zip(peaks, res).filter { $0.1 <= threshold }.map { $0.0 }
        return inliers.count >= minimumPeaks && inliers.count < peaks.count ? fit(inliers) : first
    }

    @Test func robustFitDropsOutliersAndMatchesTranscription() throws {
        var peaks = lattice(rowScale: 0.99)
        // Two off-lattice peaks inside the indexing gate, indexed to real nodes.
        peaks.append(.init(row: 0.99 * (g1.row + g2.row) + 0.9, column: g1.column + g2.column - 0.5, intensity: 2, h: 1, k: 1))
        peaks.append(.init(row: -0.99 * g1.row - 0.7, column: -g1.column + 0.8, intensity: 1.5, h: -1, k: 0))
        let plain = try #require(try LatticeStrain.fitIntensityOnce(peaks))
        let robust = try #require(try LatticeStrain.fitRobust(peaks))
        // Truth: the planted, row-scaled lattice; the robust fit recovers it, the plain fit is pulled off.
        close(robust.basis.g1.row, 0.99 * g1.row, 1e-6); close(robust.basis.g1.column, g1.column, 1e-6)
        close(robust.basis.g2.row, 0.99 * g2.row, 1e-6); close(robust.basis.g2.column, g2.column, 1e-6)
        #expect(abs(plain.basis.g1.row - 0.99 * g1.row) > 1e-2)
        // Hand transcription: same inliers, same normal equations.
        let t = transcribedFit(peaks)
        for (a, b) in [(robust.basis.g1.row, t.g1.row), (robust.basis.g1.column, t.g1.column), (robust.basis.g2.row, t.g2.row),
                       (robust.basis.g2.column, t.g2.column), (robust.origin.row, t.origin.row)] {
            close(a, b, 1e-6, scale: 10)
        }
        #expect(robust.residual < 1e-4)
    }

    @Test func robustFitScalesBrightWeightsAndHandlesFewPeaks() throws {
        // Weights far above 1 (a hot peak of 1e9) give the same fit as the same ratios scaled down.
        let base = lattice { h, k in 1 + Double(abs(h)) }
        let scaled = base.map { var p = $0; p.intensity *= 1e9; return p }
        let a = try #require(try LatticeStrain.fitRobust(base)), b = try #require(try LatticeStrain.fitRobust(scaled))
        close(a.basis.g1.row, b.basis.g1.row, 1e-6, scale: 10)
        close(a.basis.g2.column, b.basis.g2.column, 1e-6, scale: 10)
        #expect(try LatticeStrain.fitRobust(Array(base.prefix(4)), minimumPeaks: 5) == nil)
        // 1e10 and above is rescaled in Double first (0.9.0; earlier versions refused it): see RobustLargeIntensityTests.
        var bad = base; bad[3].intensity = -1
        #expect(throws: LatticeStrainError.self) { try LatticeStrain.fitRobust(bad) }
    }

    @Test func basisHelpersMatchTranscription() {
        let basis = LatticeBasis(g1: g1, g2: g2)
        close(LatticeStrain.indexingTolerance(basis: basis), max(0.5, 0.18 * min(hypot(10.2, 1.3), hypot(-1.1, 9.6))), 1e-9)
        let small = LatticeBasis(g1: .init(row: 2, column: 0.5), g2: .init(row: -0.4, column: 3))
        #expect(LatticeStrain.indexingTolerance(basis: small) == 0.5)
        // Condition number by the 2x2 singular values (closed form), independent of the Gram-matrix formula.
        func cond(_ a: Double, _ b: Double, _ c: Double, _ d: Double) -> Double {
            let p = hypot(a + d, b - c), m = hypot(a - d, b + c)
            return (p + m) / abs(p - m)
        }
        close(LatticeStrain.conditionNumber(basis: basis)!, cond(10.2, 1.3, -1.1, 9.6), 1e-9)
        close(LatticeStrain.conditionNumber(basis: small)!, cond(2, 0.5, -0.4, 3), 1e-9)
        #expect(LatticeStrain.conditionNumber(basis: .init(g1: .init(row: 1, column: 2), g2: .init(row: 2, column: 4))) == nil)

        // scoreBasis: nodes plus a vector off by 1.2 (outside 0.9), one at the origin (skipped), one beyond order 8.
        let tol = 0.9
        let nodes = [(1, 0), (0, 1), (2, -1), (-1, -2), (3, 3)].map { h, k in
            LatticeVector(row: Double(h) * g1.row + Double(k) * g2.row + 0.3, column: Double(h) * g1.column + Double(k) * g2.column - 0.2)
        }
        let others = [LatticeVector(row: g1.row + 1.2, column: g1.column), LatticeVector(row: 0.1, column: -0.1),
                      LatticeVector(row: 9 * g1.row, column: 9 * g1.column)]
        let s = LatticeStrain.scoreBasis(basis: basis, observations: nodes + others, tolerance: tol)
        #expect(s.support == 5)
        close(s.residual, hypot(0.3, 0.2), 1e-9)
        let none = LatticeStrain.scoreBasis(basis: basis, observations: others, tolerance: tol)
        #expect(none.support == 0 && none.residual == .infinity)
        let singular = LatticeStrain.scoreBasis(basis: .init(g1: .init(row: 1, column: 2), g2: .init(row: 2, column: 4)), observations: nodes, tolerance: 5)
        #expect(singular.support == 0 && singular.residual == .infinity)
    }

    @Test func indexByRoundingMatchesTranscription() throws {
        let basis = LatticeBasis(g1: .init(row: 10, column: 0), g2: .init(row: 0, column: 10))
        let origin = LatticeVector(row: 100, column: 50)
        func at(_ r: Double, _ c: Double, i: Double = 1) -> LatticePeak { .init(row: 100 + r, column: 50 + c, intensity: i) }
        let peaks = [at(10.4, -9.7, i: 3),      // (1, -1), 0.5 off: kept
                     at(5, 0),                  // h = 0.5 rounds away from zero to (1, 0): residual 5 > tolerance 2, dropped
                     at(0.5, 0.5),              // (0, 0) kept at 0.7 off
                     at(-20.2, 30.1),           // (-2, 3)
                     at(95, 0),                 // order 9.5 → 10, out of order 8
                     at(14, 14)]                // (1, 1), 6.4 off: dropped
        let out = try LatticeStrain.indexByRounding(peaks: peaks, origin: origin, basis: basis, tolerance: 2, maximumOrder: 8)
        #expect(out.count == 3)
        #expect(out.map { [$0.h, $0.k] } == [[1, -1], [0, 0], [-2, 3]])
        close(out[0].vector.row, 10.4, 1e-9); close(out[0].vector.column, -9.7, 1e-9)   // centred frame
        #expect(out[0].intensity == 3)
        // Ties round away from zero: (15, 0) is (1.5, 0) → (2, 0), a distance of 5 from the node.
        let tie = try LatticeStrain.indexByRounding(peaks: [at(15, 0)], origin: origin, basis: basis, tolerance: 6)
        #expect(tie.map { [$0.h, $0.k] } == [[2, 0]])
        // Skewed basis against the app's formula h = round(β⁻¹ v).
        let skew = LatticeBasis(g1: g1, g2: g2)
        let det = g1.row * g2.column - g2.row * g1.column
        let p = LatticeVector(row: 2 * g1.row - g2.row + 0.2, column: 2 * g1.column - g2.column - 0.1)
        let hf = (g2.column * p.row - g2.row * p.column) / det, kf = (-g1.column * p.row + g1.row * p.column) / det
        let r = try LatticeStrain.indexByRounding(peaks: [.init(row: p.row, column: p.column, intensity: 1)], origin: .init(row: 0, column: 0),
                                                  basis: skew, tolerance: 1)
        #expect(r.map { [$0.h, $0.k] } == [[hf.rounded(), kf.rounded()]] && r[0].h == 2 && r[0].k == -1)
        #expect(throws: LatticeStrainError.self) {
            try LatticeStrain.indexByRounding(peaks: peaks, origin: origin, basis: .init(g1: .init(row: 1, column: 2), g2: .init(row: 2, column: 4)), tolerance: 1)
        }
    }

    // MARK: - the scan-level path

    /// A 3 × 3 scan: position 4 carries exx = 0.01 (row axis scaled by 0.99); every position has one off-lattice peak
    /// inside the indexing gate; the last carries a negative intensity the kit must refuse.
    func scan() -> [[StrainPeak]] {
        (0..<9).map { position in
            var list = lattice(rowScale: position == 4 ? 0.99 : 1).map { StrainPeak(row: $0.vector.row, column: $0.vector.column, intensity: $0.intensity) }
            list.append(StrainPeak(row: g1.row + 0.9, column: g1.column - 0.4, intensity: 1.2))
            if position == 8 { list[2].intensity = -1 }
            return list
        }
    }

    @Test func computeReproducesTheAppPath() throws {
        let basis = LatticeBasis(g1: g1, g2: g2)
        let peaks = scan()
        var o = StrainMapOptions(maximumPeakSpacing: 5)
        o.indexing = .basisRounding(maximumOrder: 8); o.fitMethod = .robustIntensityOnce; o.referenceMethod = .madRejection
        let result = try StrainMapping.compute(peaks: peaks, scanRows: 3, scanColumns: 3, detectorRows: 64, detectorColumns: 64,
                                               origin: .init(row: 0, column: 0), directions: StrainDirections.lattice(basis: basis), options: o)
        // The refused position is unfitted and counted; the other eight are fitted.
        #expect(result.refusedFitCount == 1)
        #expect(result.firstRefusal?.contains("mac4DSTEM") == false && result.firstRefusal != nil)
        #expect(result.valid == [true, true, true, true, true, true, true, true, false])
        // The same positions by direct kit calls.
        let tol = LatticeStrain.indexingTolerance(basis: basis)
        var fits: [LatticeBasis] = []
        for p in 0..<8 {
            let measured = peaks[p].map { LatticePeak(row: $0.row, column: $0.column, intensity: $0.intensity) }
            let indexed = try LatticeStrain.indexByRounding(peaks: measured, origin: .init(row: 0, column: 0), basis: basis, tolerance: tol)
            #expect(indexed.count == 25)
            let fit = try #require(try LatticeStrain.fitRobust(indexed))
            #expect(result.g1Row[p] == fit.basis.g1.row && result.g2Column[p] == fit.basis.g2.column && result.residual[p] == fit.residual)
            fits.append(fit.basis)
        }
        let reference = try LatticeStrain.referenceMADResult(fits)
        #expect(result.reference == reference.reference && result.referenceInlierCount == reference.inlierCount)
        for p in 0..<8 {
            let s = try LatticeStrain.strain(local: fits[p], reference: reference.reference)
            close(result.exx[p], s.exx, 1e-9, scale: 1e-2); close(result.exy[p], s.exy, 1e-9, scale: 1e-2)
            close(result.theta[p], s.theta, 1e-9, scale: 1e-2)
        }
        // Planted strain recovered through the Float32 fit (absolute 1e-5).
        for p in 0..<8 {
            #expect(abs(result.exx[p] - (p == 4 ? 0.01 : 0)) < 1e-5, "exx[\(p)] = \(result.exx[p])")
            #expect(abs(result.eyy[p]) < 1e-5 && abs(result.exy[p]) < 1e-5 && abs(result.theta[p]) < 1e-5)
        }
        // Basis diagnostics of the given-basis path: all peaks with radius > 0 scored at the indexing tolerance.
        let observations = peaks.flatMap { $0 }.map { LatticeVector(row: $0.row, column: $0.column) }
        let score = LatticeStrain.scoreBasis(basis: basis, observations: observations, tolerance: tol)
        close(result.basisSupportFraction, Double(score.support) / Double(observations.count), 1e-9)
        #expect(result.basisConditionNumber == LatticeStrain.conditionNumber(basis: basis))
        close(result.basisConditionNumber!, 1.0, 0.2)
    }

    @Test func defaultsKeepTheNearestDirectionPath() throws {
        let o = StrainMapOptions(maximumPeakSpacing: 5)
        #expect(o.indexing == .nearestDirection && o.indexingTolerance == nil && o.fitMethod == .py4DSTEM)
        let basis = LatticeBasis(g1: g1, g2: g2)
        var peaks = scan().map { Array($0.prefix(24)) }
        peaks[8][2].intensity = 1e12   // the py4DSTEM fit takes a bright peak (the refused one of scan() is the negative intensity)
        let r = try StrainMapping.compute(peaks: peaks, scanRows: 3, scanColumns: 3, detectorRows: 64, detectorColumns: 64,
                                          origin: .init(row: 0, column: 0), directions: StrainDirections.lattice(basis: basis), options: o)
        #expect(r.refusedFitCount == 0 && r.firstRefusal == nil && r.valid.allSatisfy { $0 })
        #expect(abs(r.exx[4] - 0.01) < 1e-9)
    }

    // MARK: - the frame rotation

    @Test func frameRotationMatchesTheAppAndStrainConvention() throws {
        let exx = [0.01, -0.02, .nan], eyy = [0.0, 0.03, 0.5], exy = [0.004, -0.001, 0.2], theta = [0.002, -0.003, 0.1]
        // Hand values at 45 degrees: c = s = 1/sqrt2, so exx' = (exx + eyy)/2 - exy, eyy' = (exx + eyy)/2 + exy, exy' = (exx - eyy)/2.
        let q = StrainFrame.rotate(exx: exx, eyy: eyy, exy: exy, theta: theta, rotation: .pi / 4, transposed: false)
        for i in 0..<2 {
            close(q.exx[i], (exx[i] + eyy[i]) / 2 - exy[i], 1e-9, scale: 0.03)
            close(q.eyy[i], (exx[i] + eyy[i]) / 2 + exy[i], 1e-9, scale: 0.03)
            close(q.exy[i], (exx[i] - eyy[i]) / 2, 1e-9, scale: 0.03)
            #expect(q.theta[i] == theta[i])
        }
        #expect(q.exx[2].isNaN && q.eyy[2].isNaN && q.exy[2].isNaN)
        // Transposed: exx and eyy swap before rotating, theta changes sign; at 0 rad that is the plain swap.
        let t = StrainFrame.rotate(exx: exx, eyy: eyy, exy: exy, theta: theta, rotation: 0, transposed: true)
        #expect(Array(t.exx.prefix(2)) == Array(eyy.prefix(2)) && Array(t.eyy.prefix(2)) == Array(exx.prefix(2)) && t.exy.prefix(2) == exy.prefix(2))
        #expect(t.theta == theta.map { -$0 })
        // Sign convention: rotating a coordinateRotation = 0 strain by `rotation` equals LatticeStrain.strain with
        // coordinateRotation = -rotation in degrees (theta unchanged).
        let reference = LatticeBasis(g1: g1, g2: g2)
        let local = LatticeBasis(g1: .init(row: 0.985 * g1.row + 0.1, column: g1.column + 0.05), g2: .init(row: g2.row - 0.04, column: 1.01 * g2.column))
        let raw = try LatticeStrain.strain(local: local, reference: reference)
        let angle = 0.37
        let rotated = StrainFrame.rotate(exx: [raw.exx], eyy: [raw.eyy], exy: [raw.exy], theta: [raw.theta], rotation: angle, transposed: false)
        let direct = try LatticeStrain.strain(local: local, reference: reference, coordinateRotation: -angle * 180 / .pi)
        close(rotated.exx[0], direct.exx, 1e-9, scale: 0.05); close(rotated.eyy[0], direct.eyy, 1e-9, scale: 0.05)
        close(rotated.exy[0], direct.exy, 1e-9, scale: 0.05); close(rotated.theta[0], direct.theta, 1e-9, scale: 0.05)
    }
}
