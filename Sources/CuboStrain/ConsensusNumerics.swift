import Foundation

// Numerical compatibility routines from the pinned GPL source; see tools/strain/pins.py.
enum ConsensusNumerics {
    struct BasisEstimate {
        let g1: (x: Float, y: Float)
        let g2: (x: Float, y: Float)
        let observationCount: Int
        let supportCount: Int
        let residualPixels: Float
        let conditionNumber: Float
        let indexingTolerancePixels: Float
    }

    // MARK: - Lattice-vector selection

    struct VectorObservation {
        let x: Float
        let y: Float
        let radius: Float
    }

    struct VectorCluster {
        var sumX: Float
        var sumY: Float
        var count: Int

        var x: Float { sumX / Float(count) }
        var y: Float { sumY / Float(count) }
        var radius: Float { hypot(x, y) }
    }

    struct VectorCell: Hashable {
        let x: Int
        let y: Int
    }

    /// Compatibility entry point for callers that only need the vectors. The
    /// full solver consumes `estimateLatticeBasis` so it can retain confidence
    /// and residual diagnostics.
    static func chooseLatticeVectors(bragg: ConsensusPeakGrid,
                                                 x0: Float, y0: Float,
                                                 minRadius: Float = 2,
                                                 cancellation: ConsensusCancellation? = nil)
        -> (g1: (x: Float, y: Float), g2: (x: Float, y: Float))? {
        guard let estimate = estimateLatticeBasis(
            bragg: bragg, x0: x0, y0: y0,
            minRadius: minRadius, cancellation: cancellation
        ) else { return nil }
        return (estimate.g1, estimate.g2)
    }

    static func vectorObservations(
        bragg: ConsensusPeakGrid, x0: Float, y0: Float, minRadius: Float,
        cancellation: ConsensusCancellation?
    ) -> [VectorObservation] {
        var observations: [VectorObservation] = []
        for positionPeaks in bragg.peaks {
            if cancellation?.isCancelled == true { return [] }
            for p in positionPeaks {
                let vx = p.x - x0, vy = p.y - y0
                let radius = hypot(vx, vy)
                if radius > minRadius, radius.isFinite {
                    observations.append(VectorObservation(x: vx, y: vy, radius: radius))
                }
            }
        }
        return observations
    }

    /// Cluster repeated peak vectors, test bounded candidate pairs, and choose
    /// the well-conditioned basis explaining the largest observation consensus.
    static func estimateLatticeBasis(
        bragg: ConsensusPeakGrid, x0: Float, y0: Float, minRadius: Float = 2,
        cancellation: ConsensusCancellation?
    ) -> BasisEstimate? {
        let observations = vectorObservations(
            bragg: bragg, x0: x0, y0: y0,
            minRadius: minRadius, cancellation: cancellation
        )
        guard cancellation?.isCancelled != true, observations.count >= 2 else { return nil }

        // A typical first-shell scale from each position's nearest non-central
        // peak makes clustering insensitive to a handful of globally short
        // false detections.
        var nearestRadii: [Float] = []
        for peaks in bragg.peaks {
            var nearest = Float.greatestFiniteMagnitude
            for peak in peaks {
                let radius = hypot(peak.x - x0, peak.y - y0)
                if radius > minRadius { nearest = min(nearest, radius) }
            }
            // Only positions that HAVE a non-central peak: the sentinel is
            // finite, so an `isFinite` test let every empty or central-only
            // position add 3.4e38, and at half the scan the median exploded
            // and no basis was found.
            if nearest < .greatestFiniteMagnitude { nearestRadii.append(nearest) }
        }
        guard let typicalRadius = median(nearestRadii) else { return nil }
        let clusterTolerance = max(0.5, typicalRadius * 0.18)

        func cell(x: Float, y: Float) -> VectorCell {
            VectorCell(
                x: Int(floor(Double(x / clusterTolerance))),
                y: Int(floor(Double(y / clusterTolerance)))
            )
        }

        // A spatial hash keeps this pass near-linear even when a noisy dataset
        // contains many unique outliers. Cluster centers can drift across cell
        // boundaries, so their bucket membership is updated after every merge.
        var clusters: [VectorCluster] = []
        var clusterCells: [VectorCell] = []
        var buckets: [VectorCell: [Int]] = [:]
        var visited = 0
        for observation in observations.sorted(by: { $0.radius < $1.radius }) {
            visited += 1
            if visited & 1023 == 0, cancellation?.isCancelled == true { return nil }
            var nearestIndex: Int?
            var nearestDistance = clusterTolerance
            let observationCell = cell(x: observation.x, y: observation.y)
            for cellY in (observationCell.y - 1)...(observationCell.y + 1) {
                for cellX in (observationCell.x - 1)...(observationCell.x + 1) {
                    for index in buckets[VectorCell(x: cellX, y: cellY)] ?? [] {
                        let distance = hypot(observation.x - clusters[index].x,
                                             observation.y - clusters[index].y)
                        if distance < nearestDistance {
                            nearestDistance = distance
                            nearestIndex = index
                        }
                    }
                }
            }
            if let nearestIndex {
                let oldCell = clusterCells[nearestIndex]
                clusters[nearestIndex].sumX += observation.x
                clusters[nearestIndex].sumY += observation.y
                clusters[nearestIndex].count += 1
                let newCell = cell(
                    x: clusters[nearestIndex].x, y: clusters[nearestIndex].y
                )
                if newCell != oldCell {
                    buckets[oldCell]?.removeAll { $0 == nearestIndex }
                    if buckets[oldCell]?.isEmpty == true { buckets[oldCell] = nil }
                    buckets[newCell, default: []].append(nearestIndex)
                    clusterCells[nearestIndex] = newCell
                }
            } else {
                let index = clusters.count
                clusters.append(VectorCluster(
                    sumX: observation.x, sumY: observation.y, count: 1
                ))
                clusterCells.append(observationCell)
                buckets[observationCell, default: []].append(index)
            }
        }

        // DEVIATION (occupiedPositions): the 8 %
        // support bar is mac4DSTEM's own automation (py4DSTEM picks maxima of
        // the whole-scan vector histogram and has no per-position count), and
        // it is a fraction of the positions a cluster CAN be drawn from --
        // those that hold a peak beyond `minRadius`, i.e. `nearestRadii`. It
        // counted every position with any peak, so central-beam-only positions
        // raised the bar: at 95 % central-only (lattice in 5 %) no cluster
        // reached 8 % and no basis was found.
        let occupiedPositions = max(1, nearestRadii.count)
        let minimumClusterSupport = max(1, Int(ceil(Double(occupiedPositions) * 0.08)))
        let candidates = clusters
            .filter { $0.count >= minimumClusterSupport && $0.radius > minRadius }
            .sorted {
                if $0.count != $1.count { return $0.count > $1.count }
                return $0.radius < $1.radius
            }
            .prefix(24)

        guard candidates.count >= 2 else { return nil }

        var best: BasisEstimate?
        for first in candidates.indices {
            for second in candidates.indices where second > first {
                if cancellation?.isCancelled == true { return nil }
                let g1 = (x: candidates[first].x, y: candidates[first].y)
                let g2 = (x: candidates[second].x, y: candidates[second].y)
                guard let condition = conditionNumber(g1: g1, g2: g2), condition <= 8 else {
                    continue
                }
                let tolerance = indexingTolerance(g1: g1, g2: g2)
                let score = scoreBasis(
                    g1: g1, g2: g2,
                    observations: observations, tolerance: tolerance
                )
                guard score.support >= 2,
                      Float(score.support) / Float(observations.count) >= 0.5 else { continue }
                let estimate = BasisEstimate(
                    g1: g1, g2: g2,
                    observationCount: observations.count,
                    supportCount: score.support,
                    residualPixels: score.residual,
                    conditionNumber: condition,
                    indexingTolerancePixels: tolerance
                )
                guard let current = best else { best = estimate; continue }
                let supportGain = estimate.supportCount - current.supportCount
                if supportGain > 0
                    || (supportGain == 0 && estimate.residualPixels < current.residualPixels - 1e-5)
                    || (supportGain == 0
                        && abs(estimate.residualPixels - current.residualPixels) <= 1e-5
                        && hypot(estimate.g1.x, estimate.g1.y)
                            + hypot(estimate.g2.x, estimate.g2.y)
                            < hypot(current.g1.x, current.g1.y)
                                + hypot(current.g2.x, current.g2.y)) {
                    best = estimate
                }
            }
        }
        return best
    }

    static func scoreBasis(
        g1: (x: Float, y: Float), g2: (x: Float, y: Float),
        observations: [VectorObservation], tolerance: Float
    ) -> (support: Int, residual: Float) {
        guard let inverse = invert2x2(g1.x, g2.x, g1.y, g2.y) else {
            return (0, .infinity)
        }
        var support = 0
        var squaredResidual: Double = 0
        for observation in observations {
            let h = (inverse.a * observation.x + inverse.b * observation.y).rounded()
            let k = (inverse.c * observation.x + inverse.d * observation.y).rounded()
            guard abs(h) <= 8, abs(k) <= 8, h != 0 || k != 0 else { continue }
            let residual = hypot(
                observation.x - h * g1.x - k * g2.x,
                observation.y - h * g1.y - k * g2.y
            )
            if residual <= tolerance {
                support += 1
                squaredResidual += Double(residual * residual)
            }
        }
        return (
            support,
            support > 0 ? Float(sqrt(squaredResidual / Double(support))) : .infinity
        )
    }

    static func indexingTolerance(
        g1: (x: Float, y: Float), g2: (x: Float, y: Float)
    ) -> Float {
        max(0.5, 0.18 * min(hypot(g1.x, g1.y), hypot(g2.x, g2.y)))
    }

    static func conditionNumber(
        g1: (x: Float, y: Float), g2: (x: Float, y: Float)
    ) -> Float? {
        let trace = g1.x * g1.x + g1.y * g1.y + g2.x * g2.x + g2.y * g2.y
        let determinant = g1.x * g2.y - g2.x * g1.y
        let discriminant = max(0, trace * trace - 4 * determinant * determinant)
        let root = sqrt(discriminant)
        let maximum = (trace + root) / 2
        let minimum = (trace - root) / 2
        guard minimum > 1e-10, maximum.isFinite else { return nil }
        return sqrt(maximum / minimum)
    }

    // MARK: - Weighted lattice fit (fit_lattice_vectors)

    /// Intensity-weighted least-squares fit of [origin, g1, g2] to indexed
    /// peaks. Basis row for a peak is [1, h, k]; solves the 3×3 normal
    /// equations for each of the x and y coordinates.
    ///
    /// DEVIATION (measured 2026-08-04, training-dataset campaign parity):
    /// py4DSTEM's fit_lattice_vectors multiplies both the design matrix and
    /// the data by the intensity weight before lstsq, which minimizes
    /// Σ w²·r². These normal equations weight by w once, minimizing Σ w·r² —
    /// the statistically standard weighted least squares. On sim_Au the two
    /// estimators differ by ~5e-3 median per strain component (vs ~2e-4 when
    /// the estimators are matched). The parity record reports the deviation
    /// magnitude per dataset; see parity_py4dstem.py `weighting_deviation_*`.
    static func fitLattice(
        _ peaks: [(h: Float, k: Float, x: Float, y: Float, wgt: Float)]
    ) -> (originX: Float, originY: Float,
          g1x: Float, g1y: Float, g2x: Float, g2y: Float,
          residual: Float)? {
        // Normal equations MᵀW M (3×3, symmetric) and MᵀW·[x, y] (3×2).
        var ata = [Double](repeating: 0, count: 9)
        var atbx = [Double](repeating: 0, count: 3)
        var atby = [Double](repeating: 0, count: 3)
        for p in peaks {
            let w = Double(p.wgt)
            let b = [1.0, Double(p.h), Double(p.k)]
            for i in 0..<3 {
                atbx[i] += w * b[i] * Double(p.x)
                atby[i] += w * b[i] * Double(p.y)
                for j in 0..<3 { ata[i * 3 + j] += w * b[i] * b[j] }
            }
        }
        guard let cx = solve3(ata, atbx), let cy = solve3(ata, atby) else { return nil }
        var squaredResidual: Double = 0
        var weightSum: Double = 0
        for peak in peaks {
            let predictedX = cx[0] + cx[1] * Double(peak.h) + cx[2] * Double(peak.k)
            let predictedY = cy[0] + cy[1] * Double(peak.h) + cy[2] * Double(peak.k)
            let dx = Double(peak.x) - predictedX
            let dy = Double(peak.y) - predictedY
            let weight = Double(peak.wgt)
            squaredResidual += weight * (dx * dx + dy * dy)
            weightSum += weight
        }
        // c[0] = origin, c[1] = g1, c[2] = g2 for each coordinate.
        return (
            originX: Float(cx[0]), originY: Float(cy[0]),
            g1x: Float(cx[1]), g1y: Float(cy[1]),
            g2x: Float(cx[2]), g2y: Float(cy[2]),
            residual: weightSum > 0 ? Float(sqrt(squaredResidual / weightSum)) : .infinity
        )
    }

    /// One reweighted pass removes a peak that happened to fall inside the
    /// coarse indexing gate but is inconsistent with the local lattice fit.
    static func robustFitLattice(
        _ peaks: [(h: Float, k: Float, x: Float, y: Float, wgt: Float)],
        minNumPeaks: Int
    ) -> (originX: Float, originY: Float,
          g1x: Float, g1y: Float, g2x: Float, g2y: Float,
          residual: Float)? {
        guard let initial = fitLattice(peaks) else { return nil }
        let residuals = peaks.map { peak in
            hypot(
                peak.x - initial.originX - peak.h * initial.g1x - peak.k * initial.g2x,
                peak.y - initial.originY - peak.h * initial.g1y - peak.k * initial.g2y
            )
        }
        guard let center = median(residuals) else { return initial }
        let deviations = residuals.map { abs($0 - center) }
        let mad = median(deviations) ?? 0
        let threshold = max(0.35, center + 3 * max(1.4826 * mad, 0.05))
        let inliers = zip(peaks, residuals).compactMap { peak, residual in
            residual <= threshold ? peak : nil
        }
        guard inliers.count >= minNumPeaks, inliers.count < peaks.count else { return initial }
        return fitLattice(inliers) ?? initial
    }

    /// DEVIATION from py4DSTEM: `get_reference_g1g2` (process/strain/latticevectors.py)
    /// takes the plain component-wise median of every position inside the caller's
    /// mask. This is called first, and discards candidates whose fitted (g1, g2)
    /// deviate from that median by more than a MAD-derived threshold (falling back
    /// to the full candidate set when fewer than half of the candidates survive) —
    /// the robust reference rejects distorted positions, per this file's own header
    /// above, so a strained or mis-indexed sub-region inside the reference mask
    /// cannot pull the reference lattice off the bulk.
    static func robustReferenceIndices(
        candidates: [Int],
        g1x: [Float], g1y: [Float], g2x: [Float], g2y: [Float]
    ) -> [Int] {
        guard candidates.count > 2 else { return candidates }
        let centerG1X = median(candidates.map { g1x[$0] }) ?? 0
        let centerG1Y = median(candidates.map { g1y[$0] }) ?? 0
        let centerG2X = median(candidates.map { g2x[$0] }) ?? 0
        let centerG2Y = median(candidates.map { g2y[$0] }) ?? 0
        let residuals = candidates.map { index in
            hypot(g1x[index] - centerG1X, g1y[index] - centerG1Y)
                + hypot(g2x[index] - centerG2X, g2y[index] - centerG2Y)
        }
        let center = median(residuals) ?? 0
        let mad = median(residuals.map { abs($0 - center) }) ?? 0
        let basisScale = (
            hypot(centerG1X, centerG1Y) + hypot(centerG2X, centerG2Y)
        ) / 2
        let threshold = center + max(3 * 1.4826 * mad, 0.02 * basisScale)
        let inliers = zip(candidates, residuals).compactMap { index, residual in
            residual <= threshold ? index : nil
        }
        // Do not let aggressive rejection turn a broad but valid reference
        // population into a tiny, overconfident subset.
        return inliers.count * 2 >= candidates.count ? inliers : candidates
    }

    // MARK: - Linear algebra helpers

    static func invert2x2(_ a: Float, _ b: Float, _ c: Float, _ d: Float)
        -> (a: Float, b: Float, c: Float, d: Float)? {
        // Inverse of [[a, b], [c, d]].
        let det = a * d - b * c
        guard abs(det) > 1e-12 else { return nil }
        let inv = 1 / det
        return (a: d * inv, b: -b * inv, c: -c * inv, d: a * inv)
    }

    static func median(_ values: [Float]) -> Float? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[middle - 1] + sorted[middle]) / 2
        }
        return sorted[middle]
    }

    /// Solve a 3×3 system by Gaussian elimination with partial pivoting.
    static func solve3(_ aIn: [Double], _ bIn: [Double]) -> [Double]? {
        var a = aIn, b = bIn
        for col in 0..<3 {
            var pivot = col
            for row in (col + 1)..<3 where abs(a[row * 3 + col]) > abs(a[pivot * 3 + col]) {
                pivot = row
            }
            if abs(a[pivot * 3 + col]) < 1e-12 { return nil }
            if pivot != col {
                for k in 0..<3 { a.swapAt(col * 3 + k, pivot * 3 + k) }
                b.swapAt(col, pivot)
            }
            for row in (col + 1)..<3 {
                let f = a[row * 3 + col] / a[col * 3 + col]
                for k in col..<3 { a[row * 3 + k] -= f * a[col * 3 + k] }
                b[row] -= f * b[col]
            }
        }
        var x = [Double](repeating: 0, count: 3)
        for row in stride(from: 2, through: 0, by: -1) {
            var v = b[row]
            for k in (row + 1)..<3 { v -= a[row * 3 + k] * x[k] }
            x[row] = v / a[row * 3 + row]
        }
        return x
    }
}

struct ConsensusPeakGrid { var peaks: [[ConsensusPeak]] }
struct ConsensusPeak { var x: Float; var y: Float; var intensity: Float }
struct ConsensusCancellation {
    let check: @Sendable () -> Bool
    var isCancelled: Bool { check() }
}
