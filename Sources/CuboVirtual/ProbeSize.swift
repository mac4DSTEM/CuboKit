//
//  ProbeSize.swift
//  The radius and centre of the bright central disk (registration slice 2, 3.5): a port of py4DSTEM's `get_probe_size`
//  (process/calibration/probe.py 6 at f050d207), step for step in float64, with NumPy's linspace, gradient, median and
//  pairwise sums, so it reproduces py4DSTEM's numbers bit for bit on the same pattern.
//
//  DEVIATION, in the refusing direction: where py4DSTEM returns NaN (no trusted radius, or a centre of mass of 0/0), the
//  kit throws `probeNotFound`.
//

import CuboCore
import Foundation

public struct ProbeEstimate: Sendable, Equatable {
    /// The disk radius in detector pixels, `sqrt(count / π)` averaged over the trusted thresholds.
    public let radius: Double
    /// The disk centre (py4DSTEM's `x0`, `y0`): the centre of mass above the trusted threshold.
    public let centerRow: Double
    public let centerColumn: Double
    /// The trusted threshold, as a fraction of the pattern's maximum.
    public let threshold: Double
}

public enum ProbeSize {
    public static func estimate(_ pattern: DetectorImage, lowerThreshold: Double = 0.01, upperThreshold: Double = 0.99,
                                count: Int = 100) throws -> ProbeEstimate {
        guard count >= 2 else { throw CuboVirtualError.probeNotFound("needs at least two thresholds") }
        let v = pattern.values
        let qx = pattern.detectorColumns
        // NumPy's linspace: k · step + start, the last value set to stop.
        let step = (upperThreshold - lowerThreshold) / Double(count - 1)
        var thresholds = (0..<count).map { Double($0) * step + lowerThreshold }
        thresholds[count - 1] = upperThreshold
        let maximum = v.reduce(-Double.infinity) { $0.isNaN || $1.isNaN ? .nan : Swift.max($0, $1) }
        let radii = thresholds.map { t -> Double in
            let limit = maximum * t
            return (Double(v.reduce(0) { $1 > limit ? $0 + 1 : $0 }) / Double.pi).squareRoot()
        }
        // np.gradient, unit spacing: one-sided first differences at the ends, central differences inside.
        var gradient = [Double](repeating: 0, count: count)
        gradient[0] = radii[1] - radii[0]
        gradient[count - 1] = radii[count - 1] - radii[count - 2]
        for k in 1..<(count - 1) { gradient[k] = (radii[k + 1] - radii[k - 1]) / 2.0 }
        let sorted = gradient.sorted()
        let median = count % 2 == 1 ? sorted[count / 2] : (sorted[count / 2 - 1] + sorted[count / 2]) / 2
        let trusted = (0..<count).filter { gradient[$0] <= 0 && gradient[$0] >= 2 * median }
        let radius = NumPySum.mean(trusted.map { radii[$0] })
        let threshold = NumPySum.mean(trusted.map { thresholds[$0] })
        // get_CoM(DP * mask): the masked pattern's centre of mass, row then column, with NumPy's sums.
        let limit = maximum * threshold
        let masked = v.map { $0 * ($0 > limit ? 1.0 : 0.0) }
        let total = NumPySum.pairwise(masked)
        let row = NumPySum.pairwise(masked.indices.map { Double($0 / qx) * masked[$0] }) / total
        let column = NumPySum.pairwise(masked.indices.map { Double($0 % qx) * masked[$0] }) / total
        guard radius.isFinite, row.isFinite, column.isFinite else {
            throw CuboVirtualError.probeNotFound(trusted.isEmpty ? "no trusted threshold" : "the masked pattern sums to zero")
        }
        return ProbeEstimate(radius: radius, centerRow: row, centerColumn: column, threshold: threshold)
    }
}
