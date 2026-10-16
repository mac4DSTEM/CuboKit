//
//  ProbeRingHint.swift
//  The outer edge of a ring-structured probe (a bullseye), read from the mean pattern: a suggestion for the probe radius of
//  a trench kernel. mac4DSTEM's Core/Analysis/ProbeRingHint.swift at c809679c, which has no py4DSTEM counterpart; py4DSTEM's
//  probe size reads such a probe at its first shoulder, where a trench kernel never finds the beam. The 20 % edge and the 0.25
//  dip are the app's, measured on its own datasets: a hint to offer, not a default.
//

import CuboVirtual
import Foundation

public struct ProbeRingHint: Sendable, Equatable {
    /// The outer edge of the ring structure, detector pixels (an integer radial bin).
    public let outerRadius: Int
    /// The profile minimum before the last ring maximum, over the profile maximum.
    public let dip: Double

    /// The hint is worth showing only when the radius in use is not already within half a pixel of the edge.
    public func isWorthOffering(currentRadius: Double) -> Bool { abs(Double(outerRadius) - currentRadius) >= 0.5 }

    /// The azimuthal mean about `ProbeSize.estimate`'s centre (integer-radius bins) is read from the outside in: the outer edge
    /// is the largest bin still at least 20 % of the profile maximum. The probe counts as ring-structured when, out to twice
    /// its measured radius, the profile has a local maximum of at least 20 % of its maximum after a dip below 25 % of it. The
    /// hint is returned only when the edge lies beyond the measured radius and is at most min(rows, columns) / 8. Non-finite
    /// pixels count as 0 for the probe size and are left out of the profile. Nil when there is no hint, or no probe can be
    /// measured (no positive finite pixel, or no mass above the trusted threshold).
    public static func outerEdge(of meanPattern: DetectorImage) -> ProbeRingHint? {
        let qy = meanPattern.detectorRows, qx = meanPattern.detectorColumns
        guard qy > 0, qx > 0, meanPattern.values.count == qy * qx else { return nil }
        let v = meanPattern.values
        guard v.contains(where: { $0.isFinite && $0 > 0 }) else { return nil }
        let cleaned = DetectorImage(detectorRows: qy, detectorColumns: qx, values: v.map { $0.isFinite ? $0 : 0 }, path: .cpu)
        guard let probe = try? ProbeSize.estimate(cleaned) else { return nil }
        let bins = Int(Double(qx * qx + qy * qy).squareRoot()) + 2
        var sum = [Double](repeating: 0, count: bins)
        var count = [Double](repeating: 0, count: bins)
        for y in 0..<qy {
            for x in 0..<qx {
                let value = v[y * qx + x]
                guard value.isFinite else { continue }
                let r = Int(hypot(Double(y) - probe.centerRow, Double(x) - probe.centerColumn))
                sum[r] += value; count[r] += 1
            }
        }
        let profile = (0..<bins).map { count[$0] > 0 ? sum[$0] / count[$0] : 0 }
        guard let peak = profile.max(), peak > 0 else { return nil }
        var outer = 0
        for i in 0..<bins where profile[i] >= 0.2 * peak { outer = i + 1 }
        var ring = 0
        let upper = min(Int(2 * probe.radius) + 1, bins - 1)
        guard upper > 1 else { return nil }
        for i in 1..<upper where profile[i] > profile[i - 1] && profile[i] >= profile[i + 1] && profile[i] >= 0.2 * peak {
            ring = i
        }
        guard ring > 1, let dip = profile[1..<ring].min(), dip / peak < 0.25 else { return nil }
        guard Double(outer) > probe.radius, Double(outer) <= Double(min(qx, qy)) / 8 else { return nil }
        return ProbeRingHint(outerRadius: outer, dip: dip / peak)
    }
}
