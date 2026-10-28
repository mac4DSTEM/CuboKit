//
//  CenterOfMass.swift
//  The intensity-weighted centre of mass of every pattern: py4DSTEM's
//  `_calculate_intensities_center_of_mass` (process/phase/phase_base_class.py 613 at f050d207), in Double.
//
//  DEVIATION from a common Float32 implementation, toward py4DSTEM: negative intensities are summed as they are (that
//  implementation clamps them at 0), a zero sum gives NaN (it substitutes the reference centre), and no reference is subtracted here (CuboPhase normalizes).
//

import CuboCore

/// The centre of mass per scan position, in detector pixels, and the sums it is made of.
public struct CenterOfMassMaps: Sendable, Equatable {
    /// Detector-row coordinate of the centre of mass (py4DSTEM's `com_x`).
    public let row: ScanImage
    /// Detector-column coordinate (py4DSTEM's `com_y`).
    public let column: ScanImage
    /// Σ w·P per position.
    public let intensity: ScanImage
    /// Σ i·w·P and Σ j·w·P per position.
    public let momentRow: ScanImage
    public let momentColumn: ScanImage
}

public enum CenterOfMass {
    /// The centre of mass of every pattern of `source`, weighted by `mask`'s weights when one is given.
    public static func measure(of source: some CubeSource, mask: DetectorGeometry? = nil, path: ComputePath = .cpu,
                               tiling: ScanTiling = .automatic,
                               progress: (@Sendable (Double) -> Void)? = nil,
                               isCancelled: (@Sendable () -> Bool)? = nil) async throws -> CenterOfMassMaps {
        let shape = source.shape
        let weights = try mask.map { try DetectorMask.make($0, shape: shape) }
            ?? [Float](repeating: 1, count: shape.detectorCount)
        let sums: (s: [Double], r: [Double], c: [Double]) = try await withInterruption(isCancelled) {
            switch path {
            case .cpu: return try await cpuSums(of: source, weights: weights, tiling: tiling, progress: progress)
            case .metal: return try await MetalReducers.centerOfMassSums(of: source, weights: weights, tiling: tiling,
                                                                          progress: progress)
            }
        }
        func image(_ v: [Double]) -> ScanImage {
            ScanImage(scanRows: shape.scanRows, scanColumns: shape.scanColumns, values: v, path: path)
        }
        return CenterOfMassMaps(row: image(zip(sums.r, sums.s).map { $0 / $1 }),
                                column: image(zip(sums.c, sums.s).map { $0 / $1 }),
                                intensity: image(sums.s), momentRow: image(sums.r), momentColumn: image(sums.c))
    }

    /// Per position, Σ wP, Σ (w i) P and Σ (w j) P in Double with Neumaier compensation. The weights times the index
    /// are formed first in Double (exact for Float32 weights and indices below 2^24), as the float64 definition does.
    static func cpuSums(of source: some CubeSource, weights: [Float], tiling: ScanTiling,
                        progress: (@Sendable (Double) -> Void)?) async throws -> (s: [Double], r: [Double], c: [Double]) {
        let shape = source.shape
        let n = shape.detectorCount, qx = shape.detectorColumns
        let w = weights.map(Double.init)
        let wi = (0..<n).map { w[$0] * Double($0 / qx) }
        let wj = (0..<n).map { w[$0] * Double($0 % qx) }
        var s = [Double](repeating: 0, count: shape.scanCount)
        var r = s, c = s
        for range in try tiling.ranges(for: shape) {
            try checkInterrupted()
            let tile = try await source.readScanTile(rows: range)
            let first = range.lowerBound * shape.scanColumns
            // Positions in parallel: each keeps its own sums in pixel order, so the result is
            // bit identical to the sequential loop.
            let values = tile.values
            let found = parallelMap(tile.positionCount) { (k: Int) -> (Double, Double, Double) in
                var a = Neumaier(), b = Neumaier(), d = Neumaier()
                for i in 0..<n {
                    let x = Double(values[k * n + i])
                    a.add(x * w[i]); b.add(x * wi[i]); d.add(x * wj[i])
                }
                return (a.value, b.value, d.value)
            }
            for k in found.indices { s[first + k] = found[k].0; r[first + k] = found[k].1; c[first + k] = found[k].2 }
            progress?(Double(range.upperBound) / Double(shape.scanRows))
        }
        return (s, r, c)
    }
}

/// Neumaier's compensated sum; a non-finite step carries no correction (∞ − ∞ would be NaN).
struct Neumaier {
    var sum = 0.0
    var correction = 0.0
    mutating func add(_ x: Double) {
        let t = sum + x
        if t.isFinite { correction += abs(sum) >= abs(x) ? (sum - t) + x : (x - t) + sum }
        sum = t
    }
    var value: Double { sum + correction }
}
