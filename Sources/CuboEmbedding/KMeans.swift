import Foundation

/// The outcome of a k-means clustering.
public struct KMeansResult: Sendable, Equatable {
    /// Cluster index per row.
    public let labels: [Int]
    /// `clusters × columns`, row-major.
    public let centroids: [Double]
    /// Sum of squared distances of rows to their centroid.
    public let inertia: Double
    /// Lloyd iterations run (scikit-learn's `n_iter_`; the mac4DSTEM variant counts passes).
    public let iterations: Int
    /// True when iteration stopped before the cap.
    public let converged: Bool
}

/// Lloyd's k-means.
public enum KMeans {
    /// scikit-learn 1.4.2 `KMeans(init=initialCentroids, n_init=1, algorithm="lloyd")` in Double.
    ///
    /// Each iteration assigns rows to the nearest centroid (the lowest index wins an exact tie), moves empty
    /// clusters onto the rows farthest from their centroids, and stops when the labels repeat, or when the summed
    /// squared centroid shift is at most `tolerance` times the mean per-column variance of the data (population
    /// variance). When it did not stop on repeated labels, the labels are recomputed against the final centroids.
    /// With several empty clusters in one iteration the order in which they take the farthest rows is unspecified
    /// by scikit-learn (an `argpartition`) and not pinned here; rows are given out by descending distance, lower
    /// index first.
    public static func fit(_ data: [Double], rows n: Int, columns d: Int, initialCentroids: [Double],
                           maxIterations: Int = 300, tolerance: Double = 1e-4) throws -> KMeansResult {
        guard n > 0, d > 0, data.count == n * d else {
            throw EmbeddingError.invalidShape("data.count \(data.count) is not rows \(n) × columns \(d)")
        }
        guard !initialCentroids.isEmpty, initialCentroids.count % d == 0 else {
            throw EmbeddingError.invalidShape("initialCentroids.count must be a positive multiple of columns \(d)")
        }
        let k = initialCentroids.count / d
        guard k <= n else { throw EmbeddingError.invalidCount("clusters \(k) exceed rows \(n)") }
        guard maxIterations >= 1, tolerance >= 0 else {
            throw EmbeddingError.invalidCount("maxIterations must be at least 1 and tolerance not negative")
        }
        guard data.allSatisfy(\.isFinite), initialCentroids.allSatisfy(\.isFinite) else {
            throw EmbeddingError.nonFinite
        }
        var meanVariance = 0.0
        for j in 0..<d {
            var m = 0.0
            for i in 0..<n { m += data[i * d + j] }
            m /= Double(n)
            var v = 0.0
            for i in 0..<n { let t = data[i * d + j] - m; v += t * t }
            meanVariance += v / Double(n)
        }
        let absoluteTolerance = meanVariance / Double(d) * tolerance

        var centers = initialCentroids
        var labels = [Int](repeating: -1, count: n)
        var oldLabels = labels
        var strict = false
        var iterations = 0
        var stoppedEarly = false
        for _ in 0..<maxIterations {
            iterations += 1
            var sums = [Double](repeating: 0, count: k * d)
            var weights = [Double](repeating: 0, count: k)
            for i in 0..<n {
                let c = nearest(data, i, d, centers, k)
                labels[i] = c
                weights[c] += 1
                for j in 0..<d { sums[c * d + j] += data[i * d + j] }
            }
            let empty = (0..<k).filter { weights[$0] == 0 }
            if !empty.isEmpty {
                var dist = [Double](repeating: 0, count: n)
                for i in 0..<n {
                    var s = 0.0
                    for j in 0..<d { let t = data[i * d + j] - centers[labels[i] * d + j]; s += t * t }
                    dist[i] = s
                }
                let far = (0..<n).sorted { dist[$0] != dist[$1] ? dist[$0] > dist[$1] : $0 < $1 }
                for (e, cluster) in empty.enumerated() {
                    let idx = far[e], old = labels[idx]
                    for j in 0..<d {
                        sums[old * d + j] -= data[idx * d + j]
                        sums[cluster * d + j] = data[idx * d + j]
                    }
                    weights[cluster] = 1
                    weights[old] -= 1
                }
            }
            var next = [Double](repeating: 0, count: k * d)
            var shift = 0.0
            for c in 0..<k {
                for j in 0..<d {
                    next[c * d + j] = weights[c] > 0 ? sums[c * d + j] / weights[c] : sums[c * d + j]
                    let t = next[c * d + j] - centers[c * d + j]
                    shift += t * t
                }
            }
            centers = next
            if labels == oldLabels { strict = true; stoppedEarly = true; break }
            if shift <= absoluteTolerance { stoppedEarly = true; break }
            oldLabels = labels
        }
        if !strict {
            for i in 0..<n { labels[i] = nearest(data, i, d, centers, k) }
        }
        var inertia = 0.0
        for i in 0..<n {
            for j in 0..<d { let t = data[i * d + j] - centers[labels[i] * d + j]; inertia += t * t }
        }
        return KMeansResult(labels: labels, centroids: centers, inertia: inertia, iterations: iterations,
                            converged: stoppedEarly)
    }

    private static func nearest(_ x: [Double], _ i: Int, _ d: Int, _ centers: [Double], _ k: Int) -> Int {
        var best = 0, bestD = Double.infinity
        for c in 0..<k {
            var s = 0.0
            for j in 0..<d { let t = x[i * d + j] - centers[c * d + j]; s += t * t }
            if s < bestD { bestD = s; best = c }
        }
        return best
    }

    /// mac4DSTEM `DiffractionEmbedding.kMeans` (c809679c): Float32 coordinates, k-means++ seeding from a
    /// splitmix64 generator started at `seed &+ 0x5A17` through Swift's `Int.random(in:using:)` and
    /// `Double.random(in:using:)`, at most `maxIterations` (50) passes, no tolerance (stops when no label changes),
    /// and an empty cluster is re-seeded at the row farthest from its own pre-update centroid. `groups` is clamped to
    /// `1...rows`. Distances and means are Double; centroids are returned rounded to Float32.
    ///
    /// The seeding depends on the Swift standard library's range sampling, so it is reproducible for a given
    /// toolchain's algorithm, not across languages.
    public static func fitSeeded(_ data: [Double], rows n: Int, columns d: Int, groups: Int, seed: UInt64,
                                 maxIterations: Int = 50) throws -> KMeansResult {
        guard n > 0, d > 0, data.count == n * d else {
            throw EmbeddingError.invalidShape("data.count \(data.count) is not rows \(n) × columns \(d)")
        }
        let x = data.map { Float($0) }
        guard x.allSatisfy(\.isFinite) else { throw EmbeddingError.nonFinite }
        let k = max(1, min(groups, n))
        var generator = SplitMix64(seed: seed &+ 0x5A17)
        func point(_ i: Int) -> [Double] { (0..<d).map { Double(x[i * d + $0]) } }
        func squaredDistance(_ i: Int, _ target: [Double]) -> Double {
            var sum = 0.0
            for j in 0..<d { let diff = Double(x[i * d + j]) - target[j]; sum += diff * diff }
            return sum
        }
        var indices = [Int.random(in: 0..<n, using: &generator)]
        var minDistances = [Double](repeating: .infinity, count: n)
        while indices.count < k {
            let last = point(indices[indices.count - 1])
            for i in 0..<n {
                let dist = squaredDistance(i, last)
                if dist < minDistances[i] { minDistances[i] = dist }
            }
            let total = minDistances.reduce(0, +)
            if !total.isFinite || total <= 0 {
                for i in 0..<n where !indices.contains(i) {
                    indices.append(i)
                    if indices.count == k { break }
                }
                continue
            }
            let target = Double.random(in: 0..<total, using: &generator)
            var cumulative = 0.0
            var chosen = n - 1
            for i in 0..<n {
                cumulative += minDistances[i]
                if cumulative >= target { chosen = i; break }
            }
            indices.append(chosen)
        }
        var centroids = indices.flatMap(point)
        var assignment = [Int](repeating: 0, count: n)
        var passes = 0
        var settled = false
        for _ in 0..<maxIterations {
            passes += 1
            var changed = false
            for i in 0..<n {
                var bestCluster = 0
                var bestDistance = Double.infinity
                for c in 0..<k {
                    var sum = 0.0
                    for j in 0..<d {
                        let diff = Double(x[i * d + j]) - centroids[c * d + j]
                        sum += diff * diff
                    }
                    if sum < bestDistance { bestDistance = sum; bestCluster = c }
                }
                if assignment[i] != bestCluster { changed = true }
                assignment[i] = bestCluster
            }
            var sums = [Double](repeating: 0, count: k * d)
            var counts = [Int](repeating: 0, count: k)
            for i in 0..<n {
                let c = assignment[i]
                counts[c] += 1
                for j in 0..<d { sums[c * d + j] += Double(x[i * d + j]) }
            }
            let old = centroids
            for c in 0..<k where counts[c] > 0 {
                for j in 0..<d { centroids[c * d + j] = sums[c * d + j] / Double(counts[c]) }
            }
            for c in 0..<k where counts[c] == 0 {
                var farthest = 0
                var farthestDistance = -1.0
                for i in 0..<n {
                    let own = assignment[i]
                    let d2 = squaredDistance(i, Array(old[(own * d)..<(own * d + d)]))
                    if d2 > farthestDistance { farthestDistance = d2; farthest = i }
                }
                assignment[farthest] = c
                centroids.replaceSubrange(c * d..<(c + 1) * d, with: point(farthest))
                changed = true
            }
            if !changed { settled = true; break }
        }
        let out = centroids.map { Double(Float($0)) }
        var inertia = 0.0
        for i in 0..<n {
            for j in 0..<d { let t = Double(x[i * d + j]) - out[assignment[i] * d + j]; inertia += t * t }
        }
        return KMeansResult(labels: assignment, centroids: out, inertia: inertia, iterations: passes, converged: settled)
    }
}

/// splitmix64, as in the mac4DSTEM source.
private struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
