import Foundation
import Testing
@testable import CuboEmbedding

@Suite struct EmbeddingTests {
    let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")
    func array(_ name: String) throws -> [Double] {
        let d = try Data(contentsOf: directory.appendingPathComponent(name + ".f64"))
        return d.withUnsafeBytes { raw in
            stride(from: 0, to: raw.count, by: 8).map {
                Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: $0, as: UInt64.self)))
            }
        }
    }
    func gap(_ a: [Double], _ b: [Double]) -> Double {
        #expect(a.count == b.count)
        return zip(a, b).map { abs($0 - $1) }.max() ?? 0
    }
    func close(_ label: String, _ a: [Double], _ b: [Double], _ tolerance: Double) {
        let g = gap(a, b)
        print("embedding \(label) maximum absolute gap: \(g), tolerance: \(tolerance)")
        #expect(g <= tolerance)
    }
    /// Cuts `expected` into consecutive pieces of the given lengths.
    func split(_ a: [Double], _ lengths: [Int]) -> [[Double]] {
        var at = 0
        return lengths.map { n in defer { at += n }; return Array(a[at..<(at + n)]) }
    }

    // MARK: scikit-learn PCA

    @Test(arguments: [("pcaA", 60, 8, 4), ("pcaB", 6, 10, 5)])
    func pcaMatchesScikitLearn(_ c: (String, Int, Int, Int)) throws {
        let (name, n, d, k) = c
        let r = try PCA.fit(try array(name + ".in"), rows: n, columns: d, components: k)
        let e = split(try array(name + ".sk"), [d, k * d, k, k, k, n * k])
        close(name + " mean", r.mean, e[0], 1e-12)
        close(name + " components", r.components, e[1], 1e-9)
        close(name + " explainedVariance", r.explainedVariance, e[2], 1e-10)
        close(name + " ratio", r.explainedVarianceRatio, e[3], 1e-12)
        close(name + " singularValues", r.singularValues, e[4], 1e-10)
        close(name + " scores", r.scores, e[5], 1e-9)
        let t = try PCA.transform(try array(name + ".in"), rows: n, result: r)
        close(name + " transform", t, r.scores, 1e-12)
    }

    @Test func pcaRecoversPlantedAxis() throws {
        // All variance on the first column: the single axis is column 0 with the svd_flip sign, ratio exactly 1.
        // Variance: deviations -3.25, -1.25, 0.75, 3.75 sum to 26.75 squares; divided by rows - 1.
        let data: [Double] = [-3, 0, -1, 0, 1, 0, 4, 0]   // the largest |score| (4) is positive
        let r = try PCA.fit(data, rows: 4, columns: 2, components: 1)
        close("planted axis", r.components, [1, 0], 1e-14)
        close("planted ratio", r.explainedVarianceRatio, [1], 1e-14)
        close("planted variance", r.explainedVariance, [107.0 / 12.0], 1e-13)
        let flipped = try PCA.fit(data.map { -$0 }, rows: 4, columns: 2, components: 1)
        // The negated data have their largest |score| (-4) at the last row, so the axis is made to point at it: (-1, 0).
        close("negated data follow the convention", flipped.components, [-1, 0], 1e-14)
    }

    // MARK: mac4DSTEM PCA

    @Test func pcaAppVariantMatchesAppSource() throws {
        let n = 60, d = 8, k = 4
        let r = try PCA.fit(try array("pcaA.in"), rows: n, columns: d, components: k, variant: .diffractionEmbeddingApp)
        let e = split(try array("pcaA.app"), [d, k * d, k, k, n * k])
        close("app mean", r.mean, e[0], 0)
        // LAPACK's sign is not specified across platforms, so align each axis to the pinned sign first.
        var aligned = r.components, coords = r.scores
        for c in 0..<k {
            var dot = 0.0
            for j in 0..<d { dot += r.components[c * d + j] * e[1][c * d + j] }
            if dot < 0 {
                for j in 0..<d { aligned[c * d + j] = -aligned[c * d + j] }
                for i in 0..<n { coords[i * k + c] = -coords[i * k + c] }
            }
        }
        close("app components", aligned, e[1], 1e-6)
        close("app eigenvalues", r.explainedVariance, e[2], 1e-10)
        close("app ratio", r.explainedVarianceRatio, e[3], 1e-6)
        close("app coordinates", coords, e[4], 1e-4)
        #expect(r.singularValues.isEmpty)
    }

    @Test func pcaAppVariantClampsAndSklearnRefuses() throws {
        let data = try array("pcaB.in")
        let app = try PCA.fit(data, rows: 6, columns: 10, components: 99, variant: .diffractionEmbeddingApp)
        #expect(app.componentCount == 5)   // min(columns, rows - 1)
        #expect(throws: EmbeddingError.invalidCount("components 99 must be in 1...6")) {
            try PCA.fit(data, rows: 6, columns: 10, components: 99)
        }
        // The two variants differ in variance scale (rows vs rows - 1) but not in ratios.
        let sk = try PCA.fit(data, rows: 6, columns: 10, components: 5)
        close("variance scale", app.explainedVariance.map { $0 * 6 / 5 }, sk.explainedVariance, 1e-4)
    }

    @Test func pcaRefusesBadInput() {
        #expect(throws: EmbeddingError.nonFinite) { try PCA.fit([1, .nan, 3, 4], rows: 2, columns: 2, components: 1) }
        #expect(throws: EmbeddingError.self) { try PCA.fit([1, 2, 3], rows: 2, columns: 2, components: 1) }
        #expect(throws: EmbeddingError.self) { try PCA.fit([1, 2], rows: 1, columns: 2, components: 1) }
        #expect(throws: EmbeddingError.self) { try PCA.fit([1, 2, 3, 4], rows: 2, columns: 2, components: 0) }
    }

    // MARK: k-means

    @Test(arguments: ["kmBlobs", "kmEmpty", "kmCapped", "kmTol"])
    func kMeansMatchesScikitLearn(_ name: String) throws {
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("embedding.json"))) as! [String: Any]
        let c = (meta["cases"] as! [String: Any])[name] as! [String: Any]
        let n = c["rows"] as! Int, d = c["columns"] as! Int, k = c["clusters"] as! Int
        let input = try array(name + ".in")
        let e = split(try array(name + ".sk"), [k * d, n, 2])
        let r = try KMeans.fit(Array(input[0..<(n * d)]), rows: n, columns: d,
                               initialCentroids: Array(input[(n * d)...]),
                               maxIterations: c["maxIterations"] as! Int, tolerance: c["tolerance"] as! Double)
        #expect(r.labels.map(Double.init) == e[1])
        #expect(r.iterations == Int(e[2][1]))
        close(name + " centroids", r.centroids, e[0], 1e-11)
        let relative = abs(r.inertia - e[2][0]) / e[2][0]
        print("embedding \(name) inertia relative gap: \(relative), tolerance: 1e-13")
        #expect(relative <= 1e-13)
    }

    @Test func kMeansPlantedTruthAndRefusals() throws {
        let data: [Double] = [0, 0, 0, 1, 10, 10, 10, 11]
        let r = try KMeans.fit(data, rows: 4, columns: 2, initialCentroids: [0, 0, 10, 10])
        #expect(r.labels == [0, 0, 1, 1])
        close("planted centroids", r.centroids, [0, 0.5, 10, 10.5], 1e-15)
        #expect(r.inertia == 1)
        #expect(r.converged)
        #expect(throws: EmbeddingError.self) { try KMeans.fit(data, rows: 4, columns: 2, initialCentroids: [0, 0, 1]) }
        #expect(throws: EmbeddingError.self) { try KMeans.fit([0, 0, 1, 1], rows: 2, columns: 2, initialCentroids: [0, 0, 1, 1, 2, 2]) }
        #expect(throws: EmbeddingError.nonFinite) { try KMeans.fit([.nan, 0], rows: 1, columns: 2, initialCentroids: [0, 0]) }
    }

    @Test(arguments: [("appKm3", 90, 3, 3, 1), ("appKm4", 90, 3, 4, 7), ("appKm5", 60, 8, 5, 12345)])
    func seededKMeansMatchesAppSource(_ c: (String, Int, Int, Int, UInt64)) throws {
        let (name, n, d, k, seed) = c
        let r = try KMeans.fitSeeded(try array(name + ".in"), rows: n, columns: d, groups: k, seed: seed)
        let e = split(try array(name + ".app"), [k * d, n])
        close(name + " centroids", r.centroids, e[0], 0)
        #expect(r.labels.map(Double.init) == e[1])
    }

    // MARK: features

    @Test(arguments: [("appEmbedA", 12, 10, 4), ("appEmbedB", 5, 6, 8), ("appEmbedC", 16, 16, 16)])
    func binnedVectorMatchesAppSource(_ c: (String, Int, Int, Int)) throws {
        let (name, qy, qx, b) = c
        let pattern = try array(name + ".in").map { Float($0) }
        let v = try DiffractionFeatures.binnedVector(pattern: pattern, rows: qy, columns: qx, binnedSize: b)
        let e = try array(name + ".app")
        let g = gap(v.map(Double.init), e)
        let scale = max(1, e.map(abs).max() ?? 1)
        print("embedding \(name) binned vector gap: \(g), relative to \(scale), tolerance: 1e-5")
        #expect(g / scale <= 1e-5)
        #expect(v.count == b * b)
    }

    @Test func binnedVectorRefusesBadShape() {
        #expect(throws: EmbeddingError.self) { try DiffractionFeatures.binnedVector(pattern: [1, 2, 3], rows: 2, columns: 2, binnedSize: 2) }
    }

    // MARK: streamed PCA (0.6.0)

    /// The largest relative gap between two vectors, against the larger of 1 and the biggest value of `b`.
    func relativeGap(_ a: [Double], _ b: [Double]) -> Double {
        #expect(a.count == b.count)
        let scale = max(1, b.map(abs).max() ?? 1)
        return zip(a, b).map { abs($0 - $1) }.max().map { $0 / scale } ?? 0
    }

    /// Streams `data` through an accumulator in the given row blocks, in the given order.
    func stream(_ data: [Double], columns d: Int, blocks: [Range<Int>]) throws -> PCACovarianceAccumulator {
        var acc = try PCACovarianceAccumulator(columns: d)
        for b in blocks { try acc.add(Array(data[(b.lowerBound * d)..<(b.upperBound * d)]), rows: b.count) }
        return acc
    }

    func cuts(_ n: Int, _ size: Int) -> [Range<Int>] {
        stride(from: 0, to: n, by: size).map { $0..<min(n, $0 + size) }
    }

    func resolvedSigns(_ r: PCAResult, _ data: [Double], blocks: [Range<Int>]) throws -> PCAResult {
        var res = PCASignResolver(result: r)
        for b in blocks {
            try res.add(Array(data[(b.lowerBound * r.columns)..<(b.upperBound * r.columns)]), rows: b.count,
                        firstRow: b.lowerBound)
        }
        return res.resolved()
    }

    @Test(arguments: [("pcaA", 60, 8, 4), ("pcaB", 6, 10, 5)])
    func streamedPCAMatchesScikitLearnAndInMemory(_ c: (String, Int, Int, Int)) throws {
        let (name, n, d, k) = c
        let data = try array(name + ".in")
        let e = split(try array(name + ".sk"), [d, k * d, k, k, k, n * k])
        let memory = try PCA.fit(data, rows: n, columns: d, components: k)
        let blocks = cuts(n, 7)
        let acc = try stream(data, columns: d, blocks: blocks)
        #expect(acc.count == n)
        let provisional = try PCA.fit(acc, components: k)
        #expect(provisional.scores.isEmpty && provisional.rows == n)
        let r = try resolvedSigns(provisional, data, blocks: blocks)
        // Signs equal scikit-learn's, so the components compare without any alignment.
        let g = [gap(r.mean, e[0]), gap(r.components, e[1]), relativeGap(r.explainedVariance, e[2]),
                 gap(r.explainedVarianceRatio, e[3]), relativeGap(r.singularValues, e[4]),
                 gap(r.components, memory.components)]
        print("embedding streamed \(name) gaps vs scikit-learn (mean, components, variance rel, ratio, singular rel) and vs in-memory components: \(g)")
        #expect(g[0] <= 1e-12)
        #expect(g[1] <= 1e-9)
        #expect(g[2] <= 1e-10)
        #expect(g[3] <= 1e-12)
        #expect(g[4] <= 1e-10)
        #expect(g[5] <= 1e-9)
        // The projection of the data on the streamed axes is scikit-learn's scores.
        close(name + " streamed scores", try PCA.transform(data, rows: n, result: r), e[5], 1e-9)
    }

    @Test func streamedPCAIsTileOrderInvariantAndMergeable() throws {
        let n = 60, d = 8, k = 4
        let data = try array("pcaA.in")
        let reference = try PCA.fit(try stream(data, columns: d, blocks: [0..<n]), components: k)
        var worst = (mean: 0.0, ev: 0.0, comps: 0.0)
        for (size, reversed) in [(1, false), (5, true), (13, false), (59, true)] {
            var blocks = cuts(n, size)
            if reversed { blocks.reverse() }
            let acc = try stream(data, columns: d, blocks: blocks)
            let r = try PCA.fit(acc, components: k)
            worst.mean = max(worst.mean, gap(r.mean, reference.mean))
            worst.ev = max(worst.ev, relativeGap(r.explainedVariance, reference.explainedVariance))
            worst.comps = max(worst.comps, gap(r.components, reference.components))
        }
        // Two halves reduced separately, then merged.
        var a = try stream(data, columns: d, blocks: [0..<23]), b = try stream(data, columns: d, blocks: [23..<n])
        try b.merge(PCACovarianceAccumulator(columns: d))   // merging an empty accumulator changes nothing
        try a.merge(b)
        let m = try PCA.fit(a, components: k)
        worst.mean = max(worst.mean, gap(m.mean, reference.mean))
        worst.ev = max(worst.ev, relativeGap(m.explainedVariance, reference.explainedVariance))
        worst.comps = max(worst.comps, gap(m.components, reference.components))
        print("embedding streamed tile-order invariance, worst gaps (mean, variance rel, components): \(worst)")
        #expect(worst.mean <= 1e-14)
        #expect(worst.ev <= 1e-13)
        #expect(worst.comps <= 1e-11)
        // Sign resolution is independent of block order too, ties included.
        let r1 = try resolvedSigns(reference, data, blocks: cuts(n, 9))
        let r2 = try resolvedSigns(reference, data, blocks: cuts(n, 9).reversed())
        #expect(r1 == r2)
    }

    @Test func streamedPCAOnOffsetDataKeepsRelativeAccuracy() throws {
        // 3000 rows × 12 columns around a mean of 1e6: a sum-of-squares covariance would lose about ten digits here.
        let n = 3000, d = 12
        var g = SplitMix(seed: 42)
        var data = [Double](repeating: 0, count: n * d)
        for i in 0..<n {
            let a = g.gaussian(), b = g.gaussian()
            for j in 0..<d { data[i * d + j] = 1e6 + 5 * a * Double(j % 3 + 1) + 2 * b * Double(j % 5) + 0.3 * g.gaussian() }
        }
        let k = 3
        let memory = try PCA.fit(data, rows: n, columns: d, components: k)
        let blocks = cuts(n, 333)
        let streamed = try PCA.fit(try stream(data, columns: d, blocks: blocks.reversed()), components: k)
        let r = try resolvedSigns(streamed, data, blocks: blocks)
        let ev = relativeGap(r.explainedVariance, memory.explainedVariance)
        let cg = gap(r.components, memory.components)
        let sg = relativeGap(r.explainedVarianceRatio, memory.explainedVarianceRatio)
        print("embedding streamed offset data: variance rel \(ev), components \(cg), ratio \(sg)")
        #expect(ev <= 1e-10)
        #expect(cg <= 1e-8)
        #expect(sg <= 1e-10)
    }

    @Test func streamedPCARefusesBadInput() throws {
        #expect(throws: EmbeddingError.self) { try PCACovarianceAccumulator(columns: 0) }
        var acc = try PCACovarianceAccumulator(columns: 2)
        #expect(throws: EmbeddingError.self) { try acc.add([1, 2, 3] as [Double], rows: 2) }
        #expect(throws: EmbeddingError.nonFinite) { try acc.add([1, .nan] as [Double], rows: 1) }
        try acc.add([Double](), rows: 0)
        #expect(acc.count == 0)
        #expect(throws: EmbeddingError.self) { try acc.covariance() }
        #expect(throws: EmbeddingError.self) { try acc.merge(PCACovarianceAccumulator(columns: 3)) }
        #expect(throws: EmbeddingError.self) { try PCA.fit(covariance: [1, 0, 0, 1], count: 1, mean: [0, 0], components: 1) }
        #expect(throws: EmbeddingError.self) { try PCA.fit(covariance: [1, 0, 0, 1], count: 5, mean: [0, 0], components: 3) }
        #expect(throws: EmbeddingError.self) { try PCA.fit(covariance: [1, 0, 0], count: 5, mean: [0, 0], components: 1) }
        // Float rows give the same accumulator as the same values in Double.
        try acc.add([1, 2, 3, 5, -1, 0.5] as [Float], rows: 3)
        var viaDouble = try PCACovarianceAccumulator(columns: 2)
        try viaDouble.add([1, 2, 3, 5, -1, 0.5] as [Double], rows: 3)
        #expect(acc == viaDouble)
        // The planted case: variance only on column 0 (as pcaRecoversPlantedAxis), streamed.
        var planted = try PCACovarianceAccumulator(columns: 2)
        try planted.add([-3, 0, -1, 0] as [Double], rows: 2)
        try planted.add([1, 0, 4, 0] as [Double], rows: 2)
        let r = try PCA.fit(planted, components: 1)
        close("streamed planted axis", r.components, [1, 0], 1e-14)
        close("streamed planted variance", r.explainedVariance, [107.0 / 12.0], 1e-13)
    }

    // MARK: vDSP binning (0.6.0)

    @Test(arguments: [("appEmbedA", 12, 10, 4), ("appEmbedB", 5, 6, 8), ("appEmbedC", 16, 16, 16)])
    func acceleratedBinnedVectorMatchesScalar(_ c: (String, Int, Int, Int)) throws {
        let (name, qy, qx, b) = c
        let pattern = try array(name + ".in").map { Float($0) }
        let s = try DiffractionFeatures.binnedVector(pattern: pattern, rows: qy, columns: qx, binnedSize: b)
        let v = try DiffractionFeatures.binnedVector(pattern: pattern, rows: qy, columns: qx, binnedSize: b,
                                                      implementation: .accelerate)
        let p = try pattern.withUnsafeBufferPointer {
            try DiffractionFeatures.binnedVector(pattern: $0, rows: qy, columns: qx, binnedSize: b, implementation: .accelerate)
        }
        let g = gap(v.map(Double.init), s.map(Double.init))
        let scale = Double(max(1, s.map(abs).max() ?? 1))
        print("embedding \(name) accelerated vs scalar binned vector gap: \(g), relative to \(scale), tolerance: 1e-6")
        #expect(g / scale <= 1e-6)
        #expect(p == v)
    }

    @Test func acceleratedBinnedVectorOnRandomAndSpecialPatterns() throws {
        var g = SplitMix(seed: 7)
        var worst = 0.0
        for (qy, qx, b) in [(64, 64, 16), (130, 127, 32), (50, 50, 64), (7, 300, 8)] {
            let pattern = (0..<(qy * qx)).map { _ in Float(pow(10, 4 * g.uniform()) - 1) * (g.uniform() < 0.1 ? -1 : 1) }
            let s = try DiffractionFeatures.binnedVector(pattern: pattern, rows: qy, columns: qx, binnedSize: b)
            let v = try DiffractionFeatures.binnedVector(pattern: pattern, rows: qy, columns: qx, binnedSize: b,
                                                          implementation: .accelerate)
            let scale = max(1, s.map(abs).max() ?? 1)
            worst = max(worst, gap(v.map(Double.init), s.map(Double.init)) / Double(scale))
        }
        print("embedding accelerated vs scalar binned vector, random patterns, worst relative gap: \(worst), tolerance: 1e-6")
        #expect(worst <= 1e-6)
        // Exact cases: all zeros, a single bright pixel (log1p scaled to 1, a box sum of one 1 and zeros).
        let zeros = [Float](repeating: 0, count: 64)
        #expect(try DiffractionFeatures.binnedVector(pattern: zeros, rows: 8, columns: 8, binnedSize: 4, implementation: .accelerate)
                == DiffractionFeatures.binnedVector(pattern: zeros, rows: 8, columns: 8, binnedSize: 4))
        var one = zeros; one[27] = 1000
        let a = try DiffractionFeatures.binnedVector(pattern: one, rows: 8, columns: 8, binnedSize: 4, implementation: .accelerate)
        #expect(a == (try DiffractionFeatures.binnedVector(pattern: one, rows: 8, columns: 8, binnedSize: 4)))
        #expect(a.reduce(0, +) == 1)
        // NaN falls back to the scalar loop and is kept; -inf counts as zero.
        var bad = zeros; bad[3] = .nan; bad[9] = -.infinity; bad[20] = 5
        let sb = try DiffractionFeatures.binnedVector(pattern: bad, rows: 8, columns: 8, binnedSize: 4)
        let ab = try DiffractionFeatures.binnedVector(pattern: bad, rows: 8, columns: 8, binnedSize: 4, implementation: .accelerate)
        #expect(zip(sb, ab).allSatisfy { $0.isNaN ? $1.isNaN : $0 == $1 })
        #expect(throws: EmbeddingError.self) {
            try DiffractionFeatures.binnedVector(pattern: [1, 2, 3], rows: 2, columns: 2, binnedSize: 2, implementation: .accelerate)
        }
    }

    // MARK: cancellation (0.6.0)

    @Test func kMeansCancellationThrowsAndDefaultsAreUnchanged() throws {
        let data = try array("appKm3.in")
        let start = Array(data[0..<9])
        var asked = 0
        #expect(throws: CancellationError.self) {
            try KMeans.fit(data, rows: 90, columns: 3, initialCentroids: start, isCancelled: { asked += 1; return asked > 1 })
        }
        #expect(asked == 2)   // one iteration ran, the second was refused at its start
        #expect(throws: CancellationError.self) {
            try KMeans.fit(data, rows: 90, columns: 3, initialCentroids: start, isCancelled: { true })
        }
        asked = 0
        #expect(throws: CancellationError.self) {
            try KMeans.fitSeeded(data, rows: 90, columns: 3, groups: 3, seed: 1, isCancelled: { asked += 1; return asked > 3 })
        }
        let plain = try KMeans.fitSeeded(data, rows: 90, columns: 3, groups: 3, seed: 1)
        let never = try KMeans.fitSeeded(data, rows: 90, columns: 3, groups: 3, seed: 1, isCancelled: { false })
        #expect(plain == never)
        let a = try KMeans.fit(data, rows: 90, columns: 3, initialCentroids: start)
        let b = try KMeans.fit(data, rows: 90, columns: 3, initialCentroids: start, isCancelled: { false })
        #expect(a == b)
    }
}

/// A small deterministic generator for the tests' random data.
private struct SplitMix {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func uniform() -> Double { Double(next() >> 11) / Double(1 << 53) }
    mutating func gaussian() -> Double {
        let u = max(uniform(), 1e-300), v = uniform()
        return (-2 * log(u)).squareRoot() * cos(2 * .pi * v)
    }
}

/// `PCA.project`: the Float32 vDSP projection the app used. Tolerance: against a Double transcription of the sum,
/// `dims · 6e-8 · Σ|centred · basis|` per coordinate (Float32 rounding, vDSP's order); centring is exact.
@Suite struct ProjectFloatTests {
    @Test(arguments: [(32, 4, 50), (1024, 8, 20), (7, 3, 11)])
    func matchesDoubleSum(_ c: (Int, Int, Int)) {
        let (dims, k, n) = c
        var state: UInt64 = UInt64(dims * 31 + n)
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int64(bitPattern: state >> 11) % 20_000) / 10_000 - 1
        }
        let vectors = (0..<n * dims).map { _ in next() * 5 + 3 }
        let mean = (0..<dims).map { _ in next() }
        let basis = (0..<k * dims).map { _ in next() }
        var centred = [Float](repeating: 0, count: dims)
        var out = [Float](repeating: 0, count: n * k)
        for i in 0..<n {
            vectors.withUnsafeBufferPointer { v in
                out.withUnsafeMutableBufferPointer { o in
                    PCA.project(v.baseAddress! + i * dims, mean: mean, basis: basis, dims: dims, components: k,
                                centred: &centred, into: o.baseAddress! + i * k)
                }
            }
            for j in 0..<dims { #expect(centred[j] == vectors[i * dims + j] - mean[j]) }
        }
        var worst = 0.0
        for i in 0..<n {
            for c in 0..<k {
                var exact = 0.0, mass = 0.0
                for j in 0..<dims {
                    let t = Double(vectors[i * dims + j] - mean[j]) * Double(basis[c * dims + j])
                    exact += t; mass += abs(t)
                }
                let gap = abs(Double(out[i * k + c]) - exact)
                worst = max(worst, gap / max(mass, 1e-30))
                #expect(gap <= Double(dims) * 6e-8 * mass + 1e-30)
            }
        }
        print("PCA.project dims \(dims): worst gap / mass \(worst), tolerance \(Double(dims) * 6e-8)")
    }
}
