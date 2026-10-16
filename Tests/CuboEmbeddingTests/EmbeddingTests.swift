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
}
