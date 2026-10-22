import Foundation
import Testing
import CuboEmbedding

// Pins from py4DSTEM 0.14.19 / scikit-learn 1.9.1 / NumPy 2.5.4 (tools/classification/pins.py). Tolerance: np.allclose defaults
// (rtol 1e-5, atol 1e-8), the default of the upstream tests for these functions (they state none); the measured worst ratio of the gap to
// that bound is printed for each comparison.

private struct RandomStatePin: Decodable {
    let sample: [Double], randint: [Int], normal: [Double], choice: [Int], highSeed: [Double], randint2: [Int], long: [Double]
}
private struct NMFPin: Decodable {
    struct Init: Decodable { let kind: String; let seed: UInt32?; let w: [Double]?; let h: [Double]? }
    let name: String, rows: Int, columns: Int, components: Int, x: [Double], initialization: Init, maxIter: Int, tol: Double
    let w: [Double], h: [Double], error: Double, nIter: Int, converged: Bool
}
private struct ScoresPin: Decodable { let rows: Int, columns: Int, components: Int, x: [Double], scores: [Double] }
private struct PreprocessingPin: Decodable {
    let rows: Int, columns: Int, x: [Double], minMax: [Double], shiftPositive: [Double], reductions: [String: [Double]]
}
private struct ModelPin: Decodable { let rows: Int, columns: Int, values: [Double] }
private struct FeatNMFPin: Decodable {
    struct Args: Decodable {
        let max_components: Int, num_models: Int
        let merge_thresh: Double?, max_iterations: Int?, random_seed: UInt32?, save_all_models: Bool?
    }
    let name: String, error: String?, rows: Int?, columns: Int?, scan: [Int]?, x: [Double]?, args: Args?, models: [ModelPin]?
}
private struct FitPin: Decodable {
    let covarianceType: String, components: Int, labels: [Int], proba: [Double], weights: [Double], means: [Double]
    let covariances: [Double], precisionsCholesky: [Double], bic: Double, nIter: Int, lowerBound: Double, converged: Bool
}
private struct GMMPin: Decodable {
    let name: String, rows: Int, columns: Int, scan: [Int], x: [Double], numModels: Int, seed: UInt32?
    let covarianceTypes: [String], components: [Int], fits: [FitPin]
}
private struct ClassPatternPin: Decodable {
    let positions: Int, pixels: Int, patterns: [Double], weights: [Double], classes: Int, threshold: Double, expected: [[Double]]
}
private struct Pins: Decodable {
    let randomState: RandomStatePin, nmf: [NMFPin], pca: ScoresPin, pcaWide: ScoresPin, preprocessing: PreprocessingPin
    let featurizationNMF: [FeatNMFPin], gmm: [GMMPin], classPatterns: ClassPatternPin
}
private let pins: Pins = {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/featurization.json")
    return try! JSONDecoder().decode(Pins.self, from: Data(contentsOf: url))
}()

/// np.allclose(got, want) with the defaults; returns the worst |got − want| / (atol + rtol·|want|) (at most 1 is agreement).
@discardableResult
private func allclose(_ label: String, _ got: [Double], _ want: [Double], rtol: Double = 1e-5, atol: Double = 1e-8) -> Double {
    #expect(got.count == want.count, "\(label): \(got.count) values against \(want.count)")
    var worst = 0.0
    for (g, w) in zip(got, want) {
        if g.isNaN && w.isNaN { continue }
        worst = max(worst, abs(g - w) / (atol + rtol * abs(w)))
    }
    print("featurization \(label): worst gap / allclose bound = \(worst)")
    #expect(worst <= 1, "\(label)")
    return worst
}

@Suite struct FeaturizationTests {
    // MARK: NumPy RandomState (exact)

    @Test func randomStateDrawsMatchNumPy() {
        var r = NumPyRandomState(seed: 12345)
        #expect((0..<5).map { _ in r.randomSample() } == pins.randomState.sample)
        r = NumPyRandomState(seed: 42)
        #expect((0..<12).map { _ in r.randint(5000) } == pins.randomState.randint)
        r = NumPyRandomState(seed: 7)
        #expect((0..<9).map { _ in r.standardNormal() } == pins.randomState.normal)
        r = NumPyRandomState(seed: 99)
        let p = [Double](repeating: 1.0 / 17.0, count: 17)
        #expect((0..<6).map { _ in r.choice(probabilities: p) } == pins.randomState.choice)
        r = NumPyRandomState(seed: UInt32.max)
        #expect((0..<3).map { _ in r.randomSample() } == pins.randomState.highSeed)
        r = NumPyRandomState(seed: 0)
        let drawn = [r.randint(1), r.randint(1)] + (0..<4).map { _ in r.randint(2) } + [r.randint(1 << 32), r.randint(1 << 32)]
        #expect(drawn == pins.randomState.randint2)
        r = NumPyRandomState(seed: 31337)
        let long = (0..<400).map { _ in r.randomSample() }
        #expect(Array(long.suffix(3)) == pins.randomState.long)
    }

    // MARK: NMF

    private func initialisation(_ pin: NMFPin) -> NMF.Initialization {
        switch pin.initialization.kind {
        case "automatic": return .automatic(seed: pin.initialization.seed ?? 0)
        case "custom": return .custom(w: pin.initialization.w!, h: pin.initialization.h!)
        case "nndsvd": return .nndsvd
        case "nndsvda": return .nndsvda
        case "nndsvdar": return .nndsvdar(seed: pin.initialization.seed!)
        default: return .random(seed: pin.initialization.seed!)
        }
    }

    @Test func nmfMatchesScikitLearn() throws {
        for pin in pins.nmf {
            let r = try NMF.fit(pin.x, rows: pin.rows, columns: pin.columns, components: pin.components,
                                initialization: initialisation(pin), tolerance: pin.tol, maxIterations: pin.maxIter)
            #expect(r.iterations == pin.nIter, "\(pin.name): iterations \(r.iterations) against \(pin.nIter)")
            #expect(r.converged == pin.converged, "\(pin.name) converged")
            allclose("NMF \(pin.name) W", r.w, pin.w)
            allclose("NMF \(pin.name) H", r.h, pin.h)
            allclose("NMF \(pin.name) error", [r.reconstructionError], [pin.error])
        }
    }

    @Test func nmfRecoversAPlantedFactorisation() throws {
        // Exactly rank 2 and non-negative: the factorisation reproduces X.
        let w: [Double] = [1, 0, 0, 2, 1, 1, 3, 0.5, 0, 1, 2, 0]
        let h: [Double] = [1, 2, 0, 0, 1, 0, 0, 1, 3, 1]
        var x = [Double](repeating: 0, count: 6 * 5)
        for i in 0..<6 { for j in 0..<5 { for c in 0..<2 { x[i * 5 + j] += w[i * 2 + c] * h[c * 5 + j] } } }
        let r = try NMF.fit(x, rows: 6, columns: 5, components: 2, tolerance: 1e-12, maxIterations: 5000)
        #expect(r.reconstructionError < 1e-4 * x.reduce(0, +))
        #expect(r.w.allSatisfy { $0 >= 0 } && r.h.allSatisfy { $0 >= 0 })
    }

    @Test func nmfRefusesBadInput() {
        #expect(throws: FeaturizationError.negativeValues) {
            _ = try NMF.fit([1, -1, 2, 3], rows: 2, columns: 2, components: 1)
        }
        #expect(throws: FeaturizationError.nonFinite) {
            _ = try NMF.fit([1, .nan, 2, 3], rows: 2, columns: 2, components: 1)
        }
        #expect(throws: FeaturizationError.self) {
            _ = try NMF.fit([1, 2, 3], rows: 2, columns: 2, components: 1)
        }
        #expect(throws: FeaturizationError.self) {
            _ = try NMF.fit([1, 2, 3, 4], rows: 2, columns: 2, components: 2, initialization: .nndsvd, maxIterations: 0)
        }
    }

    @Test func nmfCancels() {
        let x = pins.nmf[0].x
        #expect(throws: CancellationError.self) {
            _ = try NMF.fit(x, rows: 30, columns: 10, components: 3, isCancelled: { true })
        }
    }

    // MARK: Featurization.PCA and preprocessing

    @Test func pcaMatchesScikitLearn() throws {
        for (label, pin) in [("PCA", pins.pca), ("PCA wide", pins.pcaWide)] {
            let f = try Featurization(features: FeatureMatrix(rows: pin.rows, columns: pin.columns, values: pin.x), scanRows: pin.rows, scanColumns: 1)
            let r = try f.pca(components: pin.components)
            allclose(label + " scores", r.scores, pin.scores)
        }
    }

    @Test func preprocessingMatchesScikitLearn() throws {
        let pin = pins.preprocessing
        let f = try Featurization(features: FeatureMatrix(rows: pin.rows, columns: pin.columns, values: pin.x), scanRows: 3, scanColumns: 4)
        allclose("MinMaxScaler", f.minMaxScaled().features.values, pin.minMax)
        allclose("shift_positive", f.shiftedPositive().features.values, pin.shiftPositive)
        allclose("mean_feature", try f.reduceFeatures([0, 3, 4], with: .mean).features.values, pin.reductions["mean_feature"]!)
        allclose("median_feature", try f.reduceFeatures([0, 3, 4], with: .median).features.values, pin.reductions["median_feature"]!)
        allclose("median_feature two", try f.reduceFeatures([1, 2], with: .median).features.values, pin.reductions["median_two"]!)
        allclose("max_feature", try f.reduceFeatures([0, 3, 4], with: .maximum).features.values, pin.reductions["max_feature"]!)
        let both = try Featurization.concatenate([f, f.shiftedPositive()])
        #expect(both.featureCount == 10 && both.features[2, 7] == f.shiftedPositive().features[2, 2])
    }

    // MARK: Featurization.NMF

    @Test func iterativeNMFMatchesPy4DSTEM() throws {
        var compared = 0
        for pin in pins.featurizationNMF {
            let f = try Featurization(features: FeatureMatrix(rows: pin.rows ?? 36, columns: pin.columns ?? 14, values: pin.x ?? []),
                                      scanRows: 6, scanColumns: 6)
            if pin.error != nil {
                #expect(throws: FeaturizationError.self, "\(pin.name) is an error upstream") {
                    _ = try f.nmf(maxComponents: pin.args!.max_components, numModels: pin.args!.num_models,
                                  maxIterations: pin.args!.max_iterations ?? 1)
                }
                continue
            }
            let a = pin.args!
            let r = try f.nmf(maxComponents: a.max_components, numModels: a.num_models, mergeThreshold: a.merge_thresh ?? 1,
                              maxIterations: a.max_iterations ?? 1, randomSeed: a.random_seed, saveAllModels: a.save_all_models ?? true)
            #expect(r.models.count == pin.models!.count, "\(pin.name) model count")
            for (m, want) in zip(r.models, pin.models!) {
                #expect(m.rows == want.rows && m.columns == want.columns, "\(pin.name) shape \(m.columns) against \(want.columns)")
                if m.columns == want.columns { allclose("Featurization.NMF \(pin.name)", m.values, want.values) }
                compared += 1
            }
        }
        #expect(compared >= 14)
    }

    // MARK: Featurization.GMM

    private func type(_ s: String) -> GaussianMixture.CovarianceType { .init(rawValue: s)! }

    @Test func gaussianMixturesMatchScikitLearn() throws {
        for pin in pins.gmm {
            let f = try Featurization(features: FeatureMatrix(rows: pin.rows, columns: pin.columns, values: pin.x),
                                      scanRows: pin.scan[0], scanColumns: pin.scan[1])
            let fits = try f.gmm(covarianceTypes: pin.covarianceTypes.map(type), components: pin.components, numModels: pin.numModels,
                                 randomSeed: pin.seed)
            #expect(fits.count == pin.fits.count, "\(pin.name) fit count")
            for (k, (fit, want)) in zip(fits, pin.fits).enumerated() {
                let tag = "GMM \(pin.name)[\(k)] \(want.covarianceType) k=\(want.components)"
                #expect(fit.labels == want.labels, "\(tag) labels")
                #expect(fit.model.iterations == want.nIter, "\(tag) iterations \(fit.model.iterations) against \(want.nIter)")
                #expect(fit.model.converged == want.converged, "\(tag) converged")
                allclose(tag + " weights", fit.model.weights, want.weights)
                allclose(tag + " means", fit.model.means, want.means)
                allclose(tag + " covariances", fit.model.covariances, want.covariances)
                allclose(tag + " precisionsCholesky", fit.model.precisionsCholesky, want.precisionsCholesky)
                allclose(tag + " proba", fit.probabilities, want.proba)
                allclose(tag + " bic", [fit.bic], [want.bic])
                allclose(tag + " lowerBound", [fit.model.lowerBound], [want.lowerBound])
                let again = try fit.model.bic(of: pin.x, rows: pin.rows)
                allclose(tag + " bic(of:)", [again], [want.bic])
            }
        }
    }

    @Test func gaussianMixtureRecoversPlantedClasses() throws {
        // Three well separated blobs of 20: every blob gets one label, whatever the seed.
        var x = [Double]()
        var truth = [Int]()
        var rng = NumPyRandomState(seed: 5)
        for (c, centre) in [[0.0, 0, 0], [8, 8, 0], [-8, 4, 8]].enumerated() {
            for _ in 0..<20 { for j in 0..<3 { x.append(centre[j] + 0.5 * rng.standardNormal()) }; truth.append(c) }
        }
        for type in GaussianMixture.CovarianceType.allCases {
            for seed in [1, 2, 3] as [UInt32] {
                let fit = try GaussianMixture.fit(x, rows: 60, columns: 3, components: 3, covarianceType: type, seed: seed)
                for c in 0..<3 {
                    let labels = Set((0..<60).filter { truth[$0] == c }.map { fit.labels[$0] })
                    #expect(labels.count == 1, "\(type) seed \(seed): blob \(c) split over \(labels)")
                }
                #expect(Set(fit.labels).count == 3)
            }
        }
    }

    @Test func gaussianMixtureRefusesBadInput() {
        #expect(throws: FeaturizationError.self) { _ = try GaussianMixture.fit([1, 2, 3, 4], rows: 2, columns: 2, components: 3, seed: 1) }
        #expect(throws: FeaturizationError.illDefinedCovariance) {
            // a duplicated point pair with reg_covar 0 collapses a component
            _ = try GaussianMixture.fit([0, 0, 0, 0, 5, 5, 5, 5], rows: 4, columns: 2, components: 2, seed: 1, regCovar: 0)
        }
        #expect(throws: CancellationError.self) {
            _ = try GaussianMixture.fit([0, 1, 2, 3, 4, 5, 6, 7], rows: 4, columns: 2, components: 2, seed: 1, isCancelled: { true })
        }
    }

    // MARK: class patterns

    @Test func classPatternsMatchPy4DSTEM() throws {
        let p = pins.classPatterns
        let weights = try FeatureMatrix(rows: p.positions, columns: p.classes, values: p.weights)
        let got = try Featurization.classPatterns(weights: weights, patterns: p.patterns.map { Float($0) }, pixelCount: p.pixels,
                                                  threshold: p.threshold)
        #expect(got.count == p.expected.count)
        for (c, (g, e)) in zip(got, p.expected).enumerated() { allclose("class pattern \(c)", g, e) }
    }
}
