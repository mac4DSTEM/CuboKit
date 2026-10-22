//
//  BraggVectorClassification.swift
//  py4DSTEM's `process/classification/braggvectorclassification.py` and `classutils.get_class_DP`: classify the scan positions by
//  which Bragg peaks they show. The detected peaks are labelled by the nearest of a set of reference points (a Voronoi
//  tessellation of diffraction space), the labels of every position form the data matrix X (peaks x positions), initial classes
//  come from the co-occurrence of labels, and the classes `X ≈ W·H` are refined by NMF, split by connected region in real
//  space and merged by correlation. Rows and columns, not x and y: py4DSTEM's `qx`, `R_Nx` are rows here.
//
//  Like py4DSTEM the refinements (`nmf`, `split`, `merge`, `removeClass`, ...) leave the classes untouched and write a candidate
//  (`W_next`, `H_next`) that `accept()` makes current or `reject()` discards. The classification is a value: copy it to keep a
//  state. py4DSTEM's own `merge()` has an empty body in 0.14.19 (the docstring describes `merge_iterative`); `merge` here does what
//  `merge_iterative` does, one round at a time if asked.
//

import CuboCore
import Foundation

/// The classification of scan positions by their Bragg peaks.
public struct BraggVectorClassification: Sendable, Equatable {
    public let scanRows: Int
    public let scanColumns: Int
    /// The number of reference points (py4DSTEM's `N_feat`).
    public let featureCount: Int
    /// `scanRows · scanColumns` (py4DSTEM's `N_meas`).
    public var positionCount: Int { scanRows * scanColumns }
    /// For every scan position (row-major) the sorted reference-point labels of its peaks (`braggpeak_labels`).
    public let peakLabels: [[Int]]
    /// The data matrix X, `featureCount × positionCount`, row-major.
    public let data: [Double]

    /// The number of classes (`N_c`); 0 until the initial classes are set.
    public private(set) var classCount = 0
    /// The Bragg peak weights `W`, `featureCount × classCount`, row-major.
    public private(set) var w: [Double] = []
    /// The scan position weights `H`, `classCount × positionCount`, row-major.
    public private(set) var h: [Double] = []

    /// A set of refined classes waiting to be accepted or rejected (`W_next`, `H_next`, `N_c_next`).
    public struct Candidate: Sendable, Equatable {
        public let classCount: Int
        public let w: [Double]
        public let h: [Double]
    }
    public private(set) var candidate: Candidate?

    // MARK: labels and the data matrix

    /// `get_braggpeak_labels_by_scan_position`: for each scan position the sorted set of labels, where a peak's label is the
    /// index of the nearest reference point (the first on a tie). With `maximumDistance`, a peak further than that (strictly
    /// less is kept) from its nearest point is ignored.
    public static func peakLabels(vectors: BraggVectors, referenceRows: [Double], referenceColumns: [Double],
                                  maximumDistance: Double? = nil) throws -> [[Int]] {
        guard referenceRows.count == referenceColumns.count, !referenceRows.isEmpty else {
            throw CuboBraggError.shapeMismatch("the reference rows and columns must have the same, non-zero length")
        }
        return vectors.peaks.map { peaks in
            var labels = Set<Int>()
            for peak in peaks {
                var best = 0
                var bestDistance = Double.infinity
                for i in referenceRows.indices {
                    let d = hypot(referenceRows[i] - peak.row, referenceColumns[i] - peak.column)
                    if d.isNaN { best = i; bestDistance = d; break }       // np.argmin returns the first NaN
                    if d < bestDistance { bestDistance = d; best = i }
                }
                if let maximumDistance {
                    if hypot(referenceRows[best] - peak.row, referenceColumns[best] - peak.column) < maximumDistance {
                        labels.insert(best)
                    }
                } else {
                    labels.insert(best)
                }
            }
            return labels.sorted()
        }
    }

    /// `BraggVectorClassification(braggpeaks, Qx, Qy, X_is_boolean, max_dist)`: labels the peaks of every position by the nearest
    /// reference point and builds X. With `booleanData` X is 1 where a label is present; otherwise it holds the intensity of
    /// the peak nearest to the reference point (at the positions where the label is present).
    public init(vectors: BraggVectors, referenceRows: [Double], referenceColumns: [Double], booleanData: Bool = true,
                maximumDistance: Double? = nil) throws {
        let labels = try Self.peakLabels(vectors: vectors, referenceRows: referenceRows, referenceColumns: referenceColumns,
                                         maximumDistance: maximumDistance)
        let features = referenceRows.count, positions = vectors.scanRows * vectors.scanColumns
        var x = [Double](repeating: 0, count: features * positions)
        for position in 0..<positions {
            let peaks = vectors.peaks[position]
            for i in labels[position] {
                if booleanData {
                    x[i * positions + position] = 1
                } else {
                    var best = 0
                    var bestDistance = Double.infinity
                    for (index, peak) in peaks.enumerated() {
                        let d = hypot(peak.row - referenceRows[i], peak.column - referenceColumns[i])
                        if d.isNaN { best = index; break }
                        if d < bestDistance { bestDistance = d; best = index }
                    }
                    x[i * positions + position] = peaks[best].intensity
                }
            }
        }
        self.scanRows = vectors.scanRows
        self.scanColumns = vectors.scanColumns
        self.featureCount = features
        self.peakLabels = labels
        self.data = x
    }

    // MARK: initial classes

    /// `get_initial_classes`: sets of reference points that tend to appear together. The n-point function (n = 2 or 3, the
    /// probability that n labels share a position) is searched for its maximum, which seeds a class; the class grows by the label
    /// most frequent among the positions that hold all its members as long as that frequency exceeds `threshold`; the seed's
    /// slices are then zeroed and the search repeats, until a class repeats or sits inside an earlier one, fewer than
    /// `fractionThreshold` of the labels are unused, or more than `maximumIterations` rounds were made. Each class is returned
    /// as an ascending array of labels, in the order found.
    ///
    /// `isCancelled` is asked at the start of every round; when it answers true the call throws `CancellationError`.
    public static func initialClasses(peakLabels: [[Int]], featureCount n: Int, threshold: Double = 0.3,
                                      fractionThreshold: Double = 0.1, maximumIterations: Int = 200, correlationOrder: Int = 2,
                                      isCancelled: (@Sendable () -> Bool)? = nil) throws -> [[Int]] {
        guard n > 0, !peakLabels.isEmpty else { throw CuboBraggError.shapeMismatch("no labels or no reference points") }
        guard (0.0...1.0).contains(threshold), (0.0...1.0).contains(fractionThreshold), maximumIterations >= 0 else {
            throw CuboBraggError.invalidOption("threshold and fractionThreshold must lie in [0, 1] and maximumIterations must not be negative")
        }
        guard correlationOrder == 2 || correlationOrder == 3 else {
            throw CuboBraggError.invalidOption("correlationOrder must be 2 or 3")
        }
        guard peakLabels.allSatisfy({ $0.allSatisfy { (0..<n).contains($0) } }) else {
            throw CuboBraggError.invalidOption("a label lies outside 0..<\(n)")
        }
        let (squared, o1) = n.multipliedReportingOverflow(by: n)
        let (cubed, o2) = squared.multipliedReportingOverflow(by: n)
        guard correlationOrder == 2 ? (!o1 && squared <= 100_000_000) : (!o1 && !o2 && cubed <= 100_000_000) else {
            throw CuboBraggError.invalidOption("featureCount \(n) is too large for correlationOrder \(correlationOrder)")
        }
        let sets = peakLabels.map { Set($0) }
        let positions = Double(sets.count)
        var function = [Double](repeating: 0, count: correlationOrder == 2 ? n * n : n * n * n)
        for s in peakLabels {
            if correlationOrder == 2 {
                for a in s { for b in s where a != b { function[a * n + b] += 1 } }
            } else {
                for a in s { for b in s where b != a { for c in s where c != a && c != b { function[(a * n + b) * n + c] += 1 } } }
            }
        }
        for i in function.indices { function[i] /= positions }

        var classes = [Set<Int>]()
        var iteration = 0
        var unused = [Bool](repeating: true, count: n)
        var seeding = true
        while seeding {
            if isCancelled?() == true { throw CancellationError() }
            var top = 0
            for i in function.indices where function[i] > function[top] { top = i }
            var members: Set<Int>
            if correlationOrder == 2 {
                members = [top / n, top % n]
            } else {
                members = [top / (n * n), (top / n) % n, top % n]
            }
            while true {
                var frequencies = [Double](repeating: 0, count: n)
                var count = 0
                for s in sets where members.isSubset(of: s) {
                    count += 1
                    for i in s { frequencies[i] += 1 }
                }
                if count == 0 { break }              // 0/0: every entry NaN, no frequency exceeds the threshold
                for i in 0..<n { frequencies[i] /= Double(count) }
                for i in members { frequencies[i] = 0 }
                var next = 0
                for i in 1..<max(1, n) where frequencies[i] > frequencies[next] { next = i }
                if frequencies[next] > threshold { members.insert(next) } else { break }
            }
            for i in members {
                if correlationOrder == 2 {
                    for j in 0..<n { function[i * n + j] = 0; function[j * n + i] = 0 }
                } else {
                    for a in 0..<n { for b in 0..<n {
                        function[(i * n + a) * n + b] = 0
                        function[(a * n + i) * n + b] = 0
                        function[(a * n + b) * n + i] = 0
                    } }
                }
                unused[i] = false
            }
            for s in classes where members.isSubset(of: s) { seeding = false }
            if seeding { classes.append(members) }
            iteration += 1
            let unusedCount = unused.filter { $0 }.count
            if iteration > maximumIterations || Double(unusedCount) < Double(n) * fractionThreshold { seeding = false }
        }
        return classes.map { $0.sorted() }
    }

    /// `get_initial_classes_by_cooccurrence`: sets `W` to the 0/1 membership of the classes of `initialClasses` and `H` to the
    /// minimum-norm least-squares fit of X by them, negative values set to 0. Discards any candidate.
    public mutating func setInitialClassesByCooccurrence(threshold: Double = 0.3, fractionThreshold: Double = 0.1,
                                                         maximumIterations: Int = 200, correlationOrder: Int = 2,
                                                         isCancelled: (@Sendable () -> Bool)? = nil) throws {
        let sets = try Self.initialClasses(peakLabels: peakLabels, featureCount: featureCount, threshold: threshold,
                                           fractionThreshold: fractionThreshold, maximumIterations: maximumIterations,
                                           correlationOrder: correlationOrder, isCancelled: isCancelled)
        let k = sets.count
        var w = [Double](repeating: 0, count: featureCount * k)
        for (c, set) in sets.enumerated() { for j in set { w[j * k + c] = 1 } }
        var h = ClassificationNumerics.leastSquares(w, featureCount, k, data, positionCount)
        for i in h.indices where h[i] < 0 { h[i] = 0 }
        self.classCount = k
        self.w = w
        self.h = h
        self.candidate = nil
    }

    /// `get_initial_classes_from_images`: `H` is the given class images (`scanRows · scanColumns` values each, row-major scan
    /// order) and `W` the minimum-norm least-squares fit of X by them, negative values set to 0. Discards any candidate.
    public mutating func setInitialClasses(images: [[Double]]) throws {
        guard !images.isEmpty, images.allSatisfy({ $0.count == positionCount }) else {
            throw CuboBraggError.shapeMismatch("each class image must hold \(scanRows) x \(scanColumns) values")
        }
        let k = images.count
        let h = images.flatMap { $0 }
        // W = lstsq(H.T, X.T).T: (positions x k) against (positions x features) gives k x features
        var ht = [Double](repeating: 0, count: positionCount * k)
        for c in 0..<k { for p in 0..<positionCount { ht[p * k + c] = h[c * positionCount + p] } }
        var xt = [Double](repeating: 0, count: positionCount * featureCount)
        for f in 0..<featureCount { for p in 0..<positionCount { xt[p * featureCount + f] = data[f * positionCount + p] } }
        let solution = ClassificationNumerics.leastSquares(ht, positionCount, k, xt, featureCount)     // k × features
        var w = [Double](repeating: 0, count: featureCount * k)
        for f in 0..<featureCount { for c in 0..<k { w[f * k + c] = max(0, solution[c * featureCount + f]) } }
        self.classCount = k
        self.w = w
        self.h = h
        self.candidate = nil
    }

    /// Sets the classes directly (py4DSTEM lets the attributes `W`, `H` and `N_c` be assigned): `w` is `featureCount × classCount`
    /// and `h` is `classCount × positionCount`, row-major. Discards any candidate.
    public mutating func setClasses(w: [Double], h: [Double], classCount: Int) throws {
        guard classCount >= 1, w.count == featureCount * classCount, h.count == classCount * positionCount else {
            throw CuboBraggError.shapeMismatch("W must be \(featureCount) x \(classCount) and H \(classCount) x \(positionCount)")
        }
        self.classCount = classCount
        self.w = w
        self.h = h
        self.candidate = nil
    }

    // MARK: refinement (writes the candidate)

    private func requireClass(_ i: Int, of count: Int) throws {
        guard (0..<count).contains(i) else { throw CuboBraggError.invalidOption("class \(i) does not exist") }
    }

    private func requireClasses() throws {
        guard classCount > 0 else { throw CuboBraggError.invalidOption("the initial classes have not been set") }
    }

    /// `nmf(max_iterations)`: up to `maximumIterations` sweeps of scikit-learn's coordinate-descent NMF (tolerance 1e-4) from the
    /// current `W` and `H`, X as the data. The result is the candidate.
    ///
    /// `isCancelled` is asked at the start of every sweep; when it answers true the call throws `CancellationError`.
    public mutating func nmf(maximumIterations: Int = 1, isCancelled: (@Sendable () -> Bool)? = nil) throws {
        try requireClasses()
        guard maximumIterations >= 1 else { throw CuboBraggError.invalidOption("maximumIterations must be at least 1") }
        guard data.allSatisfy({ $0.isFinite && $0 >= 0 }), w.allSatisfy({ $0.isFinite && $0 >= 0 }),
              h.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw CuboBraggError.invalidOption("NMF needs finite, non-negative X, W and H")
        }
        let result = try ClassificationNumerics.coordinateDescentNMF(data, featureCount, positionCount, classCount, w, h,
                                                                     maxIterations: maximumIterations, isCancelled: isCancelled)
        candidate = Candidate(classCount: classCount, w: result.w, h: result.h)
    }

    /// The connected regions of class `i` in real space as py4DSTEM's `split` builds them: the class image smoothed by a
    /// Gaussian of `sigma`, thresholded at `thresholdSplit` times its maximum, opened, closed and dilated by `expandMask`, its
    /// 8-connected regions numbered in raster order of their first pixel and each eroded by `expandMask` again; regions of fewer
    /// than `minimumPixels` pixels are left out. Returns one mask (`scanRows · scanColumns`) per region.
    private func regions(ofClass i: Int, sigma: Double, thresholdSplit: Double, expandMask: Int, minimumPixels: Int) -> [[Bool]] {
        let image = Array(h[(i * positionCount)..<((i + 1) * positionCount)])
        let smooth = ClassificationNumerics.gaussianFilter(image, scanRows, scanColumns, sigma: sigma)
        let limit = (smooth.max() ?? 0) * thresholdSplit
        var mask = smooth.map { $0 > limit }
        mask = ClassificationNumerics.morph(mask, scanRows, scanColumns, erode: true, iterations: 1)      // opening
        mask = ClassificationNumerics.morph(mask, scanRows, scanColumns, erode: false, iterations: 1)
        mask = ClassificationNumerics.morph(mask, scanRows, scanColumns, erode: false, iterations: 1)     // closing
        mask = ClassificationNumerics.morph(mask, scanRows, scanColumns, erode: true, iterations: 1)
        mask = ClassificationNumerics.morph(mask, scanRows, scanColumns, erode: false, iterations: expandMask)
        let (labels, count) = ClassificationNumerics.label(mask, scanRows, scanColumns)
        var out = [[Bool]]()
        for j in 0..<count {
            let region = ClassificationNumerics.morph(labels.map { $0 == j + 1 }, scanRows, scanColumns, erode: true,
                                                      iterations: expandMask)
            if region.filter({ $0 }).count >= minimumPixels { out.append(region) }
        }
        return out
    }

    /// The Gaussian kernel must be finite, positive and no wider than four times the scan's longer side.
    private func requireSplitSigma(_ sigma: Double) throws {
        guard sigma.isFinite, sigma > 0, sigma <= 4 * Double(max(scanRows, scanColumns)) else {
            throw CuboBraggError.invalidOption("sigma must be finite, positive and at most 4 x the longer scan side")
        }
    }

    /// `split(sigma, threshold_split, expand_mask, minimum_pixels)`: every class becomes one class per contiguous region of its
    /// smoothed image (see `regions`): the same `W` column, and `H` kept inside the region. A class without a region disappears.
    /// The result is the candidate.
    public mutating func split(sigma: Double = 2, thresholdSplit: Double = 0.25, expandMask: Int = 1,
                               minimumPixels: Int = 1) throws {
        try requireClasses()
        try requireSplitSigma(sigma)
        var wColumns = [[Double]](), hRows = [[Double]]()
        for i in 0..<classCount {
            for region in regions(ofClass: i, sigma: sigma, thresholdSplit: thresholdSplit, expandMask: expandMask,
                                  minimumPixels: minimumPixels) {
                wColumns.append((0..<featureCount).map { w[$0 * classCount + i] })
                hRows.append((0..<positionCount).map { region[$0] ? h[i * positionCount + $0] : 0 })
            }
        }
        candidate = Self.assemble(wColumns, hRows, featureCount, positionCount)
    }

    /// `split_by_class_index(i, ...)`: class `i` is split as in `split` and the regions come first, followed by the other
    /// classes in their order. The result is the candidate.
    public mutating func split(classIndex i: Int, sigma: Double = 2, thresholdSplit: Double = 0.25, expandMask: Int = 1,
                               minimumPixels: Int = 1) throws {
        try requireClasses()
        guard (0..<classCount).contains(i) else { throw CuboBraggError.invalidOption("class \(i) does not exist") }
        try requireSplitSigma(sigma)
        var wColumns = [[Double]](), hRows = [[Double]]()
        for region in regions(ofClass: i, sigma: sigma, thresholdSplit: thresholdSplit, expandMask: expandMask,
                              minimumPixels: minimumPixels) {
            wColumns.append((0..<featureCount).map { w[$0 * classCount + i] })
            hRows.append((0..<positionCount).map { region[$0] ? h[i * positionCount + $0] : 0 })
        }
        for c in 0..<classCount where c != i {
            wColumns.append((0..<featureCount).map { w[$0 * classCount + c] })
            hRows.append(Array(h[(c * positionCount)..<((c + 1) * positionCount)]))
        }
        candidate = Self.assemble(wColumns, hRows, featureCount, positionCount)
    }

    /// `remove_class(i)`: the candidate is the classes without class `i`.
    public mutating func removeClass(_ i: Int) throws {
        try requireClasses()
        guard (0..<classCount).contains(i) else { throw CuboBraggError.invalidOption("class \(i) does not exist") }
        var wColumns = [[Double]](), hRows = [[Double]]()
        for c in 0..<classCount where c != i {
            wColumns.append((0..<featureCount).map { w[$0 * classCount + c] })
            hRows.append(Array(h[(c * positionCount)..<((c + 1) * positionCount)]))
        }
        candidate = Self.assemble(wColumns, hRows, featureCount, positionCount)
    }

    /// `merge_by_class_index(i, j)`: classes `i` and `j` become one: the `H` rows add, the `W` columns average weighted by the
    /// total of each `H` row. As upstream, column `j` is deleted first and the merged class written to position `i` of what is
    /// left, so `i < j` (the order `merge` produces) puts it where class `i` was; for `i > j` it overwrites the class that moved
    /// into place (and `i` beyond the end throws, where NumPy raises `IndexError`). The result is the candidate.
    public mutating func merge(classIndices i: Int, _ j: Int) throws {
        try requireClasses()
        guard (0..<classCount).contains(i), (0..<classCount).contains(j) else {
            throw CuboBraggError.invalidOption("class \(i) or \(j) does not exist")
        }
        var weightI = 0.0, weightJ = 0.0
        for p in 0..<positionCount { weightI += h[i * positionCount + p]; weightJ += h[j * positionCount + p] }
        let wNew = (0..<featureCount).map { (w[$0 * classCount + i] * weightI + w[$0 * classCount + j] * weightJ) / (weightI + weightJ) }
        let hNew = (0..<positionCount).map { h[i * positionCount + $0] + h[j * positionCount + $0] }
        var wColumns = [[Double]](), hRows = [[Double]]()
        for c in 0..<classCount where c != j {
            wColumns.append((0..<featureCount).map { w[$0 * classCount + c] })
            hRows.append(Array(h[(c * positionCount)..<((c + 1) * positionCount)]))
        }
        guard i < wColumns.count else { throw CuboBraggError.invalidOption("class \(i) is beyond the classes left after removing \(j)") }
        wColumns[i] = wNew
        hRows[i] = hNew
        candidate = Self.assemble(wColumns, hRows, featureCount, positionCount)
    }

    /// `merge_iterative(threshBPs, threshScanPosition)` (what the docstring of the empty `merge()` describes): classes whose Bragg
    /// peak weights (columns of `W`) correlate above `thresholdPeaks` and whose real-space weights (rows of `H`) correlate above
    /// `thresholdScan` are merge candidates, taken in order of decreasing product of the two correlations (ties by larger first
    /// index, then larger second). Each round merges the pairs whose classes are not yet merged in that round; as upstream, a
    /// class that appears in a skipped pair is also marked merged and so dropped. Rounds repeat until no candidate pair is left
    /// (`maximumRounds == nil`), or for at most `maximumRounds`. Surviving classes come first, merged ones after. The result is
    /// the candidate.
    ///
    /// `isCancelled` is asked at the start of every round; when it answers true the call throws `CancellationError`.
    public mutating func merge(thresholdPeaks: Double = 0.1, thresholdScan: Double = 0.1, maximumRounds: Int? = nil,
                               isCancelled: (@Sendable () -> Bool)? = nil) throws {
        try requireClasses()
        var wCurrent = w, hCurrent = h
        var classes = classCount
        var rounds = 0
        while true {
            if isCancelled?() == true { throw CancellationError() }
            // upstream's `np.corrcoef` of one class is a 0-d array that its candidate indexing then fails on (IndexError); this
            // happens for a single class to start with and when merging reduces the classes to one
            guard classes >= 2 else { throw CuboBraggError.invalidOption("merging needs at least two classes") }
            if let maximumRounds, rounds >= maximumRounds { break }
            rounds += 1
            var wRows = [Double](repeating: 0, count: classes * featureCount)       // W.T
            for f in 0..<featureCount { for c in 0..<classes { wRows[c * featureCount + f] = wCurrent[f * classes + c] } }
            let wCorrelation = classes > 1
                ? ClassificationNumerics.correlation(wRows, variables: classes, observations: featureCount) : [1.0]
            let hCorrelation = classes > 1
                ? ClassificationNumerics.correlation(hCurrent, variables: classes, observations: positionCount) : [1.0]
            var pairs = [(i: Int, j: Int, score: Double)]()
            for i in 0..<classes { for j in (i + 1)..<max(i + 1, classes)
                where wCorrelation[i * classes + j] > thresholdPeaks && hCorrelation[i * classes + j] > thresholdScan {
                pairs.append((i, j, wCorrelation[i * classes + j] * hCorrelation[i * classes + j]))
            } }
            // np.sort(order="score")[::-1]: ascending by (score, i, j), then reversed
            pairs.sort { ($0.score, $0.i, $0.j) > ($1.score, $1.i, $1.j) }
            var merged = [Bool](repeating: false, count: classes)
            var mergedW = [[Double]](), mergedH = [[Double]]()
            for pair in pairs {
                if !(merged[pair.i] || merged[pair.j]) {
                    var weightI = 0.0, weightJ = 0.0
                    for p in 0..<positionCount { weightI += hCurrent[pair.i * positionCount + p]; weightJ += hCurrent[pair.j * positionCount + p] }
                    mergedW.append((0..<featureCount).map {
                        (wCurrent[$0 * classes + pair.i] * weightI + wCurrent[$0 * classes + pair.j] * weightJ) / (weightI + weightJ)
                    })
                    mergedH.append((0..<positionCount).map { hCurrent[pair.i * positionCount + $0] + hCurrent[pair.j * positionCount + $0] })
                }
                merged[pair.i] = true
                merged[pair.j] = true
            }
            var wColumns = [[Double]](), hRows = [[Double]]()
            for c in 0..<classes where !merged[c] {
                wColumns.append((0..<featureCount).map { wCurrent[$0 * classes + c] })
                hRows.append(Array(hCurrent[(c * positionCount)..<((c + 1) * positionCount)]))
            }
            wColumns += mergedW
            hRows += mergedH
            let next = Self.assemble(wColumns, hRows, featureCount, positionCount)
            wCurrent = next.w
            hCurrent = next.h
            classes = next.classCount
            if pairs.isEmpty { break }
        }
        candidate = Candidate(classCount: classes, w: wCurrent, h: hCurrent)
    }

    private static func assemble(_ wColumns: [[Double]], _ hRows: [[Double]], _ features: Int, _ positions: Int) -> Candidate {
        let k = wColumns.count
        var w = [Double](repeating: 0, count: features * k)
        for c in 0..<k { for f in 0..<features { w[f * k + c] = wColumns[c][f] } }
        return Candidate(classCount: k, w: w, h: hRows.flatMap { $0 })
    }

    // MARK: accept, reject

    /// `accept()`: the candidate becomes the classes (nothing happens without one).
    public mutating func accept() {
        guard let candidate else { return }
        classCount = candidate.classCount
        w = candidate.w
        h = candidate.h
        self.candidate = nil
    }

    /// `reject()`: discards the candidate.
    public mutating func reject() { candidate = nil }

    // MARK: reading the classes

    /// `get_class_BPs(i)`: the Bragg peak weights of class `i` (`featureCount` values).
    ///
    /// Throws `CuboBraggError.invalidOption` when `i` is not a class.
    public func classPeakWeights(_ i: Int) throws -> [Double] {
        try requireClass(i, of: classCount)
        return (0..<featureCount).map { w[$0 * classCount + i] }
    }

    /// `get_class_image(i)`: the scan position weights of class `i`, `scanRows · scanColumns` values in row-major scan order.
    ///
    /// Throws `CuboBraggError.invalidOption` when `i` is not a class.
    public func classImage(_ i: Int) throws -> [Double] {
        try requireClass(i, of: classCount)
        return Array(h[(i * positionCount)..<((i + 1) * positionCount)])
    }

    /// `get_candidate_class_BPs(i)`; nil without a candidate, and `invalidOption` when `i` is not a candidate class.
    public func candidateClassPeakWeights(_ i: Int) throws -> [Double]? {
        guard let c = candidate else { return nil }
        try requireClass(i, of: c.classCount)
        return (0..<featureCount).map { c.w[$0 * c.classCount + i] }
    }

    /// `get_candidate_class_image(i)`; nil without a candidate, and `invalidOption` when `i` is not a candidate class.
    public func candidateClassImage(_ i: Int) throws -> [Double]? {
        guard let c = candidate else { return nil }
        try requireClass(i, of: c.classCount)
        return Array(c.h[(i * positionCount)..<((i + 1) * positionCount)])
    }
}

// MARK: - class diffraction patterns

extension BraggVectorClassification {
    /// `get_class_DP`: the average diffraction pattern of a class described in real space by `classImage`: the patterns of the
    /// positions whose weight is at least `threshold`, weighted by it and divided by the sum of those weights, negative values
    /// (and NaN) set to 0. Row-major `detectorRows · detectorColumns`.
    ///
    /// With `rowShifts` and `columnShifts` (one per position, row-major scan order) each pattern is first moved by minus its shift:
    /// rounded to the nearest integer (ties to even) and rolled when `integerShifts`, otherwise shifted with the Fourier shift
    /// theorem (`get_shifted_ar`, periodic). `darkReference` is subtracted, weighted, from the running sum after each pattern.
    public static func classPattern(of source: some CubeSource, classImage: [Double], threshold: Double = 0.01,
                                    rowShifts: [Double]? = nil, columnShifts: [Double]? = nil, darkReference: [Double]? = nil,
                                    integerShifts: Bool = true,
                                    isCancelled: (@Sendable () -> Bool)? = nil) async throws -> [Double] {
        let shape = source.shape
        guard classImage.count == shape.scanCount else {
            throw CuboBraggError.shapeMismatch("the class image must hold \(shape.scanRows) x \(shape.scanColumns) values")
        }
        if let rowShifts { guard rowShifts.count == shape.scanCount else { throw CuboBraggError.shapeMismatch("rowShifts") } }
        if let columnShifts { guard columnShifts.count == shape.scanCount else { throw CuboBraggError.shapeMismatch("columnShifts") } }
        if let darkReference { guard darkReference.count == shape.detectorCount else { throw CuboBraggError.shapeMismatch("darkReference") } }
        let rows = shape.detectorRows, columns = shape.detectorColumns
        var pattern = [Double](repeating: 0, count: rows * columns)
        var used = [Double]()
        for position in 0..<shape.scanCount {
            if isCancelled?() == true { throw CancellationError() }
            let weight = classImage[position]
            guard weight >= threshold else { continue }
            used.append(weight)
            let raw = try await source.readPattern(row: position / shape.scanColumns, column: position % shape.scanColumns)
            var current = raw.values.map { weight * Double($0) }
            if let rowShifts, let columnShifts {
                let dr = rowShifts[position], dc = columnShifts[position]
                guard dr.isFinite, dc.isFinite else { throw CuboBraggError.invalidOption("a shift is not finite") }
                if integerShifts {
                    current = roll(current, rows, columns, rows: integerShift(dr, modulo: rows), columns: integerShift(dc, modulo: columns))
                } else {
                    current = fourierShift(current, rows, columns, rows: -dr, columns: -dc)
                }
            }
            for i in 0..<pattern.count { pattern[i] += current[i] }
            if let darkReference { for i in 0..<pattern.count { pattern[i] -= darkReference[i] * weight } }
        }
        let total = NumPySum.pairwise(used)             // np.sum(class_image[class_image >= thresh])
        return pattern.map { value in
            let v = value / total
            return v > 0 ? v : 0
        }
    }

    /// `get_class_DP_without_Bragg_scattering`: the class average with a disc of `radius` masked around every detected peak of a
    /// position except the one nearest the optic axis (`originRow`, `originColumn`), the masks being averaged with the same
    /// weights (a pixel never unmasked stays 0). `vectors` are the peaks in the raw detector coordinates, before any shift.
    public static func classPatternWithoutBraggScattering(of source: some CubeSource, classImage: [Double], vectors: BraggVectors,
                                                          radius: Double, originRow: Double, originColumn: Double,
                                                          threshold: Double = 0.01, rowShifts: [Double]? = nil,
                                                          columnShifts: [Double]? = nil, darkReference: [Double]? = nil,
                                                          integerShifts: Bool = true,
                                    isCancelled: (@Sendable () -> Bool)? = nil) async throws -> [Double] {
        let shape = source.shape
        guard classImage.count == shape.scanCount, vectors.scanRows == shape.scanRows, vectors.scanColumns == shape.scanColumns else {
            throw CuboBraggError.shapeMismatch("the class image and the vectors must match the scan \(shape.scanRows) x \(shape.scanColumns)")
        }
        if let rowShifts { guard rowShifts.count == shape.scanCount else { throw CuboBraggError.shapeMismatch("rowShifts") } }
        if let columnShifts { guard columnShifts.count == shape.scanCount else { throw CuboBraggError.shapeMismatch("columnShifts") } }
        if let darkReference { guard darkReference.count == shape.detectorCount else { throw CuboBraggError.shapeMismatch("darkReference") } }
        let rows = shape.detectorRows, columns = shape.detectorColumns
        var pattern = [Double](repeating: 0, count: rows * columns)
        var maskWeights = [Double](repeating: 0, count: rows * columns)
        for position in 0..<shape.scanCount {
            if isCancelled?() == true { throw CancellationError() }
            let weight = classImage[position]
            guard weight >= threshold else { continue }
            let peaks = vectors.peaks[position]
            var mask = [Double](repeating: 1, count: rows * columns)
            if peaks.count > 1 {
                var centre = 0
                var best = Double.infinity
                for (i, p) in peaks.enumerated() {
                    let d = hypot(p.row - originRow, p.column - originColumn)
                    if d < best { best = d; centre = i }
                }
                for (i, p) in peaks.enumerated() where i != centre {
                    for r in 0..<rows { for c in 0..<columns {
                        let dr = Double(r) - p.row, dc = Double(c) - p.column
                        if !((dr * dr + dc * dc) >= radius * radius) { mask[r * columns + c] = 0 }
                    } }
                }
            }
            let raw = try await source.readPattern(row: position / shape.scanColumns, column: position % shape.scanColumns)
            var current = (0..<(rows * columns)).map { Double(raw.values[$0]) * mask[$0] * weight }
            if let rowShifts, let columnShifts {
                let dr = rowShifts[position], dc = columnShifts[position]
                guard dr.isFinite, dc.isFinite else { throw CuboBraggError.invalidOption("a shift is not finite") }
                if integerShifts {
                    let sr = integerShift(dr, modulo: rows), sc = integerShift(dc, modulo: columns)
                    current = roll(current, rows, columns, rows: sr, columns: sc)
                    mask = roll(mask, rows, columns, rows: sr, columns: sc)
                } else {
                    current = fourierShift(current, rows, columns, rows: -dr, columns: -dc)
                    mask = fourierShift(mask, rows, columns, rows: -dr, columns: -dc)
                }
            }
            if let darkReference { for i in 0..<current.count { current[i] -= darkReference[i] * weight } }
            for i in 0..<pattern.count { pattern[i] += current[i]; maskWeights[i] += mask[i] * weight }
        }
        return (0..<pattern.count).map { maskWeights[$0] != 0 ? pattern[$0] / maskWeights[$0] : 0 }
    }

    /// Minus the shift rounded to the nearest integer (ties to even), reduced modulo `n` as a Double first so a huge shift
    /// cannot overflow `Int`; a roll by it equals a roll by the unreduced value. `shift` must be finite.
    private static func integerShift(_ shift: Double, modulo n: Int) -> Int {
        Int((-shift.rounded(.toNearestOrEven)).truncatingRemainder(dividingBy: Double(n)))
    }

    /// `np.roll(np.roll(a, rows, axis=0), columns, axis=1)`.
    private static func roll(_ a: [Double], _ nr: Int, _ nc: Int, rows dr: Int, columns dc: Int) -> [Double] {
        var out = a
        for r in 0..<nr { for c in 0..<nc {
            let rr = ((r + dr) % nr + nr) % nr, cc = ((c + dc) % nc + nc) % nc
            out[rr * nc + cc] = a[r * nc + c]
        } }
        return out
    }

    /// `get_shifted_ar(ar, xshift, yshift)`: periodic Fourier shift, `real(ifft2(fft2(a)·exp(-2πi(yshift·qy + xshift·qx))))` with
    /// `fftfreq` coordinates of spacing 1.
    private static func fourierShift(_ a: [Double], _ nr: Int, _ nc: Int, rows dr: Double, columns dc: Double) -> [Double] {
        var re = a, im = [Double](repeating: 0, count: a.count)
        let plan = FFT2D(rows: nr, columns: nc)
        plan.transform(&re, &im, inverse: false)
        func frequency(_ i: Int, _ n: Int) -> Double { Double(i < (n + 1) / 2 ? i : i - n) / Double(n) }
        for r in 0..<nr { for c in 0..<nc {
            let phase = -(2 * Double.pi) * (dc * frequency(c, nc) + dr * frequency(r, nr))
            let wr = cos(phase), wi = sin(phase)
            let k = r * nc + c
            let x = re[k], y = im[k]
            re[k] = x * wr - y * wi
            im[k] = x * wi + y * wr
        } }
        plan.transform(&re, &im, inverse: true)
        return re
    }
}
