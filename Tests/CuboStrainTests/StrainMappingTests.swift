import CuboBragg
import Foundation
import Testing
@testable import CuboStrain

/// Scan-level strain mapping against py4DSTEM f050d207's real `BraggVectors.histogram` and `StrainMap`
/// (`tools/strain/mapping_pins.py`, fixtures `map_*.f64`, described in `strain_mapping.json`; `docs/strain.md`).
/// The datasets: d1 a noise-free planted lattice, d2 a noisy one with dropped peaks, thin positions, a scan mask and
/// a reference region, d3 peaks measured on the public simulated Au dataset.
@Suite struct StrainMappingTests {
    let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")

    func array(_ name: String) throws -> [Double] {
        let d = try Data(contentsOf: directory.appendingPathComponent(name + ".f64"))
        return d.withUnsafeBytes { raw in
            stride(from: 0, to: raw.count, by: 8).map { Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: $0, as: UInt64.self))) }
        }
    }
    struct Dataset {
        let rows: Int, columns: Int, detectorRows: Int, detectorColumns: Int
        let origin: LatticeVector
        let peaks: [[StrainPeak]]
    }
    func dataset(_ name: String) throws -> Dataset {
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("strain_mapping.json"))) as! [String: Any]
        let d = (meta["datasets"] as! [String: Any])[name] as! [String: Any]
        let scan = d["scan"] as! [Int], det = d["detector"] as! [Int], o = d["origin"] as! [Double]
        let counts = try array("map_\(name)_counts"), flat = try array("map_\(name)_peaks")
        var lists: [[StrainPeak]] = [], at = 0
        for c in counts {
            lists.append((0..<Int(c)).map { k in StrainPeak(row: flat[(at + k) * 3], column: flat[(at + k) * 3 + 1], intensity: flat[(at + k) * 3 + 2]) })
            at += Int(c)
        }
        #expect(at * 3 == flat.count)
        return .init(rows: scan[0], columns: scan[1], detectorRows: det[0], detectorColumns: det[1],
                     origin: .init(row: o[0], column: o[1]), peaks: lists)
    }
    func bools(_ a: [Double]) -> [Bool] { a.map { $0 != 0 } }

    /// Largest absolute difference, NaN-aware (NaN against NaN is no difference, NaN against a number is infinite).
    func gap(_ a: [Double], _ b: [Double]) -> Double {
        #expect(a.count == b.count)
        return zip(a, b).map { x, y in x.isNaN && y.isNaN ? 0 : (x.isNaN || y.isNaN ? .infinity : abs(x - y)) }.max() ?? 0
    }
    func check(_ label: String, _ a: [Double], _ b: [Double], _ tolerance: Double) {
        let g = gap(a, b)
        print("strain mapping \(label): maximum absolute gap \(g), tolerance \(tolerance)")
        #expect(g <= tolerance, "\(label): gap \(g) above \(tolerance)")
    }

    // MARK: the Bragg vector map

    /// BraggVectors.histogram: same accumulation order and arithmetic, so equal to rounding of the sums.
    /// Tolerance: absolute 1e-12 times the map's maximum (measured gaps are in the printed lines).
    @Test func histogramMatchesPy4DSTEM() throws {
        let d1 = try dataset("d1"), d2 = try dataset("d2")
        func make(_ d: Dataset, sampling: Double = 1, mask: [Bool]? = nil, weights: [Double]? = nil) throws -> BraggVectorHistogram {
            try StrainMapping.histogram(peaks: d.peaks, scanRows: d.rows, scanColumns: d.columns, detectorRows: d.detectorRows,
                                        detectorColumns: d.detectorColumns, origin: d.origin, sampling: sampling,
                                        scanMask: mask, scanWeights: weights)
        }
        let cases: [(String, BraggVectorHistogram, [Double])] = [
            ("d1", try make(d1), try array("map_d1_hist")),
            ("d2", try make(d2), try array("map_d2_hist_default")),
            ("d2 sampling 1.5", try make(d2, sampling: 1.5), try array("map_d2_hist_s15")),
            ("d2 scan mask", try make(d2, mask: bools(try array("map_d2_scanmask"))), try array("map_d2_hist_bool")),
            ("d2 scan weights", try make(d2, weights: try array("map_d2_weights")), try array("map_d2_hist_num")),
        ]
        for (label, h, expected) in cases {
            let scale = expected.map(abs).max() ?? 1
            check("histogram \(label)", h.values, expected, 1e-12 * scale)
        }
        let h15 = try make(d2, sampling: 1.5)
        #expect(h15.rows == 72 && h15.columns == 72)
        #expect(h15.origin == .init(row: 1.5 * d2.origin.row, column: 1.5 * d2.origin.column))
    }

    /// The grid size rounds ties to even, as NumPy's round does: 0.5 * 5 = 2.5 -> 2, 0.5 * 7 = 3.5 -> 4.
    @Test func histogramGridRounding() throws {
        let h = try StrainMapping.histogram(peaks: [[]], scanRows: 1, scanColumns: 1, detectorRows: 5, detectorColumns: 7,
                                            origin: .init(row: 2, column: 3), sampling: 0.5)
        #expect(h.rows == 2 && h.columns == 4)
    }

    /// One peak at an exact integer cell and one at the corner of four cells, counted by hand.
    @Test func histogramHandCount() throws {
        let peaks = [[StrainPeak(row: 1, column: 1, intensity: 8), StrainPeak(row: 1.25, column: -0.5, intensity: 4),
                      StrainPeak(row: 100, column: 0, intensity: 9)]]
        let h = try StrainMapping.histogram(peaks: peaks, scanRows: 1, scanColumns: 1, detectorRows: 4, detectorColumns: 4,
                                            origin: .init(row: 1, column: 1))
        // peak A at (2,2): all 8 in that cell. peak B at (2.25, 0.5): weights .75*.5, .75*.5, .25*.5, .25*.5 of 4.
        var expected = [Double](repeating: 0, count: 16)
        expected[2 * 4 + 2] = 8
        expected[2 * 4 + 0] = 4 * 0.75 * 0.5; expected[2 * 4 + 1] = 4 * 0.75 * 0.5
        expected[3 * 4 + 0] = 4 * 0.25 * 0.5; expected[3 * 4 + 1] = 4 * 0.25 * 0.5
        #expect(h.values == expected)                        // the peak far outside the grid is dropped
    }

    // MARK: choose_basis_vectors, fit_basis_vectors, get_strain

    struct Run {
        let name: String, dataset: String
        var basis = StrainBasisOptions()
        var spacing: Double
        var mask: String?
        var reference: String = "median"          // median | roi | supplied
        var rotation = 0.0
        var tolerances: (maxima: Double, directions: Double, fit: Double, strain: Double)
    }
    func runs() -> [Run] {
        var r1 = Run(name: "r1", dataset: "d1", spacing: 3, tolerances: (1e-9, 1e-9, 1e-9, 1e-9))                       // multicorr
        r1.basis = StrainBasisOptions()
        var r4 = Run(name: "r4", dataset: "d1", spacing: 3, reference: "supplied", tolerances: (1e-9, 1e-9, 1e-9, 1e-9))
        r4.basis.subpixel = .parabolic; r4.basis.sigma = 1; r4.basis.minimumSpacing = 4; r4.basis.maximumPeaks = 12
        var r2 = Run(name: "r2", dataset: "d2", spacing: 2.5, reference: "roi", rotation: 20, tolerances: (1e-9, 1e-9, 1e-9, 1e-9))
        r2.basis.subpixel = .parabolic; r2.basis.sigma = 1; r2.basis.minimumSpacing = 4; r2.basis.maximumPeaks = 12
        var r3 = Run(name: "r3", dataset: "d2", spacing: 2.5, mask: "map_r3_scanmask", rotation: -35, tolerances: (1e-9, 1e-9, 1e-9, 1e-9))
        r3.basis.subpixel = .pixel; r3.basis.minimumSpacing = 3; r3.basis.minimumRelativeIntensity = 0.02; r3.basis.maximumPeaks = 14
        var r5 = Run(name: "r5", dataset: "d2", spacing: 2.5, tolerances: (1e-9, 1e-9, 1e-9, 1e-9))                       // multicorr
        r5.basis.indexG1 = 1; r5.basis.sigma = 1; r5.basis.minimumSpacing = 2; r5.basis.maximumPeaks = 10
        var r6 = Run(name: "r6", dataset: "d3", spacing: 4, tolerances: (1e-9, 1e-9, 1e-9, 1e-9))
        r6.basis.subpixel = .parabolic; r6.basis.sigma = 1; r6.basis.minimumSpacing = 6; r6.basis.maximumPeaks = 8
        return [r1, r4, r2, r3, r5, r6]
    }
    func execute(_ r: Run) throws -> (StrainMapResult, Dataset) {
        let d = try dataset(r.dataset)
        var o = StrainMapOptions(maximumPeakSpacing: r.spacing)
        o.coordinateRotation = r.rotation
        if let m = r.mask { o.scanMask = bools(try array(m)) }
        switch r.reference {
        case "roi": o.reference = .region(bools(try array("map_\(r.name)_roi")))
        case "supplied":
            let g = try array("map_\(r.name)_reference")
            o.reference = .supplied(.init(g1: .init(row: g[0], column: g[1]), g2: .init(row: g[2], column: g[3])))
        default: break
        }
        let result = try StrainMapping.compute(peaks: d.peaks, scanRows: d.rows, scanColumns: d.columns, detectorRows: d.detectorRows,
                                               detectorColumns: d.detectorColumns, origin: d.origin, basisOptions: r.basis, options: o)
        return (result, d)
    }

    /// Every stage of py4DSTEM's StrainMap against its pinned outputs: maxima, origin/g1/g2, indexed directions (exact
    /// indices), the per-position fits, the validity mask (exact), the reference and the four strain maps.
    /// Tolerance 1e-9 absolute for every stage and run, written in `runs()` (the quantities are of order 1 to 100;
    /// measured gaps, printed by this test, are at most 2.3e-13 for the maxima, including the multicorr refinement, and
    /// 1.4e-14 for fits and strains; `docs/strain.md`).
    @Test func scanLevelMatchesStrainMap() throws {
        for r in runs() {
            let (res, _) = try execute(r)
            let p = "map_\(r.name)"
            let t = r.tolerances
            let maxima = try array(p + "_maxima")
            #expect(res.directions.maxima.count * 3 == maxima.count, "\(r.name): maxima count")
            check("\(r.name) maxima", res.directions.maxima.flatMap { [$0.row, $0.column, $0.intensity] }, maxima, t.maxima)
            let basis = try array(p + "_basis"), c = try #require(res.directions.choice)
            check("\(r.name) g0 g1 g2", [c.g0.row, c.g0.column, c.basis.g1.row, c.basis.g1.column, c.basis.g2.row, c.basis.g2.column], basis, t.maxima)
            let dirs = try array(p + "_directions")
            #expect(res.directions.directions.count * 4 == dirs.count)
            for (i, d) in res.directions.directions.enumerated() {
                #expect(d.h == dirs[i * 4 + 2] && d.k == dirs[i * 4 + 3], "\(r.name): direction \(i) indices")     // exact
            }
            check("\(r.name) directions", res.directions.directions.flatMap { [$0.vector.row, $0.vector.column] },
                  (0..<res.directions.directions.count).flatMap { [dirs[$0 * 4], dirs[$0 * 4 + 1]] }, t.directions)
            let fits = try array(p + "_fits"), n = res.valid.count
            func slice(_ k: Int) -> [Double] { Array(fits[k * n..<(k + 1) * n]) }
            let mask = try array(p + "_mask")
            #expect(res.valid.map { $0 ? 1.0 : 0 } == mask, "\(r.name): validity mask")                           // exact
            let fitted = res.valid.map { $0 ? 1.0 : Double.nan }
            func masked(_ a: [Double]) -> [Double] { zip(a, fitted).map { $1.isNaN ? .nan : $0 } }
            // py4DSTEM stores 0 where a position was not fitted; compare the fitted positions, NaN elsewhere.
            check("\(r.name) fits", [res.originRow, res.originColumn, res.g1Row, res.g1Column, res.g2Row, res.g2Column, res.residual].flatMap { $0 }.map { $0.isNaN ? 0 : $0 },
                  (0..<7).flatMap { masked(slice($0)) }.map { $0.isNaN ? 0 : $0 }, t.fit)
            let ref = try array(p + "_reference")
            check("\(r.name) reference", [res.reference.g1.row, res.reference.g1.column, res.reference.g2.row, res.reference.g2.column], ref, t.fit)
            let strain = try array(p + "_strain")
            for (k, a) in [res.exx, res.eyy, res.exy, res.theta].enumerated() {
                check("\(r.name) strain \(["exx", "eyy", "exy", "theta"][k])", a.map { $0.isNaN ? 0 : $0 },
                      masked(Array(strain[k * n..<(k + 1) * n])).map { $0.isNaN ? 0 : $0 }, t.strain)
            }
            #expect(res.valid.indices.allSatisfy { res.valid[$0] == !res.exx[$0].isNaN })
        }
    }

    /// The runs must exercise what they claim: invalid positions exist, the scan mask excludes, the reference differs
    /// from the all-position median, rotation changes exx.
    @Test func pinsAreNotVacuous() throws {
        let all = runs()
        let r2 = try execute(all[2]).0, r3 = try execute(all[3]).0, r1 = try execute(all[0]).0
        #expect(r2.valid.filter { !$0 }.count >= 4)               // thin positions are not fitted
        #expect(r3.valid.filter { !$0 }.count >= 6)               // thin positions and the two masked ones
        #expect(r1.valid.allSatisfy { $0 })
        #expect(r2.exx.contains { $0.isNaN } && r2.exx.contains { !$0.isNaN && abs($0) > 1e-3 })
    }

    // MARK: planted truth

    /// Recovery of imposed strain, independent of py4DSTEM: d1 carries exact planted tensors with columns 0...2
    /// unstrained; taking those columns as the reference region must give back the planted exx, eyy, exy, theta.
    /// Tolerance 1e-9 (noise-free peaks, the fit is exact to rounding).
    @Test func plantedStrainIsRecovered() throws {
        let d = try dataset("d1"), truth = try array("map_d1_truth")
        let n = d.rows * d.columns
        var o = StrainMapOptions(maximumPeakSpacing: 3)
        o.reference = .region((0..<n).map { $0 % d.columns < 3 })
        var b = StrainBasisOptions(); b.subpixel = .parabolic; b.sigma = 1
        let r = try StrainMapping.compute(peaks: d.peaks, scanRows: d.rows, scanColumns: d.columns, detectorRows: d.detectorRows,
                                          detectorColumns: d.detectorColumns, origin: d.origin, basisOptions: b, options: o)
        check("planted exx", r.exx, Array(truth[0..<n]), 1e-9)
        check("planted eyy", r.eyy, Array(truth[n..<2 * n]), 1e-9)
        check("planted exy", r.exy, Array(truth[2 * n..<3 * n]), 1e-9)
        check("planted theta", r.theta, Array(truth[3 * n..<4 * n]), 1e-9)
        #expect(truth[0..<n].contains { abs($0) > 0.005 })       // the planted field is not trivial
        // The reference is the planted lattice.
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("strain_mapping.json"))) as! [String: Any]
        let d1 = (meta["datasets"] as! [String: Any])["d1"] as! [String: Any]
        let g1 = d1["refG1"] as! [Double], g2 = d1["refG2"] as! [Double]
        check("planted reference", [r.reference.g1.row, r.reference.g1.column, r.reference.g2.row, r.reference.g2.column], g1 + g2, 1e-9)
        // A coordinate rotation keeps trace, rotation and the shear magnitude of the planted tensor.
        o.coordinateRotation = 37
        let q = try StrainMapping.compute(peaks: d.peaks, scanRows: d.rows, scanColumns: d.columns, detectorRows: d.detectorRows,
                                          detectorColumns: d.detectorColumns, origin: d.origin, basisOptions: b, options: o)
        check("rotated trace", zip(q.exx, q.eyy).map { $0 + $1 }, zip(r.exx, r.eyy).map { $0 + $1 }, 1e-12)
        check("rotated theta", q.theta, r.theta, 1e-15)
        check("rotated shear invariant", zip(zip(q.exx, q.eyy), q.exy).map { pow($0.0 - $0.1, 2) + 4 * $1 * $1 },
              zip(zip(r.exx, r.eyy), r.exy).map { pow($0.0 - $0.1, 2) + 4 * $1 * $1 }, 1e-12)
        #expect(gap(q.exx, r.exx) > 1e-4)
    }

    /// Equal intensities are ordered as NumPy's sort does it: larger row first, then larger column.
    @Test func equalMaximaFollowNumPyOrder() throws {
        let ar = try array("map_ties_ar"), expected = try array("map_ties_maxima")
        var o = StrainBasisOptions()
        o.subpixel = .pixel
        let h = BraggVectorHistogram(rows: 14, columns: 14, values: ar, origin: .init(row: 6.2, column: 6.4))
        let dirs = try StrainMapping.chooseBasisVectors(histogram: h, options: o)
        #expect(dirs.maxima.flatMap { [$0.row, $0.column, $0.intensity] } == expected)
        #expect(dirs.maxima.first?.row == 9 && dirs.maxima.first?.column == 9)
    }

    // MARK: options composed from the existing numerics

    @Test func variantsComposeTheNamedOptions() throws {
        let d = try dataset("d2")
        var o = StrainMapOptions(maximumPeakSpacing: 2.5)
        var b = StrainBasisOptions(); b.subpixel = .parabolic; b.sigma = 1; b.minimumSpacing = 4; b.maximumPeaks = 12
        func run(_ o: StrainMapOptions) throws -> StrainMapResult {
            try StrainMapping.compute(peaks: d.peaks, scanRows: d.rows, scanColumns: d.columns, detectorRows: d.detectorRows,
                                      detectorColumns: d.detectorColumns, origin: d.origin, basisOptions: b, options: o)
        }
        let base = try run(o)
        o.fitMethod = .intensityOnce
        let once = try run(o)
        // The same position fitted by hand with the compatibility fit.
        let k = try #require(base.valid.firstIndex(of: true))
        let indexed = try LatticeStrain.index(peaks: d.peaks[k].map { LatticePeak(row: $0.row, column: $0.column, intensity: $0.intensity) },
                                              directions: base.directions.directions, maximumSpacing: 2.5)
        let by = try #require(try LatticeStrain.fitIntensityOnce(indexed))
        #expect(once.g1Row[k] == by.basis.g1.row && once.residual[k] == by.residual)
        #expect(once.g1Row[k] != base.g1Row[k])
        o.fitMethod = .py4DSTEM; o.referenceMethod = .madRejection
        let mad = try run(o)
        let bases = base.valid.indices.filter { base.valid[$0] }.map { LatticeBasis(g1: .init(row: base.g1Row[$0], column: base.g1Column[$0]), g2: .init(row: base.g2Row[$0], column: base.g2Column[$0])) }
        #expect(mad.reference == (try LatticeStrain.referenceMAD(bases)))
        #expect(mad.referenceInlierCount == (try LatticeStrain.referenceMADResult(bases)).inlierCount && mad.referenceInlierCount <= mad.referencePositionCount)
        #expect(base.referenceInlierCount == base.referencePositionCount)
        #expect(base.reference == (try LatticeStrain.reference(bases)))
        // A caller-supplied lattice replaces the histogram stage.
        let lattice = try StrainDirections.lattice(basis: base.directions.basis, order: 2)
        #expect(lattice.directions.count == 25)
        let viaLattice = try StrainMapping.compute(peaks: d.peaks, scanRows: d.rows, scanColumns: d.columns, detectorRows: d.detectorRows,
                                                   detectorColumns: d.detectorColumns, origin: d.origin, directions: lattice,
                                                   options: StrainMapOptions(maximumPeakSpacing: 2.5))
        #expect(viaLattice.histogram == nil)
        #expect(viaLattice.valid.filter { $0 }.count >= base.valid.filter { $0 }.count - 4)
    }

    @Test func braggVectorsConvertToCentredPeaks() {
        let v = BraggVectors(scanRows: 1, scanColumns: 2, peaks: [[BraggPeak(row: 10, column: 12, intensity: 3)], []])
        let c = StrainMapping.centredPeaks(v, origin: .init(row: 8, column: 9))
        #expect(c == [[StrainPeak(row: 2, column: 3, intensity: 3)], []])
    }

    // MARK: validation

    @Test func invalidInputIsRefused() throws {
        let d = try dataset("d1")
        let o = StrainMapOptions(maximumPeakSpacing: 3)
        func compute(peaks: [[StrainPeak]]? = nil, rows: Int? = nil, options: StrainMapOptions? = nil, directions: StrainDirections? = nil) throws {
            _ = try StrainMapping.compute(peaks: peaks ?? d.peaks, scanRows: rows ?? d.rows, scanColumns: d.columns,
                                          detectorRows: d.detectorRows, detectorColumns: d.detectorColumns, origin: d.origin,
                                          directions: directions, options: options ?? o)
        }
        func hist(rows: Int = 8, sampling: Double = 1, mask: [Bool]? = nil, weights: [Double]? = nil, peaks: [[StrainPeak]]? = nil) throws {
            _ = try StrainMapping.histogram(peaks: peaks ?? [[], []], scanRows: 1, scanColumns: 2, detectorRows: rows, detectorColumns: 8,
                                            origin: .init(row: 4, column: 4), sampling: sampling, scanMask: mask, scanWeights: weights)
        }
        #expect(throws: StrainMappingError.self) { try compute(rows: d.rows + 1) }                  // peak lists vs scan shape
        #expect(throws: StrainMappingError.self) { try compute(rows: 0) }
        #expect(throws: StrainMappingError.self) { try compute(options: StrainMapOptions(maximumPeakSpacing: -1)) }
        #expect(throws: StrainMappingError.self) { try compute(options: StrainMapOptions(maximumPeakSpacing: .nan)) }
        var bad = o; bad.minimumPeaks = 0
        #expect(throws: StrainMappingError.self) { try compute(options: bad) }
        bad = o; bad.coordinateRotation = .infinity
        #expect(throws: StrainMappingError.self) { try compute(options: bad) }
        bad = o; bad.scanMask = [true]
        #expect(throws: StrainMappingError.self) { try compute(options: bad) }
        bad = o; bad.reference = .region([true])
        #expect(throws: StrainMappingError.self) { try compute(options: bad) }
        bad = o; bad.reference = .supplied(.init(g1: .init(row: .nan, column: 0), g2: .init(row: 0, column: 1)))
        #expect(throws: StrainMappingError.self) { try compute(options: bad) }
        #expect(throws: LatticeStrainError.self) {                                                   // empty reference region
            var e = o; e.reference = .region([Bool](repeating: false, count: d.rows * d.columns)); try compute(options: e)
        }
        #expect(throws: LatticeStrainError.self) { _ = try StrainDirections.lattice(basis: .init(g1: .init(row: 1, column: 2), g2: .init(row: 2, column: 4))) }
        #expect(throws: StrainMappingError.self) { _ = try StrainDirections.lattice(basis: .init(g1: .init(row: 1, column: 0), g2: .init(row: 0, column: 1)), order: 0) }
        #expect(throws: StrainMappingError.self) { try hist(rows: 0) }
        #expect(throws: StrainMappingError.self) { try hist(sampling: 0) }
        #expect(throws: StrainMappingError.self) { try hist(sampling: .nan) }
        #expect(throws: StrainMappingError.self) { try hist(mask: [true]) }
        #expect(throws: StrainMappingError.self) { try hist(weights: [1, .nan]) }
        #expect(throws: StrainMappingError.self) { try hist(mask: [true, true], weights: [1, 1]) }
        #expect(throws: StrainMappingError.self) { try hist(peaks: [[StrainPeak(row: .nan, column: 0, intensity: 1)], []]) }
        // Basis choice: too few maxima, index out of range, a map that is not sampling 1, a degenerate basis.
        let flat = BraggVectorHistogram(rows: 16, columns: 16, values: [Double](repeating: 0, count: 256), origin: .init(row: 8, column: 8))
        #expect(throws: StrainMappingError.self) { _ = try StrainMapping.chooseBasisVectors(histogram: flat) }
        var peaky = [Double](repeating: 0, count: 256)
        for (r, c) in [(4, 4), (4, 10), (10, 4), (10, 10)] { peaky[r * 16 + c] = 1 }
        peaky[8 * 16 + 8] = 2
        let map = BraggVectorHistogram(rows: 16, columns: 16, values: peaky, origin: .init(row: 8, column: 8))
        var b = StrainBasisOptions(); b.subpixel = .pixel
        _ = try StrainMapping.chooseBasisVectors(histogram: map, options: b)
        b.indexG2 = 99
        #expect(throws: LatticeStrainError.self) { _ = try StrainMapping.chooseBasisVectors(histogram: map, options: b) }
        b.indexG2 = nil; b.indexG1 = 0; b.indexOrigin = 0                                           // g1 = origin: a zero vector
        #expect(throws: LatticeStrainError.self) { _ = try StrainMapping.chooseBasisVectors(histogram: map, options: b) }
        b = StrainBasisOptions(); b.subpixel = .pixel; b.sigma = -1
        #expect(throws: StrainMappingError.self) { _ = try StrainMapping.chooseBasisVectors(histogram: map, options: b) }
        let sampled = BraggVectorHistogram(rows: 16, columns: 16, values: peaky, origin: .init(row: 8, column: 8), sampling: 2)
        #expect(throws: StrainMappingError.self) { _ = try StrainMapping.chooseBasisVectors(histogram: sampled) }
    }

    /// An all-positions-unfit scan still returns (NaN maps) when the reference is supplied, and throws without one.
    @Test func scanWithNoFittedPosition() throws {
        let d = try dataset("d1")
        let empty = [[StrainPeak]](repeating: [], count: d.rows * d.columns)
        let lattice = try StrainDirections.lattice(basis: .init(g1: .init(row: 9, column: 1.5), g2: .init(row: -1.2, column: 10.5)))
        var o = StrainMapOptions(maximumPeakSpacing: 3)
        #expect(throws: LatticeStrainError.self) {
            _ = try StrainMapping.compute(peaks: empty, scanRows: d.rows, scanColumns: d.columns, detectorRows: 48, detectorColumns: 48,
                                          origin: d.origin, directions: lattice, options: o)
        }
        o.reference = .supplied(lattice.basis)
        let r = try StrainMapping.compute(peaks: empty, scanRows: d.rows, scanColumns: d.columns, detectorRows: 48, detectorColumns: 48,
                                          origin: d.origin, directions: lattice, options: o)
        #expect(r.valid.allSatisfy { !$0 } && r.exx.allSatisfy(\.isNaN) && r.indexedPeakCount.allSatisfy { $0 == 0 })
    }
}
