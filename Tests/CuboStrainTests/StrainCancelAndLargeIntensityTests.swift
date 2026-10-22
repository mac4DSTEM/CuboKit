import Foundation
import Testing
import CuboStrain

/// 0.9.0: `LatticeStrain.fitRobust` accepts intensities of 1e10 and above (rescaled in Double first), and
/// `StrainMapping.compute` / `histogram` take `isCancelled` and `progress` in the style of `RotationSolver.curve`.
final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    private var seen: [Double] = []
    func next() -> Int { lock.lock(); defer { lock.unlock() }; n += 1; return n }
    var count: Int { lock.lock(); defer { lock.unlock() }; return n }
    func record(_ v: Double) { lock.lock(); seen.append(v); lock.unlock() }
    var values: [Double] { lock.lock(); defer { lock.unlock() }; return seen }
}

@Suite struct RobustLargeIntensityTests {
    let g1 = LatticeVector(row: 10.2, column: 1.3)
    let g2 = LatticeVector(row: -1.1, column: 9.6)

    func lattice(_ intensity: (Int, Int) -> Double, rowScale: Double = 1) -> [LatticePeak] {
        var out: [LatticePeak] = []
        for h in -2...2 { for k in -2...2 where h != 0 || k != 0 {
            out.append(.init(row: rowScale * (Double(h) * g1.row + Double(k) * g2.row), column: Double(h) * g1.column + Double(k) * g2.column,
                             intensity: intensity(h, k), h: Double(h), k: Double(k)))
        } }
        return out
    }

    func same(_ a: LatticeFit, _ b: LatticeFit, _ rel: Double, _ comment: String) {
        for (x, y) in [(a.basis.g1.row, b.basis.g1.row), (a.basis.g1.column, b.basis.g1.column), (a.basis.g2.row, b.basis.g2.row),
                       (a.basis.g2.column, b.basis.g2.column), (a.origin.row, b.origin.row), (a.origin.column, b.origin.column)] {
            #expect(abs(x - y) <= rel * 10, "\(comment): \(x) vs \(y)")
        }
    }

    @Test(arguments: [1e10, 1e12, 1e20, 1e30, 1e100, 1e300])
    func intensitiesOfOneE10AndAboveGiveTheFitOfTheSameRatios(_ scale: Double) throws {
        var planted = lattice({ h, k in 1 + 0.3 * Double(abs(h) + abs(k)) }, rowScale: 0.99)
        planted.append(.init(row: 0.99 * (g1.row + g2.row) + 0.9, column: g1.column + g2.column - 0.5, intensity: 2, h: 1, k: 1))   // an outlier
        let base = try #require(try LatticeStrain.fitRobust(planted))
        let big = planted.map { var p = $0; p.intensity *= scale; return p }
        let fit = try #require(try LatticeStrain.fitRobust(big))
        same(base, fit, 2e-6, "scale \(scale)")
        #expect(abs(fit.basis.g1.row - 0.99 * g1.row) < 1e-4)   // the planted strain is recovered
    }

    @Test func aHotPeakOfOneE12AmongOrdinaryOnesDominatesAsItsWeightSays() throws {
        // One peak at 1e12, the rest 1 to 3: the ordinary peaks weigh 1e-12 and are floored at 1e-6 (as after any division by the
        // largest), so the fit is the weighted fit a Double transcription gives, not a refusal.
        var peaks = lattice({ h, k in 1 + Double(abs(h)) })
        peaks[3].intensity = 1e12
        let fit = try #require(try LatticeStrain.fitRobust(peaks))
        var reference = peaks; reference[3].intensity = 1e6   // the same weights after dividing by the largest: 1, then the 1e-6 floor
        let want = try #require(try LatticeStrain.fitRobust(reference))
        same(fit, want, 1e-5, "hot peak")
        // Ordinary intensities (below 1e10) are untouched: one more decade of the same ratios changes nothing.
        let ordinary = try #require(try LatticeStrain.fitRobust(lattice({ h, k in 1 + Double(abs(h)) })))
        let times1e9 = try #require(try LatticeStrain.fitRobust(lattice({ h, k in (1 + Double(abs(h))) * 1e9 })))
        same(ordinary, times1e9, 1e-6, "1e9")
    }

    @Test func nonFiniteAndNegativeIntensitiesAreStillRefused() {
        var peaks = lattice({ _, _ in 1 })
        peaks[0].intensity = .infinity
        #expect(throws: LatticeStrainError.self) { try LatticeStrain.fitRobust(peaks) }
        peaks[0].intensity = .nan
        #expect(throws: LatticeStrainError.self) { try LatticeStrain.fitRobust(peaks) }
        peaks[0].intensity = -1e12
        #expect(throws: LatticeStrainError.self) { try LatticeStrain.fitRobust(peaks) }
        peaks = lattice({ _, _ in 1 }); peaks[0].vector.row = 1e12   // a coordinate of 1e10 or more stays unrepresentable
        #expect(throws: LatticeStrainError.self) { try LatticeStrain.fitRobust(peaks) }
    }
}

@Suite struct StrainCancelAndProgressTests {
    let g1 = LatticeVector(row: 10.2, column: 1.3)
    let g2 = LatticeVector(row: -1.1, column: 9.6)

    /// `side` x `side` positions, each the planted lattice plus an off-lattice peak.
    func scan(_ side: Int) -> [[StrainPeak]] {
        (0..<(side * side)).map { position in
            var list: [StrainPeak] = []
            for h in -2...2 { for k in -2...2 where h != 0 || k != 0 {
                let r = Double(h) * g1.row + Double(k) * g2.row, c = Double(h) * g1.column + Double(k) * g2.column
                list.append(StrainPeak(row: r * (position == side + 1 ? 0.99 : 1), column: c, intensity: 1 + 0.3 * Double(abs(h) + abs(k))))
            } }
            list.append(StrainPeak(row: g1.row + 0.9, column: g1.column - 0.4, intensity: 1.2))
            return list
        }
    }

    func options() -> StrainMapOptions {
        var o = StrainMapOptions(maximumPeakSpacing: 5)
        o.indexing = .basisRounding(maximumOrder: 8); o.fitMethod = .robustIntensityOnce; o.referenceMethod = .madRejection
        return o
    }

    @Test func progressIsMonotoneAndEndsAtOne() throws {
        let side = 12
        for supplied in [true, false] {
            let seen = Counter()
            let result = try StrainMapping.compute(
                peaks: scan(side), scanRows: side, scanColumns: side, detectorRows: 64, detectorColumns: 64, origin: .init(row: 0, column: 0),
                directions: supplied ? try StrainDirections.lattice(basis: .init(g1: g1, g2: g2)) : nil,
                basisOptions: StrainBasisOptions(), options: options(), progress: { seen.record($0) })
            let values = seen.values
            #expect(!values.isEmpty && values.last == 1, "ends at 1 (supplied \(supplied))")
            #expect(zip(values, values.dropFirst()).allSatisfy { $0 <= $1 }, "monotone (supplied \(supplied))")
            #expect(values.allSatisfy { $0 >= 0 && $0 <= 1 })
            #expect(values.count > 5, "reports along the way")
            #expect(result.scanRows == side)
        }
        let seen = Counter()
        _ = try StrainMapping.histogram(peaks: scan(side), scanRows: side, scanColumns: side, detectorRows: 64, detectorColumns: 64,
                                        origin: .init(row: 32, column: 32), progress: { seen.record($0) })
        #expect(seen.values.last == 1 && zip(seen.values, seen.values.dropFirst()).allSatisfy { $0 <= $1 })
    }

    @Test func progressAndTheResultAreUnchangedByTheDefaultArguments() throws {
        let side = 6
        let a = try StrainMapping.compute(peaks: scan(side), scanRows: side, scanColumns: side, detectorRows: 64, detectorColumns: 64,
                                          origin: .init(row: 0, column: 0), directions: StrainDirections.lattice(basis: .init(g1: g1, g2: g2)),
                                          options: options())
        let b = try StrainMapping.compute(peaks: scan(side), scanRows: side, scanColumns: side, detectorRows: 64, detectorColumns: 64,
                                          origin: .init(row: 0, column: 0), directions: StrainDirections.lattice(basis: .init(g1: g1, g2: g2)),
                                          options: options(), isCancelled: { false }, progress: { _ in })
        #expect(a.exx.map(\.bitPattern) == b.exx.map(\.bitPattern) && a.g1Row.map(\.bitPattern) == b.g1Row.map(\.bitPattern))
    }

    @Test func cancellingStopsEarlyAndReturnsNothing() throws {
        let side = 40   // 1600 positions
        let all = scan(side)
        // Never cancelled: the number of times the closure is asked in a full run.
        let full = Counter()
        _ = try StrainMapping.compute(peaks: all, scanRows: side, scanColumns: side, detectorRows: 64, detectorColumns: 64,
                                      origin: .init(row: 0, column: 0), directions: StrainDirections.lattice(basis: .init(g1: g1, g2: g2)),
                                      options: options(), isCancelled: { _ = full.next(); return false })
        #expect(full.count >= side * side)
        // Cancelled after 100 questions: throws CancellationError, asked about 100 times, never to the end.
        let asked = Counter(), last = Counter()
        #expect(throws: CancellationError.self) {
            try StrainMapping.compute(peaks: all, scanRows: side, scanColumns: side, detectorRows: 64, detectorColumns: 64,
                                      origin: .init(row: 0, column: 0), directions: StrainDirections.lattice(basis: .init(g1: g1, g2: g2)),
                                      options: options(), isCancelled: { asked.next() > 100 }, progress: { last.record($0) })
        }
        #expect(asked.count <= 102 && asked.count < full.count / 10, "stopped early (\(asked.count) of \(full.count) questions)")
        #expect((last.values.last ?? 0) < 0.2, "progress stopped with the work")
        // Cancelled before the start.
        #expect(throws: CancellationError.self) {
            try StrainMapping.compute(peaks: all, scanRows: side, scanColumns: side, detectorRows: 64, detectorColumns: 64,
                                      origin: .init(row: 0, column: 0), directions: StrainDirections.lattice(basis: .init(g1: g1, g2: g2)),
                                      options: options(), isCancelled: { true })
        }
        // The Bragg vector map alone, and compute while it builds the map.
        let h = Counter()
        #expect(throws: CancellationError.self) {
            try StrainMapping.histogram(peaks: all, scanRows: side, scanColumns: side, detectorRows: 64, detectorColumns: 64,
                                        origin: .init(row: 32, column: 32), isCancelled: { h.next() > 10 })
        }
        #expect(h.count == 11)
        let c = Counter()
        #expect(throws: CancellationError.self) {
            try StrainMapping.compute(peaks: all, scanRows: side, scanColumns: side, detectorRows: 64, detectorColumns: 64,
                                      origin: .init(row: 32, column: 32), options: options(), isCancelled: { c.next() > 20 })
        }
        #expect(c.count == 21)
    }
}
