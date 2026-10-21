import Foundation
import Testing
@testable import CuboStrain

@Suite struct LatticeStrainTests {
    let directory = URL(fileURLWithPath:#filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")
    func array(_ name:String) throws -> [Double] {
        let d=try Data(contentsOf:directory.appendingPathComponent(name+".f64"))
        return d.withUnsafeBytes { raw in stride(from:0,to:raw.count,by:8).map { Double(bitPattern:UInt64(littleEndian:raw.loadUnaligned(fromByteOffset:$0,as:UInt64.self))) } }
    }
    func peaks(_ i:Int) throws -> [LatticePeak] {
        let a=try array("input\(i)")
        return stride(from:0,to:a.count,by:5).map { .init(row:a[$0],column:a[$0+1],intensity:a[$0+2],h:a[$0+3],k:a[$0+4]) }
    }
    func basis(_ a:[Double]) -> LatticeBasis { .init(g1:.init(row:a[0],column:a[1]),g2:.init(row:a[2],column:a[3])) }
    func values(_ f:LatticeFit) -> [Double] { [f.origin.row,f.origin.column,f.basis.g1.row,f.basis.g1.column,f.basis.g2.row,f.basis.g2.column,f.residual] }
    func close(_ a:[Double],_ b:[Double],_ tolerance:Double = 1e-10) {
        #expect(a.count == b.count)
        let gap=zip(a,b).map { abs($0-$1) }.max() ?? 0
        print("strain maximum absolute gap: \(gap), tolerance: \(tolerance)")
        #expect(gap <= tolerance)
    }
    @Test func pinnedFitsAndStrains() throws {
        let expected=try array("fits"), ref=basis(try array("reference"))
        var fits:[LatticeFit]=[]
        for i in 0..<4 {
            let f=try #require(try LatticeStrain.fit(peaks(i))); fits.append(f)
            close(values(f),Array(expected[(i*7)..<(i*7+7)]))
        }
        let measuredReference = try LatticeStrain.reference(fits.prefix(3).map(\.basis))
        close([measuredReference.g1.row,measuredReference.g1.column,measuredReference.g2.row,measuredReference.g2.column],[ref.g1.row,ref.g1.column,ref.g2.row,ref.g2.column])
        for (i,angle) in [0.0,31.0].enumerated() {
            let expected=try array("strain\(i)")
            let actual=try fits.map { try LatticeStrain.strain(local:$0.basis,reference:ref,coordinateRotation:angle) }
            close(actual.map(\.exx)+actual.map(\.eyy)+actual.map(\.exy)+actual.map(\.theta),expected)
        }
        // Planted transform for case 1 gives exx=0.02, eyy=-0.03, exy=-0.005, theta=-0.02.
        let truth=try LatticeStrain.strain(local:fits[1].basis,reference:basis([8,1,-0.5,10]))
        close([truth.exx,truth.eyy,truth.exy,truth.theta],[0.02,-0.03,-0.005,-0.02])
    }
    @Test func originalAppOptions() throws {
        let expected=try array("appFits")
        for i in 0..<4 { close(values(try #require(try LatticeStrain.fitIntensityOnce(peaks(i)))),Array(expected[i*7..<i*7+7]),0) }
        let b=try array("appBases"), bases=stride(from:0,to:b.count,by:4).map { basis(Array(b[$0..<$0+4])) }
        let r=try LatticeStrain.referenceMAD(bases)
        close([r.g1.row,r.g1.column,r.g2.row,r.g2.column],try array("appReference"),0)
        let c=try array("consensus")
        let positions=(0..<4).map { i in (0..<5).map { j in let n=(i*5+j)*3; return LatticePeak(row:c[n],column:c[n+1],intensity:c[n+2]) } }
        let chosen=try #require(try LatticeStrain.chooseBasisConsensus(positions))
        close([chosen.g1.row,chosen.g1.column,chosen.g2.row,chosen.g2.column],try array("appBasis"),0)
        #expect(try LatticeStrain.reference(bases) != r)
        let primary = try #require(try LatticeStrain.fit(peaks(2)))
        let compatibility = try #require(try LatticeStrain.fitIntensityOnce(peaks(2)))
        #expect(primary.basis != compatibility.basis)
    }
    @Test func indexingAndSelection() throws {
        let mx=try array("maxima")
        let maxima=stride(from:0,to:mx.count,by:2).map { LatticeVector(row:mx[$0],column:mx[$0+1]) }
        let selected=try LatticeStrain.chooseBasis(maxima:maxima,origin:.init(row:56,column:56))
        close([selected.g1.row,selected.g1.column,selected.g2.row,selected.g2.column],try array("selectedBasis"))
        let zero=LatticeVector(row:0,column:0)
        let b=try LatticeStrain.chooseBasis(maxima:[zero,.init(row:8,column:0),.init(row:0,column:10),.init(row:16,column:0)],origin:zero)
        #expect(b == basis([8,0,0,10]))
        let ds=try LatticeStrain.index(directions:[zero,.init(row:8,column:0),.init(row:0,column:10)],basis:b)
        let indexed=try LatticeStrain.index(peaks:[.init(row:7.9,column:0,intensity:4),.init(row:3,column:3,intensity:1)],directions:ds,maximumSpacing:0.2)
        #expect(indexed.count == 1 && indexed[0].h == 1)
        let indexInput=try array("indexInputs"), indexOutput=try array("indexOutputs")
        for i in 0..<3 {
            let x=Array(indexInput[i*6..<i*6+6])
            let result=try LatticeStrain.index(directions:[.init(row:x[4],column:x[5])],basis:basis(Array(x.prefix(4))))[0]
            close([result.h,result.k],Array(indexOutput[i*2..<i*2+2]),0)
        }
        let tiny = basis([1e-8,0,0,1e-8])
        #expect(try LatticeStrain.index(directions:[.init(row:2e-8,column:-3e-8)],basis:tiny)[0].h == 2)
        let deficient = try LatticeStrain.index(directions:[.init(row:4,column:99)],basis:basis([2,0,0,0]))
        #expect(deficient[0].h == 2 && deficient[0].k == 0)
        let ties = try LatticeStrain.index(directions:[.init(row:0.5,column:1.5)],basis:basis([1,0,0,1]))
        #expect(ties[0].h == 0 && ties[0].k == 2)
        #expect(try LatticeStrain.fit([]) == nil)
        #expect(throws:LatticeStrainError.self) { try LatticeStrain.reference([]) }
        #expect(throws:LatticeStrainError.self) { try LatticeStrain.fit([.init(row:.nan,column:0,intensity:1)]) }
        #expect(try LatticeStrain.chooseBasisConsensus([]) == nil)
    }

    // MARK: - Round 6: consensus diagnostics, cancellation, MAD inliers

    final class Counter: @unchecked Sendable {
        private let lock = NSLock(); private var n = 0
        func tick() -> Int { lock.lock(); defer { lock.unlock() }; n += 1; return n }
        var count: Int { lock.lock(); defer { lock.unlock() }; return n }
    }
    /// 60 positions of a planted lattice (g1 = (8, 0.5), g2 = (-0.4, 10)) with deterministic sub-tolerance jitter and an
    /// occasional off-lattice peak.
    func plantedPositions() -> [[LatticePeak]] {
        (0..<60).map { p in
            var peaks: [LatticePeak] = []
            for (h, k) in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (2.0, 0.0)] {
                let j1 = 0.05 * Double((p * 7 + Int(h * 3 + k * 5 + 20)) % 5 - 2), j2 = 0.05 * Double((p * 3 + Int(h * 5 + k * 2 + 20)) % 5 - 2)
                peaks.append(.init(row: 8 * h - 0.4 * k + j1, column: 0.5 * h + 10 * k + j2, intensity: 1 + Double(p % 3)))
            }
            if p % 7 == 0 { peaks.append(.init(row: 3.1 + Double(p) * 0.01, column: -13.3, intensity: 1)) }
            return peaks
        }
    }
    /// The scoring of the app's original inline search, written out in Double for the test (not the kit's code).
    func rescore(_ b: LatticeBasis, _ positions: [[LatticePeak]], origin: LatticeVector, minimumRadius: Double) -> (obs: Int, support: Int, residual: Double, condition: Double, tolerance: Double) {
        let obs = positions.flatMap { $0 }.map { (x: $0.vector.row - origin.row, y: $0.vector.column - origin.column) }
            .filter { hypot($0.x, $0.y) > minimumRadius }
        let g1 = b.g1, g2 = b.g2, det = g1.row * g2.column - g2.row * g1.column
        let tol = max(0.5, 0.18 * min(hypot(g1.row, g1.column), hypot(g2.row, g2.column)))
        var support = 0, sq = 0.0
        for o in obs {
            let h = ((g2.column * o.x - g2.row * o.y) / det).rounded(), k = ((-g1.column * o.x + g1.row * o.y) / det).rounded()
            guard abs(h) <= 8, abs(k) <= 8, h != 0 || k != 0 else { continue }
            let r = hypot(o.x - h * g1.row - k * g2.row, o.y - h * g1.column - k * g2.column)
            if r <= tol { support += 1; sq += r * r }
        }
        let tr = g1.row*g1.row + g1.column*g1.column + g2.row*g2.row + g2.column*g2.column
        let root = sqrt(max(0, tr*tr - 4*det*det)), cond = sqrt((tr + root) / (tr - root))
        return (obs.count, support, support > 0 ? sqrt(sq / Double(support)) : .infinity, cond, tol)
    }
    @Test func consensusResultMatchesPlainCallAndRescoring() throws {
        let positions = plantedPositions(), origin = LatticeVector(row: 0, column: 0)
        let plain = try #require(try LatticeStrain.chooseBasisConsensus(positions))
        let full = try #require(try LatticeStrain.chooseBasisConsensusResult(positions))
        #expect(full.basis == plain)                                  // default unchanged: identical basis, bit for bit
        let again = try #require(try LatticeStrain.chooseBasisConsensusResult(positions, isCancelled: { false }))
        #expect(again == full)                                        // a closure that never cancels changes nothing
        let r = rescore(full.basis, positions, origin: origin, minimumRadius: 2)
        #expect(full.observationCount == r.obs)                       // integers exact
        #expect(full.supportCount == r.support)
        #expect(full.supportCount * 2 >= full.observationCount)
        let gaps = [abs(full.residual - r.residual), abs(full.conditionNumber - r.condition), abs(full.indexingTolerance - r.tolerance)]
        print("consensus diagnostics gaps (residual, condition, tolerance): \(gaps), tolerance 1e-5 (Float32)")
        #expect(gaps.allSatisfy { $0 <= 1e-5 * max(1, r.condition) })
        #expect(full.conditionNumber <= 8)
        // The pinned fixture case: the same basis as appBasis, with its diagnostics.
        let c = try array("consensus")
        let pinned = (0..<4).map { i in (0..<5).map { j in let n = (i*5+j)*3; return LatticePeak(row: c[n], column: c[n+1], intensity: c[n+2]) } }
        let pr = try #require(try LatticeStrain.chooseBasisConsensusResult(pinned))
        close([pr.basis.g1.row, pr.basis.g1.column, pr.basis.g2.row, pr.basis.g2.column], try array("appBasis"), 0)
        let pre = rescore(pr.basis, pinned, origin: origin, minimumRadius: 2)
        #expect(pr.observationCount == pre.obs && pr.supportCount == pre.support)
        #expect(abs(pr.residual - pre.residual) <= 1e-5)
        #expect(try LatticeStrain.chooseBasisConsensusResult([]) == nil)
    }
    @Test func consensusCancellation() throws {
        let positions = plantedPositions()
        #expect(throws: CancellationError.self) { try LatticeStrain.chooseBasisConsensusResult(positions, isCancelled: { true }) }
        let counter = Counter()
        // Polls: the entry, one per scan position (observation collection), then clustering and the candidate pairs. Cancel
        // only after every position has been polled, so the cancellation lands in the clustering or the pair loop.
        let polls = Counter()
        _ = try LatticeStrain.chooseBasisConsensusResult(positions, isCancelled: { _ = polls.tick(); return false })
        #expect(polls.count > positions.count + 2, "polled in the pair loop, not only per position")
        let threshold = positions.count + 2
        #expect(throws: CancellationError.self) {
            try LatticeStrain.chooseBasisConsensusResult(positions, isCancelled: { counter.tick() > threshold })
        }
        #expect(counter.count > threshold)
        // Invalid input still throws the input error first.
        #expect(throws: LatticeStrainError.self) { try LatticeStrain.chooseBasisConsensusResult([[.init(row: .nan, column: 0, intensity: 1)]], isCancelled: { true }) }
    }
    @Test func referenceMADInliers() throws {
        let b = try array("appBases"), bases = stride(from: 0, to: b.count, by: 4).map { basis(Array(b[$0..<$0+4])) }
        let plain = try LatticeStrain.referenceMAD(bases), full = try LatticeStrain.referenceMADResult(bases)
        #expect(full.reference == plain)                              // default unchanged
        #expect(full.inlierCount == full.inlierIndices.count && full.inlierIndices == full.inlierIndices.sorted())
        #expect(full.inlierIndices.allSatisfy { bases.indices.contains($0) })
        #expect(full.inlierCount * 2 >= bases.count)
        // Planted: nine near-identical bases and two grossly distorted ones are rejected, exactly.
        var planted = (0..<9).map { i in basis([8 + 0.001 * Double(i), 0.5, -0.4, 10 + 0.002 * Double(i)]) }
        planted.insert(basis([9.5, 1.5, -1.4, 8.5]), at: 3); planted.append(basis([6.5, -0.5, 0.6, 11.5]))
        let mad = try LatticeStrain.referenceMADResult(planted)
        #expect(mad.inlierIndices == [0, 1, 2, 4, 5, 6, 7, 8, 9])
        #expect(mad.inlierCount == 9)
        // Fallback keeps every index; two or fewer bases are kept whole.
        #expect(try LatticeStrain.referenceMADResult([planted[0], planted[3]]).inlierIndices == [0, 1])
    }
}
