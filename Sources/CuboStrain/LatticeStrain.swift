import Foundation
import CStrainLAPACK

/// Detector coordinates: first axis is row (py4DSTEM x), second is column (y).
public struct LatticeVector: Sendable, Equatable, Codable {
    /// Row coordinate.
    public var row: Double
    /// Column coordinate.
    public var column: Double
    /// Creates a detector vector.
    public init(row: Double, column: Double) { self.row = row; self.column = column }
}
/// A measured Bragg vector, optionally carrying integer lattice indices.
public struct LatticePeak: Sendable, Codable {
    /// Detector coordinates, in the same centered frame as the directions.
    public var vector: LatticeVector
    /// Intensity used by the fit.
    public var intensity: Double
    /// First lattice index.
    public var h: Double
    /// Second lattice index.
    public var k: Double
    /// Creates an indexed or unindexed peak.
    public init(row: Double, column: Double, intensity: Double, h: Double = 0, k: Double = 0) {
        vector = .init(row: row, column: column); self.intensity = intensity; self.h = h; self.k = k
    }
}
/// Two reciprocal lattice vectors.
public struct LatticeBasis: Sendable, Equatable, Codable {
    /// First reciprocal vector.
    public var g1: LatticeVector
    /// Second reciprocal vector.
    public var g2: LatticeVector
    /// Creates a basis.
    public init(g1: LatticeVector, g2: LatticeVector) { self.g1 = g1; self.g2 = g2 }
}
/// Fitted lattice and intensity-weighted mean Euclidean residual.
public struct LatticeFit: Sendable {
    /// Fitted intercept relative to the supplied origin.
    public let origin: LatticeVector
    /// Fitted reciprocal basis.
    public let basis: LatticeBasis
    /// Mean distance in detector units (RMS for the compatibility fit).
    public let residual: Double
}
/// Infinitesimal strain in the requested coordinate frame; theta is in radians.
public struct StrainTensor: Sendable {
    /// Normal strain along the first axis.
    public let exx: Double
    /// Normal strain along the second axis.
    public let eyy: Double
    /// Symmetric shear strain.
    public let exy: Double
    /// Antisymmetric rotation, unchanged by a coordinate rotation.
    public let theta: Double
}
/// Invalid numerical input or an undefined reference.
public enum LatticeStrainError: Error, Sendable { case invalidInput, singularBasis, emptyReference }

/// CPU Double strain numerics following py4DSTEM f050d207.
public enum LatticeStrain {
    /// Selects a basis from histogram maxima ordered by decreasing intensity.
    /// Maxima must include the direct beam. The nearest peak is the origin, the next
    /// nearest is g1, and g2 minimizes distance * (cos(angle-angle_g1)^2 + 0.1).
    /// This is the numeric selection step of `StrainMap.choose_basis_vectors`.
    public static func chooseBasis(maxima: [LatticeVector], origin: LatticeVector) throws -> LatticeBasis {
        try chooseBasis(maxima: maxima, origin: origin, indexOrigin: nil, indexG1: nil, indexG2: nil).basis
    }
    /// The three maxima chosen as origin, g1 and g2, with the basis they span.
    public struct BasisChoice: Sendable, Equatable {
        /// Index into the maxima of the lattice origin (py4DSTEM `index_origin`).
        public let originIndex: Int
        /// Index of the maximum that defines g1.
        public let g1Index: Int
        /// Index of the maximum that defines g2.
        public let g2Index: Int
        /// The position of the origin maximum (py4DSTEM `g0`).
        public let g0: LatticeVector
        /// g1 and g2 relative to `g0`.
        public let basis: LatticeBasis
    }
    /// `StrainMap.choose_basis_vectors`' selection with optional explicit indices: an index left nil is guessed exactly
    /// as py4DSTEM guesses it (origin nearest `origin`; g1 next nearest; g2 minimizing distance * (cos² + 0.1)), and an
    /// index given is used as is, so a given origin index leaves its distance unchanged, as in py4DSTEM.
    /// Throws `invalidInput` for fewer than three maxima or an index out of range.
    public static func chooseBasis(maxima: [LatticeVector], origin: LatticeVector,
                                   indexOrigin: Int?, indexG1: Int?, indexG2: Int?) throws -> BasisChoice {
        guard maxima.count >= 3, maxima.allSatisfy(finite), finite(origin) else { throw LatticeStrainError.invalidInput }
        for i in [indexOrigin, indexG1, indexG2] { if let i, !maxima.indices.contains(i) { throw LatticeStrainError.invalidInput } }
        var d = maxima.map { hypot($0.row-origin.row, $0.column-origin.column) }
        let angles = maxima.map { atan2($0.column-origin.column, $0.row-origin.row) }
        var i0 = indexOrigin, i1 = indexG1, i2 = indexG2
        if i0 == nil || i1 == nil || i2 == nil {
            if i0 == nil { i0 = d.indices.min(by: { d[$0] < d[$1] })!; d[i0!] = 2 * d.max()! }
            if i1 == nil { i1 = d.indices.min(by: { d[$0] < d[$1] })!; d[i1!] = 2 * d.max()! }
            if i2 == nil {
                let score = d.indices.map { d[$0] * (pow(cos(angles[$0]-angles[i1!]), 2) + 0.1) }
                i2 = score.indices.min(by: { score[$0] < score[$1] })!
            }
        }
        let (a, b, c) = (i0!, i1!, i2!)
        return .init(originIndex: a, g1Index: b, g2Index: c, g0: maxima[a],
                     basis: .init(g1: .init(row: maxima[b].row-maxima[a].row, column: maxima[b].column-maxima[a].column),
                                  g2: .init(row: maxima[c].row-maxima[a].row, column: maxima[c].column-maxima[a].column)))
    }
    /// Indexes histogram directions by NumPy's ties-to-even rounding.
    public static func index(directions: [LatticeVector], basis: LatticeBasis) throws -> [LatticePeak] {
        let a = basis.g1, b = basis.g2
        guard [a,b].allSatisfy(finite), directions.allSatisfy(finite) else { throw LatticeStrainError.invalidInput }
        let m = [[a.row,b.row],[a.column,b.column]]
        return try directions.map { p in
            let hk = try solve(m,[p.row,p.column])
            return .init(row:p.row,column:p.column,intensity:1,
                         h:hk[0].rounded(.toNearestOrEven),k:hk[1].rounded(.toNearestOrEven))
        }
    }
    /// Assigns peaks to the nearest measured histogram direction, within the inclusive distance limit.
    /// Equal distances retain the first direction, as NumPy argmin does.
    public static func index(peaks: [LatticePeak], directions: [LatticePeak], maximumSpacing: Double) throws -> [LatticePeak] {
        guard maximumSpacing.isFinite, maximumSpacing >= 0, valid(peaks), valid(directions), !directions.isEmpty else {
            throw LatticeStrainError.invalidInput
        }
        return peaks.compactMap { p in
            let ds = directions.map { hypot(p.vector.row-$0.vector.row, p.vector.column-$0.vector.column) }
            let i = ds.indices.min(by: { ds[$0] < ds[$1] })!
            guard ds[i] <= maximumSpacing else { return nil }
            var q = p; q.h = directions[i].h; q.k = directions[i].k; return q
        }
    }
    /// Fits [1,h,k] after multiplying both design and observations by intensity.
    /// Thus the minimized objective is sum(intensity² * distance²). Rank-deficient
    /// designs use LAPACK's minimum-norm solution, like NumPy lstsq. Too few peaks return nil.
    public static func fit(_ peaks: [LatticePeak], origin: LatticeVector = .init(row: 0, column: 0), minimumPeaks: Int = 5) throws -> LatticeFit? {
        guard valid(peaks), finite(origin), minimumPeaks >= 1 else { throw LatticeStrainError.invalidInput }
        guard peaks.count >= minimumPeaks else { return nil }
        let design = peaks.map { [$0.intensity, $0.h*$0.intensity, $0.k*$0.intensity] }
        let r = try solve(design, peaks.map { ($0.vector.row-origin.row)*$0.intensity })
        let c = try solve(design, peaks.map { ($0.vector.column-origin.column)*$0.intensity })
        let weight = peaks.reduce(0) { $0 + $1.intensity }
        guard weight > 0 else { throw LatticeStrainError.invalidInput }
        let error = peaks.reduce(0.0) { s,p in
            s + p.intensity * hypot(p.vector.row-origin.row-r[0]-p.h*r[1]-p.k*r[2],
                                    p.vector.column-origin.column-c[0]-p.h*c[1]-p.k*c[2])
        } / weight
        return .init(origin: .init(row:r[0],column:c[0]),
                     basis:.init(g1:.init(row:r[1],column:c[1]),g2:.init(row:r[2],column:c[2])), residual:error)
    }
    /// Component-wise median of the explicitly selected bases; caller excludes invalid positions.
    public static func reference(_ bases: [LatticeBasis]) throws -> LatticeBasis {
        guard !bases.isEmpty else { throw LatticeStrainError.emptyReference }
        guard bases.allSatisfy({ finite($0.g1) && finite($0.g2) }) else { throw LatticeStrainError.invalidInput }
        return .init(g1:.init(row:median(bases.map { $0.g1.row }),column:median(bases.map { $0.g1.column })),
                     g2:.init(row:median(bases.map { $0.g2.row }),column:median(bases.map { $0.g2.column })))
    }
    /// Computes beta = lstsq(reference rows, local rows).transpose, then
    /// exx=1-beta00, eyy=1-beta11, exy=-(beta01+beta10)/2, theta=(beta01-beta10)/2.
    /// Coordinate rotation is counterclockwise in degrees, matching StrainMap.get_strain.
    public static func strain(local: LatticeBasis, reference: LatticeBasis, coordinateRotation: Double = 0) throws -> StrainTensor {
        guard [local.g1,local.g2,reference.g1,reference.g2].allSatisfy(finite), coordinateRotation.isFinite else { throw LatticeStrainError.invalidInput }
        let m = [[reference.g1.row,reference.g1.column],[reference.g2.row,reference.g2.column]]
        let x = try solve(m,[local.g1.row,local.g2.row]), y = try solve(m,[local.g1.column,local.g2.column])
        let xx = 1-x[0], yy = 1-y[1], xy = -(x[1]+y[0])/2
        let angle = -coordinateRotation * .pi/180, c = cos(angle), s = sin(angle)
        return .init(exx:c*c*xx-2*c*s*xy+s*s*yy, eyy:s*s*xx+2*c*s*xy+c*c*yy,
                     exy:c*s*(xx-yy)+(c*c-s*s)*xy,theta:(x[1]-y[0])/2)
    }
    /// Compatibility option: intensity-once normal-equation fit, Float32 vectors and RMS residual.
    /// Uses the original numerical helper; no local outlier rejection is applied here.
    public static func fitIntensityOnce(_ peaks: [LatticePeak]) throws -> LatticeFit? {
        guard valid(peaks), representable(peaks.flatMap { [$0.vector.row,$0.vector.column,$0.h,$0.k,$0.intensity] }) else { throw LatticeStrainError.invalidInput }
        guard let f = ConsensusNumerics.fitLattice(peaks.map { (Float($0.h),Float($0.k),Float($0.vector.row),Float($0.vector.column),max(Float($0.intensity),1e-6)) }) else { return nil }
        return .init(origin:.init(row:Double(f.originX),column:Double(f.originY)),basis:.init(g1:.init(row:Double(f.g1x),column:Double(f.g1y)),g2:.init(row:Double(f.g2x),column:Double(f.g2y))),residual:Double(f.residual))
    }
    /// The reference of ``referenceMADResult(_:)`` and the candidates that produced it.
    public struct ReferenceMAD: Sendable, Equatable {
        /// Component-wise Float32 median of the retained candidates (what ``referenceMAD(_:)`` returns).
        public let reference: LatticeBasis
        /// Indices into the input bases of the candidates kept, ascending. All indices when fewer than half survive
        /// the rejection (the fallback) or when there are two bases or fewer.
        public let inlierIndices: [Int]
        /// `inlierIndices.count`.
        public var inlierCount: Int { inlierIndices.count }
    }
    /// Compatibility option: reject reference bases beyond median + max(3*1.4826*MAD, 2% basis scale).
    /// Falls back to the entire set when fewer than half survive. Arithmetic follows Float32 source.
    public static func referenceMAD(_ bases: [LatticeBasis]) throws -> LatticeBasis {
        try referenceMADResult(bases).reference
    }
    /// ``referenceMAD(_:)`` with the retained candidates: the same reference, plus which bases were kept.
    public static func referenceMADResult(_ bases: [LatticeBasis]) throws -> ReferenceMAD {
        _ = try reference(bases)
        guard representable(bases.flatMap { [$0.g1.row,$0.g1.column,$0.g2.row,$0.g2.column] }) else { throw LatticeStrainError.invalidInput }
        let a = bases.map { Float($0.g1.row) }, b = bases.map { Float($0.g1.column) }, c = bases.map { Float($0.g2.row) }, d = bases.map { Float($0.g2.column) }
        let ids = ConsensusNumerics.robustReferenceIndices(candidates:Array(bases.indices),g1x:a,g1y:b,g2x:c,g2y:d)
        func med(_ v:[Float]) -> Double { Double(ConsensusNumerics.median(ids.map { v[$0] })!) }
        return .init(reference: .init(g1:.init(row:med(a),column:med(b)),g2:.init(row:med(c),column:med(d))), inlierIndices: ids)
    }
    /// The basis ``chooseBasisConsensus(_:origin:minimumRadius:)`` selects, with the diagnostics of its scoring.
    /// Counts are exact; the Float32 diagnostics are widened to Double unchanged.
    public struct ConsensusBasis: Sendable, Equatable {
        /// The chosen basis (identical to the basis ``chooseBasisConsensus(_:origin:minimumRadius:)`` returns).
        public let basis: LatticeBasis
        /// Peaks beyond `minimumRadius` from the origin, over all positions (the denominator of the support fraction).
        public let observationCount: Int
        /// Observations indexed by the basis within `indexingTolerance` (|h|,|k| at most 8, not both zero).
        public let supportCount: Int
        /// RMS distance of the supporting observations to their lattice node, in detector units.
        public let residual: Double
        /// Condition number of the basis (at most 8 for a chosen basis).
        public let conditionNumber: Double
        /// Indexing tolerance in detector units: max(0.5, 0.18 * the shorter vector).
        public let indexingTolerance: Double
    }
    /// ``chooseBasisConsensus(_:origin:minimumRadius:)`` with the scoring diagnostics and an optional cancellation closure,
    /// polled while peaks are clustered and before each candidate pair is scored. Throws `CancellationError` when it
    /// returns true. The basis is identical to the plain call's; no consensus returns nil.
    public static func chooseBasisConsensusResult(_ positions: [[LatticePeak]], origin: LatticeVector = .init(row:0,column:0), minimumRadius: Double = 2,
                                                  isCancelled: (@Sendable () -> Bool)? = nil) throws -> ConsensusBasis? {
        guard positions.allSatisfy(valid), finite(origin), minimumRadius.isFinite, minimumRadius >= 0,
              representable([origin.row,origin.column,minimumRadius]+positions.flatMap { $0.flatMap { [$0.vector.row,$0.vector.column] } }) else { throw LatticeStrainError.invalidInput }
        if isCancelled?() == true { throw CancellationError() }
        let grid = ConsensusPeakGrid(peaks:positions.map { $0.map { ConsensusPeak(x:Float($0.vector.row),y:Float($0.vector.column),intensity:Float($0.intensity)) } })
        let token = isCancelled.map { ConsensusCancellation(check: $0) }
        let e = ConsensusNumerics.estimateLatticeBasis(bragg:grid,x0:Float(origin.row),y0:Float(origin.column),minRadius:Float(minimumRadius),cancellation:token)
        if isCancelled?() == true { throw CancellationError() }
        guard let e else { return nil }
        return .init(basis: .init(g1:.init(row:Double(e.g1.x),column:Double(e.g1.y)),g2:.init(row:Double(e.g2.x),column:Double(e.g2.y))),
                     observationCount: e.observationCount, supportCount: e.supportCount, residual: Double(e.residualPixels),
                     conditionNumber: Double(e.conditionNumber), indexingTolerance: Double(e.indexingTolerancePixels))
    }
    /// Compatibility option: cluster repeated vectors, score bounded candidate pairs by consensus.
    /// Uses Float32 source arithmetic, min radius 2 by default, condition limit 8,
    /// cluster support 8% of occupied positions, and at least 50% observation support.
    public static func chooseBasisConsensus(_ positions: [[LatticePeak]], origin: LatticeVector = .init(row:0,column:0), minimumRadius: Double = 2) throws -> LatticeBasis? {
        try chooseBasisConsensusResult(positions, origin: origin, minimumRadius: minimumRadius)?.basis
    }
    private static func finite(_ v:LatticeVector) -> Bool { v.row.isFinite && v.column.isFinite }
    private static func valid(_ p:[LatticePeak]) -> Bool { p.allSatisfy { finite($0.vector) && $0.intensity.isFinite && $0.intensity >= 0 && $0.h.isFinite && $0.k.isFinite } }
    private static func representable(_ a:[Double]) -> Bool { a.allSatisfy { Float($0).isFinite && abs($0) < 1e10 } }
    private static func median(_ a:[Double]) -> Double { let v=a.sorted(), i=v.count/2; return v.count%2 == 1 ? v[i] : (v[i-1]+v[i])/2 }
    private static func solve(_ design:[[Double]], _ rhs:[Double]) throws -> [Double] {
        let m=design.count, n=design[0].count
        guard m <= Int(Int32.max), n <= Int(Int32.max) else { throw LatticeStrainError.invalidInput }
        var a=(0..<n).flatMap { j in design.map { $0[j] } }, b=rhs+Array(repeating:0.0,count:max(m,n)-rhs.count)
        let info=cubo_strain_least_squares(Int32(m),Int32(n),&a,&b,Double(max(m,n))*Double.ulpOfOne)
        guard info == 0 else { throw LatticeStrainError.invalidInput }
        return Array(b.prefix(Int(n)))
    }
}
