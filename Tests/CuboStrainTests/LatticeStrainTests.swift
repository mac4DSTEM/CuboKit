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
}
