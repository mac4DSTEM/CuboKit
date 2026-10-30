import Foundation
import Testing
import CuboParallax

@Suite struct ParallaxAberrationTests {
    struct Pin: Decodable {
        let detectorShape, indices: [Int]
        let angles, shifts: [Double]
        let terms: [[Int]]
        let rows,columns: Int
        let image: [Double]; let wavelength: Double; let cases: [Case]
    }
    struct Case: Decodable {
        let transpose: Bool; let forced: Double?; let method: String
        let rotation: Double; let low,coefs,appCoefs,corrected,appCorrected: [Double]
    }
    func pin() throws -> Pin {
        try JSONDecoder().decode(Pin.self,from:Data(contentsOf:parallaxFixture("aberrations")!))
    }
    func pairs(_ a:[Double]) -> [ParallaxShift] { stride(from:0,to:a.count,by:2).map{.init(row:a[$0],column:a[$0+1])} }
    func check(_ a:[Double],_ b:[Double],tol:Double=2e-4) {
        #expect(a.count == b.count)
        let gap = zip(a,b).map{abs($0-$1)/max(1,abs($1))}.max() ?? 0
        print("aberration gap \(gap)")
        #expect(gap < tol) // per-coefficient relative tolerance, unit floor; Float32 reference basis.
    }
    @Test func pinnedFits() throws {
        let p=try pin(), angles=pairs(p.angles), shifts=pairs(p.shifts)
        let indices=stride(from:0,to:p.indices.count,by:2).map{[p.indices[$0],p.indices[$0+1]]}
        for c in p.cases {
            var o=ParallaxHigherOrderAberrationOptions();o.maxRadialOrder=4
            o.lowOrder.forceTranspose=c.transpose;o.lowOrder.forceRotationAngleDegrees=c.forced
            o.fitMethod=c.method == "global" ? .global : c.method == "recursive-exclusive" ? .recursiveExclusive : .recursive
            for app in [false,true] {
                o.basisConvention=app ? .selectedCentroid : .detectorGrid
                let basis=try ParallaxAberrationFitter.basisAngles(detectorIndices:indices,detectorRows:9,detectorColumns:10,
                    reciprocalSampling:.init(row:0.08,column:0.08),wavelength:p.wavelength,convention:o.basisConvention)
                let f=try ParallaxAberrationFitter.fitHigherOrder(probeAngles:angles,pixelShifts:shifts,scanSampling:.init(row:1.25,column:1.25),basisAngles:basis,options:o)
                check([f.lowOrder.rotationRad],[c.rotation],tol:1e-10)
                check([f.lowOrder.c1Angstrom,f.lowOrder.c12aAngstrom,f.lowOrder.c12bAngstrom],c.low,tol:1e-10)
                check(f.coefficientsAngstrom,app ? c.appCoefs : c.coefs)
                #expect(f.terms.map{[$0.radialOrder,$0.angularOrder,$0.component]} == p.terms)
            }
        }
    }
    @Test func pinnedCorrection() throws {
        let p=try pin();let terms=p.terms.map{ParallaxAberrationTerm(radialOrder:$0[0],angularOrder:$0[1],component:$0[2])}
        for c in p.cases {
            for app in [false,true] {
                let r=try ParallaxAberrationCorrection.correct(p.image,rows:p.rows,columns:p.columns,sampling:.init(row:1.25,column:1.25),wavelength:p.wavelength,
                    terms:terms,coefficients:c.coefs,lowpass:0.3,highpass:0.12,highpassBehavior:app ? .apply : .reference)
                check(r.image,app ? c.appCorrected : c.corrected,tol:2e-6)
            }
        }
    }
    @Test func invalidFitAndCorrection() throws {
        #expect(throws: (any Error).self) { try ParallaxAberrationFitter.fitLowOrder(probeAngles:[],pixelShifts:[],scanSampling:.init(row:1,column:1)) }
        #expect(throws: ParallaxError.invalidInput) { try ParallaxAberrationCorrection.correct([],rows:Int.max,columns:8,sampling:.init(row:1,column:1),wavelength:0.025,terms:[],coefficients:[]) }
    }
}
