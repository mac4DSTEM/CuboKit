import Foundation
import Testing
import CuboPtycho

private struct ComplexPin: Decodable { let real: [Double]; let imaginary: [Double] }
private struct Variant: Decodable {
    let fixProbe: Bool, purePhase: Bool, minimum: Double, step: Double, object: ComplexPin, probe: ComplexPin, errors: [Double]
}
private struct Pin: Decodable {
    let variants: [Variant], rectangularProbe: ComplexPin
    let rows: Int, columns: Int, objectRows: Int, objectColumns: Int
    let wavelength: Double, positions: [[Double]], amplitudes: [Double]
    let initialProbe: ComplexPin, initialObject: ComplexPin, builtProbe: ComplexPin, object: ComplexPin, probe: ComplexPin
    let errors: [Double]
}
private func pin() throws -> Pin {
    let path = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/ptycho.json")
    return try JSONDecoder().decode(Pin.self, from: Data(contentsOf: path))
}
private func gap(_ a: [Double], _ b: [Double]) -> Double {
    guard a.count == b.count else { return .infinity }
    return zip(a,b).map { abs($0-$1) }.max()! / max(1, b.map(abs).max()!)
}
@Suite struct PtychographyTests {
    @Test func probeAgainstComplexProbe() throws {
        let p = try pin()
        let probe = try PtychographyProbe.build(rows:p.rows, columns:p.columns, rowSampling:0.8, columnSampling:1.1,
            wavelength:p.wavelength, cutoff:0.018, rolloff:0.002, defocus:70, astigmatism:25,
            astigmatismAngle:0.31, sphericalAberration:2e4)
        // ComplexProbe evaluates chi/frequencies in Float32; unit-scale absolute gap.
        let g = max(gap(probe.real,p.builtProbe.real),gap(probe.imaginary,p.builtProbe.imaginary))
        print("PTYCHO probe max normwise gap: \(g)")
        #expect(g < 2e-6)
        let rectangular = try PtychographyProbe.build(rows:7,columns:10,rowSampling:0.8,columnSampling:1.1,
            wavelength:p.wavelength,cutoff:0.018,rolloff:0,defocus:70,astigmatism:25,astigmatismAngle:0.31,sphericalAberration:2e4)
        let rg = max(gap(rectangular.real,p.rectangularProbe.real),gap(rectangular.imaginary,p.rectangularProbe.imaginary))
        print("PTYCHO rectangular probe gap: \(rg)")
        #expect(rg < 2e-6)
    }
    @Test func threeIterationsAgainstSingleslice() throws {
        let p = try pin()
        let initialProbe = try PtychographyArray(rows:p.rows, columns:p.columns, real:p.initialProbe.real, imaginary:p.initialProbe.imaginary)
        let initialObject = try PtychographyArray(rows:p.objectRows, columns:p.objectColumns, real:p.initialObject.real, imaginary:p.initialObject.imaginary)
        let start = ContinuousClock.now
        let result = try SingleslicePtychography.reconstruct(amplitudes:p.amplitudes,
            positions:p.positions.map { .init(row:$0[0],column:$0[1]) }, object:initialObject, probe:initialProbe,
            iterations:3, stepSize:0.35, normalizationMinimum:0.7)
        let g = [gap(result.object.real,p.object.real),gap(result.object.imaginary,p.object.imaginary),
                 gap(result.probe.real,p.probe.real),gap(result.probe.imaginary,p.probe.imaginary),gap(result.errors,p.errors)].max()!
        print("PTYCHO engine max normwise gap: \(g)")
        // Python complex64 state, Float32 probe shifts and accumulation order versus Swift Double.
        #expect(g < 2e-6)
        for v in p.variants {
            let r = try SingleslicePtychography.reconstruct(amplitudes:p.amplitudes,
                positions:p.positions.map { .init(row:$0[0],column:$0[1]) },object:initialObject,probe:initialProbe,
                iterations:3,stepSize:v.step,normalizationMinimum:v.minimum,fixProbe:v.fixProbe,purePhaseObject:v.purePhase)
            let vg = [gap(r.object.real,v.object.real),gap(r.object.imaginary,v.object.imaginary),
                      gap(r.probe.real,v.probe.real),gap(r.probe.imaginary,v.probe.imaginary),gap(r.errors,v.errors)].max()!
            print("PTYCHO variant \(v.minimum) gap: \(vg)")
            #expect(vg < 2e-6)
        }
        #expect(start.duration(to:.now) < .seconds(10))
    }
    @Test func invalidInputsAreRefused() throws {
        #expect(throws: PtychographyError.self) {
            try PtychographyArray(rows:Int.max,columns:2,real:[],imaginary:[])
        }
        let p = try pin()
        let probe = try PtychographyArray(rows:p.rows,columns:p.columns,real:p.initialProbe.real,imaginary:p.initialProbe.imaginary)
        #expect(throws: PtychographyError.self) {
            try SingleslicePtychography.reconstruct(amplitudes:[],positions:[],object:probe,probe:probe)
        }
        #expect(throws: PtychographyError.self) {
            try PtychographyProbe.build(rows:2,columns:2,rowSampling:0,columnSampling:1,wavelength:0.02,cutoff:0.01)
        }
    }
}
