import Foundation
import Testing
import CuboKitTestSupport
import CuboPhase

private struct IDPCPin: Decodable {
    let rows: Int, columns: Int, factor: Int
    let regularization: Double, rowSampling: Double, columnSampling: Double, scale: Double
    let row: [Double], column: [Double], phase: [Double], truth: [Double]
}
private struct IDPCPins: Decodable { let cases: [IDPCPin] }
private func idpcPins() throws -> [IDPCPin] {
    let path = URL(fileURLWithPath:#filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/idpc.json")
    return try JSONDecoder().decode(IDPCPins.self,from:Data(contentsOf:path)).cases
}
@Suite struct IDPCTests {
    @Test func numpyFourierPins() throws {
        for p in try idpcPins() {
            let field = try ScanVectorField(scanRows:p.rows,scanColumns:p.columns,row:p.row,column:p.column)
            let boundary: IDPCBoundary = p.factor == 1 ? .periodic : .zeroPadded(factor:p.factor)
            let result = try IDPC.integrate(field,rowSampling:p.rowSampling,columnSampling:p.columnSampling,
                                           gradientScale:p.scale,regularization:p.regularization,boundary:boundary)
            let gap = zip(result,p.phase).map {abs($0-$1)}.max()!
            print("IDPC factor \(p.factor) max absolute gap: \(gap)")
            // Double FFT against the pinned NumPy float64 transcription of app Fourier iDPC.
            #expect(gap < 1e-12)
            if p.factor == 1 {
                #expect(zip(result,p.truth).allSatisfy {abs($0-$1) < 1e-12})
            }
            if p.scale != 1 {
                let calibration = try IDPCPhysicalCalibration(rowSamplingAngstrom:p.rowSampling,
                    columnSamplingAngstrom:p.columnSampling,reciprocalAngstromPerDetectorPixel:p.scale/(2 * .pi))
                let physical = try IDPC.integratePhysical(field,calibration:calibration,
                    regularization:p.regularization,boundary:boundary)
                #expect(zip(physical,p.phase).allSatisfy {abs($0-$1) < 1e-12})
            }
        }
    }
    @Test func invalidInputsAndCalibration() throws {
        let field = try ScanVectorField(scanRows:2,scanColumns:2,row:[1,2,3,4],column:[0,0,0,0])
        #expect(throws:CuboPhaseError.self) { try IDPC.integrate(field,rowSampling:0) }
        #expect(throws:CuboPhaseError.self) { try IDPC.integrate(field,boundary:.zeroPadded(factor:0)) }
        #expect(throws:CuboPhaseError.self) { try IDPC.integrate(field,boundary:.zeroPadded(factor:Int.max)) }
        #expect(throws:CuboPhaseError.self) { try IDPC.integrate(field,regularization: -.infinity) }
        #expect(throws:CuboPhaseError.self) { try IDPCPhysicalCalibration(rowSamplingAngstrom:1,columnSamplingAngstrom:1,reciprocalAngstromPerDetectorPixel: .nan) }
        let bad = try ScanVectorField(scanRows:2,scanColumns:2,row:[1,2,.nan,4],column:[0,0,0,0])
        #expect(throws:CuboPhaseError.self) { try IDPC.integrate(bad) }
    }
}
