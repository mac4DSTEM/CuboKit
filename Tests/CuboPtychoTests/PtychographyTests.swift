import Foundation
import Testing
import CuboCore
import CuboPtycho

private struct ComplexPin: Decodable { let real: [Double]; let imaginary: [Double] }
private struct Variant: Decodable {
    let fixProbe: Bool, purePhase: Bool, minimum: Double, step: Double, object: ComplexPin, probe: ComplexPin, errors: [Double]
}
private struct ConstraintCase: Decodable {
    let name: String, fixProbeCenterOfMass: Bool, constrainProbeAmplitude: Bool, radius: Double, width: Double
    let object: ComplexPin, probe: ComplexPin, errors: [Double]
}
/// The raw cube py4DSTEM's `_normalize_diffraction_intensities` turned into `amplitudes` and `meanIntensity`.
private struct Pre: Decodable { let raw: [Double], origin: [Double], meanIntensity: Double, energyPerPosition: Double }
private struct Pin: Decodable {
    let preprocessing: Pre, offcentreProbe: ComplexPin, constraintCases: [ConstraintCase]
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
            iterations:3, stepSize:0.35, normalizationMinimum:0.7, errorNormalization:.meanIntensity(p.preprocessing.meanIntensity))
        let g = [gap(result.object.real,p.object.real),gap(result.object.imaginary,p.object.imaginary),
                 gap(result.probe.real,p.probe.real),gap(result.probe.imaginary,p.probe.imaginary),gap(result.errors,p.errors)].max()!
        print("PTYCHO engine max normwise gap: \(g)")
        // Python complex64 state, Float32 probe shifts and accumulation order versus Swift Double.
        #expect(g < 2e-6)
        for v in p.variants {
            let r = try SingleslicePtychography.reconstruct(amplitudes:p.amplitudes,
                positions:p.positions.map { .init(row:$0[0],column:$0[1]) },object:initialObject,probe:initialProbe,
                iterations:3,stepSize:v.step,normalizationMinimum:v.minimum,fixProbe:v.fixProbe,purePhaseObject:v.purePhase,
                errorNormalization:.meanIntensity(p.preprocessing.meanIntensity))
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

    private func start() throws -> (Pin, PtychographyArray, PtychographyArray, PtychographyArray, [PtychographyPosition]) {
        let p = try pin()
        return (p,
            try PtychographyArray(rows:p.rows, columns:p.columns, real:p.initialProbe.real, imaginary:p.initialProbe.imaginary),
            try PtychographyArray(rows:p.rows, columns:p.columns, real:p.offcentreProbe.real, imaginary:p.offcentreProbe.imaginary),
            try PtychographyArray(rows:p.objectRows, columns:p.objectColumns, real:p.initialObject.real, imaginary:p.initialObject.imaginary),
            p.positions.map { .init(row:$0[0],column:$0[1]) })
    }
    /// Intensity centre of mass in pixels from the corner pixel, wrapped (row, column).
    private func centreOfMass(_ a: PtychographyArray) -> (Double, Double) {
        var t = 0.0, r = 0.0, c = 0.0
        for i in 0..<a.rows { for j in 0..<a.columns {
            let w = a.real[i*a.columns+j]*a.real[i*a.columns+j]+a.imaginary[i*a.columns+j]*a.imaginary[i*a.columns+j]
            t += w; r += Double(i < (a.rows+1)/2 ? i : i-a.rows)*w; c += Double(j < (a.columns+1)/2 ? j : j-a.columns)*w
        } }
        return (r/t, c/t)
    }
    @Test func probeConstraintsAgainstSingleslice() throws {
        let (p, _, off, object, positions) = try start()
        #expect(p.constraintCases.count == 4)
        for k in p.constraintCases {
            let r = try SingleslicePtychography.reconstruct(amplitudes:p.amplitudes, positions:positions, object:object, probe:off,
                iterations:3, stepSize:0.35, normalizationMinimum:0.7, fixProbeCenterOfMass:k.fixProbeCenterOfMass,
                constrainProbeAmplitude:k.constrainProbeAmplitude, probeAmplitudeRelativeRadius:k.radius,
                probeAmplitudeRelativeWidth:k.width, errorNormalization:.meanIntensity(p.preprocessing.meanIntensity))
            let g = [gap(r.object.real,k.object.real),gap(r.object.imaginary,k.object.imaginary),
                     gap(r.probe.real,k.probe.real),gap(r.probe.imaginary,k.probe.imaginary),gap(r.errors,k.errors)].max()!
            print("PTYCHO constraint \(k.name) gap: \(g)")
            // Same bound as the unconstrained pins: Float32 oracle state against Double.
            #expect(g < 2e-6)
        }
    }
    @Test func defaultsKeepTheOldBehaviourAndFixedProbeIgnoresConstraints() throws {
        let (p, _, off, object, positions) = try start()
        let plain = try SingleslicePtychography.reconstruct(amplitudes:p.amplitudes, positions:positions, object:object, probe:off,
            iterations:3, stepSize:0.35, normalizationMinimum:0.7)
        let explicit = try SingleslicePtychography.reconstruct(amplitudes:p.amplitudes, positions:positions, object:object, probe:off,
            iterations:3, stepSize:0.35, normalizationMinimum:0.7, fixProbeCenterOfMass:false, constrainProbeAmplitude:false,
            probeAmplitudeRelativeRadius:0.5, probeAmplitudeRelativeWidth:0.05)
        #expect(plain.object.real == explicit.object.real && plain.probe.real == explicit.probe.real
                && plain.probe.imaginary == explicit.probe.imaginary && plain.errors == explicit.errors)
        // With the probe fixed py4DSTEM applies no probe constraint; the probe stays the input.
        let fixed = try SingleslicePtychography.reconstruct(amplitudes:p.amplitudes, positions:positions, object:object, probe:off,
            iterations:3, stepSize:0.35, normalizationMinimum:0.7, fixProbe:true, fixProbeCenterOfMass:true,
            constrainProbeAmplitude:true, probeAmplitudeRelativeRadius:0.3, probeAmplitudeRelativeWidth:0.1)
        #expect(fixed.probe.real == off.real && fixed.probe.imaginary == off.imaginary)
    }
    @Test func shiftedProbeIsRecentred() throws {
        // Planted truth: a Gaussian probe (amplitude sigma 2.5 px, 32x32) centred at (1.3, -0.8) px from the corner pixel.
        // Its wrap-around tails (~e^-41) and band-limit aliasing (~e^-30) are far below 1e-9 px. The step size is
        // negligible, so the probe update leaves it as planted and the constraint alone acts.
        let n = 32, sigma = 2.5
        func wrap(_ d: Double) -> Double { d - Double(n) * (d / Double(n)).rounded() }
        var re = [Double](repeating:0, count:n*n)
        for r in 0..<n { for c in 0..<n {
            let dr = wrap(Double(r)-1.3), dc = wrap(Double(c)+0.8)
            re[r*n+c] = exp(-(dr*dr+dc*dc)/(2*sigma*sigma))
        } }
        let planted = try PtychographyArray(rows:n, columns:n, real:re, imaginary:[Double](repeating:0, count:n*n))
        let before = centreOfMass(planted)
        #expect(abs(before.0-1.3) < 1e-9 && abs(before.1+0.8) < 1e-9)
        let object = try PtychographyArray(rows:n, columns:n, real:[Double](repeating:1, count:n*n), imaginary:[Double](repeating:0, count:n*n))
        let amplitudes = [Double](repeating:1, count:n*n)
        let positions = [PtychographyPosition(row:0, column:0)]
        let r = try SingleslicePtychography.reconstruct(amplitudes:amplitudes, positions:positions, object:object, probe:planted,
            iterations:1, stepSize:1e-14, normalizationMinimum:1, fixProbeCenterOfMass:true)
        let after = centreOfMass(r.probe)
        print("PTYCHO planted recentred CoM: \(after)")
        #expect(abs(after.0) < 1e-9 && abs(after.1) < 1e-9)
        // The constraint is a pure translation: total intensity is kept.
        let i0 = zip(planted.real, planted.imaginary).reduce(0) { $0+$1.0*$1.0+$1.1*$1.1 }
        let i1 = zip(r.probe.real, r.probe.imaginary).reduce(0) { $0+$1.0*$1.0+$1.1*$1.1 }
        #expect(abs(i1-i0)/i0 < 1e-12)
        let free = try SingleslicePtychography.reconstruct(amplitudes:amplitudes, positions:positions, object:object, probe:planted,
            iterations:1, stepSize:1e-14, normalizationMinimum:1)
        #expect(abs(centreOfMass(free.probe).0-1.3) < 1e-9)
        // Pinned off-centre probe (not compact, so the periodic CoM is not a fixed point): the constraint reduces the offset.
        let (p, _, off, pobject, ppositions) = try start()
        let o = centreOfMass(off)
        let c = try SingleslicePtychography.reconstruct(amplitudes:p.amplitudes, positions:ppositions, object:pobject, probe:off,
            iterations:2, stepSize:0.35, normalizationMinimum:0.7, fixProbeCenterOfMass:true)
        let cm = centreOfMass(c.probe)
        print("PTYCHO pinned probe CoM before \(o) after \(cm)")
        #expect(hypot(cm.0,cm.1) < 0.1 * hypot(o.0,o.1))
    }
    @Test func invalidConstraintParametersAreRefused() throws {
        let (p, probe, _, object, positions) = try start()
        for (radius, width) in [(-0.1,0.05),(0.51,0.05),(.nan,0.05),(0.5,0),(0.5,0.51),(0.5,.infinity)] {
            #expect(throws: PtychographyError.self) {
                try SingleslicePtychography.reconstruct(amplitudes:p.amplitudes, positions:positions, object:object, probe:probe,
                    iterations:1, constrainProbeAmplitude:true, probeAmplitudeRelativeRadius:radius, probeAmplitudeRelativeWidth:width)
            }
        }
    }
}

@Suite struct SingleSliceErrorNormalizationTests {
    // `errors` are py4DSTEM's: divided by mean_intensity * positions, with the mean from preprocessing (negatives counted).
    @Test func errorsFromRawCubeThroughPreprocessing() async throws {
        let p = try pin()
        let cube = try MemoryCube(shape: CubeShape(scanRows: 8, scanColumns: 8, detectorRows: p.rows, detectorColumns: p.columns),
                                  values: p.preprocessing.raw.map { Float($0) })
        let a = try await PtychographyPreparation.amplitudes(
            from: cube, origins: .uniform(row: p.preprocessing.origin[0], column: p.preprocessing.origin[1]), shift: .bilinear)
        #expect(gap(a.values, p.amplitudes) < 2e-6)
        #expect(abs(a.meanIntensity - p.preprocessing.meanIntensity) / p.preprocessing.meanIntensity < 2e-6)
        let probe = try PtychographyArray(rows:p.rows, columns:p.columns, real:p.initialProbe.real, imaginary:p.initialProbe.imaginary)
        let object = try PtychographyArray(rows:p.objectRows, columns:p.objectColumns, real:p.initialObject.real, imaginary:p.initialObject.imaginary)
        let positions = p.positions.map { PtychographyPosition(row:$0[0], column:$0[1]) }
        func run(_ n: PtychographyErrorNormalization) throws -> SingleslicePtychography.Result {
            try SingleslicePtychography.reconstruct(amplitudes:a.values, positions:positions, object:object, probe:probe,
                iterations:3, stepSize:0.35, normalizationMinimum:0.7, errorNormalization:n)
        }
        let r = try run(a.errorNormalization)
        let g = [gap(r.errors, p.errors), gap(r.object.real, p.object.real), gap(r.probe.real, p.probe.real)].max()!
        print("ERRORNORM single through preprocessing: \(g)")
        #expect(g < 2e-6)
        let old = try run(.suppliedAmplitudeEnergy)
        let relative = zip(old.errors, p.errors).map { abs($0 - $1) }.max()! / p.errors.map(abs).max()!
        #expect(relative > 5e-3)                     // the pin tells the two normalisations apart
        #expect(old.object.real == r.object.real)    // and the normalisation touches nothing else
    }
}
