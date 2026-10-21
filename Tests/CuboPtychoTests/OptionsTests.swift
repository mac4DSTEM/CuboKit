import Foundation
import Testing
import CuboPtycho

// The single-slice options the mac4DSTEM app uses: the object amplitude toggle and the higher-order probe aberration terms.
// Fixtures: tools/ptycho/options_pins.py (Fixtures/options.json; its engine input is Fixtures/ptycho.json).
private struct CPin: Decodable { let real: [Double]; let imaginary: [Double] }
private struct Source: Decodable {
    let preprocessing: Pre, rows: Int, columns: Int, objectRows: Int, objectColumns: Int
    let positions: [[Double]], amplitudes: [Double], initialProbe: CPin
    struct Pre: Decodable { let meanIntensity: Double }
}
private struct AmplitudeCase: Decodable {
    let name: String, fixProbe: Bool, purePhase: Bool, step: Double, iterations: Int, maxAmplitude: Double
    let object: CPin, probe: CPin, errors: [Double]
}
private struct Polar: Decodable { let n: Int, m: Int, magnitude: Double, azimuth: Double }
private struct ProbeCase: Decodable {
    let name: String, terms: [Polar], defocus: Double, astigmatism: Double, astigmatismAngle: Double, sphericalAberration: Double, probe: CPin
}
private struct AppTerm: Decodable { let radialOrder: Int, angularOrder: Int, sine: Bool, coefficient: Double }
private struct AppCase: Decodable { let name: String, defocus: Double, c12a: Double, c12b: Double, terms: [AppTerm], probe: CPin }
private struct Pin: Decodable {
    struct Amp: Decodable { let startObject: CPin, cases: [AmplitudeCase] }
    struct Probes: Decodable {
        let rows: Int, columns: Int, rowSampling: Double, columnSampling: Double, wavelength: Double, cutoff: Double, rolloff: Double
        let cases: [ProbeCase]
    }
    struct Initial: Decodable {
        let rows: Int, columns: Int, meanIntensity: Double, semiangle: Double, rolloff: Double, defocus: Double, astigmatism: Double
        let astigmatismAngle: Double, sphericalAberration: Double, terms: [Polar], samplingRow: Double, samplingColumn: Double
        let wavelength: Double, real: [Double], imaginary: [Double]
    }
    let objectAmplitude: Amp, probes: Probes, initialProbe: Initial, appTranscription: [AppCase]
}
private func fixture<T: Decodable>(_ name: String, _: T.Type) throws -> T {
    let path = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/\(name)")
    return try JSONDecoder().decode(T.self, from: Data(contentsOf: path))
}
private func gap(_ a: [Double], _ b: [Double]) -> Double {
    guard a.count == b.count, a.allSatisfy(\.isFinite) else { return .infinity }
    return zip(a,b).map { abs($0-$1) }.max()! / max(1, b.map(abs).max()!)
}
private func terms(_ polar: [Polar]) -> [PtychographyAberrationTerm] {
    polar.flatMap { PtychographyAberrationTerm.polar(radialOrder: $0.n, angularOrder: $0.m, magnitude: $0.magnitude, azimuth: $0.azimuth) }
}

@Suite struct ObjectAmplitudeToggleTests {
    private func run(_ s: Source, _ start: CPin, _ c: AmplitudeCase, constrain: Bool?, pure: Bool? = nil) throws -> SingleslicePtychography.Result {
        let probe = try PtychographyArray(rows: s.rows, columns: s.columns, real: s.initialProbe.real, imaginary: s.initialProbe.imaginary)
        let object = try PtychographyArray(rows: s.objectRows, columns: s.objectColumns, real: start.real, imaginary: start.imaginary)
        let positions = s.positions.map { PtychographyPosition(row: $0[0], column: $0[1]) }
        let mean = PtychographyErrorNormalization.meanIntensity(s.preprocessing.meanIntensity)
        if let constrain {
            return try SingleslicePtychography.reconstruct(amplitudes: s.amplitudes, positions: positions, object: object, probe: probe,
                iterations: c.iterations, stepSize: c.step, normalizationMinimum: 0.7, fixProbe: c.fixProbe, purePhaseObject: pure ?? c.purePhase,
                constrainObjectAmplitude: constrain, errorNormalization: mean)
        }
        return try SingleslicePtychography.reconstruct(amplitudes: s.amplitudes, positions: positions, object: object, probe: probe,
            iterations: c.iterations, stepSize: c.step, normalizationMinimum: 0.7, fixProbe: c.fixProbe, purePhaseObject: pure ?? c.purePhase,
            errorNormalization: mean)
    }
    @Test func uncappedObjectAgainstSingleslice() throws {
        let s = try fixture("ptycho.json", Source.self), p = try fixture("options.json", Pin.self).objectAmplitude
        for c in p.cases {
            let r = try run(s, p.startObject, c, constrain: false)
            let g = [gap(r.object.real, c.object.real), gap(r.object.imaginary, c.object.imaginary),
                     gap(r.probe.real, c.probe.real), gap(r.probe.imaginary, c.probe.imaginary), gap(r.errors, c.errors)].max()!
            print("PTYCHO uncapped object \(c.name) gap: \(g)")
            #expect(g < 2e-6, "\(c.name)")
            if !c.purePhase {
                // The cap would have changed the result: the pinned run really leaves |object| > 1, and so does the kit's.
                let amplitude = zip(r.object.real, r.object.imaginary).map { hypot($0, $1) }.max()!
                #expect(amplitude > 1.002 && abs(amplitude - c.maxAmplitude) < 1e-5)
                let capped = try run(s, p.startObject, c, constrain: nil)
                #expect(zip(capped.object.real, capped.object.imaginary).map { hypot($0, $1) }.max()! <= 1 + 1e-12)
                #expect(gap(capped.object.real, r.object.real) > 1e-3)
            }
        }
    }
    @Test func defaultIsTheCapAndPurePhaseWins() throws {
        let s = try fixture("ptycho.json", Source.self), p = try fixture("options.json", Pin.self).objectAmplitude
        let c = p.cases[0]
        let omitted = try run(s, p.startObject, c, constrain: nil), explicit = try run(s, p.startObject, c, constrain: true)
        #expect(omitted.object.real == explicit.object.real && omitted.object.imaginary == explicit.object.imaginary)
        #expect(omitted.probe.real == explicit.probe.real && omitted.errors == explicit.errors)
        // purePhaseObject gives unit amplitude whatever the toggle says (the app's `constrain || pure`).
        let pure = p.cases[2]
        let a = try run(s, p.startObject, pure, constrain: true), b = try run(s, p.startObject, pure, constrain: false)
        #expect(a.object.real == b.object.real && a.object.imaginary == b.object.imaginary)
    }
}

@Suite struct HigherOrderProbeTests {
    private func build(_ g: Pin.Probes, _ c: ProbeCase) throws -> PtychographyArray {
        try PtychographyProbe.build(rows: g.rows, columns: g.columns, rowSampling: g.rowSampling, columnSampling: g.columnSampling,
            wavelength: g.wavelength, cutoff: g.cutoff, rolloff: g.rolloff, defocus: c.defocus, astigmatism: c.astigmatism,
            astigmatismAngle: c.astigmatismAngle, sphericalAberration: c.sphericalAberration, higherOrder: terms(c.terms))
    }
    @Test func polarTermsAgainstComplexProbe() throws {
        let g = try fixture("options.json", Pin.self).probes
        #expect(g.cases.count == 12)
        for c in g.cases {
            let probe = try build(g, c)
            let e = max(gap(probe.real, c.probe.real), gap(probe.imaginary, c.probe.imaginary))
            print("PTYCHO higher-order \(c.name) gap: \(e)")
            #expect(e < 2e-6, "\(c.name)")
            // The term is not a no-op: without it the probe is far from the pinned one.
            let plain = try PtychographyProbe.build(rows: g.rows, columns: g.columns, rowSampling: g.rowSampling, columnSampling: g.columnSampling,
                wavelength: g.wavelength, cutoff: g.cutoff, rolloff: g.rolloff, defocus: c.defocus, astigmatism: c.astigmatism,
                astigmatismAngle: c.astigmatismAngle, sphericalAberration: c.sphericalAberration)
            #expect(max(gap(plain.real, c.probe.real), gap(plain.imaginary, c.probe.imaginary)) > 1e-3, "\(c.name)")
        }
    }
    @Test func startingProbeAgainstInitializeProbe() throws {
        let q = try fixture("options.json", Pin.self).initialProbe
        let probe = try PtychographyPreparation.initialProbe(rows: q.rows, columns: q.columns, samplingRow: q.samplingRow,
            samplingColumn: q.samplingColumn, wavelength: q.wavelength, semiangle: q.semiangle, rolloff: q.rolloff, defocus: q.defocus,
            astigmatism: q.astigmatism, astigmatismAngle: q.astigmatismAngle, sphericalAberration: q.sphericalAberration,
            higherOrder: terms(q.terms), meanIntensity: q.meanIntensity)
        let e = max(gap(probe.real, q.real), gap(probe.imaginary, q.imaginary))
        print("PTYCHO higher-order starting probe gap: \(e)")
        #expect(e < 2e-6)
        // The preparation options carry the same terms to the same probe.
        let o = PtychographyPreparationOptions(defocus: q.defocus, astigmatism: q.astigmatism, astigmatismAngle: q.astigmatismAngle,
                                               sphericalAberration: q.sphericalAberration, higherOrderAberrations: terms(q.terms))
        #expect(o.higherOrderAberrations == terms(q.terms))
        #expect(PtychographyPreparationOptions().higherOrderAberrations.isEmpty)
    }
    @Test func appFormulaTermsAgainstTranscription() throws {
        let p = try fixture("options.json", Pin.self)
        let g = p.probes
        for c in p.appTranscription {
            // The app enters defocus and the cartesian astigmatism (C12a, C12b); the kit's polar C12 / phi12 is the same function.
            let magnitude = hypot(c.c12a, c.c12b), azimuth = 0.5 * atan2(c.c12b, c.c12a)
            let higher = c.terms.map { PtychographyAberrationTerm(radialOrder: $0.radialOrder, angularOrder: $0.angularOrder,
                                                                  component: $0.sine ? .sine : .cosine, coefficient: $0.coefficient) }
            let probe = try PtychographyProbe.build(rows: g.rows, columns: g.columns, rowSampling: g.rowSampling, columnSampling: g.columnSampling,
                wavelength: g.wavelength, cutoff: g.cutoff, rolloff: g.rolloff, defocus: c.defocus, astigmatism: magnitude,
                astigmatismAngle: azimuth, higherOrder: higher)
            let e = max(gap(probe.real, c.probe.real), gap(probe.imaginary, c.probe.imaginary))
            print("PTYCHO app-formula terms \(c.name) gap: \(e)")
            #expect(e < 2e-6, "\(c.name)")
        }
    }
    @Test func emptyTermsLeaveTheProbeUnchangedAndInvalidTermsAreRefused() throws {
        let g = try fixture("options.json", Pin.self).probes
        func make(_ t: [PtychographyAberrationTerm]?) throws -> PtychographyArray {
            if let t {
                return try PtychographyProbe.build(rows: g.rows, columns: g.columns, rowSampling: g.rowSampling, columnSampling: g.columnSampling,
                    wavelength: g.wavelength, cutoff: g.cutoff, defocus: 70, astigmatism: 25, astigmatismAngle: 0.31, sphericalAberration: 2e4, higherOrder: t)
            }
            return try PtychographyProbe.build(rows: g.rows, columns: g.columns, rowSampling: g.rowSampling, columnSampling: g.columnSampling,
                wavelength: g.wavelength, cutoff: g.cutoff, defocus: 70, astigmatism: 25, astigmatismAngle: 0.31, sphericalAberration: 2e4)
        }
        let a = try make(nil), b = try make([]), z = try make([.init(radialOrder: 2, angularOrder: 1, coefficient: 0)])
        #expect(a.real == b.real && a.imaginary == b.imaginary && a.real == z.real && a.imaginary == z.imaginary)
        let bad: [PtychographyAberrationTerm] = [
            .init(radialOrder: 0, angularOrder: 0, coefficient: 1), .init(radialOrder: 2, angularOrder: -1, coefficient: 1),
            .init(radialOrder: 2, angularOrder: 0, component: .sine, coefficient: 1), .init(radialOrder: 2, angularOrder: 1, coefficient: .nan),
            .init(radialOrder: 2, angularOrder: 1, coefficient: .infinity)]
        for t in bad { #expect(throws: PtychographyError.self) { try make([t]) } }
        // polar() is py4DSTEM's C cos(m (phi - phi0)): one symmetric term for m = 0, a cosine and a sine term otherwise.
        #expect(PtychographyAberrationTerm.polar(radialOrder: 5, angularOrder: 0, magnitude: 3, azimuth: 1).count == 1)
        let two = PtychographyAberrationTerm.polar(radialOrder: 2, angularOrder: 3, magnitude: 2, azimuth: 0.2)
        #expect(two.count == 2 && abs(two[0].coefficient - 2 * cos(0.6)) < 1e-15 && abs(two[1].coefficient - 2 * sin(0.6)) < 1e-15)
    }
}
