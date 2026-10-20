import Foundation
import Testing
import CuboPtycho

private struct Cx: Decodable { let real: [Double]; let imaginary: [Double] }
private struct MixedCase: Decodable {
    let name: String, numProbes: Int, fixProbeCenterOfMass: Bool, orthogonalize: Bool, constrainProbeAmplitude: Bool
    let radius: Double, width: Double, fixProbe: Bool, purePhase: Bool
    let object: Cx, probes: Cx, errors: [Double]
}
private struct Orth: Decodable { let input: Cx, output: Cx, count: Int }
private struct Init: Decodable { let seed: UInt32, count: Int, first: Cx, meanIntensity: Double, states: Cx, draws: [Double] }
private struct Mixed: Decodable { let amplitudes: [Double], startStates: Cx, cases: [MixedCase], orthogonalization: Orth, inits: [Init] }
private struct SliceCase: Decodable {
    let name: String, slices: Int, thicknesses: [Double], fixProbeCenterOfMass: Bool, constrainProbeAmplitude: Bool
    let radius: Double, width: Double, fixProbe: Bool, purePhase: Bool, identicalSlices: Bool
    let object: Cx, probe: Cx, errors: [Double]
}
private struct Multi: Decodable {
    let sliceThicknesses: [Double], propagators: Cx, amplitudes: [Double], startSlices: Cx, startProbe: Cx, cases: [SliceCase]
}
private struct ExtraCase: Decodable {
    let engine: String, name: String, fixProbeCenterOfMass: Bool, fixProbeAperture: Bool, fixProbe: Bool, constrainProbeAmplitude: Bool
    let radius: Double, width: Double, object: Cx, probe: Cx, errors: [Double]
}
private struct Extras: Decodable { let support: [Bool], cases: [ExtraCase] }
private struct Pin: Decodable {
    let extras: Extras
    let rows: Int, columns: Int, objectRows: Int, objectColumns: Int, rowSampling: Double, columnSampling: Double
    let wavelength: Double, positions: [[Double]], initialObject: Cx, mixed: Mixed, multislice: Multi
}
private func pin() throws -> Pin {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/engines.json")
    return try JSONDecoder().decode(Pin.self, from: Data(contentsOf: url))
}
private func gap(_ a: [Double], _ b: [Double]) -> Double {
    guard a.count == b.count else { return .infinity }
    return zip(a, b).map { abs($0 - $1) }.max()! / max(1, b.map(abs).max()!)
}
private func positions(_ p: Pin) -> [PtychographyPosition] { p.positions.map { .init(row: $0[0], column: $0[1]) } }
/// Normwise gap of a complex field: max |a - b| (real and imaginary) over max(1, max |b|).
private func gap(_ a: PtychographyArray, _ b: Cx) -> Double { max(gap(a.real, b.real), gap(a.imaginary, b.imaginary)) }
private func gap(_ a: PtychographyStack, _ b: Cx) -> Double { max(gap(a.real, b.real), gap(a.imaginary, b.imaginary)) }

/// Rotates each state of `a` by the global phase that best matches `b` (the eigenvector phase convention is free).
private func phaseAligned(_ a: PtychographyStack, to b: Cx) -> PtychographyStack {
    let n = a.rows * a.columns
    var re = a.real, im = a.imaginary
    for s in 0..<a.count {
        var sr = 0.0, si = 0.0          // sum conj(b) a
        for k in 0..<n {
            let x = a.real[s*n+k], y = a.imaginary[s*n+k], u = b.real[s*n+k], v = b.imaginary[s*n+k]
            sr += u*x + v*y; si += u*y - v*x
        }
        let norm = hypot(sr, si)
        guard norm > 0 else { continue }
        let c = sr/norm, d = -si/norm       // multiply by e^{-i angle}
        for k in 0..<n {
            let x = a.real[s*n+k], y = a.imaginary[s*n+k]
            re[s*n+k] = x*c - y*d; im[s*n+k] = x*d + y*c
        }
    }
    return try! PtychographyStack(count: a.count, rows: a.rows, columns: a.columns, real: re, imaginary: im)
}
private func stack(_ c: Cx, count: Int, rows: Int, columns: Int) throws -> PtychographyStack {
    try PtychographyStack(count: count, rows: rows, columns: columns, real: c.real, imaginary: c.imaginary)
}
private func gramOffDiagonal(_ s: PtychographyStack) -> Double {
    let n = s.rows * s.columns
    var worst = 0.0, scale = 0.0
    for i in 0..<s.count { for j in 0..<s.count {
        var re = 0.0, im = 0.0
        for k in 0..<n {
            let a = s.real[i*n+k], b = s.imaginary[i*n+k], c = s.real[j*n+k], d = s.imaginary[j*n+k]
            re += a*c + b*d; im += a*d - b*c
        }
        if i == j { scale = max(scale, re) } else { worst = max(worst, hypot(re, im)) }
    } }
    return worst / scale
}

@Suite struct MixedstateTests {
    // Python state is complex64 and its eigenvectors come from complex64 LAPACK; the engine is Double.
    // Bound: 2e-6 (max abs error over max(1, max |oracle|)) on the object, the errors and the (phase-aligned) probes.
    @Test func reconstructionAgainstMixedstatePtychography() throws {
        let p = try pin(), m = p.mixed
        let start = try stack(m.startStates, count: 3, rows: p.rows, columns: p.columns)
        let object = try PtychographyArray(rows: p.objectRows, columns: p.objectColumns, real: p.initialObject.real, imaginary: p.initialObject.imaginary)
        for c in m.cases {
            let probes = try PtychographyStack((0..<c.numProbes).map { try start.array(at: $0) })
            let r = try MixedstatePtychography.reconstruct(
                amplitudes: m.amplitudes, positions: positions(p), object: object, probes: probes, iterations: 3,
                stepSize: 0.35, normalizationMinimum: 0.7, fixProbe: c.fixProbe, purePhaseObject: c.purePhase,
                fixProbeCenterOfMass: c.fixProbeCenterOfMass, constrainProbeAmplitude: c.constrainProbeAmplitude,
                probeAmplitudeRelativeRadius: c.radius, probeAmplitudeRelativeWidth: c.width, orthogonalizeProbes: c.orthogonalize)
            let aligned = phaseAligned(r.probes, to: c.probes)
            let g = [gap(r.object, c.object), gap(r.errors, c.errors), gap(aligned, c.probes)]
            print("MIXED \(c.name) object \(g[0]) errors \(g[1]) probes \(g[2])")
            #expect(g.max()! < 2e-6)
            #expect(r.probes.count == c.numProbes)
            if !c.orthogonalize {      // no eigenvectors involved: the phases are not free
                #expect(gap(r.probes, c.probes) < 2e-6)
            }
        }
    }

    @Test func orthogonalizationAgainstProbeOrthogonalizationConstraint() throws {
        let p = try pin(), o = p.mixed.orthogonalization
        let input = try stack(o.input, count: o.count, rows: p.rows, columns: p.columns)
        let out = try MixedstatePtychography.orthogonalize(input)
        let g = gap(phaseAligned(out, to: o.output), o.output)
        print("MIXED orthogonalization gap \(g), Gram off-diagonal \(gramOffDiagonal(out))")
        #expect(g < 2e-6)
        // truth, independent of the reference: the states are mutually orthogonal and the total intensity is kept
        #expect(gramOffDiagonal(out) < 1e-13)
        let before = input.real.reduce(0) { $0 + $1*$1 } + input.imaginary.reduce(0) { $0 + $1*$1 }
        let after = out.real.reduce(0) { $0 + $1*$1 } + out.imaginary.reduce(0) { $0 + $1*$1 }
        #expect(abs(after - before) < 1e-12 * before)
        // intensities decrease
        let n = out.rows * out.columns
        let energy = (0..<out.count).map { s in (0..<n).reduce(0.0) { $0 + out.real[s*n+$1]*out.real[s*n+$1] + out.imaginary[s*n+$1]*out.imaginary[s*n+$1] } }
        #expect(zip(energy, energy.dropFirst()).allSatisfy { $0 >= $1 })
    }

    // The NumPy-compatible generator: the first draws of np.random.seed(seed) (indices and bit patterns: exact), then py4DSTEM's states.
    @Test func initialProbesAgainstInitializeProbe() throws {
        let p = try pin()
        for c in p.mixed.inits {
            // py4DSTEM scales the first state so that sum |fft2|^2 equals the mean intensity (Parseval: N sum |p|^2)
            let n = p.rows * p.columns
            let energy = zip(c.first.real, c.first.imaginary).reduce(0) { $0 + $1.0*$1.0 + $1.1*$1.1 } * Double(n)
            let scale = (c.meanIntensity / energy).squareRoot()
            let first = try PtychographyArray(rows: p.rows, columns: p.columns, real: c.first.real.map { $0 * scale },
                                              imaginary: c.first.imaginary.map { $0 * scale })
            let states = try MixedstatePtychography.initialProbes(from: first, count: c.count, seed: c.seed)
            let g = gap(states, c.states)
            print("MIXED init seed \(c.seed) count \(c.count) gap \(g)")
            #expect(g < 2e-6)
            #expect(states.count == c.count && states.real.count == c.count * n)
        }
        // the generator itself: NumPy's np.random.seed(0); np.random.rand() is 0.5488135039273248, then 0.7151893663724195
        let draws = p.mixed.inits[0].draws
        #expect(draws[0] == 0.5488135039273248 && draws[1] == 0.7151893663724195)
    }

    // The draws the kit makes are NumPy's: the second state of seed 0 is the first state times the phase ramps of
    // u = 0.5488135039273248 (rows) and v = 0.7151893663724195 (columns), computed here from those literals.
    @Test func initialProbesUseNumPysDraws() throws {
        let n = 8
        let ones = try PtychographyArray(rows: n, columns: n, real: [Double](repeating: 1, count: n*n), imaginary: [Double](repeating: 0, count: n*n))
        let states = try MixedstatePtychography.initialProbes(from: ones, count: 2, seed: 0)
        let u = 0.5488135039273248, v = 0.7151893663724195
        func f(_ k: Int) -> Double { Double(k < (n+1)/2 ? k : k-n)/Double(n) }
        for r in 0..<n { for c in 0..<n {
            let phase = -2 * Double.pi * ((u-0.5)*f(r) + (v-0.5)*f(c))
            #expect(abs(states.real[n*n + r*n+c] - cos(phase)) < 1e-14 && abs(states.imaginary[n*n + r*n+c] - sin(phase)) < 1e-14)
        } }
    }

    @Test func fixedProbeLeavesTheStatesAlone() throws {
        let p = try pin()
        let start = try stack(p.mixed.startStates, count: 3, rows: p.rows, columns: p.columns)
        let object = try PtychographyArray(rows: p.objectRows, columns: p.objectColumns, real: p.initialObject.real, imaginary: p.initialObject.imaginary)
        let r = try MixedstatePtychography.reconstruct(amplitudes: p.mixed.amplitudes, positions: positions(p), object: object,
                                                       probes: start, iterations: 2, fixProbe: true, fixProbeCenterOfMass: true,
                                                       constrainProbeAmplitude: true, orthogonalizeProbes: false)
        #expect(r.probes.real == start.real && r.probes.imaginary == start.imaginary)
    }

    @Test func invalidInputsAreRefused() throws {
        let p = try pin()
        let start = try stack(p.mixed.startStates, count: 3, rows: p.rows, columns: p.columns)
        let object = try PtychographyArray(rows: p.objectRows, columns: p.objectColumns, real: p.initialObject.real, imaginary: p.initialObject.imaginary)
        #expect(throws: PtychographyError.self) {
            try MixedstatePtychography.reconstruct(amplitudes: [], positions: [], object: object, probes: start)
        }
        #expect(throws: PtychographyError.self) {
            try MixedstatePtychography.reconstruct(amplitudes: p.mixed.amplitudes, positions: positions(p), object: object,
                                                   probes: start, constrainProbeAmplitude: true, probeAmplitudeRelativeRadius: 0.9)
        }
        #expect(throws: PtychographyError.self) { try PtychographyStack(count: 0, rows: 4, columns: 4, real: [], imaginary: []) }
        #expect(throws: PtychographyError.self) { try MixedstatePtychography.initialProbes(from: object, count: 0, seed: 1) }
    }
}

@Suite struct MultisliceTests {
    // Propagators: py4DSTEM builds them from Float32 frequencies in complex64.
    @Test func propagatorsAgainstPrecomputePropagatorArrays() throws {
        let p = try pin(), m = p.multislice
        let props = try MultislicePtychography.propagators(rows: p.rows, columns: p.columns, rowSampling: p.rowSampling,
                                                           columnSampling: p.columnSampling, wavelength: p.wavelength,
                                                           sliceThicknesses: m.sliceThicknesses)
        let g = gap(props, m.propagators)
        print("MULTISLICE propagator gap \(g)")
        #expect(g < 2e-6)
        // truth: a propagator is a unit-modulus phase
        #expect(zip(props.real, props.imaginary).allSatisfy { abs(hypot($0, $1) - 1) < 1e-14 })
    }

    @Test func reconstructionAgainstMultislicePtychography() throws {
        let p = try pin(), m = p.multislice
        let probe = try PtychographyArray(rows: p.rows, columns: p.columns, real: m.startProbe.real, imaginary: m.startProbe.imaginary)
        let plane = p.objectRows * p.objectColumns
        for c in m.cases {
            let object = try PtychographyStack(count: c.slices, rows: p.objectRows, columns: p.objectColumns,
                                               real: Array(m.startSlices.real[0..<c.slices*plane]),
                                               imaginary: Array(m.startSlices.imaginary[0..<c.slices*plane]))
            let r = try MultislicePtychography.reconstruct(
                amplitudes: m.amplitudes, positions: positions(p), object: object, probe: probe, sliceThicknesses: c.thicknesses,
                wavelength: p.wavelength, rowSampling: p.rowSampling, columnSampling: p.columnSampling, iterations: 3,
                stepSize: 0.35, normalizationMinimum: 0.7, fixProbe: c.fixProbe, purePhaseObject: c.purePhase,
                identicalSlices: c.identicalSlices, fixProbeCenterOfMass: c.fixProbeCenterOfMass,
                constrainProbeAmplitude: c.constrainProbeAmplitude, probeAmplitudeRelativeRadius: c.radius,
                probeAmplitudeRelativeWidth: c.width)
            let g = [gap(r.object, c.object), gap(r.probe, c.probe), gap(r.errors, c.errors)]
            print("MULTISLICE \(c.name) object \(g[0]) probe \(g[1]) errors \(g[2])")
            #expect(g.max()! < 2e-6)
        }
    }

    // With one slice the multislice engine is the single-slice engine (no propagator is used).
    @Test func oneSliceEqualsSingleslice() throws {
        let p = try pin(), m = p.multislice
        let probe = try PtychographyArray(rows: p.rows, columns: p.columns, real: m.startProbe.real, imaginary: m.startProbe.imaginary)
        let plane = p.objectRows * p.objectColumns
        let slice = try PtychographyArray(rows: p.objectRows, columns: p.objectColumns,
                                          real: Array(m.startSlices.real[0..<plane]), imaginary: Array(m.startSlices.imaginary[0..<plane]))
        let single = try SingleslicePtychography.reconstruct(amplitudes: m.amplitudes, positions: positions(p), object: slice,
                                                             probe: probe, iterations: 3, stepSize: 0.35, normalizationMinimum: 0.7)
        let multi = try MultislicePtychography.reconstruct(
            amplitudes: m.amplitudes, positions: positions(p), object: PtychographyStack([slice]), probe: probe,
            sliceThicknesses: [], wavelength: p.wavelength, rowSampling: p.rowSampling, columnSampling: p.columnSampling,
            iterations: 3, stepSize: 0.35, normalizationMinimum: 0.7)
        let g = max(gap(multi.object.real, single.object.real), gap(multi.object.imaginary, single.object.imaginary),
                    gap(multi.probe.real, single.probe.real), gap(multi.errors, single.errors))
        print("MULTISLICE one-slice vs single-slice gap \(g)")
        #expect(g < 1e-12)
    }

    @Test func invalidInputsAreRefused() throws {
        let p = try pin(), m = p.multislice
        let probe = try PtychographyArray(rows: p.rows, columns: p.columns, real: m.startProbe.real, imaginary: m.startProbe.imaginary)
        let plane = p.objectRows * p.objectColumns
        let object = try PtychographyStack(count: 3, rows: p.objectRows, columns: p.objectColumns,
                                           real: Array(m.startSlices.real[0..<3*plane]), imaginary: Array(m.startSlices.imaginary[0..<3*plane]))
        // thickness count must be slices - 1
        #expect(throws: PtychographyError.self) {
            try MultislicePtychography.reconstruct(amplitudes: m.amplitudes, positions: positions(p), object: object, probe: probe,
                                                   sliceThicknesses: [10], wavelength: p.wavelength, rowSampling: 0.8, columnSampling: 1.1)
        }
        #expect(throws: PtychographyError.self) {
            try MultislicePtychography.propagators(rows: 8, columns: 8, rowSampling: 0, columnSampling: 1, wavelength: 0.02, sliceThicknesses: [1])
        }
    }
}

@Suite struct ProbeSupportAndApertureTests {
    // Fixed-aperture and real-space-support constraints on all three engines (2e-6, as the other engine pins).
    @Test func apertureAndSupportAgainstPy4DSTEM() throws {
        let p = try pin(), m = p.multislice, x = p.extras
        let plane = p.objectRows * p.objectColumns
        let startProbe = try PtychographyArray(rows: p.rows, columns: p.columns, real: m.startProbe.real, imaginary: m.startProbe.imaginary)
        let startObject = try PtychographyArray(rows: p.objectRows, columns: p.objectColumns, real: p.initialObject.real, imaginary: p.initialObject.imaginary)
        let mixedStart = try stack(p.mixed.startStates, count: 3, rows: p.rows, columns: p.columns)
        for c in x.cases {
            let g: Double
            switch c.engine {
            case "single":
                let r = try SingleslicePtychography.reconstruct(
                    amplitudes: m.amplitudes, positions: positions(p), object: startObject, probe: startProbe, iterations: 3,
                    stepSize: 0.35, normalizationMinimum: 0.7, fixProbe: c.fixProbe, fixProbeCenterOfMass: c.fixProbeCenterOfMass,
                    constrainProbeAmplitude: c.constrainProbeAmplitude, probeAmplitudeRelativeRadius: c.radius,
                    probeAmplitudeRelativeWidth: c.width, fixProbeAperture: c.fixProbeAperture, probeSupportMask: x.support)
                g = [gap(r.object, c.object), gap(r.probe, c.probe), gap(r.errors, c.errors)].max()!
            case "mixed":
                let probes = try PtychographyStack((0..<2).map { try mixedStart.array(at: $0) })
                let r = try MixedstatePtychography.reconstruct(
                    amplitudes: p.mixed.amplitudes, positions: positions(p), object: startObject, probes: probes, iterations: 3,
                    stepSize: 0.35, normalizationMinimum: 0.7, fixProbe: c.fixProbe, fixProbeCenterOfMass: c.fixProbeCenterOfMass,
                    constrainProbeAmplitude: c.constrainProbeAmplitude, probeAmplitudeRelativeRadius: c.radius,
                    probeAmplitudeRelativeWidth: c.width, fixProbeAperture: c.fixProbeAperture, probeSupportMask: x.support)
                g = [gap(r.object, c.object), gap(phaseAligned(r.probes, to: c.probe), c.probe), gap(r.errors, c.errors)].max()!
            default:
                let object = try PtychographyStack(count: 3, rows: p.objectRows, columns: p.objectColumns,
                                                   real: Array(m.startSlices.real[0..<3*plane]), imaginary: Array(m.startSlices.imaginary[0..<3*plane]))
                let r = try MultislicePtychography.reconstruct(
                    amplitudes: m.amplitudes, positions: positions(p), object: object, probe: startProbe, sliceThicknesses: [55, 80],
                    wavelength: p.wavelength, rowSampling: p.rowSampling, columnSampling: p.columnSampling, iterations: 3,
                    stepSize: 0.35, normalizationMinimum: 0.7, fixProbe: c.fixProbe, fixProbeCenterOfMass: c.fixProbeCenterOfMass,
                    constrainProbeAmplitude: c.constrainProbeAmplitude, probeAmplitudeRelativeRadius: c.radius,
                    probeAmplitudeRelativeWidth: c.width, fixProbeAperture: c.fixProbeAperture, probeSupportMask: x.support)
                g = [gap(r.object, c.object), gap(r.probe, c.probe), gap(r.errors, c.errors)].max()!
            }
            print("EXTRAS \(c.engine) \(c.name) gap \(g)")
            #expect(g < 2e-6)
        }
    }

    @Test func supportMustMatchTheProbe() throws {
        let p = try pin(), m = p.multislice
        let probe = try PtychographyArray(rows: p.rows, columns: p.columns, real: m.startProbe.real, imaginary: m.startProbe.imaginary)
        let object = try PtychographyArray(rows: p.objectRows, columns: p.objectColumns, real: p.initialObject.real, imaginary: p.initialObject.imaginary)
        #expect(throws: PtychographyError.self) {
            try SingleslicePtychography.reconstruct(amplitudes: m.amplitudes, positions: positions(p), object: object, probe: probe,
                                                    probeSupportMask: [true])
        }
    }
}
