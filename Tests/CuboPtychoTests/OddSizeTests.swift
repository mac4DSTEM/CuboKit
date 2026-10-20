import Foundation
import Testing
import CuboCore
import CuboPtycho

// Odd and non-square sizes: preprocessing of a 9 x 7 detector, and the three engines on a 15 x 13 probe and a 31 x 29 object
// (tools/ptycho/odd_pins.py). Odd lengths have no Nyquist bin, so the frequency grid, the corner-centred patch offsets, the
// Fourier shifts, the centre of mass and the top-hat all change. Tolerance 2e-6 (max abs over max(1, max |oracle|)).

private struct Cx: Decodable { let real: [Double]; let imaginary: [Double] }
private struct AmpCase: Decodable { let amplitudes: [Double], meanIntensity: Double, clippedMeanIntensity: Double }
private struct PosCase: Decodable {
    let angle: Double, transpose: Bool, positions: [[Double]], padding: [[Double]], objectShape: [Int], centred: [[Double]]
}
private struct ProbeCase: Decodable {
    let name: String, semiangle: Double, rolloff: Double, defocus: Double, astigmatism: Double
    let astigmatismAngle: Double, sphericalAberration: Double, meanIntensity: Double, real: [Double], imaginary: [Double]
}
private struct Prep: Decodable {
    let scanRows: Int, scanColumns: Int, detectorRows: Int, detectorColumns: Int
    let cube: [Double], originRows: [Double], originColumns: [Double], amplitudes: [String: AmpCase]
    let wavelength: Double, angularSampling: [Double], objectSampling: [Double], scanSampling: [Double]
    let positionCases: [PosCase], probes: [ProbeCase]
}
private struct Pre: Decodable { let raw: [Double], origin: [Double], meanIntensity: Double, energyPerPosition: Double }
private struct Case: Decodable {
    let name: String, fixProbeCenterOfMass: Bool, constrainProbeAmplitude: Bool, radius: Double, width: Double
    let aperture: Bool, support: Bool, errors: [Double]
    let object: Cx
    let probe: Cx?, probes: Cx?
    let numProbes: Int?, orthogonalize: Bool?, slices: Int?, thicknesses: [Double]?
}
private struct Init: Decodable { let seed: UInt32, count: Int, first: Cx, meanIntensity: Double, states: Cx }
private struct Eng: Decodable {
    let rows: Int, columns: Int, objectRows: Int, objectColumns: Int, rowSampling: Double, columnSampling: Double, wavelength: Double
    let positions: [[Double]], initialObject: Cx, support: [Bool], startProbe: Cx, startStates: Cx, startSlices: Cx
    let mixedPreprocessing: Pre, mixedAmplitudes: [Double], multislicePreprocessing: Pre, amplitudes: [Double]
    let propagators: Cx, sliceThicknesses: [Double]
    let single: [Case], mixed: [Case], multislice: [Case], `init`: Init, builtProbe: Cx
}
private struct Pin: Decodable { let preprocessing: Prep, engines: Eng }
private func pin() throws -> Pin {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/odd.json")
    return try JSONDecoder().decode(Pin.self, from: Data(contentsOf: url))
}
private func gap(_ a: [Double], _ b: [Double]) -> Double {
    guard a.count == b.count, a.allSatisfy(\.isFinite) else { return .infinity }   // a NaN must fail, max() would skip it
    return zip(a, b).map { abs($0 - $1) }.max()! / max(1, b.map(abs).max()!)
}
private func gap(_ a: PtychographyArray, _ b: Cx) -> Double { max(gap(a.real, b.real), gap(a.imaginary, b.imaginary)) }
private func gap(_ a: PtychographyStack, _ b: Cx) -> Double { max(gap(a.real, b.real), gap(a.imaginary, b.imaginary)) }
private func relative(_ a: Double, _ b: Double) -> Double { abs(a - b) / max(1e-300, abs(b)) }
private func positions(_ e: Eng) -> [PtychographyPosition] { e.positions.map { .init(row: $0[0], column: $0[1]) } }
private func array(_ c: Cx, _ rows: Int, _ columns: Int) throws -> PtychographyArray {
    try PtychographyArray(rows: rows, columns: columns, real: c.real, imaginary: c.imaginary)
}
private func phaseAligned(_ a: PtychographyStack, to b: Cx) -> PtychographyStack {
    let n = a.rows * a.columns
    var re = a.real, im = a.imaginary
    for s in 0..<a.count {
        var sr = 0.0, si = 0.0
        for k in 0..<n {
            let x = a.real[s*n+k], y = a.imaginary[s*n+k], u = b.real[s*n+k], v = b.imaginary[s*n+k]
            sr += u*x + v*y; si += u*y - v*x
        }
        let norm = hypot(sr, si)
        guard norm > 0 else { continue }
        let c = sr/norm, d = -si/norm
        for k in 0..<n { let x = a.real[s*n+k], y = a.imaginary[s*n+k]; re[s*n+k] = x*c - y*d; im[s*n+k] = x*d + y*c }
    }
    return try! PtychographyStack(count: a.count, rows: a.rows, columns: a.columns, real: re, imaginary: im)
}

@Suite struct OddSizePreprocessingTests {
    private func source(_ p: Prep) throws -> MemoryCube {
        try MemoryCube(shape: CubeShape(scanRows: p.scanRows, scanColumns: p.scanColumns, detectorRows: p.detectorRows,
                                        detectorColumns: p.detectorColumns), values: p.cube.map { Float($0) })
    }
    @Test func amplitudesPositionsAndProbeOnAnOddDetector() async throws {
        let p = try pin().preprocessing, cube = try source(p)
        #expect(p.detectorRows % 2 == 1 && p.detectorColumns % 2 == 1)
        for (key, shift) in [("1", PtychographyShift.bilinear), ("3", .cubicSpline)] {
            let ref = p.amplitudes[key]!
            let a = try await PtychographyPreparation.amplitudes(
                from: cube, origins: .perPosition(rows: p.originRows, columns: p.originColumns), shift: shift)
            let g = gap(a.values, ref.amplitudes)
            print("ODD amplitudes order \(key) gap \(g)")
            #expect(g < 2e-6)
            #expect(relative(a.meanIntensity, ref.meanIntensity) < 2e-6)
            #expect(relative(a.clippedMeanIntensity, ref.clippedMeanIntensity) < 2e-6)
        }
        for c in p.positionCases {
            let geometry = PtychographyScanGeometry(scanSamplingRow: p.scanSampling[0], scanSamplingColumn: p.scanSampling[1],
                                                    rotation: c.angle, transpose: c.transpose)
            let pad = PtychographyPadding.reference(detectorRows: p.detectorRows, detectorColumns: p.detectorColumns)
            #expect(pad.startRow == c.padding[0][0] && pad.extentRow == c.padding[0][1])
            #expect(pad.startColumn == c.padding[1][0] && pad.extentColumn == c.padding[1][1])
            let pos = try PtychographyPreparation.scanPositions(
                scanRows: p.scanRows, scanColumns: p.scanColumns, geometry: geometry,
                objectSamplingRow: p.objectSampling[0], objectSamplingColumn: p.objectSampling[1], padding: pad)
            #expect(max(gap(pos.map(\.row), c.positions.map { $0[0] }), gap(pos.map(\.column), c.positions.map { $0[1] })) < 1e-9)
            let shape = try PtychographyPreparation.objectShape(positions: pos, detectorRows: p.detectorRows,
                                                                detectorColumns: p.detectorColumns, padding: pad)
            #expect([shape.rows, shape.columns] == c.objectShape)
            let centred = PtychographyPreparation.centerPositions(pos, objectRows: shape.rows, objectColumns: shape.columns)
            #expect(max(gap(centred.map(\.row), c.centred.map { $0[0] }), gap(centred.map(\.column), c.centred.map { $0[1] })) < 2e-6)
        }
        for c in p.probes {
            let probe = try PtychographyPreparation.initialProbe(
                rows: p.detectorRows, columns: p.detectorColumns, samplingRow: p.objectSampling[0],
                samplingColumn: p.objectSampling[1], wavelength: p.wavelength, semiangle: c.semiangle, rolloff: c.rolloff,
                defocus: c.defocus, astigmatism: c.astigmatism, astigmatismAngle: c.astigmatismAngle,
                sphericalAberration: c.sphericalAberration, meanIntensity: c.meanIntensity)
            let g = max(gap(probe.real, c.real), gap(probe.imaginary, c.imaginary))
            print("ODD probe \(c.name) gap \(g)")
            #expect(g < 2e-6)
        }
    }
}

@Suite struct OddSizeEngineTests {
    @Test func builtProbeOnAnOddGrid() throws {
        let e = try pin().engines
        let probe = try PtychographyProbe.build(rows: e.rows, columns: e.columns, rowSampling: e.rowSampling, columnSampling: e.columnSampling,
                                                wavelength: e.wavelength, cutoff: 0.018, rolloff: 0.002, defocus: 70, astigmatism: 25,
                                                astigmatismAngle: 0.31, sphericalAberration: 2e4)
        let g = gap(probe, e.builtProbe)
        print("ODD built probe gap \(g)")
        #expect(g < 2e-6)
    }

    @Test func singleSliceOnOddSizes() throws {
        let e = try pin().engines
        #expect(e.rows % 2 == 1 && e.columns % 2 == 1 && e.objectRows % 2 == 1 && e.objectColumns % 2 == 1)
        let probe = try array(e.startProbe, e.rows, e.columns), object = try array(e.initialObject, e.objectRows, e.objectColumns)
        for c in e.single {
            let r = try SingleslicePtychography.reconstruct(
                amplitudes: e.amplitudes, positions: positions(e), object: object, probe: probe, iterations: 3, stepSize: 0.35,
                normalizationMinimum: 0.7, fixProbeCenterOfMass: c.fixProbeCenterOfMass, constrainProbeAmplitude: c.constrainProbeAmplitude,
                probeAmplitudeRelativeRadius: c.radius, probeAmplitudeRelativeWidth: c.width, fixProbeAperture: c.aperture,
                probeSupportMask: c.support ? e.support : nil, errorNormalization: .meanIntensity(e.multislicePreprocessing.meanIntensity))
            let g = [gap(r.object, c.object), gap(r.probe, c.probe!), gap(r.errors, c.errors)]
            print("ODD single \(c.name) \(g)")
            #expect(g.max()! < 2e-6)
        }
    }

    @Test func mixedStateOnOddSizes() throws {
        let e = try pin().engines
        let object = try array(e.initialObject, e.objectRows, e.objectColumns)
        let start = try PtychographyStack(count: 3, rows: e.rows, columns: e.columns, real: e.startStates.real, imaginary: e.startStates.imaginary)
        for c in e.mixed {
            let probes = try PtychographyStack((0..<c.numProbes!).map { try start.array(at: $0) })
            let r = try MixedstatePtychography.reconstruct(
                amplitudes: e.mixedAmplitudes, positions: positions(e), object: object, probes: probes, iterations: 3, stepSize: 0.35,
                normalizationMinimum: 0.7, fixProbeCenterOfMass: c.fixProbeCenterOfMass, constrainProbeAmplitude: c.constrainProbeAmplitude,
                probeAmplitudeRelativeRadius: c.radius, probeAmplitudeRelativeWidth: c.width, orthogonalizeProbes: c.orthogonalize!,
                fixProbeAperture: c.aperture, probeSupportMask: c.support ? e.support : nil,
                errorNormalization: .meanIntensity(e.mixedPreprocessing.meanIntensity))
            let g = [gap(r.object, c.object), gap(r.errors, c.errors), gap(phaseAligned(r.probes, to: c.probes!), c.probes!)]
            print("ODD mixed \(c.name) \(g)")
            #expect(g.max()! < 2e-6)
        }
        let n = e.rows * e.columns
        let energy = zip(e.`init`.first.real, e.`init`.first.imaginary).reduce(0) { $0 + $1.0*$1.0 + $1.1*$1.1 } * Double(n)
        let scale = (e.`init`.meanIntensity / energy).squareRoot()
        let first = try PtychographyArray(rows: e.rows, columns: e.columns, real: e.`init`.first.real.map { $0 * scale },
                                          imaginary: e.`init`.first.imaginary.map { $0 * scale })
        let states = try MixedstatePtychography.initialProbes(from: first, count: e.`init`.count, seed: e.`init`.seed)
        let g = gap(states, e.`init`.states)
        print("ODD init gap \(g)")
        #expect(g < 2e-6)
    }

    @Test func multisliceOnOddSizes() throws {
        let e = try pin().engines
        let props = try MultislicePtychography.propagators(rows: e.rows, columns: e.columns, rowSampling: e.rowSampling,
                                                           columnSampling: e.columnSampling, wavelength: e.wavelength,
                                                           sliceThicknesses: e.sliceThicknesses)
        let pg = gap(props, e.propagators)
        print("ODD propagators gap \(pg)")
        #expect(pg < 2e-6)
        let probe = try array(e.startProbe, e.rows, e.columns)
        let plane = e.objectRows * e.objectColumns
        for c in e.multislice {
            let k = c.slices!
            let object = try PtychographyStack(count: k, rows: e.objectRows, columns: e.objectColumns,
                                               real: Array(e.startSlices.real[0..<k*plane]), imaginary: Array(e.startSlices.imaginary[0..<k*plane]))
            let r = try MultislicePtychography.reconstruct(
                amplitudes: e.amplitudes, positions: positions(e), object: object, probe: probe, sliceThicknesses: c.thicknesses!,
                wavelength: e.wavelength, rowSampling: e.rowSampling, columnSampling: e.columnSampling, iterations: 3, stepSize: 0.35,
                normalizationMinimum: 0.7, fixProbeCenterOfMass: c.fixProbeCenterOfMass, constrainProbeAmplitude: c.constrainProbeAmplitude,
                probeAmplitudeRelativeRadius: c.radius, probeAmplitudeRelativeWidth: c.width, fixProbeAperture: c.aperture,
                probeSupportMask: c.support ? e.support : nil, errorNormalization: .meanIntensity(e.multislicePreprocessing.meanIntensity))
            let g = [gap(r.object, c.object), gap(r.probe, c.probe!), gap(r.errors, c.errors)]
            print("ODD multislice \(c.name) \(g)")
            #expect(g.max()! < 2e-6)
        }
    }
}
