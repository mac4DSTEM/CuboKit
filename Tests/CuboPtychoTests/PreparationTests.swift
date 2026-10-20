import Foundation
import Testing
import CuboCore
import CuboPtycho

private struct PosCase: Decodable {
    let angle: Double, transpose: Bool, positions: [[Double]], padding: [[Double]], objectShape: [Int], centred: [[Double]]
}
private struct AmpCase: Decodable { let amplitudes: [Double], meanIntensity: Double, clippedMeanIntensity: Double }
private struct ProbeCase: Decodable {
    let name: String, semiangle: Double, rolloff: Double, defocus: Double, astigmatism: Double
    let astigmatismAngle: Double, sphericalAberration: Double, meanIntensity: Double, real: [Double], imaginary: [Double]
}
private struct Pin: Decodable {
    let scanRows: Int, scanColumns: Int, detectorRows: Int, detectorColumns: Int
    let cube: [Double], originRows: [Double], originColumns: [Double]
    let amplitudes: [String: AmpCase]
    let wavelength: Double, angularSampling: [Double], objectSampling: [Double], scanSampling: [Double]
    let positionCases: [PosCase], probes: [ProbeCase]
}
private func pin() throws -> Pin {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/preprocess.json")
    return try JSONDecoder().decode(Pin.self, from: Data(contentsOf: url))
}
private func source(_ p: Pin) throws -> MemoryCube {
    try MemoryCube(shape: CubeShape(scanRows: p.scanRows, scanColumns: p.scanColumns, detectorRows: p.detectorRows,
                                    detectorColumns: p.detectorColumns), values: p.cube.map { Float($0) })
}
/// max |a - b| / max(1, max |b|)
private func gap(_ a: [Double], _ b: [Double]) -> Double {
    guard a.count == b.count else { return .infinity }
    return zip(a, b).map { abs($0 - $1) }.max()! / max(1, b.map(abs).max()!)
}
private func relative(_ a: Double, _ b: Double) -> Double { abs(a - b) / max(1e-300, abs(b)) }

@Suite struct PreparationTests {
    // py4DSTEM shifts in Float32 (order 1: float32 weights; order 3: scipy's Float32 output), the kit in Double.
    // Bound: 2e-6 of max(1, max|amplitude|); mean intensity relative 2e-6.
    @Test func amplitudesAgainstNormalizeDiffractionIntensities() async throws {
        let p = try pin(), cube = try source(p)
        for (key, shift) in [("1", PtychographyShift.bilinear), ("3", .cubicSpline)] {
            let ref = p.amplitudes[key]!
            let a = try await PtychographyPreparation.amplitudes(
                from: cube, origins: .perPosition(rows: p.originRows, columns: p.originColumns), shift: shift)
            let g = gap(a.values, ref.amplitudes)
            print("PREP amplitudes order \(key) gap \(g), mean rel \(relative(a.meanIntensity, ref.meanIntensity)), clipped rel \(relative(a.clippedMeanIntensity, ref.clippedMeanIntensity))")
            #expect(g < 2e-6)
            #expect(relative(a.meanIntensity, ref.meanIntensity) < 2e-6)
            // the app's clipped mean: a NumPy transcription (sum of max(shifted, 0))
            #expect(relative(a.clippedMeanIntensity, ref.clippedMeanIntensity) < 2e-6)
            #expect(a.positionCount == p.scanRows * p.scanColumns && a.detectorRows == p.detectorRows)
        }
    }

    @Test func tilingDoesNotChangeAmplitudes() async throws {
        let p = try pin(), cube = try source(p)
        let origins = PtychographyOrigins.perPosition(rows: p.originRows, columns: p.originColumns)
        let one = try await PtychographyPreparation.amplitudes(from: cube, origins: origins, tiling: .rows(1))
        let all = try await PtychographyPreparation.amplitudes(from: cube, origins: origins, tiling: .rows(p.scanRows))
        #expect(one.values == all.values && one.meanIntensity == all.meanIntensity)
    }

    @Test func zeroOriginIsTheClippedSquareRoot() async throws {
        let p = try pin(), cube = try source(p)
        for shift in [PtychographyShift.bilinear, .cubicSpline] {
            let a = try await PtychographyPreparation.amplitudes(from: cube, origins: .uniform(row: 0, column: 0), shift: shift)
            // an integer shift of zero reproduces the data: exact for bilinear, FFT roundoff for the spline
            // (compared as intensities: the square root turns FFT roundoff at exact zeros into 1e-8)
            let expected = p.cube.map { max($0, 0) }
            let g = gap(a.values.map { $0 * $0 }, expected)       // normwise, relative to max(1, max)
            #expect(g < (shift == .bilinear ? 1e-15 : 1e-13))
        }
        // an integer shift is a roll of the pattern
        let a = try await PtychographyPreparation.amplitudes(from: cube, origins: .uniform(row: 2, column: -3), shift: .bilinear)
        let qr = p.detectorRows, qc = p.detectorColumns
        for r in 0..<qr { for c in 0..<qc {
            let src = ((r + 2) % qr) * qc + ((c - 3 + 3*qc) % qc)
            #expect(a.values[r*qc + c] == max(p.cube[src], 0).squareRoot())
        } }
    }

    // Positions are float64 in py4DSTEM until the Float32 state: pinned at 1e-9 absolute (px); the Float32 centring at 2e-6 of max(1, max).
    @Test func positionsAgainstCalculateScanPositions() throws {
        let p = try pin()
        for c in p.positionCases {
            let geometry = PtychographyScanGeometry(scanSamplingRow: p.scanSampling[0], scanSamplingColumn: p.scanSampling[1],
                                                    rotation: c.angle, transpose: c.transpose)
            let pad = PtychographyPadding.reference(detectorRows: p.detectorRows, detectorColumns: p.detectorColumns)
            #expect(pad.startRow == c.padding[0][0] && pad.extentRow == c.padding[0][1])
            #expect(pad.startColumn == c.padding[1][0] && pad.extentColumn == c.padding[1][1])
            let pos = try PtychographyPreparation.scanPositions(
                scanRows: p.scanRows, scanColumns: p.scanColumns, geometry: geometry,
                objectSamplingRow: p.objectSampling[0], objectSamplingColumn: p.objectSampling[1], padding: pad)
            let g = max(gap(pos.map(\.row), c.positions.map { $0[0] }), gap(pos.map(\.column), c.positions.map { $0[1] }))
            print("PREP positions angle \(c.angle) transpose \(c.transpose) gap \(g)")
            #expect(g < 1e-9)
            let shape = try PtychographyPreparation.objectShape(positions: pos, detectorRows: p.detectorRows,
                                                                detectorColumns: p.detectorColumns, padding: pad)
            #expect([shape.rows, shape.columns] == c.objectShape)       // indices: exact
            let centred = PtychographyPreparation.centerPositions(pos, objectRows: shape.rows, objectColumns: shape.columns)
            let cg = max(gap(centred.map(\.row), c.centred.map { $0[0] }), gap(centred.map(\.column), c.centred.map { $0[1] }))
            print("PREP centred gap \(cg)")
            #expect(cg < 2e-6)
        }
    }

    // The app pads each axis by its own half detector extent. Differs from the reference only along the column
    // start offset on this 12 x 10 detector: (10 - 12) / 2 = -1 pixel; the extents use the row half (6) on the rows and
    // the column half (5) on the columns in both. Analytic, from the app's source.
    @Test func perAxisPaddingDiffersByTheDocumentedOffsets() throws {
        let p = try pin()
        let g = PtychographyScanGeometry(scanSamplingRow: p.scanSampling[0], scanSamplingColumn: p.scanSampling[1], rotation: 0.35)
        let ref = PtychographyPadding.reference(detectorRows: p.detectorRows, detectorColumns: p.detectorColumns)
        let app = PtychographyPadding.perAxis(detectorRows: p.detectorRows, detectorColumns: p.detectorColumns)
        let a = try PtychographyPreparation.scanPositions(scanRows: p.scanRows, scanColumns: p.scanColumns, geometry: g,
            objectSamplingRow: p.objectSampling[0], objectSamplingColumn: p.objectSampling[1], padding: ref)
        let b = try PtychographyPreparation.scanPositions(scanRows: p.scanRows, scanColumns: p.scanColumns, geometry: g,
            objectSamplingRow: p.objectSampling[0], objectSamplingColumn: p.objectSampling[1], padding: app)
        #expect(zip(a, b).allSatisfy { $0.row == $1.row && abs($0.column - $1.column - 1) < 1e-12 })
        #expect(app.extentRow == 6 && app.extentColumn == 5 && ref.extentRow == 5 && ref.extentColumn == 5)
        let square = PtychographyPadding.perAxis(detectorRows: 8, detectorColumns: 8)
        #expect(square == PtychographyPadding.reference(detectorRows: 8, detectorColumns: 8))
    }

    @Test func probeAgainstInitializeProbe() throws {
        let p = try pin()
        for c in p.probes {
            let probe = try PtychographyPreparation.initialProbe(
                rows: p.detectorRows, columns: p.detectorColumns, samplingRow: p.objectSampling[0],
                samplingColumn: p.objectSampling[1], wavelength: p.wavelength, semiangle: c.semiangle, rolloff: c.rolloff,
                defocus: c.defocus, astigmatism: c.astigmatism, astigmatismAngle: c.astigmatismAngle,
                sphericalAberration: c.sphericalAberration, meanIntensity: c.meanIntensity)
            let scale = max(1, c.real.map(abs).max()!, c.imaginary.map(abs).max()!)
            let g = max(zip(probe.real, c.real).map { abs($0 - $1) }.max()!, zip(probe.imaginary, c.imaginary).map { abs($0 - $1) }.max()!) / scale
            print("PREP probe \(c.name) gap \(g)")
            #expect(g < 2e-6)         // ComplexProbe builds in Float32
        }
    }

    @Test func objectSamplingFormula() throws {
        let p = try pin()
        let s = PtychographyPreparation.objectSampling(wavelength: p.wavelength, angularSampling: p.angularSampling[0], size: p.detectorRows)
        #expect(relative(s, p.objectSampling[0]) < 1e-14)
    }

    @Test func prepareComposesThePieces() async throws {
        let p = try pin(), cube = try source(p)
        let origins = PtychographyOrigins.perPosition(rows: p.originRows, columns: p.originColumns)
        let geometry = PtychographyScanGeometry(scanSamplingRow: p.scanSampling[0], scanSamplingColumn: p.scanSampling[1], rotation: 0.35)
        let prepared = try await PtychographyPreparation.prepare(
            source: cube, origins: origins, geometry: geometry, objectSamplingRow: p.objectSampling[0],
            objectSamplingColumn: p.objectSampling[1], wavelength: p.wavelength, semiangle: 12)
        let c = p.positionCases[1]
        #expect(prepared.object.rows == c.objectShape[0] && prepared.object.columns == c.objectShape[1])
        #expect(gap(prepared.positions.map(\.row), c.centred.map { $0[0] }) < 2e-6)
        #expect(gap(prepared.amplitudes.values, p.amplitudes["3"]!.amplitudes) < 2e-6)
        // probe energy follows the mean intensity: sum |fft2|^2 = N sum |probe|^2
        let energy = zip(prepared.probe.real, prepared.probe.imaginary).reduce(0) { $0 + $1.0*$1.0 + $1.1*$1.1 }
            * Double(p.detectorRows * p.detectorColumns)
        #expect(relative(energy, prepared.amplitudes.meanIntensity) < 1e-12)
        let app = try await PtychographyPreparation.prepare(
            source: cube, origins: origins, geometry: geometry, objectSamplingRow: p.objectSampling[0],
            objectSamplingColumn: p.objectSampling[1], wavelength: p.wavelength, semiangle: 12,
            options: .mac4DSTEM(detectorRows: p.detectorRows, detectorColumns: p.detectorColumns))
        let appEnergy = zip(app.probe.real, app.probe.imaginary).reduce(0) { $0 + $1.0*$1.0 + $1.1*$1.1 }
            * Double(p.detectorRows * p.detectorColumns)
        #expect(relative(appEnergy, app.amplitudes.clippedMeanIntensity) < 1e-12)
    }

    @Test func invalidInputsAreRefused() async throws {
        let p = try pin(), cube = try source(p)
        await #expect(throws: PtychographyError.self) {
            _ = try await PtychographyPreparation.amplitudes(from: cube, origins: .perPosition(rows: [0], columns: [0]))
        }
        await #expect(throws: PtychographyError.self) {
            _ = try await PtychographyPreparation.amplitudes(from: cube, origins: .uniform(row: .nan, column: 0))
        }
        #expect(throws: PtychographyError.self) {
            try PtychographyPreparation.scanPositions(scanRows: 2, scanColumns: 2,
                geometry: .init(scanSamplingRow: 0, scanSamplingColumn: 1), objectSamplingRow: 1, objectSamplingColumn: 1,
                padding: .perAxis(detectorRows: 4, detectorColumns: 4))
        }
        #expect(throws: PtychographyError.self) {
            try PtychographyPreparation.initialProbe(rows: 8, columns: 8, samplingRow: 1, samplingColumn: 1,
                wavelength: 0.025, semiangle: 10, meanIntensity: 0)
        }
    }
}
