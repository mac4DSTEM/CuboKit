import Foundation
import Testing
import CuboPtycho

// Position correction of the single-slice engine against py4DSTEM `reconstruct(fix_positions=False, ...)`
// (tools/ptycho/position_pins.py): 8 x 8 positions that start up to 0.3 px from the data's, four iterations.
// Tolerance on the object, probe and errors 2e-6 (max abs over max(1, max |oracle|)); on the positions 2e-6 px (absolute, over max(1, max))
// (measured 6.2e-7; py4DSTEM keeps them in Float32 and solves a 2 x 2 normal system per pattern in Float32).

private struct Cx: Decodable { let real: [Double]; let imaginary: [Double] }
private struct Settings: Decodable {
    let stepSize: Double, maxUpdateDistance: Double?, maxTotalDistance: Double?, constrainCenterOfMass: Bool?, fixProbe: Bool?
}
private struct Case: Decodable {
    let name: String, settings: Settings, fixProbe: Bool, object: Cx, probe: Cx, errors: [Double], positions: [[Double]]
}
private struct Exact: Decodable { let amplitudes: [Double], meanIntensity: Double, object: Cx, probe: Cx, positions: [[Double]], errors: [Double] }
private struct Pin: Decodable {
    let exact: Exact
    let rows: Int, columns: Int, objectRows: Int, objectColumns: Int, rowSampling: Double, columnSampling: Double
    let nominalPositions: [[Double]], truthPositions: [[Double]], amplitudes: [Double], meanIntensity: Double
    let startObject: Cx, startProbe: Cx, cases: [Case]
}
private func pin() throws -> Pin {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/positions.json")
    return try JSONDecoder().decode(Pin.self, from: Data(contentsOf: url))
}
private func gap(_ a: [Double], _ b: [Double]) -> Double {
    guard a.count == b.count, a.allSatisfy(\.isFinite) else { return .infinity }   // a NaN must fail, max() would skip it
    return zip(a, b).map { abs($0 - $1) }.max()! / max(1, b.map(abs).max()!)
}
private func positions(_ a: [[Double]]) -> [PtychographyPosition] { a.map { .init(row: $0[0], column: $0[1]) } }
private func start(_ p: Pin) throws -> (PtychographyArray, PtychographyArray) {
    (try PtychographyArray(rows: p.objectRows, columns: p.objectColumns, real: p.startObject.real, imaginary: p.startObject.imaginary),
     try PtychographyArray(rows: p.rows, columns: p.columns, real: p.startProbe.real, imaginary: p.startProbe.imaginary))
}
private func correction(_ s: Settings, _ p: Pin) -> PtychographyPositionCorrection {
    PtychographyPositionCorrection(stepSize: s.stepSize, maxUpdateDistance: s.maxUpdateDistance, maxTotalDistance: s.maxTotalDistance,
                                   rowSampling: p.rowSampling, columnSampling: p.columnSampling,
                                   constrainCenterOfMass: s.constrainCenterOfMass ?? false)
}

@Suite struct PositionCorrectionTests {
    @Test func positionsAgainstPositionCorrection() throws {
        let p = try pin(), (object, probe) = try start(p)
        for c in p.cases {
            let r = try SingleslicePtychography.reconstruct(
                amplitudes: p.amplitudes, positions: positions(p.nominalPositions), object: object, probe: probe, iterations: 4,
                stepSize: 0.35, normalizationMinimum: 0.7, fixProbe: c.fixProbe,
                errorNormalization: .meanIntensity(p.meanIntensity), positionCorrection: correction(c.settings, p))
            let pg = max(gap(r.positions.map(\.row), c.positions.map { $0[0] }), gap(r.positions.map(\.column), c.positions.map { $0[1] }))
            let g = [gap(r.object.real, c.object.real), gap(r.object.imaginary, c.object.imaginary),
                     gap(r.probe.real, c.probe.real), gap(r.probe.imaginary, c.probe.imaginary), gap(r.errors, c.errors)]
            print("POSITIONS \(c.name) positions \(pg) object/probe/errors \(g)")
            #expect(pg < 2e-6)
            #expect(g.max()! < 2e-6)
            // the positions did move (the pin would not catch an engine that ignores the option)
            let moved = zip(r.positions, p.nominalPositions).map { max(abs($0.row - $1[0]), abs($0.column - $1[1])) }.max()!
            #expect(moved > 0.05)
        }
    }

    @Test func offByDefaultAndPositionsReturnedUnchanged() throws {
        let p = try pin(), (object, probe) = try start(p)
        let given = positions(p.nominalPositions)
        let plain = try SingleslicePtychography.reconstruct(amplitudes: p.amplitudes, positions: given, object: object, probe: probe,
                                                            iterations: 2, stepSize: 0.35, normalizationMinimum: 0.7)
        #expect(plain.positions.map(\.row) == given.map(\.row) && plain.positions.map(\.column) == given.map(\.column))
        let zero = try SingleslicePtychography.reconstruct(amplitudes: p.amplitudes, positions: given, object: object, probe: probe,
                                                           iterations: 2, stepSize: 0.35, normalizationMinimum: 0.7,
                                                           positionCorrection: PtychographyPositionCorrection(stepSize: 0))
        // a zero step moves nothing and leaves every other result exactly as without the option
        #expect(zero.positions.map(\.row) == given.map(\.row) && zero.object.real == plain.object.real && zero.errors == plain.errors)
    }

    // Exact object and probe, data taken at `truthPositions`, the object frozen (step 1e-12): the correction alone moves the positions.
    // Against py4DSTEM (2e-6), and toward the data's positions (a sign check that does not depend on the reference).
    @Test func positionsMoveTowardTheDataPositions() throws {
        let p = try pin(), e = p.exact
        let object = try PtychographyArray(rows: p.objectRows, columns: p.objectColumns, real: e.object.real, imaginary: e.object.imaginary)
        let probe = try PtychographyArray(rows: p.rows, columns: p.columns, real: e.probe.real, imaginary: e.probe.imaginary)
        let r = try SingleslicePtychography.reconstruct(
            amplitudes: e.amplitudes, positions: positions(p.nominalPositions), object: object, probe: probe, iterations: 1,
            stepSize: 1e-12, normalizationMinimum: 0.7, fixProbe: true, errorNormalization: .meanIntensity(e.meanIntensity),
            positionCorrection: PtychographyPositionCorrection(stepSize: 0.5))
        let pg = max(gap(r.positions.map(\.row), e.positions.map { $0[0] }), gap(r.positions.map(\.column), e.positions.map { $0[1] }))
        #expect(pg < 2e-6)
        func meanDistance(_ a: [PtychographyPosition]) -> Double {
            zip(a, p.truthPositions).map { hypot($0.row - $1[0], $0.column - $1[1]) }.reduce(0, +) / Double(a.count)
        }
        let before = meanDistance(positions(p.nominalPositions)), after = meanDistance(r.positions)
        print("POSITIONS exact data: position gap \(pg), mean distance to the data's positions \(before) -> \(after)")
        #expect(after < 0.7 * before)
    }

    @Test func limitsAreHonoured() throws {
        let p = try pin(), (object, probe) = try start(p)
        let given = positions(p.nominalPositions)
        let scale = (p.rowSampling*p.rowSampling + p.columnSampling*p.columnSampling).squareRoot()
        let r = try SingleslicePtychography.reconstruct(
            amplitudes: p.amplitudes, positions: given, object: object, probe: probe, iterations: 1, stepSize: 0.35,
            normalizationMinimum: 0.7, positionCorrection: PtychographyPositionCorrection(
                stepSize: 0.5, maxUpdateDistance: 0.2, rowSampling: p.rowSampling, columnSampling: p.columnSampling))
        let limit = 0.2 / scale
        let moves = zip(r.positions, given).map { hypot($0.row - $1.row, $0.column - $1.column) }
        #expect(moves.max()! <= limit * (1 + 1e-12) && moves.max()! > 0.5 * limit)   // clipped, and some were clipped to the limit
        // a total-distance limit smaller than any single move drops those moves
        let held = try SingleslicePtychography.reconstruct(
            amplitudes: p.amplitudes, positions: given, object: object, probe: probe, iterations: 1, stepSize: 0.35,
            normalizationMinimum: 0.7, positionCorrection: PtychographyPositionCorrection(
                stepSize: 0.5, maxTotalDistance: 1e-6, rowSampling: p.rowSampling, columnSampling: p.columnSampling))
        #expect(held.positions.map(\.row) == given.map(\.row) && held.positions.map(\.column) == given.map(\.column))
    }

    @Test func invalidSettingsAreRefused() throws {
        let p = try pin(), (object, probe) = try start(p)
        let given = positions(p.nominalPositions)
        let bad: [PtychographyPositionCorrection] = [
            .init(stepSize: .nan), .init(stepSize: .infinity),
            .init(stepSize: 0.5, maxUpdateDistance: 1),                                                   // distance without samplings
            .init(stepSize: 0.5, maxTotalDistance: 1, rowSampling: 1),
            .init(stepSize: 0.5, maxUpdateDistance: -1, rowSampling: 1, columnSampling: 1),
            .init(stepSize: 0.5, maxTotalDistance: 0, rowSampling: 1, columnSampling: 1),
            .init(stepSize: 0.5, maxUpdateDistance: 1, rowSampling: 0, columnSampling: 1),
            .init(stepSize: 0.5, maxUpdateDistance: .nan, rowSampling: 1, columnSampling: 1)]
        for b in bad {
            #expect(throws: PtychographyError.self) {
                try SingleslicePtychography.reconstruct(amplitudes: p.amplitudes, positions: given, object: object, probe: probe,
                                                        iterations: 1, positionCorrection: b)
            }
        }
    }
}
