import Foundation
import Testing
import CuboCore
@testable import CuboVirtual

// Sampling and sigma that used to trap or allocate without bound now throw CuboVirtualError.invalidInput.

@Suite struct PolarHardeningTests {
    @Test func samplingRefusals() {
        let bad: [PolarSampling] = [
            PolarSampling(maximumRadius: 1e300, radialStep: 1e-300),
            PolarSampling(maximumRadius: 1e300),
            PolarSampling(radialStep: 1e-9),
            PolarSampling(annularCount: 1 << 17),
            PolarSampling(maximumRadius: 2000, annularCount: 1 << 16),          // 2000 x 65536 cells
            PolarSampling(maximumRadius: 0.01, radialStep: 1e-3),                // angular kernel of 100k samples at radius 0
        ]
        for sampling in bad {
            #expect(throws: CuboVirtualError.self, "\(sampling)") { _ = try PolarCube(detectorRows: 8, detectorColumns: 8, sampling: sampling) }
        }
        #expect(throws: Never.self) { _ = try PolarCube(detectorRows: 8, detectorColumns: 8) }
    }

    private func fixture() throws -> (MemoryCube, Calibration, PolarCube) {
        let shape = try CubeShape(scanRows: 2, scanColumns: 3, detectorRows: 8, detectorColumns: 8)
        let cube = try MemoryCube(shape: shape, values: (0..<shape.elementCount).map { Float($0 % 13) })
        var cal = Calibration(); cal.setOrigin(row: 3.5, column: 3.5)
        return (cube, cal, try PolarCube(detectorRows: 8, detectorColumns: 8, sampling: PolarSampling(annularCount: 24)))
    }

    @Test func sigmaRefusals() async throws {
        let (cube, cal, polar) = try fixture()
        let pattern = try polar.transform(pattern: [Float](repeating: 1, count: 64), origin: (3.5, 3.5), ellipse: nil)
        for sigma in [Double.infinity, .nan, 1e300, 1e7] {
            #expect(throws: CuboVirtualError.self) { _ = try RadialBackground.profile(of: pattern, sigma: sigma) }
            await #expect(throws: CuboVirtualError.self) {
                _ = try await RadialBackground.background(of: cube, polar: polar, calibration: cal, scanRow: 0, scanColumn: 0, sigma: sigma)
            }
            await #expect(throws: CuboVirtualError.self) {
                _ = try await RadialBackground.subtracted(from: cube, polar: polar, calibration: cal, scanRow: 0, scanColumn: 0, sigma: sigma)
            }
            await #expect(throws: CuboVirtualError.self) {
                _ = try await RadialBackground.localAverage(of: cube, polar: polar, calibration: cal, scanRow: 0, scanColumn: 0, sigma: sigma)
            }
        }
        #expect(try RadialBackground.profile(of: pattern, sigma: 0).count == polar.radialCount)
    }

    @Test func wholeCubeTransformRefusesWhatCannotFit() async throws {
        // 2048 x 2048 positions of 2 x 2 detectors (64 MB), each asking 32768 x 1000 polar cells: far beyond the memory
        let shape = try CubeShape(scanRows: 2048, scanColumns: 2048, detectorRows: 2, detectorColumns: 2)
        let cube = try MemoryCube(shape: shape, values: [Float](repeating: 1, count: shape.elementCount))
        let polar = try PolarCube(detectorRows: 2, detectorColumns: 2, sampling: PolarSampling(maximumRadius: 1000, annularCount: 1 << 15))
        var cal = Calibration(); cal.setOrigin(row: 0.5, column: 0.5)
        await #expect(throws: CuboVirtualError.self) { _ = try await polar.transform(source: cube, calibration: cal) }
    }

    @Test func patternAtAnInvalidPositionThrows() async throws {
        let (cube, cal, polar) = try fixture()
        let all = try await polar.transform(source: cube, calibration: cal)
        #expect(throws: CuboError.self) { _ = try all.pattern(row: 2, column: 0) }
        #expect(throws: CuboError.self) { _ = try all.pattern(row: 0, column: -1) }
        #expect(try all.pattern(row: 1, column: 2).valid.count == polar.annularCount * polar.radialCount)
    }
}
