import Foundation
import Testing
import CuboCore
import CuboBragg

// Inputs that used to trap (a huge or non-finite shift or sigma, an out-of-range class, an n^3 allocation) now throw.

@Suite struct ClassificationHardeningTests {
    private func classification() throws -> BraggVectorClassification {
        let peaks = (0..<9).map { _ in [BraggPeak(row: 1, column: 1, intensity: 1), BraggPeak(row: 3, column: 3, intensity: 1)] }
        var c = try BraggVectorClassification(vectors: BraggVectors(scanRows: 3, scanColumns: 3, peaks: peaks),
                                              referenceRows: [1, 3], referenceColumns: [1, 3])
        try c.setClasses(w: [1, 0, 0, 1], h: [Double](repeating: 1, count: 18), classCount: 2)
        return c
    }

    private func cube() throws -> MemoryCube {
        let shape = try CubeShape(scanRows: 3, scanColumns: 3, detectorRows: 4, detectorColumns: 5)
        return try MemoryCube(shape: shape, values: (0..<(9 * 20)).map { Float($0 % 17) })
    }

    @Test func classPatternShifts() async throws {
        let source = try cube(), image = [Double](repeating: 1, count: 9)
        let ones = [Double](repeating: 1, count: 9)
        for bad in [Double.nan, .infinity, -.infinity] {
            await #expect(throws: CuboBraggError.self) {
                _ = try await BraggVectorClassification.classPattern(of: source, classImage: image, rowShifts: [Double](repeating: bad, count: 9), columnShifts: ones)
            }
        }
        // a huge shift equals its reduction modulo the detector size
        let huge = 1e18   // a multiple of 4 and of 5? 1e18 mod 4 = 0, mod 5 = 0
        let a = try await BraggVectorClassification.classPattern(of: source, classImage: image,
                rowShifts: [Double](repeating: huge, count: 9), columnShifts: [Double](repeating: huge, count: 9))
        let b = try await BraggVectorClassification.classPattern(of: source, classImage: image)
        #expect(a == b)
        await #expect(throws: CancellationError.self) {
            _ = try await BraggVectorClassification.classPattern(of: source, classImage: image, isCancelled: { true })
        }
        let vectors = BraggVectors(scanRows: 3, scanColumns: 3, peaks: [[BraggPeak]](repeating: [], count: 9))
        await #expect(throws: CuboBraggError.self) {
            _ = try await BraggVectorClassification.classPatternWithoutBraggScattering(
                of: source, classImage: image, vectors: vectors, radius: 1, originRow: 0, originColumn: 0,
                rowShifts: [Double](repeating: .nan, count: 9), columnShifts: ones)
        }
    }

    @Test func splitSigma() throws {
        var c = try classification()
        for bad in [Double.infinity, .nan, 0, -1, 1e300, 13] {
            #expect(throws: CuboBraggError.self) { try c.split(sigma: bad) }
            #expect(throws: CuboBraggError.self) { try c.split(classIndex: 0, sigma: bad) }
        }
        try c.split(sigma: 12)
    }

    @Test func classGettersRefuseMissingClasses() throws {
        var c = try classification()
        #expect(throws: CuboBraggError.self) { _ = try c.classImage(2) }
        #expect(throws: CuboBraggError.self) { _ = try c.classPeakWeights(-1) }
        #expect(try c.classImage(1).count == 9)
        #expect(try c.candidateClassImage(5) == nil)
        try c.removeClass(0)
        #expect(throws: CuboBraggError.self) { _ = try c.candidateClassPeakWeights(1) }
        #expect(throws: CuboBraggError.self) { _ = try c.candidateClassImage(1) }
        #expect(try c.candidateClassPeakWeights(0)?.count == 2)
    }

    @Test func initialClassesRefusesHugeCorrelation() {
        #expect(throws: CuboBraggError.self) {
            _ = try BraggVectorClassification.initialClasses(peakLabels: [[0]], featureCount: 1_000_000, correlationOrder: 3)
        }
        #expect(throws: CuboBraggError.self) {
            _ = try BraggVectorClassification.initialClasses(peakLabels: [[0]], featureCount: Int.max, correlationOrder: 2)
        }
    }
}
