import Foundation
import Testing
import CuboCore
import CuboBragg

// CuboKit#92 item 6: inputs that gave a silent all-zero pattern or a NaN class are refused.

@Suite struct SilentNaNRefusalTests {
    /// No position reaches the class threshold: py4DSTEM divides by an empty sum; this gave an all-zero pattern. Mutation: drop
    /// the guard - an all-zero pattern comes back, red.
    @Test func aClassWithNoPositionHasNoPattern() async throws {
        let shape = try CubeShape(scanRows: 3, scanColumns: 3, detectorRows: 4, detectorColumns: 5)
        let source = try MemoryCube(shape: shape, values: (0..<(9 * 20)).map { Float($0 % 17) })
        await #expect(throws: CuboBraggError.self) {
            _ = try await BraggVectorClassification.classPattern(of: source, classImage: [Double](repeating: 0, count: 9))
        }
        let vectors = BraggVectors(scanRows: 3, scanColumns: 3, peaks: [[BraggPeak]](repeating: [], count: 9))
        await #expect(throws: CuboBraggError.self) {
            _ = try await BraggVectorClassification.classPatternWithoutBraggScattering(
                of: source, classImage: [Double](repeating: 0, count: 9), vectors: vectors, radius: 1, originRow: 2, originColumn: 2)
        }
        // One position at the threshold is a pattern.
        var image = [Double](repeating: 0, count: 9); image[4] = 0.5
        let one = try await BraggVectorClassification.classPattern(of: source, classImage: image)
        #expect(one.contains { $0 > 0 })
    }

    /// Two classes with no weight at any position: their weighted average is 0/0. Mutation: drop the guard - a NaN column, red.
    @Test func twoClassesWithoutWeightDoNotMerge() throws {
        let peaks = (0..<9).map { _ in [BraggPeak(row: 1, column: 1, intensity: 1), BraggPeak(row: 3, column: 3, intensity: 1)] }
        var c = try BraggVectorClassification(vectors: BraggVectors(scanRows: 3, scanColumns: 3, peaks: peaks),
                                              referenceRows: [1, 3], referenceColumns: [1, 3])
        try c.setClasses(w: [1, 0, 0, 1, 0, 0], h: [Double](repeating: 1, count: 9) + [Double](repeating: 0, count: 18), classCount: 3)
        #expect(throws: CuboBraggError.self) { try c.merge(classIndices: 1, 2) }
        try c.merge(classIndices: 0, 1)   // one class with weight: the average is that class's
    }
}
