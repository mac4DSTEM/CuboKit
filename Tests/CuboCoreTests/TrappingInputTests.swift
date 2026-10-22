import Foundation
import Testing
@testable import CuboCore

// Inputs that used to trap (Int of a huge Double, an unchecked product, a precondition) now throw, or give the same result as a
// bound input.

@Suite struct TrappingInputTests {
    private func cube() throws -> MemoryCube {
        let shape = try CubeShape(scanRows: 2, scanColumns: 2, detectorRows: 5, detectorColumns: 6)
        return try MemoryCube(shape: shape, values: (0..<shape.elementCount).map { Float(($0 * 7) % 23) })
    }

    @Test func fourierResampleRefusesHugeOutputs() async throws {
        let source = try cube()
        for spec in [FourierResampleSpecification.scale(1e300), .scale(1e7), .outputSize(rows: Int.max, columns: Int.max),
                     .outputSize(rows: 1 << 40, columns: 2), .outputSize(rows: 1 << 20, columns: 1 << 20)] {
            await #expect(throws: CuboError.self) { _ = try await CubeEditing.resampleDetectorFourier(of: source, spec) }
        }
        #expect(throws: CuboError.self) {
            _ = try CubeEditing.fourierResample([1, 2, 3, 4], rows: 2, columns: 2, outputRows: Int.max, outputColumns: 2)
        }
        #expect(throws: CuboError.self) {
            _ = try CubeEditing.fourierResample([1, 2, 3, 4], rows: Int.max, columns: 4, outputRows: 2, outputColumns: 2)
        }
        let ok = try await CubeEditing.resampleDetectorFourier(of: source, .scale(2))
        #expect(ok.cube.shape.detectorRows == 10)
    }

    @Test func medianKernelWiderThanTheDetectorIsTheWholeDetector() async throws {
        let source = try cube()
        var mask = [Bool](repeating: false, count: 30)
        mask[7] = true; mask[20] = true
        let exact = try await CubeEditing.medianFilterMaskedPixels(of: source, mask: mask, kernelWidth: 2 * 6 + 1)
        for width in [Int.max, Int.max - 1, 1 << 40, 2 * 6 + 2] {
            let wide = try await CubeEditing.medianFilterMaskedPixels(of: source, mask: mask, kernelWidth: width)
            let a = try await wide.cube.readScanTile(rows: 0..<2).values, b = try await exact.cube.readScanTile(rows: 0..<2).values
            #expect(a == b, "kernel width \(width)")
        }
    }

    @Test func ellipseFitInputsThatTrapped() throws {
        #expect(throws: EllipseFitError.self) { _ = try EllipseFitPlane(rows: Int.max, columns: 2, values: [1]) }
        let plane = try EllipseFitPlane(rows: 20, columns: 20, values: (0..<400).map { Double($0 % 7) })
        for centre in [1e300, 1e12, -1e12] {
            #expect(throws: EllipseFitError.self) {
                _ = try EllipseFit.fitAmorphousRing(plane, center: (centre, 10), fitRadii: (2, 8))
            }
        }
        #expect(throws: EllipseFitError.self) { _ = try EllipseFit.ellipseError(parameters: [1, 2], rows: [1], columns: [1], values: [1]) }
        #expect(throws: EllipseFitError.self) { _ = try EllipseFit.ellipseError(parameters: [0, 0, 1, 0, 1], rows: [1, 2], columns: [1], values: [1]) }
        #expect(throws: EllipseFitError.self) { _ = try EllipseFit.doubleSidedGaussian(parameters: [1], rows: [1], columns: [1]) }
        #expect(throws: EllipseFitError.self) {
            _ = try EllipseFit.doubleSidedGaussianFitError(parameters: [Double](repeating: 1, count: 11), rows: [1], columns: [1], values: [])
        }
    }

    @Test func hyperbolicSolutionIsNotFinite() throws {
        // bright pixels on the two diagonals: the best conic through them is a hyperbola, whose semi-axes are NaN
        var v = [Double](repeating: 0, count: 41 * 41)
        for r in 0..<41 { for c in 0..<41 where abs((r - 20) * (r - 20) - (c - 20) * (c - 20)) <= 20 { v[r * 41 + c] = 1 } }
        let plane = try EllipseFitPlane(rows: 41, columns: 41, values: v)
        do {
            let fit = try EllipseFit.fit1D(plane, center: (20, 20), fitRadii: (3, 19))
            #expect(fit.ellipse.a.isFinite && fit.ellipse.b.isFinite && fit.ellipse.theta.isFinite)
        } catch let error as EllipseFitError {
            #expect(error == .notFinite)
        }
    }

    @Test func kernelsNeverTrap() {
        #expect(SciPyKernels.arange(0, 1e300, 1e-300).isEmpty)
        #expect(SciPyKernels.arange(0, .infinity, 1).isEmpty)
        #expect(SciPyKernels.arange(0, 1, .nan).isEmpty)
        #expect(SciPyKernels.arangeLength(0, 1e300, 1e-300) == nil)
        #expect(SciPyKernels.arange(0, 3, 1) == [0, 1, 2])
        #expect(SciPyKernels.gaussianRadius(sigma: .infinity) == nil)
        #expect(SciPyKernels.gaussianRadius(sigma: 1e300) == nil)
        #expect(SciPyKernels.gaussianRadius(sigma: 2) == 8)
        #expect(SciPyKernels.gaussianWeights(sigma: .infinity).count == 1)
    }
}
