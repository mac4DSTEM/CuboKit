import Testing
@testable import CuboBragg

// `ClassificationNumerics.leastSquares` against `np.linalg.lstsq(a, b, rcond=None)[0]` (NumPy 2.5.4) on planted tall, wide and
// rank-deficient systems, at np.allclose defaults. The values were printed by:
//   a, b = np.array(...), np.array(...); print([repr(v) for v in np.linalg.lstsq(a, b, rcond=None)[0].ravel()])
@Suite struct LeastSquaresTests {
    private func allclose(_ x: [Double], _ y: [Double]) -> Bool {
        x.count == y.count && zip(x, y).allSatisfy { abs($0 - $1) <= 1e-8 + 1e-5 * abs($1) }
    }

    @Test func tallMatchesNumPy() throws {
        let a: [Double] = [1, 2, 0.5, 0.3, -1, 2, 4, 0.1, 1, 2, 2, -3, 0.7, 1.5, 0.2]
        let b: [Double] = [1, 0, 2, 1, -1, 3, 0.5, 0.5, 2, -2]
        let x = try ClassificationNumerics.leastSquares(a, 5, 3, b, 2)
        #expect(allclose(x, [-0.2526151349095737, 0.8243045142981714, 0.8190730524586134, -0.8369340924137171,
                             0.532812781224323, -0.12577281868886783]))
    }

    @Test func wideIsTheMinimumNormSolution() throws {
        let a: [Double] = [1, 2, 0.5, -1, 0.3, -1, 2, 4]
        let b: [Double] = [1, 0.25, 2, -1]
        let x = try ClassificationNumerics.leastSquares(a, 2, 4, b, 2)
        #expect(allclose(x, [0.32491056984665867, -0.008658205928592507, 0.3990065847934564, 0.05668846407983787,
                             0.45245961402629364, -0.08989724076647904, 0.3491535464467177, -0.19022989815215657]))
    }

    @Test func rankDeficientDropsTheNullSpace() throws {
        // the third column is the sum of the first two
        let a: [Double] = [1, 2, 3, 0.5, -1, -0.5, 2, 0.25, 2.25, -1, 3, 2]
        let b: [Double] = [1, 0, 0, 1, 2, 2, -1, 0.5]
        let x = try ClassificationNumerics.leastSquares(a, 4, 3, b, 2)
        #expect(allclose(x, [0.6325001498531438, 0.39992807049091883, -0.2997062878379189, -0.14098183779895704,
                             0.33279386201522515, 0.2589462326919619]))
    }

    @Test func nonFiniteMatrixThrowsAsNumPyDoes() throws {
        // np.linalg.lstsq raises LinAlgError("SVD did not converge") on a NaN in `a`; this used to return [0, 0] silently.
        #expect(throws: CuboBraggError.self) { try ClassificationNumerics.leastSquares([1, .nan, 2, 3], 2, 2, [1, 2], 1) }
        // A non-finite `b` does not raise in NumPy (lstsq(eye(2), [1, inf]) is [nan, nan]): it carries through here too.
        let x = try ClassificationNumerics.leastSquares([1, 0, 0, 1], 2, 2, [1, .infinity], 1)
        #expect(x.contains { !$0.isFinite })
    }
}
