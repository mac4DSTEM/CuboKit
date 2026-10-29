import Testing
import CuboCore
import CuboKitTestSupport
@testable import CuboVirtual

/// scipy 1.18.0 `gaussian_filter(a, 1.7, mode=...)` on a 3 × 4 image whose kernel radius (7) exceeds both extents, so the
/// edge folding is exercised repeatedly. Generated with the pinned venv (tools/bragg/pins.py documents it).
@Suite struct GaussianEdgeTests {
    let image: [Double] = [1, 5, 2, 0, 7, 3, 9, 4, 0, 8, 6, 1]
    let reflect: [Double] = [3.804407399533657, 3.774679385371966, 3.613976334908968, 3.415874500747273, 3.9404564736662033, 3.9461242977561954,
                             3.8214491050250134, 3.6388755123888243, 4.066434520159526, 4.107741334447454, 4.0188171860346324, 3.8511639499602914]
    let nearest: [Double] = [2.803236387012084, 2.9705488713242847, 2.7296663644285024, 2.157407960645013, 3.0685471740239083, 3.3736527719125027,
                             3.2158473828174583, 2.6412819610689797, 3.024672944791127, 3.5305883010732977, 3.4763856040938528, 2.888242927488524]

    @Test func edgeModesMatchScipy() throws {
        let f = try GaussianFilter(sigma: 1.7)
        let r = f.apply(image, rows: 3, columns: 4, edge: .reflect)
        let n = f.apply(image, rows: 3, columns: 4)
        let gapReflect = zip(r, reflect).map { abs($0 - $1) }.max()!, gapNearest = zip(n, nearest).map { abs($0 - $1) }.max()!
        print("GaussianFilter edge gaps: reflect \(gapReflect), nearest \(gapNearest), tolerance 1e-13")
        #expect(gapReflect <= 1e-13 && gapNearest <= 1e-13)
        #expect(n == f.apply(image, rows: 3, columns: 4, edge: .nearest))
    }
}

/// One Gaussian filter in the kit (#14): CuboCore's `SciPyKernels.gaussianFilter2D` (mode reflect) and the public
/// `GaussianFilter.apply(edge: .reflect)` give the same bits, kernels wider and narrower than the image.
@Suite struct GaussianIdentityTests {
    @Test(arguments: [(3, 4, 1.7), (16, 11, 0.8), (9, 23, 2.6), (1, 7, 1.0), (5, 1, 3.3), (32, 32, 0.4)])
    func reflectIsSciPyKernels(_ rows: Int, _ columns: Int, _ sigma: Double) throws {
        var state: UInt64 = UInt64(rows * 131 + columns)
        let image = (0..<(rows * columns)).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Double(state >> 11) / 0x1p53 * 100 - 20
        }
        let filter = try GaussianFilter(sigma: sigma)
        #expect(filter.kernel == SciPyKernels.gaussianWeights(sigma: sigma))
        #expect(filter.apply(image, rows: rows, columns: columns, edge: .reflect)
                == SciPyKernels.gaussianFilter2D(image, rows: rows, columns: columns, sigma: sigma))
    }
}
