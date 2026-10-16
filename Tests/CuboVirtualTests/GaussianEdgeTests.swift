import Testing
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
