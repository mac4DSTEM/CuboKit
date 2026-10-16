import Foundation
import Testing
import CuboParallax

@Suite struct ParallaxDepthTests {
    struct Pin: Decodable {
        let rows, columns, count: Int
        let stack, angles, coefs, depths: [Double]
        let terms: [[Int]]; let c1, wavelength: Double; let cases: [Case]
    }
    struct Case: Decodable { let limit: Double?; let power: Double; let reference, app: [Double] }
    func pin() throws -> Pin {
        try JSONDecoder().decode(Pin.self, from: Data(contentsOf: Bundle.module.url(forResource: "depth", withExtension: "json", subdirectory: "Fixtures")!))
    }
    func gap(_ a: [Double], _ b: [Double]) -> Double {
        let scale = b.map { abs($0) }.max() ?? 1
        return zip(a, b).map { abs($0 - $1) / scale }.max() ?? 0
    }
    @Test func pinnedSections() throws {
        let p = try pin(), n = p.rows * p.columns
        let stack = (0..<p.count).map { Array(p.stack[$0 * n..<($0 + 1) * n]) }
        let angles = stride(from: 0, to: p.angles.count, by: 2).map { ParallaxShift(row: p.angles[$0], column: p.angles[$0 + 1]) }
        let terms = p.terms.map { ParallaxAberrationTerm(radialOrder: $0[0], angularOrder: $0[1], component: $0[2]) }
        for c in p.cases {
            for app in [false, true] {
                let planes = try ParallaxDepthSectioning.section(stack: stack, rows: p.rows, columns: p.columns, probeAngles: angles,
                    sampling: .init(row: 1.25, column: 1.25), wavelength: p.wavelength,
                    ctf: app ? .fullFit(terms: terms, coefficientsAngstrom: p.coefs) : .defocusOnly(c1Angstrom: p.c1),
                    depthsAngstrom: p.depths, informationLimit: c.limit, informationPower: c.power)
                let g = gap(planes.flatMap { $0 }, app ? c.app : c.reference)
                print("depth gap app=\(app) \(g)")
                #expect(g < 1e-5) // relative to the plane maximum; the reference runs complex64.
            }
        }
    }
    @Test func invalidInput() {
        #expect(throws: ParallaxError.invalidInput) {
            try ParallaxDepthSectioning.section(stack: [], rows: 4, columns: 4, probeAngles: [], sampling: .init(row: 1, column: 1),
                wavelength: 0.025, ctf: .defocusOnly(c1Angstrom: 1), depthsAngstrom: [0])
        }
    }
}
