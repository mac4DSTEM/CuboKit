import Foundation
import Testing
import CuboCore
import CuboParallax

@Suite struct ParallaxPreprocessingTests {
    struct Pin: Decodable { let cases: [Case] }
    struct Case: Decodable {
        let rows, columns, detectorRows, detectorColumns: Int
        let padding: [Int]; let blend: Double; let app: Bool
        let input: [Double]; let mask: [Bool]; let indices: [Int]
        let vectors, angles, window, stack, unshifted, mean: [Double]
        let error: Double; let appSourceUnshifted: [Double]?
    }
    @Test func pinnedPreprocessing() throws {
        let url = parallaxFixture("preprocess")!
        let pin = try JSONDecoder().decode(Pin.self, from: Data(contentsOf: url))
        for c in pin.cases {
            let shape = try CubeShape([c.rows,c.columns,c.detectorRows,c.detectorColumns])
            let r = try ParallaxPreprocessing.prepare(c.input, shape: shape, reciprocalSampling: .init(row: 0.08,column: 0.08), energyEV: 200000,
                threshold: 0.65, edgeBlend: c.blend, paddingRows: c.padding[0], paddingColumns: c.padding[1], retention: c.app ? .unwindowed : .windowed)
            #expect(r.detectorIndices.flatMap{$0} == c.indices)
            check(r.reciprocalVectors.flatMap{[$0.row,$0.column]}, c.vectors)
            check(r.probeAngles.flatMap{[$0.row,$0.column]}, c.angles)
            check(r.window,c.window); check(r.stack.flatMap{$0},c.stack)
            check(r.unshiftedStack.flatMap{$0},c.unshifted)
            check(r.incoherentImage,c.mean); check([r.initialError],[c.error])
            if let app = c.appSourceUnshifted { check(r.unshiftedStack.flatMap{$0},app) }
        }
    }
    @Test func rejectsInvalidPreprocessing() throws {
        let s = try CubeShape([3,3,2,2])
        #expect(throws: ParallaxError.degenerateData) { try ParallaxPreprocessing.prepare([Double](repeating: 0,count:36),shape:s,reciprocalSampling:.init(row:1,column:1),energyEV:200000) }
        #expect(throws: ParallaxError.invalidInput) { try ParallaxPreprocessing.prepare([],shape:s,reciprocalSampling:.init(row:1,column:1),energyEV:200000) }
    }
    @Test func rejectsOverflowedGeometry() throws {
        #expect(throws: ParallaxError.nonfiniteValues) { try ParallaxPreprocessing.electronWavelength(energyEV: .leastNonzeroMagnitude) }
        let s = try CubeShape([3,3,5,5])
        #expect(throws: ParallaxError.nonfiniteValues) {
            try ParallaxPreprocessing.prepare([Double](repeating: 1,count: s.elementCount), shape: s,
                reciprocalSampling: .init(row: .greatestFiniteMagnitude,column: 1), energyEV: 200000)
        }
    }
    @Test func explicitDetectorMask() throws {
        let s = try CubeShape([3,4,2,3])
        let values = [Double](repeating: 2,count: s.elementCount)
        let r = try ParallaxPreprocessing.prepare(values, shape: s, reciprocalSampling: .init(row: 1,column: 1),
            energyEV: 200000, detectorMask: [false,true,false,false,false,false])
        #expect(r.detectorIndices == [[0,1]])
        #expect(r.stack.allSatisfy { $0.allSatisfy { abs($0-1) < 1e-14 } })
        #expect(throws: ParallaxError.invalidInput) {
            try ParallaxPreprocessing.prepare(values, shape: s, reciprocalSampling: .init(row: 1,column: 1),energyEV: 200000,detectorMask: [])
        }
        #expect(throws: ParallaxError.degenerateData) {
            try ParallaxPreprocessing.prepare(values, shape: s, reciprocalSampling: .init(row: 1,column: 1),energyEV: 200000,detectorMask: [Bool](repeating: false,count: 6))
        }
    }
    func check(_ actual: [Double],_ expected: [Double]) {
        #expect(actual.count == expected.count)
        let gap = zip(actual,expected).map{abs($0-$1)}.max() ?? 0
        let scale = max(1,expected.map(abs).max() ?? 0)
        print("preprocess gap \(gap / scale)")
        #expect(gap / scale < 2e-6) // Float32 reference storage/window arithmetic.
    }
}
