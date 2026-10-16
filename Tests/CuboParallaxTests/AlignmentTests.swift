import Foundation
import Testing
import CuboCore
@testable import CuboParallax

private struct Pins: Decodable {
    let upsampleFactor: Int
    let patchCenter: [Double]
    let cases: [StackPin]
}
private struct FactorPin: Decodable {
    let factor: Int
    let shift: [Double]
}
private struct StackPin: Decodable {
    let variants: [FactorPin]
    let name: String
    let rows: Int, columns: Int
    let reference: [Double]
    let planted: [[Double]]
    let frames: [[Double]]
    let shifts: [[Double]]
    let aligned: [[Double]]
    let mean: [Double]
    let patches: [[Double]]
    let running: Bool
}
private func pins() throws -> Pins {
    let url = try #require(Bundle.module.url(forResource: "alignment", withExtension: "json", subdirectory: "Fixtures"))
    return try JSONDecoder().decode(Pins.self, from: Data(contentsOf: url))
}

// Normwise relative tolerance, with a unit floor for zeros and subpixel shifts.
// Double FFT/BLAS summation differs from NumPy: no claim of bit-for-bit parity.
private func close(_ actual: [Double], _ expected: [Double], tolerance: Double = 1e-10,
                   sourceLocation: SourceLocation = #_sourceLocation) {
    #expect(actual.count == expected.count, sourceLocation: sourceLocation)
    let scale = max(1, expected.map(abs).max() ?? 0)
    let error = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
    #expect(actual.allSatisfy(\.isFinite) && error <= tolerance * scale,
            "maximum error \(error), scale \(scale)", sourceLocation: sourceLocation)
}

@Test func pythonStackAlignmentPins() throws {
    let fixture = try pins()
    #expect(fixture.cases.count == 4)
    for item in fixture.cases {
        let result = try ParallaxAlignment.align(item.frames, reference: item.reference,
            rows: item.rows, columns: item.columns, upsampleFactor: fixture.upsampleFactor,
            runningAverage: item.running)
        #expect(result.shifts.count == item.frames.count)
        #expect(result.images.count == item.frames.count)
        for i in item.frames.indices {
            close([result.shifts[i].row, result.shifts[i].column], item.shifts[i])
            close(result.images[i], item.aligned[i])
            if !item.running {
                let shift = try ParallaxAlignment.shift(reference: item.reference, moving: item.frames[i],
                    rows: item.rows, columns: item.columns)
                close([shift.row, shift.column], item.shifts[i])
            }
        }
        close(result.mean, item.mean)
        for variant in item.variants {
            let shift = try ParallaxAlignment.shift(reference: item.reference, moving: item.frames[0],
                rows: item.rows, columns: item.columns, upsampleFactor: variant.factor)
            close([shift.row, shift.column], variant.shift)
        }
    }
}

@Test func pythonFourierShiftPins() throws {
    for item in try pins().cases {
        for i in item.frames.indices {
            let result = try ParallaxAlignment.shifted(item.frames[i], rows: item.rows, columns: item.columns,
                by: ParallaxShift(row: item.shifts[i][0], column: item.shifts[i][1]))
            close(result, item.aligned[i])
            // Generator inputs are also pinned get_shifted_ar outputs (noise-free cases only).
            if item.name != "noisy-running" {
                let planted = try ParallaxAlignment.shifted(item.reference, rows: item.rows, columns: item.columns,
                    by: ParallaxShift(row: item.planted[i][0], column: item.planted[i][1]))
                close(planted, item.frames[i])
            }
        }
    }
}

@Test func plantedShiftSignAndWrapping() throws {
    let fixture = try pins()
    let integer = fixture.cases[0]
    for i in integer.frames.indices {
        let result = try ParallaxAlignment.shift(reference: integer.reference, moving: integer.frames[i],
            rows: integer.rows, columns: integer.columns)
        // Integer planted shifts recover their inverse to floating-point rounding (1e-10 relative/unit floor).
        close([result.row, result.column], integer.planted[i].map { -$0 })
    }
    let odd = fixture.cases[1]
    for i in odd.frames.indices {
        let result = try ParallaxAlignment.shift(reference: odd.reference, moving: odd.frames[i],
            rows: odd.rows, columns: odd.columns)
        // Factor-8 interpolation has finite accuracy; oracle agreement above remains 1e-10.
        close([result.row, result.column], odd.planted[i].map { -$0 }, tolerance: 5e-4)
    }
}

@Test func invalidInputsAreRefused() throws {
    let image = try pins().cases[0].reference
    #expect(throws: ParallaxAlignmentError.invalidDimensions) {
        try ParallaxAlignment.shift(reference: [], moving: [], rows: Int.max, columns: 3)
    }
    #expect(throws: ParallaxAlignmentError.invalidDimensions) {
        try ParallaxAlignment.shifted([], rows: 0, columns: 0, by: .init(row: 0, column: 0))
    }
    #expect(throws: ParallaxAlignmentError.inconsistentShape) {
        try ParallaxAlignment.shift(reference: image, moving: [], rows: 8, columns: 10)
    }
    #expect(throws: ParallaxAlignmentError.inconsistentShape) {
        try ParallaxAlignment.align([], reference: image, rows: 8, columns: 10)
    }
    for factor in [Int.min, 0, 1, 2, 65, Int.max] {
        #expect(throws: ParallaxAlignmentError.invalidUpsampleFactor) {
            try ParallaxAlignment.shift(reference: image, moving: image, rows: 8, columns: 10, upsampleFactor: factor)
        }
    }
    for bad in [Double.nan, .infinity, -.infinity] {
        var invalid = image; invalid[0] = bad
        #expect(throws: ParallaxAlignmentError.nonfiniteValues) {
            try ParallaxAlignment.shift(reference: invalid, moving: image, rows: 8, columns: 10)
        }
        #expect(throws: ParallaxAlignmentError.nonfiniteValues) {
            try ParallaxAlignment.shifted(image, rows: 8, columns: 10, by: .init(row: bad, column: 0))
        }
    }
    #expect(throws: ParallaxAlignmentError.nonfiniteValues) {
        let huge = image.map { $0 * 1e200 }
        return try ParallaxAlignment.shift(reference: huge, moving: huge, rows: 8, columns: 10)
    }
    for value in [0.0, 1.0] {
        #expect(throws: ParallaxAlignmentError.degenerateCorrelation) {
            try ParallaxAlignment.shift(reference: image, moving: Array(repeating: value, count: 80), rows: 8, columns: 10)
        }
    }
}

@Test func pythonMatrixDFTPatchPins() throws {
    let fixture = try pins()
    for item in fixture.cases {
        let fft = FFT2D(rows: item.rows, columns: item.columns)
        var rr = item.reference, ri = Array(repeating: 0.0, count: rr.count)
        var mr = item.frames[0], mi = ri
        fft.transform(&rr, &ri, inverse: false)
        fft.transform(&mr, &mi, inverse: false)
        let real = rr.indices.map { rr[$0] * mr[$0] + ri[$0] * mi[$0] }
        let imaginary = rr.indices.map { ri[$0] * mr[$0] - rr[$0] * mi[$0] }
        let patch = MatrixDFTCorrelation.dftUpsampleOfConjugate(
            correlationRe: real, correlationIm: imaginary,
            width: item.columns, height: item.rows,
            centerRow: fixture.patchCenter[0], centerColumn: fixture.patchCenter[1],
            upsampleFactor: fixture.upsampleFactor, patchSize: 12)
        close(patch, item.patches[0])
    }
}
