import Testing
import CuboCore

/// 0.7.0: the in-place loop entry `HotPixelFilter.replacing(in:base:plan:)` is bit-identical to the one-off `replacing`.
@Suite struct ReplacingPlanTests {
    private func frame(_ rows: Int, _ columns: Int) -> [Float] {
        var state: UInt32 = 12345
        return (0..<rows*columns).map { _ in state = state &* 1664525 &+ 1013904223; return Float(state >> 12) / 1000 }
    }

    @Test func planEntryEqualsOneOffEntryAcrossFrames() throws {
        let rows = 17, columns = 23
        var mask = [Bool](repeating: false, count: rows * columns)
        for k in [0, 5, 22, 23, 24, 100, 200, rows * columns - 1] { mask[k] = true }
        let plan = try HotPixelFilter.Replacer(rows: rows, columns: columns, mask: mask)
        let frames = (0..<3).map { _ in frame(rows, columns) }
        var stack = frames.flatMap { $0 }
        try stack.withUnsafeMutableBufferPointer { buffer in
            for index in frames.indices { try HotPixelFilter.replacing(in: buffer, base: index * rows * columns, plan: plan) }
        }
        let expected = try frames.flatMap { try HotPixelFilter.replacing($0, rows: rows, columns: columns, mask: mask) }
        #expect(stack == expected)
    }

    @Test func refusesNonfiniteNeighbourhoodsAndFramesOutsideTheBuffer() throws {
        var mask = [Bool](repeating: false, count: 36); mask[7] = true
        let plan = try HotPixelFilter.Replacer(rows: 6, columns: 6, mask: mask)
        var pixels = frame(6, 6)
        pixels[8] = .nan                                   // a neighbour of pixel 7
        let copy = pixels
        try pixels.withUnsafeMutableBufferPointer { buffer in
            #expect(throws: HotPixelFilterError.invalidInput) { try HotPixelFilter.replacing(in: buffer, plan: plan) }
            #expect(throws: HotPixelFilterError.invalidInput) { try HotPixelFilter.replacing(in: buffer, base: 1, plan: plan) }
            try HotPixelFilter.replacing(in: buffer, plan: plan, validatingFinite: false)   // caller took responsibility
        }
        #expect(pixels != copy || pixels[7].isNaN)
        var far = frame(6, 6); far[35] = .nan                // far from the mask: not read, not checked
        try far.withUnsafeMutableBufferPointer { try HotPixelFilter.replacing(in: $0, plan: plan) }
    }
}
