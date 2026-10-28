import Foundation
import Testing
import CuboCore

private struct HotPixelPin: Decodable {
    let shape: [Int], threshold: Double, index: Int
    let values: [Float], mean: [Float], mask: [Bool], filtered: [Float]
}
private struct HotPixelPins: Decodable { let cases: [HotPixelPin] }
private func hotPixelPins() throws -> [HotPixelPin] {
    let file = URL(fileURLWithPath:#filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/hot-pixels.json")
    return try JSONDecoder().decode(HotPixelPins.self,from:Data(contentsOf:file)).cases
}
@Suite struct HotPixelTests {
    @Test func filterAgainstPy4DSTEM() throws {
        for p in try hotPixelPins() {
            let shape = try CubeShape(p.shape)
            let cube = try MemoryCube(shape:shape,values:p.values)
            let result = try HotPixelFilter.filter(cube,threshold:p.threshold,comparisonIndex:p.index)
            // Pinned actual Float32 py4DSTEM calls: means, masks and all output samples, asserted exact today (a tolerance against py4DSTEM would be allowed for the samples).
            #expect(result.mean.map(\.bitPattern) == p.mean.map(\.bitPattern))
            #expect(result.mask == p.mask)
            #expect(result.cube.values.map(\.bitPattern) == p.filtered.map(\.bitPattern))
            #expect(cube.values == p.values)
        }
    }
    @Test func meanAndFrameEntryPoints() throws {
        for p in try hotPixelPins() {
            let shape = try CubeShape(p.shape)
            let mask = try HotPixelFilter.mask(mean:p.mean,rows:shape.detectorRows,columns:shape.detectorColumns,
                threshold:p.threshold,comparisonIndex:p.index)
            #expect(mask == p.mask)
            for scan in 0..<shape.scanCount {
                let range = scan*shape.detectorCount..<(scan+1)*shape.detectorCount
                let filtered = try HotPixelFilter.replacing(Array(p.values[range]),rows:shape.detectorRows,columns:shape.detectorColumns,mask:mask)
                #expect(filtered.map(\.bitPattern) == p.filtered[range].map(\.bitPattern))
            }
        }
    }
    @Test func invalidInputIsRefused() throws {
        #expect(throws:HotPixelFilterError.self) { try HotPixelFilter.mask(mean:[1],rows:Int.max,columns:2,threshold:1) }
        #expect(throws:HotPixelFilterError.self) { try HotPixelFilter.mask(mean:[1],rows:1,columns:1,threshold:.nan) }
        #expect(throws:HotPixelFilterError.self) { try HotPixelFilter.mask(mean:[1],rows:1,columns:1,threshold:1,comparisonIndex:21) }
        #expect(throws:HotPixelFilterError.self) { try HotPixelFilter.replacing([1],rows:1,columns:1,mask:[]) }
    }

    @Test func replacerMatchesSequentialReplacementBitForBit() throws {
        var rng = SystemRandomNumberGenerator()
        for (rows, columns) in [(1,1),(1,7),(2,2),(5,1),(16,16),(33,17)] {
            for density in [0.0, 0.05, 1.0] {
                let mask = (0..<rows*columns).map { _ in Double.random(in:0..<1,using:&rng) < density }
                var frame = (0..<rows*columns).map { _ in Float.random(in:-3...3,using:&rng) }
                if frame.count > 3 { frame[1] = -0.0; frame[2] = 0.0 }
                let want = referenceReplace(frame,rows:rows,columns:columns,mask:mask)
                let replacer = try HotPixelFilter.Replacer(rows:rows,columns:columns,mask:mask)
                var got = frame
                replacer.apply(to:&got)
                #expect(got.map(\.bitPattern) == want.map(\.bitPattern))
                // Offset frame inside a larger buffer; neighbours untouched.
                var padded = [Float](repeating:7,count:3)+frame+[Float](repeating:9,count:2)
                replacer.apply(to:&padded,base:3)
                #expect(Array(padded[3..<3+frame.count]).map(\.bitPattern) == want.map(\.bitPattern))
                #expect(padded.prefix(3).allSatisfy { $0 == 7 } && padded.suffix(2).allSatisfy { $0 == 9 })
                let viaPublic = try HotPixelFilter.replacing(frame,rows:rows,columns:columns,mask:mask)
                #expect(viaPublic.map(\.bitPattern) == want.map(\.bitPattern))
            }
        }
        #expect(throws:HotPixelFilterError.self) { try HotPixelFilter.Replacer(rows:2,columns:2,mask:[true]) }
        #expect(throws:HotPixelFilterError.self) { try HotPixelFilter.Replacer(rows:0,columns:2,mask:[]) }
    }

    /// Timing, printed only when CUBO_BENCH is set: 256x256 Float32, 40 masked pixels, one thread.
    @Test func replacementTiming() throws {
        guard ProcessInfo.processInfo.environment["CUBO_BENCH"] != nil else { return }
        let rows = 256, columns = 256, patterns = 2000
        var rng = SystemRandomNumberGenerator()
        var mask = [Bool](repeating:false,count:rows*columns)
        for _ in 0..<40 { mask[Int.random(in:0..<rows*columns,using:&rng)] = true }
        let base = (0..<rows*columns).map { _ in Float.random(in:0...100,using:&rng) }
        let clock = ContinuousClock()
        var sink: Float = 0
        let before = clock.measure {
            for _ in 0..<patterns {
                guard base.allSatisfy(\.isFinite), mask.count == base.count else { continue }  // the validation `replacing` always did
                sink += referenceReplace(base,rows:rows,columns:columns,mask:mask)[5]
            }
        }
        let current = clock.measure {
            for _ in 0..<patterns { sink += (try? HotPixelFilter.replacing(base,rows:rows,columns:columns,mask:mask))?[5] ?? 0 }
        }
        let replacer = try HotPixelFilter.Replacer(rows:rows,columns:columns,mask:mask)
        var work = base
        let fast = clock.measure {
            for _ in 0..<patterns {
                work.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from:base,count:base.count) }
                replacer.apply(to:&work); sink += work[5]
            }
        }
        func ms(_ d: Duration) -> Double { (Double(d.components.seconds)*1e3+Double(d.components.attoseconds)/1e15)/Double(patterns) }
        print("HOTPIXEL-BENCH old \(ms(before)) ms, replacing \(ms(current)) ms, Replacer(incl. 256 KB copy) \(ms(fast)) ms per pattern (\(sink))")
    }
}

/// The pre-Replacer implementation, kept verbatim as the bit-for-bit reference.
private func referenceReplace(_ frame: [Float], rows: Int, columns: Int, mask: [Bool]) -> [Float] {
    var values = frame
    var window = [Float](); window.reserveCapacity(9)
    for k in mask.indices where mask[k] {
        let r = k/columns, c = k%columns
        window.removeAll(keepingCapacity:true)
        for rr in max(0,r-1)..<min(rows,r+2) { for cc in max(0,c-1)..<min(columns,c+2) {
            window.append(values[rr*columns+cc])
        } }
        window.sort()
        let middle = window.count/2
        values[k] = window.count%2 == 1 ? 0 + window[middle] : (0 + window[middle-1]+window[middle])/2
    }
    return values
}
