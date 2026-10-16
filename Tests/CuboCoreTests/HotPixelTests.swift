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
            // Pinned actual Float32 py4DSTEM calls: means, masks and all output samples bit for bit.
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
}
