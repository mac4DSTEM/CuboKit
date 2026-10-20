import Foundation
import Testing
import CuboCore

private struct ViewCase: Decodable {
    let name: String, integer: Bool, bin: Int, shape: [Int], values: [Double], discardedRows: Int, discardedColumns: Int
    let scanCrop: [Int]?, detectorCrop: [Int]?
}
private struct Pin: Decodable { let sourceShape: [Int], floats: [Double], ints: [Double], cases: [ViewCase] }
private func pin() throws -> Pin {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/cubeview.json")
    return try JSONDecoder().decode(Pin.self, from: Data(contentsOf: url))
}
private func crop(_ a: [Int]?) -> AxisCrop? { a.map { AxisCrop(rowOffset: $0[0], columnOffset: $0[2], rows: $0[1], columns: $0[3]) } }

@Suite struct CubeViewTests {
    // Shapes, crops, indices and integer-valued sums are exact; non-integer sums are Double-accumulated and rounded once,
    // against NumPy's Float32 accumulation: relative 1e-6 of max(1, max|value|).
    @Test func viewAgainstCropAndBinData() async throws {
        let p = try pin()
        let shape = try CubeShape(p.sourceShape)
        for c in p.cases {
            let values = (c.integer ? p.ints : p.floats).map { Float($0) }
            let cube = try MemoryCube(shape: shape, values: values)
            let spec = CubeViewSpecification(scanCrop: crop(c.scanCrop), detectorCrop: crop(c.detectorCrop), detectorBin: c.bin)
            let view = try ViewedCube(base: cube, specification: spec)
            #expect(view.shape.extents == c.shape)
            #expect(view.geometry.discardedDetectorRows == c.discardedRows && view.geometry.discardedDetectorColumns == c.discardedColumns)
            let all = try await MemoryCube(loading: view, tiling: .rows(2)).values
            #expect(all.count == c.values.count)
            let worst = zip(all, c.values).map { abs(Double($0) - $1) }.max() ?? 0
            let scale = max(1, c.values.map(abs).max() ?? 1)
            print("CUBEVIEW \(c.name) max abs gap \(worst)")
            if c.integer { #expect(worst == 0) } else { #expect(worst / scale < 1e-6) }
            // one pattern equals the same pattern of the loaded view
            let last = view.shape.scanRows - 1, col = view.shape.scanColumns - 1
            let pattern = try await view.readPattern(row: last, column: col)
            let start = ((last * view.shape.scanColumns) + col) * view.shape.detectorCount
            #expect(pattern.values == Array(all[start..<(start + view.shape.detectorCount)]))
        }
    }

    @Test func fullExtentIsTheSourceAndRefusalsAreNotClamps() async throws {
        let shape = try CubeShape(scanRows: 3, scanColumns: 4, detectorRows: 5, detectorColumns: 6)
        let cube = try MemoryCube(shape: shape, values: (0..<shape.elementCount).map(Float.init))
        let view = try ViewedCube(base: cube, specification: .fullExtent)
        #expect(view.shape == shape && CubeViewSpecification.fullExtent.isFullExtent)
        #expect(try await MemoryCube(loading: view).values == cube.values)
        for bad in [CubeViewSpecification(scanCrop: AxisCrop(rowOffset: 2, columnOffset: 0, rows: 2, columns: 4)),
                    CubeViewSpecification(scanCrop: AxisCrop(rowOffset: -1, columnOffset: 0, rows: 2, columns: 2)),
                    CubeViewSpecification(detectorCrop: AxisCrop(rowOffset: 0, columnOffset: 1, rows: 5, columns: 6)),
                    CubeViewSpecification(detectorCrop: AxisCrop(rowOffset: 0, columnOffset: 0, rows: 0, columns: 2)),
                    CubeViewSpecification(detectorBin: 0), CubeViewSpecification(detectorBin: 7),
                    CubeViewSpecification(detectorCrop: AxisCrop(rowOffset: 0, columnOffset: 0, rows: 3, columns: 6), detectorBin: 4)] {
            #expect(throws: CuboError.self) { _ = try CubeViewGeometry(source: shape, specification: bad) }
        }
        await #expect(throws: CuboError.self) { _ = try await view.readScanTile(rows: 2..<5) }
    }

    // The crop arithmetic of the app's drag selection (a transcription, evaluated by hand): corners floored, far corner included,
    // scaled by the overview stride, clamped, at least one pixel, nil when it is the whole plane.
    @Test func enclosingCropArithmetic() {
        // rows 1.2...3.9, columns 2.7...5.1 on a 10 x 12 plane: rows 1..<4, columns 2..<6
        #expect(AxisCrop.enclosing(firstRow: 1.2, firstColumn: 2.7, lastRow: 3.9, lastColumn: 5.1, planeRows: 10, planeColumns: 12)
                == AxisCrop(rowOffset: 1, columnOffset: 2, rows: 3, columns: 4))
        // reversed corners give the same crop
        #expect(AxisCrop.enclosing(firstRow: 3.9, firstColumn: 5.1, lastRow: 1.2, lastColumn: 2.7, planeRows: 10, planeColumns: 12)
                == AxisCrop(rowOffset: 1, columnOffset: 2, rows: 3, columns: 4))
        // overview stride 4 x 2: rows (1)*4..<(3+1)*4 = 4..<16 clamped to 10; columns 2*2..<(5+1)*2 = 4..<12
        #expect(AxisCrop.enclosing(firstRow: 1.2, firstColumn: 2.7, lastRow: 3.9, lastColumn: 5.1, strideRows: 4, strideColumns: 2,
                                   planeRows: 10, planeColumns: 12) == AxisCrop(rowOffset: 4, columnOffset: 4, rows: 6, columns: 8))
        // a point is one pixel; outside the plane clamps to the edge pixel
        #expect(AxisCrop.enclosing(firstRow: 20, firstColumn: -5, lastRow: 20, lastColumn: -5, planeRows: 10, planeColumns: 12)
                == AxisCrop(rowOffset: 9, columnOffset: 0, rows: 1, columns: 1))
        // the whole plane is no crop; non-finite input is none
        #expect(AxisCrop.enclosing(firstRow: -3, firstColumn: -3, lastRow: 50, lastColumn: 50, planeRows: 10, planeColumns: 12) == nil)
        #expect(AxisCrop.enclosing(firstRow: .nan, firstColumn: 0, lastRow: 1, lastColumn: 1, planeRows: 10, planeColumns: 12) == nil)
    }
}
