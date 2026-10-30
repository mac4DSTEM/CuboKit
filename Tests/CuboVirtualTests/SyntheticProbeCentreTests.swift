import Foundation
import Testing
import CuboVirtual

// `SyntheticProbe.make(..., centre:)`: a caller-chosen centre for the logistic disk. nil keeps the (rows/2, columns/2) centre
// in Double, a half pixel off the grid for an odd side. A pixel centre gives the disk a pixel-centred kernel builds.
@Suite struct SyntheticProbeCentreTests {
    /// The logistic disk 1 / (1 + exp(4 (|q - centre| - radius) / width)), written out independently of the kit.
    private func disk(radius: Double, width: Double, rows: Int, columns: Int, centre: (Double, Double)) -> [Double] {
        (0..<(rows * columns)).map { k in
            let r = Double(k / columns) - centre.0, c = Double(k % columns) - centre.1
            return 1 / (1 + exp(4 * ((r * r + c * c).squareRoot() - radius) / width))
        }
    }

    /// A pixel-centred disk the way a pixel-centred kernel builds it: integer offsets from (rows / 2, columns / 2), integer
    /// division, Float radius and width widened to Double.
    private func pixelCentredDisk(radius: Float, width: Float, rows: Int, columns: Int) -> [Double] {
        let centreRow = rows / 2, centreColumn = columns / 2
        var disk = [Double](repeating: 0, count: rows * columns)
        for y in 0..<rows {
            for x in 0..<columns {
                let r = Double(y - centreRow), c = Double(x - centreColumn)
                disk[y * columns + x] = 1 / (1 + exp(4 * ((r * r + c * c).squareRoot() - Double(radius)) / Double(width)))
            }
        }
        return disk
    }

    @Test func pixelCentreOnAnOddDetectorIsThePixelCentredDisk() throws {
        for (rows, columns) in [(25, 23), (125, 125), (32, 30)] {
            let radius: Float = 4.5, width: Float = 1.7
            let probe = try SyntheticProbe.make(radius: Double(radius), width: Double(width), detectorRows: rows, detectorColumns: columns,
                                                centre: (row: Double(rows / 2), column: Double(columns / 2)))
            #expect(probe.values == pixelCentredDisk(radius: radius, width: width, rows: rows, columns: columns))
        }
    }

    @Test func defaultCentreIsUnchanged() throws {
        for (rows, columns) in [(25, 23), (32, 30)] {
            let plain = try SyntheticProbe.make(radius: 6, width: 1.5, detectorRows: rows, detectorColumns: columns)
            let none = try SyntheticProbe.make(radius: 6, width: 1.5, detectorRows: rows, detectorColumns: columns, centre: nil)
            let half = disk(radius: 6, width: 1.5, rows: rows, columns: columns, centre: (Double(rows) / 2, Double(columns) / 2))
            #expect(plain.values == half)
            #expect(none.values == half)
        }
    }

    @Test func nonFiniteCentreIsInvalidInput() {
        #expect(throws: CuboVirtualError.self) {
            _ = try SyntheticProbe.make(radius: 4, width: 2, detectorRows: 9, detectorColumns: 9, centre: (row: .nan, column: 4))
        }
    }
}
