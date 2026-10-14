//
//  TruthTests.swift
//  Planted truth independent of py4DSTEM (registration 5.5) and the refusals of 3.1.
//

import Foundation
import XCTest
import CuboCore
import CuboVirtual

final class TruthTests: XCTestCase {
    /// digits: value s·10000 + i·100 + j; mean ((N−1)/2)·10000 + i·100 + j and max (N−1)·10000 + i·100 + j, exactly;
    /// one virtual image recomputed here by hand from the predicates.
    func testDigitsClosedForms() async throws {
        let source = try MappedCube(npy: Fixtures.url("digits.npy"))
        let shape = source.shape
        XCTAssertEqual(shape.extents, [2, 3, 5, 7])
        let n = Double(shape.scanCount)
        let (mean, max) = try await VirtualDiffraction.meanAndMax(of: source)
        for i in 0..<5 { for j in 0..<7 {
            XCTAssertEqual(mean[i, j], (n - 1) / 2 * 10000 + Double(i * 100 + j))
            XCTAssertEqual(max[i, j], (n - 1) * 10000 + Double(i * 100 + j))
        } }
        // The off-centre circle, by hand: (i − 2.3)² + (j − 4.7)² < 2.9².
        let image = try await VirtualImaging.image(of: source, geometry: .circle(centerRow: 2.3, centerColumn: 4.7, radius: 2.9))
        for s in 0..<shape.scanCount {
            var want = 0.0
            for i in 0..<5 { for j in 0..<7 {
                let di = Double(i) - 2.3, dj = Double(j) - 4.7
                if di * di + dj * dj < 2.9 * 2.9 { want += Double(s * 10000 + i * 100 + j) }
            } }
            XCTAssertEqual(image.values[s], want, "scan \(s)")
        }
    }

    /// A cube of ones: every image is the mask's weight sum; the mean and max are 1.
    func testOnesCube() async throws {
        let shape = try CubeShape(scanRows: 3, scanColumns: 2, detectorRows: 9, detectorColumns: 11)
        let cube = try MemoryCube(shape: shape, values: [Float](repeating: 1, count: shape.elementCount))
        let weights: [Float] = (0..<shape.detectorCount).map { [0, 0.5, 1, 2, -1][$0 % 5] }
        for geometry: DetectorGeometry in [.circle(centerRow: 4, centerColumn: 5, radius: 3.5),
                                           .annulus(centerRow: 4.5, centerColumn: 5.5, inner: 1.2, outer: 4.1),
                                           .rectangle(rowMin: 1, rowMax: 7, columnMin: 2, columnMax: 30),
                                           .point(row: 8, column: 10), .mask(weights)] {
            let want = try DetectorMask.make(geometry, shape: shape).reduce(0.0) { $0 + Double($1) }
            for path in paths {
                let image = try await VirtualImaging.image(of: cube, geometry: geometry, path: path)
                XCTAssertEqual(image.values, [Double](repeating: want, count: shape.scanCount), "\(geometry) \(path)")
            }
        }
        for path in paths {
            let (mean, max) = try await VirtualDiffraction.meanAndMax(of: cube, path: path)
            XCTAssertEqual(mean.values, [Double](repeating: 1, count: shape.detectorCount))
            XCTAssertEqual(max.values, [Double](repeating: 1, count: shape.detectorCount))
        }
    }

    /// Counted by hand: i² + j² < 4 holds 9 pixels, < 9 holds 25; 1 < i² + j² < 6.25 holds 16.
    func testBoundaryCountsByHand() throws {
        for (qy, qx) in [(16, 16), (32, 32)] {
            let c = Double(qy / 2), d = Double(qx / 2)
            func count(_ g: DetectorGeometry) throws -> Int {
                try DetectorMask.make(g, detectorRows: qy, detectorColumns: qx).filter { $0 != 0 }.count
            }
            XCTAssertEqual(try count(.circle(centerRow: c, centerColumn: d, radius: 2)), 9)
            XCTAssertEqual(try count(.circle(centerRow: c, centerColumn: d, radius: 3)), 25)
            XCTAssertEqual(try count(.annulus(centerRow: c, centerColumn: d, inner: 1, outer: 2.5)), 16)
        }
    }

    /// np.round is half to even: (0.5, 2.5, 1.5, 3.5) selects rows 0..<2 and columns 2..<4; a stop past the edge clips.
    func testRectangleRoundsHalfToEvenAndClips() throws {
        let m = try DetectorMask.make(.rectangle(rowMin: 0.5, rowMax: 2.5, columnMin: 1.5, columnMax: 3.5),
                                      detectorRows: 5, detectorColumns: 7)
        XCTAssertEqual(m.indices.filter { m[$0] != 0 }, [2, 3, 9, 10])
        let clipped = try DetectorMask.make(.rectangle(rowMin: 3, rowMax: 99, columnMin: 5, columnMax: 99),
                                            detectorRows: 5, detectorColumns: 7)
        XCTAssertEqual(clipped.indices.filter { clipped[$0] != 0 }, [26, 27, 33, 34])
        let empty = try DetectorMask.make(.rectangle(rowMin: 4, rowMax: 2, columnMin: 0, columnMax: 7),
                                          detectorRows: 5, detectorColumns: 7)
        XCTAssertEqual(empty.filter { $0 != 0 }.count, 0)
    }

    /// A cube whose pattern at scan index s is s·P: the mean is ((N − 1)/2)·P and the image s·Σ P·mask, exactly.
    func testScaledPatternCube() async throws {
        let shape = try CubeShape(scanRows: 4, scanColumns: 5, detectorRows: 6, detectorColumns: 7)
        let p: [Float] = (0..<shape.detectorCount).map { Float(($0 * 37) % 23) - 5 }
        var values = [Float]()
        for s in 0..<shape.scanCount { values += p.map { Float(s) * $0 } }
        let cube = try MemoryCube(shape: shape, values: values)
        let n = Double(shape.scanCount)
        let geometry = DetectorGeometry.annulus(centerRow: 2.5, centerColumn: 3, inner: 1, outer: 3)
        let mask = try DetectorMask.make(geometry, shape: shape)
        let pm = zip(p, mask).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }
        for path in paths {
            let (mean, _) = try await VirtualDiffraction.meanAndMax(of: cube, path: path)
            XCTAssertEqual(mean.values, p.map { (n - 1) / 2 * Double($0) }, "\(path)")
            let image = try await VirtualImaging.image(of: cube, geometry: geometry, path: path)
            XCTAssertEqual(image.values, (0..<shape.scanCount).map { Double($0) * pm }, "\(path)")
        }
    }

    /// The maximum is seeded at −∞: an all-negative cube's maximum is negative.
    func testAllNegativeMaximum() async throws {
        let source = try MappedCube(npy: Fixtures.url("negative.npy"))
        for path in paths {
            let max = try await VirtualDiffraction.max(of: source, path: path)
            XCTAssertTrue(max.values.allSatisfy { $0 < 0 }, "\(path)")
        }
    }

    /// Registration 3.1 and the DEVIATION in the refusing direction.
    func testRefusals() throws {
        func refuses(_ g: DetectorGeometry, _ want: CuboVirtualError, line: UInt = #line) {
            XCTAssertThrowsError(try DetectorMask.make(g, detectorRows: 8, detectorColumns: 9), line: line) {
                XCTAssertEqual($0 as? CuboVirtualError, want, line: line)
            }
        }
        refuses(.annulus(centerRow: 4, centerColumn: 4, inner: 3, outer: 3), .innerNotBelowOuter(inner: 3, outer: 3))
        refuses(.annulus(centerRow: 4, centerColumn: 4, inner: 3, outer: 2), .innerNotBelowOuter(inner: 3, outer: 2))
        refuses(.annulus(centerRow: 4, centerColumn: 4, inner: -1, outer: 2), .negativeRadius(-1))
        refuses(.circle(centerRow: 4, centerColumn: 4, radius: -2), .negativeRadius(-2))
        refuses(.circle(centerRow: .nan, centerColumn: 4, radius: 2), .nonFiniteGeometry)
        refuses(.circle(centerRow: 4, centerColumn: 4, radius: .infinity), .nonFiniteGeometry)
        refuses(.point(row: 8, column: 0), .pointOutsideDetector(row: 8, column: 0))
        refuses(.point(row: 0, column: -1), .pointOutsideDetector(row: 0, column: -1))
        refuses(.rectangle(rowMin: -1, rowMax: 3, columnMin: 0, columnMax: 2), .negativeRectangleBound)
        refuses(.mask([Float](repeating: 1, count: 71)), .maskLengthMismatch(expected: 72, received: 71))
        refuses(.mask([Float](repeating: .nan, count: 72)), .nonFiniteGeometry)
        // −0.4 rounds to −0 (not below zero): accepted, as NumPy's int(np.round(−0.4)) == 0.
        XCTAssertNoThrow(try DetectorMask.make(.rectangle(rowMin: -0.4, rowMax: 3, columnMin: 0, columnMax: 2),
                                               detectorRows: 8, detectorColumns: 9))
    }

    private var paths: [ComputePath] { VirtualDiffraction.metalIsAvailable ? [.cpu, .metal] : [.cpu] }
}
