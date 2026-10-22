//
//  BraggVirtualImage.swift
//  py4DSTEM's `BraggVectors.get_virtual_image`: the intensity of the Bragg peaks that fall inside a circular or annular detector,
//  summed at every scan position. The peaks are taken as they are: py4DSTEM first applies its calibrations (centre, ellipse,
//  pixel size, rotation) to the vectors; here the caller hands over vectors already in the coordinates of the detector.
//  py4DSTEM's x is the row and y the column.
//

import CuboCore
import Foundation

/// A point in the detector plane.
public struct BraggDetectorCenter: Sendable, Equatable {
    public var row: Double
    public var column: Double
    public init(row: Double, column: Double) {
        self.row = row
        self.column = column
    }
}

/// The detector `get_virtual_image` integrates over.
public enum BraggVirtualDetector: Sendable, Equatable {
    /// Every peak, whatever its position (`mode=None` with an empty radial range): the plain sum of the intensities, so a NaN
    /// intensity gives NaN.
    case allPeaks
    /// `mode="circle"`: peaks at distance `< radius` from `center`; a nil centre is the origin (0, 0).
    case circle(center: BraggDetectorCenter?, radius: Double)
    /// `mode="annulus"`: peaks at distance `>= inner` and `< outer` from `center`; a nil centre is the origin (0, 0).
    /// `mode=None` without geometry is `.annulus(center: nil, inner: 0, outer: .infinity)`.
    case annulus(center: BraggDetectorCenter?, inner: Double, outer: Double)
}

/// A virtual image from Bragg vectors: `scanRows × scanColumns` values, row-major.
public struct BraggVirtualImage: Sendable, Equatable {
    public let scanRows: Int
    public let scanColumns: Int
    public let values: [Double]

    public subscript(row: Int, column: Int) -> Double { values[row * scanColumns + column] }
}

extension BraggVectors {
    /// `get_virtual_image(mode, geometry)` on vectors already in detector coordinates (all four calibrations off in py4DSTEM
    /// terms). At each scan position the peaks inside the detector are summed (NumPy's pairwise order); a position with no peak
    /// inside gets 0. NaN positions fall outside every detector but `.allPeaks`.
    public func virtualImage(detector: BraggVirtualDetector) -> BraggVirtualImage {
        var values = [Double](repeating: 0, count: peaks.count)
        for (position, list) in peaks.enumerated() where !list.isEmpty {
            switch detector {
            case .allPeaks:
                values[position] = NumPySum.pairwise(list.map(\.intensity))
            case .circle(let center, let radius):
                values[position] = Self.sum(list, center: center, inner: 0, outer: radius)
            case .annulus(let center, let inner, let outer):
                values[position] = Self.sum(list, center: center, inner: inner, outer: outer)
            }
        }
        return BraggVirtualImage(scanRows: scanRows, scanColumns: scanColumns, values: values)
    }

    private static func sum(_ list: [BraggPeak], center: BraggDetectorCenter?, inner: Double, outer: Double) -> Double {
        var selected = [Double]()
        for peak in list {
            let r = center.map { hypot(peak.row - $0.row, peak.column - $0.column) } ?? hypot(peak.row, peak.column)
            if r >= inner && r < outer { selected.append(peak.intensity) }
        }
        return selected.isEmpty ? 0 : NumPySum.pairwise(selected)
    }
}
