//
//  Results.swift
//  What CuboVirtual returns: Double images that record the path that made them (registration 1.2).
//
//  Double on purpose: py4DSTEM's virtual image of an integer cube is an exact integer sum (NumPy sums uint16 × bool in
//  uint64); a Float32 result would round it above 2^24, and a kit whose numbers are the reference cannot round where the
//  reference does not.
//

import Foundation

/// Where a reduction ran. There is no automatic choice: a number never depends on the hardware unless the caller asks.
public enum ComputePath: String, Sendable, Hashable, Codable, CaseIterable {
    /// Accelerate on the CPU, Double accumulation: the reference path.
    case cpu
    /// Metal compute kernels, Float32 accumulation, Double combination across tiles; within the registered tolerance of
    /// the CPU path (registration 3.4).
    case metal
}

/// A real-space image, one value per scan position, `scanRows × scanColumns`, row-major.
public struct ScanImage: Sendable, Equatable {
    public let scanRows: Int
    public let scanColumns: Int
    public let values: [Double]
    public let path: ComputePath

    public init(scanRows: Int, scanColumns: Int, values: [Double], path: ComputePath) {
        precondition(values.count == scanRows * scanColumns, "ScanImage: \(values.count) values for \(scanRows) × \(scanColumns)")
        self.scanRows = scanRows
        self.scanColumns = scanColumns
        self.values = values
        self.path = path
    }

    public subscript(row: Int, column: Int) -> Double { values[row * scanColumns + column] }
}

/// A diffraction-space image, one value per detector pixel, `detectorRows × detectorColumns`, row-major.
public struct DetectorImage: Sendable, Equatable {
    public let detectorRows: Int
    public let detectorColumns: Int
    public let values: [Double]
    public let path: ComputePath

    public init(detectorRows: Int, detectorColumns: Int, values: [Double], path: ComputePath) {
        precondition(values.count == detectorRows * detectorColumns,
                     "DetectorImage: \(values.count) values for \(detectorRows) × \(detectorColumns)")
        self.detectorRows = detectorRows
        self.detectorColumns = detectorColumns
        self.values = values
        self.path = path
    }

    public subscript(row: Int, column: Int) -> Double { values[row * detectorColumns + column] }
}

public enum CuboVirtualError: Error, Equatable, Sendable, LocalizedError {
    case innerNotBelowOuter(inner: Double, outer: Double)
    case negativeRadius(Double)
    case negativeRectangleBound
    case nonFiniteGeometry
    case pointOutsideDetector(row: Int, column: Int)
    case maskLengthMismatch(expected: Int, received: Int)
    /// A Metal path was asked for on a machine without a Metal device; the caller may run `.cpu` instead.
    case noMetalDevice
    case metal(String)

    public var errorDescription: String? {
        switch self {
        case .innerNotBelowOuter(let i, let o): "The annulus's outer radius (\(o)) must exceed its inner radius (\(i))."
        case .negativeRadius(let r): "A detector radius cannot be negative (\(r))."
        case .negativeRectangleBound: "A rectangle bound cannot round below zero."
        case .nonFiniteGeometry: "A detector geometry needs finite values."
        case .pointOutsideDetector(let r, let c): "The point (\(r), \(c)) is outside the detector."
        case .maskLengthMismatch(let e, let r): "A mask needs \(e) weights, one per detector pixel; got \(r)."
        case .noMetalDevice: "No Metal device is available; run the CPU path."
        case .metal(let detail): "The Metal path failed: \(detail)."
        }
    }
}
