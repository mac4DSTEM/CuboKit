//
//  CompensatedSum.swift
//  The CPU reference sum of the mean patterns: per detector pixel a running Double sum and the exact rounding error of every
//  addition (Knuth's TwoSum, branch free, four lanes at a time). It replaces the scalar Neumaier loop with its per-element
//  finite check and its magnitude branch; the error of an addition is the same number either way (both are exact for
//  round-to-nearest), so the result is the bit pattern of the old loop. The finite check moved to ``finish``: once a sum is
//  ∞ or NaN it stays so, and then the correction (NaN from ∞ − ∞) is dropped, as the old loop never added it.
//

import CuboCore
import Foundation

enum CompensatedSum {
    /// Adds `count` Float32 values to `sum[0..<count]`, their rounding errors to `correction`. With `weight` each value is
    /// multiplied in Double first (the product is the one a scalar `Double(x) * w` gives).
    @inline(__always)
    static func add(_ values: UnsafePointer<Float>, weight: Double?, sum: UnsafeMutablePointer<Double>,
                    correction: UnsafeMutablePointer<Double>, count: Int) {
        let raw = UnsafeRawPointer(values)
        var i = 0
        if let w = weight {
            while i + 4 <= count {
                let x = SIMD4<Double>(raw.loadUnaligned(fromByteOffset: i * 4, as: SIMD4<Float>.self)) * w
                step(x, sum + i, correction + i)
                i += 4
            }
            while i < count { scalar(Double(values[i]) * w, sum + i, correction + i); i += 1 }
        } else {
            while i + 4 <= count {
                step(SIMD4<Double>(raw.loadUnaligned(fromByteOffset: i * 4, as: SIMD4<Float>.self)), sum + i, correction + i)
                i += 4
            }
            while i < count { scalar(Double(values[i]), sum + i, correction + i); i += 1 }
        }
    }

    /// `rows` calls of `add(_:weight: nil, …)`, row k at `values + k * stride`, bit for bit: each pixel still takes the rows in
    /// order with the same TwoSum. The rows go in passes of four with a pixel's sum and correction held in registers, so the
    /// running sum is not stored and reloaded between rows (that store-to-load chain was most of the time, CuboKit#7).
    @inline(__always)
    static func addRows(_ values: UnsafePointer<Float>, stride: Int, rows: Int, sum: UnsafeMutablePointer<Double>,
                        correction: UnsafeMutablePointer<Double>, count: Int) {
        var k = 0
        while k + 4 <= rows {
            fourRows(values + k * stride, stride: stride, sum, correction, count)
            k += 4
        }
        while k < rows {
            add(values + k * stride, weight: nil, sum: sum, correction: correction, count: count)
            k += 1
        }
    }

    @inline(__always)
    private static func fourRows(_ values: UnsafePointer<Float>, stride: Int, _ s: UnsafeMutablePointer<Double>,
                                 _ c: UnsafeMutablePointer<Double>, _ count: Int) {
        let raw = UnsafeRawPointer(values), rawS = UnsafeMutableRawPointer(s), rawC = UnsafeMutableRawPointer(c)
        var i = 0
        while i + 4 <= count {
            var a = rawS.loadUnaligned(fromByteOffset: i * 8, as: SIMD4<Double>.self)
            var e = rawC.loadUnaligned(fromByteOffset: i * 8, as: SIMD4<Double>.self)
            for r in 0..<4 {
                let x = SIMD4<Double>(raw.loadUnaligned(fromByteOffset: (r * stride + i) * 4, as: SIMD4<Float>.self))
                let t = a + x
                let bb = t - a
                e = e + ((a - (t - bb)) + (x - bb))
                a = t
            }
            rawS.storeBytes(of: a, toByteOffset: i * 8, as: SIMD4<Double>.self)
            rawC.storeBytes(of: e, toByteOffset: i * 8, as: SIMD4<Double>.self)
            i += 4
        }
        while i < count {
            for r in 0..<4 { scalar(Double(values[r * stride + i]), s + i, c + i) }
            i += 1
        }
    }

    @inline(__always)
    private static func step(_ x: SIMD4<Double>, _ s: UnsafeMutablePointer<Double>, _ c: UnsafeMutablePointer<Double>) {
        let raw = UnsafeMutableRawPointer(s), rawC = UnsafeMutableRawPointer(c)
        let a = raw.loadUnaligned(as: SIMD4<Double>.self)
        let t = a + x
        let bb = t - a
        let e = (a - (t - bb)) + (x - bb)
        raw.storeBytes(of: t, as: SIMD4<Double>.self)
        rawC.storeBytes(of: rawC.loadUnaligned(as: SIMD4<Double>.self) + e, as: SIMD4<Double>.self)
    }

    @inline(__always)
    private static func scalar(_ x: Double, _ s: UnsafeMutablePointer<Double>, _ c: UnsafeMutablePointer<Double>) {
        let a = s.pointee
        let t = a + x
        let bb = t - a
        c.pointee += (a - (t - bb)) + (x - bb)
        s.pointee = t
    }

    /// The compensated sum `sum + correction` of one pixel. A sum that is ∞ or NaN is returned as it is (its correction is
    /// NaN from ∞ − ∞); a sum of −0 (every value −0, the identity of a column that started at −0) keeps its sign.
    @inline(__always)
    static func value(sum: Double, correction: Double) -> Double {
        if !sum.isFinite { return sum }
        if sum == 0 && sum.sign == .minus && correction == 0 { return sum }
        return sum + correction
    }
}

/// The values of scan rows `range` of `source` as a pointer valid inside `body`: read in place for a ``MemoryCube`` (no tile
/// copy), else from `readScanTile`.
func withTileValues<R>(of source: some CubeSource, rows range: Range<Int>,
                       _ body: (UnsafePointer<Float>) throws -> R) async throws -> R {
    if let memory = source as? MemoryCube {
        try source.shape.requireRows(range)
        let offset = range.lowerBound * source.shape.elementsPerScanRow
        return try memory.values.withUnsafeBufferPointer { try body($0.baseAddress! + offset) }
    }
    let tile = try await source.readScanTile(rows: range)
    return try tile.values.withUnsafeBufferPointer { try body($0.baseAddress!) }
}
