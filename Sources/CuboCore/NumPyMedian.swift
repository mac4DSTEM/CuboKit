//
//  NumPyMedian.swift
//  `np.median` of one set of values, the one median helper the modules share: NaN when any value is NaN, otherwise the
//  values sorted and the middle one taken, or the two middle ones averaged as `(a + b) / 2` for an even count. The
//  windowed medians of `HotPixelFilter` and `CubeEditing.medianFilterMaskedPixels` keep their own sorted-window loops.
//

package enum NumPyMedian {
    /// The median of `values`, `nil` when there are none.
    package static func median(of values: [Double]) -> Double? { sortedMedian(values) }

    /// The median of `values` in Float arithmetic, `nil` when there are none.
    package static func median(of values: [Float]) -> Float? { sortedMedian(values) }

    /// The median of the values in `buffer`, sorting them in place; `nil` when it is empty.
    package static func medianInPlace(_ buffer: UnsafeMutableBufferPointer<Double>) -> Double? {
        guard !buffer.isEmpty else { return nil }
        if buffer.contains(where: \.isNaN) { return .nan }
        var v = buffer
        v.sort()
        return middle(v)
    }

    @inline(__always)
    private static func sortedMedian<T: BinaryFloatingPoint>(_ values: [T]) -> T? {
        guard !values.isEmpty else { return nil }
        if values.contains(where: \.isNaN) { return .nan }
        return middle(values.sorted())
    }

    @inline(__always)
    private static func middle<C: RandomAccessCollection>(_ s: C) -> C.Element where C.Element: BinaryFloatingPoint, C.Index == Int {
        let n = s.count, i = s.startIndex + n / 2
        return n % 2 == 1 ? s[i] : (s[i - 1] + s[i]) / 2
    }
}
