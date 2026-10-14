//
//  NumPySum.swift
//  NumPy's float64 summation order (numpy/_core/src/umath/loops_utils.h, pairwise_sum): fewer than 8 values summed in
//  order from 0; up to 128 in eight interleaved partial sums combined as ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
//  plus the remainder in order; above 128 split in two at a multiple of 8 and recursed. np.sum and np.mean of a contiguous
//  array (a 2-D array flattened) follow it exactly, so a port that sums this way reproduces NumPy's bits (checked against
//  NumPy 2.5.0 on 400 random arrays, 2026-10-15).
//

package enum NumPySum {
    /// NumPy's pairwise sum of `values`.
    package static func pairwise(_ values: UnsafeBufferPointer<Double>) -> Double {
        let n = values.count
        guard let base = values.baseAddress else { return 0 }
        return sum(base, n)
    }

    package static func pairwise(_ values: [Double]) -> Double {
        values.withUnsafeBufferPointer { pairwise($0) }
    }

    /// NumPy's mean: the pairwise sum divided by the count (NaN for no values).
    package static func mean(_ values: [Double]) -> Double {
        values.isEmpty ? .nan : pairwise(values) / Double(values.count)
    }

    private static func sum(_ a: UnsafePointer<Double>, _ n: Int) -> Double {
        if n < 8 {
            var res = 0.0
            for i in 0..<n { res += a[i] }
            return res
        } else if n <= 128 {
            var r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3], r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7]
            var i = 8
            while i < n - (n % 8) {
                r0 += a[i]; r1 += a[i + 1]; r2 += a[i + 2]; r3 += a[i + 3]
                r4 += a[i + 4]; r5 += a[i + 5]; r6 += a[i + 6]; r7 += a[i + 7]
                i += 8
            }
            var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
            while i < n { res += a[i]; i += 1 }
            return res
        } else {
            var n2 = n / 2
            n2 -= n2 % 8
            return sum(a, n2) + sum(a + n2, n - n2)
        }
    }
}
