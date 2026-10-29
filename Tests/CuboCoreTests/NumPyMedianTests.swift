import Testing
@testable import CuboCore

/// `np.median` (NumPy 2.5.4) of a few sets, the one median helper the modules share (#14): the middle value, the mean of the
/// two middle ones, NaN when a NaN is present, nothing for no values; the in-place and Float forms agree with the array form.
@Suite struct NumPyMedianTests {
    @Test func matchesNumPy() {
        #expect(NumPyMedian.median(of: [3.0, 1, 2]) == 2)                 // np.median([3, 1, 2]) = 2.0
        #expect(NumPyMedian.median(of: [4.0, 1, 3, 2]) == 2.5)            // 2.5
        #expect(NumPyMedian.median(of: [0.1, 0.7]) == (0.1 + 0.7) / 2)    // 0.39999999999999997
        #expect(NumPyMedian.median(of: [1.0, .nan, 2])!.isNaN)
        #expect(NumPyMedian.median(of: [Double]()) == nil)
        #expect(NumPyMedian.median(of: [Float(0.1), 0.7, 5, -1]) == (Float(0.1) + Float(0.7)) / 2)
        #expect(NumPyMedian.median(of: [Float]()) == nil)
    }

    @Test(arguments: [1, 2, 7, 8, 33, 64, 1001])
    func inPlaceAgreesWithArray(_ count: Int) {
        var state = UInt64(count)
        let values = (0..<count).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Double(state >> 11) / 0x1p53 - 0.5
        }
        var copy = values
        let inPlace = copy.withUnsafeMutableBufferPointer { NumPyMedian.medianInPlace($0) }
        #expect(inPlace == NumPyMedian.median(of: values))
        let s = values.sorted()
        #expect(inPlace == (count % 2 == 1 ? s[count / 2] : (s[count / 2 - 1] + s[count / 2]) / 2))
        #expect(NumPyMedian.median(of: values.map(Float.init)) == {
            let f = values.map(Float.init).sorted()
            return count % 2 == 1 ? f[count / 2] : (f[count / 2 - 1] + f[count / 2]) / 2
        }())
    }
}
