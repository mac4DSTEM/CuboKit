import Foundation

/// py4DSTEM's hot-pixel classification and sequential median replacement on Float32 cubes.
public enum HotPixelFilter {
    /// The cleaned cube, scan-mean pattern and detector mask (row-major).
    public struct Result: Sendable {
        /// Filtered Float32 data in the input cube's storage order.
        public let cube: MemoryCube
        /// Float32 scan mean: NumPy's strided sequential sum, or pairwise for a 1×1 detector.
        public let mean: [Float]
        /// True where the detector pixel exceeded its comparison value by the threshold.
        public let mask: [Bool]
    }
    /// Filters an eager Float32 cube. Classifies the mean against the `comparisonIndex`-th
    /// brightest of 21 wrapped detector neighbors (default 1 = second brightest).
    /// In every frame, replaces masked pixels in row-major order by their clipped 3×3
    /// median, in place: later medians see earlier replacements.
    /// The Float32 accumulation is intentional: a Double mean can change threshold masks.
    public static func filter(_ cube: MemoryCube, threshold: Double,
                              comparisonIndex: Int = 1) throws -> Result {
        let shape = cube.shape, n = shape.detectorCount
        guard cube.values.allSatisfy(\.isFinite) else { throw HotPixelFilterError.invalidInput }
        var mean = [Float](repeating:0,count:n)
        // NumPy reduces contiguous scan values pairwise only when the detector
        // contains one pixel; all other detector patterns have a strided reduction.
        if n == 1 {
            mean[0] = 0 + pairwise(cube.values,start:0,count:shape.scanCount)
        } else {
            for scan in 0..<shape.scanCount {
                for k in 0..<n { mean[k] += cube.values[scan*n+k] }
            }
        }
        let denominator = Float(shape.scanCount)
        for k in 0..<n { mean[k] /= denominator }
        let mask = try mask(mean:mean,rows:shape.detectorRows,columns:shape.detectorColumns,
                            threshold:threshold,comparisonIndex:comparisonIndex)
        var values = cube.values
        for scan in 0..<shape.scanCount {
            let base = scan*n
            replaceUnchecked(&values,base:base,rows:shape.detectorRows,columns:shape.detectorColumns,mask:mask)
        }
        return try Result(cube:MemoryCube(shape:shape,values:values),mean:mean,mask:mask)
    }

    /// Classifies a supplied Float32 mean pattern with the same wrapped 21-pixel footprint.
    /// `comparisonIndex` lies in 0...20. The threshold is compared in Float32 arithmetic.
    public static func mask(mean: [Float], rows: Int, columns: Int, threshold: Double,
                            comparisonIndex: Int = 1) throws -> [Bool] {
        try validate(mean,rows:rows,columns:columns)
        guard threshold.isFinite, Float(threshold).isFinite, (0...20).contains(comparisonIndex) else {
            throw HotPixelFilterError.invalidInput
        }
        var offsets = [(Int,Int)]()
        for r in -1...1 { for c in -1...1 { offsets.append((r,c)) } }
        for r in -1...1 { offsets.append((r,-2)); offsets.append((r,2)) }
        for c in -1...1 { offsets.append((-2,c)); offsets.append((2,c)) }
        var output = [Bool](repeating:false,count:mean.count)
        var window = [Float](repeating:0,count:21)
        func wrapped(_ i: Int, _ n: Int) -> Int { let v = i % n; return v < 0 ? v+n : v }
        for r in 0..<rows { for c in 0..<columns {
            for (j,offset) in offsets.enumerated() {
                window[j] = mean[wrapped(r+offset.0,rows)*columns+wrapped(c+offset.1,columns)]
            }
            window.sort()
            output[r*columns+c] = mean[r*columns+c]-window[20-comparisonIndex] > Float(threshold)
        } }
        return output
    }

    /// Replaces selected pixels in one frame, sequentially in row-major mask order.
    /// The median window is clipped at detector boundaries. The input frame is unchanged.
    public static func replacing(_ frame: [Float], rows: Int, columns: Int, mask: [Bool]) throws -> [Float] {
        try validate(frame,rows:rows,columns:columns)
        guard mask.count == frame.count else { throw HotPixelFilterError.invalidInput }
        var output = frame
        replaceUnchecked(&output,base:0,rows:rows,columns:columns,mask:mask)
        return output
    }
    // NumPy's contiguous Float32 fast-axis pairwise reduction (block size 128).
    private static func pairwise(_ values: [Float], start: Int, count: Int) -> Float {
        if count < 8 {
            var sum: Float = -0.0
            for i in 0..<count { sum += values[start+i] }
            return sum
        }
        if count <= 128 {
            var partial = Array(values[start..<start+8])
            let limit = count-count%8
            for i in stride(from:8,to:limit,by:8) {
                for j in 0..<8 { partial[j] += values[start+i+j] }
            }
            var sum = ((partial[0]+partial[1])+(partial[2]+partial[3]))
                    + ((partial[4]+partial[5])+(partial[6]+partial[7]))
            for i in limit..<count { sum += values[start+i] }
            return sum
        }
        let half = (count/2)/8*8
        return pairwise(values,start:start,count:half)+pairwise(values,start:start+half,count:count-half)
    }

    private static func validate(_ values: [Float], rows: Int, columns: Int) throws {
        guard rows > 0, columns > 0, rows <= Int.max/columns,
              values.count == rows*columns, values.allSatisfy(\.isFinite) else {
            throw HotPixelFilterError.invalidInput
        }
    }

    private static func replaceUnchecked(_ values: inout [Float], base: Int, rows: Int, columns: Int, mask: [Bool]) {
        var window = [Float](); window.reserveCapacity(9)
        for k in mask.indices where mask[k] {
            let r = k/columns, c = k%columns
            window.removeAll(keepingCapacity:true)
            for rr in max(0,r-1)..<min(rows,r+2) { for cc in max(0,c-1)..<min(columns,c+2) {
                window.append(values[base+rr*columns+cc])
            } }
            window.sort()
            let middle = window.count/2
            // np.median takes the mean of the middle one or two samples;
            // its +0 accumulator canonicalizes input -0 before division.
            values[base+k] = window.count%2 == 1 ? 0 + window[middle] : (0 + window[middle-1]+window[middle])/2
        }
    }

}

/// Invalid shape, nonfinite values, threshold or comparison rank.
public enum HotPixelFilterError: Error, Sendable {
    /// Requires a consistent finite Float32 image, a finite Float32-representable
    /// threshold and a comparison index in 0...20.
    case invalidInput
}
