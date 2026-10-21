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
        let replacer = try Replacer(rows:shape.detectorRows,columns:shape.detectorColumns,mask:mask)
        values.withUnsafeMutableBufferPointer { buffer in
            for scan in 0..<shape.scanCount { replacer.apply(to:buffer,base:scan*n) }
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
        guard rows > 0, columns > 0 else { throw HotPixelFilterError.invalidInput }
        output.withUnsafeMutableBufferPointer { buffer in
            withUnsafeTemporaryAllocation(of:Float.self,capacity:9) { window in
                for k in mask.indices where mask[k] {
                    let r = k/columns, c = k%columns
                    var count = 0
                    for rr in max(0,r-1)..<min(rows,r+2) { for cc in max(0,c-1)..<min(columns,c+2) {
                        insert(buffer[rr*columns+cc],into:window,count:&count)
                    } }
                    buffer[k] = median(window,count:count)
                }
            }
        }
        return output
    }

    /// A reusable replacement plan for one mask: the clipped 3×3 neighbour indices of every
    /// masked pixel are computed once, so `apply` allocates nothing per frame. Results are
    /// bit-identical to `replacing`, which sequentially replaces masked pixels in row-major
    /// order (later medians see earlier replacements).
    public struct Replacer: Sendable {
        /// Detector rows.
        public let rows: Int
        /// Detector columns.
        public let columns: Int
        private let targets: [Int]
        private let starts: [Int]
        private let neighbours: [Int]

        /// Builds the plan. Throws `HotPixelFilterError.invalidInput` unless
        /// `mask.count == rows*columns` with positive dimensions.
        public init(rows: Int, columns: Int, mask: [Bool]) throws {
            guard rows > 0, columns > 0, rows <= Int.max/columns, mask.count == rows*columns else {
                throw HotPixelFilterError.invalidInput
            }
            var targets = [Int](), starts = [0], neighbours = [Int]()
            for k in mask.indices where mask[k] {
                let r = k/columns, c = k%columns
                for rr in max(0,r-1)..<min(rows,r+2) { for cc in max(0,c-1)..<min(columns,c+2) {
                    neighbours.append(rr*columns+cc)
                } }
                targets.append(k); starts.append(neighbours.count)
            }
            self.rows = rows; self.columns = columns
            self.targets = targets; self.starts = starts; self.neighbours = neighbours
        }

        /// Replaces the masked pixels of the `rows*columns` frame starting at `base`, in place.
        /// The caller guarantees finite values and that the frame lies inside the buffer.
        public func apply(to pixels: UnsafeMutableBufferPointer<Float>, base: Int = 0) {
            precondition(base >= 0 && base+rows*columns <= pixels.count)
            guard let data = pixels.baseAddress else { return }
            let frame = data+base
            withUnsafeTemporaryAllocation(of:Float.self,capacity:9) { window in
                for t in targets.indices {
                    let lo = starts[t], count = starts[t+1]-lo
                    var n = 0
                    for i in 0..<count { HotPixelFilter.insert(frame[neighbours[lo+i]],into:window,count:&n) }
                    frame[targets[t]] = HotPixelFilter.median(window,count:n)
                }
            }
        }

        /// Array convenience for `apply(to:base:)`; `base` is checked against the array.
        public func apply(to pixels: inout [Float], base: Int = 0) {
            pixels.withUnsafeMutableBufferPointer { apply(to:$0,base:base) }
        }
    }
    @inline(__always)
    private static func insert(_ v: Float, into window: UnsafeMutableBufferPointer<Float>, count: inout Int) {
        var j = count
        while j > 0 && window[j-1] > v { window[j] = window[j-1]; j -= 1 }
        window[j] = v; count += 1
    }
    @inline(__always)
    private static func median(_ window: UnsafeMutableBufferPointer<Float>, count: Int) -> Float {
        let middle = count/2
        // np.median takes the mean of the middle one or two samples;
        // its +0 accumulator canonicalizes input -0 before division.
        return count%2 == 1 ? 0 + window[middle] : (0 + window[middle-1]+window[middle])/2
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
}

/// Invalid shape, nonfinite values, threshold or comparison rank.
public enum HotPixelFilterError: Error, Sendable {
    /// Requires a consistent finite Float32 image, a finite Float32-representable
    /// threshold and a comparison index in 0...20.
    case invalidInput
}
