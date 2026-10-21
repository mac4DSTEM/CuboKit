import Accelerate
import CuboCore
import Foundation

/// A complex spectrum of a real `rows * columns` image, row-major, in the layout of NumPy's `fft2`
/// (unnormalised forward transform, frequencies in `fftfreq` order).
public struct ParallaxSpectrum: Sendable, Equatable {
    /// Rows of the image this spectrum belongs to.
    public let rows: Int
    /// Columns of the image this spectrum belongs to.
    public let columns: Int
    /// Real parts, `rows * columns` values.
    public var real: [Double]
    /// Imaginary parts, `rows * columns` values.
    public var imaginary: [Double]

    /// Creates a spectrum. Its use checks the lengths and finiteness.
    public init(rows: Int, columns: Int, real: [Double], imaginary: [Double]) {
        self.rows = rows
        self.columns = columns
        self.real = real
        self.imaginary = imaginary
    }
}

/// A reusable Fourier-domain toolbox for one image size: the FFT plan is built once, then any number of
/// images are shifted, transformed or correlated with it.
///
/// ``ParallaxAlignment/shifted(_:rows:columns:by:)`` builds a new plan on every call; this type is its
/// plan-reusing form and returns bit-identical results for the same input. The plan is immutable, the type
/// is `Sendable`, and every method may be called from several threads at once.
public struct ParallaxFourierShifter: Sendable {
    /// Image rows.
    public let rows: Int
    /// Image columns.
    public let columns: Int
    private let fft: FFT2D
    private let singleFFT: FloatFFT2D?

    /// Builds the plan. Dimensions must be at least three on each axis and their product must fit in `Int`.
    public init(rows: Int, columns: Int) throws {
        let (count, overflow) = rows.multipliedReportingOverflow(by: columns)
        guard rows >= 3, columns >= 3, !overflow, count > 0 else { throw ParallaxAlignmentError.invalidDimensions }
        self.rows = rows
        self.columns = columns
        fft = FFT2D(rows: rows, columns: columns)
        singleFFT = FloatFFT2D(rows: rows, columns: columns)
    }

    /// Applies a periodic Fourier translation; identical to ``ParallaxAlignment/shifted(_:rows:columns:by:)``.
    public func shifted(_ image: [Double], by shift: ParallaxShift) throws -> [Double] {
        try ParallaxAlignment.validate(image, rows: rows, columns: columns)
        guard shift.row.isFinite, shift.column.isFinite else { throw ParallaxAlignmentError.nonfiniteValues }
        var transformed = try ParallaxAlignment.spectrum(image, fft: fft)
        ParallaxAlignment.apply(shift, to: &transformed, rows: rows, columns: columns)
        return try ParallaxAlignment.inverse(transformed, fft: fft)
    }

    /// Shifts every image by its own translation with the one plan, in input order.
    ///
    /// `isCancelled` is polled before each image and throws `CancellationError` when it returns true;
    /// `progress` receives `(completed, total)` after each image.
    public func shifted(_ images: [[Double]], by shifts: [ParallaxShift],
                        isCancelled: (() -> Bool)? = nil,
                        progress: ((_ completed: Int, _ total: Int) -> Void)? = nil) throws -> [[Double]] {
        guard images.count == shifts.count else { throw ParallaxAlignmentError.inconsistentShape }
        var out: [[Double]] = []
        out.reserveCapacity(images.count)
        for (index, image) in images.enumerated() {
            try parallaxCheckCancelled(isCancelled)
            out.append(try shifted(image, by: shifts[index]))
            progress?(index + 1, images.count)
        }
        return out
    }

    /// Whether the Float32 shifter is available for this size: vDSP's DFT must take both lengths (`2^n × {1, 3, 5, 15}`).
    public var supportsSingle: Bool { singleFFT != nil }

    /// ``shifted(_:by:)`` in Float32 (named apart so integer-literal arrays keep resolving to the Double form) on vDSP's Float DFT, for pipelines that hold their images in Float: agrees with the Double
    /// shifter to 1e-4 px-equivalent (about 1e-5 relative to the image's range; D12 Float32) at about half the cost. Throws
    /// ``ParallaxAlignmentError/invalidDimensions`` when ``supportsSingle`` is false.
    public func shiftedSingle(_ image: [Float], by shift: ParallaxShift) throws -> [Float] {
        guard let single = singleFFT else { throw ParallaxAlignmentError.invalidDimensions }
        guard image.count == rows * columns else { throw ParallaxAlignmentError.inconsistentShape }
        guard image.allSatisfy(\.isFinite) else { throw ParallaxAlignmentError.nonfiniteValues }
        guard shift.row.isFinite, shift.column.isFinite else { throw ParallaxAlignmentError.nonfiniteValues }
        let n = rows * columns
        let rowShift = shift.row.truncatingRemainder(dividingBy: Double(rows))
        let columnShift = shift.column.truncatingRemainder(dividingBy: Double(columns))
        let inverseCount = 1 / Float(n)
        // the spectrum comes out transposed (columns x rows); the ramp and the 1/n are applied there
        let rowRamp = (0..<rows).map { row -> (Float, Float) in
            let angle = -2 * Double.pi * rowShift * (Double(row < (rows + 1) / 2 ? row : row - rows) / Double(rows))
            return (Float(cos(angle)), Float(sin(angle)))
        }
        let columnRamp = (0..<columns).map { column -> (Float, Float) in
            let angle = -2 * Double.pi * columnShift * (Double(column < (columns + 1) / 2 ? column : column - columns) / Double(columns))
            return (Float(cos(angle)) * inverseCount, Float(sin(angle)) * inverseCount)
        }
        var aRe = image, aIm = [Float](repeating: 0, count: n), bRe = aIm, bIm = aIm
        aRe.withUnsafeMutableBufferPointer { ar in aIm.withUnsafeMutableBufferPointer { ai in
        bRe.withUnsafeMutableBufferPointer { br in bIm.withUnsafeMutableBufferPointer { bi in
            single.forward(aRe: ar.baseAddress!, aIm: ai.baseAddress!, bRe: br.baseAddress!, bIm: bi.baseAddress!)
            for column in 0..<columns {
                let (cc, cs) = columnRamp[column]
                for row in 0..<rows {
                    let (rc, rs) = rowRamp[row]
                    let cosine = rc * cc - rs * cs, sine = rs * cc + rc * cs
                    let k = column * rows + row, re = br[k], im = bi[k]
                    br[k] = re * cosine - im * sine; bi[k] = re * sine + im * cosine
                }
            }
            single.inverse(aRe: br.baseAddress!, aIm: bi.baseAddress!, bRe: ar.baseAddress!, bIm: ai.baseAddress!)
        } } } }
        guard aRe.allSatisfy(\.isFinite) else { throw ParallaxAlignmentError.nonfiniteValues }
        return aRe
    }

    /// Shifts every Float image by its own translation, as the Double batch form does (`isCancelled` before each image,
    /// `progress` after each).
    public func shiftedSingle(_ images: [[Float]], by shifts: [ParallaxShift],
                        isCancelled: (() -> Bool)? = nil,
                        progress: ((_ completed: Int, _ total: Int) -> Void)? = nil) throws -> [[Float]] {
        guard images.count == shifts.count else { throw ParallaxAlignmentError.inconsistentShape }
        var out: [[Float]] = []
        out.reserveCapacity(images.count)
        for (index, image) in images.enumerated() {
            try parallaxCheckCancelled(isCancelled)
            out.append(try shiftedSingle(image, by: shifts[index]))
            progress?(index + 1, images.count)
        }
        return out
    }

    /// The forward transform of a finite real image (`np.fft.fft2`).
    public func spectrum(of image: [Double]) throws -> ParallaxSpectrum {
        try ParallaxAlignment.validate(image, rows: rows, columns: columns)
        let result = try ParallaxAlignment.spectrum(image, fft: fft)
        return ParallaxSpectrum(rows: rows, columns: columns, real: result.real, imaginary: result.imaginary)
    }

    /// The real part of the inverse transform of a spectrum (`np.fft.ifft2(...).real`).
    public func inverse(_ spectrum: ParallaxSpectrum) throws -> [Double] {
        try ParallaxAlignment.inverse(try internalSpectrum(spectrum), fft: fft)
    }

    /// Multiplies a spectrum by the Fourier-shift phase ramp of `shift`, as ``shifted(_:by:)`` does
    /// before its inverse transform.
    public func shifting(_ spectrum: ParallaxSpectrum, by shift: ParallaxShift) throws -> ParallaxSpectrum {
        guard shift.row.isFinite, shift.column.isFinite else { throw ParallaxAlignmentError.nonfiniteValues }
        var inner = try internalSpectrum(spectrum)
        ParallaxAlignment.apply(shift, to: &inner, rows: rows, columns: columns)
        return ParallaxSpectrum(rows: rows, columns: columns, real: inner.real, imaginary: inner.imaginary)
    }

    /// The spectrum-level shift estimate; see ``ParallaxAlignment/correlationShift(reference:moving:upsampleFactor:)``.
    public func correlationShift(reference: ParallaxSpectrum, moving: ParallaxSpectrum,
                                 upsampleFactor: Int = 8) throws -> ParallaxShift {
        guard (1...64).contains(upsampleFactor) else { throw ParallaxAlignmentError.invalidUpsampleFactor }
        let referenceSpectrum = try internalSpectrum(reference), movingSpectrum = try internalSpectrum(moving)
        if upsampleFactor > 2 {
            return try ParallaxAlignment.estimate(reference: referenceSpectrum, moving: movingSpectrum, fft: fft,
                                                  upsampleFactor: upsampleFactor)
        }
        return try ParallaxAlignment.integerEstimate(reference: referenceSpectrum, moving: movingSpectrum, fft: fft)
    }

    private func internalSpectrum(_ spectrum: ParallaxSpectrum) throws -> ParallaxAlignment.Spectrum {
        guard spectrum.rows == rows, spectrum.columns == columns,
              spectrum.real.count == rows * columns, spectrum.imaginary.count == rows * columns
        else { throw ParallaxAlignmentError.inconsistentShape }
        guard spectrum.real.allSatisfy(\.isFinite), spectrum.imaginary.allSatisfy(\.isFinite)
        else { throw ParallaxAlignmentError.nonfiniteValues }
        return ParallaxAlignment.Spectrum(real: spectrum.real, imaginary: spectrum.imaginary)
    }
}

extension ParallaxAlignment {
    /// The spectrum-level shift estimate of the source application: the translation to apply to `moving`
    /// (a spectrum, for instance an average of shifted spectra) to align it to `reference`.
    ///
    /// The cross-power is `reference * conj(moving)` with no phase normalisation. For `upsampleFactor > 2`
    /// the peak is the parabolic estimate of the circular-correlation maximum refined on a `ceil(1.5 * factor)`
    /// matrix-DFT patch with a final parabolic polish (as ``shift(reference:moving:rows:columns:upsampleFactor:)``);
    /// for factors 1 and 2 it is the integer maximum alone. The result is wrapped into `[-size/2, size/2)`.
    /// A flat correlation throws ``ParallaxAlignmentError/degenerateCorrelation``. This builds an FFT plan per
    /// call; use ``ParallaxFourierShifter`` to reuse one.
    public static func correlationShift(reference: ParallaxSpectrum, moving: ParallaxSpectrum,
                                        upsampleFactor: Int = 8) throws -> ParallaxShift {
        try ParallaxFourierShifter(rows: reference.rows, columns: reference.columns)
            .correlationShift(reference: reference, moving: moving, upsampleFactor: upsampleFactor)
    }

    /// The integer-pixel estimate (factors 1 and 2): argmax of the inverse of the cross-power, first row-major
    /// maximum, wrapped.
    package static func integerEstimate(reference: Spectrum, moving: Spectrum, fft: FFT2D) throws -> ParallaxShift {
        var real = Array(repeating: 0.0, count: reference.real.count), imaginary = real
        for pixel in real.indices {
            real[pixel] = reference.real[pixel] * moving.real[pixel] + reference.imaginary[pixel] * moving.imaginary[pixel]
            imaginary[pixel] = reference.imaginary[pixel] * moving.real[pixel] - reference.real[pixel] * moving.imaginary[pixel]
        }
        guard real.allSatisfy(\.isFinite), imaginary.allSatisfy(\.isFinite) else {
            throw ParallaxAlignmentError.nonfiniteValues
        }
        fft.transform(&real, &imaginary, inverse: true)
        guard real.allSatisfy(\.isFinite) else { throw ParallaxAlignmentError.nonfiniteValues }
        guard real.min() != real.max() else { throw ParallaxAlignmentError.degenerateCorrelation }
        var peak = 0
        for index in 1..<real.count where real[index] > real[peak] { peak = index }
        return ParallaxShift(row: wrapped(Double(peak / fft.columns), size: fft.rows),
                             column: wrapped(Double(peak % fft.columns), size: fft.columns))
    }
}
