import CuboCore
import Foundation

/// A periodic translation in image pixels. Positive values move content toward larger indices.
public struct ParallaxShift: Sendable, Equatable {
    /// Translation along the first (row) axis, in pixels.
    public let row: Double
    /// Translation along the second (column) axis, in pixels.
    public let column: Double

    /// Creates a translation. Alignment and shifting reject nonfinite components.
    public init(row: Double, column: Double) {
        self.row = row
        self.column = column
    }
}

/// Aligned images and the translations applied to them, in input order.
public struct ParallaxAlignmentResult: Sendable {
    /// One translation per input image, wrapped to `[-size/2, size/2)` on each axis.
    public let shifts: [ParallaxShift]
    /// Periodically Fourier-shifted images, each stored in row-major order.
    public let images: [[Double]]
    /// Pixelwise arithmetic mean of the aligned images.
    public let mean: [Double]
}

/// Invalid input or an undefined correlation peak.
public enum ParallaxAlignmentError: Error, Sendable, Equatable {
    /// Dimensions must be at least three on each axis and their product must fit in `Int`.
    case invalidDimensions
    /// Image lengths must equal the product of the dimensions; a stack must be nonempty.
    case inconsistentShape
    /// This CPU slice supports matrix-DFT upsampling factors from 3 through 64, inclusive.
    case invalidUpsampleFactor
    /// An input, translation, Fourier product or output is nonfinite.
    case nonfiniteValues
    /// A constant image or a flat correlation has no identifiable translation.
    case degenerateCorrelation
}

/// CPU, Double-precision cross-correlation alignment from py4DSTEM's parallax path.
///
/// Uses the unnormalised cross-power spectrum `FFT(reference) * conj(FFT(moving))`,
/// a half-pixel parabolic estimate, matrix-DFT upsampling and parabolic polishing.
/// It does not perform detector grouping, shift-basis fitting or aberration reconstruction.
public enum ParallaxAlignment: Sendable {
    /// Estimates the periodic translation to apply to `moving` to align it to `reference`.
    ///
    /// Inputs are finite, nonconstant row-major images with identical dimensions (at least 3 × 3).
    /// The default factor 8 matches the parallax alignment slice. Output components are in
    /// `[-rows/2, rows/2)` and `[-columns/2, columns/2)`. Equal maxima keep the first row-major
    /// sample. Constant images and nonfinite arithmetic throw instead of returning NaN.
    public static func shift(reference: [Double], moving: [Double], rows: Int, columns: Int,
                             upsampleFactor: Int = 8) throws -> ParallaxShift {
        try validateFactor(upsampleFactor)
        try validate(reference, rows: rows, columns: columns, requireContrast: true)
        try validate(moving, rows: rows, columns: columns, requireContrast: true)
        let fft = FFT2D(rows: rows, columns: columns)
        return try estimate(reference: spectrum(reference, fft: fft), moving: spectrum(moving, fft: fft),
                            fft: fft, upsampleFactor: upsampleFactor)
    }

    /// Applies a periodic Fourier translation, matching `get_shifted_ar(periodic: true, bilinear: false)`.
    ///
    /// Returns the real part of the inverse transform, including on even-sized axes with a
    /// Nyquist component. Consequently, a fractional translation followed by its inverse
    /// need not recover the Nyquist component of a real image.
    public static func shifted(_ image: [Double], rows: Int, columns: Int,
                               by shift: ParallaxShift) throws -> [Double] {
        try validate(image, rows: rows, columns: columns)
        guard shift.row.isFinite, shift.column.isFinite else { throw ParallaxAlignmentError.nonfiniteValues }
        let fft = FFT2D(rows: rows, columns: columns)
        var transformed = try spectrum(image, fft: fft)
        apply(shift, to: &transformed, rows: rows, columns: columns)
        return try inverse(transformed, fft: fft)
    }

    /// Aligns a nonempty stack against a supplied reference and returns the images and their mean.
    ///
    /// With `runningAverage: false`, every image uses the supplied reference. With `true`,
    /// follows py4DSTEM's running Fourier reference: after the first image it replaces the
    /// supplied reference with that shifted image, then averages subsequent shifted spectra.
    /// The reference retains its complex Nyquist components until the next correlation.
    ///
    /// `isCancelled` is polled before each image and throws `CancellationError` when it returns true;
    /// `progress` receives `(completed, total)` after each image. Both default to nothing.
    public static func align(_ images: [[Double]], reference: [Double], rows: Int, columns: Int,
                             upsampleFactor: Int = 8, runningAverage: Bool = false,
                             isCancelled: (() -> Bool)? = nil,
                             progress: ((_ completed: Int, _ total: Int) -> Void)? = nil) throws -> ParallaxAlignmentResult {
        try validateFactor(upsampleFactor)
        try validate(reference, rows: rows, columns: columns, requireContrast: true)
        guard !images.isEmpty else { throw ParallaxAlignmentError.inconsistentShape }
        for image in images { try validate(image, rows: rows, columns: columns, requireContrast: true) }
        let fft = FFT2D(rows: rows, columns: columns)
        var referenceSpectrum = try spectrum(reference, fft: fft)
        var shifts: [ParallaxShift] = []
        var aligned: [[Double]] = []
        var mean = Array(repeating: 0.0, count: reference.count)
        for (index, image) in images.enumerated() {
            try parallaxCheckCancelled(isCancelled)
            var moving = try spectrum(image, fft: fft)
            let shift = try estimate(reference: referenceSpectrum, moving: moving,
                                     fft: fft, upsampleFactor: upsampleFactor)
            apply(shift, to: &moving, rows: rows, columns: columns)
            let shifted = try inverse(moving, fft: fft)
            shifts.append(shift)
            aligned.append(shifted)
            for pixel in mean.indices { mean[pixel] += shifted[pixel] / Double(images.count) }
            if runningAverage {
                let count = Double(index + 1)
                for pixel in referenceSpectrum.real.indices {
                    referenceSpectrum.real[pixel] = referenceSpectrum.real[pixel] * Double(index) / count
                        + moving.real[pixel] / count
                    referenceSpectrum.imaginary[pixel] = referenceSpectrum.imaginary[pixel] * Double(index) / count
                        + moving.imaginary[pixel] / count
                }
            }
            progress?(index + 1, images.count)
        }
        guard mean.allSatisfy(\.isFinite) else { throw ParallaxAlignmentError.nonfiniteValues }
        return ParallaxAlignmentResult(shifts: shifts, images: aligned, mean: mean)
    }

    package struct Spectrum {
        var real: [Double]
        var imaginary: [Double]
    }

    package static func validateFactor(_ factor: Int) throws {
        guard (3...64).contains(factor) else { throw ParallaxAlignmentError.invalidUpsampleFactor }
    }

    package static func validate(_ image: [Double], rows: Int, columns: Int,
                                 requireContrast: Bool = false) throws {
        let (count, overflow) = rows.multipliedReportingOverflow(by: columns)
        guard rows >= 3, columns >= 3, !overflow else { throw ParallaxAlignmentError.invalidDimensions }
        guard image.count == count else { throw ParallaxAlignmentError.inconsistentShape }
        guard image.allSatisfy(\.isFinite) else { throw ParallaxAlignmentError.nonfiniteValues }
        if requireContrast, image.min() == image.max() { throw ParallaxAlignmentError.degenerateCorrelation }
    }

    package static func spectrum(_ image: [Double], fft: FFT2D) throws -> Spectrum {
        var real = image, imaginary = Array(repeating: 0.0, count: image.count)
        fft.transform(&real, &imaginary, inverse: false)
        guard real.allSatisfy(\.isFinite), imaginary.allSatisfy(\.isFinite) else {
            throw ParallaxAlignmentError.nonfiniteValues
        }
        return Spectrum(real: real, imaginary: imaginary)
    }

    package static func inverse(_ spectrum: Spectrum, fft: FFT2D) throws -> [Double] {
        var real = spectrum.real, imaginary = spectrum.imaginary
        fft.transform(&real, &imaginary, inverse: true)
        guard real.allSatisfy(\.isFinite), imaginary.allSatisfy(\.isFinite) else {
            throw ParallaxAlignmentError.nonfiniteValues
        }
        return real
    }

    package static func estimate(reference: Spectrum, moving: Spectrum, fft: FFT2D,
                                 upsampleFactor: Int) throws -> ParallaxShift {
        var real = Array(repeating: 0.0, count: reference.real.count), imaginary = real
        for pixel in real.indices {
            real[pixel] = reference.real[pixel] * moving.real[pixel]
                + reference.imaginary[pixel] * moving.imaginary[pixel]
            imaginary[pixel] = reference.imaginary[pixel] * moving.real[pixel]
                - reference.real[pixel] * moving.imaginary[pixel]
        }
        guard real.allSatisfy(\.isFinite), imaginary.allSatisfy(\.isFinite) else {
            throw ParallaxAlignmentError.nonfiniteValues
        }
        let peak = try MatrixDFTCorrelation.refinedPeak(correlationRe: real, correlationIm: imaginary,
            width: fft.columns, height: fft.rows, fft: fft, upsampleFactor: upsampleFactor)
        guard peak.row.isFinite, peak.column.isFinite else { throw ParallaxAlignmentError.nonfiniteValues }
        return ParallaxShift(row: wrapped(peak.row, size: fft.rows), column: wrapped(peak.column, size: fft.columns))
    }

    package static func wrapped(_ value: Double, size: Int) -> Double {
        let period = Double(size), half = period / 2
        let remainder = (value + half).truncatingRemainder(dividingBy: period)
        return (remainder < 0 ? remainder + period : remainder) - half
    }

    package static func apply(_ shift: ParallaxShift, to spectrum: inout Spectrum, rows: Int, columns: Int) {
        // Periodic reduction also keeps finite, very large user-supplied shifts from overflowing the phase.
        let rowShift = shift.row.truncatingRemainder(dividingBy: Double(rows))
        let columnShift = shift.column.truncatingRemainder(dividingBy: Double(columns))
        for row in 0..<rows {
            let rowFrequency = Double(row < (rows + 1) / 2 ? row : row - rows) / Double(rows)
            for column in 0..<columns {
                let columnFrequency = Double(column < (columns + 1) / 2 ? column : column - columns) / Double(columns)
                let angle = -2 * Double.pi * (rowShift * rowFrequency + columnShift * columnFrequency)
                let cosine = cos(angle), sine = sin(angle)
                let pixel = row * columns + column
                let real = spectrum.real[pixel], imaginary = spectrum.imaginary[pixel]
                spectrum.real[pixel] = real * cosine - imaginary * sine
                spectrum.imaginary[pixel] = real * sine + imaginary * cosine
            }
        }
    }
}
