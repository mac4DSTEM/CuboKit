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

    /// Builds the plan. Dimensions must be at least three on each axis and their product must fit in `Int`.
    public init(rows: Int, columns: Int) throws {
        let (count, overflow) = rows.multipliedReportingOverflow(by: columns)
        guard rows >= 3, columns >= 3, !overflow, count > 0 else { throw ParallaxAlignmentError.invalidDimensions }
        self.rows = rows
        self.columns = columns
        fft = FFT2D(rows: rows, columns: columns)
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
