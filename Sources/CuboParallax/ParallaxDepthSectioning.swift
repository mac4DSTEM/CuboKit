import Foundation
import CuboCore

/// Which contrast-transfer function selects the sign for depth sectioning.
public enum ParallaxDepthCTF: Sendable {
    /// `sin(pi * wavelength * C1 * q^2)`, the only path py4DSTEM 0.14.19 can run
    /// (its `use_CTF_fit=True` raises a `TypeError`).
    case defocusOnly(c1Angstrom: Double)
    /// `sin(chi)` from every fitted term, as the source application evaluates it.
    case fullFit(terms: [ParallaxAberrationTerm], coefficientsAngstrom: [Double])
}

/// A prepared depth-sectioning problem: the sign-CTF filter and the forward transform of every image are
/// computed once, then any number of planes are formed one at a time.
///
/// ``ParallaxDepthSectioning/section(stack:rows:columns:probeAngles:sampling:wavelength:ctf:depthsAngstrom:informationLimit:informationPower:isCancelled:progress:)``
/// is a loop over ``plane(atDepth:)`` and returns the same values. The type is immutable and `Sendable`;
/// ``plane(atDepth:)`` may be called from several threads at once.
public struct ParallaxDepthSectioner: Sendable {
    /// Image rows.
    public let rows: Int
    /// Image columns.
    public let columns: Int
    /// Number of images in the stack.
    public var imageCount: Int { spectra.count }
    private let probeAngles: [ParallaxShift]
    private let sampling: ParallaxShift
    private let sign: [Double], envelope: [Double]
    private let fft: FFT2D
    private let spectra: [([Double], [Double])]

    /// Validates the inputs, builds the sign-CTF and envelope and transforms every image once.
    /// Parameters are those of ``ParallaxDepthSectioning/section(stack:rows:columns:probeAngles:sampling:wavelength:ctf:depthsAngstrom:informationLimit:informationPower:isCancelled:progress:)``.
    public init(stack: [[Double]], rows: Int, columns: Int, probeAngles: [ParallaxShift],
        sampling: ParallaxShift, wavelength: Double, ctf: ParallaxDepthCTF,
        informationLimit: Double? = nil, informationPower: Double = 1,
        isCancelled: (() -> Bool)? = nil) throws {
        guard !stack.isEmpty, stack.count == probeAngles.count,
            sampling.row.isFinite, sampling.column.isFinite,
            sampling.row > 0, sampling.column > 0, wavelength.isFinite, wavelength > 0,
            informationPower.isFinite, informationPower > 0, validCutoff(informationLimit),
            probeAngles.allSatisfy({ $0.row.isFinite && $0.column.isFinite }) else { throw ParallaxError.invalidInput }
        for image in stack { try validateParallaxImage(image, rows: rows, columns: columns) }
        let count = rows * columns
        var sign = [Double](repeating: 0, count: count), envelope = [Double](repeating: 1, count: count)
        for r in 0..<rows { for c in 0..<columns {
            let kr = parallaxFrequency(r, rows) / sampling.row, kc = parallaxFrequency(c, columns) / sampling.column
            let q2 = kr * kr + kc * kc, i = r * columns + c
            var chi: Double
            switch ctf {
            case .defocusOnly(let c1):
                guard c1.isFinite else { throw ParallaxError.invalidInput }
                chi = .pi * wavelength * c1 * q2
            case .fullFit(let terms, let coefficients):
                guard terms.count == coefficients.count, coefficients.allSatisfy(\.isFinite),
                    terms.allSatisfy(validTerm) else { throw ParallaxError.invalidInput }
                let alpha = q2.squareRoot() * wavelength, theta = atan2(kc, kr)
                chi = 0
                for (t, coef) in zip(terms, coefficients) {
                    let angular = t.angularOrder == 0 ? 1 : t.component == 0
                        ? cos(Double(t.angularOrder) * theta) : sin(Double(t.angularOrder) * theta)
                    chi += pow(alpha, Double(t.radialOrder + 1)) / Double(t.radialOrder + 1) * angular * coef
                }
                chi *= 2 * .pi / wavelength
            }
            let s = sin(chi)
            sign[i] = s > 0 ? 1 : s < 0 ? -1 : 0
            if let limit = informationLimit { envelope[i] = 1 / (1 + pow(q2, informationPower) / pow(limit, 2 * informationPower)) }
        } }
        sign[0] = 0
        let fft = FFT2D(rows: rows, columns: columns)
        var spectra: [([Double], [Double])] = []
        for image in stack {
            try parallaxCheckCancelled(isCancelled)
            var re = image, im = [Double](repeating: 0, count: count)
            fft.transform(&re, &im, inverse: false)
            spectra.append((re, im))
        }
        self.rows = rows; self.columns = columns; self.probeAngles = probeAngles; self.sampling = sampling
        self.sign = sign; self.envelope = envelope; self.fft = fft; self.spectra = spectra
    }

    /// The padded real plane at one depth: the mean over images of the sign-CTF-corrected stack after a
    /// Fourier shift of `-angle * depth / sampling` pixels. Only the shift phase and the inverse transform
    /// are computed here; the forward transforms were done in ``init(stack:rows:columns:probeAngles:sampling:wavelength:ctf:informationLimit:informationPower:isCancelled:)``.
    public func plane(atDepth depth: Double, isCancelled: (() -> Bool)? = nil) throws -> [Double] {
        guard depth.isFinite else { throw ParallaxError.invalidInput }
        let count = rows * columns
        var mean = [Double](repeating: 0, count: count)
        for (b, spectrum) in spectra.enumerated() {
            try parallaxCheckCancelled(isCancelled)
            let sr = -probeAngles[b].row * depth / sampling.row, sc = -probeAngles[b].column * depth / sampling.column
            var re = spectrum.0, im = spectrum.1
            for r in 0..<rows { for c in 0..<columns {
                let i = r * columns + c
                let phase = -2 * Double.pi * (parallaxFrequency(r, rows) * sr + parallaxFrequency(c, columns) * sc)
                let m = sign[i] * envelope[i], cs = cos(phase) * m, sn = sin(phase) * m
                let a = re[i], bb = im[i]
                re[i] = a * cs - bb * sn; im[i] = a * sn + bb * cs
            } }
            fft.transform(&re, &im, inverse: true)
            for i in 0..<count { mean[i] += re[i] }
        }
        let n = Double(spectra.count)
        let plane = mean.map { $0 / n }
        guard plane.allSatisfy(\.isFinite) else { throw ParallaxError.nonfiniteValues }
        return plane
    }
}

/// Depth sectioning of an aligned bright-field stack (py4DSTEM `Parallax.depth_section`).
public enum ParallaxDepthSectioning: Sendable {
    /// Returns one padded real plane per depth, each the mean over images of the
    /// sign-CTF-corrected stack after a Fourier shift of `-angle * depth / sampling` pixels.
    ///
    /// The forward transforms are computed once for all depths (see ``ParallaxDepthSectioner`` to form planes
    /// one at a time). `isCancelled` is polled before every image of every plane and throws
    /// `CancellationError` when it returns true; `progress` receives `(completed, total)` planes.
    ///
    /// - Parameters:
    ///   - stack: aligned images, each `rows * columns`, row-major.
    ///   - probeAngles: per-image probe angles in radians (row, column).
    ///   - sampling: scan sampling in angstroms per pixel (row, column).
    ///   - informationLimit: optional Butterworth-style envelope `1 / (1 + q^(2p) / limit^(2p))`.
    public static func section(stack: [[Double]], rows: Int, columns: Int, probeAngles: [ParallaxShift],
        sampling: ParallaxShift, wavelength: Double, ctf: ParallaxDepthCTF, depthsAngstrom: [Double],
        informationLimit: Double? = nil, informationPower: Double = 1,
        isCancelled: (() -> Bool)? = nil,
        progress: ((_ completed: Int, _ total: Int) -> Void)? = nil) throws -> [[Double]] {
        guard !depthsAngstrom.isEmpty, depthsAngstrom.allSatisfy(\.isFinite) else { throw ParallaxError.invalidInput }
        let sectioner = try ParallaxDepthSectioner(stack: stack, rows: rows, columns: columns, probeAngles: probeAngles,
            sampling: sampling, wavelength: wavelength, ctf: ctf, informationLimit: informationLimit,
            informationPower: informationPower, isCancelled: isCancelled)
        var planes: [[Double]] = []
        planes.reserveCapacity(depthsAngstrom.count)
        for (index, depth) in depthsAngstrom.enumerated() {
            try parallaxCheckCancelled(isCancelled)
            planes.append(try sectioner.plane(atDepth: depth, isCancelled: isCancelled))
            progress?(index + 1, depthsAngstrom.count)
        }
        return planes
    }
}
