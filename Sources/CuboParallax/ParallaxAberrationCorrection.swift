import Foundation
import CuboCore

/// Handling of the high-pass argument, which the pinned Python implementation ignores.
public enum ParallaxHighpassBehavior: Sendable {
    /// Match the pinned reference: accept but ignore the high-pass cutoff.
    case reference
    /// Apply the Butterworth high-pass envelope used in the source application.
    case apply
}

/// Corrected padded image, the CTF phase surfaces and the transfer function used to form it.
public struct ParallaxCorrectionResult: Sendable {
    /// Real inverse FFT of the corrected spectrum, row-major.
    public let image: [Double]
    /// Even radial phase surface in radians.
    public let chiEven: [Double]
    /// Odd radial phase surface in radians.
    public let chiOdd: [Double]
    /// Real part of the transfer function multiplied into the image spectrum,
    /// `sign(sin(chiEven)) * cos(chiOdd) * envelope`, with DC set to zero.
    public let transferReal: [Double]
    /// Imaginary part of the transfer function, `-sign(sin(chiEven)) * sin(chiOdd) * envelope`.
    public let transferImaginary: [Double]
}

/// Phase correction using Cartesian aberration coefficients in angstroms.
public enum ParallaxAberrationCorrection: Sendable {
    /// Applies `sign(sin(chiEven)) * exp(-i*chiOdd)`, suppressing DC.
    ///
    /// An empty even-order surface becomes one, matching Python. Use `defocusOnly`
    /// to select C1 instead of the full fit. Sampling is angstroms per image pixel.
    /// High-pass filtering is ignored by default as in the pinned reference; `.apply`
    /// preserves the source application's filter behavior.
    public static func correct(_ image: [Double], rows: Int, columns: Int,
        sampling: ParallaxShift, wavelength: Double, terms: [ParallaxAberrationTerm],
        coefficients: [Double], defocusOnly: Double? = nil, lowpass: Double? = nil,
        highpass: Double? = nil, butterworthOrder: Int = 2,
        highpassBehavior: ParallaxHighpassBehavior = .reference
    ) throws -> ParallaxCorrectionResult {
        try validateParallaxImage(image, rows: rows, columns: columns)
        guard sampling.row.isFinite, sampling.column.isFinite, sampling.row > 0, sampling.column > 0,
            wavelength.isFinite, wavelength > 0, terms.count == coefficients.count,
            coefficients.allSatisfy(\.isFinite), defocusOnly == nil || defocusOnly!.isFinite,
            validCutoff(lowpass), validCutoff(highpass), (1...100).contains(butterworthOrder),
            terms.allSatisfy(validTerm) else { throw ParallaxError.invalidInput }
        var even = [Double](repeating: 0,count: image.count), odd = even
        var frequencies = even
        for r in 0..<rows { for c in 0..<columns {
            let kr = parallaxFrequency(r, rows) / sampling.row
            let kc = parallaxFrequency(c, columns) / sampling.column
            let i = r * columns + c, q = hypot(kr,kc)
            frequencies[i] = q
            if let defocusOnly { even[i] = .pi * wavelength * defocusOnly * q * q }
            else {
                let alpha = q * wavelength, theta = atan2(kc,kr)
                for (t,coef) in zip(terms,coefficients) {
                    let angular = t.angularOrder == 0 ? 1 : t.component == 0 ? cos(Double(t.angularOrder)*theta) : sin(Double(t.angularOrder)*theta)
                    let phase = pow(alpha,Double(t.radialOrder+1)) / Double(t.radialOrder+1) * angular * coef * 2 * .pi / wavelength
                    if t.radialOrder % 2 == 1 { even[i] += phase } else { odd[i] += phase }
                }
            }
        } }
        if defocusOnly == nil, !even.contains(where: {$0 != 0}) { even = [Double](repeating: 1,count: image.count) }
        var tr = even, ti = odd
        for i in image.indices {
            let s = sin(even[i]); let sign = s > 0 ? 1.0 : s < 0 ? -1.0 : 0
            var envelope = 1.0
            if let lowpass { envelope /= 1 + pow(frequencies[i] / lowpass, Double(2*butterworthOrder)) }
            if let highpass, highpassBehavior == .apply { envelope *= 1 - 1/(1+pow(frequencies[i]/highpass,Double(2*butterworthOrder))) }
            tr[i] = sign * cos(odd[i]) * envelope; ti[i] = -sign * sin(odd[i]) * envelope
        }
        tr[0] = 0; ti[0] = 0
        let corrected = try parallaxTransfer(image, rows: rows, columns: columns, real: tr, imaginary: ti)
        guard even.allSatisfy(\.isFinite), odd.allSatisfy(\.isFinite) else { throw ParallaxError.nonfiniteValues }
        return ParallaxCorrectionResult(image: corrected,chiEven: even,chiOdd: odd,transferReal: tr,transferImaginary: ti)
    }
}

func validCutoff(_ value: Double?) -> Bool { value == nil || (value!.isFinite && value! > 0) }
func validTerm(_ t: ParallaxAberrationTerm) -> Bool {
    (1...8).contains(t.radialOrder) && (0...t.radialOrder+1).contains(t.angularOrder)
        && (t.radialOrder+t.angularOrder)%2 == 1 && (0...1).contains(t.component)
        && (t.angularOrder > 0 || t.component == 0)
}
func parallaxFrequency(_ index: Int, _ count: Int) -> Double {
    Double(index < (count+1)/2 ? index : index-count) / Double(count)
}
func validateParallaxImage(_ image: [Double], rows: Int, columns: Int) throws {
    let (count,overflow) = rows.multipliedReportingOverflow(by: columns)
    guard rows >= 3, columns >= 3, !overflow, image.count == count else { throw ParallaxError.invalidInput }
    guard image.allSatisfy(\.isFinite) else { throw ParallaxError.nonfiniteValues }
}
func parallaxTransfer(_ image: [Double], rows: Int, columns: Int, real: [Double], imaginary: [Double]) throws -> [Double] {
    let fft = FFT2D(rows: rows,columns: columns)
    var re = image, im = [Double](repeating:0,count:image.count)
    fft.transform(&re,&im,inverse:false)
    for i in re.indices {
        let old = re[i];re[i] = old*real[i]-im[i]*imaginary[i];im[i] = old*imaginary[i]+im[i]*real[i]
    }
    fft.transform(&re,&im,inverse:true)
    guard re.allSatisfy(\.isFinite), im.allSatisfy(\.isFinite) else { throw ParallaxError.nonfiniteValues }
    return re
}
