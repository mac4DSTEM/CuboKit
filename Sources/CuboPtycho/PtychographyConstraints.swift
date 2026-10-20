import Foundation
import CuboCore

// Probe constraints shared by the single-slice, mixed-state and multislice engines (py4DSTEM
// `ProbeConstraintsMixin` / `ProbeMixedConstraintsMixin`), all in Double.

/// Corner-centred intensity centre of mass of a probe, in pixels (py4DSTEM `get_CoM(corner_centered=True)`);
/// `nil` for a probe without intensity.
func probeCenterOfMass(_ re: [Double], _ im: [Double], rows: Int, columns: Int) -> (row: Double, column: Double)? {
    var total = 0.0, comRow = 0.0, comColumn = 0.0
    for r in 0..<rows { for c in 0..<columns {
        let k = r*columns+c, w = re[k]*re[k]+im[k]*im[k]
        total += w
        comRow += frequency(r,rows)*Double(rows)*w; comColumn += frequency(c,columns)*Double(columns)*w
    } }
    guard total > 0, total.isFinite else { return nil }
    return (comRow/total, comColumn/total)
}

/// Periodic Fourier translation by (`row`, `column`) pixels (py4DSTEM `fft_shift`): `out[x] = in[x - shift]`.
func fourierShift(_ re: inout [Double], _ im: inout [Double], _ fft: FFT2D, row: Double, column: Double) {
    let rows = fft.rows, columns = fft.columns
    fft.transform(&re,&im,inverse:false)
    for r in 0..<rows { for c in 0..<columns {
        let k = r*columns+c
        let phase = -2 * Double.pi * (frequency(r,rows)*row+frequency(c,columns)*column)
        let cosine = cos(phase), sine = sin(phase), a = re[k], b = im[k]
        re[k] = a*cosine-b*sine; im[k] = a*sine+b*cosine
    } }
    fft.transform(&re,&im,inverse:true)
}

/// py4DSTEM `_probe_center_of_mass_constraint`: corner-centred intensity CoM, then a Fourier shift by its negative.
func centerOfMassConstraint(_ re: inout [Double], _ im: inout [Double], _ fft: FFT2D) {
    guard let com = probeCenterOfMass(re, im, rows: fft.rows, columns: fft.columns) else { return }
    fourierShift(&re, &im, fft, row: -com.row, column: -com.column)
}

/// py4DSTEM `_probe_amplitude_constraint`: smooth top-hat in cycles per pixel, total intensity restored.
func amplitudeConstraint(_ re: inout [Double], _ im: inout [Double], rows: Int, columns: Int,
                         radius: Double, width: Double) {
    let before = zip(re,im).reduce(0) { $0 + $1.0*$1.0 + $1.1*$1.1 }
    let sigma = Double.pi.squareRoot()/width
    var after = 0.0
    for r in 0..<rows { for c in 0..<columns {
        let k = r*columns+c
        let d = hypot(frequency(r,rows),frequency(c,columns)) - radius
        let mask = 0.5*(1-erf(sigma*d/(1-d*d)))
        re[k] *= mask; im[k] *= mask
        after += re[k]*re[k]+im[k]*im[k]
    } }
    guard before > 0, after > 0 else { return }
    let scale = (before/after).squareRoot()
    for k in re.indices { re[k] *= scale; im[k] *= scale }
}

/// py4DSTEM `_probe_aperture_constraint`: keep the probe's Fourier phase, restore the initial Fourier amplitude
/// (`aperture`, row-major, `|fft2(initial probe)|`), then the original total intensity.
func apertureConstraint(_ re: inout [Double], _ im: inout [Double], _ fft: FFT2D, aperture: [Double]) {
    let before = zip(re,im).reduce(0) { $0 + $1.0*$1.0 + $1.1*$1.1 }
    fft.transform(&re, &im, inverse: false)
    for k in re.indices {
        let phase = atan2(im[k], re[k])
        re[k] = aperture[k]*cos(phase); im[k] = aperture[k]*sin(phase)
    }
    fft.transform(&re, &im, inverse: true)
    let after = zip(re,im).reduce(0) { $0 + $1.0*$1.0 + $1.1*$1.1 }
    guard before > 0, after > 0 else { return }
    let scale = (before/after).squareRoot()
    for k in re.indices { re[k] *= scale; im[k] *= scale }
}

/// py4DSTEM `_probe_real_space_support_constraint`: zero outside the corner-centred boolean `mask`.
func supportConstraint(_ re: inout [Double], _ im: inout [Double], mask: [Bool]) {
    for k in re.indices where !mask[k] { re[k] = 0; im[k] = 0 }
}

/// The Fourier amplitude of a probe, row-major (py4DSTEM `_probe_initial_aperture`).
func fourierAmplitude(_ re: [Double], _ im: [Double], _ fft: FFT2D) -> [Double] {
    var r = re, i = im
    fft.transform(&r, &i, inverse: false)
    return zip(r, i).map { hypot($0, $1) }
}

/// Positions finite and small enough to round to an integer pixel.
func positionsAreUsable(_ positions: [PtychographyPosition]) -> Bool {
    positions.allSatisfy { $0.row.isFinite && $0.column.isFinite && abs($0.row) < Double(Int.max/4) && abs($0.column) < Double(Int.max/4) }
}

func probeConstraintParametersAreValid(radius: Double, width: Double) -> Bool {
    radius.isFinite && (0...0.5).contains(radius) && width.isFinite && width > 0 && width <= 0.5
}
