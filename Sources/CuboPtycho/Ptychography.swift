import Foundation
import CuboCore

/// A finite row-major complex field in Double precision.
public struct PtychographyArray: Sendable {
    /// Grid dimensions and split complex samples.
    public let rows: Int, columns: Int
    /// Real samples, in row-major order.
    public let real: [Double]
    /// Imaginary samples, in row-major order.
    public let imaginary: [Double]
    /// Validates positive dimensions, lengths and finite samples.
    public init(rows: Int, columns: Int, real: [Double], imaginary: [Double]) throws {
        guard rows > 1, columns > 1, rows <= Int.max / columns,
              real.count == rows * columns, imaginary.count == real.count,
              real.allSatisfy(\.isFinite), imaginary.allSatisfy(\.isFinite) else {
            throw PtychographyError.invalidInput
        }
        self.rows = rows; self.columns = columns; self.real = real; self.imaginary = imaginary
    }
}

/// Invalid numerical input or unsupported dimensions.
public enum PtychographyError: Error, Sendable {
    /// Shapes, samples or reconstruction parameters violate their documented domain.
    case invalidInput
}

/// How an engine divides the summed squared amplitude residual of an iteration to make its `errors`.
public enum PtychographyErrorNormalization: Sendable, Equatable {
    /// py4DSTEM: the residual divided by `meanIntensity * positionCount`, where `meanIntensity` is the preprocessing
    /// mean (``PtychographyAmplitudes/meanIntensity``, which counts negative shifted values before they are clipped).
    case meanIntensity(Double)
    /// The residual divided by the energy `sum(amplitudes^2)` of the amplitudes the engine was given. This was the kit's
    /// behaviour before 0.5.0 and is the only choice for amplitudes that did not come from ``PtychographyPreparation``;
    /// it equals `.meanIntensity(m)` when `m * positionCount` is that energy (no negative value was clipped).
    case suppliedAmplitudeEnergy

    /// The divisor, or `nil` when it is not a positive finite number.
    func divisor(amplitudes: [Double], positionCount: Int) -> Double? {
        let d: Double
        switch self {
        case .meanIntensity(let mean): d = mean * Double(positionCount)
        case .suppliedAmplitudeEnergy: d = amplitudes.reduce(0) { $0 + $1 * $1 }
        }
        return d.isFinite && d > 0 ? d : nil
    }
}

/// py4DSTEM's position correction for ``SingleslicePtychography`` (`fix_positions=False`): after each object update every
/// scan position moves by a least-squares step along the gradient of the detector intensity with respect to a one-pixel shift of
/// the object patch (`_position_correction`).
public struct PtychographyPositionCorrection: Sendable, Equatable {
    /// Scale of the update (`positions_step_size`, py4DSTEM default 0.5).
    public var stepSize: Double
    /// Largest move of one position in one iteration, in Å (`max_position_update_distance`); `nil` is unlimited. Needs the samplings.
    public var maxUpdateDistance: Double?
    /// Largest distance of a position from its starting value, in Å (`max_position_total_distance`); an update that would
    /// exceed it is dropped. `nil` is unlimited. Needs the samplings.
    public var maxTotalDistance: Double?
    /// The object sampling in Å per row and per column; py4DSTEM converts the two distances to pixels with
    /// `sqrt(rowSampling^2 + columnSampling^2)`.
    public var rowSampling: Double?, columnSampling: Double?
    /// Keep the centre of mass of the positions at its starting value after every iteration. This is py4DSTEM's
    /// `fix_positions_com=False` (its flag is inverted in use: the default `True` applies no constraint).
    public var constrainCenterOfMass: Bool
    /// Creates the settings; validated when a reconstruction starts.
    public init(stepSize: Double = 0.5, maxUpdateDistance: Double? = nil, maxTotalDistance: Double? = nil,
                rowSampling: Double? = nil, columnSampling: Double? = nil, constrainCenterOfMass: Bool = false) {
        self.stepSize = stepSize; self.maxUpdateDistance = maxUpdateDistance; self.maxTotalDistance = maxTotalDistance
        self.rowSampling = rowSampling; self.columnSampling = columnSampling; self.constrainCenterOfMass = constrainCenterOfMass
    }
    /// The two distances in pixels, validated; `nil` when the settings are invalid.
    fileprivate func pixelDistances() -> (update: Double, total: Double)? {
        guard stepSize.isFinite else { return nil }
        let needsSampling = maxUpdateDistance != nil || maxTotalDistance != nil
        var scale = 1.0
        if needsSampling {
            guard let r = rowSampling, let c = columnSampling, r.isFinite, r > 0, c.isFinite, c > 0 else { return nil }
            scale = (r*r + c*c).squareRoot()
        }
        for d in [maxUpdateDistance, maxTotalDistance] { if let d, !(d.isFinite && d > 0) { return nil } }
        return ((maxUpdateDistance ?? .infinity) / scale, (maxTotalDistance ?? .infinity) / scale)
    }
}

/// A probe center in object pixels; fractional offsets use periodic Fourier translation.
public struct PtychographyPosition: Sendable {
    /// Row and column coordinates.
    public let row: Double, column: Double
    /// Creates a position, validated when reconstruction starts.
    public init(row: Double, column: Double) { self.row = row; self.column = column }
}

/// Probe construction following py4DSTEM ComplexProbe, without coherence envelopes.
public enum PtychographyProbe {
    /// Builds a corner-centered, unit-norm probe. Sampling and wavelength are in Å;
    /// aperture angles are radians. Defocus uses C10 = -defocus; C12 is polar
    /// two-fold astigmatism in Å, its azimuth in radians; C30 is in Å.
    public static func build(rows: Int, columns: Int, rowSampling: Double, columnSampling: Double,
                             wavelength: Double, cutoff: Double, rolloff: Double = 0.002,
                             defocus: Double = 0, astigmatism: Double = 0,
                             astigmatismAngle: Double = 0, sphericalAberration: Double = 0) throws -> PtychographyArray {
        guard rows > 1, columns > 1, rows <= Int.max / columns,
              [rowSampling,columnSampling,wavelength,cutoff].allSatisfy({ $0.isFinite && $0 > 0 }),
              rolloff.isFinite, rolloff >= 0,
              [defocus,astigmatism,astigmatismAngle,sphericalAberration].allSatisfy(\.isFinite) else {
            throw PtychographyError.invalidInput
        }
        var re = [Double](repeating: 0, count: rows * columns), im = re
        for r in 0..<rows { for c in 0..<columns {
            let qr = frequency(r, rows) / rowSampling, qc = frequency(c, columns) / columnSampling
            let alpha = hypot(qr,qc) * wavelength, angle = atan2(qc,qr)
            let aperture: Double
            if rolloff > 0, alpha > cutoff - rolloff {
                aperture = alpha > cutoff ? 0 : 0.5 * (1 + cos(.pi * (alpha-cutoff+rolloff)/rolloff))
            } else { aperture = alpha < cutoff ? 1 : 0 }
            let a2 = alpha * alpha
            let chi = 2 * Double.pi / wavelength * (a2/2 * (-defocus + astigmatism * cos(2*(angle-astigmatismAngle))) + a2*a2/4*sphericalAberration)
            re[r*columns+c] = aperture*cos(chi); im[r*columns+c] = -aperture*sin(chi)
        } }
        FFT2D(rows:rows,columns:columns).transform(&re,&im,inverse:true)
        let norm = sqrt(zip(re,im).reduce(0) { $0 + $1.0*$1.0 + $1.1*$1.1 })
        guard norm > 0, norm.isFinite else { throw PtychographyError.invalidInput }
        return try PtychographyArray(rows:rows,columns:columns,real:re.map{$0/norm},imaginary:im.map{$0/norm})
    }
}

/// A CPU full-batch complex-object gradient-descent reconstruction.
public enum SingleslicePtychography {
    /// Corner-centered probe and object after the last iteration, and normalized
    /// squared amplitude residual measured before each update.
    public struct Result: Sendable {
        /// Reconstructed object.
        public let object: PtychographyArray
        /// Reconstructed corner-centered probe.
        public let probe: PtychographyArray
        /// Per-iteration normalized amplitude errors.
        public let errors: [Double]
        /// The scan positions after the last iteration: those given, unless `positionCorrection` moved them.
        public let positions: [PtychographyPosition]
    }
    /// Reconstructs from corner-centered amplitudes `[position][row][column]`.
    /// Uses periodic object patches, ties-to-even center rounding, fractional probe
    /// shifts, py4DSTEM's normalization and |object| ≤ 1 constraint after each update.
    /// Positions stay fixed. `purePhaseObject` forces unit object amplitude. No preprocessing is implicit.
    ///
    /// Probe constraints, applied after each probe update and only when `fixProbe` is false, in
    /// py4DSTEM's order: `fixProbeCenterOfMass` re-centres the probe intensity centre of mass on
    /// the corner pixel by a periodic Fourier shift (`fix_probe_com`; py4DSTEM defaults it on,
    /// here it is off like every added option); `constrainProbeAmplitude` multiplies the probe by
    /// a smooth top-hat of the given relative radius and width in cycles per pixel and restores
    /// the probe's total intensity (`constrain_probe_amplitude`, `_relative_radius`,
    /// `_relative_width`; the radius lies in 0...0.5 and the width in (0, 0.5]).
    /// `fixProbeAperture` restores the Fourier amplitude of the probe given at the start (`fix_probe_aperture`) and
    /// `probeSupportMask` (corner-centred, row-major, `true` = kept) zeroes the probe outside it
    /// (`probe_real_space_support_mask`); both follow the centre of mass and the aperture precedes the amplitude
    /// constraint, as in py4DSTEM. The object amplitude cap has no switch in py4DSTEM and stays unconditional.
    ///
    /// `errorNormalization` divides each iteration's squared residual: pass `.meanIntensity(m)` with the
    /// ``PtychographyAmplitudes/meanIntensity`` of the preprocessing (``PtychographyAmplitudes/errorNormalization``) to get
    /// py4DSTEM's `errors`; the default divides by the energy of `amplitudes` (they agree when no negative value was clipped).
    ///
    /// `positionCorrection` (off by default, as `fix_positions=True` in py4DSTEM) refines the positions: see
    /// ``PtychographyPositionCorrection``. Each iteration, after the object and probe update and before the object amplitude cap,
    /// every position moves by `stepSize` times the least-squares solution of `d I/dr, d I/dc ~ measured^2 - |F|^2` over the detector,
    /// the derivatives being the intensity change when the object patch is read one row (column) further. An iteration is one
    /// batch of all positions, as py4DSTEM's default. A pattern whose 2 x 2 normal matrix is singular keeps its position
    /// (py4DSTEM raises). The corrected positions are in `Result.positions`.
    public static func reconstruct(amplitudes: [Double], positions: [PtychographyPosition],
                                   object: PtychographyArray, probe: PtychographyArray,
                                   iterations: Int = 8, stepSize: Double = 0.5,
                                   normalizationMinimum: Double = 1, fixProbe: Bool = false,
                                   purePhaseObject: Bool = false,
                                   fixProbeCenterOfMass: Bool = false,
                                   constrainProbeAmplitude: Bool = false,
                                   probeAmplitudeRelativeRadius: Double = 0.5,
                                   probeAmplitudeRelativeWidth: Double = 0.05,
                                   fixProbeAperture: Bool = false,
                                   probeSupportMask: [Bool]? = nil,
                                   errorNormalization: PtychographyErrorNormalization = .suppliedAmplitudeEnergy,
                                   positionCorrection: PtychographyPositionCorrection? = nil) throws -> Result {
        let n = probe.real.count, count = positions.count
        guard count > 0, count <= Int.max / n, amplitudes.count == count*n,
              amplitudes.allSatisfy({ $0.isFinite && $0 >= 0 }),
              iterations >= 0, stepSize.isFinite, stepSize > 0,
              normalizationMinimum.isFinite, (0...1).contains(normalizationMinimum),
              positions.allSatisfy({ $0.row.isFinite && $0.column.isFinite && abs($0.row) < Double(Int.max/4) && abs($0.column) < Double(Int.max/4) }),
              probeAmplitudeRelativeRadius.isFinite, (0...0.5).contains(probeAmplitudeRelativeRadius),
              probeAmplitudeRelativeWidth.isFinite, probeAmplitudeRelativeWidth > 0, probeAmplitudeRelativeWidth <= 0.5,
              object.rows >= probe.rows, object.columns >= probe.columns,
              probeSupportMask == nil || probeSupportMask!.count == n else { throw PtychographyError.invalidInput }
        guard let intensity = errorNormalization.divisor(amplitudes: amplitudes, positionCount: count) else {
            throw PtychographyError.invalidInput
        }
        var distances: (update: Double, total: Double)?
        if let correction = positionCorrection {
            guard let d = correction.pixelDistances() else { throw PtychographyError.invalidInput }
            distances = d
        }
        let fft = FFT2D(rows:probe.rows,columns:probe.columns)
        let initialAperture = fixProbeAperture ? fourierAmplitude(probe.real, probe.imaginary, fft) : []
        var or = object.real, oi = object.imaginary, pr = probe.real, pi = probe.imaginary
        var errors = [Double]()
        var currentPositions = positions
        let initialMean = (positions.reduce(0) { $0 + $1.row } / Double(count), positions.reduce(0) { $0 + $1.column } / Double(count))
        func wrapped(_ v: Int, _ n: Int) -> Int { let m = v % n; return m < 0 ? m+n : m }
        for _ in 0..<iterations {
            var onr = [Double](repeating:0,count:or.count), oni = onr, pn = onr
            var pnr = [Double](repeating:0,count:n), pni = pnr, on = pnr
            var psr = pr, psi = pi
            fft.transform(&psr,&psi,inverse:false)
            var error = 0.0
            for pattern in 0..<count {
                let position = currentPositions[pattern]
                let rr = Int(position.row.rounded(.toNearestOrEven)), cc = Int(position.column.rounded(.toNearestOrEven))
                let dr = position.row-Double(rr), dc = position.column-Double(cc)
                var sr = psr, si = psi
                for r in 0..<probe.rows { for c in 0..<probe.columns {
                    let k = r*probe.columns+c
                    let phase = -2 * Double.pi * (frequency(r,probe.rows)*dr+frequency(c,probe.columns)*dc)
                    let cosine = cos(phase), sine = sin(phase)
                    sr[k] = psr[k]*cosine-psi[k]*sine; si[k] = psr[k]*sine+psi[k]*cosine
                } }
                fft.transform(&sr,&si,inverse:true)
                var indices = [Int](repeating:0,count:n)
                var fr = sr, fi = si
                for r in 0..<probe.rows { for c in 0..<probe.columns {
                    let k = r*probe.columns+c
                    let rowOffset = r < (probe.rows+1)/2 ? r : r-probe.rows
                    let colOffset = c < (probe.columns+1)/2 ? c : c-probe.columns
                    let j = wrapped(rr+rowOffset,object.rows)*object.columns+wrapped(cc+colOffset,object.columns)
                    indices[k] = j
                    fr[k] = sr[k]*or[j]-si[k]*oi[j]; fi[k] = sr[k]*oi[j]+si[k]*or[j]
                } }
                fft.transform(&fr,&fi,inverse:false)
                for k in 0..<n {
                    let magnitude = hypot(fr[k],fi[k]), measured = amplitudes[pattern*n+k]
                    error += (measured-magnitude)*(measured-magnitude)
                    let angle = atan2(fi[k],fr[k])
                    fr[k] = measured*cos(angle)-fr[k]; fi[k] = measured*sin(angle)-fi[k]
                }
                fft.transform(&fr,&fi,inverse:true)
                for k in 0..<n {
                    let j = indices[k]
                    onr[j] += sr[k]*fr[k]+si[k]*fi[k]; oni[j] += sr[k]*fi[k]-si[k]*fr[k]
                    pn[j] += sr[k]*sr[k]+si[k]*si[k]
                    pnr[k] += or[j]*fr[k]+oi[j]*fi[k]; pni[k] += or[j]*fi[k]-oi[j]*fr[k]
                    on[k] += or[j]*or[j]+oi[j]*oi[j]
                }
            }
            let maxPN = pn.max()!, maxON = on.max()!, m = normalizationMinimum
            func inverse(_ local: Double, _ maximum: Double) -> Double {
                1/sqrt(1e-16 + pow((1-m)*local,2) + pow(m*maximum,2))
            }
            let objectBefore = positionCorrection == nil ? ([], []) : (or, oi)
            for j in or.indices {
                let s = stepSize * inverse(pn[j],maxPN)
                or[j] += s*onr[j]; oi[j] += s*oni[j]
            }
            if let correction = positionCorrection, let limits = distances {
                // py4DSTEM corrects the positions with the updated object and before the amplitude cap
                currentPositions = try positionUpdate(
                    before: objectBefore, after: (or, oi), objectRows: object.rows, objectColumns: object.columns,
                    probeSpectrum: (psr, psi), rows: probe.rows, columns: probe.columns, fft: fft, amplitudes: amplitudes,
                    positions: currentPositions, initial: positions, stepSize: correction.stepSize, limits: limits)
            }
            for j in or.indices {
                let amplitude = hypot(or[j],oi[j])
                let phase = atan2(oi[j],or[j])
                let a = purePhaseObject ? 1 : min(amplitude,1)
                or[j] = a*cos(phase); oi[j] = a*sin(phase)
            }
            if !fixProbe { for k in 0..<n {
                let s = stepSize * inverse(on[k],maxON)
                pr[k] += s*pnr[k]; pi[k] += s*pni[k]
            }
            if fixProbeCenterOfMass { centerOfMassConstraint(&pr,&pi,fft) }
            if fixProbeAperture { apertureConstraint(&pr,&pi,fft,aperture:initialAperture) }
            if constrainProbeAmplitude {
                amplitudeConstraint(&pr,&pi,rows:probe.rows,columns:probe.columns,
                                    radius:probeAmplitudeRelativeRadius,width:probeAmplitudeRelativeWidth)
            }
            }
            // py4DSTEM does not gate the support mask on fix_probe
            if let mask = probeSupportMask { supportConstraint(&pr,&pi,mask:mask) }
            if let correction = positionCorrection, correction.constrainCenterOfMass {
                let mean = (currentPositions.reduce(0) { $0 + $1.row } / Double(count), currentPositions.reduce(0) { $0 + $1.column } / Double(count))
                currentPositions = currentPositions.map { PtychographyPosition(row: $0.row - (mean.0 - initialMean.0), column: $0.column - (mean.1 - initialMean.1)) }
            }
            errors.append(error/intensity)
        }
        return try Result(object:PtychographyArray(rows:object.rows,columns:object.columns,real:or,imaginary:oi),
                          probe:PtychographyArray(rows:probe.rows,columns:probe.columns,real:pr,imaginary:pi),errors:errors,
                          positions:currentPositions)
    }
}

/// py4DSTEM `_position_correction` for one full batch. `before` is the object the forward model used, `after` the object once its
/// update was added; the probe is the one the forward model used (given by its spectrum). Positions are returned in input order.
private func positionUpdate(before: ([Double], [Double]), after: ([Double], [Double]), objectRows: Int, objectColumns: Int,
                            probeSpectrum: ([Double], [Double]), rows: Int, columns: Int, fft: FFT2D, amplitudes: [Double],
                            positions: [PtychographyPosition], initial: [PtychographyPosition], stepSize: Double,
                            limits: (update: Double, total: Double)) throws -> [PtychographyPosition] {
    let n = rows * columns
    func wrapped(_ v: Int, _ n: Int) -> Int { let m = v % n; return m < 0 ? m+n : m }
    var out = positions
    for pattern in positions.indices {
        let position = positions[pattern]
        let rr = Int(position.row.rounded(.toNearestOrEven)), cc = Int(position.column.rounded(.toNearestOrEven))
        let dr = position.row - Double(rr), dc = position.column - Double(cc)
        var sr = probeSpectrum.0, si = probeSpectrum.1
        for r in 0..<rows { for c in 0..<columns {
            let k = r*columns+c
            let phase = -2 * Double.pi * (frequency(r,rows)*dr + frequency(c,columns)*dc)
            let cosine = cos(phase), sine = sin(phase)
            sr[k] = probeSpectrum.0[k]*cosine - probeSpectrum.1[k]*sine; si[k] = probeSpectrum.0[k]*sine + probeSpectrum.1[k]*cosine
        } }
        fft.transform(&sr, &si, inverse: true)
        /// The far field of the shifted probe times the object patch read `(rowStep, columnStep)` further.
        func farField(_ object: ([Double], [Double]), _ rowStep: Int, _ columnStep: Int) -> ([Double], [Double]) {
            var fr = [Double](repeating: 0, count: n), fi = fr
            for r in 0..<rows {
                let rowOffset = r < (rows+1)/2 ? r : r-rows
                let base = wrapped(rr+rowStep+rowOffset, objectRows) * objectColumns
                for c in 0..<columns {
                    let colOffset = c < (columns+1)/2 ? c : c-columns
                    let j = base + wrapped(cc+columnStep+colOffset, objectColumns), k = r*columns+c
                    fr[k] = sr[k]*object.0[j] - si[k]*object.1[j]; fi[k] = sr[k]*object.1[j] + si[k]*object.0[j]
                }
            }
            fft.transform(&fr, &fi, inverse: false)
            return (fr, fi)
        }
        let f = farField(before, 0, 0), fdx = farField(after, 1, 0), fdy = farField(after, 0, 1)
        var axx = 0.0, axy = 0.0, ayy = 0.0, bx = 0.0, by = 0.0
        for k in 0..<n {
            let measured = amplitudes[pattern*n+k]
            let difference = measured*measured - (f.0[k]*f.0[k] + f.1[k]*f.1[k])
            let dx = 2 * ((f.0[k]-fdx.0[k])*f.0[k] + (f.1[k]-fdx.1[k])*f.1[k])
            let dy = 2 * ((f.0[k]-fdy.0[k])*f.0[k] + (f.1[k]-fdy.1[k])*f.1[k])
            axx += dx*dx; axy += dx*dy; ayy += dy*dy; bx += dx*difference; by += dy*difference
        }
        let determinant = axx*ayy - axy*axy
        var u0 = 0.0, u1 = 0.0
        if determinant != 0, determinant.isFinite {
            u0 = (ayy*bx - axy*by) / determinant * stepSize
            u1 = (axx*by - axy*bx) / determinant * stepSize
        }
        guard u0.isFinite, u1.isFinite else { continue }
        var norm = hypot(u0, u1)
        if norm > limits.update { u0 /= norm/limits.update; u1 /= norm/limits.update; norm = limits.update }
        if limits.total.isFinite {
            let start = initial[pattern]
            if hypot(position.row - start.row - u0, position.column - start.column - u1) > limits.total { u0 = 0; u1 = 0 }
        }
        let row = position.row - u0, column = position.column - u1
        guard abs(row) < Double(Int.max/4), abs(column) < Double(Int.max/4) else { throw PtychographyError.invalidInput }
        out[pattern] = PtychographyPosition(row: row, column: column)
    }
    return out
}

func frequency(_ k: Int, _ n: Int) -> Double { Double(k < (n+1)/2 ? k : k-n)/Double(n) }
