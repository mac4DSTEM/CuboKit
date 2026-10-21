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

/// One probe aberration term beyond defocus, two-fold astigmatism and C30: radial order `n` (the power `alpha^(n+1)/(n+1)`),
/// angular order `m` and a cosine or sine component, in Cartesian form: `chi += 2 pi / wavelength * coefficient *
/// alpha^(n+1)/(n+1) * cos(m phi)` (or `sin(m phi)`), with `phi = atan2(column frequency, row frequency)`. This is py4DSTEM's
/// `ComplexProbe` Cartesian coefficient `C_nm a` / `C_nm b` (`polar_aberrations_to_cartesian`) and the mac4DSTEM app's
/// higher-order term; the sign is not negated (only the defocus is: `C10 = -defocus`). Use ``polar(radialOrder:angularOrder:magnitude:azimuth:)``
/// for py4DSTEM's `C21`/`phi21` style input.
public struct PtychographyAberrationTerm: Sendable, Equatable {
    /// Cosine or sine of `angularOrder * phi`.
    public enum Component: Sendable, Equatable { case cosine, sine }
    /// `n >= 1`: the term goes with `alpha^(n+1)`.
    public var radialOrder: Int
    /// `m >= 0` (`m = 0` is rotationally symmetric and has no sine component).
    public var angularOrder: Int
    public var component: Component
    /// The coefficient in Å.
    public var coefficient: Double
    public init(radialOrder: Int, angularOrder: Int, component: Component = .cosine, coefficient: Double) {
        self.radialOrder = radialOrder; self.angularOrder = angularOrder
        self.component = component; self.coefficient = coefficient
    }
    /// py4DSTEM's polar form `C_nm cos(m (phi - phi_nm))` as the cosine and sine terms it equals
    /// (`C cos(m phi_nm)` and `C sin(m phi_nm)`; one term for `m = 0`, where the azimuth is ignored).
    public static func polar(radialOrder: Int, angularOrder: Int, magnitude: Double, azimuth: Double) -> [PtychographyAberrationTerm] {
        if angularOrder == 0 { return [.init(radialOrder: radialOrder, angularOrder: 0, coefficient: magnitude)] }
        let m = Double(angularOrder)
        return [.init(radialOrder: radialOrder, angularOrder: angularOrder, component: .cosine, coefficient: magnitude * cos(m * azimuth)),
                .init(radialOrder: radialOrder, angularOrder: angularOrder, component: .sine, coefficient: magnitude * sin(m * azimuth))]
    }
    var isValid: Bool {
        coefficient.isFinite && radialOrder >= 1 && radialOrder <= 64 && angularOrder >= 0 && (component == .cosine || angularOrder > 0)
    }
    /// The term's contribution to `chi * wavelength / (2 pi)` (Å).
    func phase(alpha: Double, azimuth: Double) -> Double {
        let radial = pow(alpha, Double(radialOrder + 1)) / Double(radialOrder + 1)
        if angularOrder == 0 { return coefficient * radial }
        let angle = Double(angularOrder) * azimuth
        return coefficient * radial * (component == .cosine ? cos(angle) : sin(angle))
    }
}

/// Probe construction following py4DSTEM ComplexProbe, without coherence envelopes.
public enum PtychographyProbe {
    /// Builds a corner-centered, unit-norm probe. Sampling and wavelength are in Å;
    /// aperture angles are radians. Defocus uses C10 = -defocus; C12 is polar
    /// two-fold astigmatism in Å, its azimuth in radians; C30 is in Å.
    /// `higherOrder` adds terms to chi (see ``PtychographyAberrationTerm``): py4DSTEM's `C21 ... C56` (pinned) and, beyond its
    /// set, any `(n, m)` the mac4DSTEM app can enter (pinned to a transcription of the app's formula). Empty (default): unchanged.
    public static func build(rows: Int, columns: Int, rowSampling: Double, columnSampling: Double,
                             wavelength: Double, cutoff: Double, rolloff: Double = 0.002,
                             defocus: Double = 0, astigmatism: Double = 0,
                             astigmatismAngle: Double = 0, sphericalAberration: Double = 0,
                             higherOrder: [PtychographyAberrationTerm] = []) throws -> PtychographyArray {
        guard rows > 1, columns > 1, rows <= Int.max / columns,
              [rowSampling,columnSampling,wavelength,cutoff].allSatisfy({ $0.isFinite && $0 > 0 }),
              rolloff.isFinite, rolloff >= 0,
              [defocus,astigmatism,astigmatismAngle,sphericalAberration].allSatisfy(\.isFinite),
              higherOrder.allSatisfy(\.isValid) else {
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
            var phase = a2/2 * (-defocus + astigmatism * cos(2*(angle-astigmatismAngle))) + a2*a2/4*sphericalAberration
            for term in higherOrder { phase += term.phase(alpha: alpha, azimuth: angle) }
            let chi = 2 * Double.pi / wavelength * phase
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
    /// constraint, as in py4DSTEM. The object amplitude cap (`|object| <= 1`) has no switch in py4DSTEM; `constrainObjectAmplitude: false` (default true = py4DSTEM)
    /// is the mac4DSTEM app's option (DEVIATION from py4DSTEM, pinned to py4DSTEM with its threshold constraint replaced by the identity):
    /// no cap, and unit amplitude only if `purePhaseObject` (which always wins, as in the app).
    ///
    /// `errorNormalization` divides each iteration's squared residual: pass `.meanIntensity(m)` with the
    /// ``PtychographyAmplitudes/meanIntensity`` of the preprocessing (``PtychographyAmplitudes/errorNormalization``) to get
    /// py4DSTEM's `errors`; the default divides by the energy of `amplitudes` (they agree when no negative value was clipped).
    ///
    /// `positionCorrection` (off by default, as `fix_positions=True` in py4DSTEM) refines the positions: see
    /// ``PtychographyPositionCorrection``. Each iteration, after the object update and before the object amplitude cap,
    /// every position moves by `stepSize` times the least-squares solution of `d I/dr, d I/dc ~ measured^2 - |F|^2` over the detector,
    /// the derivatives being the intensity change when the object patch is read one row (column) further. An iteration is one
    /// batch of all positions, as py4DSTEM's default. A pattern whose 2 x 2 normal matrix is singular keeps its position
    /// (py4DSTEM raises). The corrected positions are in `Result.positions`.
    public static func reconstruct(amplitudes: [Double], positions: [PtychographyPosition],
                                   object: PtychographyArray, probe: PtychographyArray,
                                   iterations: Int = 8, stepSize: Double = 0.5,
                                   normalizationMinimum: Double = 1, fixProbe: Bool = false,
                                   purePhaseObject: Bool = false,
                                   constrainObjectAmplitude: Bool = true,
                                   fixProbeCenterOfMass: Bool = false,
                                   constrainProbeAmplitude: Bool = false,
                                   probeAmplitudeRelativeRadius: Double = 0.5,
                                   probeAmplitudeRelativeWidth: Double = 0.05,
                                   fixProbeAperture: Bool = false,
                                   probeSupportMask: [Bool]? = nil,
                                   errorNormalization: PtychographyErrorNormalization = .suppliedAmplitudeEnergy,
                                   positionCorrection: PtychographyPositionCorrection? = nil,
                                   precision: PtychographyPrecision = .double,
                                   isCancelled: (() -> Bool)? = nil,
                                   progress: ((_ completed: Int, _ total: Int) -> Void)? = nil) throws -> Result {
        if precision == .single {
            guard positionCorrection == nil else { throw PtychographyError.invalidInput }
            guard amplitudes.allSatisfy({ $0.isFinite && $0 >= 0 }) else { throw PtychographyError.invalidInput }
            return try reconstructSingle(
                amplitudes: amplitudes.map { Float($0) }, positions: positions, object: object, probe: probe,
                iterations: iterations, stepSize: stepSize, normalizationMinimum: normalizationMinimum, fixProbe: fixProbe,
                purePhaseObject: purePhaseObject, constrainObjectAmplitude: constrainObjectAmplitude,
                fixProbeCenterOfMass: fixProbeCenterOfMass, constrainProbeAmplitude: constrainProbeAmplitude,
                probeAmplitudeRelativeRadius: probeAmplitudeRelativeRadius, probeAmplitudeRelativeWidth: probeAmplitudeRelativeWidth,
                fixProbeAperture: fixProbeAperture, probeSupportMask: probeSupportMask,
                errorNormalization: errorNormalization, isCancelled: isCancelled, progress: progress)
        }
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
        let rows = probe.rows, columns = probe.columns, objectRows = object.rows, objectColumns = object.columns
        let rowFrequency = (0..<rows).map { frequency($0, rows) }, columnFrequency = (0..<columns).map { frequency($0, columns) }
        var rowCos = [Double](repeating: 0, count: rows), rowSin = rowCos
        var columnCos = [Double](repeating: 0, count: columns), columnSin = columnCos
        var rowBase = [Int](repeating: 0, count: rows), columnIndex = [Int](repeating: 0, count: columns)
        var indices = [Int](repeating: 0, count: n)
        var sr = [Double](repeating: 0, count: n), si = sr, fr = sr, fi = sr
        var scratch = [Double](repeating: 0, count: fft.usesDFT ? 4*n : 1)
        for iteration in 0..<iterations {
            try checkCancelled(isCancelled)
            var onr = [Double](repeating:0,count:or.count), oni = onr, pn = onr
            var pnr = [Double](repeating:0,count:n), pni = pnr, on = pnr
            var psr = pr, psi = pi
            fft.transform(&psr,&psi,inverse:false)
            var error = 0.0
            var cancelled = false
            pn.withUnsafeMutableBufferPointer { pnP in on.withUnsafeMutableBufferPointer { onP in
            onr.withUnsafeMutableBufferPointer { onrP in oni.withUnsafeMutableBufferPointer { oniP in
            pnr.withUnsafeMutableBufferPointer { pnrP in pni.withUnsafeMutableBufferPointer { pniP in
            or.withUnsafeBufferPointer { orP in oi.withUnsafeBufferPointer { oiP in
            amplitudes.withUnsafeBufferPointer { ampP in
            psr.withUnsafeBufferPointer { psrP in psi.withUnsafeBufferPointer { psiP in
            sr.withUnsafeMutableBufferPointer { srP in si.withUnsafeMutableBufferPointer { siP in
            fr.withUnsafeMutableBufferPointer { frP in fi.withUnsafeMutableBufferPointer { fiP in
            scratch.withUnsafeMutableBufferPointer { scP in
                let srp = srP.baseAddress!, sip = siP.baseAddress!, frp = frP.baseAddress!, fip = fiP.baseAddress!
                for pattern in 0..<count {
                    if pattern & 3 == 0, isCancelled?() == true { cancelled = true; break }
                    let position = currentPositions[pattern]
                    let rr = Int(position.row.rounded(.toNearestOrEven)), cc = Int(position.column.rounded(.toNearestOrEven))
                    let dr = position.row-Double(rr), dc = position.column-Double(cc)
                    // the shifted probe: spectrum times exp(-2 pi i (f_r dr + f_c dc)), a product of a row and a column ramp
                    for r in 0..<rows {
                        let a = -2 * Double.pi * rowFrequency[r] * dr
                        rowCos[r] = cos(a); rowSin[r] = sin(a)
                    }
                    for c in 0..<columns {
                        let a = -2 * Double.pi * columnFrequency[c] * dc
                        columnCos[c] = cos(a); columnSin[c] = sin(a)
                    }
                    for r in 0..<rows {
                        let rc = rowCos[r], rs = rowSin[r], base = r*columns
                        for c in 0..<columns {
                            let cosine = rc*columnCos[c]-rs*columnSin[c], sine = rs*columnCos[c]+rc*columnSin[c]
                            let a = psrP[base+c], b = psiP[base+c]
                            srp[base+c] = a*cosine-b*sine; sip[base+c] = a*sine+b*cosine
                        }
                    }
                    fft.transform(real: srp, imaginary: sip, inverse: true, scratch: scP.baseAddress!)
                    for r in 0..<rows {
                        let rowOffset = r < (rows+1)/2 ? r : r-rows
                        rowBase[r] = wrapped(rr+rowOffset,objectRows)*objectColumns
                    }
                    for c in 0..<columns {
                        let colOffset = c < (columns+1)/2 ? c : c-columns
                        columnIndex[c] = wrapped(cc+colOffset,objectColumns)
                    }
                    for r in 0..<rows {
                        let base = r*columns, rowStart = rowBase[r]
                        for c in 0..<columns {
                            let k = base+c, j = rowStart+columnIndex[c]
                            indices[k] = j
                            frp[k] = srp[k]*orP[j]-sip[k]*oiP[j]; fip[k] = srp[k]*oiP[j]+sip[k]*orP[j]
                        }
                    }
                    fft.transform(real: frp, imaginary: fip, inverse: false, scratch: scP.baseAddress!)
                    // Fourier-modulus projection: measured * F/|F| - F (a vanishing F takes phase 0, as atan2(0, 0))
                    for k in 0..<n {
                        let a = frp[k], b = fip[k]
                        let magnitude = (a*a+b*b).squareRoot(), measured = ampP[pattern*n+k]
                        error += (measured-magnitude)*(measured-magnitude)
                        if magnitude > 0 {
                            let scale = measured/magnitude
                            frp[k] = scale*a-a; fip[k] = scale*b-b
                        } else {
                            frp[k] = measured; fip[k] = 0
                        }
                    }
                    fft.transform(real: frp, imaginary: fip, inverse: true, scratch: scP.baseAddress!)
                    for k in 0..<n {
                        let j = indices[k]
                        let sk = srp[k], ik = sip[k], fk = frp[k], gk = fip[k]
                        onrP[j] += sk*fk+ik*gk; oniP[j] += sk*gk-ik*fk
                        pnP[j] += sk*sk+ik*ik
                        pnrP[k] += orP[j]*fk+oiP[j]*gk; pniP[k] += orP[j]*gk-oiP[j]*fk
                        onP[k] += orP[j]*orP[j]+oiP[j]*oiP[j]
                    }
                }
            } } } } } } } } } } } } } } } }
            if cancelled { throw CancellationError() }
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
            if constrainObjectAmplitude || purePhaseObject { for j in or.indices {
                let amplitude = (or[j]*or[j]+oi[j]*oi[j]).squareRoot()
                let a = purePhaseObject ? 1 : min(amplitude,1)
                if amplitude > 0 { let scale = a/amplitude; or[j] *= scale; oi[j] *= scale } else { or[j] = a; oi[j] = 0 }
            } }
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
            progress?(iteration + 1, iterations)
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
