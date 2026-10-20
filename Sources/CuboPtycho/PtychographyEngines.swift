import Foundation
import CuboCore

/// A finite stack of equally sized complex grids: probe states or object slices, `[index][row][column]`, Double.
public struct PtychographyStack: Sendable {
    /// Number of grids, and the rows and columns of each.
    public let count: Int, rows: Int, columns: Int
    /// Real samples, `[index][row][column]`.
    public let real: [Double]
    /// Imaginary samples, `[index][row][column]`.
    public let imaginary: [Double]
    /// Validates positive dimensions, lengths and finite samples.
    public init(count: Int, rows: Int, columns: Int, real: [Double], imaginary: [Double]) throws {
        guard count > 0, rows > 1, columns > 1, rows <= Int.max / columns, count <= Int.max / (rows * columns),
              real.count == count * rows * columns, imaginary.count == real.count,
              real.allSatisfy(\.isFinite), imaginary.allSatisfy(\.isFinite) else { throw PtychographyError.invalidInput }
        self.count = count; self.rows = rows; self.columns = columns; self.real = real; self.imaginary = imaginary
    }
    /// Stacks grids of one shape.
    public init(_ arrays: [PtychographyArray]) throws {
        guard let first = arrays.first, arrays.allSatisfy({ $0.rows == first.rows && $0.columns == first.columns }) else {
            throw PtychographyError.invalidInput
        }
        try self.init(count: arrays.count, rows: first.rows, columns: first.columns,
                      real: arrays.flatMap(\.real), imaginary: arrays.flatMap(\.imaginary))
    }
    /// The grid at `index`.
    public func array(at index: Int) throws -> PtychographyArray {
        guard (0..<count).contains(index) else { throw PtychographyError.invalidInput }
        let n = rows * columns
        return try PtychographyArray(rows: rows, columns: columns, real: Array(real[index*n..<(index+1)*n]),
                                     imaginary: Array(imaginary[index*n..<(index+1)*n]))
    }
}

// MARK: - shared kernels

/// Geometry shared by the engines: a probe-sized frame on a larger periodic object.
private struct Frame {
    let rows: Int, columns: Int, objectRows: Int, objectColumns: Int
    let fft: FFT2D
    var n: Int { rows * columns }
    init(rows: Int, columns: Int, objectRows: Int, objectColumns: Int) {
        self.rows = rows; self.columns = columns; self.objectRows = objectRows; self.objectColumns = objectColumns
        fft = FFT2D(rows: rows, columns: columns)
    }
    static func wrapped(_ v: Int, _ n: Int) -> Int { let m = v % n; return m < 0 ? m+n : m }
    /// Nearest-integer centre (ties to even) and the remaining fractional shift.
    func centre(_ p: PtychographyPosition) -> (rr: Int, cc: Int, dr: Double, dc: Double) {
        let rr = Int(p.row.rounded(.toNearestOrEven)), cc = Int(p.column.rounded(.toNearestOrEven))
        return (rr, cc, p.row - Double(rr), p.column - Double(cc))
    }
    /// Object index of every probe pixel for a patch centred at (`rr`, `cc`), periodic.
    func indices(_ rr: Int, _ cc: Int) -> [Int] {
        var out = [Int](repeating: 0, count: n)
        for r in 0..<rows {
            let rowOffset = r < (rows+1)/2 ? r : r-rows
            let base = Self.wrapped(rr+rowOffset, objectRows) * objectColumns
            for c in 0..<columns {
                let colOffset = c < (columns+1)/2 ? c : c-columns
                out[r*columns+c] = base + Self.wrapped(cc+colOffset, objectColumns)
            }
        }
        return out
    }
    /// The probe (given by its spectrum) shifted by the fractional (`dr`, `dc`) pixels, in real space.
    func shifted(_ spectrumRe: [Double], _ spectrumIm: [Double], _ dr: Double, _ dc: Double) -> (re: [Double], im: [Double]) {
        var sr = spectrumRe, si = spectrumIm
        for r in 0..<rows { for c in 0..<columns {
            let k = r*columns+c
            let phase = -2 * Double.pi * (frequency(r,rows)*dr+frequency(c,columns)*dc)
            let cosine = cos(phase), sine = sin(phase)
            sr[k] = spectrumRe[k]*cosine-spectrumIm[k]*sine; si[k] = spectrumRe[k]*sine+spectrumIm[k]*cosine
        } }
        fft.transform(&sr,&si,inverse:true)
        return (sr, si)
    }
}

/// py4DSTEM's update normalisation `1 / sqrt(1e-16 + ((1-m) local)^2 + (m max)^2)`.
private func updateNormalization(_ local: Double, _ maximum: Double, _ m: Double) -> Double {
    1 / sqrt(1e-16 + pow((1-m)*local,2) + pow(m*maximum,2))
}

private func commonInputsAreValid(amplitudes: [Double], positions: [PtychographyPosition], probeCount n: Int,
                                  iterations: Int, stepSize: Double, normalizationMinimum: Double,
                                  radius: Double, width: Double) -> Bool {
    let count = positions.count
    return count > 0 && n > 0 && count <= Int.max / n && amplitudes.count == count*n
        && amplitudes.allSatisfy({ $0.isFinite && $0 >= 0 })
        && iterations >= 0 && stepSize.isFinite && stepSize > 0
        && normalizationMinimum.isFinite && (0...1).contains(normalizationMinimum)
        && positionsAreUsable(positions) && probeConstraintParametersAreValid(radius: radius, width: width)
}

/// The object update of one iteration: step times the accumulated gradient, normalised.
private func applyObjectUpdate(_ re: inout [Double], _ im: inout [Double], _ gradRe: [Double], _ gradIm: [Double],
                               _ norm: [Double], step: Double, m: Double) {
    let maximum = norm.max()!
    for j in re.indices {
        let s = step * updateNormalization(norm[j], maximum, m)
        re[j] += s*gradRe[j]; im[j] += s*gradIm[j]
    }
}

/// py4DSTEM `_object_threshold_constraint`: amplitude capped at 1, or set to 1 for a pure-phase object.
private func threshold(_ re: inout Double, _ im: inout Double, pure: Bool) {
    let amplitude = hypot(re, im), phase = atan2(im, re)
    let a = pure ? 1 : min(amplitude, 1)
    re = a*cos(phase); im = a*sin(phase)
}

// MARK: - mixed state

/// CPU full-batch mixed-state ptychography after py4DSTEM's `MixedstatePtychography`: several mutually incoherent probe
/// states share one complex object; the detector intensity is the sum over states. Fixed positions, Double.
public enum MixedstatePtychography {
    /// Object, probe states (ordered by decreasing intensity when orthogonalised) and per-iteration normalised errors.
    public struct Result: Sendable {
        /// Reconstructed object.
        public let object: PtychographyArray
        /// Reconstructed corner-centred probe states.
        public let probes: PtychographyStack
        /// Per-iteration normalised squared amplitude residual measured before each update.
        public let errors: [Double]
    }

    /// Reconstructs from corner-centred amplitudes `[position][row][column]`.
    ///
    /// The update is py4DSTEM's gradient descent: far-field amplitude `sqrt(sum_states |F_s|^2)` replaced by the measured one
    /// for every state at once, object and probe updates normalised by the summed state intensity. After each iteration, and
    /// only when the probes are not fixed, in py4DSTEM's order: `fixProbeCenterOfMass` shifts all states by the centre of
    /// mass of state 0; `constrainProbeAmplitude` applies the top-hat to each state; `orthogonalizeProbes` replaces the states
    /// by the eigenvectors of their Gram matrix, sorted by decreasing intensity (also when `fixProbe` is set, as in
    /// py4DSTEM; pass `orthogonalizeProbes: false` to keep fixed states untouched). The overall phase of each orthogonalised
    /// state is an eigenvector convention (LAPACK's differs from this one's) and does not affect the object, the errors or any
    /// intensity. An exactly zero far-field pixel contributes `measured^2` to the error and no update (py4DSTEM divides
    /// by infinity there).
    public static func reconstruct(amplitudes: [Double], positions: [PtychographyPosition],
                                   object: PtychographyArray, probes: PtychographyStack,
                                   iterations: Int = 8, stepSize: Double = 0.5, normalizationMinimum: Double = 1,
                                   fixProbe: Bool = false, purePhaseObject: Bool = false,
                                   fixProbeCenterOfMass: Bool = false, constrainProbeAmplitude: Bool = false,
                                   probeAmplitudeRelativeRadius: Double = 0.5, probeAmplitudeRelativeWidth: Double = 0.05,
                                   orthogonalizeProbes: Bool = true) throws -> Result {
        let n = probes.rows * probes.columns, count = positions.count, states = probes.count
        guard commonInputsAreValid(amplitudes: amplitudes, positions: positions, probeCount: n, iterations: iterations,
                                   stepSize: stepSize, normalizationMinimum: normalizationMinimum,
                                   radius: probeAmplitudeRelativeRadius, width: probeAmplitudeRelativeWidth),
              object.rows >= probes.rows, object.columns >= probes.columns else { throw PtychographyError.invalidInput }
        let intensity = amplitudes.reduce(0) { $0+$1*$1 }
        guard intensity > 0, intensity.isFinite else { throw PtychographyError.invalidInput }
        let frame = Frame(rows: probes.rows, columns: probes.columns, objectRows: object.rows, objectColumns: object.columns)
        var or = object.real, oi = object.imaginary, pr = probes.real, pi = probes.imaginary
        var errors = [Double]()
        let m = normalizationMinimum
        for _ in 0..<iterations {
            var onr = [Double](repeating: 0, count: or.count), oni = onr, pn = onr
            var pnr = [Double](repeating: 0, count: states*n), pni = pnr, on = [Double](repeating: 0, count: n)
            var spectraRe = [[Double]](), spectraIm = [[Double]]()
            for s in 0..<states {
                var re = Array(pr[s*n..<(s+1)*n]), im = Array(pi[s*n..<(s+1)*n])
                frame.fft.transform(&re, &im, inverse: false)
                spectraRe.append(re); spectraIm.append(im)
            }
            var error = 0.0
            for pattern in 0..<count {
                let centre = frame.centre(positions[pattern])
                let indices = frame.indices(centre.rr, centre.cc)
                var sr = [[Double]](), si = [[Double]](), fr = [[Double]](), fi = [[Double]]()
                var farField = [Double](repeating: 0, count: n)
                for s in 0..<states {
                    let shifted = frame.shifted(spectraRe[s], spectraIm[s], centre.dr, centre.dc)
                    var ar = [Double](repeating: 0, count: n), ai = ar
                    for k in 0..<n {
                        let j = indices[k]
                        ar[k] = shifted.re[k]*or[j]-shifted.im[k]*oi[j]; ai[k] = shifted.re[k]*oi[j]+shifted.im[k]*or[j]
                    }
                    frame.fft.transform(&ar, &ai, inverse: false)
                    for k in 0..<n { farField[k] += ar[k]*ar[k]+ai[k]*ai[k] }
                    sr.append(shifted.re); si.append(shifted.im); fr.append(ar); fi.append(ai)
                }
                var factor = [Double](repeating: 0, count: n)
                for k in 0..<n {
                    let amplitude = farField[k].squareRoot(), measured = amplitudes[pattern*n+k]
                    if amplitude == 0 { error += measured*measured; factor[k] = 0 }
                    else { error += (measured-amplitude)*(measured-amplitude); factor[k] = measured/amplitude - 1 }
                }
                for s in 0..<states {
                    var er = fr[s], ei = fi[s]
                    for k in 0..<n { er[k] *= factor[k]; ei[k] *= factor[k] }
                    frame.fft.transform(&er, &ei, inverse: true)
                    for k in 0..<n {
                        let j = indices[k]
                        onr[j] += sr[s][k]*er[k]+si[s][k]*ei[k]; oni[j] += sr[s][k]*ei[k]-si[s][k]*er[k]
                        pn[j] += sr[s][k]*sr[s][k]+si[s][k]*si[s][k]
                        pnr[s*n+k] += or[j]*er[k]+oi[j]*ei[k]; pni[s*n+k] += or[j]*ei[k]-oi[j]*er[k]
                    }
                }
                for k in 0..<n { let j = indices[k]; on[k] += or[j]*or[j]+oi[j]*oi[j] }
            }
            applyObjectUpdate(&or, &oi, onr, oni, pn, step: stepSize, m: m)
            for j in or.indices { threshold(&or[j], &oi[j], pure: purePhaseObject) }
            if !fixProbe {
                let maximum = on.max()!
                for s in 0..<states { for k in 0..<n {
                    let w = stepSize * updateNormalization(on[k], maximum, m)
                    pr[s*n+k] += w*pnr[s*n+k]; pi[s*n+k] += w*pni[s*n+k]
                } }
                if fixProbeCenterOfMass,
                   let com = probeCenterOfMass(Array(pr[0..<n]), Array(pi[0..<n]), rows: probes.rows, columns: probes.columns) {
                    for s in 0..<states {
                        var re = Array(pr[s*n..<(s+1)*n]), im = Array(pi[s*n..<(s+1)*n])
                        fourierShift(&re, &im, frame.fft, row: -com.row, column: -com.column)
                        pr.replaceSubrange(s*n..<(s+1)*n, with: re); pi.replaceSubrange(s*n..<(s+1)*n, with: im)
                    }
                }
                if constrainProbeAmplitude { for s in 0..<states {
                    var re = Array(pr[s*n..<(s+1)*n]), im = Array(pi[s*n..<(s+1)*n])
                    amplitudeConstraint(&re, &im, rows: probes.rows, columns: probes.columns,
                                        radius: probeAmplitudeRelativeRadius, width: probeAmplitudeRelativeWidth)
                    pr.replaceSubrange(s*n..<(s+1)*n, with: re); pi.replaceSubrange(s*n..<(s+1)*n, with: im)
                } }
            }
            // py4DSTEM does not gate the orthogonalisation on fix_probe: a fixed probe stack is still rotated into
            // its orthogonal, intensity-ordered basis.
            if orthogonalizeProbes && states > 1 { (pr, pi) = orthogonalized(pr, pi, states: states, n: n) }
            errors.append(error/intensity)
        }
        return try Result(object: PtychographyArray(rows: object.rows, columns: object.columns, real: or, imaginary: oi),
                          probes: PtychographyStack(count: states, rows: probes.rows, columns: probes.columns, real: pr, imaginary: pi),
                          errors: errors)
    }

    /// py4DSTEM `_probe_orthogonalization_constraint`: the states become the eigenvectors of their Gram matrix
    /// `G[i][j] = sum conj(p_i) p_j`, `new_i = sum_j V[j][i] p_j`, ordered by decreasing intensity.
    public static func orthogonalize(_ probes: PtychographyStack) throws -> PtychographyStack {
        let (re, im) = orthogonalized(probes.real, probes.imaginary, states: probes.count, n: probes.rows * probes.columns)
        return try PtychographyStack(count: probes.count, rows: probes.rows, columns: probes.columns, real: re, imaginary: im)
    }

    /// The starting states of py4DSTEM's `_initialize_probe` for `count` probes: state 0 is `first`; each further
    /// state is the previous one times a random linear phase ramp per axis, `exp(-2 pi i (u - 1/2) fftfreq(n))`, with `u`
    /// drawn row ramp first from NumPy's legacy `RandomState(seed)` (`np.random.seed(seed)`; `np.random.rand()`).
    /// The caller scales `first` to the mean diffraction intensity (see ``PtychographyPreparation``).
    public static func initialProbes(from first: PtychographyArray, count: Int, seed: UInt32) throws -> PtychographyStack {
        guard count > 0, count < 4096 else { throw PtychographyError.invalidInput }
        let n = first.rows * first.columns
        var re = first.real, im = first.imaginary
        var generator = NumPyRandomState(seed: seed)
        var cur = first
        for _ in 1..<max(count, 1) {
            let u = generator.random(), v = generator.random()
            var nr = [Double](repeating: 0, count: n), ni = nr
            for r in 0..<first.rows {
                let pr = -2 * Double.pi * (u - 0.5) * frequency(r, first.rows)
                for c in 0..<first.columns {
                    let phase = pr - 2 * Double.pi * (v - 0.5) * frequency(c, first.columns)
                    let k = r*first.columns+c
                    let a = cos(phase), b = sin(phase)
                    nr[k] = cur.real[k]*a - cur.imaginary[k]*b; ni[k] = cur.real[k]*b + cur.imaginary[k]*a
                }
            }
            re += nr; im += ni
            cur = try PtychographyArray(rows: first.rows, columns: first.columns, real: nr, imaginary: ni)
        }
        return try PtychographyStack(count: count, rows: first.rows, columns: first.columns, real: re, imaginary: im)
    }
}

private func orthogonalized(_ pr: [Double], _ pi: [Double], states: Int, n: Int) -> ([Double], [Double]) {
    var gr = [[Double]](repeating: [Double](repeating: 0, count: states), count: states), gi = gr
    for i in 0..<states { for j in i..<states {
        var sr = 0.0, si = 0.0
        for k in 0..<n {
            let a = pr[i*n+k], b = pi[i*n+k], c = pr[j*n+k], d = pi[j*n+k]
            sr += a*c + b*d; si += a*d - b*c            // conj(p_i) p_j
        }
        gr[i][j] = sr; gi[i][j] = si
        gr[j][i] = sr; gi[j][i] = -si
    } }
    for i in 0..<states { gi[i][i] = 0 }
    let (values, vr, vi) = hermitianEigen(gr, gi)
    var outR = [Double](repeating: 0, count: states*n), outI = outR
    // new_i = sum_j V[j][i] p_j
    for i in 0..<states { for j in 0..<states {
        let a = vr[j][i], b = vi[j][i]
        for k in 0..<n {
            outR[i*n+k] += a*pr[j*n+k] - b*pi[j*n+k]; outI[i*n+k] += a*pi[j*n+k] + b*pr[j*n+k]
        }
    } }
    // intensities are the eigenvalues; sort descending, stable for equal values (NumPy: argsort ascending then reversed)
    let order = (0..<states).sorted { values[$0] != values[$1] ? values[$0] > values[$1] : $0 > $1 }
    var sortedR = outR, sortedI = outI
    for (target, source) in order.enumerated() {
        sortedR.replaceSubrange(target*n..<(target+1)*n, with: outR[source*n..<(source+1)*n])
        sortedI.replaceSubrange(target*n..<(target+1)*n, with: outI[source*n..<(source+1)*n])
    }
    return (sortedR, sortedI)
}

/// Eigenvalues and eigenvectors (columns) of a small Hermitian matrix by cyclic complex Jacobi rotations.
private func hermitianEigen(_ ar: [[Double]], _ ai: [[Double]]) -> (values: [Double], vr: [[Double]], vi: [[Double]]) {
    let n = ar.count
    var a = ar, b = ai
    var vr = (0..<n).map { i in (0..<n).map { $0 == i ? 1.0 : 0.0 } }, vi = ar.map { _ in [Double](repeating: 0, count: n) }
    var scale = 0.0
    for i in 0..<n { for j in 0..<n { scale += a[i][j]*a[i][j] + b[i][j]*b[i][j] } }
    scale = scale.squareRoot()
    for _ in 0..<100 {
        var off = 0.0
        for i in 0..<n { for j in 0..<n where i != j { off += a[i][j]*a[i][j] + b[i][j]*b[i][j] } }
        if off.squareRoot() <= 1e-15 * scale || scale == 0 { break }
        for p in 0..<n-1 { for q in p+1..<n {
            let r = hypot(a[p][q], b[p][q])
            if r <= 1e-300 { continue }
            let phi = atan2(b[p][q], a[p][q])
            let theta = (a[q][q] - a[p][p]) / (2 * r)
            let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + (theta*theta + 1).squareRoot())
            let c = 1 / (t*t + 1).squareRoot(), s = t * c
            // U = D P with P the real rotation and D = diag(1, e^{-i phi}):
            // U[p][p] = c, U[p][q] = s, U[q][p] = -s e^{-i phi}, U[q][q] = c e^{-i phi}
            let upp = (c, 0.0), upq = (s, 0.0)
            let uqp = (-s * cos(phi), s * sin(phi)), uqq = (c * cos(phi), -c * sin(phi))
            func mul(_ x: (Double, Double), _ y: (Double, Double)) -> (Double, Double) { (x.0*y.0 - x.1*y.1, x.0*y.1 + x.1*y.0) }
            // columns update  X <- X U  (for A and V)
            func rightMultiply(_ xr: inout [[Double]], _ xi: inout [[Double]]) {
                for row in 0..<n {
                    let xp = (xr[row][p], xi[row][p]), xq = (xr[row][q], xi[row][q])
                    let np = mul(xp, upp), np2 = mul(xq, uqp)
                    let nq = mul(xp, upq), nq2 = mul(xq, uqq)
                    xr[row][p] = np.0 + np2.0; xi[row][p] = np.1 + np2.1
                    xr[row][q] = nq.0 + nq2.0; xi[row][q] = nq.1 + nq2.1
                }
            }
            rightMultiply(&a, &b)
            rightMultiply(&vr, &vi)
            // rows update  A <- U^H A : new row p = conj(Upp) A[p] + conj(Uqp) A[q]; row q = conj(Upq) A[p] + conj(Uqq) A[q]
            let cpp = (upp.0, -upp.1), cqp = (uqp.0, -uqp.1), cpq = (upq.0, -upq.1), cqq = (uqq.0, -uqq.1)
            for col in 0..<n {
                let xp = (a[p][col], b[p][col]), xq = (a[q][col], b[q][col])
                let rp1 = mul(cpp, xp), rp2 = mul(cqp, xq), rq1 = mul(cpq, xp), rq2 = mul(cqq, xq)
                a[p][col] = rp1.0 + rp2.0; b[p][col] = rp1.1 + rp2.1
                a[q][col] = rq1.0 + rq2.0; b[q][col] = rq1.1 + rq2.1
            }
            a[p][q] = 0; b[p][q] = 0; a[q][p] = 0; b[q][p] = 0
        } }
    }
    return ((0..<n).map { a[$0][$0] }, vr, vi)
}

/// NumPy's legacy `RandomState`: MT19937 seeded as `np.random.seed(int)` and `random_sample()` of 53 bits.
struct NumPyRandomState {
    private var state = [UInt32](repeating: 0, count: 624)
    private var index = 624
    init(seed: UInt32) {
        state[0] = seed
        for i in 1..<624 { state[i] = 1812433253 &* (state[i-1] ^ (state[i-1] >> 30)) &+ UInt32(i) }
    }
    private mutating func generate() {
        for i in 0..<624 {
            let y = (state[i] & 0x80000000) | (state[(i+1) % 624] & 0x7fffffff)
            var next = state[(i+397) % 624] ^ (y >> 1)
            if y & 1 != 0 { next ^= 0x9908b0df }
            state[i] = next
        }
        index = 0
    }
    private mutating func next32() -> UInt32 {
        if index >= 624 { generate() }
        var y = state[index]; index += 1
        y ^= y >> 11; y ^= (y << 7) & 0x9d2c5680; y ^= (y << 15) & 0xefc60000; y ^= y >> 18
        return y
    }
    mutating func random() -> Double {
        let a = Double(next32() >> 5), b = Double(next32() >> 6)
        return (a * 67108864.0 + b) / 9007199254740992.0
    }
}

// MARK: - multislice

/// CPU full-batch multislice ptychography after py4DSTEM's `MultislicePtychography`: a stack of thin complex object
/// slices separated by Fresnel propagation, one probe, fixed positions, Double.
public enum MultislicePtychography {
    /// Object slices, probe and per-iteration normalised errors.
    public struct Result: Sendable {
        /// Reconstructed object slices, entrance first.
        public let object: PtychographyStack
        /// Reconstructed corner-centred probe.
        public let probe: PtychographyArray
        /// Per-iteration normalised squared amplitude residual measured before each update.
        public let errors: [Double]
    }

    /// Fresnel propagators between consecutive slices (py4DSTEM `_precompute_propagator_arrays`):
    /// `exp(-i pi lambda dz (kx^2 + ky^2))` with `k = fftfreq(n, d)`; one per entry of `sliceThicknesses` (Å).
    public static func propagators(rows: Int, columns: Int, rowSampling: Double, columnSampling: Double, wavelength: Double,
                                   sliceThicknesses: [Double]) throws -> PtychographyStack {
        guard rows > 1, columns > 1, rowSampling.isFinite, rowSampling > 0, columnSampling.isFinite, columnSampling > 0,
              wavelength.isFinite, wavelength > 0, !sliceThicknesses.isEmpty, sliceThicknesses.allSatisfy(\.isFinite) else {
            throw PtychographyError.invalidInput
        }
        var re = [Double](), im = [Double]()
        for dz in sliceThicknesses {
            for r in 0..<rows { for c in 0..<columns {
                let kr = frequency(r, rows) / rowSampling, kc = frequency(c, columns) / columnSampling
                let phase = -Double.pi * wavelength * dz * (kr*kr + kc*kc)
                re.append(cos(phase)); im.append(sin(phase))
            } }
        }
        return try PtychographyStack(count: sliceThicknesses.count, rows: rows, columns: columns, real: re, imaginary: im)
    }

    /// Reconstructs from corner-centred amplitudes `[position][row][column]` through `object.count` slices;
    /// `sliceThicknesses` has `object.count - 1` entries (Å), `wavelength` and the samplings are in Å.
    ///
    /// Gradient descent as py4DSTEM: forward through the slices with a propagation between each, amplitude replacement at
    /// the detector, then back through them. Each slice is updated with its own probe-intensity normalisation, the probe
    /// with slice 0's object intensity. After each iteration: `identicalSlices` replaces every slice by their mean
    /// (`identical_slices`), then the amplitude cap or unit amplitude (`purePhaseObject`) applies to all slices; then the
    /// probe constraints as in ``SingleslicePtychography``. Tilted propagators, the z-regularisation filter and TV denoising
    /// are absent.
    public static func reconstruct(amplitudes: [Double], positions: [PtychographyPosition],
                                   object: PtychographyStack, probe: PtychographyArray, sliceThicknesses: [Double],
                                   wavelength: Double, rowSampling: Double, columnSampling: Double,
                                   iterations: Int = 8, stepSize: Double = 0.5, normalizationMinimum: Double = 1,
                                   fixProbe: Bool = false, purePhaseObject: Bool = false, identicalSlices: Bool = false,
                                   fixProbeCenterOfMass: Bool = false, constrainProbeAmplitude: Bool = false,
                                   probeAmplitudeRelativeRadius: Double = 0.5,
                                   probeAmplitudeRelativeWidth: Double = 0.05) throws -> Result {
        let n = probe.real.count, count = positions.count, slices = object.count, plane = object.rows * object.columns
        guard commonInputsAreValid(amplitudes: amplitudes, positions: positions, probeCount: n, iterations: iterations,
                                   stepSize: stepSize, normalizationMinimum: normalizationMinimum,
                                   radius: probeAmplitudeRelativeRadius, width: probeAmplitudeRelativeWidth),
              object.rows >= probe.rows, object.columns >= probe.columns,
              sliceThicknesses.count == slices - 1 else { throw PtychographyError.invalidInput }
        let intensity = amplitudes.reduce(0) { $0+$1*$1 }
        guard intensity > 0, intensity.isFinite else { throw PtychographyError.invalidInput }
        let props = try propagators(rows: probe.rows, columns: probe.columns, rowSampling: rowSampling,
                                    columnSampling: columnSampling, wavelength: wavelength,
                                    sliceThicknesses: sliceThicknesses.isEmpty ? [0] : sliceThicknesses)
        let frame = Frame(rows: probe.rows, columns: probe.columns, objectRows: object.rows, objectColumns: object.columns)
        var or = object.real, oi = object.imaginary, pr = probe.real, pi = probe.imaginary
        var errors = [Double]()
        let m = normalizationMinimum

        func propagate(_ re: inout [Double], _ im: inout [Double], slice s: Int, conjugate: Bool) {
            frame.fft.transform(&re, &im, inverse: false)
            for k in 0..<n {
                let a = props.real[s*n+k], b = conjugate ? -props.imaginary[s*n+k] : props.imaginary[s*n+k]
                let x = re[k], y = im[k]
                re[k] = x*a - y*b; im[k] = x*b + y*a
            }
            frame.fft.transform(&re, &im, inverse: true)
        }

        for _ in 0..<iterations {
            var onr = [Double](repeating: 0, count: or.count), oni = onr, pn = onr
            var pnr = [Double](repeating: 0, count: n), pni = pnr, on = pnr
            var spectrumRe = pr, spectrumIm = pi
            frame.fft.transform(&spectrumRe, &spectrumIm, inverse: false)
            var error = 0.0
            for pattern in 0..<count {
                let centre = frame.centre(positions[pattern])
                let indices = frame.indices(centre.rr, centre.cc)
                // forward
                var wavesRe = [[Double]](), wavesIm = [[Double]]()       // probe entering each slice
                var patchRe = [[Double]](), patchIm = [[Double]]()       // object patch of each slice
                var current = frame.shifted(spectrumRe, spectrumIm, centre.dr, centre.dc)
                var overlapRe = current.re, overlapIm = current.im
                for s in 0..<slices {
                    wavesRe.append(current.re); wavesIm.append(current.im)
                    var tr = [Double](repeating: 0, count: n), ti = tr
                    for k in 0..<n { tr[k] = or[s*plane+indices[k]]; ti[k] = oi[s*plane+indices[k]] }
                    patchRe.append(tr); patchIm.append(ti)
                    overlapRe = [Double](repeating: 0, count: n); overlapIm = overlapRe
                    for k in 0..<n {
                        overlapRe[k] = current.re[k]*tr[k]-current.im[k]*ti[k]; overlapIm[k] = current.re[k]*ti[k]+current.im[k]*tr[k]
                    }
                    if s + 1 < slices {
                        propagate(&overlapRe, &overlapIm, slice: s, conjugate: false)
                        current = (overlapRe, overlapIm)
                    }
                }
                // detector
                var fr = overlapRe, fi = overlapIm
                frame.fft.transform(&fr, &fi, inverse: false)
                for k in 0..<n {
                    let magnitude = hypot(fr[k], fi[k]), measured = amplitudes[pattern*n+k]
                    error += (measured-magnitude)*(measured-magnitude)
                    let angle = atan2(fi[k], fr[k])
                    fr[k] = measured*cos(angle)-fr[k]; fi[k] = measured*sin(angle)-fi[k]
                }
                frame.fft.transform(&fr, &fi, inverse: true)
                // backward
                for s in stride(from: slices-1, through: 0, by: -1) {
                    for k in 0..<n {
                        let j = s*plane+indices[k]
                        onr[j] += wavesRe[s][k]*fr[k]+wavesIm[s][k]*fi[k]; oni[j] += wavesRe[s][k]*fi[k]-wavesIm[s][k]*fr[k]
                        pn[j] += wavesRe[s][k]*wavesRe[s][k]+wavesIm[s][k]*wavesIm[s][k]
                    }
                    for k in 0..<n {            // exit *= conj(object patch)
                        let a = fr[k], b = fi[k], c = patchRe[s][k], d = patchIm[s][k]
                        fr[k] = a*c + b*d; fi[k] = b*c - a*d
                    }
                    if s > 0 {
                        propagate(&fr, &fi, slice: s-1, conjugate: true)
                    } else {
                        for k in 0..<n {
                            pnr[k] += fr[k]; pni[k] += fi[k]
                            on[k] += patchRe[0][k]*patchRe[0][k]+patchIm[0][k]*patchIm[0][k]
                        }
                    }
                }
            }
            for s in 0..<slices {
                let range = s*plane..<(s+1)*plane
                var re = Array(or[range]), im = Array(oi[range])
                applyObjectUpdate(&re, &im, Array(onr[range]), Array(oni[range]), Array(pn[range]),
                                  step: stepSize, m: m)
                or.replaceSubrange(range, with: re); oi.replaceSubrange(range, with: im)
            }
            if !fixProbe {
                let maximum = on.max()!
                for k in 0..<n {
                    let w = stepSize * updateNormalization(on[k], maximum, m)
                    pr[k] += w*pnr[k]; pi[k] += w*pni[k]
                }
            }
            if identicalSlices {
                for j in 0..<plane {
                    var sr = 0.0, si = 0.0
                    for s in 0..<slices { sr += or[s*plane+j]; si += oi[s*plane+j] }
                    for s in 0..<slices { or[s*plane+j] = sr/Double(slices); oi[s*plane+j] = si/Double(slices) }
                }
            }
            for j in or.indices { threshold(&or[j], &oi[j], pure: purePhaseObject) }
            if !fixProbe {
                if fixProbeCenterOfMass { centerOfMassConstraint(&pr, &pi, frame.fft) }
                if constrainProbeAmplitude {
                    amplitudeConstraint(&pr, &pi, rows: probe.rows, columns: probe.columns,
                                        radius: probeAmplitudeRelativeRadius, width: probeAmplitudeRelativeWidth)
                }
            }
            errors.append(error/intensity)
        }
        return try Result(object: PtychographyStack(count: slices, rows: object.rows, columns: object.columns, real: or, imaginary: oi),
                          probe: PtychographyArray(rows: probe.rows, columns: probe.columns, real: pr, imaginary: pi),
                          errors: errors)
    }
}
