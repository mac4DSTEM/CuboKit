//
//  VacuumProbe.swift
//  The vacuum probe and the cross-correlation kernels Bragg disk detection uses (registration slice 4, 3.5, 3.6):
//  py4DSTEM's DataCube.get_vacuum_probe (datacube/datacube.py 516 at f050d207), Probe.measure_disk (= get_probe_size,
//  slice 2's ProbeSize.estimate), zero_vacuum, zero_vacuum_sinusoid, generate_synthetic_probe, get_probe_kernel_flat,
//  get_probe_kernel_edge_gaussian and get_probe_kernel_edge_sigmoid (braggvectors/probe.py), CPU, Double.
//
//  A Float32 reference implementation builds kernels on wrapped coordinates with a cosine shoulder in place of
//  py4DSTEM's vacuum mask; the kit follows py4DSTEM's shifts and sums.
//  DEVIATION: a boolean scan mask only (py4DSTEM's 4-tuple ROI slices an all-true array, so it always takes the top-left
//  block of the scan). Recorded quirks kept as py4DSTEM's: the first selected pattern has weight 0 in the running mean
//  (it anchors the first alignment), and the sinusoidal damping scales the disk itself by sin(1).
//

import CuboCore
import Foundation

/// A vacuum probe: the aligned running mean of the selected patterns, its disk, and the probe after the vacuum damping.
public struct VacuumProbe: Sendable, Equatable {
    /// The probe after `dampVacuum` with py4DSTEM's defaults (s 0.1, w 1, p 1), or the averaged probe without it.
    public let probe: DetectorImage
    /// The aligned running mean before any damping.
    public let averaged: DetectorImage
    /// `measure_disk` of the averaged probe: its radius (py4DSTEM's alpha) and centre (its origin).
    public let estimate: ProbeEstimate

    /// py4DSTEM's get_vacuum_probe: the selected positions (`scanMask`, row-major, default all) in scan order; each after
    /// the first multiplied by `detectorMask` (default 1), rolled onto the running probe by the argmax of their
    /// cross-correlation when `align`, and folded in as probe · (n − 1)/n + pattern/n.
    public static func measure(of source: some CubeSource, scanMask: [Bool]? = nil, align: Bool = true,
                               zeroVacuum: Bool = true, detectorMask: [Double]? = nil,
                               tiling: ScanTiling = .automatic) async throws -> VacuumProbe {
        let shape = source.shape
        let qy = shape.detectorRows, qx = shape.detectorColumns, n = shape.detectorCount
        if let scanMask, scanMask.count != shape.scanCount {
            throw CuboVirtualError.maskLengthMismatch(expected: shape.scanCount, received: scanMask.count)
        }
        if let detectorMask, detectorMask.count != n {
            throw CuboVirtualError.maskLengthMismatch(expected: n, received: detectorMask.count)
        }
        let fft = FFT2D(rows: qy, columns: qx)
        var probe: [Double]? = nil
        var count = 0
        for range in try tiling.ranges(for: shape) {
            try Task.checkCancellation()
            let tile = try await source.readScanTile(rows: range)
            let first = range.lowerBound * shape.scanColumns
            for k in 0..<tile.positionCount where scanMask?[first + k] ?? true {
                guard let current = probe else {
                    probe = (0..<n).map { Double(tile.values[k * n + $0]) }   // p₀, not masked
                    count = 1
                    continue
                }
                var d = (0..<n).map { Double(tile.values[k * n + $0]) * (detectorMask?[$0] ?? 1) }
                if align {
                    let s = alignmentShift(current, d, qy, qx, fft)
                    d = VirtualImaging.roll(d, qy, qx, s.row, s.column)
                }
                let w = Double(count)
                probe = (0..<n).map { i in current[i] * Double(count - 1) / w + d[i] / w }
                count += 1
            }
        }
        guard let averaged = probe else { throw CuboVirtualError.invalidInput("the selection holds no scan position") }
        let image = DetectorImage(detectorRows: qy, detectorColumns: qx, values: averaged, path: .cpu)
        let estimate = try ProbeSize.estimate(image)
        let final = zeroVacuum
            ? dampVacuum(image, center: (estimate.centerRow, estimate.centerColumn), alpha: estimate.radius) : image
        return VacuumProbe(probe: final, averaged: image, estimate: estimate)
    }

    /// py4DSTEM's get_shift: the first argmax of max(Re(IFFT2(FFT2(a) · conj(FFT2(b)))), 0), the roll that lays b on a.
    static func alignmentShift(_ a: [Double], _ b: [Double], _ qy: Int, _ qx: Int, _ fft: FFT2D) -> (row: Int, column: Int) {
        var ar = a, ai = [Double](repeating: 0, count: a.count)
        var br = b, bi = [Double](repeating: 0, count: b.count)
        fft.transform(&ar, &ai, inverse: false)
        fft.transform(&br, &bi, inverse: false)
        for k in ar.indices {
            let x = ar[k], y = ai[k], u = br[k], v = -bi[k]
            ar[k] = x * u - y * v; ai[k] = x * v + y * u
        }
        fft.transform(&ar, &ai, inverse: true)
        let cc = ar.map { $0.isNaN ? $0 : Swift.max($0, 0) }
        let best = FriedelOrigin.Plan.argmax(cc)
        return (best / qx, best % qx)
    }

    /// py4DSTEM's Probe.zero_vacuum: the probe times (hypot(i − r₀, j − c₀) < alpha · alphaMax).
    public static func zeroVacuum(_ probe: DetectorImage, center: (row: Double, column: Double), alpha: Double,
                                  alphaMax: Double = 1.2) -> DetectorImage {
        let qx = probe.detectorColumns, limit = alpha * alphaMax
        let v = probe.values.indices.map { k in
            probe.values[k] * (hypot(Double(k / qx) - center.row, Double(k % qx) - center.column) < limit ? 1.0 : 0.0)
        }
        return DetectorImage(detectorRows: probe.detectorRows, detectorColumns: qx, values: v, path: probe.path)
    }

    /// py4DSTEM's Probe.zero_vacuum_sinusoid: the probe times sin(f)^p, f = clip(1 + (alpha (1 + s) − r) / (alpha w), 0, 1).
    public static func dampVacuum(_ probe: DetectorImage, center: (row: Double, column: Double), alpha: Double,
                                  start s: Double = 0.1, width w: Double = 1, power p: Double = 1) -> DetectorImage {
        let qx = probe.detectorColumns
        let r = alpha * (1 + s), ww = alpha * w
        let v = probe.values.indices.map { k -> Double in
            let qrr = hypot(Double(k / qx) - center.row, Double(k % qx) - center.column)
            let f = Swift.min(Swift.max(1 + (r - qrr) / ww, 0), 1)
            return probe.values[k] * pow(sin(f), p)
        }
        return DetectorImage(detectorRows: probe.detectorRows, detectorColumns: qx, values: v, path: probe.path)
    }
}

/// The radial profile of the subtracted trench of a sigmoid kernel (py4DSTEM's `type`).
public enum SigmoidProfile: String, Sendable, Hashable, CaseIterable {
    /// cos(π/2 · clip((r − ri)/(ro − ri), 0, 1))², py4DSTEM's default.
    case sineSquared
    /// 1 / (1 + exp((r − (ri + ro)/2) / ((ro − ri)/4))).
    case logistic
}

/// Cross-correlation kernels from a probe, corner-centred (the probe's centre moved to pixel (0, 0)), for Bragg disk
/// detection. `origin` is the probe's centre; nil measures it with `ProbeSize.estimate` (py4DSTEM's default).
public enum ProbeKernel {
    /// get_probe_kernel_flat: the probe over its sum, shifted to the corner (Fourier unless `bilinear`). Sums to 1.
    public static func flat(_ probe: DetectorImage, origin: (row: Double, column: Double)? = nil,
                            bilinear: Bool = false) throws -> DetectorImage {
        let (r0, c0) = try center(probe, origin)
        let total = NumPySum.pairwise(probe.values)
        guard total != 0, total.isFinite else { throw CuboVirtualError.invalidInput("the probe sums to \(total)") }
        return image(probe, shift(probe.values.map { $0 / total }, probe, -r0, -c0, bilinear))
    }

    /// get_probe_kernel_edge_gaussian: the shifted probe over its sum minus a Gaussian over its sum, the Gaussian's width
    /// `sigma` times the probe's own radial standard deviation. Sums to 0.
    public static func gaussian(_ probe: DetectorImage, sigma: Double, origin: (row: Double, column: Double)? = nil,
                                bilinear: Bool = true) throws -> DetectorImage {
        guard sigma > 0, sigma.isFinite else { throw CuboVirtualError.invalidInput("sigma must be positive (\(sigma))") }
        let (r0, c0) = try center(probe, origin)
        let k = shift(probe.values, probe, -r0, -c0, bilinear)
        let q2 = wrappedSquares(probe)
        let sk = NumPySum.pairwise(k)
        let qstd2 = NumPySum.pairwise(zip(q2, k).map { $0 * $1 }) / sk
        let g = q2.map { exp(-$0 / (2 * qstd2 * (sigma * sigma))) }
        let sg = NumPySum.pairwise(g)
        guard sk != 0, sk.isFinite else { throw CuboVirtualError.invalidInput("the probe sums to \(sk)") }
        guard sg > 0, sg.isFinite else {
            throw CuboVirtualError.invalidInput("the Gaussian sums to \(sg) (σ_q² = \(qstd2), σ = \(sigma))")
        }
        return image(probe, k.indices.map { k[$0] / sk - g[$0] / sg })
    }

    /// get_probe_kernel_edge_sigmoid: the shifted probe over its sum minus a radial trench over its sum, 1 inside
    /// `innerRadius` falling to 0 at `outerRadius` (sine² or logistic). Sums to 0.
    public static func sigmoid(_ probe: DetectorImage, innerRadius ri: Double, outerRadius ro: Double,
                               profile: SigmoidProfile = .sineSquared, origin: (row: Double, column: Double)? = nil,
                               bilinear: Bool = true) throws -> DetectorImage {
        guard ri.isFinite, ro.isFinite, ro > ri else { throw CuboVirtualError.innerNotBelowOuter(inner: ri, outer: ro) }
        let (r0, c0) = try center(probe, origin)
        let k = shift(probe.values, probe, -r0, -c0, bilinear)
        let q = wrappedSquares(probe).map { $0.squareRoot() }
        let s: [Double]
        switch profile {
        case .sineSquared:
            s = q.map { let c = cos((Double.pi / 2) * Swift.min(Swift.max((($0 - ri) / (ro - ri)), 0.0), 1.0)); return c * c }
        case .logistic:
            let mid = 0.5 * (ro + ri), width = 0.25 * (ro - ri)
            s = q.map { 1 / (1 + exp(($0 - mid) / width)) }
        }
        let sk = NumPySum.pairwise(k), ss = NumPySum.pairwise(s)
        guard sk != 0, sk.isFinite else { throw CuboVirtualError.invalidInput("the probe sums to \(sk)") }
        guard ss > 0, ss.isFinite else { throw CuboVirtualError.invalidInput("the trench sums to \(ss) (radii \(ri), \(ro))") }
        return image(probe, k.indices.map { k[$0] / sk - s[$0] / ss })
    }

    static func center(_ probe: DetectorImage, _ origin: (row: Double, column: Double)?) throws -> (Double, Double) {
        guard probe.detectorRows > 0, probe.detectorColumns > 0 else {
            throw CuboVirtualError.invalidInput("the probe is empty (\(probe.detectorRows) × \(probe.detectorColumns))")
        }
        if let origin {
            guard origin.row.isFinite, origin.column.isFinite else { throw CuboVirtualError.invalidInput("the origin must be finite") }
            return (origin.row, origin.column)
        }
        let e = try ProbeSize.estimate(probe)
        return (e.centerRow, e.centerColumn)
    }

    static func image(_ like: DetectorImage, _ values: [Double]) -> DetectorImage {
        DetectorImage(detectorRows: like.detectorRows, detectorColumns: like.detectorColumns, values: values, path: .cpu)
    }

    /// q_r² + q_c² on the wrapped grid q = mod(arange(N) + N//2, N) − N//2 (integers, exact in Double).
    static func wrappedSquares(_ p: DetectorImage) -> [Double] {
        let qy = p.detectorRows, qx = p.detectorColumns
        let wr = (0..<qy).map { ($0 + qy / 2) % qy - qy / 2 }, wc = (0..<qx).map { ($0 + qx / 2) % qx - qx / 2 }
        return (0..<(qy * qx)).map { k in Double(wr[k / qx] * wr[k / qx] + wc[k % qx] * wc[k % qx]) }
    }

    /// py4DSTEM's get_shifted_ar: the Fourier shift, or the four bilinear rolls summed left to right.
    static func shift(_ v: [Double], _ like: DetectorImage, _ sr: Double, _ sc: Double, _ bilinear: Bool) -> [Double] {
        let qy = like.detectorRows, qx = like.detectorColumns
        guard bilinear else { return VirtualDiffraction.fourierShift(v, qy, qx, sr, sc, FFT2D(rows: qy, columns: qx)) }
        let fr = sr.rounded(.down), fc = sc.rounded(.down), wr = sr - fr, wc = sc - fc
        let ir = VirtualImaging.wrap(fr, qy), ic = VirtualImaging.wrap(fc, qx)
        let a = VirtualImaging.roll(v, qy, qx, ir, ic), b = VirtualImaging.roll(v, qy, qx, ir + 1, ic)
        let c = VirtualImaging.roll(v, qy, qx, ir, ic + 1), d = VirtualImaging.roll(v, qy, qx, ir + 1, ic + 1)
        let w00 = (1 - wr) * (1 - wc), w10 = wr * (1 - wc), w01 = (1 - wr) * wc, w11 = wr * wc
        return v.indices.map { a[$0] * w00 + b[$0] * w10 + c[$0] * w01 + d[$0] * w11 }
    }
}

/// py4DSTEM's Probe.generate_synthetic_probe: a disk of `radius` with a logistic edge of full width `width`, centred on
/// (rows/2, columns/2): 1 / (1 + exp(4 (r − radius) / width)).
public enum SyntheticProbe {
    public static func make(radius: Double, width: Double, detectorRows qy: Int, detectorColumns qx: Int) throws
        -> DetectorImage {
        guard radius.isFinite, width > 0, width.isFinite, qy > 0, qx > 0 else {
            throw CuboVirtualError.invalidInput("radius finite, width positive, a non-empty detector")
        }
        let v = (0..<(qy * qx)).map { k -> Double in
            let r = Double(k / qx) - Double(qy) / 2.0, c = Double(k % qx) - Double(qx) / 2.0
            let qr = (r * r + c * c).squareRoot() - radius
            return 1 / (1 + exp(4 * qr / width))
        }
        return DetectorImage(detectorRows: qy, detectorColumns: qx, values: v, path: .cpu)
    }
}
