"""CuboKit slice 4: break every new test once (registration slice 4, section 6). Slice 1's runner, slice 4's mutants (28 planned, 3 after Gate B).

    python3 -I tools/slice-4/mutants.py <scratch-dir> [mutant-id ...]
"""
import importlib.util, os
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("slice1", os.path.join(HERE, "..", "slice-1", "mutants.py"))
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
V = "Sources/CuboVirtual/"
SEL, FR, BS, VP, OM = V + "SelectedDiffraction.swift", V + "FriedelOrigin.swift", V + "BeamstopMask.swift", V + "VacuumProbe.swift", V + "OriginMeasurement.swift"
T = "Slice4Tests/"
runner.MUTANTS = [
    ("D1", "an even median as the upper middle", SEL, "return c % 2 == 1 ? v[c / 2] : (v[c / 2 - 1] + v[c / 2]) / 2",
     "return c % 2 == 1 ? v[c / 2] : v[c / 2]", T, "red"),
    ("D2", "NaN ignored by the median", SEL, "if v.contains(where: \\.isNaN) { return .nan }",
     "if false, v.contains(where: \\.isNaN) { return .nan }", T, "red"),
    ("D3", "the mask mean divided by the scan count", SEL, "let count = weights == nil ? Double(selected.count) : total.value",
     "let count = weights == nil ? Double(take.count) : total.value", T, "red"),
    ("D4", "weights normalised by the count", SEL, "let count = weights == nil ? Double(selected.count) : total.value",
     "let count = Double(selected.count)", T, "red"),
    ("D5", "the selected maximum seeded at 0", SEL, "var maximum = [Double](repeating: -.infinity, count: n)",
     "var maximum = [Double](repeating: 0, count: n)", T, "red"),
    ("D6", "the weighted median of the unweighted values", SEL, "gathered[i * count + s] = w.map { v * $0 } ?? v",
     "gathered[i * count + s] = v", T, "red"),
    ("O1", "the scan mask transposed", OM, "if let scanMask, !scanMask[first + k] {",
     "if let scanMask, !scanMask[((first + k) % shape.scanColumns) * shape.scanRows % shape.scanCount + (first + k) / shape.scanColumns] {", T, "red"),
    ("O2", "the fast centre refined anyway", OM, "if fastCenter { rows[first + k] = i0; cols[first + k] = j0; continue }",
     "if false && fastCenter { rows[first + k] = i0; cols[first + k] = j0; continue }", T, "red"),
    ("O3", "excluded origins 0 instead of NaN", OM, "rows[first + k] = .nan; cols[first + k] = .nan; continue",
     "rows[first + k] = 0; cols[first + k] = 0; continue", T, "red"),
    ("O4", "remove_NaN with the mean over all values (NaN as 0)", OM, "let fill = NumPySum.mean(m.values.filter { !$0.isNaN })",
     "let fill = NumPySum.mean(m.values.map { $0.isNaN ? 0 : $0 })", T, "red"),
    ("F1", "Friedel with |G|² instead of G²", FR, "            Self.square(&re, &im)\n            fft.transform(&re, &im, inverse: true)          // IFFT2(FFT2(im)²)",
     "            for k in re.indices { re[k] = re[k] * re[k] + im[k] * im[k]; im[k] = 0 }\n            fft.transform(&re, &im, inverse: true)          // IFFT2(FFT2(im)²)", T, "red"),
    ("F2", "the masked form's padding at 0", FR, "var mp = [Double](repeating: 1, count: 4 * qy * qx)",
     "var mp = [Double](repeating: 0, count: 4 * qy * qx)", T, "red"),
    ("F3", "term3 dropped", FR, "let term3 = pad[k] * maskPad[k]", "let term3 = 0.0", T, "red"),
    ("F4", "the parabola's denominator sign", FR, "(plus - minus) / (4.0 * c0 - 2.0 * plus - 2.0 * minus)",
     "(plus - minus) / (2.0 * plus + 2.0 * minus - 4.0 * c0)", T, "red"),
    ("F5", "the remainder as fmod (a negative origin kept negative)", FR,
     "return (Self.remainder(sx / 2, Double(qy)), Self.remainder(sy / 2, Double(qx)))",
     "return (fmod(sx / 2, Double(qy)), fmod(sy / 2, Double(qx)))", T, "red"),
    # Equivalent on every fixture: a tie needs (x + dx) · u to be an exact half-integer.
    ("F6", "the upsampled grid rounded half away from zero", FR,
     "let sx = (x * ud).rounded(.toNearestOrEven) / ud, sy = (y * ud).rounded(.toNearestOrEven) / ud",
     "let sx = (x * ud).rounded(.toNearestOrAwayFromZero) / ud, sy = (y * ud).rounded(.toNearestOrAwayFromZero) / ud", T, "survives"),
    ("F7", "the upsampled refinement dropped", FR, "if a > 0, a < n - 1, b > 0, b < n - 1 {", "if false {", T, "red"),
    ("B1", "the threshold index rounded half away from zero", BS,
     "let index = Int((Swift.min(Swift.max(Double(n) * threshold, 0), Double(n))).rounded(.toNearestOrEven))",
     "let index = Int((Swift.min(Swift.max(Double(n) * threshold, 0), Double(n))).rounded(.toNearestOrAwayFromZero))", T, "red"),
    ("B2", "holes filled with 8-connectivity", BS, "push(i - 1, j); push(i + 1, j); push(i, j - 1); push(i, j + 1)",
     "push(i - 1, j); push(i + 1, j); push(i, j - 1); push(i, j + 1); push(i - 1, j - 1); push(i + 1, j + 1); push(i - 1, j + 1); push(i + 1, j - 1)", T, "red"),
    ("B3", "the border kept with includeEdges", BS, "        if includeEdges {", "        if false && includeEdges {", T, "red"),
    ("B4", "<= for < against distanceEdge", BS, "return d2.map { $0.squareRoot() < distanceEdge }",
     "return d2.map { $0.squareRoot() <= distanceEdge }", T, "red"),
    ("P1", "the first vacuum pattern weighted like the others", VP,
     "probe = (0..<n).map { i in current[i] * Double(count - 1) / w + d[i] / w }",
     "probe = (0..<n).map { i in current[i] * w / (w + 1) + d[i] / (w + 1) }", T, "red"),
    ("P2", "the alignment shift's sign", VP, "d = VirtualImaging.roll(d, qy, qx, s.row, s.column)",
     "d = VirtualImaging.roll(d, qy, qx, -s.row, -s.column)", T, "red"),
    ("P3", "sin(f) as f in the damping", VP, "return probe.values[k] * pow(sin(f), p)", "return probe.values[k] * pow(f, p)", T, "red"),
    ("K1", "the Gaussian kernel's sigma not squared", VP, "let g = q2.map { exp(-$0 / (2 * qstd2 * (sigma * sigma))) }",
     "let g = q2.map { exp(-$0 / (2 * qstd2 * sigma)) }", T, "red"),
    ("K2", "the sigmoid's radii swapped", VP, "Swift.min(Swift.max((($0 - ri) / (ro - ri)), 0.0), 1.0)",
     "Swift.min(Swift.max((($0 - ro) / (ri - ro)), 0.0), 1.0)", T, "red"),
    ("K3", "the flat kernel not normalised", VP, "shift(probe.values.map { $0 / total }, probe, -r0, -c0, bilinear)",
     "shift(probe.values, probe, -r0, -c0, bilinear)", T, "red"),
    ("K4", "the bilinear weights swapped", VP, "let w00 = (1 - wr) * (1 - wc), w10 = wr * (1 - wc), w01 = (1 - wr) * wc, w11 = wr * wc",
     "let w00 = (1 - wr) * (1 - wc), w10 = (1 - wr) * wc, w01 = wr * (1 - wc), w11 = wr * wc", T, "red"),
    ("R1", "Gate B: an all-false Friedel mask accepted", FR, "if let mask, !mask.contains(true) {", "if let mask, mask.isEmpty {", T, "red"),
    ("R2", "Gate B: no upper bound on the upsampling factor", FR, "guard u == 1 || (u > 2 && u <= 1024) else {",
     "guard u == 1 || u > 2 else {", T, "red"),
    ("R3", "Gate B: of equal maxima the later kept", SEL, "maximum[i] = (m.isNaN || v.isNaN) ? .nan : (v > m ? v : m)",
     "maximum[i] = (m.isNaN || v.isNaN) ? .nan : (v >= m ? v : m)", T, "red"),
]
if __name__ == "__main__":
    runner.main()
