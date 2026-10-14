"""CuboKit slice 3: break every new test once (registration slice 3, section 6). Slice 1's runner, slice 3's mutants.

    python3 -I tools/slice-3/mutants.py <scratch-dir> [mutant-id ...]
"""
import importlib.util, os
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("slice1", os.path.join(HERE, "..", "slice-1", "mutants.py"))
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
O = "Sources/CuboVirtual/OriginMeasurement.swift"
S = "Sources/CuboVirtual/ShiftedImaging.swift"
F = "Sources/CuboCore/OriginFit.swift"
T = "Slice3Tests/"
runner.MUTANTS = [
    ("G1", "Gaussian radius int(4σ) without + 0.5", O, "let R = Int(4.0 * sigma + 0.5)", "let R = Int(4.0 * sigma)", T, "red"),
    ("G2", "mode wrap instead of nearest (rows)", O, "p[min(max(i + t, 0), rows - 1) * columns + j]",
     "p[(((i + t) % rows) + rows) % rows * columns + j]", T, "red"),
    ("G4", "the last argmax instead of the first", O, "for i in 1..<n where g[i] > g[best]", "for i in 1..<n where g[i] >= g[best]", T, "red"),
    ("G5", "<= for < at r · rscale", O, "guard hypot(di, dj) < limit", "guard hypot(di, dj) <= limit", T, "red"),
    ("G6", "rscale ignored", O, "let limit = r * radiusScale", "let limit = r", T, "red"),
    ("G7", "the radius from the mean pattern", O, "let max = try await VirtualDiffraction.max(of: source, tiling: tiling)",
     "let max = try await VirtualDiffraction.mean(of: source, tiling: tiling)", T, "red"),
    ("F1", "parabola's rc term dropped", F, "case .parabola: a += [1, x, y, x * x, y * y, x * y]",
     "case .parabola: a += [1, x, y, x * x, y * y, 0.5 * x]", T, "red"),
    ("F2", "the robust mask reset each round", F, "for k in values.indices where squared[k] > limit { mask[k] = false }",
     "mask = values.map(\\.isFinite); for k in values.indices where squared[k] > limit { mask[k] = false }", T, "survives"),
    ("F3", "the robust mean over the kept positions only", F, "let limit = NumPySum.mean(squared) * robustThreshold * robustThreshold",
     "let limit = NumPySum.mean(values.indices.filter { mask[$0] }.map { squared[$0] }) * robustThreshold * robustThreshold", T, "red"),
    ("C1", "centring the column ignored (circle)", S, "return .circle(centerRow: (cr + r0) / s, centerColumn: (cc + c0) / s, radius: r / s)",
     "return .circle(centerRow: (cr + r0) / s, centerColumn: cc / s, radius: r / s)", T, "red"),
    ("C2", "the pixel size multiplied (circle radius)", S, "return .circle(centerRow: (cr + r0) / s, centerColumn: (cc + c0) / s, radius: r / s)",
     "return .circle(centerRow: (cr + r0) / s, centerColumn: (cc + c0) / s, radius: r * s)", T, "red"),
    ("S1", "the mask shift's sign flipped", S, "let dr = map.row.values[first + k] - map.meanRow", "let dr = map.meanRow - map.row.values[first + k]", T, "red"),
    ("S2", "rounding half away from zero", S, "m = roll(mask, qy, qx, Int(dr.rounded(.toNearestOrEven)), Int(dc.rounded(.toNearestOrEven)))",
     "m = roll(mask, qy, qx, Int(dr.rounded(.toNearestOrAwayFromZero)), Int(dc.rounded(.toNearestOrAwayFromZero)))", T, "red"),
    ("S3", "bilinear weights swapped", S, "let w00 = (1 - wr) * (1 - wc), w10 = wr * (1 - wc), w01 = (1 - wr) * wc, w11 = wr * wc",
     "let w00 = (1 - wr) * (1 - wc), w10 = (1 - wr) * wc, w01 = wr * (1 - wc), w11 = wr * wc", T, "red"),
    ("S4", "the Fourier shift's sign flipped", S, "let angle = -2 * Double.pi * (sc * fc[j] + sr * fr[i])",
     "let angle = 2 * Double.pi * (sc * fc[j] + sr * fr[i])", T, "red"),
    ("S5", "the shifted maximum seeded at 0", S, "var maximum = [Double](repeating: -.infinity, count: n)",
     "var maximum = [Double](repeating: 0, count: n)", T, "red"),
    ("S6", "the shifted mean divided by the scan rows", S, "        let count = Double(shape.scanCount)\n        return (DetectorImage",
     "        let count = Double(shape.scanRows)\n        return (DetectorImage", T, "red"),
]
if __name__ == "__main__":
    runner.main()
