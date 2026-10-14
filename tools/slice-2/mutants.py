"""CuboKit slice 2: break every new test once (registration slice 2, section 6, gate 4).

    python3 -I tools/slice-2/mutants.py <scratch-dir> [mutant-id ...]

Slice 1's runner (tools/slice-1/mutants.py: an exported copy of HEAD, one exact replacement per mutant, the catchers
run with `swift test -c release --filter`) with slice 2's mutants.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("slice1", os.path.join(HERE, "..", "slice-1", "mutants.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

V = "Sources/CuboVirtual/"
P = "Sources/CuboPhase/"
COM = "Slice2PinsTests/testCenterOfMassAgrees|Slice2TruthTests/testPlantedShift"
PROBE = "Slice2PinsTests/testProbeSize|Slice2TruthTests/testUniformDisk"
CHAIN = "CuboPhasePinsTests"
REAL = "CuboPhaseRealDataTests"

runner.MUTANTS = [
    ("S1", "CoM: the column index used for the row moment", V + "CenterOfMass.swift",
     "let wi = (0..<n).map { w[$0] * Double($0 / qx) }", "let wi = (0..<n).map { w[$0] * Double($0 % qx) }", COM, "red"),
    ("S2", "CoM: the mask ignored in the intensity", V + "CenterOfMass.swift",
     "a.add(x * w[i]); b.add", "a.add(x); b.add", COM, "red"),
    ("S3", "CoM: the row moment without the index", V + "CenterOfMass.swift",
     "b.add(x * wi[i])", "b.add(x * w[i])", COM, "red"),
    ("S4", "CoM: Neumaier compensation dropped", V + "CenterOfMass.swift",
     "if t.isFinite { correction +=", "if false { correction +=", COM + "|" + REAL, "survives"),
    ("S5", "CoM Metal: the correction term dropped", V + "MetalReducers.swift",
     "out[3 * gid] = float2(s0, c0);", "out[3 * gid] = float2(s0, 0.0f);",
     "Slice2PinsTests/testCenterOfMassMetal|" + REAL, "red"),
    ("P1", "plane: the constant term dropped", P + "CenterOfMassReference.swift",
     "return (0..<(R * C)).map { a + b *", "return (0..<(R * C)).map { 0 + b *", CHAIN + "|CuboPhaseTruthTests/testReferences", "red"),
    ("P2", "plane: non-finite positions not left out", P + "CenterOfMassReference.swift",
     "let finite = m.indices.filter { m[$0].isFinite }", "let finite = m.indices.filter { _ in true }",
     "CuboPhaseTruthTests/testReferences", "red"),
    ("P3", "plane: row and column slopes swapped", P + "CenterOfMassReference.swift",
     "a + b * (Double($0 / C) - rMean) + c * (Double($0 % C) - cMean)",
     "a + c * (Double($0 / C) - rMean) + b * (Double($0 % C) - cMean)", CHAIN + "|CuboPhaseTruthTests/testReferences", "red"),
    ("R1", "rotation: the sign of sin flipped", P + "RotationSolver.swift",
     "(zip(a, b).map { c * $0 - s * $1 }", "(zip(a, b).map { c * $0 + s * $1 }", CHAIN, "red"),
    ("R2", "rotation: the transpose ignored", P + "RotationSolver.swift",
     "let a = transpose ? f.column : f.row, b = transpose ? f.row : f.column", "let a = f.row, b = f.column", CHAIN, "red"),
    ("R3", "rotation: ties go to the transposed orientation", P + "RotationSolver.swift",
     "transposed = better(trans[j], than: plain[i])", "transposed = !better(plain[i], than: trans[j])",
     "CuboPhaseTruthTests/testTiesKeep", "red"),
    ("R4", "rotation: the curl's differences transposed", P + "RotationSolver.swift",
     "abs((y[(r + 1) * C + c] - y[(r - 1) * C + c]) - (x[r * C + c + 1] - x[r * C + c - 1]))",
     "abs((x[(r + 1) * C + c] - x[(r - 1) * C + c]) - (y[r * C + c + 1] - y[r * C + c - 1]))", CHAIN, "red"),
    ("R5", "rotation: the default range includes +90", P + "RotationSolver.swift",
     "(0..<179).map { Double($0) - 89 }", "(0..<180).map { Double($0) - 89 }", CHAIN, "red"),
    ("D1", "DPC: the 0.25 dropped", P + "DPC.swift",
     "gx[r * Cp + c] = -0.25 * kx[r] * kden", "gx[r * Cp + c] = -1.0 * kx[r] * kden", CHAIN, "red"),
    ("D2", "DPC: +i instead of -i in the operators", P + "DPC.swift",
     "ur[k] = -xi[k] * gx[k] + -yi[k] * gy[k]\n                ui[k] = dx[k] * gx[k] + dy[k] * gy[k]",
     "ur[k] = xi[k] * gx[k] + yi[k] * gy[k]\n                ui[k] = -(dx[k] * gx[k] + dy[k] * gy[k])", CHAIN, "red"),
    ("D3", "DPC: the padding ignored", P + "DPC.swift",
     "let Rp = Int((Double(R) * paddingFactor)", "let Rp = Int((Double(R) * 1.0)", CHAIN, "red"),
    ("D4", "DPC: backtracking without halving", P + "DPC.swift",
     "                step /= 2\n", "                step /= 1\n", "CuboPhaseTruthTests/testBacktrackingHalves", "red"),
    ("D5", "DPC: the error without its denominator", P + "DPC.swift",
     "let newError = NumPySum.mean(maskValues) / denominator", "let newError = NumPySum.mean(maskValues)", CHAIN, "red"),
    ("D6", "DPC: the inverse FFT unnormalized", P + "DoubleFFT.swift",
     "            let s = 1.0 / Double(n)\n", "            let s = 1.0\n", CHAIN, "red"),
    ("D7", "DPC: the roll direction reversed", P + "DPC.swift",
     "dx[r * Cp + c] = (up - down)", "dx[r * Cp + c] = (down - up)", CHAIN, "red"),
    ("D8", "DPC: Bluestein's chirp sign flipped", P + "DoubleFFT.swift",
     "let a = -Double.pi * Double(kk) / Double(n)", "let a = Double.pi * Double(kk) / Double(n)", CHAIN, "red"),
    ("Q1", "probe: >= instead of > at a threshold", V + "ProbeSize.swift",
     "$1 > limit ? $0 + 1 : $0", "$1 >= limit ? $0 + 1 : $0", PROBE, "red"),
    ("Q2", "probe: the gradient's first end zero", V + "ProbeSize.swift",
     "gradient[0] = radii[1] - radii[0]", "gradient[0] = 0", PROBE, "red"),
    ("Q3", "probe: the mean instead of the median", V + "ProbeSize.swift",
     "let median = count % 2 == 1 ?", "let median = true ? gradient.reduce(0, +) / Double(count) :", PROBE, "red"),
    ("Q4", "probe: the centre from the unmasked pattern", V + "ProbeSize.swift",
     "let masked = v.map { $0 * ($0 > limit ? 1.0 : 0.0) }", "let masked = v.map { $0 * 1.0 }", PROBE, "red"),
    ("Q5", "probe: linspace's last value left inexact", V + "ProbeSize.swift",
     "thresholds[count - 1] = upperThreshold", "_ = upperThreshold", PROBE, "red"),
    ("Q6", "NumPy's eight partial sums combined in order, not pairwise", "Sources/CuboCore/NumPySum.swift",
     "var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))",
     "var res = (((((((r0 + r1) + r2) + r3) + r4) + r5) + r6) + r7)", PROBE, "red"),
]

if __name__ == "__main__":
    runner.main()
