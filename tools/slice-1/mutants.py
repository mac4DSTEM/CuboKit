"""CuboKit slice 1: break every new test once (registration section 6, gate 4).

    python3 -I tools/slice-1/mutants.py <scratch-dir> [mutant-id ...]

Exports HEAD (`git archive`) into <scratch-dir>/tree, then for each mutant: applies one exact text replacement (it must
occur exactly once), runs `swift test -c release --filter <its catchers>` in the copy, records RED (the tests failed or the
process trapped) or GREEN (a surviving mutant), and restores the file. The repository itself is never touched. Real-data
tests read $CUBOKIT_REAL_DATA as tools/test.sh does. Prints a markdown table; exit 1 if an expected-red mutant survived.
"""
import os
import shutil
import subprocess
import sys
import time

KIT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CORE = "Sources/CuboCore/"
VIRT = "Sources/CuboVirtual/"

# (id, what it breaks, file, old, new, catcher filter regex, expected: "red" or "survives")
MUTANTS = [
    ("M1", "mask transposed (centre row and column swapped)", VIRT + "DetectorGeometry.swift",
     "fill(&mask, qy: qy, qx: qx, cr: cr, cc: cc) { $0 < r2 }", "fill(&mask, qy: qy, qx: qx, cr: cc, cc: cr) { $0 < r2 }",
     "PinsTests/testMasksAreMakeDetectors|TruthTests/testDigitsClosedForms", "red"),
    ("M2", "outer radius <= instead of <", VIRT + "DetectorGeometry.swift",
     "{ $0 < r2 }", "{ $0 <= r2 }", "PinsTests/testMasksAreMakeDetectors|TruthTests/testBoundaryCountsByHand", "red"),
    ("M3", "inner radius >= instead of >", VIRT + "DetectorGeometry.swift",
     "{ $0 > in2 && $0 < out2 }", "{ $0 >= in2 && $0 < out2 }",
     "PinsTests/testMasksAreMakeDetectors|TruthTests/testBoundaryCountsByHand", "red"),
    ("M4", "rectangle rounded half up instead of half to even", VIRT + "DetectorGeometry.swift",
     ".rounded(.toNearestOrEven)", ".rounded(.toNearestOrAwayFromZero)",
     "PinsTests/testMasksAreMakeDetectors|TruthTests/testRectangleRoundsHalfToEvenAndClips", "red"),
    ("M5", "Float32 predicate instead of Double (mac4DSTEM's evaluation)", VIRT + "DetectorGeometry.swift",
     """            let di = Double(i) - cr
            let di2 = di * di
            for j in 0..<qx {
                let dj = Double(j) - cc
                if predicate(di2 + dj * dj) { mask[i * qx + j] = 1 }""",
     """            let di = Float(i) - Float(cr)
            let di2 = di * di
            for j in 0..<qx {
                let dj = Float(j) - Float(cc)
                if predicate(Double(di2 + dj * dj)) { mask[i * qx + j] = 1 }""",
     "PinsTests/testMasksAreMakeDetectors", "red"),
    ("M5b", "Float32 radius squared", VIRT + "DetectorGeometry.swift",
     "let r2 = r * r", "let r2 = Double(Float(r) * Float(r))", "PinsTests/testMasksAreMakeDetectors", "red"),
    ("M6", "the point at (column, row)", VIRT + "DetectorGeometry.swift",
     "mask[row * qx + column] = 1", "mask[(column % qy) * qx + (row % qx)] = 1",
     "TruthTests/testPointReadsItsOwnPixel", "red"),
    ("M7", "inner >= outer accepted", VIRT + "DetectorGeometry.swift",
     "guard outer > inner else", "guard outer >= inner else", "TruthTests/testRefusals", "red"),
    ("M8", "weights treated as 0/1", VIRT + "DetectorGeometry.swift",
     "            mask = weights\n", "            mask = weights.map { $0 != 0 ? 1 : 0 }\n",
     "PinsTests/testMasksAreMakeDetectors|PinsTests/testVirtualImagesOnTheCPU|TruthTests/testOnesCube", "red"),
    ("M9", "mean divided by one row's positions, not N", VIRT + "CPUReducers.swift",
     "let count = Double(shape.scanCount)", "let count = Double(shape.scanColumns)",
     "PinsTests/testMeanAndMaxOnTheCPU|TruthTests/testDigitsClosedForms|TruthTests/testScaledPatternCube", "red"),
    ("M10a", "eager tile read from row 0 whatever the range", CORE + "MemoryCube.swift",
     "values: Array(values[(rows.lowerBound * stride)..<(rows.upperBound * stride)]))",
     "values: Array(values[0..<(rows.count * stride)]))",
     "PinsTests/testLazyEagerAndEveryTilingGiveTheSameBits|CuboCoreTests/testEagerCubeReadsAndRefuses", "red"),
    ("M10b", "lazy tile read from element 0 whatever the range", CORE + "MappedCube.swift",
     "let values = convert(firstElement: rows.lowerBound * stride, count: rows.count * stride)",
     "let values = convert(firstElement: 0, count: rows.count * stride)",
     "PinsTests/testMeanAndMaxOnTheCPU|CuboCoreTests/testLazyCubeEqualsTheFileForEveryFixture", "red"),
    ("M11", "Float32 accumulation in the CPU mean (no compensation)", VIRT + "CPUReducers.swift",
     """                                        if e != 0 { c[i] += e }
                                    }
                                    s[i] = t""",
     """                                        if e != 0 { c[i] += 0 }
                                    }
                                    s[i] = Double(Float(a) + Float(x))""",
     "PinsTests/testMeanAndMaxOnTheCPU|RealDataTests", "red"),
    ("M11b", "compensation dropped, Double accumulation kept", VIRT + "CPUReducers.swift",
     "if e != 0 { c[i] += e }", "if e != 0 { c[i] += 0 }",
     "PinsTests/testMeanAndMaxOnTheCPU|RealDataTests", "survives"),
    ("M12", "maximum seeded at 0 (CPU)", VIRT + "CPUReducers.swift",
     "var maximum = [Float](repeating: -.infinity, count: n)\n        for range in try tiling.ranges(for: shape) {\n            try Task.checkCancellation()\n            let tile = try await source.readScanTile(rows: range)\n            tile.values",
     "var maximum = [Float](repeating: 0, count: n)\n        for range in try tiling.ranges(for: shape) {\n            try Task.checkCancellation()\n            let tile = try await source.readScanTile(rows: range)\n            tile.values",
     "PinsTests/testMeanAndMaxOnTheCPU|TruthTests/testAllNegativeMaximum", "red"),
    ("M12b", "maximum seeded at 0 (Metal kernel)", VIRT + "MetalReducers.swift",
     "float vmax = -INFINITY;", "float vmax = 0.0f;",
     "PinsTests/testMetalAgrees|TruthTests/testAllNegativeMaximum", "red"),
    ("M13", ".npy shape read in reverse", CORE + "NPYFile.swift",
     "fields.shape = shape\n", "fields.shape = shape.reversed()\n",
     "PinsTests/testFixturesAreTheGeneratorsBytes|CuboCoreTests/testHeaderOfANumPyFile", "red"),
    ("M14", ".npy v1 header length read big-endian", CORE + "NPYFile.swift",
     "headerLength = Int(prefix[8]) | Int(prefix[9]) << 8\n", "headerLength = Int(prefix[8]) << 8 | Int(prefix[9])\n",
     "CuboCoreTests/testHeaderOfANumPyFile|PinsTests/testFixturesAreTheGeneratorsBytes", "red"),
    ("M14b", ".npy v2 header length bytes swapped", CORE + "NPYFile.swift",
     "Int(prefix[10]) << 16 | Int(prefix[11]) << 24", "Int(prefix[11]) << 16 | Int(prefix[10]) << 24",
     "CuboCoreTests/testVersion2HeaderIsRead", "red"),
    ("M15", "uint16 read as int16", CORE + "MappedCube.swift",
     "vDSP_vfltu16(source.assumingMemoryBound(to: UInt16.self), 1, destination, 1, n)",
     "vDSP_vflt16(source.assumingMemoryBound(to: Int16.self), 1, destination, 1, n)",
     "CuboCoreTests/testEveryElementTypeConverts|RealDataTests/testPy4DSTEMSmallDatacube", "red"),
    ("M15b", "uint32 read as int32", CORE + "MappedCube.swift",
     "vDSP_vfltu32(source.assumingMemoryBound(to: UInt32.self), 1, destination, 1, n)",
     "vDSP_vflt32(source.assumingMemoryBound(to: Int32.self), 1, destination, 1, n)",
     "CuboCoreTests/testEveryElementTypeConverts", "red"),
    ("M16", "float64 bytes read as float32", CORE + "MappedCube.swift",
     "vDSP_vdpsp(source.assumingMemoryBound(to: Double.self), 1, destination, 1, n)",
     "destination.update(from: source.assumingMemoryBound(to: Float.self), count: count)",
     "CuboCoreTests/testFloat64RoundsToNearest|PinsTests/testVirtualImagesOnTheCPU", "red"),
    ("M16b", "int64 read as uint64", CORE + "MappedCube.swift",
     "let s = source.assumingMemoryBound(to: Int64.self)", "let s = source.assumingMemoryBound(to: UInt64.self)",
     "CuboCoreTests/testEveryElementTypeConverts", "red"),
    ("M17", "Metal tile written at offset 0", VIRT + "MetalReducers.swift",
     "            let first = range.lowerBound * shape.scanColumns\n            let pairs = MetalContext",
     "            let first = 0\n            let pairs = MetalContext",
     "PinsTests/testMetalAgrees|PinsTests/testLazyEager", "red"),
    ("M17b", "Metal mean keeps only the last tile", VIRT + "MetalReducers.swift",
     "sum[i] += Double(s[2 * i]) + Double(s[2 * i + 1])", "sum[i] = Double(s[2 * i]) + Double(s[2 * i + 1])",
     "PinsTests/testMetalAgrees", "red"),
    # The first kernels (two-level plain Float32 sums) failed Gate B F1; M18 on them survived as registered. On the
    # compensated kernels each half of the compensation is a mutant of its own (amendment 2026-10-15).
    ("M18", "Metal image: the correction term dropped", VIRT + "MetalReducers.swift",
     "image[gid] = float2(s, c);", "image[gid] = float2(s, 0.0f);", "RefuterRegressionTests", "red"),
    ("M18b", "Metal image: the TwoProduct error dropped", VIRT + "MetalReducers.swift",
     "            c += error;\n", "            c += 0.0f;\n", "RefuterRegressionTests", "red"),
    ("M18c", "Metal mean: the correction term dropped", VIRT + "MetalReducers.swift",
     "sums[gid] = float2(s, c);", "sums[gid] = float2(s, 0.0f);", "RefuterRegressionTests", "red"),
    ("M18d", "Metal max drops NaN", VIRT + "MetalReducers.swift",
     "vmax = (isnan(vmax) || isnan(v)) ? NAN : max(vmax, v);", "vmax = max(vmax, v);",
     "RefuterRegressionTests/testNonFiniteAndSignedZeroBehaviour", "red"),
    ("M19", "automatic tile budget halved", CORE + "CubeSource.swift",
     "public static let automaticTileBytes = 64 * 1024 * 1024", "public static let automaticTileBytes = 32 * 1024 * 1024",
     "CuboCoreTests/testAutomaticTilingIsAFunctionOfTheShape", "red"),
    # Equivalent: a zero extent is also refused by the byte-count guard, and negatives stay refused (recorded, 2026-10-15).
    ("M20", "zero extent accepted", CORE + "CubeShape.swift",
     "guard extents.allSatisfy({ $0 > 0 })", "guard extents.allSatisfy({ $0 >= 0 })",
     "CuboCoreTests/testShapeValidatesAndCounts", "survives"),
    ("M20b", "the per-extent guard dropped (two negatives multiply to a positive count)", CORE + "CubeShape.swift",
     "guard extents.allSatisfy({ $0 > 0 })", "guard extents.allSatisfy({ _ in true })",
     "CuboCoreTests/testShapeValidatesAndCounts", "red"),
    ("M21", "big-endian read as little-endian", CORE + "NPYFile.swift",
     "            throw CuboError.bigEndianNotSupported(descr)\n", "            return try elementType(\"<\" + code)\n",
     "CuboCoreTests/testRefusals", "red"),
    ("M22", "write pads to 16 bytes, not 64", CORE + "NPYFile.swift",
     "(64 - unpadded % 64) % 64", "(16 - unpadded % 16) % 16",
     "CuboCoreTests/testWriteRoundTripsAndPadsAsNumPy", "red"),
    ("M26", "CPU mean: compensation on a non-finite step (inf - inf)", VIRT + "CPUReducers.swift",
     "                                    if t.isFinite {\n", "                                    if true {\n",
     "RefuterRegressionTests/testNonFiniteAndSignedZeroBehaviour", "red"),
    ("M27", "CPU mean accumulator starts at +0", VIRT + "CPUReducers.swift",
     "var sum = [Double](repeating: -0.0, count: n)", "var sum = [Double](repeating: 0, count: n)",
     "RefuterRegressionTests/testNonFiniteAndSignedZeroBehaviour", "red"),
    ("M28", ".npy byte-count overflow not guarded", CORE + "NPYFile.swift",
     "            guard !overflow else { throw CuboError.notNPY(\"'shape' \\(shape) overflows the byte count\") }\n",
     "            if overflow { break }\n",
     "CuboCoreTests/testOverflowingShapesAreRefused", "red"),
    ("M23", "pins file altered (a version)", "Tests/CuboVirtualTests/Fixtures/slice-1-pins.json",
     '"numpy": "2.5.0"', '"numpy": "2.5.1"', "PinsTests/testPinsCarryTheirVersionsAndEveryCube", "red"),
    ("M24", "placeholder scope text", "Sources/CuboStrain/CuboStrain.swift",
     "not registered yet.", "registered.", "CuboStrainPlaceholderTests", "red"),
    ("M25", "placeholder scope text", "Sources/CuboPtycho/CuboPtycho.swift",
     "not registered yet.", "registered.", "CuboPtychoPlaceholderTests", "red"),
]


def run(cmd, cwd, timeout=1800):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr, time.time() - t0


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    scratch = os.path.abspath(sys.argv[1])
    only = set(sys.argv[2:])
    tree = os.path.join(scratch, "tree")
    head = subprocess.check_output(["git", "-C", KIT, "rev-parse", "--short", "HEAD"], text=True).strip()
    if os.path.isdir(tree):
        shutil.rmtree(tree)
    os.makedirs(tree)
    subprocess.run(f"git -C '{KIT}' archive HEAD | tar -x -C '{tree}'", shell=True, check=True)
    build = os.path.join(scratch, "build")
    print(f"# Mutants on {head} ({len(MUTANTS)} planned)\n")
    print("| id | mutant | catchers | expected | result | seconds |")
    print("|---|---|---|---|---|---|")
    survivors = []
    for mid, what, rel, old, new, filt, expected in MUTANTS:
        if only and mid not in only:
            continue
        path = os.path.join(tree, rel)
        text = open(path).read()
        n = text.count(old)
        if n != 1:
            print(f"| {mid} | {what} | `{filt}` | {expected} | NOT APPLIED (found {n}x) | |")
            survivors.append(mid)
            continue
        open(path, "w").write(text.replace(old, new))
        try:
            rc, out, dt = run(["swift", "test", "-c", "release", "--scratch-path", build, "--filter", filt], tree)
        finally:
            open(path, "w").write(text)
        # XCTest prints "Test Case '-[...]' started."; Swift Testing prints "Test name() started." (suites excluded).
        ran = sum(1 for l in out.splitlines() if "started" in l and ("Test Case '" in l or " Test " in l)
                  and "Suite" not in l and "Test run" not in l)
        result = "RED" if rc != 0 else "GREEN"
        if rc == 0 and ran == 0:
            result = "NO TEST RAN"
        if rc != 0 and ran == 0:
            result = "DID NOT COMPILE"       # not a catch: the mutant must build for a test to have caught it
        if (expected == "red") != result.startswith("RED"):
            survivors.append(mid)
        print(f"| {mid} | {what} | `{filt}` | {expected} | {result} ({ran} started) | {dt:.0f} |", flush=True)
    print()
    print(f"unexpected: {', '.join(survivors) if survivors else 'none'}")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
