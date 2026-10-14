# Slice 2 mutants, 2026-10-15

Registration slice 2, section 6, gate 4. `tools/slice-2/mutants.py` (slice 1's runner) on exported copies. The record is the run on the Gate B fixes (`a67805c`): 29 red, and S4 (the CPU Neumaier compensation dropped) green as registered, a robustness guard no fixture distinguishes. Earlier runs on the commits before it: P2, Q1, Q3 and Q5 survived unexpectedly and were closed by planted cases (the fitted plane checked under a NaN; a ring exactly at a trusted threshold; a disk exactly at an inexact last threshold with the trusted threshold pinned; Q3 rewritten to compile). Slice 1's 40 mutants were re-run on the same commit: as registered, none unexpected.

# Mutants on a67805c (30 planned)

| id | mutant | catchers | expected | result | seconds |
|---|---|---|---|---|---|
| S1 | CoM: the column index used for the row moment | `Slice2PinsTests/testCenterOfMassAgrees|Slice2TruthTests/testPlantedShift` | red | RED (2 started) | 9 |
| S2 | CoM: the mask ignored in the intensity | `Slice2PinsTests/testCenterOfMassAgrees|Slice2TruthTests/testPlantedShift` | red | RED (2 started) | 3 |
| S3 | CoM: the row moment without the index | `Slice2PinsTests/testCenterOfMassAgrees|Slice2TruthTests/testPlantedShift` | red | RED (2 started) | 3 |
| S4 | CoM: Neumaier compensation dropped | `Slice2PinsTests/testCenterOfMassAgrees|Slice2TruthTests/testPlantedShift|CuboPhaseRealDataTests` | survives | GREEN (7 started) | 12 |
| S5 | CoM Metal: the correction term dropped | `Slice2PinsTests/testCenterOfMassMetal|CuboPhaseRealDataTests` | red | RED (6 started) | 17 |
| S6 | CoM Metal: the double-float renormalisation dropped | `Slice2RefuterRegressionTests` | red | RED (2 started) | 8 |
| R6 | rotation: deg · π / 180 instead of deg · (π / 180) | `CuboPhaseTruthTests/testRadians` | red | RED (1 started) | 8 |
| R7 | rotation: a non-finite field accepted | `CuboPhaseTruthTests/testRadians` | red | RED (1 started) | 4 |
| P1 | plane: the constant term dropped | `CuboPhasePinsTests|CuboPhaseTruthTests/testReferences` | red | RED (3 started) | 3 |
| P2 | plane: non-finite positions not left out | `CuboPhaseTruthTests/testReferences` | red | RED (1 started) | 3 |
| P3 | plane: row and column slopes swapped | `CuboPhasePinsTests|CuboPhaseTruthTests/testReferences` | red | RED (3 started) | 3 |
| R1 | rotation: the sign of sin flipped | `CuboPhasePinsTests` | red | RED (2 started) | 3 |
| R2 | rotation: the transpose ignored | `CuboPhasePinsTests` | red | RED (2 started) | 5 |
| R3 | rotation: ties go to the transposed orientation | `CuboPhaseTruthTests/testTiesKeep` | red | RED (1 started) | 5 |
| R4 | rotation: the curl's differences transposed | `CuboPhasePinsTests` | red | RED (2 started) | 3 |
| R5 | rotation: the default range includes +90 | `CuboPhasePinsTests` | red | RED (2 started) | 5 |
| D1 | DPC: the 0.25 dropped | `CuboPhasePinsTests` | red | RED (2 started) | 5 |
| D2 | DPC: +i instead of -i in the operators | `CuboPhasePinsTests` | red | RED (2 started) | 3 |
| D3 | DPC: the padding ignored | `CuboPhasePinsTests` | red | RED (2 started) | 3 |
| D4 | DPC: backtracking without halving | `CuboPhaseTruthTests/testBacktrackingHalves` | red | RED (1 started) | 3 |
| D5 | DPC: the error without its denominator | `CuboPhasePinsTests` | red | RED (2 started) | 3 |
| D6 | DPC: the inverse FFT unnormalized | `CuboPhasePinsTests` | red | RED (2 started) | 4 |
| D7 | DPC: the roll direction reversed | `CuboPhasePinsTests` | red | RED (2 started) | 3 |
| D8 | DPC: Bluestein's chirp sign flipped | `CuboPhasePinsTests` | red | RED (2 started) | 3 |
| Q1 | probe: >= instead of > at a threshold | `Slice2PinsTests/testProbeSize|Slice2TruthTests/testUniformDisk` | red | RED (2 started) | 3 |
| Q2 | probe: the gradient's first end zero | `Slice2PinsTests/testProbeSize|Slice2TruthTests/testUniformDisk` | red | RED (2 started) | 3 |
| Q3 | probe: the mean instead of the median | `Slice2PinsTests/testProbeSize|Slice2TruthTests/testUniformDisk` | red | RED (2 started) | 3 |
| Q4 | probe: the centre from the unmasked pattern | `Slice2PinsTests/testProbeSize|Slice2TruthTests/testUniformDisk` | red | RED (2 started) | 3 |
| Q5 | probe: linspace's last value left inexact | `Slice2PinsTests/testProbeSize|Slice2TruthTests/testUniformDisk` | red | RED (2 started) | 3 |
| Q6 | NumPy's eight partial sums combined in order, not pairwise | `Slice2PinsTests/testProbeSize|Slice2TruthTests/testUniformDisk` | red | RED (2 started) | 5 |

unexpected: none

## Slice 1's mutants on the same commit

| id | mutant | catchers | expected | result | seconds |
|---|---|---|---|---|---|
| M1 | mask transposed (centre row and column swapped) | `PinsTests/testMasksAreMakeDetectors|TruthTests/testDigitsClosedForms` | red | RED (2 started) | 13 |
| M2 | outer radius <= instead of < | `PinsTests/testMasksAreMakeDetectors|TruthTests/testBoundaryCountsByHand` | red | RED (2 started) | 4 |
| M3 | inner radius >= instead of > | `PinsTests/testMasksAreMakeDetectors|TruthTests/testBoundaryCountsByHand` | red | RED (2 started) | 4 |
| M4 | rectangle rounded half up instead of half to even | `PinsTests/testMasksAreMakeDetectors|TruthTests/testRectangleRoundsHalfToEvenAndClips` | red | RED (2 started) | 3 |
| M5 | Float32 predicate instead of Double (mac4DSTEM's evaluation) | `PinsTests/testMasksAreMakeDetectors` | red | RED (1 started) | 3 |
| M5b | Float32 radius squared | `PinsTests/testMasksAreMakeDetectors` | red | RED (1 started) | 3 |
| M6 | the point at (column, row) | `TruthTests/testPointReadsItsOwnPixel` | red | RED (1 started) | 3 |
| M7 | inner >= outer accepted | `TruthTests/testRefusals` | red | RED (2 started) | 3 |
| M8 | weights treated as 0/1 | `PinsTests/testMasksAreMakeDetectors|PinsTests/testVirtualImagesOnTheCPU|TruthTests/testOnesCube` | red | RED (3 started) | 6 |
| M9 | mean divided by one row's positions, not N | `PinsTests/testMeanAndMaxOnTheCPU|TruthTests/testDigitsClosedForms|TruthTests/testScaledPatternCube` | red | RED (3 started) | 5 |
| M10a | eager tile read from row 0 whatever the range | `PinsTests/testLazyEagerAndEveryTilingGiveTheSameBits|CuboCoreTests/testEagerCubeReadsAndRefuses` | red | RED (2 started) | 7 |
| M10b | lazy tile read from element 0 whatever the range | `PinsTests/testMeanAndMaxOnTheCPU|CuboCoreTests/testLazyCubeEqualsTheFileForEveryFixture` | red | RED (2 started) | 7 |
| M11 | Float32 accumulation in the CPU mean (no compensation) | `PinsTests/testMeanAndMaxOnTheCPU|RealDataTests` | red | RED (11 started) | 42 |
| M11b | compensation dropped, Double accumulation kept | `PinsTests/testMeanAndMaxOnTheCPU|RealDataTests` | survives | GREEN (11 started) | 39 |
| M12 | maximum seeded at 0 (CPU) | `PinsTests/testMeanAndMaxOnTheCPU|TruthTests/testAllNegativeMaximum` | red | RED (2 started) | 6 |
| M12b | maximum seeded at 0 (Metal kernel) | `PinsTests/testMetalAgrees|TruthTests/testAllNegativeMaximum` | red | RED (2 started) | 7 |
| M13 | .npy shape read in reverse | `PinsTests/testFixturesAreTheGeneratorsBytes|CuboCoreTests/testHeaderOfANumPyFile` | red | RED (2 started) | 9 |
| M14 | .npy v1 header length read big-endian | `CuboCoreTests/testHeaderOfANumPyFile|PinsTests/testFixturesAreTheGeneratorsBytes` | red | RED (2 started) | 7 |
| M14b | .npy v2 header length bytes swapped | `CuboCoreTests/testVersion2HeaderIsRead` | red | RED (1 started) | 8 |
| M15 | uint16 read as int16 | `CuboCoreTests/testEveryElementTypeConverts|RealDataTests/testPy4DSTEMSmallDatacube` | red | RED (3 started) | 9 |
| M15b | uint32 read as int32 | `CuboCoreTests/testEveryElementTypeConverts` | red | RED (1 started) | 4 |
| M16 | float64 bytes read as float32 | `CuboCoreTests/testFloat64RoundsToNearest|PinsTests/testVirtualImagesOnTheCPU` | red | RED (2 started) | 5 |
| M16b | int64 read as uint64 | `CuboCoreTests/testEveryElementTypeConverts` | red | RED (1 started) | 6 |
| M17 | Metal tile written at offset 0 | `PinsTests/testMetalAgrees|PinsTests/testLazyEager` | red | RED (2 started) | 7 |
| M17b | Metal mean keeps only the last tile | `PinsTests/testMetalAgrees` | red | RED (1 started) | 5 |
| M18 | Metal image: the correction term dropped | `RefuterRegressionTests` | red | RED (8 started) | 10 |
| M18b | Metal image: the TwoProduct error dropped | `RefuterRegressionTests` | red | RED (8 started) | 9 |
| M18c | Metal mean: the correction term dropped | `RefuterRegressionTests` | red | RED (8 started) | 10 |
| M18d | Metal max drops NaN | `RefuterRegressionTests/testNonFiniteAndSignedZeroBehaviour` | red | RED (1 started) | 8 |
| M19 | automatic tile budget halved | `CuboCoreTests/testAutomaticTilingIsAFunctionOfTheShape` | red | RED (1 started) | 11 |
| M20 | zero extent accepted | `CuboCoreTests/testShapeValidatesAndCounts` | survives | GREEN (1 started) | 11 |
| M20b | the per-extent guard dropped (two negatives multiply to a positive count) | `CuboCoreTests/testShapeValidatesAndCounts` | red | RED (1 started) | 5 |
| M21 | big-endian read as little-endian | `CuboCoreTests/testRefusals` | red | RED (1 started) | 5 |
| M22 | write pads to 16 bytes, not 64 | `CuboCoreTests/testWriteRoundTripsAndPadsAsNumPy` | red | RED (1 started) | 5 |
| M26 | CPU mean: compensation on a non-finite step (inf - inf) | `RefuterRegressionTests/testNonFiniteAndSignedZeroBehaviour` | red | RED (1 started) | 6 |
| M27 | CPU mean accumulator starts at +0 | `RefuterRegressionTests/testNonFiniteAndSignedZeroBehaviour` | red | RED (1 started) | 4 |
| M28 | .npy byte-count overflow not guarded | `CuboCoreTests/testOverflowingShapesAreRefused` | red | RED (5 started) | 6 |
| M23 | pins file altered (a version) | `PinsTests/testPinsCarryTheirVersionsAndEveryCube` | red | RED (1 started) | 5 |
| M24 | placeholder scope text | `CuboStrainPlaceholderTests` | red | RED (1 started) | 3 |
| M25 | placeholder scope text | `CuboPtychoPlaceholderTests` | red | RED (1 started) | 3 |

unexpected: none
