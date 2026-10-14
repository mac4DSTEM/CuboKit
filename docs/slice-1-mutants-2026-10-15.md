# Slice 1 mutants, 2026-10-15

Registration section 6, gate 4: every new test broken once. `tools/slice-1/mutants.py` on exported copies of HEAD (the repository is never touched), `swift test -c release --filter <catchers>`. Three runs, in order: on the XCTest tests (`6dd78fc`), on the Swift Testing tests (`9beaf45`), and on the Gate B fixes (`1b15f21`), the last one the record. Expected survivors, registered in advance: M11b (the CPU mean's compensation dropped: a robustness guard no fixture distinguishes, kept for the bound of 3.3) and M20 (a zero extent: equivalent, the byte-count guard refuses it too; M20b drops the per-extent guard and is red). On `6dd78fc`, M14b and M20 survived unexpectedly and two planted cases were added (registration §9); M18 then targeted the first, uncompensated kernel and survived as registered.

# Mutants on 1b15f21 (40 planned)

| id | mutant | catchers | expected | result | seconds |
|---|---|---|---|---|---|
| M1 | mask transposed (centre row and column swapped) | `PinsTests/testMasksAreMakeDetectors|TruthTests/testDigitsClosedForms` | red | RED (2 started) | 8 |
| M2 | outer radius <= instead of < | `PinsTests/testMasksAreMakeDetectors|TruthTests/testBoundaryCountsByHand` | red | RED (2 started) | 3 |
| M3 | inner radius >= instead of > | `PinsTests/testMasksAreMakeDetectors|TruthTests/testBoundaryCountsByHand` | red | RED (2 started) | 3 |
| M4 | rectangle rounded half up instead of half to even | `PinsTests/testMasksAreMakeDetectors|TruthTests/testRectangleRoundsHalfToEvenAndClips` | red | RED (2 started) | 3 |
| M5 | Float32 predicate instead of Double (mac4DSTEM's evaluation) | `PinsTests/testMasksAreMakeDetectors` | red | RED (1 started) | 3 |
| M5b | Float32 radius squared | `PinsTests/testMasksAreMakeDetectors` | red | RED (1 started) | 3 |
| M6 | the point at (column, row) | `TruthTests/testPointReadsItsOwnPixel` | red | RED (1 started) | 4 |
| M7 | inner >= outer accepted | `TruthTests/testRefusals` | red | RED (1 started) | 3 |
| M8 | weights treated as 0/1 | `PinsTests/testMasksAreMakeDetectors|PinsTests/testVirtualImagesOnTheCPU|TruthTests/testOnesCube` | red | RED (3 started) | 7 |
| M9 | mean divided by one row's positions, not N | `PinsTests/testMeanAndMaxOnTheCPU|TruthTests/testDigitsClosedForms|TruthTests/testScaledPatternCube` | red | RED (3 started) | 7 |
| M10a | eager tile read from row 0 whatever the range | `PinsTests/testLazyEagerAndEveryTilingGiveTheSameBits|CuboCoreTests/testEagerCubeReadsAndRefuses` | red | RED (2 started) | 6 |
| M10b | lazy tile read from element 0 whatever the range | `PinsTests/testMeanAndMaxOnTheCPU|CuboCoreTests/testLazyCubeEqualsTheFileForEveryFixture` | red | RED (2 started) | 4 |
| M11 | Float32 accumulation in the CPU mean (no compensation) | `PinsTests/testMeanAndMaxOnTheCPU|RealDataTests` | red | RED (6 started) | 18 |
| M11b | compensation dropped, Double accumulation kept | `PinsTests/testMeanAndMaxOnTheCPU|RealDataTests` | survives | GREEN (6 started) | 14 |
| M12 | maximum seeded at 0 (CPU) | `PinsTests/testMeanAndMaxOnTheCPU|TruthTests/testAllNegativeMaximum` | red | RED (2 started) | 4 |
| M12b | maximum seeded at 0 (Metal kernel) | `PinsTests/testMetalAgrees|TruthTests/testAllNegativeMaximum` | red | RED (2 started) | 7 |
| M13 | .npy shape read in reverse | `PinsTests/testFixturesAreTheGeneratorsBytes|CuboCoreTests/testHeaderOfANumPyFile` | red | RED (2 started) | 13 |
| M14 | .npy v1 header length read big-endian | `CuboCoreTests/testHeaderOfANumPyFile|PinsTests/testFixturesAreTheGeneratorsBytes` | red | RED (2 started) | 5 |
| M14b | .npy v2 header length bytes swapped | `CuboCoreTests/testVersion2HeaderIsRead` | red | RED (1 started) | 5 |
| M15 | uint16 read as int16 | `CuboCoreTests/testEveryElementTypeConverts|RealDataTests/testPy4DSTEMSmallDatacube` | red | RED (2 started) | 6 |
| M15b | uint32 read as int32 | `CuboCoreTests/testEveryElementTypeConverts` | red | RED (1 started) | 5 |
| M16 | float64 bytes read as float32 | `CuboCoreTests/testFloat64RoundsToNearest|PinsTests/testVirtualImagesOnTheCPU` | red | RED (2 started) | 5 |
| M16b | int64 read as uint64 | `CuboCoreTests/testEveryElementTypeConverts` | red | RED (1 started) | 4 |
| M17 | Metal tile written at offset 0 | `PinsTests/testMetalAgrees|PinsTests/testLazyEager` | red | RED (2 started) | 6 |
| M17b | Metal mean keeps only the last tile | `PinsTests/testMetalAgrees` | red | RED (1 started) | 3 |
| M18 | Metal image: the correction term dropped | `RefuterRegressionTests` | red | RED (6 started) | 6 |
| M18b | Metal image: the TwoProduct error dropped | `RefuterRegressionTests` | red | RED (6 started) | 6 |
| M18c | Metal mean: the correction term dropped | `RefuterRegressionTests` | red | RED (6 started) | 6 |
| M18d | Metal max drops NaN | `RefuterRegressionTests/testNonFiniteAndSignedZeroBehaviour` | red | RED (1 started) | 6 |
| M19 | automatic tile budget halved | `CuboCoreTests/testAutomaticTilingIsAFunctionOfTheShape` | red | RED (1 started) | 8 |
| M20 | zero extent accepted | `CuboCoreTests/testShapeValidatesAndCounts` | survives | GREEN (1 started) | 8 |
| M20b | the per-extent guard dropped (two negatives multiply to a positive count) | `CuboCoreTests/testShapeValidatesAndCounts` | red | RED (1 started) | 4 |
| M21 | big-endian read as little-endian | `CuboCoreTests/testRefusals` | red | RED (1 started) | 4 |
| M22 | write pads to 16 bytes, not 64 | `CuboCoreTests/testWriteRoundTripsAndPadsAsNumPy` | red | RED (1 started) | 4 |
| M26 | CPU mean: compensation on a non-finite step (inf - inf) | `RefuterRegressionTests/testNonFiniteAndSignedZeroBehaviour` | red | RED (1 started) | 5 |
| M27 | CPU mean accumulator starts at +0 | `RefuterRegressionTests/testNonFiniteAndSignedZeroBehaviour` | red | RED (1 started) | 3 |
| M28 | .npy byte-count overflow not guarded | `CuboCoreTests/testOverflowingShapesAreRefused` | red | RED (5 started) | 5 |
| M23 | pins file altered (a version) | `PinsTests/testPinsCarryTheirVersionsAndEveryCube` | red | RED (1 started) | 4 |
| M24 | placeholder scope text | `CuboStrainPlaceholderTests` | red | RED (1 started) | 2 |
| M25 | placeholder scope text | `CuboPtychoPlaceholderTests` | red | RED (1 started) | 3 |

unexpected: none

## The first run (`6dd78fc`, XCTest), for the record

| id | mutant | catchers | expected | result | seconds |
|---|---|---|---|---|---|
| M1 | mask transposed (centre row and column swapped) | `PinsTests/testMasksAreMakeDetectors|TruthTests/testDigitsClosedForms` | red | RED (2 started) | 16 |
| M2 | outer radius <= instead of < | `PinsTests/testMasksAreMakeDetectors|TruthTests/testBoundaryCountsByHand` | red | RED (2 started) | 3 |
| M3 | inner radius >= instead of > | `PinsTests/testMasksAreMakeDetectors|TruthTests/testBoundaryCountsByHand` | red | RED (2 started) | 3 |
| M4 | rectangle rounded half up instead of half to even | `PinsTests/testMasksAreMakeDetectors|TruthTests/testRectangleRoundsHalfToEvenAndClips` | red | RED (2 started) | 3 |
| M5 | Float32 predicate instead of Double (mac4DSTEM's evaluation) | `PinsTests/testMasksAreMakeDetectors` | red | RED (1 started) | 3 |
| M5b | Float32 radius squared | `PinsTests/testMasksAreMakeDetectors` | red | RED (1 started) | 3 |
| M6 | the point at (column, row) | `TruthTests/testPointReadsItsOwnPixel` | red | RED (1 started) | 3 |
| M7 | inner >= outer accepted | `TruthTests/testRefusals` | red | RED (1 started) | 4 |
| M8 | weights treated as 0/1 | `PinsTests/testMasksAreMakeDetectors|PinsTests/testVirtualImagesOnTheCPU|TruthTests/testOnesCube` | red | RED (3 started) | 7 |
| M9 | mean divided by one row's positions, not N | `PinsTests/testMeanAndMaxOnTheCPU|TruthTests/testDigitsClosedForms|TruthTests/testScaledPatternCube` | red | RED (3 started) | 7 |
| M10a | eager tile read from row 0 whatever the range | `PinsTests/testLazyEagerAndEveryTilingGiveTheSameBits|CuboCoreTests/testEagerCubeReadsAndRefuses` | red | RED (2 started) | 9 |
| M10b | lazy tile read from element 0 whatever the range | `PinsTests/testMeanAndMaxOnTheCPU|CuboCoreTests/testLazyCubeEqualsTheFileForEveryFixture` | red | RED (2 started) | 6 |
| M11 | Float32 accumulation in the CPU mean (no compensation) | `PinsTests/testMeanAndMaxOnTheCPU|RealDataTests` | red | RED (6 started) | 14 |
| M11b | compensation dropped, Double accumulation kept | `PinsTests/testMeanAndMaxOnTheCPU|RealDataTests` | survives | GREEN (6 started) | 11 |
| M12 | maximum seeded at 0 (CPU) | `PinsTests/testMeanAndMaxOnTheCPU|TruthTests/testAllNegativeMaximum` | red | RED (2 started) | 4 |
| M12b | maximum seeded at 0 (Metal kernel) | `PinsTests/testMetalAgrees|TruthTests/testAllNegativeMaximum` | red | RED (2 started) | 7 |
| M13 | .npy shape read in reverse | `PinsTests/testFixturesAreTheGeneratorsBytes|CuboCoreTests/testHeaderOfANumPyFile` | red | RED (2 started) | 8 |
| M14 | .npy v1 header length read big-endian | `CuboCoreTests/testHeaderOfANumPyFile|PinsTests/testFixturesAreTheGeneratorsBytes` | red | RED (2 started) | 4 |
| M14b | .npy v2 header length bytes swapped | `CuboCoreTests/testVersion2HeaderIsRead` | red | GREEN (1 started) | 4 |
| M15 | uint16 read as int16 | `CuboCoreTests/testEveryElementTypeConverts|RealDataTests/testPy4DSTEMSmallDatacube` | red | RED (2 started) | 6 |
| M15b | uint32 read as int32 | `CuboCoreTests/testEveryElementTypeConverts` | red | RED (1 started) | 4 |
| M16 | float64 bytes read as float32 | `CuboCoreTests/testFloat64RoundsToNearest|PinsTests/testVirtualImagesOnTheCPU` | red | RED (2 started) | 4 |
| M16b | int64 read as uint64 | `CuboCoreTests/testEveryElementTypeConverts` | red | RED (1 started) | 5 |
| M17 | Metal tile written at offset 0 | `PinsTests/testMetalAgrees|PinsTests/testLazyEager` | red | RED (2 started) | 6 |
| M17b | Metal mean keeps only the last tile | `PinsTests/testMetalAgrees` | red | RED (1 started) | 5 |
| M18 | Metal image summed one level (no row sums) | `PinsTests/testMetalAgrees|RealDataTests` | survives | GREEN (6 started) | 15 |
| M19 | automatic tile budget halved | `CuboCoreTests/testAutomaticTilingIsAFunctionOfTheShape` | red | RED (1 started) | 8 |
| M20 | zero extent accepted | `CuboCoreTests/testShapeValidatesAndCounts` | red | GREEN (1 started) | 9 |
| M21 | big-endian read as little-endian | `CuboCoreTests/testRefusals` | red | RED (1 started) | 5 |
| M22 | write pads to 16 bytes, not 64 | `CuboCoreTests/testWriteRoundTripsAndPadsAsNumPy` | red | RED (1 started) | 6 |
| M23 | pins file altered (a version) | `PinsTests/testPinsCarryTheirVersionsAndEveryCube` | red | RED (1 started) | 6 |
| M24 | placeholder scope text | `CuboStrainPlaceholderTests` | red | RED (1 started) | 3 |
| M25 | placeholder scope text | `CuboPtychoPlaceholderTests` | red | RED (1 started) | 5 |

unexpected: M14b, M20
