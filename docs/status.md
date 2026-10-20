# Status

**Hand-off 2026-10-20 (app-resident computation):** done: ptychography preprocessing, mixed-state and multislice engines, aperture and support constraints, crop-and-bin views, DPC views and the R-Q sign convention, all pinned (CHANGELOG "Unreleased (0.5.0 candidate)", parity 79/37/66); next: owner pushes and tags 0.4.0 then 0.5.0, the app switches (preparer, load view, DPC views), then position correction, the Fourier-amplitude constraint, `MixedstateMultislicePtychography`.

Previous hand-off (kit gaps, 2026-10-20): done: probe-centre and probe-amplitude constraints, `Parallax.reconstruct` plus end-to-end `process`, scan-level `StrainMapping`, all pinned to py4DSTEM (CHANGELOG "Unreleased", parity 76/35/71); next: owner pushes and tags 0.4.0, then multislice/mixed-state ptychography, the remaining ptycho probe constraints, parallax initial-aberration shifts.

Earlier: D12 wording in CLAUDE.md, docs and test comments (d43b531); convert exact-asserting numeric tests to written tolerances (one per release).

What is live, one table; then the hand-off. Rules are in `CLAUDE.md`; the registration in `docs/registration-slice-1.md`;
parity with py4DSTEM in `docs/parity.md`; what mac4DSTEM changes to import the kit in `docs/consumers-0.0.1.md` to
`docs/consumers-0.1.0.md`, `docs/consumers-0.2.0.md`; from 0.3.0 the `CHANGELOG.md` Migration paragraph.

## 0.3.0 (2026-10-16)

Seven pieces on the CPU (Double unless stated), each pinned to py4DSTEM 0.14.19 (`f050d207`), scikit-learn 1.4.2
or a transcription of the app source, with the mac4DSTEM differences as named options (details in `CHANGELOG.md` and the per-module
documents):

| Module | State | Document |
|---|---|---|
| `CuboParallax` | alignment (matrix-DFT, fixed/running reference), constant-order preprocessing, aberration fit and correction, depth sectioning, subpixel KDE upsampling with position correction | `docs/parallax-alignment.md` |
| `CuboPtycho` | probe construction and full-batch single-slice complex-object gradient descent (fixed positions) | `docs/ptycho.md` |
| `CuboStrain` | basis selection, indexing, lattice fits, references, rotated strain; three named compatibility options | `docs/strain.md` |
| `CuboBragg` | `find_Bragg_disks` (pattern, stack, cube) and `ProbeRingHint` | `docs/bragg.md` |
| `CuboEmbedding` | PCA, k-means; the app's variants as named options | `docs/embedding.md` |
| `CuboPhase` | Fourier iDPC | `docs/idpc.md` |
| `CuboCore` | `HotPixelFilter`, `FrameTransform` (origin re-reference under crop and bin) | `docs/hot-pixels.md` |

Parity with py4DSTEM (182 rows in scope): 73 done, 36 partial, 73 absent (`docs/parity.md`). Gate: `tools/test.sh`
(`swift test -c release`); its result for the tagged commit is the CI run after the owner's push. Not yet provided:
full parallax reconstruction (grouping, schedules, regularisation), multi-slice and other ptychography methods, GPU
paths for the new modules. Large-stack memory and iOS runtime behaviour of the new modules are unverified.


## Record through 0.2.0 (2026-10-15)

| Area | State | Evidence |
|---|---|---|
| Name | CuboKit passed the clash check on 2026-10-15: no Apple framework, no Swift Package Index package, no 4D-STEM tool of that name (the nearest are GoCubeKit, a Go `cube-kit`, and a Python `cubo` for satellite data cubes) | the session's searches |
| Registration of slice 1 | Written before any code (`b620ba5`); pins before any Swift (`6833b4e`); two dated amendments (§9), the second after Gate B | `docs/registration-slice-1.md` |
| `CuboCore` | `CubeShape`, `CubeSource` (async reads), `MemoryCube`, memory-mapped `MappedCube` (ten element types to Float32), `NPYFile` | `CuboCoreTests` |
| `CuboVirtual` | `DetectorGeometry` and `DetectorMask` (py4DSTEM's `make_detector` in Double), `VirtualImaging.image`, `VirtualDiffraction.mean/max/meanAndMax`; `.cpu` (Accelerate, Double, compensated mean) and `.metal` (compensated Float32 kernels, Dot2) | `PinsTests`, `TruthTests`, `RefuterRegressionTests` |
| Pins against py4DSTEM `f050d207` | 7 synthetic cubes, 71 geometries: masks identical; CPU at distance 0.0 from py4DSTEM's float64 definition; integer cubes bit for bit against py4DSTEM's own calls | `tools/slice-1/pins.py`, `Fixtures/slice-1-pins.json` |
| Real data | The owner's Al–Mg–Si study cube (330 × 330 × 64 × 64), Au reference, simulated poly-Au, the particle cube, and py4DSTEM's `small_datacube.dm4`: CPU mean and BF, ADF, DF images at distance 0.0 from float64; Metal within 2.4e-13; eager equals lazy; py4DSTEM's own float32 mean of the study cube is 1.9e-5 from its float64 definition | `RealDataTests`, log `test-1b15f21` (session scratchpad), exports in `~/Library/Caches/CuboKit/real-data` |
| Slice 2 registration | Before code (`9e12232`), pins before Swift (`0603136`), two amendments (§9: first run; Gate B) | `docs/registration-slice-2.md` |
| `CuboVirtual` slice 2 | `CenterOfMass` (Neumaier CPU; Metal with renormalised double-float sums), `ProbeSize` (py4DSTEM's `get_probe_size`, NumPy's sums) | `Slice2PinsTests`, `Slice2TruthTests`, `Slice2RefuterRegressionTests` |
| `CuboPhase` | `CenterOfMassReference` (plane, constant, none, given), `RotationSolver` (curl / divergence, py4DSTEM's selection), `DPC.reconstruct` (iterative, Double, Bluestein FFT) | `CuboPhasePinsTests`, `CuboPhaseTruthTests` |
| Slice 2 on real data | The five cubes: CPU CoM 0.0 px from float64, Metal ≤ 1.9e-10 px; rotation and transpose equal to py4DSTEM (Al–Mg–Si 83° T, Au 2° T, sim Au 86°, particle 80° T, small cube −18°); DPC phase ≤ 3.3e-12 of max from the float64 chain (py4DSTEM's own float32: 5e-7 to 5.2e-5); probe size bit for bit | `CuboPhaseRealDataTests`, log `test-s2-gateB` |
| Slice 2 Gate B | One refuter on `84815c9`: Metal CoM fixed (F1; the Metal claim needs \|S\| ≥ 1e-3·Σ\|wP\|), radians as NumPy (F3), py4DSTEM's float32 tie flips recorded (F2), non-finite input refused (F4) | `registration-slice-2.md` §9, `a67805c` |
| Slice 2 break-once proof | 30 mutants on `a67805c`: 29 red, the registered survivor green; slice 1's 40 re-run there as registered | `docs/slice-2-mutants-2026-10-15.md` |
| Slice 3 | Registration `22c3bdd`, pins `d272160`, code `5c82508`, Gate B fixes `63dbfef`: `OriginMeasurement` (get_origin in Double, scipy's filter order), `OriginFit` (four models, robust, lstsq's projection when rank-deficient), centred, calibrated and shifted virtual images, shifted mean and max (rounded, Fourier) | `Slice3Tests`, `Slice3RealDataTests`, `docs/registration-slice-3.md` |
| Slice 3 on real data | The five cubes: origins 0.0 px from float64; py4DSTEM's float32 filter moves 2 of 63 550 Au argmaxes (recorded); integer cubes bit for bit against py4DSTEM's centred, calibrated and rounded images | log `test-s3-final` |
| Slice 3 Gate B and mutants | Refuter on `5c82508`: symmetric ties (F1), traps (F2), rank-deficient fits (F3), NaN (F4) fixed; the Fourier claim restated normwise. 19 mutants on `63dbfef`: 17 red, 2 registered survivors | `docs/slice-3-mutants-2026-10-15.md` |
| Slice 4 | Registration `086999f`, pins `b4b4efd`, code `bc41094`, Gate B fixes `006fe47`: `VirtualDiffraction.median` and `.pattern(of:statistic:scan:shift:)` (mean, max, median over `.all`, `.mask`, `.weights`; shifted), `OriginMeasurement.measure(scanMask:fastCenter:)`, `OriginMap.replacingNaNWithMean`, `FriedelOrigin` (masked, DFT upsampling), `BeamstopMask`, `VacuumProbe` (`measure`, `zeroVacuum`, `dampVacuum`), `ProbeKernel` (flat, Gaussian, sine² and logistic sigmoid), `SyntheticProbe`; CPU, Double | `Slice4Tests`, `Slice4RealDataTests`, `docs/registration-slice-4.md` |
| Slice 4 on real data | Five cubes and the owner's bullseye vacuum probe (a new export): medians, maxima and beamstop masks bit for bit; Friedel origins within 2e-14 px of the float64 definition (py4DSTEM's complex64 8.2e-7 px on Al–Mg–Si; its masked form squares uint16 in uint16, up to 30 px wrong on integer cubes, recorded); vacuum probes within 1e-12; kernels within 1e-14 of py4DSTEM's | `tools/slice-4/pins.py real`, gate log `gate-bc41094` |
| Slice 4 break-once proof | 31 mutants on `006fe47`: 30 red, the registered survivor green (D2 needed a stronger NaN test, `537462d`) | `docs/slice-4-mutants-2026-10-15.md` |
| Slice 4 Gate B | One refuter on `bc41094`, about 20 000 cases, no wrong number on realistic inputs: traps refused (an absurd upsampling factor, empty patterns), an all-false Friedel mask refused, NumPy's keep-first rule for equal maxima, messages fixed; the masked Friedel, flat-peak and weighted-mean claims restated with their conditions | `registration-slice-4.md` §9, `006fe47` |
| 0.1.0 candidate | README standing without the suite; example executable `cubo-first-look` (`Examples/CuboFirstLook`: mean pattern, probe, BF, CoM, PNGs and numbers.json; `--synthetic` runs without data); `docs/parity.md` one row per py4DSTEM method (182 in scope: 61 done, 23 partial, 98 absent); consumer check 79 checks with 21 negative controls (`controls.sh`); CI (`.github/workflows/ci.yml`, macOS, the hook proof and `tools/test.sh`) and `.spi.yml`; the hook refuses `killall` and unpinned `pkill` | README, `docs/consumers-0.1.0.md` |
| Platform floor | Declared macOS 15 and iOS 18 from 0.2.0 (owner, 2026-10-15 13:58): library, tests and consumer checks build with no availability diagnostic, the tests pass at a 15.0 deployment target (on macOS 27), the iOS 18 Simulator builds (not run) | the floor lane's report (session scratchpad) |
| 0.2.0 generality pass | Policies as parameters with the old value as the default (OriginFit `rankTolerance`, plane fit `degeneracyTolerance`, GaussianFilter `maximumRadius`, Friedel `maximumUpsampleFactor`, median `medianWorkingBytes`; tier-2 tests); no app or suite name in Sources, test comments, README, Package.swift, the example, CI or the consumer checks (`git grep` empty; provenance in NOTICE); a DocC catalogue per module; `tools/readme-consumer` compiles and runs the README's code with a drift check; the floor declared macOS 15 / iOS 18 (owner, 13:58) | `docs/consumers-0.2.0.md` |
| Kit asks from the app's switch | CPU centre of mass parallel per position (bit identical); `RotationSolver` parallel per angle (0.767 → 0.031 s on 330 × 330, bit identical), `curve(...)`, `isCancelled`; the Debug `-Onone` remedy documented; the zero-copy resident source deferred to 0.2.1 | `d4d5917`, consumers-0.2.0 |
| CI | The first run failed on Swift 6.3.3 (macOS 26 runner): a test expression the older compiler could not type-check in time (fixed, `2bc04b3`), and the macOS 27 declaration above the runner (fixed by the 15 floor); not yet re-run | `gh run list -R mac4DSTEM/CuboKit` |
| References | `References/` is a link to the org folder's `../References/CuboKit` (2026-10-15, gitignored; the pinned py4DSTEM clone, venv and test data live there; scripts and venv shebangs resolve through the link); the real-data exports stay in `~/Library/Caches/CuboKit/real-data` | `ls -l References` |
| `CuboBragg` | py4DSTEM `find_Bragg_disks` in Double (pattern, stack, cube), `ProbeRingHint`; mac4DSTEM differences as named options | `BraggDiskTests` (9 tests), `GaussianEdgeTests`, `docs/bragg.md` |
| `CuboStrain` | CPU basis selection from maxima, indexing, fits, reference and rotated strain; three named compatibility options | `LatticeStrainTests`, `docs/strain.md` |
| `CuboEmbedding` | scikit-learn PCA (svd_flip) and Lloyd k-means with given centroids; app variants (Float32 PCA, seeded k-means, binned vector) as named options | `EmbeddingTests`, `docs/embedding.md` |
| Tests | Swift Testing (owner, 2026-10-15): 90 tests on main, 0 failures, release, real data included (CuboStrain adds LatticeStrainTests, 3 tests, debug-run green; full gate not re-stamped on this branch) (6.5 min; the Au and Al–Mg–Si slice-4 suites take 2 min each) | `tools/test.sh` |
| Gate | green and stamped at `bb0a02e`, the 0.1.0 candidate (and earlier at `bc41094`, slice 4's code) | `.build/last-green`, `tools/last-green.sh` |
| Consumer check | `tools/consumer-check`: 79 checks through the public API (slices 1 to 4), Debug, 0 warnings, at `bb0a02e`; 21 negative controls fail to build as required | `tools/consumer-check/run.sh`, `controls.sh` |
| Break-once proof | `tools/slice-1/mutants.py`, 40 mutants on the fixed commit `1b15f21`: 38 red, the 2 registered survivors green, none unexpected | `docs/slice-1-mutants-2026-10-15.md` |
| Gate B refuter | One refuter on `85318cd`: F1 (Metal tolerance on zero-mean data) and F2 (`.npy` overflow trap) fixed in `1b15f21`; F3, F4 made consistent; three behaviours recorded | `registration-slice-1.md` §9 |

Validation: nothing here is validated on a microscope. Agreement with py4DSTEM and recovery of planted cubes are what is
claimed (`registration-slice-1.md` §8).


## Hand-off

The owner pushes `main` and tags **0.3.0** (`docs/consumers-0.2.0.md` is the last per-version consumer note; from 0.3.0
the Migration paragraph in `CHANGELOG.md` replaces it); CI then runs on the push (`gh run list -R mac4DSTEM/CuboKit`).
The new modules are additive: mac4DSTEM re-pins when it chooses. Next: the remaining parity rows (`docs/parity.md`
absent and partial), full parallax reconstruction, further ptychography methods.

## Unverified

- iPadOS: declared iOS 18 and built for the Simulator (2026-10-15), not run on a device or simulator; macOS 15 tested only as a deployment target on macOS 27.
- Metal: Float32 subnormals flush to zero (recorded); the Metal CoM claim covers |S| ≥ 1e-3 · Σ|wP| only (the Float32 noise
  floor SimulatoKit measured at about 3e-10 of the total intensity is the same kind of floor). Tile buffers come from
  Metal's `new…` selectors and are freed per tile (checked against SimulatoKit's autorelease finding, 2026-10-15); not
  measured under memory pressure.
- Tiles larger than Metal's maximum buffer length, and Float32 overflow in Metal sums of huge finite values: not tested.
- `.npy` files with an unaligned data offset are refused (only hand-made headers have one).
- Slice 4: Friedel with upsampling factors of 10⁴ and above, and on detectors of 1024 pixels and more, not refuted on this
  machine (memory rule); the masked Friedel claim excludes vanishing denominators (registration-slice-4 §9).
- CI: the first run (Swift 6.3.3, macOS 26 runner) failed on the type-check and the macOS 27 declaration, both fixed; the next push re-runs it.

