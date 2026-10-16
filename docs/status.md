# Status

What is live, one table; then the hand-off. Rules are in `CLAUDE.md`; the registration in `docs/registration-slice-1.md`;
parity with py4DSTEM in `docs/parity.md`; what mac4DSTEM changes to import the kit in `docs/consumers-0.0.1.md` to
`docs/consumers-0.1.0.md`.

## Parallax alignment hand-off (2026-10-16)

Local branch `codex/parallax-alignment`, based on `main` at `c1f0293`; no push or tag.
The first parallax alignment slice adds `CuboParallax` as a separate product, keeping the README's
DPC-focused `CuboPhase` independently usable. Public Sendable API: `ParallaxShift`,
`ParallaxAlignment.shift`, `.shifted`, `.align` and `ParallaxAlignmentResult`.
CPU Double, Accelerate/Bluestein FFT, matrix-DFT refinement (default factor 8), fixed/running references;
no app types, Metal or UI. Full reconstruction, grouping, schedules and shift-basis fitting remain future work.

- Regression fixture: `Tests/CuboParallaxTests/Fixtures/alignment.json`, four three-frame stacks
  (integer/wrapped, odd fractional, noisy running-reference, public DM4 crop), including shifts, shifted pixels,
  means and DFT patches. Relative normwise tolerance `1e-10` with unit scale floor; no bit-for-bit claim.
  Extra shift pins cover factors 3, 7, 16 and 64. The planted smooth fractional check separately allows
  `5e-4` with the same scale, reflecting the estimator's interpolation accuracy.
- Regeneration: `../tools/memcap.sh -- ../References/CuboKit/py4dstem-venv/bin/python -I -B tools/parallax/pins.py`.
  py4DSTEM `f050d207` (0.14.19), Python 3.12.4, NumPy 2.5.0, SciPy 1.18.0; the generator enforces versions,
  checks the source pin and compares installed oracle files to that checkout. Only public test data is cropped;
  the input SHA-256 and selection are in the fixture. No owner's data is copied.
- Red baseline: four tests failed against placeholders, then the direct matrix-patch test also failed with a
  zero patch (five tests, 23 issues). Final `tools/test.sh`: **104 tests, all green, no skips**, including the
  five new tests and existing real data. Log: `/private/tmp/cuboparallax-full.log`; the slowest test product took
  517.862 seconds. No new green stamp was written because the sources were uncommitted during the required gate.
- `tools/readme-consumer/run.sh`: both README Swift blocks compile warning-free in a separate package and run.
  README module map/example, DocC, `CHANGELOG.md` Unreleased and parity rows accompany the API.
- Independent pre-commit numerical/test/doc review: no correctness blockers; an obsolete internal shared-caller
  comment was removed. The available Codex review model was used because Sonnet is not exposed in this session.
- No unresolved numerical discrepancies on the pinned fixtures. Large-stack memory/performance and iOS runtime
  behavior are unverified for this slice. Numerical conventions and limits: `docs/parallax-alignment.md`.

Process limitation: the suite board was not updated because its required `ArtifactData` connector is unavailable
in this session. This repository's status and parity documents contain the hand-off; no other suite folder was edited.

## Live (2026-10-15 afternoon: slice 4, 0.1.0 tagged, the 0.2.0 candidate)

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
| Placeholder module | `CuboStrain`: a scope statement; source/reference policy question parked | its placeholder test |
| Tests | Swift Testing (owner, 2026-10-15): 90 tests, 0 failures, release, real data included (6.5 min; the Au and Al–Mg–Si slice-4 suites take 2 min each) | `tools/test.sh` |
| Gate | green and stamped at `bb0a02e`, the 0.1.0 candidate (and earlier at `bc41094`, slice 4's code) | `.build/last-green`, `tools/last-green.sh` |
| Consumer check | `tools/consumer-check`: 79 checks through the public API (slices 1 to 4), Debug, 0 warnings, at `bb0a02e`; 21 negative controls fail to build as required | `tools/consumer-check/run.sh`, `controls.sh` |
| Break-once proof | `tools/slice-1/mutants.py`, 40 mutants on the fixed commit `1b15f21`: 38 red, the 2 registered survivors green, none unexpected | `docs/slice-1-mutants-2026-10-15.md` |
| Gate B refuter | One refuter on `85318cd`: F1 (Metal tolerance on zero-mean data) and F2 (`.npy` overflow trap) fixed in `1b15f21`; F3, F4 made consistent; three behaviours recorded | `registration-slice-1.md` §9 |

Validation: nothing here is validated on a microscope. Agreement with py4DSTEM and recovery of planted cubes are what is
claimed (`registration-slice-1.md` §8).

## Hand-off

The owner pushes `main` and tags **0.2.0** on the stamped candidate (command on the board row); CI then runs on the push
(`gh run list -R mac4DSTEM/CuboKit`). mac4DSTEM re-pins to 0.2.0 when it chooses (`docs/consumers-0.2.0.md`: nothing
breaks; build the kit optimised). Next: 0.2.1 takes the zero-copy resident source (kit ask 1) first, registered; then
slice 5, Bragg disk detection and strain, registered before code; then the ptychography registration.

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

## R2 Fourier iDPC hand-back

Branch `codex/dpc-hotpixel`, based on `main` at `52dbb31` (the parallax commit now
on main), in `_worktrees/CuboKit-r2`. This branch is independent of `codex/ptycho`.
CPU `IDPC.integrate` and `integratePhysical`, explicit periodic/centered padding,
regularization, anisotropic sampling and physical calibration. `docs/idpc.md`
records the NumPy transcription oracle, fixture generation, tolerance and gaps.
Two new tests failed against the zero-output baseline (10 issues), then passed.
Full `tools/test.sh --jobs 4`: **106 tests, zero failures, zero skips**; slowest
product 202.180 s. Full log `.build/idpc-full.log`; red/green focused logs alongside.
The available Codex reviewer found no numerical/API blocker; Sonnet is unavailable.
Source fingerprint metadata was added at review. No push, tag or merge. The suite
board connector is unavailable, so this status document holds the hand-back.

## R2 hot-pixel hand-back

Second change on `codex/dpc-hotpixel`: public Sendable numerical hot-pixel APIs
in CuboCore (`HotPixelFilter.filter`, `.mask`, `.replacing`, typed input error).
The eight actual py4DSTEM fixtures pin Float32 mean samples, masks and all cleaned
pixels bit for bit. NumPy's strided/pairwise accumulation and positive-zero median
accumulator are preserved. Fixture size 30,403 bytes; measured gap and tolerance 0.
Three focused tests pass. Red baseline: 38 issues against zero/identity placeholders;
the 1×1 pairwise case separately failed the sequential implementation, and two
signed-zero median cases failed four checks before their correction.
Generator, detailed numerical conventions and fixture provenance: `docs/hot-pixels.md`.
Independent review (available Codex model; Sonnet unavailable) found no remaining
blocker after the pairwise and signed-zero fixes. Full gate is recorded below.

Cross-branch commits, fixture generation, gaps, supported scope and the parked
strain policy question are collected in `docs/r2-handback.md`. No owner approval
is implied, and no push/tag/merge was performed.

Final hot-pixel gate after the signed-zero correction: `tools/test.sh --jobs 4`
passed **109 tests**, zero failures and zero skips; slowest product 198.595 s.
Full log `.build/hotpixel-full.log`; the earlier full run is preserved separately.
All three focused hot-pixel tests also pass on the final source and eight pins.

## R2 ptychography hand-back

Local branch `codex/ptycho`, based on main `c1f0293`, in `_worktrees/CuboKit-r2`.
No push, tag or merge. CPU probe and single-slice complex-object full-batch engine;
precise scope, fixture creation, tolerance and measured gaps: `docs/ptycho.md`.
The available Codex reviewer found no operator mismatch; Sonnet was unavailable.
The board connector is unavailable, so the suite board was not updated.

Strain task parked under the explicit no-deviation rule. The read-only app source at
`origin/main` documents intensity-once weighted least squares (Python squares the
intensity), robust/MAD reference selection (Python uses plain component medians),
and cluster-consensus automatic basis selection (Python uses histogram maxima).
Open question: should CuboStrain follow Python exactly, omitting/replacing those app
algorithms, or expose the app algorithms separately with explicit deviation claims?
No owner approval is implied. No CuboStrain numbers or API were changed.

Validation: full `tools/test.sh --jobs 4` passed **102 tests**, zero failures and
zero skips (all existing real-data fixtures present). Slowest product: 292.173 s;
new numerical suite: 3 tests, approximately 0.02 s. Red baseline: the probe and
three-iteration tests failed against zero/identity placeholders. Full log is kept
in the worktree's ignored `.build/ptycho-full.log`; focused red/green logs alongside it.
