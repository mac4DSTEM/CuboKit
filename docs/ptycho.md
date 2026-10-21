# CPU single-slice ptychography

Probe construction precedes the CPU engine in the API and in the regression tests.
Public Sendable numerical values only; no app types, settings, progress, provenance, UI or Metal.

`PtychographyProbe.build` supports the app's defocus, two-fold astigmatism and C30
probe terms (and, from 0.6.0, the higher-order terms below), with py4DSTEM's sign/azimuth/aperture convention. `SingleslicePtychography`
uses full-batch complex-object gradient descent, fixed positions, periodic object
patches and the reference object amplitude constraint (switchable, below). Coherence envelopes and the Fourier-amplitude and aberration-fitting
constraints are absent; position correction is described in its section below (0.5.0). Preprocessing, the mixed-state and the multislice engines and the
aperture and support constraints are described in the sections below (0.5.0).
See the module's DocC Scope for the precise supported subset.

Regenerate from this worktree:

```sh
MPLCONFIGDIR="$PWD/.build/matplotlib" ../../tools/memcap.sh -- ../../References/python/py4dstem/bin/python -I -B tools/ptycho/pins.py
```

The script verifies py4DSTEM f050d207 / 0.14.19, NumPy 2.5.0, and the installed
oracle files against the pinned checkout. Actual `ComplexProbe.build` and public
`SingleslicePtychography.reconstruct` produce the pins. Only the already-numerical
preprocessing state is injected; CoM, rotation and calibrations are outside this
slice. Seed 20261016 creates the object; shuffle seed 1729. 8×8 scan, 16×16 detector,
32×32 object, three iterations, fractional/ties-to-even positions and anisotropic
sampling. Complete uncompressed fixture is about 535 KB. A 7×10 hard-aperture probe
also exercises the arbitrary-length FFT and row/column convention.

The main case uses step 0.35, normalization 0.7. Fixed-probe/local normalization
and pure-phase/global normalization cases use step 0.1. Tolerance is `2e-6` max
absolute error divided by max(1, max absolute oracle sample). Measured gaps:

- 16×16 probe: 4.421e-8; 7×10 probe: 9.536e-8.
- Three iterations, object/probe/errors: 3.577e-7.
- Fixed probe, normalization 0: 1.674e-6.
- Pure phase, normalization 1: 4.440e-8.

The engine is Double, while the Python state and frequency/translation construction
are Float32/complex64. These are fixture agreement bounds, not general error bounds.
The normalization-0 case at step 0.35 amplified roundoff to 4.781e-6; this is outside
the test's bound, and not a claim of agreement at that parameter combination.
A uniform starting object puts exact Fourier zeros outside a hard aperture; the
projected phase there is undefined and FFT roundoff differs between precisions.
The fixture therefore starts with a fixed weak random phase (documented in the
generator). No numerical correction to py4DSTEM's operator is implemented.

The red baseline returned a zero probe/unchanged object; both regression tests failed.
Independent review used the available inherited Codex model; Sonnet is not available
in this session. Review requested the extra constraint and rectangular pins, now added.

## Probe constraints (centre of mass, amplitude)

`SingleslicePtychography.reconstruct` takes four more parameters, all off by default
so every earlier call and pin is unchanged: `fixProbeCenterOfMass` (py4DSTEM
`fix_probe_com`), `constrainProbeAmplitude` with `probeAmplitudeRelativeRadius` (0.5)
and `probeAmplitudeRelativeWidth` (0.05) (`constrain_probe_amplitude`,
`..._relative_radius`, `..._relative_width`). They follow `_probe_constraints`: applied
after the probe update, only when the probe is not fixed, centre of mass first, then the
amplitude top-hat. The centre of mass is the corner-centered intensity centre
(`get_CoM`, `corner_centered=True`), removed by a periodic Fourier shift (`fft_shift`);
the top-hat is `0.5(1 - erf(sqrt(pi)/width * r/(1 - r^2)))` of `r = |q| - radius` in
cycles per pixel, followed by renormalisation to the probe's total intensity. Radius
must lie in 0...0.5 and width in (0, 0.5]; anything else throws `invalidInput`
(py4DSTEM does not check; mac4DSTEM does, with the same limits). py4DSTEM defaults
`fix_probe_com` to on; here it is off, so a caller who wants py4DSTEM's default asks for it.

Pins (`constraintCases`, same input as the main case, 3 iterations, step 0.35,
normalization 0.7, starting from the initial probe Fourier-shifted by (1.3, -0.8) px so
the constraint has work to do): centre of mass only; amplitude only (radius 0.3,
width 0.1); both; amplitude with the defaults (0.5, 0.05). Measured gaps (the same
`2e-6` max-absolute over max(1, max|oracle|) bound, object, probe and errors): 3.576e-7,
3.601e-7, 3.715e-7, 3.576e-7. Other tests: the default call equals an explicit
all-off call exactly; with `fixProbe` the constraints are ignored (as in py4DSTEM) and the
probe is returned unchanged; a Gaussian probe planted at (1.3, -0.8) px is re-centred
to |CoM| = 6.6e-15 px (bound 1e-9) with intensity kept; invalid radius or width throws.
On non-compact probes (such as the pinned 16x16 one) the periodic centre of mass is
not a fixed point of one shift, in py4DSTEM as here (0.017 px remains from 1.2 px).

Object amplitude cap: see "App options" below (the cap stays on by default, as py4DSTEM). mac4DSTEM's versions of the two probe constraints
compute the same formulas in Float32 (CoM shift rounded to Float32, mask in Double); the
kit does everything in Double, which is the py4DSTEM-conformant choice, so no app variant
is offered. The other py4DSTEM probe constraints (Fourier amplitude, aberration fitting) remain absent.

## Preprocessing (`PtychographyPreparation`, 0.5.0)

The step before the engines: from a 4D `CubeSource`, the amplitude stack, scan positions, object canvas and starting probe.
Pins: `tools/ptycho/preprocess_pins.py` calls py4DSTEM's own `_normalize_diffraction_intensities`,
`_calculate_scan_positions_in_pixels`, `_initialize_object` and `_initialize_probe` on a `SingleslicePtychography` whose
attributes are set directly (the centre-of-mass fit and the rotation solve are inputs: per-position origins and the angle).
A 5x6 scan of 12x10 patterns (non-square on purpose), Float32 cube with negative values, fractional and integer origins.
Tolerance `2e-6` (max absolute error over max(1, max|oracle|)), mean intensities `2e-6` relative, positions before
centring `1e-9` absolute, object shape exact. Measured: amplitudes 2.9e-8 (bilinear) and 4.2e-8 (spline); mean
intensity 1.6e-8; positions 0 (the Float32 rotation matrix and origin of `AffineTransform` are reproduced);
centred positions 2.7e-7; probe 4.6e-8 and 6.0e-8.

- `amplitudes(from:origins:shift:tiling:)`: `out[i, j] = pattern[i + originRow, j + originColumn]` with wrap-around, by
  `.bilinear` (`shifting_interpolation_order=1`) or `.cubicSpline` (order 3, SciPy `shift(mode="grid-wrap")`, py4DSTEM's
  default; the periodic B-spline coefficients are solved by division in Fourier space). Square root of the clipped values.
  `meanIntensity` is py4DSTEM's (sum of the shifted values before clipping); `clippedMeanIntensity` is the app's.
- `scanPositions`, `objectShape`, `centerPositions` follow `_calculate_scan_positions_in_pixels`,
  `_initialize_object` and `center_positions_in_fov`. Rows and columns are py4DSTEM's first and second axis
  (an app that names its axes the other way relabels them before the call, as its documentation of the mapping says).
  py4DSTEM's padding quirk is the default (`PtychographyPadding.reference`: both axes start at `rows/2`, both
  extents add `columns/2`); the app's per-axis padding is `.perAxis`, identical on a square detector.
- `initialProbe` builds the aperture probe and scales it so that `sum |fft2(probe)|^2` equals the mean intensity.
- `prepare` does all of it; `PtychographyPreparationOptions.mac4DSTEM(...)` is the app's choice (bilinear, per-axis
  padding, no re-centring, clipped mean). The padding and clipped mean are pinned to a transcription of the app source
  (NumPy for the mean, an analytic offset for the padding); the interpolation, rotation and probe are py4DSTEM's.
- Absent: `crop_patterns`, `positions_mask`, `region_of_interest_shape` resampling, `initial_scan_positions`,
  `positions_offset_ang`, `object_fov_ang`, vacuum probes, the centre-of-mass fit and rotation search
  (`CuboPhase.CenterOfMass`, `RotationSolver`, `CuboCore.OriginFit` are the kit's pieces for those).

## Mixed-state engine (`MixedstatePtychography`, 0.5.0)

Several mutually incoherent probe states, one complex object. Pins: `tools/ptycho/engines_pins.py` runs the real
`MixedstatePtychography.reconstruct` on injected preprocessing state (as `pins.py`): 16x16 probes, 32x32 object, 8x8
scan with fractional positions, data from a three-state forward model, three iterations, step 0.35, normalisation 0.7.
Cases: two states; three states with centre-of-mass, amplitude top-hat and orthogonalisation; no orthogonalisation; fixed
probe; pure phase. Tolerance `2e-6` on the object, the errors and the probes. Measured: object 3.6e-7 (0.3e-7 pure
phase), errors 1.2e-8, probes 1.3e-7. Orthogonalisation alone (four states with a separated spectrum): 8.8e-9, and the
kit's result is orthogonal to 2e-16 with the total intensity kept. py4DSTEM orthogonalises even when the probe is fixed;
so does the kit (pass `orthogonalizeProbes: false` to avoid it).

The per-state phase after orthogonalisation is an eigenvector convention (LAPACK's differs from the kit's Jacobi), so the
probe comparison removes one global phase per state; the object, errors and every intensity are unaffected by it.
`initialProbes(from:count:seed:)` reproduces py4DSTEM's random linear-phase starting states with a port of NumPy's
`RandomState` (MT19937, `np.random.seed(seed)` then `rand()`), pinned to the first draws and to two seeded
`_initialize_probe` runs (gap 3.3e-8). An exactly zero far-field pixel contributes `measured^2` to the error (py4DSTEM
divides by infinity there; see "Error normalisation"). Absent: the projection schemes, batching, position correction (single-slice only), Fourier-amplitude and
aberration constraints, `MixedstateMultislicePtychography`.

## Multislice engine (`MultislicePtychography`, 0.5.0)

A stack of thin complex slices with Fresnel propagation between them: `propagators` is py4DSTEM's
`exp(-i pi lambda dz (kx^2 + ky^2))` (gap 3.7e-7 against Float32 frequencies in complex64); `reconstruct` is the gradient
descent forward through the slices and back, each slice with its own probe-intensity normalisation, the probe with slice
0's object intensity. Cases: three slices (uniform and with constraints), fixed probe, pure phase, `identicalSlices`, and
two slices; tolerance `2e-6`, measured object 3.9e-7, probe 7.1e-8, errors 1.5e-8. With one slice the engine equals
`SingleslicePtychography` to 2e-16. `sliceThicknesses` has one entry per interval, in Å. Absent: tilted propagators
(`theta_x`, `theta_y`), `kz_regularization_filter`, TV denoising, `middle_focus`, the projection schemes, position correction.

## Aperture and support constraints (0.5.0)

All three engines take `fixProbeAperture` (restore the initial probe's Fourier amplitude, keep the phase and the total
intensity; state 0 only for mixed) and `probeSupportMask` (corner-centred, zero outside). Order as py4DSTEM: centre of
mass, aperture, amplitude, support (then orthogonalisation for mixed). Both off by default. Pinned on the three engines
(`extras` in `engines.json`), measured 3.9e-7 at tolerance `2e-6`. Still absent for all engines: Fourier-amplitude
(sigmoid-fit) constraint, aberration fitting, position correction on the mixed-state and multislice engines (`fix_positions=False`; the single-slice engine has it), object Gaussian/Butterworth/TV filters.

Known differences from py4DSTEM, all documented in the API comments: the probe support mask is not gated on `fixProbe` (as py4DSTEM;
the fixed-probe support case is pinned); the rest of this section's former deviations (error normalisation, the zero mixed-state pixel) are
now py4DSTEM's by option, below.

## Error normalisation (0.5.0)

py4DSTEM's `errors` are the summed squared amplitude residual of an iteration divided by `mean_intensity * positions`, where `mean_intensity`
is the preprocessing mean (the sum of the shifted values over the detector, negatives counted, averaged over positions). The engines used to
divide by the energy `sum(amplitudes^2)` of the amplitudes they were given, which is the same number only when no negative value was clipped.
All three engines now take `errorNormalization: PtychographyErrorNormalization`:

- `.meanIntensity(m)`: py4DSTEM. `PtychographyAmplitudes.errorNormalization` is `.meanIntensity` of the preprocessing, so
  `errorNormalization: prepared.amplitudes.errorNormalization` is all a caller of `PtychographyPreparation` writes. The app's clipped mean is
  `.meanIntensity(amplitudes.clippedMeanIntensity)` (it equals the energy per position).
- `.suppliedAmplitudeEnergy`: the previous behaviour, and the default. The default cannot be py4DSTEM's, because an engine given a bare
  `[Double]` has no preprocessing mean to divide by; a caller who wants py4DSTEM's `errors` passes the mean.

The pins now go through py4DSTEM's preprocessing instead of injecting an equal value: `pins.py` and `engines_pins.py` build a raw Float32 cube
(offset and noise make many pixels negative, a fractional origin), call `_normalize_diffraction_intensities`, and hand its amplitudes and mean
to the engine; the mean differs from the amplitude energy by more than 2 percent (asserted in the generators), so the old normalisation
fails the pins (a test checks it differs from py4DSTEM's errors by more than 5e-3 while leaving the object untouched). The Swift tests also
run `PtychographyPreparation.amplitudes` on the raw cube and compare amplitudes, mean and the engines' errors (measured: amplitudes
2.9e-8, mean 1e-8, errors 1.8e-7 to 2.5e-7, tolerance 2e-6). The fixed-probe, normalisation-0 single-slice variant of `ptycho.json` moved to
normalisation 0.02 (the exactly-0 endpoint is no longer pinned): on the new data the Float32 reference's phase at weakly illuminated edge pixels (update 1 / |probe|^2) made the gap 5.2e-6
at exactly 0; the port did not change.

Zero far-field pixel of the mixed-state engine: py4DSTEM sets the amplitude there to infinity (update 0, error term infinite, so the
iteration's error is infinite); `zeroFarFieldPixelError: .reference` (default) does the same, `.finite` adds `measured^2` as before. The
single-slice and multislice engines give `measured^2` there in py4DSTEM too (the phase of zero is 0). Tested on a probe of ones on ones
(an exact power-of-two transform).

## Position correction (single-slice engine, 0.5.0)

`positionCorrection: PtychographyPositionCorrection?` (nil = py4DSTEM's `fix_positions=True`) ports `_position_correction`, `positions_step_size`,
`max_position_update_distance`, `max_position_total_distance` and the centre-of-mass constraint of the positions. One iteration is one batch
of all positions (py4DSTEM's default `max_batch_size`). After the object update (computed with the probe of the iteration's start), and before the amplitude cap, each position moves by
`stepSize` times the least-squares solution of `d(I_row), d(I_column)` against `measured^2 - |F|^2` over the detector, where `F` is the far field of
the forward model of the iteration (old object) and the derivatives are `2 Re((F - F_shifted) conj(F))`, `F_shifted` the far field of the shifted probe
times the *updated* object patch read one row (column) further. py4DSTEM mixes the two objects in that difference; so does the kit (the planted-truth
test uses a frozen object to check the sign). The distances are in Å and converted with `sqrt(rowSampling^2 + columnSampling^2)`, as py4DSTEM.
`constrainCenterOfMass` is py4DSTEM's `fix_positions_com=False` (its flag is inverted in use; its default applies no constraint). The result carries
`positions`. A pattern whose 2 x 2 normal matrix is singular keeps its position (py4DSTEM's `inv` raises). Memory: the shifted probe and three
far fields per pattern are recomputed, none stored. Not provided for the mixed-state and multislice engines (py4DSTEM has the same method there,
with a sum over states "is this correct?" comment of its own).

Pins: `tools/ptycho/position_pins.py` (`positions.json`, 8 x 8 positions that start up to 0.3 px from the data's, four iterations, a smooth
probe because the one-pixel finite difference is useless for the strongly aberrated probe of the other pins): default step, step 0.2, both
limits, the centre-of-mass constraint, fixed probe. Measured: positions 6.2e-7 px, object, probe and errors 4e-7 (tolerance 2e-6). py4DSTEM's
estimate is only good near the truth: an exact-object, frozen-object case moves the mean distance to the data's positions from 0.230 to
0.124 px in one iteration (kit and py4DSTEM agree to 6e-8); from a rough object the positions drift (up to 1.5 px in four iterations), in the reference
and here alike.

## Odd and non-square sizes (0.5.0)

`tools/ptycho/odd_pins.py` (`odd.json`): a 9 x 7 detector through amplitudes (both interpolations), positions, the starting probe; a 15 x 13 probe on a
31 x 29 object through the three engines with the centre-of-mass, amplitude, aperture and support constraints, the seeded mixed states, the
propagators and `PtychographyProbe.build`, all through py4DSTEM's preprocessing. Measured 5e-8 (probe, amplitudes), 3.9e-7 (objects), 7.4e-7 (errors) at
2e-6. No defect was found in the port: the frequency grid, corner-centred patch offsets and Fourier shifts already follow `fftfreq` for odd lengths.

## App options: object amplitude toggle and higher-order probe terms (0.6.0 candidate)

What the mac4DSTEM app passes to its single-slice engine and probe builder (read from `SingleslicePtychographyOptions`,
`PtychographyProbeAberrations` and `PtychographySettings`) and the kit lacked: two things. Every other app option (iterations, step,
normalisation, fix probe, pure phase, centre of mass, amplitude top-hat and its radius and width, the memory budget, the rolloff) was already
in the kit. `tools/ptycho/options_pins.py` writes `Fixtures/options.json` from `ptycho.json`'s data (tolerance `2e-6` max absolute over
max(1, max|oracle|), as the other ptycho pins).

- **`constrainObjectAmplitude`** (`SingleslicePtychography.reconstruct`, default `true` = py4DSTEM and the old behaviour). `false` skips the
  `|object| <= 1` cap, as the app's toggle does; `purePhaseObject` still sets unit amplitude (the app's `constrain || pure`). DEVIATION from
  py4DSTEM, which applies the cap unconditionally to a complex object. The pin is py4DSTEM's own `reconstruct` with only its
  `_object_threshold_constraint` replaced by "unit amplitude if pure phase, else identity" (the app's behaviour with the toggle off); the data
  start from an object of amplitude 1.1 so the cap would matter (the pinned run keeps |object| up to 1.1, asserted; the capped kit result
  differs by more than 1e-3). Cases: toggle off, off with a fixed probe, off with pure phase. Measured 1.8e-7, 1.8e-7, 1.1e-7. Tests also:
  omitting the argument equals `true` exactly; pure phase gives the same result with either toggle.
  Single-slice only (the app has no other engine); the mixed-state and multislice engines still always cap.
- **Higher-order probe aberrations** (`PtychographyAberrationTerm`; `higherOrder:` of `PtychographyProbe.build` and
  `PtychographyPreparation.initialProbe`, `higherOrderAberrations:` of `PtychographyPreparationOptions`; empty by default, bit-identical to
  before). A term is `coefficient * alpha^(n+1)/(n+1) * cos(m phi)` or `sin(m phi)` in chi, `phi = atan2(column, row frequency)`, not negated
  (only the defocus is), which is both py4DSTEM's Cartesian aberration coefficients and the app's `HigherOrderTerm`.
  `PtychographyAberrationTerm.polar(radialOrder:angularOrder:magnitude:azimuth:)` converts py4DSTEM's `C_nm`, `phi_nm`
  (`C cos(m(phi - phi0))`) into the cosine and sine terms. Pinned to `ComplexProbe` for every symbol py4DSTEM has: C21, C23, C32, C34, C41,
  C43, C45, C50, C52, C54, C56 one by one and all together with defocus, C12 and C30, on a 20 x 16 probe with chi up to a few radians;
  measured 3.2e-8 to 1.2e-7 (ComplexProbe evaluates chi in Float32). The starting probe of py4DSTEM's `_initialize_probe` with the combined
  terms, through `PtychographyPreparation.initialProbe`: 5.7e-8. Terms py4DSTEM cannot express (other radial or angular orders such as (1,1),
  (6,0), (6,4), a lone sine component, as the app lets a user enter) are pinned to a float64 NumPy transcription of the app's
  `PtychographyProbeAberrations.chi` (DEVIATION from py4DSTEM, which stops at order 5 with its parities): 3 cases, 4.5e-10 to 4.8e-10.
  The app's cartesian astigmatism `C12a, C12b` is the kit's polar `C12 = hypot(a, b)`, `phi12 = atan2(b, a)/2`.
  A term needs radial order 1 to 64, angular order >= 0, a finite coefficient and no sine component at angular order 0, or `build` throws.
  The app's `ParallaxAberrationTerm` indexing is its own (radial order m, angular order n, component 0/1); the kit's field names say what they are.
