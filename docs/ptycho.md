# CPU single-slice ptychography

Probe construction precedes the CPU engine in the API and in the regression tests.
Public Sendable numerical values only; no app types, settings, progress, provenance, UI or Metal.

`PtychographyProbe.build` supports the app's defocus, two-fold astigmatism and C30
probe terms, with py4DSTEM's sign/azimuth/aperture convention. `SingleslicePtychography`
uses full-batch complex-object gradient descent, fixed positions, periodic object
patches and the reference object amplitude constraint. Other probe aberrations,
coherence envelopes, position updates and the Fourier-amplitude and aberration-fitting
constraints are absent. Preprocessing, the mixed-state and the multislice engines and the
aperture and support constraints are described in the sections below (0.5.0).
See the module's DocC Scope for the precise supported subset.

Regenerate from this worktree:

```sh
MPLCONFIGDIR="$PWD/.build/matplotlib" ../../tools/memcap.sh -- ../../References/CuboKit/py4dstem-venv/bin/python -I -B tools/ptycho/pins.py
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

Not ported: py4DSTEM has no switch for the object amplitude cap (`_object_threshold_constraint`
is unconditional for a complex object), so `constrainObjectAmplitude` of mac4DSTEM is not
ported; the cap stays unconditional. mac4DSTEM's versions of the two probe constraints
compute the same formulas in Float32 (CoM shift rounded to Float32, mask in Double); the
kit does everything in Double, which is the py4DSTEM-conformant choice, so no app variant
is offered. The other py4DSTEM probe constraints (aperture, Fourier amplitude,
aberration fitting, real-space support) remain absent.

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
divides by infinity there). Absent: the projection schemes, batching, position correction, Fourier-amplitude and
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
(sigmoid-fit) constraint, aberration fitting, position correction (`fix_positions=False`), object Gaussian/Butterworth/TV filters.

Known differences from py4DSTEM, all documented in the API comments: the engines' `errors` are normalised by the energy of
the amplitudes they are given (py4DSTEM uses the preprocessing mean intensity, which counts negative shifted values;
`PtychographyAmplitudes.meanIntensity` is that value); the probe support mask is not gated on `fixProbe` (as py4DSTEM;
the fixed-probe support case is pinned); an exactly zero mixed-state far-field pixel gives a finite error here and an infinite one in py4DSTEM.
