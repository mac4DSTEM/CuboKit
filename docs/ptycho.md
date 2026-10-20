# CPU single-slice ptychography

Probe construction precedes the CPU engine in the API and in the regression tests.
Public Sendable numerical values only; no app types, settings, progress, provenance, UI or Metal.

`PtychographyProbe.build` supports the app's defocus, two-fold astigmatism and C30
probe terms, with py4DSTEM's sign/azimuth/aperture convention. `SingleslicePtychography`
uses full-batch complex-object gradient descent, fixed positions, periodic object
patches and the reference object amplitude constraint. Other probe aberrations,
coherence envelopes, preprocessing, position updates and the other constraints are absent.
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
