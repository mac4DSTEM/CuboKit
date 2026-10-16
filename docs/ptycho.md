# CPU single-slice ptychography

Branch `codex/ptycho`, based on `main` c1f0293. Probe construction precedes the
CPU engine in the API and in the regression tests. Public Sendable numerical
values only; no app types, settings, progress, provenance, UI or Metal.

`PtychographyProbe.build` supports the app's defocus, two-fold astigmatism and C30
probe terms, with py4DSTEM's sign/azimuth/aperture convention. `SingleslicePtychography`
uses full-batch complex-object gradient descent, fixed positions, periodic object
patches and the reference object amplitude constraint. Other probe aberrations,
coherence envelopes, preprocessing, position updates and constraints are absent.
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
