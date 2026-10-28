# Polar cube, ellipse fit, Fourier resample: planted truth

`planted.py` writes synthetic elliptical-ring cubes with an analytic radial background, exact and noisy ring points for an
ellipse fit, and band-limited images for Fourier resampling, each with its truth in `truth.json`. numpy only. It does not
run py4DSTEM and claims no upstream number: `pins.py` beside it adds those.

```sh
python3 -I tools/polar-ellipse/planted.py --out "$TMPDIR/cubo-planted"   # about 1 s, ~10 MB
```

Do not commit the output. If a small case is wanted as a fixture, shrink `DET`/`SCAN`
in the script and commit the `.npy` in its own commit with the reason.

## How the tests use it
- **Polar cube:** transform `cube_<case>.npy` with the case's true (r0, c0, a, b, theta) to the polar grid in `truth.json`
  (`n_q`, `n_phi`, `q_k`, `phi_j`). Compare with `polar_truth_<case>.npy`: ring positions to 0.25 px, background to rtol 1e-2
  off the rings. With the right ellipse the result is constant in phi; with a circle assumption on `tilted` it is not
  (a useful failing-before test).
- **Radial background:** `truth.json.background` is `amp * exp(-q/length) + const`. A fit or subtraction on the phi mean
  of `polar_truth` must return those three numbers; the rings are separated from the background in q (rho 1 and 1.6).
- **Ellipse fit:** feed `ring_points_<case>.npy` (exact) and `_noisy.npy` (sigma 0.05 px, seed 1) to the fit. Exact: atol
  1e-9 px and 1e-12 rad; noisy: the tolerances in `truth.json.ellipse_fit`. `circle` has no theta; `near_circle` (a-b = 0.5 px)
  has a loose theta by construction.
- **Fourier resample:** `resample_*_input.npy` to the shape of `resample_*_truth.npy`, atol 1e-10 (exact, all modes are
  below Nyquist). Even sizes only.

Conventions are in the module docstring (rows and columns; theta of the major axis from +column toward +row; ring k at
rho = s_k; circularised q = a * rho). A kit that defines the ellipse differently converts in its test wrapper and says so in
`docs/parity.md`.

`selfcheck` entries in `truth.json` are what a plain numpy implementation reaches (conic fit, FFT padding): they show the
generator and the tolerances are consistent, not that the kit passes.
