# CuboKit slice 3: per-position origins and centred virtual imaging (registration)

Registered 2026-10-15, before any code, by the CuboKit session (the owner's standing instruction for the night, relayed
by the coordinator at 01:36 CEST). Tier 1. Slices 1 and 2 are the pattern (rows and columns; Double results; the kit held
to the float64 definition, py4DSTEM's own numbers recorded beside it; registration, pins, code, mutants, one refuter).

## 1. What slice 3 makes

- **`CuboCore`: `OriginFit`**, py4DSTEM's `fit_origin` for any scan map: `.constant`, `.plane`, `.parabola`, `.bezierTwo`,
  with the robust option (`robustSteps`, `robustThreshold`). `CuboPhase`'s `.planeFit` reference keeps its closed form and
  is checked against it. The double-precision FFT of slice 2 moves from `CuboPhase` to `CuboCore` as `package`, for the
  subpixel shifts below.
- **`CuboVirtual`: `OriginMeasurement.measure(of:radius:radiusScale:tiling:)`**, py4DSTEM's `get_origin`: per position
  the Gaussian-filtered pattern's argmax, then the centre of mass of the pattern inside `|q − argmax| < r · rscale`.
  `radius` defaults to `ProbeSize.estimate` of the maximum pattern (py4DSTEM's `dp_max` default). Returns an `OriginMap`
  (`row`, `column`: `ScanImage`s; `meanRow`, `meanColumn`: NumPy's means). CPU only in slice 3.
- **`CuboVirtual`: centred, calibrated and centre-shifted virtual images**, `VirtualImaging.image(of:geometry:centeredOn:
  pixelSize:shift:path:tiling:)`: py4DSTEM's `get_virtual_image(centered, calibrated, shift_center, subpixel)`.
  `centeredOn` adds an origin (normally `OriginMap`'s mean) to the geometry's centre and bounds; `pixelSize` divides a
  geometry given in Å⁻¹ by Å⁻¹ per pixel; `shift: .none | .rounded(OriginMap) | .bilinear(OriginMap)` moves the mask at each
  position by that position's origin minus the map's mean, rounded half to even and rolled periodically (`np.roll`), or by
  py4DSTEM's bilinear interpolation of four rolls. The shifted paths run on the CPU; `.metal` with a shift is refused.
- **`CuboVirtual`: centre-shifted mean and maximum patterns**, `VirtualDiffraction.meanAndMax(of:shift:)`: each pattern
  moved by the map's mean minus its own origin before it is added, rounded and rolled, or by a Fourier shift (py4DSTEM's
  `get_shifted_ar` default). CPU only.

## 2. The oracle

py4DSTEM `f050d207` in `References/py4dstem-venv`, as before. File:line pins (`py4DSTEM/`): `process/calibration/origin.py`
`get_origin` 222 (`gaussian_filter(dp, r, mode="nearest")`, `np.unravel_index(np.argmax(...))`, `np.hypot(qxx − qx0, qyy −
qy0) < r · rscale`, `get_CoM(dp · mask)`), `fit_origin` 76; `process/fit/fit.py` `fit_2D` 38 (curve_fit on coordinates
`linspace(0, 1, n)`, the robust loop: the squared error against the previous fit, points above `mean · thresh²` dropped,
the mask cumulative), `plane`, `parabola`, `bezier_two` 127–152; `datacube/virtualimage.py` `get_virtual_image` 67 and
`get_calibrated_detector_geometry` 480; `datacube/virtualdiffraction.py` `get_virtual_diffraction` 63 (the shifted branch
from about 180); `preprocess/utils.py` `get_shifted_ar` 59; `data/calibration.py` `set_qx0` and the `_mean`/`_shift` values
(mean = `np.mean` of the map, shift = map − mean); scipy's `ndimage.gaussian_filter` (truncate 4: radius `int(4σ + 0.5)`,
the kernel `exp(−0.5/σ² · x²)` normalised by its sum, one `correlate1d` per axis, rows first, mode `nearest`).

What py4DSTEM computes in float32 here: the Gaussian-filtered pattern of a float32 cube (scipy keeps the input dtype after
each axis), the origin's total intensity (`np.sum` of a float32 array), the Fourier shift's forward FFT (NumPy 2 keeps
float32), the bilinear-shifted mask times a float32 pattern. The fits are float64 (curve_fit). As before the pins carry
py4DSTEM's own results and the float64 evaluation written in the generator; the kit is held to the latter.

## 3. The definitions

### 3.1 The Gaussian filter and the origin

σ = r. Kernel `k[t] = exp((−0.5/σ²) · t²)`, `t = −R … R`, `R = int(4σ + 0.5)`, divided by its sum. Rows then columns, each
`out[i] = Σ_t k[t] · in[clamp(i + t)]` (mode nearest: indices clamped to the edge), in Double. The argmax is the first
maximum in row-major order. The mask: `sqrt((i − i₀)² + (j − j₀)²) < r · rscale` (`np.hypot`, Double). The origin is slice
2's centre of mass of `P · mask` (row, column), the sums in Double; a zero sum is NaN.

### 3.2 Fits of a scan map (`fit_origin`)

The model's least-squares fit over the finite positions of a mask (all finite positions at first), on the scan indices
(the fitted surface does not depend on py4DSTEM's scaling of the coordinates to [0, 1]: each model space is closed under
an affine change of each coordinate): plane `{1, r, c}`, parabola `{1, r, c, r², c², rc}`, Bézier `{1, r, r²} ⊗ {1, c, c²}`
(9 functions); constant: the mean (NumPy's `np.mean`, NaN in, NaN out). Solved by Householder QR (LAPACK `dgels` through
Accelerate) on the design matrix with columns scaled to [0, 1] coordinates, in Double. Robust: `robustSteps` more rounds;
in each, the squared residual of the previous fit at every position, and every position whose squared residual exceeds
`mean(all squared residuals) · threshold²` leaves the mask for good (py4DSTEM's `mask[_mask] = False`).

### 3.3 Centred, calibrated and shifted virtual images

Geometry: with `centeredOn (r₀, c₀)` the circle and annulus centres gain `(r₀, c₀)`, the point gains it, the rectangle's
row bounds gain `r₀` and column bounds `c₀`; with `pixelSize p` every coordinate and radius is divided by `p` (py4DSTEM's
`g / unit_conversion`). **Both together are refused** (DEVIATION): py4DSTEM adds the origin in pixels and then divides
the sum by the pixel size, which scales the origin as if it were in Å⁻¹; the kit will not draw that aperture. Slice 1's
mask rules then apply. Shift: at each position the shift `(Δr, Δc) = (origin − mean)`; `.rounded`: `np.round` (half to
even) of each, the mask rolled by it periodically; `.bilinear`: `F = floor(Δ)`, `w = Δ − F`, the mask
`roll(F)·(1−w_r)(1−w_c) + roll(F_r+1, F_c)·w_r(1−w_c) + roll(F_r, F_c+1)·(1−w_r)w_c + roll(F+1)·w_r w_c`; then slice 1's sum
of pattern × mask in Double.

### 3.4 Centre-shifted mean and maximum patterns

Each pattern moved by `(mean − origin)`: `.rounded` by `np.round` and `np.roll`; `.fourier` by
`Re(IFFT2(FFT2(P) · exp(−2πi (Δr · q_r + Δc · q_c))))`, `q = fftfreq(n, 1)` per axis, in Double. Mean: the sum of the moved
patterns in scan order (compensated) divided by the scan count; maximum: the elementwise maximum seeded at −∞. **DEVIATION**:
py4DSTEM seeds the shifted maximum at 0, so a pixel negative at every position reads 0 there; the kit gives the true maximum
and the pins compare it with py4DSTEM's wherever that maximum is ≥ 0.

## 4. Claims and tolerances

| Quantity | Against the float64 definition | Against py4DSTEM's own |
|---|---|---|
| Gaussian-filtered pattern | relative 1e-12 (slice 1's floor) | recorded (float32) |
| Origin argmax | equal | equal, except exact float32 ties (recorded) |
| Origin (row, column) | 1e-9 px | recorded (float32 total), predicted ≤ 1e-4 px |
| Fits (all four, robust or not) | 1e-9 px | constant exact; the others recorded (curve_fit's convergence, slice 2 §9) |
| Centred, calibrated images; rounded and bilinear shifted images | slice 1's claims (bit for bit on integer cubes, else 1e-10) | integer cubes and rounded shifts bit for bit; bilinear recorded (float32 mask products) |
| Shifted mean | relative 1e-10 | relative 1e-10 (py4DSTEM is float64 here) |
| Shifted maximum | bit for bit | bit for bit where ≥ 0 (§3.4) |
| Fourier-shifted mean and maximum | relative 1e-10 | recorded (float32 forward FFT) |

## 5. Truth and data

- **A known per-position beam shift.** A disk of radius 4 whose centre moves by a known integer and subpixel amount per
  position on a background: the measured origins recover the planted centres (the centre of mass of a symmetric disk is its
  centre) to 1e-9 px where the shift is integer, and the rounded shifted mean pattern of the integer-shifted cube is the
  unshifted disk exactly.
- **Fits.** A plane, a parabola and a Bézier surface sampled exactly are recovered by their own model to 1e-10; one outlier
  added is removed by the robust fit (its residual exceeds the threshold) and the fit returns to the clean surface.
- **The Gaussian kernel** sums to 1 and is symmetric; a constant pattern filters to itself.
- **Centring.** A centred circle at (0, 0) equals the uncentred circle at the mean origin, bit for bit.
- Synthetic cubes: slice 1's, the planted shift cube above (float32 and uint16 versions). Real data: slice 1's five exports
  (no new files); origins with the radius from the maximum pattern, all four fits, centred BF with a shift (rounded and
  bilinear), the shifted mean and maximum (rounded and Fourier).

## 6. Mutants (planned)

The Gaussian radius `int(4σ)` without `+ 0.5`; mode wrap for nearest; columns before rows (only matters at the edges and in
float32: expected to survive on Double, recorded); argmax last instead of first; `≤` for `<` in the origin mask; rscale
ignored; the radius from the mean pattern instead of the max; a fit column dropped (parabola's `rc`); the robust mask reset
each round; the robust mean over the kept positions only; centring applied to only one coordinate; the pixel size
multiplied; the shift's sign flipped; `round` half up; the bilinear weights swapped; the Fourier sign flipped; the shifted
maximum seeded at 0; the mean divided by the kept count of the wrong axis.

## 7. Gates

1. Pins: every claim of section 4 on the synthetic cubes. 2. Truth: section 5. 3. Real data: the five cubes; predicted
origins within 1e-4 px of py4DSTEM's own, argmax equal, the fits within 1e-9 px of the float64 definitions. 4. Every new
test broken once. 5. One Gate B refuter on the frozen commit (Sonnet). 6. `tools/test.sh` green and stamped; the consumer
check extended.

## 8. What is not claimed

`get_origin`'s `mask` (a real-space subset) and `fast_center`; `get_origin_friedel` and beam-stop origins; the Metal path
for anything new here; the median pattern; a calibration object (the caller carries the map and the pixel size). Agreement
with py4DSTEM is not validation: an origin map is right when the beam is where it says, which the planted cube checks and a
real specimen does not.

## 9. Amendments

### 2026-10-15, at the first run: py4DSTEM's float32 filter moves two argmaxes on the Au cube

On the Au reference cube (205 × 310 positions) py4DSTEM's own Gaussian filter (float32, scipy keeps the input dtype) puts
the argmax one pixel away from the float64 filter's at 2 positions (16 001 and 49 321: (17, 31) against (16, 31), (37, 41)
against (36, 40)), near-ties within float32 resolution; its origin there moves by up to 0.62 px. Everywhere else, and on the
other four cubes and the synthetic ones, the argmax is the same and py4DSTEM's origin is within 7.1e-7 px of the float64
one (prediction ≤ 1e-4 px). As gate 3 said, such a flip is recorded, not widened: the kit follows the float64 definition.
Measured at the first run: the kit's origins at 0.0 px from the float64 definition on every cube; the fits, filtered
patterns, centred, calibrated, rounded and bilinear shifted images, and rounded and Fourier shifted patterns within the
section 4 bounds (integer cubes bit for bit against py4DSTEM's own centred, calibrated and rounded images).

### 2026-10-15, after the Gate B refuter on `5c82508`

One refuter (Sonnet; scripts in the session scratchpad `refuter3/`). Held: the filter against scipy within 1e-12 at the
radius boundaries and on 1 × 1 to 2 × 2 detectors; 320 random origin cases within 1e-9 px; the default radius bitwise; the
fits within 1e-9 for every model and robust step count; every image mode within 1e-10, including exact .5 ties, negative
and huge shifts; integer cubes bit for bit against py4DSTEM's own; every tiling bitwise equal; the rounded shifted mean
and max bit for bit; Fourier shifts within 2e-15 of max up to 1024². Broke:

- **F1, a registered claim: the argmax on exactly symmetric patterns.** The filter summed `k[t] · p` from −R to R; scipy's
  symmetric-kernel path (ni_filters.c `NI_Correlate1D`) adds the centre term and then `(p[i − t] + p[i + t]) · k[t]` from
  t = R down to 1, so a mirror-symmetric pattern filters to exactly tied values and the first wins. The kit's order broke the
  ties at random: a disk centred on (11.5, 12.5) gave (12, 13) instead of (11, 12); rings and point-symmetric integer
  patterns up to 11.8 px. Fixed: scipy's order, and the kernel normalised by NumPy's sum. After the fix: 0 px on every
  reproduction (0 of 60 point-symmetric patterns).
- **F2, traps where a refusal is due.** A centred `.point` with a NaN or huge origin, huge shifts (|Δ| ≈ 1e30) and a radius of
  1e300 trapped in an integer conversion. Fixed: points refuse a non-finite or huge coordinate (Python's `int(nan)` raises);
  shifts are reduced modulo the detector in Double first (any finite shift works); `GaussianFilter(sigma:)` now throws for a
  σ that is not positive and finite, whose square underflows, or whose radius exceeds 10⁶ pixels.
- **F3, rank-deficient fits.** The kit refused a plane on a 1 × N scan and a parabola or Bézier on 2 × N, where NumPy's
  lstsq (and py4DSTEM) return a surface. Fixed: a negligible column is left out of the QR, which gives the projection onto
  the design's column space, the same fitted surface. Section 3.2's refusal applies only when no finite value is left.
- **F4, NaN in the pattern.** The kit skipped masked-out pixels and NaN in the argmax; the definition multiplies the whole
  pattern by the 0/1 mask (NaN · 0 is NaN) and NumPy's argmax returns the first NaN. Fixed: both as the definition.
- **The Fourier claim is restated in the norm the FFT obeys.** On signed data whose mean pattern has pixels near zero,
  relative 1e-10 with slice 1's floor is unsatisfiable by any FFT (both sides err by about 5e-16 of the maximum); the claim
  for the Fourier-shifted mean and maximum is |Δ| ≤ 1e-12 · max|want|, which the kit meets with three orders to spare.
- Recorded: fits with offsets of 1e5 and above exceed 1e-9 px by roundoff, as the definition's own lstsq does.
