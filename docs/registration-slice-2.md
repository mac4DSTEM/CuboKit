# CuboKit slice 2: centre of mass, probe size, the R–Q rotation and DPC (registration)

Registered 2026-10-15, before any code, by the CuboKit session (the owner's instruction at 00:55 CEST: continue with slice 2
in the same session). Tier 1: every number here is one a scientist reads (a centre-of-mass shift, a probe radius, a rotation
angle, a projected phase). Fixed before the code; a change after code starts is a dated amendment in section 9. Slice 1's
registration is the pattern; its conventions hold here (rows and columns, never x and y; Double results; the kit held to the
float64 definition, py4DSTEM's own float32 numbers recorded beside it; CPU first, Metal with compensated sums).

## 1. What slice 2 makes, and where

- **`CuboVirtual`** (reductions over a cube or a pattern) gains:
  - `CenterOfMass.measure(of:mask:path:tiling:progress:) -> CenterOfMassMaps`: per scan position the intensity-weighted
    centre of mass in detector pixels, `row` (py4DSTEM's `com_x`, first detector axis) and `column` (`com_y`), and the
    `intensity` sum, each a `ScanImage`. `mask` is an optional `DetectorGeometry` (any of slice 1's, `.mask` for py4DSTEM's
    `dp_mask` array). `.cpu` and `.metal`.
  - `ProbeSize.estimate(_ pattern: DetectorImage, lowerThreshold: 0.01, upperThreshold: 0.99, count: 100) -> ProbeEstimate`
    (`radius`, `centerRow`, `centerColumn`, the selected threshold): py4DSTEM's `get_probe_size`.
- **`CuboPhase`**, a new module and product (phase from 4D-STEM: DPC now; py4DSTEM groups DPC, parallax and ptychography
  under `process/phase`, and the shared parts land here; `CuboPtycho` stays the placeholder for ptychography). Depends on
  `CuboCore` only; it takes centre-of-mass maps, not cubes, so it can be fed by any measurement.
  - `ScanVectorField` (`scanRows`, `scanColumns`, `row: [Double]`, `column: [Double]`, row-major): a 2-vector per position.
  - `CenterOfMassReference.normalize(_ measured: ScanVectorField, reference:, reciprocalSampling:)`: the shift relative to
    a reference, `.planeFit` (py4DSTEM's default `fit_function="plane"`), `.constant` (the mean, `"constant"`), `.none`, or
    `.given(ScanVectorField)` (`force_com_shifts`; per-position origins measured elsewhere); returns the fitted reference and
    the normalized shift, `(measured − fitted) · reciprocalSampling`, NaN replaced by 0 (`np.nan_to_num`).
  - `RotationSolver.solve(_ normalized:, anglesDegrees: −89 … 89 step 1, criterion: .minimizeCurl / .maximizeDivergence,
    forceRotationDegrees:, forceTranspose:) -> RotationSolution` (`degrees`, `radians`, `transposed`, the curve or curves,
    and the corrected field): py4DSTEM's `_solve_for_center_of_mass_relative_rotation`.
  - `DPC.reconstruct(_ corrected:, scanSampling: (1, 1), paddingFactor: 2, maxIterations: 64, stepSize: 0.5,
    stoppingCriterion: 1e-6, backtrack: true) -> DPCReconstruction` (`phase`, the per-iteration `errors`, the final step
    size): py4DSTEM's `DPC.reconstruct` with its default constraints (no Gaussian, no Butterworth filter). CPU only in
    slice 2 (Accelerate's double-precision FFT, Bluestein for lengths that are not powers of two); a Metal path is a later
    slice with its own registered tolerance.

Public API only, no app types, Swift Testing, every message free of app names.

## 2. The oracle

py4DSTEM `f050d207` (0.14.19) in `References/py4dstem-venv`, as slice 1. File:line pins (`py4DSTEM/`):
- `process/phase/phase_base_class.py`: `_calculate_intensities_center_of_mass` 613 (the per-position loop: `masked_intensity
  * kxa` summed over the detector, divided by the summed intensity; kxa the first-axis index; all float32), the
  normalization at its end (`nan_to_num(measured − fitted) · reciprocal_sampling`), `_solve_for_center_of_mass_relative_
  rotation` 769 (angles −89 … 89 step 1 as float32; curl `mean|∂y/∂x − ∂x/∂y|` from centred differences on the interior;
  divergence; the four force branches; the transpose rule; ties to the non-transposed and to the first index).
- `process/phase/dpc.py`: `preprocess` 230 (padding `round(scan · 2)`, `fftfreq` of the padded grid, `k_den = kx² + ky²`
  with `[0, 0] = ∞`, `kx_op = −0.25 i kx / k_den`), `_forward` 398, `_adjoint` 463, `_update` 499, `reconstruct` 645
  (64 iterations, step 0.5, backtracking with halving, stop below 1e-6).
- `process/calibration/origin.py`: `fit_origin` 76 (plane by `fit_2D` on the finite points); `process/fit/fit.py`: `fit_2D`
  38 (`scipy.optimize.curve_fit` of `mx·x + my·y + b` on coordinates scaled to [0, 1]), `plane` 127.
- `process/calibration/probe.py`: `get_probe_size` 6 (thresholds `linspace(0.01, 0.99, 100)` of the pattern's max,
  `r = sqrt(count / π)`, `np.gradient`, the trusted range `dr ≤ 0 and dr ≥ 2·median(dr)`, the mean radius and threshold,
  the centre of mass above that threshold); `process/utils/utils.py`: `get_CoM` 166.

What py4DSTEM computes in float32 here: the centre of mass (intensities cast to float32), the rotation search, the whole
DPC reconstruction (float32 phase, complex64 FFTs). The plane fit is float64 (curve_fit). The probe size is float64 on a
float64 pattern. As in slice 1, the pins carry py4DSTEM's own numbers and a **float64 evaluation of the same definitions**
written in the generator (NumPy, float64 throughout, the same algorithm step for step; the plane by `np.linalg.lstsq`); the
kit is held to the float64 evaluation and the distances to py4DSTEM's own are recorded.

## 3. The definitions

### 3.1 Centre of mass

For the pattern `P` at a scan position and the weights `w` (1 everywhere without a mask): `S = Σᵢⱼ w P`,
`row = Σᵢⱼ i · w P / S`, `column = Σᵢⱼ j · w P / S`, `i` the detector row, `j` the column, the sums over the whole detector.
`S = 0` gives NaN (py4DSTEM's 0/0), never a substituted value. Negative intensities are summed as they are (py4DSTEM does
not clamp; mac4DSTEM's kernel clamps at 0, a DEVIATION the consumer notes name). CPU: per position, the three sums over the
detector in Double with Neumaier compensation (the products `i · w · P` are exact in Double for Float32 data, weights and
indices below 2²⁴), then two divisions. Metal: one thread per position, the three sums by the TwoSum/TwoProduct scheme of
slice 1's Gate B fix, each returned as a (sum, correction) pair added in Double on the host, then the divisions in Double.

### 3.2 The reference and the normalized shift

`.planeFit`: for each component separately, the least-squares plane `a + b·r + c·c'` over the finite positions (`r`, `c'`
the scan row and column), solved in closed form from the normal equations on centred coordinates in Double; the fitted
plane evaluated at every position. (py4DSTEM's `curve_fit` minimises the same sum of squares on coordinates scaled to
[0, 1]; the plane does not depend on that scaling, so the closed form is the definition and `curve_fit`'s result is held to
it within its own convergence.) `.constant`: the mean of each component over all positions (py4DSTEM's `np.mean`, which
includes NaN: a NaN map gives NaN, and the normalization's `nan_to_num` then gives 0). `.none`: zero. `.given`: the caller's
field. Normalized shift: `(measured − fitted)` per component, NaN → 0, times `reciprocalSampling` (default 1: pixels).

### 3.3 The rotation

For an angle θ: not transposed, `x' = cos θ · x − sin θ · y`, `y' = sin θ · x + cos θ · y`; transposed, `x' = cos θ · y −
sin θ · x`, `y' = sin θ · y + cos θ · x` (x the row component, y the column component, py4DSTEM's names). Curl:
`mean |(y'[r+1, c] − y'[r−1, c]) − (x'[r, c+1] − x'[r, c−1])|` over the interior positions; divergence:
`mean |(x'[r+1, c] − x'[r−1, c]) + (y'[r, c+1] − y'[r, c−1])|`. Both evaluated in Double for the angles given (default
−89 … 89 step 1, θ = deg · π/180). Selection exactly as py4DSTEM: with neither forced, minimise curl over each orientation
(the first index on a tie) and take the transposed one only if its minimum is strictly smaller (divergence: maximise, the
transposed one only if strictly larger); a forced rotation with an unknown transpose compares the two at that angle
(transposed if strictly better); a forced transpose searches that orientation only; both forced use both. The corrected
field is the normalized field rotated (and transposed) at the solution. The scan needs at least 3 × 3 positions; fewer is
refused.

### 3.4 DPC

py4DSTEM's iterative reconstruction, literally, in Double: the padded grid `(round(R · p), round(C · p))` (round half to
even); `kx = fftfreq(Rp, d = dr)`, `ky = fftfreq(Cp, d = dc)`; `k_den = 1/(kx² + ky²)` with `1/∞ = 0` at the origin;
`kx_op = −0.25 i · kx · k_den`, `ky_op` likewise. Each iteration: stop if the step is below the stopping criterion;
`dx = (φ[r−1, c] − φ[r+1, c]) / (2 dr)` and `dy = (φ[r, c−1] − φ[r, c+1]) / (2 dc)` with periodic indices on the padded grid;
on the scan region `dx += x`, `dy += y` (the corrected field), elsewhere 0; `error = mean(dx² + dy²) over the scan region /
mean(x² + y²)`; if the error exceeds the previous one and backtracking is on, halve the step and continue; otherwise keep
the error, `update = Re(IFFT2(FFT2(dx) · kx_op + FFT2(dy) · ky_op))` (IFFT normalised by 1/(Rp · Cp)), `φ += step · update`.
The phase is the scan region of φ. The initial φ is zero; the initial error is +∞. An all-zero field is refused (its error
is 0/0).

### 3.5 Probe size

On a Double pattern (the kit's mean or max `DetectorImage`): thresholds `t_k = lower + k · (upper − lower)/(N − 1)`,
`k = 0 … N − 1`, the last set to `upper` exactly (NumPy's `linspace`); `M = max(P)`; for each `t_k` the count of pixels with
`P > M · t_k` and `r_k = sqrt(count / π)`; `dr = np.gradient(r)` (one-sided at the ends, central inside); the trusted set
`dr_k ≤ 0 and dr_k ≥ 2 · median(dr)`; `radius = mean(r over the set)`, `t = mean(t over the set)`; the centre is the centre
of mass of `P · [P > M · t]` (`get_CoM`: row then column). An empty trusted set (NumPy: NaN with a warning) is refused
(`probeNotFound`), a DEVIATION in the refusing direction.

## 4. Claims and tolerances

D12 (2026-10-19): "bit for bit" entries in these tables are what the tests assert today. Under the numerical
testing policy in `CLAUDE.md`, numerics are held to the written tolerance against py4DSTEM, and bit for bit is only a
quick check. Readers, I/O, indices and metadata stay exact. Converting an assertion is a later code change.

| Quantity | Against the float64 definition | Against py4DSTEM's own | Metal against CPU |
|---|---|---|---|
| CoM sums (intensity, row and column moments) | integer cubes bit for bit; float cubes relative 1e-10 (slice 1's floor) | recorded | relative 1e-10 |
| CoM row and column | integer cubes bit for bit; float cubes 1e-9 px where `|S| ≥ 1e-6 · Σ|wP|`, the rest counted and not claimed | recorded (float32) | 1e-9 px, the same condition |
| Plane fit, normalized shift | 1e-10 px (lstsq against the closed form) | 1e-6 px (`curve_fit`'s convergence) | — |
| Rotation: chosen degrees and transpose | equal | equal | — |
| Rotation: the curve | relative 1e-10 | relative 1e-4 (float32) | — |
| DPC: errors per iteration, the backtracking sequence | relative 1e-9; the same sequence | recorded | — |
| DPC: phase | 1e-8 · max\|φ\| | recorded, predicted ≤ 1e-3 · max\|φ\| | — |
| Probe size: radius, centre | relative 1e-12 | relative 1e-12 (py4DSTEM is float64 here) | — |

The CoM condition exists because a dark-subtracted pattern can sum to nearly zero: there the quotient is ill-conditioned
and both the kit's and NumPy's sums are dominated by rounding; such positions are counted in the test log and are not part
of the claim. A difference beyond a bound is a failure of the port, never a tolerance to widen.

## 5. Truth and data

### 5.1 Planted truth (independent of py4DSTEM)

- **A known beam shift.** A cube whose pattern at each position is two pixels, weight `a` at `(i₀, j₀)` and `b` at
  `(i₀ + 1, j₀ + 1)`, with `i₀`, `j₀`, `a`, `b` varying by position: `row = i₀ + b/(a + b)`, `column = j₀ + b/(a + b)`
  exactly; and a disk translated by integer pixels per position: the centre of mass moves by exactly that translation.
- **A uniform disk** of radius `R` (value 1000, background 1) centred on a pixel and on a half-pixel: the probe radius is
  `sqrt(n/π)` with `n` the disk's pixel count and the centre is the disk's centre, exactly.
- **A known phase gradient.** `φ(r, c)` a smooth bump inside the scan, zero near the edges; the field `x = (φ[r+1] −
  φ[r−1])/2`, `y` likewise (the gradient py4DSTEM's forward operator inverts), rotated by `−θ` and optionally transposed,
  plus a plane: the plane fit removes the plane to 1e-10; the solver returns θ (23° not transposed, −61° transposed) and
  the transpose; the reconstruction's error falls (non-increasing over accepted iterations) and its phase correlates with
  φ (Pearson ≥ 0.99, measured by the generator before any Swift and recorded).
- The curl of an exact gradient field is 0 at θ = 0 to rounding; the divergence criterion picks the same θ on that field.

### 5.2 Synthetic cubes and fields (committed, written by the generator, seed 20261015)

Slice 1's seven cubes (centre of mass with and without slice 1's masks, probe size on their mean and max where the trusted
set is non-empty) and the planted cubes and fields of 5.1, small enough to commit.

### 5.3 Real data

The five real cubes of slice 1 (exports in `$CUBOKIT_REAL_DATA`, already in `../data/README.md`): centre of mass with no
mask and with slice 1's bright-field circle, Metal and CPU; probe size on the mean and max patterns; py4DSTEM's whole DPC
(`DPC(datacube).preprocess()` with its defaults, then `reconstruct()` with its defaults) against the kit's chain from the
same centre of mass. Arrays as `.npy` beside the slice 1 exports.

## 6. Mutants (planned, each new test broken once)

CoM: the index of the row used for the column (transpose); the sums without the mask; the sum of `P` used for the moment;
compensation dropped (the real-data condition); Metal correction dropped. Plane: the constant term dropped; the fit over
all points ignoring non-finite ones (a NaN planted); the row and column swapped in the plane. Rotation: the sign of sin
flipped; the transposed formula swapped with the plain one; the tie rule `<` for `≤`; the curl's second difference
transposed; the range including +90. DPC: the 0.25 dropped; `kx_op` with `+i`; the padding factor ignored; backtracking
without halving; the error without the denominator; the IFFT unnormalized; the roll direction reversed. Probe size: `≥`
for `>`; the gradient without the one-sided ends; the median replaced by the mean; the centre from the unmasked pattern;
`linspace` with an inexact last element.

## 7. Gates and predicted outcomes

1. Pins: every claim of section 4 on every synthetic cube and field of 5.2.
2. Truth: every closed form of 5.1 exact (CoM to 1e-12 px in Double), the rotation recovered, the correlation ≥ 0.99.
3. Real data: the claims on the five cubes. Predicted: CoM bit for bit on the integer cubes; py4DSTEM's own float32 CoM
   within 1e-4 px of float64; the same rotation angle and transpose as py4DSTEM's own on every cube (a near tie that flips
   is recorded, never widened); DPC phase within 1e-8 · max|φ| of the float64 chain.
4. Every new test broken once by the mutants of section 6, on exported copies.
5. One Gate B refuter on the frozen commit (Sonnet), told to break the port on fields and cubes the fixtures do not hold.
6. `tools/test.sh` green with the stamp; the consumer check extended to the new public API.

## 8. What is not claimed

No per-position origin measurement (py4DSTEM `get_origin`'s Gaussian-filtered argmax and masked CoM): it is calibration
and comes with a calibration slice; slice 2 takes per-position origins as `.given`. No DPC constraints (Gaussian,
Butterworth), no `maximize_divergence` by default, no physical units beyond the two samplings the caller passes, no
iDPC by direct Fourier integration (mac4DSTEM's), no Metal DPC. Agreement with py4DSTEM is not validation: whether a
reconstructed phase is the specimen's projected potential needs a calibrated rotation and a known specimen, the app's to
register.

## 9. Amendments

### 2026-10-15, at the first pins and test run: three readings settled

1. **py4DSTEM keeps the curl curve local.** Its rotation solver stores only the divergence curves on the object
   (`_rotation_div`, `_rotation_div_transpose`), not `_rotation_curl`, so "the curve against py4DSTEM's own, relative
   1e-4" (section 4) cannot be pinned. The curve is held to the float64 evaluation (1e-10); the chosen angle and transpose
   stay asserted against py4DSTEM's own choice, and agree on every cube and field.
2. **py4DSTEM's plane is `curve_fit`'s, within its stopping tolerance.** The first generator compared py4DSTEM's fit of its
   own float32 centre of mass with the least-squares fit of the float64 one, two inputs. On the same input (py4DSTEM's own
   centre of mass) `curve_fit` stays up to 8.3e-6 px from the exact least-squares plane (`py4dstem_small`; 1.9e-6 to
   3.9e-6 px on the owner's cubes). The registered 1e-6 px was a wrong prediction about the oracle's solver, not a property
   of the port: the distance is recorded, not asserted, and the kit is held to the least-squares plane at 1e-10, which it
   meets at 1e-14.
3. **The probe-size refusal covers a 0/0 centre too.** On the all-negative cube py4DSTEM returns a radius of 0 and a NaN
   centre (the thresholded pattern sums to zero); the kit refuses that as well as an empty trusted set (section 3.5).

Measured at the first run (log `test-s2-first`, 2026-10-15): the kit's centre of mass equals the float64 definition at
0.0 px on every cube (no ill-conditioned position in any of them); Metal at most 1.9e-10 px from the CPU; the rotation and
transpose equal py4DSTEM's on all five real cubes, four synthetic cubes and three planted fields; the DPC phase at most
3.3e-12 of max from the float64 chain, the backtracking sequences equal; py4DSTEM's own float32 phase 5e-7 to 5.2e-5 of max
from the float64 chain (predicted ≤ 1e-3); the probe size bit for bit on every pattern.

### 2026-10-15, after the Gate B refuter on `84815c9`

One refuter (Sonnet; scripts in the session scratchpad `refuter2/`). Held: the CPU centre of mass exact against rational
arithmetic on every dtype, shape and mask tried; the probe size bit-identical to `get_probe_size` on 928 patterns (NaN,
±∞, ties, non-default thresholds, 1 × 1 to 1024²); the plane fit at 1e-14 of `lstsq`; DPC within about 1e-16 of max of the
float64 reference on 100+ cases (Bluestein lengths up to 2039, rounding paddings, anisotropic sampling, steps 0.01 to
1000) and within about 2e-7 of max of py4DSTEM's own, the backtracking identical. Broke:

- **F1, the Metal centre of mass (section 4's Metal column).** The first kernel compensated each product but accumulated
  the corrections in plain Float32: on detectors of 700² and up the row moment passes 3e10 and the corrections lost bits
  (1024² uint16: 3.9e-7 relative, 2e-4 px), and on dark-subtracted data the error (about 1e-12 · Σ|wP|) exceeded 1e-9 px
  inside the CPU's condition |S| ≥ 1e-6 · Σ|wP|. Fixed: every term goes into a renormalised double-float accumulator (about
  48 bits; integer sums exact to 2^48). Measured on the refuter's batteries after the fix: integer cubes bit for bit up to
  1024² and 2048 × 512; float data at most 1.3e-12 · Σ|wP| on the moments; the centre of mass within 4.3e-10 px where
  |S| ≥ 1e-3 · Σ|wP|, but 1.8e-9 px at 5e-4 and 5.3e-7 px at 2e-6. Float32 hardware cannot meet 1e-9 px at the CPU's
  condition, so **the Metal claim's condition is |S| ≥ 1e-3 · Σ|wP|**; positions between 1e-6 and 1e-3 are claimed on the
  CPU path only, and `.cpu` is the path for strongly dark-subtracted data. The sums' claim (relative 1e-10 with slice 1's
  floor) holds on every case after the fix.
- **F2, py4DSTEM's float32 solver flips exact ties.** On noise-free gradients rotated by k + 0.5 degrees the two
  neighbouring curve values differ by about 6e-9 relative, below float32 resolution; py4DSTEM chose the other degree in 46
  of 96, the kit agreed with the float64 evaluation in 96 of 96. Section 4's "equal to py4DSTEM's own" holds except on such
  exact ties, which gate 3 already said are recorded, never widened.
- **F3, degrees to radians.** The kit computed deg · π / 180; NumPy's `deg2rad` and Python's `math.radians` compute
  deg · (π / 180), an ulp apart for 38 of the 179 default angles, which flipped a choice on constructed exact ties. Fixed.
- **F4–F6 (not claimed; behaviour made explicit):** a non-finite field is refused by the rotation solver (NumPy's argmin
  returns the first NaN); Metal now keeps ±∞ sums instead of NaN; Metal flushes Float32 subnormals (recorded); the plane fit
  refuses scans with fewer than three non-collinear finite positions (1 × N, all NaN), where `lstsq` returns a minimum-norm
  plane: a DEVIATION in the refusing direction; a field whose squares underflow (about 1e-170) is refused as zero.

Regression tests: `Slice2RefuterRegressionTests` (a 1024² integer cube bit for bit on Metal; dark-subtracted data under the
Metal condition) and `testRadiansAndNonFiniteFields`.
