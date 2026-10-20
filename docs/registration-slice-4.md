# CuboKit slice 4: the median pattern, masked virtual diffraction, origins with a beamstop, the vacuum probe and its kernels (registration)

Registered 2026-10-15, before any code, by the CuboKit session (the coordinator's brief of 13:10 CEST: slice 4, then the
0.1.0 candidate). Tier 1. Slices 1 to 3 are the pattern (rows and columns; Double results; the kit held to the float64
definition, py4DSTEM's own numbers recorded beside it; registration, pins, code, mutants, one refuter). CPU only: no Metal
path is added in this slice. Everything is additive to 0.0.3's public API.

## 1. What slice 4 makes (all in `CuboVirtual`)

- **The median pattern and virtual diffraction over a real-space selection.** `VirtualDiffraction.median(of:tiling:)`
  (py4DSTEM `get_dp_median`) and `VirtualDiffraction.pattern(of:statistic:scan:shift:tiling:) -> DetectorImage` (py4DSTEM
  `get_virtual_diffraction(method, mask, shift_center, subpixel)`), with `DiffractionStatistic` `.mean | .max | .median`
  and `ScanSelection` `.all | .mask([Bool]) | .weights([Double])` (row-major over the scan: a boolean mask selects
  positions, weights give py4DSTEM's floating-point mask). `shift` is slice 3's `PatternShift?` (`.rounded`, `.fourier`);
  the median with a shift is refused, as in py4DSTEM. Complex masks are not ported.
- **`get_origin`'s scan mask and fast centre, and `remove_NaN`.** `OriginMeasurement.measure(of:radius:radiusScale:
  scanMask:fastCenter:tiling:)`; `OriginMap.replacingNaNWithMean()`.
- **The Friedel origin**, `FriedelOrigin.measure(of:mask:upsampleFactor:tiling:) -> OriginMap` and
  `FriedelOrigin.origin(of: DetectorImage, mask:upsampleFactor:)`: py4DSTEM `get_origin_friedel` (CPU), with the
  beamstop-tolerant masked correlation and the DFT upsampling of `upsampled_correlation`.
- **The beamstop mask**, `BeamstopMask.make(from: DetectorImage, threshold:distanceEdge:includeEdges:sigma:scaleRadial:)
  -> [Bool]`: py4DSTEM `get_beamstop_mask` on a pattern the caller passes (its mean or, for `use_max_dp`, its maximum).
  `true` under the beamstop, as py4DSTEM returns it.
- **The vacuum probe**, `VacuumProbe.measure(of:scanMask:align:zeroVacuum:detectorMask:tiling:) -> VacuumProbe`
  (`probe`, `averaged`, `estimate`): py4DSTEM `DataCube.get_vacuum_probe` with a boolean ROI, then `Probe.measure_disk`
  (which is `get_probe_size`, slice 2's `ProbeSize.estimate`, line for line) and `zero_vacuum_sinusoid`;
  `VacuumProbe.zeroVacuum(_:center:alpha:alphaMax:)` (`Probe.zero_vacuum`) and `VacuumProbe.dampVacuum(_:center:alpha:
  start:width:power:)` (`zero_vacuum_sinusoid`).
- **The probe kernels for disk detection**, `ProbeKernel.flat(_:origin:bilinear:)`, `.gaussian(_:sigma:origin:bilinear:)`,
  `.sigmoid(_:innerRadius:outerRadius:profile:origin:bilinear:)` with `SigmoidProfile` `.sineSquared | .logistic`, each
  returning the corner-centred kernel as a `DetectorImage`; `SyntheticProbe.make(radius:width:detectorRows:
  detectorColumns:)` (`Probe.generate_synthetic_probe`).

## 2. The oracle

py4DSTEM `f050d207` in `References/py4dstem-venv`, as before. File:line pins (`py4DSTEM/`):
`datacube/virtualdiffraction.py` `get_virtual_diffraction` 63 (the unshifted branches 116–163, the shifted ones 166–257),
`get_dp_median` 366; `process/calibration/origin.py` `get_origin` 222 (the mask branch 293–322, `fast_center`,
`remove_NaN` 324), `get_origin_friedel` 332; `process/utils/multicorr.py` `upsampled_correlation` 22, `dftUpsample` 145;
`datacube/datacube.py` `get_vacuum_probe` 516, `get_beamstop_mask` 1013; `braggvectors/probe.py` `generate_synthetic_probe`
165, `measure_disk` 202, `zero_vacuum` 305, `zero_vacuum_sinusoid` 340, `get_kernel` 393, `get_probe_kernel_flat` 474,
`get_probe_kernel_edge_gaussian` 518, `get_probe_kernel_edge_sigmoid` 582; `process/utils/cross_correlate.py`
`get_cross_correlation` 13, `get_shift` 48; `preprocess/utils.py` `get_shifted_ar` 59; scipy `binary_fill_holes`,
`distance_transform_edt`, `gaussian_filter`.

What py4DSTEM computes in float32 here, recorded and never asserted: the median of a float32 cube (NumPy keeps the dtype;
an even count's midpoint is a float32 mean), the masked boolean mean of a float32 cube (`np.mean` of float32), the
Friedel correlation (NumPy 2's FFT keeps float32 as complex64), the vacuum probe of a float32 cube (`get_shifted_ar` of
float32 stays float32), the mean pattern `get_beamstop_mask` reads (`get_dp_mean`, float32). Float64 in py4DSTEM itself,
so claimed against its own call: the weighted (floating-point mask) mean, max and median (float32 × float64 is float64),
everything on integer cubes (NumPy promotes to float64), the kernels and the synthetic probe of a float64 probe.

## 3. The definitions

### 3.1 Virtual diffraction over a selection

The positions in scan order (row-major). `.all`: every position. `.mask(m)`: the positions where `m` is true. For these:
mean = the sum of the patterns (Double, compensated) divided by their count; max = the elementwise maximum (NaN wins, as
`np.max`); median = per detector pixel, the middle value of the sorted values, an even count the mean `(a + b) / 2` of
the two middle values in Double, NaN if any value is NaN (`np.median`). `.weights(w)`, every position: mean = Σ P·w / Σ w
(both sums in Double, compensated), max = max over positions of P·w, median = the median of P·w (NumPy computes P·w in
float64; so does the kit). Refused, where py4DSTEM returns NaN, ∞ or raises: a selection with no position (an all-false
mask), weights summing to zero for the mean, a mask or weights of the wrong length, non-finite weights.

With a `shift` (mean and max only): each selected pattern moved as slice 3's 3.4 (`.rounded`: `np.round` and `np.roll`;
`.fourier`: the Fourier shift), then the mean (Σ moved / count, or Σ moved · w / Σ w) and the maximum (of moved, or of
moved · w). **DEVIATION, as slice 3**: the maximum is seeded at −∞; py4DSTEM seeds it at 0. `pattern(of:statistic:.mean,
scan:.all, shift: nil)` is slice 1's `mean` (the same reducer, bit for bit), and likewise `.max`.

### 3.2 Origins with a scan mask and the fast centre

`get_origin` (slice 3, 3.1) at the positions where `scanMask` is true; **NaN elsewhere**. With `fastCenter`, the origin is
the filtered argmax itself (integer row and column, as Double), no centre of mass. The radius default is unchanged (the
probe size of the whole cube's maximum pattern). `replacingNaNWithMean()`: every NaN of each map becomes NumPy's mean of
that map's non-NaN values (`remove_NaN`; NaN stays NaN if no value is finite). **DEVIATION**: py4DSTEM's mask branch
returns NumPy masked arrays whose mask becomes all-true at the first false position (`qx0.mask = True`), its `.data`
holding 0 at the excluded positions (and, with `fast_center`, later positions unmasked again); the kit returns NaN at
exactly the excluded positions. The pins compare py4DSTEM's `.data` at the selected positions.

### 3.3 The Friedel origin

Per pattern `P` (Qr × Qc), in Double. `im` = `P` at the top-left of a 2Qr × 2Qc zero array (`np.pad((0, Q))`).
Unmasked: `cc = Re(IFFT2(FFT2(im)²))`. With a mask `m` (true in the pattern, false under the beamstop): `mp` = `m` as 0/1
at the top-left, **1** in the padding (py4DSTEM pads with ones), `M = FFT2(mp)`, `term1 = Re(IFFT2(FFT2(im)²) ·
IFFT2(M²))` (the complex product, then its real part), `term2 = Re(IFFT2(FFT2(im²) · M))`, `term3 = im · mp` (py4DSTEM's
`Re(IFFT2(FFT2(im · mp)))`, the identity), `cc = (term1 − term3) / (term2 − term3)` in IEEE arithmetic. The peak
`(x, y)` = the first maximum of `cc` in row-major order (NumPy's argmax: the first NaN if any). Parabolic refinement:
`dx = (cc[x+1, y] − cc[x−1, y]) / (4 cc[x, y] − 2 cc[x+1, y] − 2 cc[x−1, y])`, `dy` likewise, IEEE division (a zero
denominator gives ±∞ or NaN, as NumPy's float64 scalars), indices periodic. **DEVIATION**: py4DSTEM wraps `x − 1 = −1` but
raises IndexError at `x + 1 = 2Qr`; the kit wraps both. Then `x ← x + dx`, `y ← y + dy`.

Upsampling, `upsampleFactor u > 2` (1 = none; 2 is refused, as py4DSTEM's assertion; below 1 refused): the shift
`s = (x, y)` rounded to the 1/u grid (`np.round` half to even of `s · u`, divided by u); `g = fix(ceil(1.5u) / 2)`;
`c = g − u · s`; `n = ceil(1.5u)`; with `C = FFT2(cc)` (Nr × Nc = 2Qr × 2Qc) the n × n real image
`U = Re(R · conj(C) · K)` where `R[a, k] = exp(−2πi/(Nr u) · (a − c₀)(f_r[k]))`, `K[k, b] = exp(−2πi/(Nc u) · f_c[k] (b −
c₁))`, `f[k] = ifftshift(arange(N))[k] − floor(N/2)` (py4DSTEM's `conj(dftUpsample(conj(C)))`, whose outer `conj` of a real
array is the identity). Its first argmax `(a, b)`; if `a` or `b` is 0 or n − 1 the refinement is (0, 0) (py4DSTEM's
`except` branch), else the parabola of `U` at `(a, b)` as above; `s ← s + ((a, b) − g + (dx, dy)) / u`.

The origin: `row = (x / 2) mod Qr`, `column = (y / 2) mod Qc`, NumPy's `np.remainder` (the sign of the divisor).

### 3.4 The beamstop mask

From a pattern `im` (Double): with `sigma > 0`, slice 3's Gaussian filter (σ, mode nearest); with `scaleRadial s`,
`im · (1 + sqrt(xa² + ya²) · s)` where `x = arange(Qr) · 2 / Qr`, `xa = x − mean(x)` (NumPy's mean), likewise `ya` over the
columns. The threshold: `sort(im)` ascending, index `round(clip(N · threshold, 0, N))` half to even; **refused** when that
index is N (py4DSTEM raises IndexError) or the pattern holds NaN. `keep = im ≥ value`; `keep = ¬fill(¬keep)`;
`keep = fill(keep)`, where `fill` is scipy's `binary_fill_holes` (a false pixel not 4-connected to the border becomes true);
with `includeEdges` the border rows and columns of `keep` are false. The result: `edt(keep) < distanceEdge`, `edt` the exact
Euclidean distance of each true pixel to the nearest false one (`sqrt` of the integer squared distance, compared in Double),
0 at a false pixel. **Refused** where `keep` has no false pixel (scipy's `edt` then measures to a virtual pixel outside the
array; a recorded quirk).

### 3.5 The vacuum probe

The selected positions (`scanMask`, default all) in scan order, `p₀ … p_{L−1}`. `probe ← p₀` (not multiplied by the
detector mask, as py4DSTEM). For `n = 1 … L − 1`: `d = pₙ · mask` (the detector mask, default 1); with `align`, the shift
`(sr, sc)` = the first argmax of `max(Re(IFFT2(FFT2(probe) · conj(FFT2(d)))), 0)` and `d ← roll(d, (sr, sc))`
(py4DSTEM's Fourier shift by an integer shift is that roll: the kit rolls); then `probe ← probe · (n − 1) / n + d / n`
(left to right, Double). **Recorded quirk**: at n = 1 the weight of `p₀` is 0, so the result is the aligned mean of
`p₁ … p_{L−1}`; `p₀` only anchors the first alignment (one position gives `p₀` itself). `estimate` = `ProbeSize.estimate`
of the averaged probe (`measure_disk`'s defaults: 0.01, 0.99, 100). With `zeroVacuum`, `dampVacuum` with py4DSTEM's
defaults (s 0.1, w 1, p 1). **DEVIATION**: py4DSTEM's 4-tuple ROI slices an all-true array and so always takes the top-left
block of the scan; the kit takes a boolean mask only. Refused: an empty selection; a probe whose estimate fails (slice 2).

`zeroVacuum`: `probe · (hypot(i − r₀, j − c₀) < alpha · alphaMax)`. `dampVacuum`: `f = clip(1 + (alpha (1 + s) − qrr) /
(alpha w), 0, 1)`, `qrr = hypot(i − r₀, j − c₀)`, `probe · sin(f)^p`. **Recorded quirk**: `sin(1) ≈ 0.841`, so py4DSTEM's
damping also scales the disk itself by 0.841; the kit keeps the definition.

### 3.6 The kernels and the synthetic probe

`SyntheticProbe`: `1 / (1 + exp(4 (sqrt((i − Qr/2)² + (j − Qc/2)²) − radius) / width))`. Kernels, from a probe (Double)
and an origin (given, or `ProbeSize.estimate(probe)`'s centre):

- shift: `get_shifted_ar(probe, −r₀, −c₀)`: the Fourier shift (slice 3) when not `bilinear`; with `bilinear`, `F =
  floor(shift)`, `w = shift − F`, the four rolls `roll(F)(1−w_r)(1−w_c) + roll(F_r+1, F_c) w_r (1−w_c) + roll(F_r, F_c+1)
  (1−w_r) w_c + roll(F+1) w_r w_c`.
- flat (default not bilinear): `probe / Σ probe`, then shifted.
- gaussian (default bilinear): `k` = shifted probe; wrapped coordinates `q = mod(arange(N) + N//2, N) − N//2` per axis,
  `qr² = q_r² + q_c²`; `σ²_q = Σ qr² k / Σ k`; `g = exp(−qr² / (2 σ²_q σ²))`; kernel `k / Σ k − g / Σ g`.
- sigmoid (default bilinear): `k` as above; `qr = hypot(q_r, q_c)`; sine²: `t = clip((qr − ri) / (ro − ri), 0, 1)`,
  `cos(π/2 · t)²`; logistic: `1 / (1 + exp((qr − (ri + ro)/2) / ((ro − ri)/4)))`; kernel `k / Σ k − s / Σ s`.

Sums are NumPy's pairwise float64 sums (`NumPySum`). **Recorded quirk**: py4DSTEM's `get_kernel(mode="sigmoid_log")`
dispatches to the sine² sigmoid; the kit's `.logistic` is `get_probe_kernel_edge_sigmoid(type="logistic")`. Refused: a
probe summing to zero, `ro ≤ ri`, a non-positive σ or width, a non-finite origin.

## 4. Claims and tolerances

D12 (2026-10-19): "bit for bit" entries in these tables are what the tests assert today. Under the numerical
testing policy in `CLAUDE.md`, numerics are held to the written tolerance against py4DSTEM, and bit for bit is only a
quick check. Readers, I/O, indices and metadata stay exact. Converting an assertion is a later code change.

"Normwise t" means `max |got − want| ≤ t · max |want|` (slice 3's restated Fourier claim). "Relative t" is slice 1's
(relative with the floor `1e-6 · max |want|`).

| Quantity | Against the float64 definition | Against py4DSTEM's own |
|---|---|---|
| Median pattern, all or a mask | bit for bit | integer cubes bit for bit; float32 cubes recorded |
| Mean over a mask | relative 1e-10; integer cubes bit for bit | integer cubes bit for bit; float32 recorded |
| Max over a mask | bit for bit | bit for bit |
| Weighted mean | relative 1e-10 | relative 1e-10 |
| Weighted max, weighted median | bit for bit | bit for bit |
| Shifted with a selection (rounded; Fourier) | slice 3's claims (mean relative 1e-10, max bit for bit; Fourier normwise 1e-12) | slice 3's (max where ≥ 0) |
| Origins with a scan mask, fast centre | slice 3's at the selected positions (1e-9 px); NaN elsewhere; the fast centre equal | `.data` at the selected positions as slice 3 (float32 recorded) |
| `replacingNaNWithMean` | bit for bit | bit for bit on the same map |
| Friedel peak (argmax) | equal, except where the top two `cc` values agree within 1e-12 · max\|cc\| (counted and recorded) | recorded (complex64) |
| Friedel origin, with or without mask and upsampling | 1e-9 px where the peak is equal | recorded, predicted ≤ 1e-3 px |
| Beamstop mask | identical | identical on integer cubes (py4DSTEM's mean is exact there); float32 recorded |
| Vacuum probe: alignment shifts | equal, except near-ties as for Friedel (recorded) | recorded |
| Vacuum probe: averaged | relative 1e-12 | integer cubes relative 1e-12 (py4DSTEM's Fourier shift noise); float32 recorded |
| Vacuum probe: radius and centre | bit for bit given the averaged probe | recorded |
| Vacuum probe after damping; zeroVacuum, dampVacuum | relative 1e-12 (`sin`, `hypot` may differ by an ulp from NumPy's) | as against the float64 definition |
| Kernels (flat Fourier; gaussian, sigmoid bilinear) and the synthetic probe | normwise 1e-12 | normwise 1e-12 (py4DSTEM is float64 here) |

## 5. Truth and data

- **Median**: a cube whose values at each detector pixel are a shuffled known sequence (odd and even scan counts): the
  median is its closed form; one huge outlier position moves the mean and not the median.
- **Selections**: a mask selecting one position gives that pattern (mean, max, median) exactly; weights all one equal the
  plain mean (relative 1e-15); one-hot weights give that pattern for the mean and `max(P, 0)` for the max.
- **Origins**: on slice 3's planted disk cube the fast centre is the planted integer centre; excluded positions are NaN and
  the selected ones equal the unmasked run bit for bit; `replacingNaNWithMean` fills with the mean of the rest.
- **Friedel**: patterns point-symmetric about a planted centre (integer and half-integer, a disk with off-centre Bragg
  disk pairs) give the centre exactly (the correlation peak is at twice the centre, the parabola is flat by symmetry); the
  same with a planted beamstop (an opaque wedge zeroed, the mask false there) with the masked form; a Gaussian at a
  fractional centre is recovered within 0.05 px with u = 16.
- **Beamstop**: a planted pattern (bright field, a dark arm reaching the edge, a dark hole inside the bright region, a
  bright speck inside the dark arm): the mask is the arm and the border dilated by `distanceEdge`, the hole filled, the
  speck removed, counted by hand.
- **Vacuum probe**: L copies of one disk rolled by known integer shifts: the aligned average is the disk at the first
  anchored position exactly; a hot pixel only in `p₀` is absent from the result (the quirk); `align: false` gives the
  plain mean of `p₁ … p_{L−1}`.
- **Kernels**: the flat kernel sums to 1, the Gaussian and sigmoid kernels to 0 (within 1e-12); a probe point-symmetric
  about an integer origin gives a flat kernel symmetric about the corner; the synthetic probe is 1/2 on the radius.
- Synthetic cubes: slice 1's and slice 3's fixtures and the planted cubes above, float32 and uint16.
- **Real data**: slice 1's five exports (no new cube files): the median pattern; mean, max and median over a boolean mask
  (the left half of the scan) and over weights (the BF image normalised to its maximum); origins with a mask and the
  fast centre; the Friedel origin unmasked on all five cubes, masked (the beamstop mask of the mean, negated) and
  upsampled (u = 16) on the particle and small cubes; the beamstop mask of each mean; the vacuum probe of the particle
  cube's first scan row and of the owner's **bullseye vacuum probe** (one new export: the 20 probe images of
  `calibrationData_bullseyeProbe.h5` `4DSTEM_experiment/data/diffractionslices/probe_template/data`, 250 × 250 × 20 uint16,
  as a 1 × 20 × 250 × 250 cube, with a manifest line in `../data/README.md`); the three kernels of each vacuum probe.

## 6. Mutants (planned)

The even median as the upper middle; NaN ignored by the median; the mask mean divided by the scan count; weights
normalised by the count; the weighted max seeded at 0; the scan mask transposed; the fast centre refined anyway; the
excluded origins 0 instead of NaN; `remove_NaN` with the mean over all values; Friedel without the square (|G|²); the
masked form with the padding at 0; the parabola's denominator sign; the peak halved before the refinement; `% Q` as
`fmod`; the upsampled grid rounding half up; the upsampled edge rule dropped; the beamstop threshold index rounded half
up; fill with 8-connectivity; the border kept with `includeEdges`; `≤` for `<` against `distanceEdge`; the vacuum
probe's first pattern weighted 1/L; the alignment shift's sign; `sin(f)` as `f`; the Gaussian kernel's σ not squared;
the sigmoid's radii swapped; the flat kernel not normalised; the bilinear weights swapped.

## 7. Gates

1. Pins (`tools/slice-4/pins.py synthetic`, before any Swift): every claim of section 4 on the synthetic cubes. 2. Truth:
section 5. 3. Real data (`pins.py real`): the five cubes and the bullseye probe. 4. Every new test broken once
(`tools/slice-4/mutants.py` on slice 1's runner). 5. One Gate B refuter on the frozen commit (Sonnet). 6. `tools/test.sh`
green and stamped; the consumer check extended; `docs/consumers-0.0.4.md`.

## 8. What is not claimed

Complex masks; the Metal path for anything here; `Probe.from_vacuum_data` (binary opening, dilation, the cosine edge);
`get_vacuum_probe`'s 4-tuple ROI; the GPU devices of `get_origin_friedel`; Bragg disk detection itself (the kernels are
its input; detection is a later slice). Agreement with py4DSTEM is not validation: a Friedel origin is right when the
pattern is centrosymmetric about the beam, which the planted patterns check and a real specimen with dynamical
scattering does not.

## 9. Amendments

### 2026-10-15, at the first run

- **A self-check restated normwise.** The generator's check that the vacuum probe written from 3.5 equals py4DSTEM's
  algorithm in float64 used slice 1's relative distance; py4DSTEM's Fourier shift (an integer shift done in Fourier space)
  leaves about 1e-14 of noise on pixels near zero, which that floor turns into 4e-10. The check, and section 4's claim for
  the averaged probe against py4DSTEM's own on integer cubes, are normwise 1e-12 (measured: at most 3.3e-15).
- **py4DSTEM's masked Friedel correlation squares integer patterns in their own dtype.** `im ** 2` of a uint16 pattern
  wraps above 255, so on integer cubes py4DSTEM's own masked origins are wrong by up to 30 px (the simulated Au cube),
  0.05 px (counts16), 3e-3 to 1e-2 px (the bullseye probe, py4DSTEM's small cube); its unmasked ones are exact. Recorded:
  the kit follows the float64 definition (section 4's prediction of 1e-3 px against py4DSTEM's own does not hold there).
- **The masked Friedel truth is the peak, not the parabola.** Behind the planted beamstop the masked form puts the peak at
  exactly twice the centre, but its parabola is not symmetric (the masked normalisation is not centrosymmetric): the
  origin is 0.033 px from the planted centre, against 1.15 px for the unmasked form on the same patterns. Section 5's
  claim for the masked case is the peak and 0.04 px; the unmasked symmetric patterns are recovered within 1e-9 px.
- **The beamstop refusal occurs on the planted cubes.** With `includeEdges: false` and threshold 0.3 the cleaned mask of
  the planted disk cubes has no false pixel; the kit refuses (3.4) and the pins record py4DSTEM's result beside it.
- **Measured** (the kit against the float64 definition; py4DSTEM's own beside it): medians bit for bit on every cube
  (py4DSTEM's own identical on integer cubes, 6e-8 relative on the float32 ones); masked means bit for bit on integer
  cubes and within 1e-10 elsewhere (py4DSTEM's float32 means 1e-5 relative on the owner's cubes); weighted means within
  1e-10 of py4DSTEM's own (measured 1.1e-14), weighted max and median bit for bit with it; origins with a mask 0.0 px,
  the fast centre equal (py4DSTEM's float32 filter flips 2 of 63 550 Au argmaxes, as in slice 3); the beamstop masks
  identical, py4DSTEM's own identical on all six inputs; Friedel origins within 2e-14 px, no near-tie on any input
  (py4DSTEM's complex64 origins 8.2e-7 px on the Al–Mg–Si cube, 5.5e-6 px on Au, exact on integer cubes unmasked);
  vacuum probes within 1e-12, py4DSTEM's own 4e-10 to 1.4e-9 normwise on float32 cubes and below 3.3e-15 on integer
  ones; the kernels within 1e-14 normwise of py4DSTEM's (all float64).

### 2026-10-15, after the Gate B refuter on `bc41094`

One refuter (Sonnet; scripts in the session scratchpad `refuter4/`, about 20 000 cases). No wrong number on realistic
inputs. Held: Friedel origins within 7e-15 px (u = 1) and 2.2e-13 px (u = 64) on random patterns of 1 × 1 to 257 × 255
(Bluestein lengths included), 9.6e-12 px on Bragg-like 257 × 255; beamstop masks identical to scipy in 16 812 cases and
574 distance-transform stress cases (512 × 512 down to 1 × 1, distances exactly √2, √5, √13, √50); selections, medians and
every tiling bit for bit in value; shifted selections normwise 1.2e-15; origins with masks as slice 3; vacuum probes,
estimates and damping exact or within 2.9e-16; kernels within 6.2e-15 of the float64 definition for origins up to ±1e7.
Broke:

- **F1, the masked Friedel claim on ill-posed masks.** Where `term2 − term3` vanishes in exact arithmetic (an all-false
  mask; a mask keeping a few percent of a pattern with exact zeros), `cc` is roundoff over roundoff and two FFTs pick
  different peaks with a large gap. Fixed in part: an all-false mask is refused. Restated: the masked claim covers peaks
  where `|term2 − term3| ≥ 1e-9 · max |term2|`; elsewhere the definition is ill-posed and nothing is claimed. Measured:
  realistic beamstop masks (70 % kept or more) and dense patterns at any mask fraction are clean.
- **F2, flat-topped correlations.** The parabola's denominator can be 1e-8 of `max |cc|` while the top-two gap is 3e-9
  (an engineered pattern): 5.3e-9 px. Restated: the 1e-9 px claim holds where `|4 c₀ − 2 c₊ − 2 c₋| ≥ 1e-6 · max |cc|`.
- **F3, the weighted mean against `np.sum`.** With cancelling weights (Σ|w| / |Σw| = 1e6 to 1e12) the kit differs from
  NumPy's sum by 1e-9 to 1e-3 relative and equals `math.fsum` exactly: NumPy's pairwise sum is the inaccurate side.
  Restated: relative 1e-10 against the float64 definition where Σ|w| / |Σw| ≤ 1e4; exact against a correctly rounded sum.
- **F4, an absurd upsampling factor trapped** (`u = Int.max`) or exhausted memory (`u = 2⁴⁰`). Fixed: factors above 1024
  are refused (py4DSTEM has no bound).
- **F5, an empty pattern or probe trapped** (`FFT2D` of 0, a roll modulo 0) in `FriedelOrigin.origin`, `ProbeKernel.gaussian`
  and `.sigmoid`. Fixed: refused.
- **F6, messages and refusals.** The Gaussian and logistic kernels reported "the probe sums to …" when their own
  normalisation was NaN (σ_q² = 0, an underflowing width); fixed: the message names it. Recorded refusals beyond 3.4:
  `BeamstopMask` with `sigma < 0` (py4DSTEM ignores it) and `sigma` below 1.5e-154 (slice 3's filter refuses it; py4DSTEM
  returns the pattern unchanged).
- **F7, the sign of zero.** Of equal maxima (±0) the kit kept the later one; NumPy's `np.max` and `np.maximum` keep the
  first. Fixed in slice 4's reducers. Recorded: slice 1's `.all` maximum (vDSP) is unchanged, and the median's choice
  between −0 and +0 follows the sort, so "bit for bit" for medians and slice 1's maximum means equal in value.
- **F8, F9, wording.** The vacuum probe's averaged claim has the same near-tie exception as the alignment shifts; origins
  are compared by circular distance (an origin within roundoff of 0 may read as Q); `replacingNaNWithMean` keeps ±∞ as
  NumPy does. The registered wrap of the parabola's neighbours is unreachable without a mask (`cc` is 0 on the last row
  and column) and occurs only in F1's ill-posed cases.
- **Not refuted on this machine** (the suite's memory rule): upsampling factors between 10⁴ and 10⁵ and Friedel on
  detectors of 1024 pixels and more. The pins of the two big cubes (Al–Mg–Si, Au) peaked above 8 GB (float64 copies) and
  ran before the rule (11:20 to 11:50 CEST, uncapped, two workers at once); they regenerate through the suite's
  `tools/memcap.sh --gb 16` with `CUBOKIT_PINS_MEM_GB=16`, one worker at a time, or with the float64 copy avoided.
