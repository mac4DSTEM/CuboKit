# CuboKit slice 1: the 4D cube model, the mean pattern and virtual imaging (registration)

Registered 2026-10-14 night, before any code, by the CuboKit session (the coordinator's side session; the owner's
direction in the suite `ROADMAP.md`, "Kits", phase 5: "CuboKit slice 1 is the 4D cube model with lazy access, the mean
pattern and virtual imaging with apertures, extracted read-only from mac4DSTEM's pushed HEAD and checked against
py4DSTEM"). Tier 1: every number here is one a scientist reads (a virtual bright- or dark-field image, a mean
diffraction pattern, a pixel of a cube). What is written here is fixed before the code; a change after code starts is
an amendment, dated, with its reason (section 9).

## 1. What slice 1 makes

Two modules and products, `CuboCore` (the cube) and `CuboVirtual` (what is computed from it), with `CuboStrain` and
`CuboPtycho` as empty modules that carry their planned scope in a doc comment. Public API only: nothing an app needs
is `package`; nothing here names an app type (`AppState`, `FourDArray`, `DatasetDescriptor`, `LoadView`, `FloatImage`
and the rest of mac4DSTEM's vocabulary stay in mac4DSTEM); no message says "mac4DSTEM". Dependencies: Foundation,
Accelerate and Metal; no third-party package. Swift 6 language mode, strict concurrency.

### 1.1 `CuboCore`: the cube

- `CubeShape`: `scanRows` (ry), `scanColumns` (rx), `detectorRows` (qy), `detectorColumns` (qx); `scanCount`,
  `detectorCount`, `elementCount`; `Hashable`, `Codable`, `Sendable`. The storage order is C order
  `[ry][rx][qy][qx]`, py4DSTEM's `(R_Nx, R_Ny, Q_Nx, Q_Ny)` (`DataCube.__init__`, dim names `Rx, Ry, Qx, Qy`). **The
  index map is stated once and used everywhere:** py4DSTEM's `x` (its first axis of a pair, `Rx`, `Qx`) is this kit's
  **row** index; its `y` (`Ry`, `Qy`) is this kit's **column** index. The kit says row and column, never x and y, so
  the swap that mac4DSTEM documents at every call site cannot be silent.
- `DiffractionPattern`: one pattern as read, `detectorRows × detectorColumns` `Float32` values, row-major.
- `ScanTile`: contiguous scan rows `rows: Range<Int>` of a cube, `[rows.count][rx][qy][qx]` `Float32` values.
- `CubeSource`, the protocol every cube conforms to (an actor; reads are serialised per source): `shape`,
  `readPattern(row:column:)`, `readScanTile(rows:)`. Every read returns `Float32` whatever the stored element type.
- `MemoryCube`: the **eager** cube, the whole array as one `[Float]` in storage order; built from an array and a shape,
  or loaded from any `CubeSource` tile by tile (`MemoryCube(loading:tiling:)`).
- `MappedCube`: the **lazy, memory-mapped** cube: a file, a byte offset, an element type and a shape; the region is
  `mmap`ed read-only and private; a read converts the elements of the requested rows to `Float32` (Accelerate's vDSP
  conversions where one exists, a scalar loop for 64-bit integers) and touches nothing else. Element types: `float32`,
  `float64`, `uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `uint64`, `int64`, all native little-endian; a
  big-endian descriptor is refused by name (slice 2 at the earliest).
- `NPYFile`: NumPy's `.npy` container (versions 1.0, 2.0, 3.0: magic, header length, the header dictionary's `descr`,
  `fortran_order`, `shape`), opened as a `MappedCube` of rank 4. `fortran_order: True` is refused; a rank other than 4
  is refused; `descr` outside the list above is refused. This is the kit's exchange format with py4DSTEM in slice 1
  (`np.save(path, datacube.data)` on the Python side); HDF5, DM4, EMPAD and the rest are FormatoKit's.
- `ScanTiling`: how a whole-cube pass walks the scan axis: `rows(perTile:)` fixed, or `.automatic`, which is a
  **function of the shape alone**: the largest row count whose Float32 tile is at most 64 MiB, at least 1. Not of the
  machine: mac4DSTEM sizes its tiles from physical memory and accepts per-machine low bits on its tiled reducers; a kit
  that promises the same numbers everywhere cannot. The CPU reducers of 1.2 do not depend on the tiling at all
  (section 3.4); the Metal reducers do, within their registered tolerance.

### 1.2 `CuboVirtual`: the mean pattern and virtual images

- `DetectorGeometry`, in detector pixels, row and column as 1.1 says:
  `.point(row:column:)` (integers), `.circle(centerRow:centerColumn:radius:)`,
  `.annulus(centerRow:centerColumn:inner:outer:)`, `.rectangle(rowMin:rowMax:columnMin:columnMax:)` (doubles, as
  py4DSTEM takes them), `.mask([Float])` (any weights on the detector grid, `detectorCount` values). Centres and radii
  are `Double`.
- `DetectorMask.make(_:shape:) -> [Float]`: the mask on the `qy × qx` grid, exactly py4DSTEM's `make_detector`
  (section 3.1), weights 0 or 1, or the given weights for `.mask`.
- `VirtualImaging.image(of:geometry:path:tiling:) -> ScanImage`: for every scan position the sum over the detector
  of pattern × mask (section 3.2); `ScanImage` holds `ry × rx` **`Double`** values and the `ComputePath` that made it.
- `VirtualDiffraction.mean(of:path:tiling:) -> DetectorImage` and `max(of:...)`: the mean and the maximum pattern over
  every scan position (section 3.3); `DetectorImage` holds `qy × qx` `Double` values and the path.
- `ComputePath`: `.cpu` (Accelerate: vDSP conversions, dot products and maxima; Double accumulation) and `.metal`
  (compute kernels compiled at run time from the source the module carries, in Metal's safe math mode so no
  fast-math reassociation moves a number; Float32 accumulation; Double combination on the host). The kernels are
  source compiled with `makeLibrary(source:options:)`, not a `.metallib` resource, so `swift build` alone builds the
  kit and the test runs headless. The
  default is `.cpu`; `.metal` is a choice the caller makes and the result records. There is no `.automatic`: a number
  must not depend on the hardware without the caller asking (the suite ROADMAP's "speed never moves a number
  silently"). A Metal request on a machine without a device throws `CuboVirtualError.noMetalDevice`; the caller falls
  back if it wants to.

Results are `Double` on purpose. py4DSTEM's virtual image of an integer cube is an exact integer sum (NumPy sums
`uint16 × bool` in `uint64`); a `Float32` result would lose it above 2²⁴, and a kit whose numbers are the reference
cannot round where the reference does not. The cube itself stays `Float32`, as py4DSTEM's data stays the file's type.

## 2. The oracle: py4DSTEM at mac4DSTEM's lock

py4DSTEM `f050d207851f8accd3aa550365baaf1f38230428` (dev, 2026-03-26, version 0.14.19), the commit mac4DSTEM pins in
`tools/lib/fetch-py4dstem.sh` at `7448401b`, cloned to the gitignored `References/py4DSTEM-f050d207` and installed into
`References/py4dstem-venv` (Python 3.12, the owner's conda `py4dstem` interpreter) with mac4DSTEM's
`tools/lib/py4dstem-ci-constraints.txt` (NumPy 2.5.0, h5py 3.11.0, SciPy 1.18.0, the stack its parity numbers were
established on), by `References/provision-py4dstem.sh`; the freeze is `References/py4dstem-venv-freeze.txt`. Oracles run
as `References/py4dstem-venv/bin/python -I -B`. An unpinned copy is never a reference; the owner's conda env holds
0.14.17 and is not used. GPL-3.0; credited in `NOTICE` and `README.md`.

File:line pins into the clone (`py4DSTEM/datacube/`):
- `virtualimage.py`: `DataCubeVirtualImager.get_virtual_image` 67 (the per-position sum `np.sum(self.data[rx, ry] *
  mask)` at about 215), `make_detector` 588 (the predicates, section 3.1), `get_calibrated_detector_geometry` 480
  (not used: slice 1 takes geometry in pixels, `centered=False, calibrated=False`).
- `virtualdiffraction.py`: `get_virtual_diffraction` 63 (`np.mean(self.data, axis=(0, 1))`, `np.max(...)` at about
  120), `get_dp_max` 308, `get_dp_mean` 337.
- `datacube.py`: `DataCube.__init__` 33 (dim names `Rx, Ry, Qx, Qy`), `R_Nx`/`Q_Nx` 172-194.

What py4DSTEM computes, exactly, for the quantities of slice 1 (read from the pinned source, 2026-10-14):
- the mask: NumPy index arrays (`np.indices`, int64) minus Python floats, squared, summed and compared in **float64**
  with strict inequalities; the rectangle bounds rounded with `np.round` (half to even) and sliced; the point set by
  integer index (a negative index wraps in NumPy; the kit refuses it, section 3.1);
- the virtual image: `pattern * mask` (float32 × bool → float32; integer × bool → the integer type) summed by
  `np.sum` (float32 pairwise for float32 data; `uint64` for unsigned integers up to 32 bits), one value per position,
  stored into a float64 array;
- the mean pattern: `np.mean` over the two scan axes, which for float32 data is a **float32** pairwise reduction
  and for integer data a float64 one; the max pattern: `np.max`, exact.

So py4DSTEM's own float32 results carry a float32 rounding the kit does not reproduce (and should not: it is an
artefact of NumPy's accumulator, not a definition). The oracle therefore records, for every quantity, **two numbers**:
py4DSTEM's own call (`get_virtual_image`, `get_dp_mean`, `get_dp_max`, as a user gets them) and the **float64
evaluation of the same definition** (`np.sum(pattern.astype(np.float64) * mask)`, `np.mean(data, axis=(0, 1),
dtype=np.float64)`). The kit is held to the float64 number (section 4); the distance between the two py4DSTEM numbers
is recorded in the pins and reported, never asserted against.

## 3. The definitions, pinned

### 3.1 The detector mask (`make_detector`)

On the `qy × qx` grid, for the pixel at row `i`, column `j` (integers), with the centre `(c_r, c_c)` and radii as
doubles, evaluated in **Double**, exactly in this form:
- circle: `(i − c_r)² + (j − c_c)² < r²` (the squares as products, `r² = r · r`; NumPy's `x ** 2` on a float64 array is
  `x · x`, and CPython's `r ** 2` is `pow(r, 2.0)`, correctly rounded, equal to `r · r`);
- annulus: `(i − c_r)² + (j − c_c)² > r_in²` **and** `< r_out²`; `r_out > r_in` is required (py4DSTEM asserts it; the
  kit throws `innerNotBelowOuter`);
- rectangle: rows `[round(rowMin), round(rowMax))`, columns `[round(columnMin), round(columnMax))`, `round` half to
  even (`np.round`), clipped to the grid (NumPy slicing clips; a negative bound is **refused**, where NumPy would wrap);
- point: the one pixel; a point outside the grid is refused (`pointOutsideDetector`); NumPy would wrap a negative index
  and raise on an index past the end, so this is a DEVIATION in the refusing direction, named here;
- mask: the given weights, which must number `detectorCount`.

The mask is evaluated in Double and not Float32 because a Float32 evaluation disagrees with py4DSTEM on pixels whose
distance lies within one Float32 ulp of the radius; the pins carry such a pixel, found by search (section 5.1 (g)).
mac4DSTEM's `VirtualDetector.makeMask` and its `virtualAperture` kernel evaluate in Float32: the kit's mask is held to
py4DSTEM, and mac4DSTEM's consumer notes say where its numbers can move at the bump (one pixel on a boundary, at a
fractional radius).

### 3.2 The virtual image

For scan position `(r, c)`: `image[r, c] = Σ_{i,j} pattern[r, c][i, j] · mask[i, j]`, the sum over the detector in
row-major order. CPU path: the pattern converted to Double, the dot product with the Double mask (`vDSP_dotprD`) with
vDSP's internal order; the result is a Double. Metal path: one thread per scan position, Float32, each detector row
summed first and the row sums then added (two levels, so the error grows with `qy + qx`, not `qy · qx`); the
Float32 result widened to Double. Tiles partition the scan axis, so no tile boundary touches a position's sum: the
tiled result equals the untiled one exactly, on both paths (mac4DSTEM's invariant I2, kept).

### 3.3 The mean and maximum pattern

`mean[i, j] = (1/N) Σ_{positions} pattern[i, j]`, N the scan count, the sum over positions in scan order (row-major
over `(r, c)`). CPU path: per position, the pattern converted to Double (`vDSP_vspdp`) and added to a Double
accumulator **with Neumaier compensation** (a sum and a running correction per detector pixel), tile after tile, the
same sequential order whatever the tiling; then the sum plus its correction divided by N. The compensation is chosen
because the owner's cubes are dark-subtracted: a pixel near the detector edge has a mean near zero made of terms of
±500, and an uncompensated sequential sum over 10⁵ positions can then miss the float64 reference by more than the
registered 1e-10 (its bound is N · 2⁻⁵³ · Σ|x|), while a compensated one is within a few ulps of the exact sum. Metal
path: per tile, one thread per detector pixel sums the tile's positions in Float32, two levels deep (chunks of 64
positions in scan order, then the chunk sums), so the error grows with `64 + n/64`, not `n`; the host adds the tile
sums into a Double accumulator in tile order and divides by N. `max[i, j]`: the maximum over positions, exact on every path
(a maximum has no rounding), seeded at `−∞` so an all-negative cube is right.

### 3.4 What is bit-identical and what has a tolerance

| Comparison | Claim |
|---|---|
| `MappedCube` against `MemoryCube` of the same bytes, every reducer, every path, every tiling | **bit for bit**: the tile floats are the same, the arithmetic is the same |
| CPU reducers against the same reducers at another tiling | **bit for bit**: the virtual image is per position; the mean is one sequential Double sum in scan order; the max is exact |
| Max pattern, CPU and Metal, against py4DSTEM `get_dp_max` | **bit for bit** (after the cube's Float32 conversion, section 3.5) |
| Virtual image, CPU, against py4DSTEM `get_virtual_image` on an **integer** cube whose values are below 2²⁴ | **bit for bit** (exact integer sums on both sides; the kit's Double sum is exact below 2⁵³) |
| Mean pattern, CPU, against `np.mean(..., dtype=float64)` on an integer cube below 2²⁴ | **bit for bit** (exact sums, one division each) |
| Virtual image, CPU, against the float64 evaluation on a float32 cube | relative **1e-10** (two Double summation orders over at most 2¹⁶ terms: bound n · 2⁻⁵³ ≈ 7e-12) |
| Mean pattern, CPU, against `np.mean(..., dtype=float64)` on a float32 cube | relative **1e-10** (compensated against NumPy's pairwise sum, whose own bound is about log₂N · 2⁻⁵³ · Σ\|x\|) |
| Virtual image, Metal, against the CPU path | relative **1e-5** |
| Mean pattern, Metal (default tiling), against the CPU path | relative **1e-5** |
| py4DSTEM's own float32 numbers against its float64 ones | **recorded, not asserted**; predicted 1e-7 to 1e-5 relative, larger than every kit distance above |

"Relative `t`" means `|got − want| ≤ t · max(|want|, f)` with the floor `f = 1e-6 · max_k |want_k|` over the image, so
a pixel near zero in an image of large values is judged against the image, not against itself. A difference beyond a
registered bound is a failure of the port (or of the kernel), never a tolerance to widen; the Metal bounds are the
ones most likely to be too tight on real data, and if the measurement on the owner's cubes exceeds 1e-5 the kernel is
changed (a finer partial-sum tree), not the number.

### 3.5 The cube's element type

Every read returns Float32. Integer sources convert exactly up to 2²⁴ (16 777 216); above it the value is rounded to
the nearest Float32 and the kit says so in `MappedCube.elementType` and the oracle's `exact_in_float32` flag per cube.
`float64` sources round to nearest even (`vDSP_vdpsp`, NumPy's `astype(np.float32)`). The bit-for-bit claims of 3.4
are made on the Float32 cube: the oracle computes py4DSTEM's numbers on `data.astype(np.float32)` for the pins, and
records beside them the same calls on the original dtype, with the distance, so a reader sees what the conversion cost
(zero on every cube of section 5 below 2²⁴). Non-finite pixels are not claimed: py4DSTEM propagates NaN through
`np.sum` and `np.max`; the kit's Double sums do too, but Metal's `max` drops NaN, and hot-pixel filtering is not in
slice 1.

## 4. Pins (agreement with py4DSTEM)

A generator, `tools/slice-1/pins.py`, run in `References/py4dstem-venv` before the Swift code exists, writes:
- `Tests/CuboVirtualTests/Fixtures/<cube>.npy` for each synthetic cube of 5.1 (small, committed) and
  `Tests/CuboVirtualTests/Fixtures/slice-1-pins.json` holding, per cube and per geometry of 5.2: the mask's pixel
  count and its row-major pixel list for the small cubes, py4DSTEM's `get_virtual_image` (its own dtype path and the
  float64 evaluation), `get_dp_mean` (own and float64), `get_dp_max`, the versions (py4DSTEM, NumPy, h5py, Python) and
  the generator's sha256 of each `.npy`;
- `<name>.npy` and `<name>-pins.json` for py4DSTEM's test data (5.3) and the owner's cubes (5.4) in the real-data
  directory, `$CUBOKIT_REAL_DATA`, default `~/Library/Caches/CuboKit/real-data`: **outside the repository** (the suite
  rule: the owner's data is never copied into a repo, and a gitignored folder inside one is still a copy there); the
  whole cube exported with `np.save`, derived, deletable and regenerable by the generator; the same quantities,
  geometries placed from the mean pattern as 5.4 says.

The Swift tests read the committed pins; the real-data tests read the real-data directory and are skipped by name
when it is absent, so `tools/test.sh` is green on a clone and complete on this machine. Agreement required as section
3.4's table. The pins carry every number as a float64 string with 17 significant digits.

## 5. Truth and data

### 5.1 Synthetic cubes (committed fixtures, written by the generator, seed 20261014)

(a) **digits**, `2 × 3 × 5 × 7` float32, value `s · 10000 + i · 100 + j` for scan index `s`, row `i`, column `j`
(mac4DSTEM's `virtual-detector-test` pattern, 7448401b): every sum is exact and a transposed or shifted mask is
conspicuous; (b) **gaussian**, `6 × 5 × 16 × 16` float32, a bright disk of radius 4 at (7.5, 8.5) on a background
`N(1000, 300)` with negative values (the owner's cubes are dark-subtracted and go negative); (c) **counts16**,
`4 × 6 × 12 × 12` uint16, Poisson(λ = 3000) so values exceed 2¹⁵ nowhere but the sum per position exceeds 2¹⁶ (an
int16 or uint16 accumulator is caught); (d) **counts32**, `3 × 4 × 9 × 9` uint32 with values up to 60 000 (above
int16, a signed read is caught); (e) **thirds**, `2 × 2 × 4 × 4` float64 with values `k/3` (not representable in
Float32: the registered rounding is exercised and the oracle's float32-cast pins differ from its float64 ones);
(f) **negative**, `2 × 2 × 3 × 3` float32, all values negative (the max seeded at 0 is caught); (g) **boundary**, a
`1 × 1 × 32 × 32` float32 cube of ones, with a geometry found by the generator's search where the Float32 and the
Double predicate of 3.1 disagree on at least one pixel (the generator fails if it finds none within the search).

### 5.2 Geometries on every synthetic cube

A bright-field circle at the grid centre `((qy − 1)/2, (qx − 1)/2)` with radius `0.3 · min(qy, qx)`; an off-centre
circle with a fractional centre and radius (centre `(2.3, 4.7)`, radius `2.9`); an annulus with a fractional centre
(centre `(1.5, 5.25)`, inner `1.4`, outer `3.6`); the **exact boundary** cases at the integer centre
`(qy div 2, qx div 2)`: a circle of radius 2 (pixels at distance exactly 2 excluded) and of radius 3, an annulus
`(1, 2.5)` (distance exactly 1 excluded, so the four axial neighbours are out); points at `(0, 0)` and at the last pixel; a
rectangle with half-integer bounds `(0.5, 2.5, 1.5, 3.5)`, rounded half to even to rows `0..<2`, columns `2..<4`; a
weighted mask with weights `0, 0.5, 1, 2, −1` (exact in Float32) in a fixed pattern; and the refusals: inner ≥ outer,
a point off the grid, a mask of the wrong length, a negative rectangle bound.

### 5.3 py4DSTEM's test data

`small_datacube.dm4` (the file `test/test_datacube.py` reads, Google Drive id `1QTcSKzZjHZd1fDimSI_q9_WsAU25NIXe`;
upstream notes the id is `vac_probe.h5`'s, so what arrives is recorded by shape and dtype, not assumed) and
`Au_sim.h5` (`1PmbCYosA1eYydWmmZebvf6uon9k_5g_S`), downloaded by the generator into `References/py4dstem-test-data/`
and read with py4DSTEM's own `import_file` / `read`; a datacube among them is exported as 4 and pinned with the
geometries of 5.4. If neither download yields a datacube, the pins say so and the real-data tests for 5.3 skip by name.

### 5.4 The owner's cubes (`$MACEM_DATA/4dstem/training_dataset/`, read-only, a manifest line each)

- `Al_Mg_Si_060_STEM SI_preprocessed_unfiltered_bin_4_20260712.h5`, `dm_dataset_root/dm_dataset/data`,
  `330 × 330 × 64 × 64` float32 (1.78 GB): the study's cube (the mean pattern, BF and DF on a weak precipitate
  reflection are the study test's 4D panels);
- `Au_ref_ROI15_preprocessed_unfiltered_bin_4_20241214.h5`, the same path, `205 × 310 × 64 × 64` float32 (1.04 GB):
  non-square scan;
- `sim_Au_data_all_binned.h5`, `4DSTEM_simulation/4DSTEM_polyAu/data`, `100 × 84 × 125 × 125` uint16 (0.53 GB, the
  py4DSTEM tutorial set): the integer bit-for-bit claims at full size, odd detector size;
- `Particle_1_Stack_1_45x90_…_300kV_bin8.h5`, `datacube_root/datacube/data`, `90 × 45 × 128 × 128` uint32 (0.27 GB):
  uint32 below 2²⁴ (sampled max 63 912), the 64-bit conversion loop is not exercised (no uint64 cube is in the list:
  `downsample_Si_SiGe_exp.h5` is uint64 and 1.3 GB; it is left for slice 2 as a named gap).

Geometries per real cube, placed by the generator from py4DSTEM's mean pattern so they are deterministic: the centre
is the argmax of the mean pattern; BF: circle radius `0.1 · min(qy, qx)` (6.4 → 6.4; radii stay fractional on
purpose); ADF: annulus `(0.28, 0.47) · min(qy, qx)`; DF on one reflection: circle radius 3.0 at the argmax of the mean
pattern outside radius `0.2 · min(qy, qx)` from the centre (the strongest off-axis spot: on the Al–Mg–Si cube that is a
matrix reflection, not the precipitate's, and the study's own aperture is mac4DSTEM's to place); and the
exact-boundary circle of radius 2 at the integer centre. Every real-data pin is recorded with the cube's sha256 and
the generator's log.

### 5.5 Planted truth (independent of py4DSTEM)

- On **digits**: every virtual image equals the standard-library Python sum written in the generator from the
  predicates of 3.1 (as mac4DSTEM's `reference.py` does), and the Swift test recomputes one case by hand; the mean
  pattern is `((N − 1)/2) · 10000 + i · 100 + j` exactly; the max is `(N − 1) · 10000 + i · 100 + j`.
- A cube of ones: every virtual image is the mask's weight sum; the mean is 1; the max is 1.
- On a grid large enough to hold it (the gaussian cube's 16 × 16 and the boundary cube's 32 × 32), the circle of
  radius 2 at an integer centre holds exactly 9 pixels and radius 3 exactly 25 (counted by hand: `i² + j² < 4` and
  `< 9`); the annulus `(1, 2.5)` holds `21 − 1 − 4 = 16`: the 21 pixels with `i² + j² < 6.25` less the
  centre and the four at distance exactly 1.
- A cube whose pattern at scan index `s` is `s · P`: the mean is `((N − 1)/2) · P` and the virtual image is
  `s · Σ P · mask`, both exact in Double for the small shapes used.
- Lazy against eager, tiling invariance of the CPU path, and the Metal tolerances, as 3.4.
- `NPYFile`: a header with padding to a 64-byte boundary (version 1.0 as `np.save` writes it), a version 2.0 header,
  a `>f4` refusal, a `fortran_order: True` refusal, a rank-3 refusal, a file shorter than its shape (refused, never
  read past the end), and the digits cube read back equal to its pins' sha256 after a round trip through
  `MemoryCube`.

## 6. Mutants (planned, each new test broken once)

The mask transposed (rows and columns swapped: the off-centre circle on a non-square detector); `<=` for `<` on the
outer radius and `>=` for `>` on the inner (the exact-boundary cases); half-up for half-even in the rectangle; the
Float32 predicate for the Double one (5.1 (g)); the point at (column, row); inner ≥ outer accepted; weights treated
as 0/1 (the weighted mask); the mean divided by the tile's position count instead of N (a multi-tile test with one
row per tile); the tile copied to offset 0 instead of `rows.lowerBound · stride` (the same test: a mean that depends on
the tiling); a Float32 CPU accumulator (the gaussian cube's 1e-10 claim, which Float32 misses by about 1e-6); the max
seeded at 0 (the negative cube); `.npy` shape read in reverse (non-square shapes); the header length read as big-endian;
uint16 read as int16 (counts32: 60 000); float64 read as float32 bytes (thirds); the Metal kernel bound at offset 0
for every tile (the multi-tile Metal mean); the Metal kernel summing one level (the tolerance test on the gaussian
cube is expected to pass either way, so this mutant's catcher is the real-data check, and the registration says so).

## 7. Gates and predicted outcomes (registered now)

1. Pins (section 4): every claim of the table in 3.4 holds on every cube of 5.1 and every geometry of 5.2.
2. Truth (5.5): every closed form exact; every count exact.
3. Real data (5.3, 5.4): the same claims hold on the four owner's cubes and py4DSTEM's test datacube, lazy from the
   `.npy` exports (the memory-mapped path at 1.78 GB), the numbers and the Metal-against-CPU distances cited in
   `docs/status.md` by the run's log. Predicted: integer cubes bit for bit; float32 cubes within 1e-10; Metal within
   1e-5 on both reducers; py4DSTEM's own float32 mean of the Al–Mg–Si cube differs from its float64 mean by more than
   the kit does.
4. Every new test broken once by a mutant of section 6, on a copy of the frozen commit, recorded in
   `docs/slice-1-mutants-<date>.md`.
5. One Gate B refuter on the frozen slice-1 commit (a second only on a named doubt); Gate D only if a discrepancy's
   cause is unproven. The refuter is asked to compare the port with py4DSTEM on cubes and apertures the fixtures do
   not hold (other shapes, other dtypes, radii at the grid edge and beyond it, masks with negative weights, tile sizes
   of 1 and of the whole cube, a Metal run on a cube larger than one tile).
6. `tools/test.sh` green with the stamp; a cold build with zero warnings; `tools/consumer-check` compiles against the
   public API from a separate package.

## 8. What is not claimed

Agreement with py4DSTEM is agreement, not truth: a virtual image is right when its mask is where the user put it and
its sum is the sum, which the planted truth of 5.5 checks; whether that aperture isolates the reflection a scientist
means is the app's to show on screen. No calibration (`centered`, `calibrated`, `shift_center`, `subpixel` of
`get_virtual_image`) is in slice 1: geometry is in detector pixels from the corner, as py4DSTEM's defaults. No
masked, median or centre-shifted virtual diffraction (`get_virtual_diffraction` beyond mean and max). No HDF5 or DM4
reading: the `.npy` container is the exchange format until FormatoKit's readers exist. No centre of mass or DPC
(slice 2). Nothing here is validated on a microscope; the study's apertures are mac4DSTEM's, drawn on its mean pattern.

## 9. Amendments

### 2026-10-15, at the first test run: the pins, the tests and the stamp

- py4DSTEM's test data: `small_datacube.dm4` arrived as a 20 × 20 × 512 × 512 uint16 datacube and is pinned
  (`py4dstem_small`); `Au_sim.h5` is in py4DSTEM's legacy v0.12 layout and is not pinned (its content duplicates the
  owner's `sim_Au_data_all_binned.h5`, which is).
- Real-data exports live outside the repository (section 4 as committed); the real-data tests are enabled by trait when
  their export is present, and `tools/test.sh` writes no stamp when they were skipped.
- The tests are Swift Testing, not XCTest (owner, 2026-10-15 00:30, the kits' standard); every name and case kept.
- One planted test added before the mutants: a point detector reads its own pixel at interior points of a non-square
  detector (the pins hold only corners, where a swapped point traps instead of failing).
- `tools/test.sh` counted only tracked files as dirty, so a run with uncommitted new sources stamped the previous commit;
  the stamp now counts untracked files under `Sources/` and `Tests/`.

No registered number changes.

### 2026-10-15, after the Gate B refuter on `85318cd`: the Metal kernels are compensated

One refuter (Sonnet, its scripts in the session's scratchpad `refuter/`) compared the frozen commit with py4DSTEM on cubes
and apertures the fixtures do not hold: 17 500 random geometries on seven detector shapes, every element type around the
2^24 and 2^53 ties, multi-tile cubes with tilings that do not divide the scan, hand-made `.npy` headers. What held: every
mask pixel-identical to `make_detector`; the CPU image and mean against the float64 definition (at most 8e-13 and 3e-16;
integer cubes bit for bit); the max bit for bit on both paths; the Float32 conversions bit-identical to
`astype(np.float32)`; lazy, eager and every CPU tiling bit for bit; the `.npy` reader's refusals. What broke:

- **F1, a registered claim: Metal against CPU, relative 1e-5 with the floor of 3.4.** The first kernels summed in plain
  Float32 (two levels deep). On zero-mean, dark-subtracted noise (N(0, 500), no signal) pixels near zero are judged
  against a floor of 1e-6 · max, and the Float32 error (about 1e-7 to 5e-7 of max) exceeded it: image up to 8.2e-3,
  mean up to 1e-3; on ±1e6 cancelling pairs 3e-4. The owner's cubes passed (6.1e-7), which is why the gate did not see
  it. Section 3.4 says the kernel changes, not the number: **both kernels now use error-free transformations** (Knuth's
  TwoSum and the fma TwoProduct, the Dot2 algorithm of Ogita, Rump and Oishi), compiled in Metal's safe math mode so
  nothing reassociates them; each thread returns its sum and its correction and the host adds the two in Double.
  Sections 3.2 and 3.3's "two levels deep" is replaced by this. Measured after the fix: the refuter's batteries at most
  1.5e-8; the owner's cubes at most 2.4e-13 (image) and 7.6e-14 (mean). The registered bound, 1e-5, does not change.
- **F2: `NPYFile.readDoubles` trapped** on a header whose element or byte count overflows `Int` (`MappedCube` refused
  them). `NPYFile.header(of:)` now refuses such a header with checked arithmetic.
- **F3, F4 (not claimed; behaviour made consistent):** the CPU mean turned ±∞ into NaN (the compensation computed
  ∞ − ∞): a non-finite step now carries no correction, so the mean is ±∞ as NumPy's. A column of −0 lost its sign: the
  accumulators start at −0, the identity of IEEE addition. Metal dropped NaN from the maximum and skipped zero-weight
  pixels (so a NaN outside a mask never reached the image): both now propagate NaN as the CPU path does.
- Recorded, not changed: `MappedCube` refuses an `.npy` whose data offset is not a multiple of the element size (an
  unpadded hand-made header; `np.save` pads to 64 bytes); Metal flushes Float32 subnormals (below 1.2e-38) to zero;
  non-finite data stay outside every claim of 3.4.

Mutants (section 6, `tools/slice-1/mutants.py`, all on exported copies): on the XCTest tests 29 of 33 behaved as
registered; M14b (the version 2.0 header length's high bytes) survived because the planted header was under 64 KiB, and
M20 (a zero extent) because the byte-count guard refuses it too. A 70 kB version 2.0 header and a shape with two negative
extents were added; M20 is recorded as equivalent and M20b (the per-extent guard dropped) takes its place. The
compensated kernels bring their own mutants (each half of the compensation, the NaN max) and the F2 to F4 fixes theirs;
the run on the fixed commit is cited in `docs/status.md`.
