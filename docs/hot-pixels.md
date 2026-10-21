# Hot-pixel filtering

`HotPixelFilter` follows actual py4DSTEM 0.14.19 (f050d207)
`DataCube.filter_hot_pixels` on Float32 input cubes. Numerical seeding source:
mac4DSTEM c809679c, Core/Data/HotPixelFilter.swift. No app export, tiling,
progress, settings or provenance is ported.

The classification window contains 21 pixels: 3×3 and the four three-pixel arms
at distance two, excluding 5×5 corners. It wraps at detector edges and includes
the center. Comparison index 1 means second brightest; 0 through 20 are accepted.
The excess comparison is strictly greater than the Float32 threshold.
Masked pixels are replaced in row-major order in every frame by the median of a
clipped 3×3 window. The replacement is sequential and later windows see earlier
replacements, exactly as in Python. Even windows average the two middle values
in Float32. Small detectors retain duplicated wrapped neighbor samples.

`filter` accepts an eager MemoryCube and returns a new cube, the mean and mask.
The scan mean uses NumPy's strided sequential Float32 reduction, including its
roundoff. A 1×1 detector uses NumPy's contiguous pairwise Float32 sum instead. This is necessary for mask parity. The app's separate Double-mean policy
is not adopted: a caller choosing such a mean can pass it explicitly to `mask`.
`replacing` exposes the single-frame numerical operation. Invalid shape, nonfinite
input/intermediate mean, nonfinite/unrepresentable threshold or rank outside
0...20 throws `HotPixelFilterError`. Source cube values are unchanged.

Regenerate from this worktree:

```sh
MPLCONFIGDIR="$PWD/.build/matplotlib" ../../tools/memcap.sh -- ../../References/CuboKit/py4dstem-venv/bin/python -I -B tools/hot-pixels/pins.py
```

The generator verifies the pinned commit, installed oracle source bytes, Python
3.12.4 and NumPy 2.5.0. Eight small synthetic Float32 cubes (seed 501) exercise edge
and adjacent hot pixels, sequential replacement, even clipped medians, wraparound,
comparison indices 0/1/8/20, a one-row detector and noninteger Float32 mean sums. A 256-position 1×1 detector cancellation case pins
NumPy's pairwise mean 0 where sequential summation gives 0.00390625.
Mean samples, masks and every filtered pixel are asserted exact today (tolerance zero;
D12 allows a written tolerance against py4DSTEM for the samples). There is no comparison against a transcribed replacement implementation.

Measured gap: 0 for means, masks and filtered pixels, tested via Float32 bit patterns.
Fixture size: 30,403 bytes. Three new Swift Testing tests failed against identity/
zero placeholders (38 issues), then passed. The intermediate sequential-only mean
implementation failed specifically on the 1×1 detector mean. The final pairwise
path makes that fixture pass without changing its expected value or tolerance.
The inherited Codex reviewer identified that case independently; Sonnet is not
available in this session. No owner approval is claimed.

Two further actual-oracle fixtures pin negative-zero input medians on 1×1 and
2×2 frames with a negative threshold so all pixels are selected. NumPy takes a
mean of the middle one/two samples using a +0 reduction accumulator. The
pre-fix median retained -0 and these fixtures failed at four bit-pattern checks;
the matching +0 accumulation fixes both odd/even windows. Tolerance remains zero.

## Reusable replacement (`HotPixelFilter.Replacer`)

`Replacer(rows:columns:mask:)` precomputes, once per mask, the clipped 3×3 neighbour
indices of every masked pixel (flat arrays, row-major target order). `apply(to:base:)`
then replaces the masked pixels of one frame in place on an `UnsafeMutableBufferPointer<Float>`
(or an `inout [Float]`), with a stack-allocated 9-sample window and an insertion sort, so
nothing is allocated per frame. `filter` uses it for every scan position; `replacing`
keeps its copy-and-validate contract and uses the same allocation-free loop.
Results are bit-identical to the previous implementation (the existing py4DSTEM pins stay
exact, and a test compares `Replacer` and `replacing` with a verbatim copy of the old
loop on random masks, including -0 samples and a frame at an offset in a larger buffer).

Measured 2026-10-21, release build, one thread, 256×256 Float32, 40 random masked pixels,
2000 patterns (`CUBO_BENCH=1 swift test -c release --filter CuboCoreTests.HotPixelTests`,
test `replacementTiming`; the mac4DSTEM app now calls this kit, so there is no separate app loop):

| path | ms per pattern |
| --- | --- |
| before: `replacing` (validate, copy, array window per masked pixel) | 0.054 |
| after: `replacing` (same contract, allocation-free loop; validation and the 256 KB copy dominate) | 0.054 |
| after: `Replacer.apply` in place (includes re-copying the 256 KB pattern for the benchmark) | 0.005 |

`Replacer.apply` is about 10x faster than the earlier per-frame path; one-frame `replacing`
is unchanged in cost because finite-validation and the copy dominate it. Callers with many
patterns should build one `Replacer` per mask and reuse it.

