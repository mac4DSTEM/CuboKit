# Slice 4 mutants, 2026-10-15

`tools/slice-4/mutants.py` on slice 1's runner (registration slice 4, section 6, and the three Gate B regressions), on
`006fe47` (the Gate B fixes), one warm `.build`, catchers `Slice4Tests/` (the synthetic tests; the real-data suite
is not run per mutant). 31 mutants: 30 red, the 1 registered survivor green, none unexpected.

- **D2** first did not compile (its replacement `v = v` is a Swift error); restated as `if false, …`, it then
  survived: the median test's NaN sat in the middle of three values after the sort. The test now puts NaN first among
  five (`537462d`, tier 1, test only), and D2 is red there.
- **F6** (the upsampled grid rounded half away from zero) is the registered survivor: a tie needs `(x + dx) · u` to be
  an exact half-integer, which no fixture or real pattern produces.

| id | mutant | catchers | expected | result | seconds |
|---|---|---|---|---|---|
| D1 | an even median as the upper middle | `Slice4Tests/` | red | RED (15 started) | 22 |
| D2 | NaN ignored by the median | `Slice4Tests/` | red | RED on 537462d (after the test change; first: did not compile, then GREEN) | 16 |
| D3 | the mask mean divided by the scan count | `Slice4Tests/` | red | RED (15 started) | 6 |
| D4 | weights normalised by the count | `Slice4Tests/` | red | RED (15 started) | 6 |
| D5 | the selected maximum seeded at 0 | `Slice4Tests/` | red | RED (15 started) | 6 |
| D6 | the weighted median of the unweighted values | `Slice4Tests/` | red | RED (15 started) | 6 |
| O1 | the scan mask transposed | `Slice4Tests/` | red | RED (15 started) | 6 |
| O2 | the fast centre refined anyway | `Slice4Tests/` | red | RED (15 started) | 6 |
| O3 | excluded origins 0 instead of NaN | `Slice4Tests/` | red | RED (15 started) | 6 |
| O4 | remove_NaN with the mean over all values (NaN as 0) | `Slice4Tests/` | red | RED (15 started) | 5 |
| F1 | Friedel with |G|² instead of G² | `Slice4Tests/` | red | RED (15 started) | 5 |
| F2 | the masked form's padding at 0 | `Slice4Tests/` | red | RED (15 started) | 6 |
| F3 | term3 dropped | `Slice4Tests/` | red | RED (15 started) | 5 |
| F4 | the parabola's denominator sign | `Slice4Tests/` | red | RED (15 started) | 11 |
| F5 | the remainder as fmod (a negative origin kept negative) | `Slice4Tests/` | red | RED (15 started) | 12 |
| F6 | the upsampled grid rounded half away from zero | `Slice4Tests/` | survives | GREEN (15 started) | 12 |
| F7 | the upsampled refinement dropped | `Slice4Tests/` | red | RED (15 started) | 6 |
| B1 | the threshold index rounded half away from zero | `Slice4Tests/` | red | RED (15 started) | 6 |
| B2 | holes filled with 8-connectivity | `Slice4Tests/` | red | RED (15 started) | 6 |
| B3 | the border kept with includeEdges | `Slice4Tests/` | red | RED (15 started) | 6 |
| B4 | <= for < against distanceEdge | `Slice4Tests/` | red | RED (15 started) | 5 |
| P1 | the first vacuum pattern weighted like the others | `Slice4Tests/` | red | RED (15 started) | 6 |
| P2 | the alignment shift's sign | `Slice4Tests/` | red | RED (15 started) | 6 |
| P3 | sin(f) as f in the damping | `Slice4Tests/` | red | RED (15 started) | 5 |
| K1 | the Gaussian kernel's sigma not squared | `Slice4Tests/` | red | RED (15 started) | 6 |
| K2 | the sigmoid's radii swapped | `Slice4Tests/` | red | RED (15 started) | 6 |
| K3 | the flat kernel not normalised | `Slice4Tests/` | red | RED (15 started) | 12 |
| K4 | the bilinear weights swapped | `Slice4Tests/` | red | RED (15 started) | 15 |
| R1 | Gate B: an all-false Friedel mask accepted | `Slice4Tests/` | red | RED (15 started) | 7 |
| R2 | Gate B: no upper bound on the upsampling factor | `Slice4Tests/` | red | RED (15 started) | 7 |
| R3 | Gate B: of equal maxima the later kept | `Slice4Tests/` | red | RED (15 started) | 7 |

