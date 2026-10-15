# Slice 3 mutants, 2026-10-15

Registration slice 3, section 6. `tools/slice-3/mutants.py` (slice 1's runner) on exported copies. Record: the run on the Gate B fixes (`63dbfef`), 17 red; two survivors registered as such: F2 (the robust mask reset each round: an outlier stays an outlier, so the cumulative and the reset masks agree on every fixture) and G8 (the NaN argmax rule: with a NaN anywhere the centre of mass is NaN whatever the argmax, an equivalent mutant). Before it: G1 survived on the synthetic sigmas (closed by a radius test); G9 was first written as a different change and was redefined as the order that actually failed (the sequential −R…R sum), then red.

# Mutants on 63dbfef (19 planned)

| id | mutant | catchers | expected | result | seconds |
|---|---|---|---|---|---|
| G1 | Gaussian radius int(4σ) without + 0.5 | `Slice3Tests/` | red | RED (13 started) | 10 |
| G2 | mode wrap instead of nearest (rows) | `Slice3Tests/` | red | RED (13 started) | 3 |
| G8 | the NaN argmax rule dropped | `Slice3Tests/` | red | GREEN (13 started) | 3 |
| G9 | the plain left-to-right kernel sum instead of scipy's pairs | `Slice3Tests/` | red | GREEN (13 started) | 3 |
| G4 | the last argmax instead of the first | `Slice3Tests/` | red | RED (13 started) | 3 |
| G5 | <= for < at r · rscale | `Slice3Tests/` | red | RED (13 started) | 3 |
| G6 | rscale ignored | `Slice3Tests/` | red | RED (13 started) | 3 |
| G7 | the radius from the mean pattern | `Slice3Tests/` | red | RED (13 started) | 7 |
| F1 | parabola's rc term dropped | `Slice3Tests/` | red | RED (13 started) | 8 |
| F2 | the robust mask reset each round | `Slice3Tests/` | survives | GREEN (13 started) | 4 |
| F3 | the robust mean over the kept positions only | `Slice3Tests/` | red | RED (13 started) | 4 |
| C1 | centring the column ignored (circle) | `Slice3Tests/` | red | RED (13 started) | 5 |
| C2 | the pixel size multiplied (circle radius) | `Slice3Tests/` | red | RED (13 started) | 3 |
| S1 | the mask shift's sign flipped | `Slice3Tests/` | red | RED (13 started) | 3 |
| S2 | rounding half away from zero | `Slice3Tests/` | red | RED (13 started) | 3 |
| S3 | bilinear weights swapped | `Slice3Tests/` | red | RED (13 started) | 3 |
| S4 | the Fourier shift's sign flipped | `Slice3Tests/` | red | RED (13 started) | 3 |
| S5 | the shifted maximum seeded at 0 | `Slice3Tests/` | red | RED (13 started) | 3 |
| S6 | the shifted mean divided by the scan rows | `Slice3Tests/` | red | RED (13 started) | 3 |

unexpected: G8, G9
