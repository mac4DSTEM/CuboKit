# PCA and k-means (CuboEmbedding)

`PCA` and `KMeans` take row-major Double matrices (`rows × columns`) and need no scan shape, tile streaming, settings,
progress or provenance. Streaming a cube into a feature matrix stays with the caller. All public values are Sendable.
Reference: scikit-learn 1.4.2 (NumPy 2.5.0, SciPy 1.18.0, Python 3.12.4); a different scikit-learn is not a reference.

## Primary path: scikit-learn

- `PCA.fit(_:rows:columns:components:)` = `PCA(n_components=k, svd_solver="full")`. The covariance (divisor rows − 1) is
  built with `dsyrk` and decomposed with LAPACK `dsyevd`; scikit-learn uses an SVD of the centred data. Same subspace and
  variances, but accuracy per component is limited by the gap to its neighbours (error about 1e-16 · λ₁ / gap).
- **Sign convention: scikit-learn 1.4.2's `svd_flip` with `u_based_decision=True`.** The score (`U·S`) of largest magnitude
  of each component is made positive; the first index wins an exact tie. This is why a component is not simply the
  eigenvector LAPACK returns. scikit-learn ≥ 1.5 switched PCA to the `v`-based decision; this kit is pinned to 1.4.2.
- `explainedVariance` (divisor rows − 1), `explainedVarianceRatio` (over the sum of all variances), `singularValues`,
  `mean`, `scores` (= `fit_transform`) and `transform`. `components` must be in `1...min(rows, columns)`, rows ≥ 2.
- `KMeans.fit(_:rows:columns:initialCentroids:maxIterations:tolerance:)` = `KMeans(init=<centroids>, n_init=1,
  algorithm="lloyd", max_iter, tol)`: strict-convergence and centre-shift stopping (tolerance scaled by the mean
  population variance of the columns), empty clusters moved to the farthest rows, final relabelling when the stop was
  not on repeated labels, inertia, `n_iter_`. Seeding (`k-means++`, `n_init`, random states) is deliberately absent:
  the caller passes the centroids, which is what makes the result reproducible against the reference.
  Not pinned: several empty clusters in one iteration (scikit-learn's order comes from `argpartition`), exact-tie
  assignment.

## Options pinned to mac4DSTEM (c809679c, `Core/Analysis/DiffractionEmbedding.swift`)

| Behaviour | scikit-learn path | mac4DSTEM option |
|---|---|---|
| Precision | Double | `PCA.Variant.diffractionEmbeddingApp`: Float32 data, mean, components, ratios, scores |
| Covariance divisor / variances | rows − 1 | rows (`explainedVariance` = eigenvalues of that covariance; ratios are the same) |
| Component sign | `svd_flip` (u-based) | raw LAPACK sign; `PCA` tests align sign before comparing |
| Component count | error unless `1...min(rows, columns)` | clamped to `1...min(columns, rows − 1)` |
| Ratio floor | none | clamped at 0 |
| k-means seeding | caller's centroids | `KMeans.fitSeeded`: k-means++ with splitmix64 (`seed &+ 0x5A17`) through Swift's `random(in:using:)` |
| k-means stopping | labels repeat or shift ≤ tol·variance, up to 300 | labels repeat, up to 50 passes, no tolerance |
| Empty cluster | farthest rows from their own centroids, before the update | farthest row from its pre-update centroid, assigned after the update |
| Features | none | `DiffractionFeatures.binnedVector`: log1p, per-pattern maximum, box bins |

`DiffractionFeatures` has no scikit-learn or py4DSTEM counterpart. The seeded k-means result depends on the Swift standard
library's range sampling, so its pin holds for that sampling algorithm.

## Pins

`tools/embedding/pins.py` (run as below) calls the real scikit-learn functions, and compiles the **original** app
numerics (sliced from the git object at the commit above, access qualifiers removed, with minimal shims for the
cancellation token and error type) with `swiftc`, then runs them. It never imports or executes CuboEmbedding.

```sh
../../tools/memcap.sh -- ../../References/CuboKit/py4dstem-venv/bin/python -I -B tools/embedding/pins.py
```

It checks Python 3.12.4, NumPy 2.5.0 and scikit-learn 1.4.2 and records versions, SHA256 of the scikit-learn sources and of
the app file in `Fixtures/embedding.json`. Data: planted-spectrum Gaussian matrices (60 × 8, and 6 × 10 with fewer rows than
columns), three Gaussian blobs (90 × 3), a pattern set with a negative count, an all-zero pattern, and a detector smaller
than the binned size. Cases cover an empty cluster in iteration 1, the iteration cap, and the centre-shift stop. Binary
little-endian f64, 41 KB in all. The oracle peaked at 371 MB under the 8 GB memcap.

Measured gaps against tolerances (tests print them):

| Quantity | Measured | Tolerance |
|---|---|---|
| PCA components | 2.5e-12 | 1e-9 |
| PCA variances, ratios, singular values | 8.9e-15, 2.2e-16, 5.0e-14 | 1e-10, 1e-12, 1e-10 |
| PCA scores | 6.2e-15 | 1e-9 |
| k-means centroids / inertia (relative) | 1.8e-15 / 2.9e-16 | 1e-11 / 1e-13 |
| k-means labels, iteration counts | equal | equal |
| App PCA (mean, components, eigenvalues, ratios, coordinates) | 0 | 0, 1e-6, 1e-10, 1e-6, 1e-4 |
| App k-means centroids and labels | 0 | 0 |
| Binned vector | 2.4e-7 (values to 2.9) | 1e-5 relative to max(1, largest value) |

The binned vector tolerance covers `vvlog1pf` against `log1pf` and the summation order. These pins establish agreement
with the reference and recovery of planted structure; they are not performance or microscope measurements.
