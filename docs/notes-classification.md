# Notes: Bragg-vector classification, Featurization and the Bragg virtual image (suite task 37)

For the session that merges CHANGELOG.md, docs/parity.md, docs/open-items.md and docs/status.md. Nothing in those files was touched.
Reference: py4DSTEM 0.14.19 (`f050d207`) with the environment of `../References/python/py4dstem` (NumPy 2.5.4, scikit-learn 1.9.1,
SciPy 1.18.1, scikit-image 0.26.0). Pins: `tools/classification/pins.py` (`featurization`, `classification`, `virtualimage`, `all`;
one BLAS thread, so a regeneration is byte for byte repeatable), fixtures `Tests/CuboEmbeddingTests/Fixtures/featurization.json`,
`Tests/CuboBraggTests/Fixtures/classification.json`, `virtualimage.json`. py4DSTEM itself produced every number; no Swift is used
to generate a pin. `swift test -c release --parallel --skip CuboKitParityTests` is green, `tools/consumer-check/run.sh` passes
(143 checks, no warnings; no check was added for the new API: see "Suggested").

## CHANGELOG bullets (suggested for the next minor, 0.9.0; one Migration sentence: "Additive")

- **CuboEmbedding `Featurization`** (py4DSTEM `process/classification/featurization.py`) on a feature matrix, one row per scan position: `pca(components:)` (scikit-learn 1.9.1's component signs, the largest-magnitude entry of each component positive), `nmf(maxComponents:numModels:mergeThreshold:maxIterations:randomSeed:saveAllModels:isCancelled:)` (`_nmf_single`, including the merge of correlated weight columns and its upstream quirks), `gmm(covarianceTypes:components:numModels:randomSeed:isCancelled:)` (`_gmm_single`, every fit kept), `minMaxScaled`, `shiftedPositive`, `reduceFeatures` (mean, median, maximum of chosen columns), `concatenate`, and `classPatterns(weights:patterns:pixelCount:threshold:)` (the weighted class patterns of `get_class_DPs`).
- **CuboEmbedding `NMF`**: scikit-learn's `NMF` (coordinate descent, Frobenius, no regularisation) with the NNDSVD, NNDSVDa, NNDSVDar, random and custom initialisations, scikit-learn's stopping rule and iteration count. **`GaussianMixture`**: scikit-learn's `GaussianMixture` with the four covariance types, the k-means initialisation (k-means++ seeding drawn from NumPy's `RandomState`), EM, `bic`, responsibilities. **`NumPyRandomState`**: NumPy's legacy `RandomState(seed)` (MT19937: `randomSample`, `randint`, `standardNormal`, `choice(probabilities:)`), bit identical. `FeaturizationError`, `FeatureMatrix`. A new C entry `cubo_embedding_svd` (LAPACK `dgesdd`) in `CEmbeddingLAPACK`.
- **CuboBragg `BraggVectorClassification`** (py4DSTEM `braggvectorclassification.py`): `peakLabels` (`get_braggpeak_labels_by_scan_position`), the data matrix X (boolean or intensity), `initialClasses` (`get_initial_classes`, two- or three-point function), `setInitialClassesByCooccurrence`, `setInitialClasses(images:)`, `nmf`, `split` and `split(classIndex:)`, `merge` (`merge_iterative`), `merge(classIndices:_:)`, `removeClass`, `accept`/`reject` with the candidate `W_next`/`H_next`, the class getters, and `classPattern` / `classPatternWithoutBraggScattering` (`classutils.get_class_DP` and `get_class_DP_without_Bragg_scattering`, integer and Fourier shifts, dark reference). The iterative steps take `isCancelled`.
- **CuboBragg `BraggVectors.virtualImage(detector:)`** (`get_virtual_image`): `.allPeaks`, `.circle(center:radius:)`, `.annulus(center:inner:outer:)` over the vectors in detector coordinates (the calibrations are the caller's), NumPy's pairwise sum order.
- Tests: `FeaturizationTests` (12), `BraggVectorClassificationTests` (11), `BraggVirtualImageTests` (2), `np.allclose` defaults throughout, worst measured gaps below.

Migration: none (additive).

## Measured agreement (worst |gap| / (atol + rtol·|expected|) with `np.allclose` defaults rtol 1e-5, atol 1e-8; 1 is the limit)

| Group | Worst ratio | Notes |
|---|---|---|
| `NumPyRandomState` draws (`random_sample`, `randint`, `standard_normal`, `choice`, high seeds, 400 draws across a refill) | 0 (bit identical) | the normal generator needs NumPy's fused `x1·x1 + x2·x2` (5000 draws checked against NumPy) |
| `NMF.fit`, 13 cases (nndsvda, nndsvd, nndsvdar, random, wide, custom at 1/2/7/200/500 iterations, zero column) | 4.5e-7 | iteration counts and `converged` equal in every case |
| `Featurization.pca` (tall and wide) | 8.0e-9 | |
| `Featurization.nmf`, 11 runs / 18 models (and the one-column error case) (default seeds, seeded, best of models, merging at 0.99/0.9/0.6/0.3) | 1.4e-6 | model shapes (the merged column counts) equal |
| `Featurization.gmm`, 6 runs / 35 fits (full, tied, diag, spherical; 2 to 6 components; seeds from `RandomState(42)`) | 1.0e-3 | labels, `n_iter_`, `converged_` equal everywhere; weights, means, covariances, precision Cholesky, responsibilities, `bic`, lower bound compared |
| Scalers, `shift_positive`, mean/median/max feature, `classPatterns` | 0 | |
| `BraggVectorClassification`: labels, X, initial classes (7 parameter sets), W of the initial states | 0 (exact) | |
| initial H (least squares, incl. a rank-deficient and a wide case) | 3.5e-8 | |
| `nmf` (1, 3, 4, 200 sweeps) | 2.3e-8 | |
| `split`, `split(classIndex:)`, `removeClass`, `merge(classIndices:_:)` | 0 | the connected-region order matched skimage's on every pin |
| `merge` (iterative; 6 parameter sets incl. the dropped-partner quirk) | 3.3e-11 | |
| `classPattern` (7), `classPatternWithoutBraggScattering` (4), integer and Fourier shifts, ties to even | 5.0e-11 | |
| `virtualImage` (13 cases: all modes, empty range, a NaN intensity, 150 peaks at one position) | 0 | |

## parity.md rows to change or add (columns: py4DSTEM | CuboKit | State | Mark | Tolerance (source))

Change the row in "Bragg vectors and disk detection":

| `BraggVectors.get_virtual_image` | `BraggVectors.virtualImage(detector:)` | present: `.allPeaks`, `.circle`, `.annulus` over vectors already in detector coordinates; the four calibration flags are the caller's | useful | `test_braggvectors.py` (counts and shapes); np.allclose defaults, measured 0 |

Replace the single `process.classification.*` row in "Further API audited" (line "missing") by:

| `BraggVectorClassification.__init__` (`braggpeaks`, `Qx`, `Qy`, `X_is_boolean`, `max_dist`), `get_braggpeak_labels_by_scan_position` | `BraggVectorClassification.init(vectors:referenceRows:referenceColumns:booleanData:maximumDistance:)`, `.peakLabels(...)` | present: labels and X exact (rows are `qx`) | useful | none upstream; labels exact, X exact |
| `get_initial_classes`, `get_initial_classes_by_cooccurrence`, `get_initial_classes_from_images` | `.initialClasses(...)`, `.setInitialClassesByCooccurrence(...)`, `.setInitialClasses(images:)` | present: class sets exact, W exact, H by minimum-norm least squares (Jacobi SVD against `lstsq`) | useful | none upstream; np.allclose defaults, measured 3.5e-8 of the bound |
| `nmf`, `split`, `split_by_class_index`, `merge_iterative`, `merge_by_class_index`, `remove_class`, `accept`, `reject`, `get_class*`, `get_candidate_class*` | `.nmf`, `.split`, `.split(classIndex:)`, `.merge`, `.merge(classIndices:_:)`, `.removeClass`, `.accept`, `.reject`, `classPeakWeights`, `classImage`, `candidate*` | present: `merge()` itself is an empty method in 0.14.19; `merge` here is `merge_iterative` (see deviations); a lone class makes `merge` throw where NumPy raises `IndexError` | useful | none upstream; np.allclose defaults, measured 2.3e-8 (nmf), 0 (split), 3.3e-11 (merge) |
| `classutils.get_class_DP`, `get_class_DP_without_Bragg_scattering` | `BraggVectorClassification.classPattern(of:...)`, `.classPatternWithoutBraggScattering(of:...)` | present: integer shifts (ties to even) and Fourier shifts, dark reference | useful | none upstream; np.allclose defaults, measured 5e-11 |
| `Featurization` (`PCA`, `NMF` incl. `_nmf_single`, `GMM` incl. `_gmm_single`, `MinMaxScaler`, `shift_positive`, `mean/median/max_feature`, `concatenate_features`, `get_class_DPs` (the `pca`, `consensus_clusters`, `spatially_separated_ims` branches), `get_class_ims`) | `Featurization.pca/nmf/gmm/minMaxScaled/shiftedPositive/reduceFeatures/concatenate/classPatterns`, `NMF`, `GaussianMixture`, `NumPyRandomState` | partial: scikit-learn 1.9.1 numbers; see deviations | useful | none upstream; np.allclose defaults, measured 4.5e-7 (NMF), 1.4e-6 (iterative), 1e-3 (GMM) |
| `Featurization.from_braggvectors`, `ICA`, `RobustScaler`, `spatial_separation`, `consensus`, `add_features`, `delete_features` | — | missing: `from_braggvectors` (peaks binned onto a grid, calibrated vectors) needs the Bragg types in CuboEmbedding; ICA and the two clustering steps are not ported | useful | none upstream |

Count (0.8.0 table): Bragg vectors and disk detection present 3 to 4, missing 11 to 10; Further API audited present 2 to 5,
partial 10 to 11, missing 12 to 12 (one row removed, one added). Total present 121 to 125, partial 38 to 39, missing 50 to 49
(213 rows).

## open-items.md

Nothing in the file mentions classification, so nothing to change. Suggested new items:

1. **`CuboBragg` carries its own small coordinate-descent NMF** (`ClassificationNumerics.coordinateDescentNMF`, custom start only,
   60 lines) because `CuboBragg` does not depend on `CuboEmbedding`, where the public `NMF` lives. With a `Package.swift` edge
   `CuboBragg -> CuboEmbedding` the copy goes away (`NMF.fit(initialization: .custom(...))` is the same arithmetic). The same
   goes for the Jacobi `leastSquares` against `NMF`'s LAPACK SVD: a second SVD in the kit.
2. **`Featurization.from_braggvectors`, ICA, `RobustScaler`, `spatial_separation` and `consensus` are not ported** (see the parity row).
3. **The k-means start of `GaussianMixture` uses `KMeans.fit`, a transcription of scikit-learn 1.4.2's Lloyd loop**, with data
   centred as 1.9.1 does; every pin agrees on labels, but the Lloyd loop of 1.9.1 was compared only through those pins, and the
   k-means++ distances use scikit-learn's expanded form (`-2xy + |x|² + |y|²`), so two candidates that tie to rounding could be
   ordered differently from NumPy's BLAS.
4. **Upstream is not bit repeatable here:** `NMF(init=None)` with `random_state=None` (the call `BraggVectorClassification.nmf`
   does not make, but a user's default does) runs a randomised SVD without a seed, so the same call differs in the 14th digit between runs.
   The pins use `random_state=0`; `Featurization.NMF` seeds it itself.

## Deviations from py4DSTEM (all documented in the doc comments)

- `BraggVectorClassification.merge`: py4DSTEM 0.14.19's `merge()` has only a docstring; the algorithm it describes is
  `merge_iterative`, which is what `merge` runs (`maximumRounds` gives the single round the docstring describes). With one class
  NumPy's `corrcoef` returns a 0-d array and the candidate indexing raises; `merge` throws `CuboBraggError.invalidOption` there.
  Upstream's quirk that a class sitting in a skipped candidate pair is also dropped is kept (pinned).
- `merge(classIndices: i, j)` follows upstream, which deletes column `j` before writing the merged class at index `i`; for `i > j`
  that overwrites the wrong class, and `i` past the end throws (NumPy `IndexError`). Pinned both ways.
- `Featurization.nmf`: `_nmf_single` appends/keeps the weights of the last merge, not of the last fit, deletes only the last merged
  column (`inds[n, 1]` with the loop's last `n`), and fails on a single weight column (`np.tril` of a scalar); a throw
  (`FeaturizationError.invalidInput`) here. A zero-norm singular vector in NNDSVD, which gives NaN upstream, throws
  `FeaturizationError.degenerateInitialisation`; inside the iterative loop it ends the model like upstream's `FloatingPointError` handler.
- NNDSVD uses the exact LAPACK SVD (`dgesdd`), scikit-learn a randomised SVD (7 or 4 power iterations) that converges to it: the
  initial factors differ at about 1e-10 relative and the converged factorisation agrees to 4.5e-7 of the allclose bound.
- `Featurization.pca` goes through `PCA.fit(variant: .scikitLearn)` (covariance eigendecomposition) with 1.9.1's v-based sign
  rule; scikit-learn picks full SVD or the covariance solver by shape (never randomised for the tall/small sizes pinned). `n_components`
  as a fraction or `"mle"` is not offered. The pre-existing `PCA.fit` docs still say 1.4.2 (u-based signs): `Featurization.pca`
  applies the 1.9.1 rule on top.
- `Featurization.classPatterns` takes any weight matrix. py4DSTEM's `get_class_DPs` `"nmf"` and `"gmm"` branches index
  `datacube.data[x, 0]` (a pattern's first row, broadcast) and the `"gmm"` branch loops to `max(label)` exclusive, which leaves out
  the last class; neither is reproduced. A class with no position above the threshold yields NaN (0/0), as upstream.
- `NumPyRandomState` is bit identical on arm64 NumPy (the polar normal needs the fused multiply-add NumPy's build uses); an x86 NumPy
  build that does not fuse could differ in the last bit of a normal.
- `GaussianMixture.fit` is `n_init=1`, `init_params="kmeans"`, `warm_start=False` only (what `Featurization.GMM` uses); `reg_covar`,
  `tol`, `max_iter` are options. Not offered: other `init_params`, `weights_init`, `means_init`, `precisions_init`.
- `get_virtual_image` does not apply `center`, `ellipse`, `pixel`, `rotate`: there is no calibration state on `BraggVectors`;
  the caller passes calibrated vectors. No `VirtualImage` data object or tree attach: `BraggVirtualImage` is a value.
- Rows and columns: py4DSTEM's `qx`, `R_Nx` are rows, `qy`, `R_Ny` columns; the Voronoi points are `referenceRows`/`referenceColumns`.
- No Metal path (Float64 CPU only), as asked.

## Package.swift

No change is needed for the tests. Suggested (not done): `CuboBragg` depends on `CuboEmbedding` to drop the duplicated NMF solver
(open item 1). The new C function lives in the existing `CEmbeddingLAPACK` target (`Sources/CEmbeddingLAPACK/SymmetricEigen.c`
and its header).

## Suggested follow-ups for the merging session

- `tools/consumer-check`: add checks for `Featurization`, `NMF`, `GaussianMixture`, `NumPyRandomState`, `BraggVectorClassification`,
  `BraggVectors.virtualImage` (the public API compiles today; the run reports 143 checks, none of them new).
- `docs/status.md`: Bragg vectors now carry classification and the virtual image; embedding carries Featurization.
- DocC: no article was added for the new types.
