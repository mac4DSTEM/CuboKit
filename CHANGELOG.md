# Changelog

## Unreleased

- Add CuboEmbedding: `PCA` (scikit-learn 1.4.2 `svd_flip` sign convention, covariance divisor rows - 1) and `KMeans`
  (scikit-learn Lloyd from caller-given centroids), pinned to scikit-learn. mac4DSTEM's Float32 covariance PCA,
  seeded k-means and binned diffraction feature vector are separate named options pinned to its original source.
  Binary fixtures under 50 KB; see `docs/embedding.md`.

- Add CuboStrain basis selection from histogram maxima, measured-direction indexing,
  intensity-squared lattice fits, median references and rotated infinitesimal strain.
  Separate intensity-once, MAD-reference and cluster-consensus compatibility APIs
  preserve the original source numerics. Small binary fixtures pin both conventions.

- Add CuboCore hot-pixel filtering, wrapped 21-pixel classification and sequential
  clipped-window median replacement. Float32 mean accumulation matches py4DSTEM,
  including the contiguous pairwise reduction for a 1×1 detector. Eight small fixtures
  pin means, masks and all filtered pixels bit for bit to py4DSTEM 0.14.19.

- Add CPU Fourier iDPC integration in CuboPhase, with periodic or centered zero-padded
  boundaries, explicit regularization, anisotropic sampling and projected-phase calibration.
  Regression pins use a NumPy float64 transcription of the app Fourier integration;
  this is separate from the existing py4DSTEM iterative DPC.

- Add the CPU-only `CuboParallax` product: Double-precision matrix-DFT correlation refinement,
  periodic Fourier translation, and fixed/running-reference stack alignment (default upsampling factor 8).
  Public inputs and results are Sendable and independent of app types. Four small stacks pin shifts,
  shifted images, means and DFT patches to py4DSTEM 0.14.19 (`f050d207`), with `1e-10` normwise tolerance.

Migration: additive; existing products and APIs are unchanged. Full parallax reconstruction is not yet provided.

- Add CuboPtycho CPU probe construction (defocus, two-fold astigmatism, C30) and
  full-batch single-slice complex-object gradient descent from preprocessed amplitudes,
  with fixed positions, optional probe updates and pure-phase object constraints.
  Pinned py4DSTEM 0.14.19 fixtures cover probe pixels and object/probe after three iterations.

Migration: additive API; preprocessing and other reconstruction methods remain unsupported.
