# Changelog

## Unreleased

- Add CPU Fourier iDPC integration in CuboPhase, with periodic or centered zero-padded
  boundaries, explicit regularization, anisotropic sampling and projected-phase calibration.
  Regression pins use a NumPy float64 transcription of the app Fourier integration;
  this is separate from the existing py4DSTEM iterative DPC.

- Add the CPU-only `CuboParallax` product: Double-precision matrix-DFT correlation refinement,
  periodic Fourier translation, and fixed/running-reference stack alignment (default upsampling factor 8).
  Public inputs and results are Sendable and independent of app types. Four small stacks pin shifts,
  shifted images, means and DFT patches to py4DSTEM 0.14.19 (`f050d207`), with `1e-10` normwise tolerance.

Migration: additive; existing products and APIs are unchanged. Full parallax reconstruction is not yet provided.
