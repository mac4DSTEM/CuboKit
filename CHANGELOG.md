# Changelog

## Unreleased

- Add CuboCore `FrameTransform`: detector-position, scan-crop and per-position origin-map re-referencing under crop and
  bin, with the pixel-centre bin rule `(v+0.5)/bin-0.5`. py4DSTEM moves no origin on crop or bin (it only multiplies the Q
  pixel size by the bin factor), so these rules are the app's, pinned Float32 bit for bit to a transcription of
  mac4DSTEM `CalibrationReReference.swift` at c809679c (`tools/rereference/pins.py`). Calibration objects, provenance
  and invalidation messages stay in the app.

- Add CuboStrain basis selection from histogram maxima, measured-direction indexing,
  intensity-squared lattice fits, median references and rotated infinitesimal strain.
  Separate intensity-once, MAD-reference and cluster-consensus compatibility APIs
  preserve the original source numerics. Small binary fixtures pin both conventions.

- Add CuboParallax subpixel kernel-density upsampling (`ParallaxSubpixel`): the upsample-factor heuristic and bilinear/Lanczos deposition with Gaussian blur and optional sinc low-pass, pinned to py4DSTEM 0.14.19 `subpixel_alignment` (measured gap 4e-7, tolerance 1e-4; the reference accumulates in Float32). The app's KDE source (mac4DSTEM c809679c) was read against the reference and uses the same formulas, including the Lanczos second-axis window quirk, so there is no separate app option; this was a reading, not a pinned comparison. Probe-position correction (`correctPositions`: gradient and checkerboard steps, Gaussian and Butterworth regularisation) is pinned to the reference over three 3-iteration cases (offset gap 1.7e-5, image gap 6e-7, tolerance 1e-4). The app's variant (no regularisation; clamps the step floor only on rejected sites; guards a zero gradient) is not ported as an option.

- Add CuboParallax depth sectioning (`ParallaxDepthSectioning`), py4DSTEM `depth_section` semantics. The reference path (defocus-only sign CTF, optional information-limit envelope) is pinned to py4DSTEM 0.14.19 at 2e-7 measured (tolerance 1e-5 of the maximum over all planes, complex64 reference); py4DSTEM's `use_CTF_fit=True` raises a TypeError at that version, so the full-fit CTF (`.fullFit`, the app's behaviour) is pinned to a NumPy transcription of the app source.

- Add CuboParallax aberration fitting (`ParallaxAberrationFitter`: polar-decomposition low order C1/C12/rotation, higher-order global/recursive fits) and CTF phase correction (`ParallaxAberrationCorrection`). Pinned to py4DSTEM 0.14.19 `aberration_fit`/`aberration_correct` (fit coefficients to 2e-4, relative with a unit floor (absolute below 1), measured 6e-7, against a Float32 reference basis, low order and rotation to 1e-10, corrected image to 2e-6). The reference ignores `q_highpass`; the app applies it, so `.reference` is the default and `.apply` is the app option, with `.detectorGrid`/`.selectedCentroid` basis conventions, the app pinned to a NumPy transcription of the app source.

- Add CuboParallax constant-order bright-field preprocessing: detector selection,
  reciprocal vectors, radian probe angles, sine-squared blending and centered padding.
  Default windowed stack retention follows py4DSTEM; `.unwindowed` retains the app KDE input.

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
