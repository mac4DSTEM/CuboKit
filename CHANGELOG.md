# Changelog

## Unreleased (0.5.0 candidate)

App-resident computation moves into the kit, each piece pinned to py4DSTEM 0.14.19 (`f050d207`), NumPy or a transcription of
the app source, with the tolerance written in its test (`docs/ptycho.md` has the measured gaps).

- **CuboPtycho preprocessing:** `PtychographyPreparation` (`amplitudes`, `scanPositions`, `objectShape`, `centerPositions`, `initialProbe`, `prepare`) builds the amplitude stack, scan positions, object canvas and starting probe from a `CubeSource`. Bilinear and cubic-spline (SciPy grid-wrap) pattern shifts, py4DSTEM's rotation/transpose/padding, Float32 rotation matrix reproduced. Measured gaps 4.2e-8 (amplitudes), 0 (positions), 6e-8 (probe) at tolerance 2e-6. The app's choices are named options (`PtychographyPreparationOptions.mac4DSTEM`: bilinear shifts, per-axis padding, no re-centring, clipped mean intensity).
- **CuboPtycho engines:** `MixedstatePtychography` (`reconstruct`, `orthogonalize`, `initialProbes` with NumPy-compatible seeded states) and `MultislicePtychography` (`reconstruct`, `propagators`, `identicalSlices`), CPU Double beside `SingleslicePtychography`; new `PtychographyStack` type. Measured gaps 3.9e-7 at 2e-6 over 11 py4DSTEM cases; one slice equals the single-slice engine to 2e-16.
- **CuboPtycho constraints:** `fixProbeAperture` and `probeSupportMask` on all three engines (off by default), pinned to py4DSTEM on each.
- **CuboCore:** `ViewedCube`, `CubeViewSpecification`, `CubeViewGeometry`, `AxisCrop`: scan crop, detector crop and detector bin applied at read time to any `CubeSource` (py4DSTEM `crop_data_real`, `crop_data_diffraction`, `bin_data_diffraction`: crop first, remainder trimmed off the end, blocks summed). Shapes and integer sums exact, float sums within 1e-6 (measured 1.2e-7). `AxisCrop.enclosing` is the app's drag-rectangle arithmetic. New error case `CuboError.invalidView`.
- **CuboPhase:** `DPCViews` (magnitude, angle in [0, 2 pi) with py4DSTEM's `row + i column` direction or the image-frame `atan2(row, column)`, milliradians per pixel) and `RotationConvention` (the sign flip between py4DSTEM's R-Q angle and an image-frame caller's, display text), NumPy-pinned.
- `docs/parity.md`: 79 done, 37 partial, 66 absent of 182 (crop_Q, crop_R, bin_Q done; mixed-state and multislice partial).

### Migration

Source-compatible except that `CuboError` gains a case (`invalidView`): an exhaustive `switch` over it needs a default. `SingleslicePtychography.reconstruct` gains two defaulted parameters (`fixProbeAperture`, `probeSupportMask`). A caller in the app moves: the preparer (`PtychographyPreparer`) to `PtychographyPreparation.prepare(..., options: .mac4DSTEM(...))` after relabelling its axes to py4DSTEM's (its x and y are the kit's column and row); the load view's crop and bin to `ViewedCube` (its bin factors 2, 4, 8 stay the app's policy); the DPC magnitude and angle images to `DPCViews` (Double, not Float32 interleaved) and the rotation sign to `RotationConvention`. Not moved: the colour wheel (a display), the byte-count and residency arithmetic of the load configuration, the UI selection state.

## Unreleased (0.4.0 candidate)

Three additions, each pinned to py4DSTEM 0.14.19 (`f050d207`) with the tolerance written in its test.

- **CuboPtycho:** `SingleslicePtychography.reconstruct` gains `fixProbeCenterOfMass`, `constrainProbeAmplitude`, `probeAmplitudeRelativeRadius` (0.5) and `probeAmplitudeRelativeWidth` (0.05), all off by default so every earlier result is unchanged (py4DSTEM turns the centre-of-mass constraint on by default; here it is the caller's choice). Four new py4DSTEM cases, measured gap 3.6e-7, tolerance 2e-6. Radius outside 0...0.5 or width outside (0, 0.5] is `invalidInput`. Other probe constraints (aperture, Fourier amplitude, aberration fitting, support) remain absent.
- **CuboParallax:** `ParallaxReconstruction.reconstruct` ports `Parallax.reconstruct` (alignment-bin schedule, running reference, Bernstein regularisation, median centring, per-bin error), and `ParallaxReconstruction.process` runs prepare, reconstruct and the optional KDE upsampling and position correction in one call. Seven py4DSTEM cases, gaps 1e-7 to 9e-7 (tolerance 5e-6 on shifts, 2e-6 on images). Named option `emptySchedule = .singlePassAtMinimumBin` is the app's behaviour where py4DSTEM raises. Aberration fit and correction and depth sectioning stay separate calls. Not provided: initial aberration shifts, real-space mask, continue/reset, batching.
- **CuboStrain:** `StrainMapping` (`histogram`, `chooseBasisVectors`, `compute`, `centredPeaks`) is the scan-level pipeline of py4DSTEM's `StrainMap`: Bragg-vector histogram, basis choice, per-position fits, reference, strain maps with coordinate rotation. Pinned to six `StrainMap` runs and five histograms (strain gap 1.8e-15, tolerance 1e-9). `LatticeStrain.chooseBasis` gains an overload with explicit `indexOrigin/G1/G2`. `BraggDiskOptions.tieOrder` (`.rasterOrder` default, `.numpy` for py4DSTEM's order of equal maxima) is new in CuboBragg. `tools/consumer-check` scope checks, which tested a string that no longer existed, now test real scope text.
- `ParallaxPreprocessingResult` gains `scanRows`, `scanColumns`, `paddedWindow` (`correctPositions(window:)` takes `paddedWindow`, as its comment now says).
- `docs/parity.md`: 76 done, 35 partial, 71 absent of 182.

### Migration

Additive for the source. `CuboStrain` now depends on `CuboBragg` (and through it `CuboVirtual` and `CuboParallax`), so a package that imports only `CuboStrain` builds those too; `StrainDirections.maxima` uses `BraggPeak`, so callers of the maxima path also `import CuboBragg`. `ParallaxPreprocessingResult` has new public fields (its initialiser is internal). Everything else keeps its default.

## 0.3.0 (2026-10-16)

0.3.0 adds seven pieces on the CPU, each pinned, with the tolerance stated in its test, to py4DSTEM 0.14.19 (`f050d207`),
scikit-learn 1.4.2 or a transcription of the mac4DSTEM source (Double, except hot pixels and origin re-referencing in Float32): `CuboParallax` (alignment, preprocessing, aberration fit and
correction, depth sectioning, subpixel upsampling), `CuboPtycho`, `CuboStrain`, `CuboBragg`, `CuboEmbedding`, Fourier
iDPC in `CuboPhase`, and hot-pixel filtering and origin re-referencing in `CuboCore`. Where the mac4DSTEM source differs
from the reference, the difference is a named option and the reference is the default. `docs/parity.md` now counts 73
done, 36 partial and 73 absent of 182 rows in scope.

### CuboParallax

- Add the CPU-only `CuboParallax` product: Double-precision matrix-DFT correlation refinement, periodic Fourier translation, and fixed/running-reference stack alignment (default upsampling factor 8). Public inputs and results are Sendable and independent of app types. Four small stacks pin shifts, shifted images, means and DFT patches to py4DSTEM 0.14.19 (`f050d207`), with `1e-10` normwise tolerance.

- Add CuboParallax constant-order bright-field preprocessing: detector selection, reciprocal vectors, radian probe angles, sine-squared blending and centered padding. Default windowed stack retention follows py4DSTEM; `.unwindowed` retains the app KDE input.

- Add CuboParallax aberration fitting (`ParallaxAberrationFitter`: polar-decomposition low order C1/C12/rotation, higher-order global/recursive fits) and CTF phase correction (`ParallaxAberrationCorrection`). Pinned to py4DSTEM 0.14.19 `aberration_fit`/`aberration_correct` (fit coefficients to 2e-4, relative with a unit floor (absolute below 1), measured 6e-7, against a Float32 reference basis, low order and rotation to 1e-10, corrected image to 2e-6). The reference ignores `q_highpass`; the app applies it, so `.reference` is the default and `.apply` is the app option, with `.detectorGrid`/`.selectedCentroid` basis conventions, the app pinned to a NumPy transcription of the app source.

- Add CuboParallax depth sectioning (`ParallaxDepthSectioning`), py4DSTEM `depth_section` semantics. The reference path (defocus-only sign CTF, optional information-limit envelope) is pinned to py4DSTEM 0.14.19 at 2e-7 measured (tolerance 1e-5 of the maximum over all planes, complex64 reference); py4DSTEM's `use_CTF_fit=True` raises a TypeError at that version, so the full-fit CTF (`.fullFit`, the app's behaviour) is pinned to a NumPy transcription of the app source.

- Add CuboParallax subpixel kernel-density upsampling (`ParallaxSubpixel`): the upsample-factor heuristic and bilinear/Lanczos deposition with Gaussian blur and optional sinc low-pass, pinned to py4DSTEM 0.14.19 `subpixel_alignment` (measured gap 4e-7, tolerance 1e-4; the reference accumulates in Float32). The app's KDE source (mac4DSTEM c809679c) was read against the reference and uses the same formulas, including the Lanczos second-axis window quirk, so there is no separate app option; this was a reading, not a pinned comparison. Probe-position correction (`correctPositions`: gradient and checkerboard steps, Gaussian and Butterworth regularisation) is pinned to the reference over three 3-iteration cases (offset gap 1.7e-5, image gap 6e-7, tolerance 1e-4). The app's variant (no regularisation; clamps the step floor only on rejected sites; guards a zero gradient) is not ported as an option.

### CuboPtycho

- Add CuboPtycho CPU probe construction (defocus, two-fold astigmatism, C30) and full-batch single-slice complex-object gradient descent from preprocessed amplitudes, with fixed positions, optional probe updates and pure-phase object constraints. Pinned py4DSTEM 0.14.19 fixtures cover probe pixels and object/probe after three iterations.

### CuboStrain

- Add CuboStrain basis selection from histogram maxima, measured-direction indexing, intensity-squared lattice fits, median references and rotated infinitesimal strain. Separate intensity-once, MAD-reference and cluster-consensus compatibility APIs preserve the original source numerics. Small binary fixtures pin both conventions.

### CuboBragg

- Add the CPU `CuboBragg` product: py4DSTEM's `find_Bragg_disks` for one pattern, a stack or a cube in Double (hybrid cross-correlation with a `ProbeKernel`, `get_maxima_2D` with its filter cascade, parabolic and multicorr subpixel refinement), plus the app's `ProbeRingHint`. Defaults are py4DSTEM's; the mac4DSTEM differences are named options (`relativeReferenceMinimumRadius`, `nonFinitePixels`, `multicorrOrder`, `BraggDiskOptions.mac4DSTEM`, `detectorAdapted`). Thirty py4DSTEM 0.14.19 cases pin positions and intensities to 1e-9 px / 1e-10 (measured 1.5e-12 / 1.3e-13); ten cases pin the options to the unchanged app source in Float32 (1e-3 px / 1e-5; measured 1.8e-4 / 1.2e-6). `GaussianFilter.apply` gains an `edge:` parameter (`.reflect`, scipy's default); the default `.nearest` is unchanged. `CuboParallax` exposes its matrix-DFT refinement to the package (`package`, no public change).

### CuboEmbedding

- Add CuboEmbedding: `PCA` (scikit-learn 1.4.2 `svd_flip` sign convention, covariance divisor rows - 1) and `KMeans` (scikit-learn Lloyd from caller-given centroids), pinned to scikit-learn. mac4DSTEM's Float32 covariance PCA, seeded k-means and binned diffraction feature vector are separate named options pinned to its original source. Binary fixtures under 50 KB; see `docs/embedding.md`.

### CuboPhase and CuboCore

- Add CPU Fourier iDPC integration in CuboPhase, with periodic or centered zero-padded boundaries, explicit regularization, anisotropic sampling and projected-phase calibration. Regression pins use a NumPy float64 transcription of the app Fourier integration; this is separate from the existing py4DSTEM iterative DPC.

- Add CuboCore hot-pixel filtering, wrapped 21-pixel classification and sequential clipped-window median replacement. Float32 mean accumulation matches py4DSTEM, including the contiguous pairwise reduction for a 1×1 detector. Eight small fixtures pin means, masks and all filtered pixels bit for bit to py4DSTEM 0.14.19.

- Add CuboCore `FrameTransform`: detector-position, scan-crop and per-position origin-map re-referencing under crop and bin, with the pixel-centre bin rule `(v+0.5)/bin-0.5`. py4DSTEM moves no origin on crop or bin (it only multiplies the Q pixel size by the bin factor), so these rules are the app's, pinned Float32 bit for bit to a transcription of mac4DSTEM `CalibrationReReference.swift` at c809679c (`tools/rereference/pins.py`). Calibration objects, provenance and invalidation messages stay in the app.

### Migration

Additive: every 0.2.0 product and API is unchanged, and the new modules (`CuboParallax`, `CuboStrain`, `CuboBragg`,
`CuboPtycho`, `CuboEmbedding`) are new library products a consumer adds only if it wants them. One public signature
gains a defaulted parameter: `GaussianFilter.apply` takes `edge:` (default `.nearest`, as before). `CuboParallax`'s
matrix-DFT refinement is shared with `CuboBragg` through `package` access (no public change). Nothing a consumer
calls moves. Full parallax reconstruction (grouping, schedules, regularisation) is not yet provided.
