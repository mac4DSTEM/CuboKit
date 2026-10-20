# ``CuboParallax``

Double-precision CPU parallax processing: alignment, preprocessing, aberration fit and correction, depth sectioning and subpixel upsampling.

Estimate the translation from a moving image to a reference with
``ParallaxAlignment/shift(reference:moving:rows:columns:upsampleFactor:)``.
Apply it using ``ParallaxAlignment/shifted(_:rows:columns:by:)``, or align a stack using
``ParallaxAlignment/align(_:reference:rows:columns:upsampleFactor:runningAverage:)``.
Images are row-major arrays; positive shifts move content toward increasing row and column indices.
The default factor-8 matrix-DFT refinement follows py4DSTEM 0.14.19.

## End to end

Run the whole bright-field pipeline with
``ParallaxReconstruction/process(_:shape:reciprocalSampling:scanSampling:energyEV:options:)``: it prepares the stack,
aligns it with ``ParallaxReconstruction/reconstruct(_:options:)`` (py4DSTEM's `Parallax.reconstruct`: the alignment-bin
schedule, running-average reference, Bernstein-basis shift regularisation, median centring and the error per bin) and,
if ``ParallaxProcessingOptions/subpixel`` is set, upsamples with the kernel density estimate and optional position
correction. The result carries every product; aberration fitting, correction and depth sectioning take its aligned
stack, shifts and probe angles as separate calls. Defaults are py4DSTEM's; the pinned gaps are in `docs/parallax-alignment.md`.

## Topics

### End to end

- ``ParallaxReconstruction``
- ``ParallaxReconstructionOptions``
- ``ParallaxReconstructionResult``
- ``ParallaxEmptyScheduleBehavior``
- ``ParallaxProcessingOptions``
- ``ParallaxSubpixelOptions``
- ``ParallaxProcessingResult``

### Alignment

- ``ParallaxAlignment``
- ``ParallaxShift``
- ``ParallaxAlignmentResult``

### Preprocessing

- ``ParallaxPreprocessing``
- ``ParallaxPreprocessingResult``

### Aberrations

- ``ParallaxAberrationFitter``
- ``ParallaxAberrationTerm``
- ``ParallaxAberrationCorrection``
- ``ParallaxCorrectionResult``

### Depth sectioning and subpixel upsampling

- ``ParallaxDepthSectioning``
- ``ParallaxDepthCTF``
- ``ParallaxSubpixel``
- ``ParallaxSubpixelResult``
- ``ParallaxPositionCorrectionOptions``
- ``ParallaxPositionCorrectionResult``
- ``ParallaxProbeOffsets``
- ``ParallaxUpsampleLimits``

### Errors

- ``ParallaxError``

- ``ParallaxAlignmentError``
