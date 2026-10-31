# ``CuboParallax``

Double-precision CPU parallax processing: alignment, preprocessing, aberration fit and correction, depth sectioning and subpixel upsampling.

Estimate the translation from a moving image to a reference with
``ParallaxAlignment/shift(reference:moving:rows:columns:upsampleFactor:)``.
Apply it using ``ParallaxAlignment/shifted(_:rows:columns:by:)``, or align a stack using
``ParallaxAlignment/align(_:reference:rows:columns:upsampleFactor:runningAverage:isCancelled:progress:)``.
Images are row-major arrays; positive shifts move content toward increasing row and column indices.
The default factor-8 matrix-DFT refinement follows py4DSTEM 0.14.19.

## Reusing a plan, cancelling, reporting progress

``ParallaxFourierShifter`` builds the FFT plan once for many shifts, spectra and spectrum-level estimates
(``ParallaxAlignment/correlationShift(reference:moving:upsampleFactor:)``), and ``ParallaxDepthSectioner`` transforms the
stack once for many planes. The long calls take optional `isCancelled` and `progress` closures; a cancelled call throws
`CancellationError`. ``ParallaxCorrectionResult`` carries the transfer function it applied, and
``ParallaxPositionCorrectionOptions/variant`` can select the source application's position-correction loop.

## End to end

Run the whole bright-field pipeline with
``ParallaxReconstruction/process(_:shape:reciprocalSampling:scanSampling:energyEV:options:isCancelled:progress:)``: it prepares the stack,
aligns it with ``ParallaxReconstruction/reconstruct(_:options:isCancelled:progress:)`` (py4DSTEM's `Parallax.reconstruct`: the alignment-bin
schedule, running-average reference, Bernstein-basis shift regularisation, median centring and the error per bin) and,
if ``ParallaxProcessingOptions/subpixel`` is set, upsamples with the kernel density estimate and optional position
correction. The result carries every product; aberration fitting, correction and depth sectioning take its aligned
stack, shifts and probe angles as separate calls. Defaults are py4DSTEM's; the pinned gaps are in `docs/parallax-alignment.md`.

## Topics

### Essentials

- <doc:Aligning-images>

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
- ``ParallaxFourierShifter``
- ``ParallaxSpectrum``

### Preprocessing

- ``ParallaxPreprocessing``
- ``ParallaxPreprocessingResult``
- ``ParallaxStreamedPreprocessingResult``
- ``ParallaxStackRetention``

### Aberrations

- ``ParallaxAberrationFitter``
- ``ParallaxAberrationTerm``
- ``ParallaxAberrationFitMethod``
- ``ParallaxLowOrderAberrationOptions``
- ``ParallaxAberrationFitResult``
- ``ParallaxHigherOrderAberrationOptions``
- ``ParallaxHigherOrderAberrationFitResult``
- ``ParallaxPhysicalShift``
- ``ParallaxBasisConvention``
- ``ParallaxAberrationCorrection``
- ``ParallaxHighpassBehavior``
- ``ParallaxCorrectionResult``

### Depth sectioning and subpixel upsampling

- ``ParallaxDepthSectioning``
- ``ParallaxDepthSectioner``
- ``ParallaxDepthCTF``
- ``ParallaxSubpixel``
- ``ParallaxSubpixelResult``
- ``ParallaxPositionCorrectionOptions``
- ``ParallaxPositionCorrectionVariant``
- ``ParallaxPositionCorrectionResult``
- ``ParallaxProbeOffsets``
- ``ParallaxUpsampleLimits``

### Errors

- ``ParallaxError``
- ``ParallaxAlignmentError``
