# ``CuboParallax``

Double-precision CPU cross-correlation alignment for parallax image stacks.

Estimate the translation from a moving image to a reference with
``ParallaxAlignment/shift(reference:moving:rows:columns:upsampleFactor:)``.
Apply it using ``ParallaxAlignment/shifted(_:rows:columns:by:)``, or align a stack using
``ParallaxAlignment/align(_:reference:rows:columns:upsampleFactor:runningAverage:)``.
Images are row-major arrays; positive shifts move content toward increasing row and column indices.
The default factor-8 matrix-DFT refinement follows py4DSTEM 0.14.19.

## Topics

### Alignment

- ``ParallaxAlignment``
- ``ParallaxShift``
- ``ParallaxAlignmentResult``

### Errors

- ``ParallaxAlignmentError``
