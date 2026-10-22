# Aligning images

Parallax reconstruction starts from a stack of bright-field images, one for each detector pixel inside the bright-field
disk, and shifts each onto a reference. ``ParallaxAlignment`` is the first step, and works on any two images of the same
size, so you can use it before running the whole pipeline.

Images are row-major `[Double]` arrays. ``ParallaxAlignment/shift(reference:moving:rows:columns:upsampleFactor:)``
estimates the translation that relates `moving` to `reference`: a cross-correlation peak found on the pixel grid, then
refined by a matrix Fourier transform to `1 / upsampleFactor` of a pixel (factor 8 by default, as in py4DSTEM 0.14.19).
It returns a ``ParallaxShift`` with a `row` and a `column`. Positive values move content toward increasing row and column
indices. A constant image has no correlation peak, so the call throws.

``ParallaxAlignment/shifted(_:rows:columns:by:)`` applies a shift in the Fourier domain.

```swift
import CuboParallax

let rows = 64, columns = 64
var reference = [Double](repeating: 0, count: rows * columns)
for row in 28..<36 { for column in 28..<36 { reference[row * columns + column] = 1 } }   // a square
let moving = try ParallaxAlignment.shifted(reference, rows: rows, columns: columns,
                                           by: ParallaxShift(row: 2.5, column: -1))
let found = try ParallaxAlignment.shift(reference: reference, moving: moving, rows: rows, columns: columns)
print(found.row, found.column)                                       // the size of the shift that was applied; the sign follows the convention above
```

To align a whole stack, ``ParallaxAlignment/align(_:reference:rows:columns:upsampleFactor:runningAverage:isCancelled:progress:)``
takes the images, a reference and the same factor. It accepts `isCancelled` and `progress` closures, and a cancelled call
throws `CancellationError`. To estimate many shifts of the same size, ``ParallaxFourierShifter`` builds the FFT plan once.

For the whole bright-field pipeline, see ``ParallaxReconstruction/process(_:shape:reciprocalSampling:scanSampling:energyEV:options:isCancelled:progress:)``.
