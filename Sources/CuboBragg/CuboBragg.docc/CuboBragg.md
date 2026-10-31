# ``CuboBragg``

Bragg disk detection by probe-kernel template matching, on the CPU in Double, or in Float32 on request.

``BraggDisks/detect(_:kernel:options:)`` follows py4DSTEM's `find_Bragg_disks`: a hybrid cross-correlation of each
pattern with a corner-centred kernel (build one with `ProbeKernel` in CuboVirtual), the maxima of the smoothed
correlation, the filter cascade, and parabolic or matrix-DFT (multicorr) subpixel refinement. Peaks are `row` and `column`
(py4DSTEM's `qx` and `qy`). Use ``BraggDetector`` to correlate many patterns against one kernel, and
``BraggDisks/detectAll(in:kernel:options:tiling:)`` for a whole cube.

Opt in to the faster Float32 path with `options.precision = .single` (``BraggPrecision``): vDSP convolution and complex DFT, a
reusable scratch per concurrent worker, about ten times faster than Double at 256 × 256 and equal to it to about 1e-5 of the
correlation maximum. It runs on every detector size: sizes vDSP's DFT takes (64, 128, 192, 240, 256, ...) directly, any other
through Bluestein's chirp-z, slower and at the same tolerances. ``BraggDetector/detectWithFunnel(_:options:)`` returns the peaks of `detect` with a ``BraggFunnel``, the count of
maxima after each stage of the selection, so a short or empty result can be explained; ``BraggDetector/fillNonFinite(_:rows:columns:)``
is the median fill of non-finite pixels, public.

``BraggDiskOptions`` defaults are py4DSTEM's. The mac4DSTEM behaviours that differ are named options, all off by default:
``BraggDiskOptions/relativeReferenceMinimumRadius``, ``BraggDiskOptions/nonFinitePixels``,
``BraggDiskOptions/multicorrOrder``, and the helpers ``BraggDiskOptions/mac4DSTEM`` and
``BraggDiskOptions/detectorAdapted(detectorRows:detectorColumns:probeRadius:base:)``.
``ProbeRingHint`` is the app's suggestion for a ring-structured probe's radius.

``BraggVectorClassification`` is py4DSTEM's `braggvectorclassification`: peaks labelled by the nearest reference vector,
an NMF of the label matrix, then split, merge, accept and reject steps, and the class images and class patterns.
``BraggVectors/virtualImage(detector:)`` sums the peaks inside a ``BraggVirtualDetector`` at each scan position
(py4DSTEM's `get_virtual_image`) into a ``BraggVirtualImage``.

## Topics

### Essentials

- <doc:Finding-Bragg-disks>

### Detection

- ``BraggDisks``
- ``BraggDetector``
- ``BraggMaxima``
- ``BraggFunnel``
- ``BraggVectors``
- ``BraggPeak``
- ``BraggScanResult``
- ``BraggFailurePolicy``
- ``BraggMasking``

### Options

- ``BraggDiskOptions``
- ``BraggSubpixel``
- ``BraggPeakOrder``
- ``BraggTieOrder``
- ``BraggNonFinitePixels``
- ``BraggPrecision``

### Classification

- ``BraggVectorClassification``

### Virtual images from Bragg vectors

- ``BraggVirtualImage``
- ``BraggVirtualDetector``
- ``BraggDetectorCenter``

### Probe

- ``ProbeRingHint``

### Errors

- ``CuboBraggError``
