# ``CuboBragg``

Bragg disk detection by probe-kernel template matching, on the CPU in Double.

``BraggDisks/detect(_:kernel:options:)`` follows py4DSTEM's `find_Bragg_disks`: a hybrid cross-correlation of each
pattern with a corner-centred kernel (build one with `ProbeKernel` in CuboVirtual), the maxima of the smoothed
correlation, the filter cascade, and parabolic or matrix-DFT (multicorr) subpixel refinement. Peaks are `row` and `column`
(py4DSTEM's `qx` and `qy`). Use ``BraggDetector`` to correlate many patterns against one kernel, and
``BraggDisks/detectAll(in:kernel:options:tiling:)`` for a whole cube.

``BraggDiskOptions`` defaults are py4DSTEM's. The mac4DSTEM behaviours that differ are named options, all off by default:
``BraggDiskOptions/relativeReferenceMinimumRadius``, ``BraggDiskOptions/nonFinitePixels``,
``BraggDiskOptions/multicorrOrder``, and the helpers ``BraggDiskOptions/mac4DSTEM`` and
``BraggDiskOptions/detectorAdapted(detectorRows:detectorColumns:probeRadius:base:)``.
``ProbeRingHint`` is the app's suggestion for a ring-structured probe's radius.

## Topics

### Detection

- ``BraggDisks``
- ``BraggDetector``
- ``BraggMaxima``
- ``BraggVectors``
- ``BraggPeak``

### Options

- ``BraggDiskOptions``
- ``BraggSubpixel``
- ``BraggPeakOrder``
- ``BraggNonFinitePixels``

### Probe

- ``ProbeRingHint``

### Errors

- ``CuboBraggError``
