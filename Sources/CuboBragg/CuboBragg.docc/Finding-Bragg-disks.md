# Finding Bragg disks

Bragg disk detection correlates each diffraction pattern with a kernel built from the probe, keeps the maxima of the
correlation, and refines their positions to a fraction of a pixel. It is py4DSTEM's `find_Bragg_disks`. The kit splits it
into three steps, so a caller can stop after any of them.

1. **Build a kernel.** `ProbeKernel` in CuboVirtual turns a vacuum probe (a ``DetectorImage`` you measured, or a mean
   pattern) into a corner-centred kernel. `ProbeKernel.flat` is py4DSTEM's default; `ProbeKernel.gaussian` and
   `ProbeKernel.sigmoid` subtract an edge trench.
2. **Detect.** ``BraggDisks/detect(_:kernel:options:)`` returns the peaks of one pattern, brightest first. Each
   ``BraggPeak`` has a `row`, a `column` and an `intensity`, in detector pixels. To correlate many patterns against one kernel,
   make a ``BraggDetector`` once and call ``BraggDetector/detect(_:options:)`` on each.
3. **Scan the cube.** ``BraggDisks/detectAll(in:kernel:options:tiling:)`` reads a cube a scan tile at a time and returns
   ``BraggVectors``: one peak list per scan position, indexed by `vectors[scanRow, scanColumn]`.

```swift
import CuboBragg
import CuboCore
import CuboVirtual
import Foundation

let cube = try MappedCube(npy: URL(fileURLWithPath: "cube.npy"))
let mean = try await VirtualDiffraction.mean(of: cube)
let kernel = try ProbeKernel.flat(mean)                                  // corner-centred, sums to 1
let first = try BraggDisks.detect(mean, kernel: kernel)                  // [BraggPeak] of one pattern
let vectors = try await BraggDisks.detectAll(in: cube, kernel: kernel)   // BraggVectors for the whole scan
let peaks = vectors[0, 0]                                                // the peaks at scan position (0, 0)
```

``BraggDiskOptions`` defaults are py4DSTEM's. Choose the subpixel method with ``BraggSubpixel`` and the Float32 path with
`options.precision = .single` (``BraggPrecision``). Float32 needs a detector size vDSP supports; ask first with
``BraggDetector/supportsSinglePrecision(detectorRows:detectorColumns:)``.

## When a result is short or empty

``BraggDetector/detectWithFunnel(_:options:)`` returns the peaks and a ``BraggFunnel``, the count of maxima left after
each stage of the selection. The stage where the count drops to zero names the option to change.

For a whole cube, ``BraggDisks/detectAll(in:kernel:options:tiling:failurePolicy:isCancelled:progress:)`` can record a pattern
that fails (``BraggFailurePolicy/recordEmpty``) instead of stopping, and returns a ``BraggScanResult`` that lists the
failed positions.

The detected peaks are in detector pixels from the corner. To measure strain, centre them on the lattice origin with
`StrainMapping.centredPeaks` in CuboStrain.
