# What a consumer changes to import CuboKit 0.0.1

Written 2026-10-15 for mac4DSTEM, the one consumer today, by the CuboKit session. **mac4DSTEM's own session makes the switch**,
as a tier-1 lane at a tag of its choosing (ROADMAP "Kits", phase 5); nothing here was changed in mac4DSTEM. Read at
mac4DSTEM's pushed `7d6fdbd1` (the extracted files are byte-identical to `7448401b`, the extraction base).

0.0.1 is the owner's tag of slice 1 (it does not exist yet; the commands are on the board's `lanes/cubokit` waiting list).
It is a 0.0.x tag: the API may still move, and mac4DSTEM pins it exactly and bumps on purpose.

## 1. The dependency

```swift
// Package.swift (DSTEMCore's dependencies)
.package(url: "https://github.com/mac4DSTEM/CuboKit", exact: "0.0.1"),
// DSTEMCore target: .product(name: "CuboCore", package: "CuboKit"), .product(name: "CuboVirtual", package: "CuboKit")
```

The Xcode project picks it up through the local package, as SpectroKit's. Licence: GPL-3.0 on both sides. CuboKit carries no
unsafe flags and no resources; `swift build` alone builds it, and its Metal kernels compile at run time from source (no
`.metallib` to bundle). It builds in Swift 6 mode, nonisolated by default; mac4DSTEM's Swift 5 mode with MainActor default
isolation imports it unchanged. Build it optimised in Debug as mac4DSTEM already does for its own Core (the `Develop`
configuration): CuboKit's CPU reducers are many times slower at -Onone.

## 2. The cube: an adapter, not a rewrite

CuboKit has no `FourDArray`, `LoadView`, `DatasetDescriptor` or residency policy, and does not need them: a reduction takes
`some CubeSource`, a public protocol any actor can adopt. The smallest switch is one adapter in mac4DSTEM that keeps the
app's load views, crops, binning, caching and resident cube exactly as they are:

```swift
import CuboCore
// In mac4DSTEM (DSTEMCore): the app's array, seen by the kit.
actor CuboAdapter: CubeSource {
    let array: FourDArray
    nonisolated let shape: CubeShape
    init(_ array: FourDArray, descriptor d: DatasetDescriptor) throws {
        self.array = array
        self.shape = try CubeShape(scanRows: d.ry, scanColumns: d.rx, detectorRows: d.qy, detectorColumns: d.qx)
    }
    func readPattern(row: Int, column: Int) async throws -> DiffractionPattern {
        let p = try await array.pattern(ry: row, rx: column)
        return try DiffractionPattern(detectorRows: p.qy, detectorColumns: p.qx, values: p.pixels)
    }
    func readScanTile(rows: Range<Int>) async throws -> ScanTile {
        let t = try await array.scanTile(yRange: rows)
        return try ScanTile(shape: shape, rows: rows, values: t.pixels)
    }
}
```

`CubeSource`'s reads are `async` (changed before the first tag, 2026-10-15), so the adapter forwards to the
app's asynchronous array without copying the cube; `CuboCoreTests.testAnAsyncAdapterIsACubeSource` holds the pattern. The
layout matches: mac4DSTEM's `[ry][rx][qy][qx]` Float32 is the kit's storage order, and its `y`/`x` are the
kit's row/column.

## 3. The call sites (mac4DSTEM `7d6fdbd1`)

| mac4DSTEM today | With CuboKit |
|---|---|
| `VirtualDetector.tiledImage(data:descriptor:shape:)` in `App/AppState+ResultPresentation.swift:57, 351, 364, 374` (circle, rectangle, point, the scan navigator's annulus) | `VirtualImaging.image(of:geometry:path:tiling:progress:)` |
| `VirtualDetector.tiledRun(data:descriptor:aperture:)` (`AppState+ResultPresentation.swift:358`, the dragged annulus) | the same call with `.annulus(...)`; the analytic `virtualAperture` kernel has no counterpart (one mask kernel serves every geometry) |
| `VirtualDetector.tiledDPStatistics` in `App/AppState+Calibration.swift:30`, `Core/Analysis/OriginCalibration.swift:565, 819` | `VirtualDiffraction.meanAndMax(of:path:tiling:progress:)` |
| `VirtualDetector.makeMask(shape:qy:qx:)` for a detector | `DetectorMask.make(_:detectorRows:detectorColumns:)` |
| `DetectorShape.circle(centerX:centerY:radius:)` | `.circle(centerRow: centerY, centerColumn: centerX, radius:)` (row first; Double) |
| `DetectorShape.annulus(centerX:centerY:inner:outer:)` | `.annulus(centerRow: centerY, centerColumn: centerX, inner:, outer:)` |
| `DetectorShape.rectangle(xMin:xMax:yMin:yMax:)` | `.rectangle(rowMin: yMin, rowMax: yMax, columnMin: xMin, columnMax: xMax)` |
| `DetectorShape.point(x:y:)` | `.point(row: y, column: x)` |
| `FloatImage` (Float) | `ScanImage` (Double, `values`, `scanRows`, `scanColumns`, `path`); the app converts to `Float` for display only |
| `DiffractionPattern(qy:qx:pixels:)` for the mean and max | `DetectorImage` (Double, `detectorRows`, `detectorColumns`) |

What stays in mac4DSTEM: the real-space region masks and selected-area diffraction (`tiledDiffraction`,
`DetectorShape.realSpaceRegion`; CuboKit has no masked virtual diffraction yet), origin measurement and centre of mass
(`tiledMeasuredOrigins`, `tiledCenterOfMass`; slice 2), the presets (`DetectorPreset`), `Aperture` and its snapping, the
tile prefetcher, residency, the Metal engine's other kernels, and every reader.

## 4. Behaviour that changes (each a reason the switch is tier 1)

1. **Refusals instead of silent empty masks.** The kit refuses a point off the detector, a rectangle bound that rounds below
   zero, a negative radius, a non-finite value, and an annulus whose outer radius does not exceed its inner one; mac4DSTEM
   answers some of these with an empty or clipped mask. The rectangle case matters: `AppState+ResultPresentation.swift:364`
   builds `centre ± half`, which goes negative near the detector's top or left edge. Clamp to 0 before calling (the result is
   the same pixels mac4DSTEM selects today, since its `makeMask` clamps).
2. **Masks in Double, as py4DSTEM.** mac4DSTEM's `makeMask` and `virtualAperture` compare in Float32; the kit compares in
   Double, as py4DSTEM's `make_detector` does. They differ only on a pixel whose distance from the centre lies within one
   Float32 ulp of a radius (a fractional radius or centre; the pins hold one such case, a single pixel on a 32 × 32 grid).
   An integer radius at an integer centre, the common case, selects the same pixels.
3. **Virtual images in Double.** mac4DSTEM sums each position in Float32 on the GPU; the kit's default `.cpu` path sums in
   Double: exact for integer cubes (bit for bit with py4DSTEM), within 1e-10 of py4DSTEM's float64 definition for float32
   cubes (measured: identical on the owner's four cubes). Numbers move in their low bits, at most about 1e-6 relative on
   the owner's cubes. `.metal` stays available: its kernels are compensated (Dot2) and sit within 2.4e-13 of the CPU
   path on the same cubes, so the path choice no longer moves a number a reader sees.
4. **The mean pattern in Double, with compensation.** mac4DSTEM's tiled mean sums each tile in Float32 on the GPU and weights
   the tiles in Double; the kit's CPU mean is a compensated Double sum (identical to py4DSTEM's float64 mean on the owner's
   cubes; py4DSTEM's own float32 mean differs from it by up to 1.9e-5 on the Al–Mg–Si cube). Every consumer of the mean
   (the probe-size estimate, the beam proxy, the ring hint, origin calibration's seeds) sees low-bit changes; its own tests
   decide whether any threshold sits that close.
5. **Tiling no longer depends on the machine.** mac4DSTEM sizes tiles from physical memory, so its tiled reducers' low bits
   differ between Macs (accepted in its `FourDArray.scanTileRows` comment); CuboKit's `.automatic` is a function of the
   shape (64 MiB tiles) and its CPU reducers give the same bits at every tiling. Pass `.rows(n)` to reproduce a specific
   grouping; the CPU result does not change.
6. **The max pattern** is unchanged (exact everywhere).

## 5. How mac4DSTEM checks the switch

- Its existing parity harness `tools/virtual-detector-test` (the digits cube, five geometries) passes unchanged through the
  kit: those are exact sums. `tools/virtual-detector-residency` (resident equals tiled, bit for bit) holds for the CPU path
  and for `.metal` images.
- The study cube: before and after the switch, the BF, ADF and DF images and the mean pattern of the Al–Mg–Si cube differ by
  at most the numbers in section 4; CuboKit's `RealDataTests` already holds the kit against py4DSTEM on that cube.
- `inventory`'s DEVIATION notes: mac4DSTEM's Float32 mask note becomes CuboKit's NOTICE line; the `DEVIATION` for the
  real-space circle's `+ 0.5` (register D025) stays in the app with `realSpaceRegion`.
