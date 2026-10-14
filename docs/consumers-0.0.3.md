# What a consumer changes to import CuboKit 0.0.3

Written 2026-10-15 for mac4DSTEM by the CuboKit session, read at mac4DSTEM's pushed `7f192da9`. mac4DSTEM's own session
makes the switch. 0.0.3 is 0.0.2 plus slice 3; nothing 0.0.1 or 0.0.2 returns changes.

## What slice 3 offers, and what it replaces

| mac4DSTEM today | CuboKit 0.0.3 |
|---|---|
| `VirtualDetector.tiledMeasuredOrigins` and `Shaders/OriginMeasure.metal` (a CoM window, Float32, no Gaussian argmax) | `OriginMeasurement.measure(of:radius:radiusScale:)`: py4DSTEM's `get_origin` (Gaussian-filtered argmax, CoM within r · rscale), Double, CPU |
| `OriginCalibration`'s smooth fit of the origins | `OriginFit.fit(_:rows:columns:model:robust:)`: `fit_origin`'s constant, plane, parabola, Bézier, robust option |
| the dragged aperture at the fitted mean origin; per-position origins fed to the CoM kernel | `VirtualImaging.image(of:geometry:centeredOn:pixelSize:shift:)`: py4DSTEM's centred, calibrated and shifted (rounded or bilinear) virtual images |
| — | `VirtualDiffraction.meanAndMax(of:shift:)`: centre-shifted mean and maximum (rounded or Fourier) |

## Behaviour to know (tier 1)

1. **Origins follow py4DSTEM's `get_origin`**, which mac4DSTEM's origin measure does not: a Gaussian-filtered argmax first,
   then the centre of mass in a disk of r · rscale around it. Expect origin maps to move where the beam is not the
   brightest pixel or where the window differs; compare on the five cubes before switching (CuboKit's numbers are in its
   status).
2. **Centred and calibrated together are refused**: py4DSTEM adds the pixel origin and then divides by the pixel size.
3. **The shifted maximum is the true maximum** (py4DSTEM seeds it at 0, so negative pixels read 0 there).
4. **The shifted paths run on the CPU** (`.metal` with a shift is refused).
5. On the Au reference cube py4DSTEM's own float32 filter picks a neighbouring argmax at 2 of 63 550 positions (0.62 px);
   CuboKit follows the float64 definition.
