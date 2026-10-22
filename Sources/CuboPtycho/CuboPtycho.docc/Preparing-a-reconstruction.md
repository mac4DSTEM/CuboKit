# Preparing a reconstruction

Ptychography fits an object and a probe to the measured patterns. Before an engine runs, three inputs must exist: the
amplitudes of the patterns, the scan positions in the object's pixels, and a starting probe. ``PtychographyPreparation``
makes the first two from a cube, and ``PtychographyProbe`` the third.

Sampling is in ångström. ``PtychographyPreparation/objectSampling(wavelength:angularSampling:size:)`` gives the object's
pixel size from the electron wavelength, the angular sampling of the detector in milliradians per pixel, and the pattern
size. ``PtychographyProbe/build(rows:columns:rowSampling:columnSampling:wavelength:cutoff:rolloff:defocus:astigmatism:astigmatismAngle:sphericalAberration:higherOrder:)``
builds a unit-norm probe with an aperture of `cutoff` radians and its aberrations. Both are in Å, and the probe is
corner-centred. It is a ``PtychographyArray`` of split real and imaginary parts.

```swift
import CuboPtycho

let wavelength = 0.0197                                                 // 300 keV, in Å
let sampling = PtychographyPreparation.objectSampling(wavelength: wavelength, angularSampling: 0.5, size: 128)
let probe = try PtychographyProbe.build(rows: 128, columns: 128, rowSampling: sampling, columnSampling: sampling,
                                        wavelength: wavelength, cutoff: 0.02, defocus: 100)
print(probe.rows, probe.columns)
```

``PtychographyPreparation/prepare(source:origins:geometry:objectSamplingRow:objectSamplingColumn:wavelength:semiangle:options:)`` and its options (``PtychographyPreparationOptions``) do the
whole preparation of a cube in one call: amplitudes, positions, object shape and the starting probe. They return a
``PreparedPtychography``. The three engines then take the same inputs: ``SingleslicePtychography``,
``MixedstatePtychography`` and ``MultislicePtychography``.

What the module does and does not provide is listed in <doc:Scope>.
