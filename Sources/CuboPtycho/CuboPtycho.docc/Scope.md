# Scope

The module builds corner-centered, unit-norm probes and performs CPU single-slice,
complex-object, full-batch gradient descent from preprocessed diffraction amplitudes.
All public inputs and outputs are Sendable values. See ``PtychographyProbe`` and
``SingleslicePtychography``. Probe sampling and wavelength use Å; angles use radians.

Supported probe aberrations are defocus (C10 = −defocus), polar two-fold astigmatism
and C30. Coherence envelopes and other aberrations are not implemented.

Reconstruction uses periodic object patches, ties-to-even rounding, Fourier shifts
of the probe for fractional positions, the reference's normalized gradient updates,
and an object amplitude cap of one (or unit amplitude for pure-phase objects).
Probe updating can be disabled. Positions stay fixed unless `positionCorrection` is given (single-slice only, py4DSTEM's `fix_positions=False`). Two probe constraints are options,
off by default: centre-of-mass centering (`fixProbeCenterOfMass`) and the real-space
top-hat amplitude constraint (`constrainProbeAmplitude` with relative radius and width),
applied in py4DSTEM's order after each probe update; a fixed Fourier aperture
(`fixProbeAperture`) and a real-space support mask (`probeSupportMask`) are options too. No smoothing, position refinement or batching is implicit. `errorNormalization` selects py4DSTEM's per-iteration error (divided by the preprocessing mean intensity) or the energy of the supplied amplitudes.
These choices correspond to explicitly disabled options in py4DSTEM.
The probe adjoint intentionally follows the reference's sum without undoing each
fractional shift. Input probes are already normalized to the measured intensities
by the caller when needed; the engine does not change their initial normalization.

The regression fixture pins the actual py4DSTEM 0.14.19 (`f050d207`) `ComplexProbe`
and `SingleslicePtychography.reconstruct`, with a fixed synthetic object and seed.
The engine uses Double arithmetic; the Python state is complex64. Agreement on these
fixtures is within `2e-6` max absolute error with a unit scale floor, not bit-exact.
Exact Fourier zeros have undefined phase; roundoff may choose different phases near
those zeros, particularly for a uniform starting object and a hard aperture.
This fixture uses a nonuniform initial object. Agreement is not microscope validation.

`MixedstatePtychography` (several incoherent probe states, orthogonalisation, NumPy-compatible seeded
starting states) and `MultislicePtychography` (a stack of slices with Fresnel propagators) follow the same
conventions and tolerance. `PtychographyPreparation` builds the amplitudes, scan positions, object canvas and
starting probe from a `CubeSource`, with py4DSTEM's conventions by default and the app's as named options.
The Fourier-amplitude and aberration-fitting constraints, position refinement of the mixed-state and multislice engines, the projection schemes,
batching and acceleration remain future work (`docs/ptycho.md`). Parallax is outside this API.
