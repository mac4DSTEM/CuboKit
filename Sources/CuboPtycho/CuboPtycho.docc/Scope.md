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
Probe updating can be disabled. Positions stay fixed. Two probe constraints are options,
off by default: centre-of-mass centering (`fixProbeCenterOfMass`) and the real-space
top-hat amplitude constraint (`constrainProbeAmplitude` with relative radius and width),
applied in py4DSTEM's order after each probe update. No probe support, probe aperture
constraint, smoothing, position refinement, batching or preprocessing is implicit.
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

Mixed-state and multislice reconstruction, further constraints, preprocessing,
position refinement and acceleration remain future work. Parallax is outside this API.
