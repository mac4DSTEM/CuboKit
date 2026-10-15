# Scope

``CuboPtycho`` is a placeholder. It holds no processing yet, and ``CuboPtycho/scope`` returns this statement as text.
The module is not registered yet, so it has no numbers to validate.

The planned scope is iterative ptychography, following py4DSTEM's phase retrieval:

- single-slice ptychography first (py4DSTEM's `SingleslicePtychography`), then mixed-state and multislice;
- tilt-corrected bright field (parallax), with its aberration fit;
- Metal first, with a CPU reference and a registered tolerance between float32 and float64.

Its tests will need truth from simulated probes and multislice data, not only from agreement with py4DSTEM.
