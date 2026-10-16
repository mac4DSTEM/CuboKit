# Scope

``LatticeStrain`` provides per-pattern lattice fitting and strain tensors, using
caller-supplied Bragg vectors and histogram maxima. The caller supplies scan shape,
valid-position masks, reference ROI selection, histogram construction and peak detection.

Primary algorithms follow py4DSTEM f050d207: intensity-squared fitting, plain
component medians and the reference strain/rotation conventions. SVD provides
minimum-norm solutions, including rank-deficient indexing and fitting.

Explicit compatibility methods preserve Float32 cluster consensus, intensity-once
normal equations and MAD-reference selection. Their independent pins execute the
original numerical source. They do not include the local robust-refit pipeline.
Compatibility inputs must convert to finite Float32 and have magnitude below 1e10;
this bound prevents unsafe spatial-hash conversion. Primary inputs use finite Double.
