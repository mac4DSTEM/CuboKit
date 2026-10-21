# ``CuboStrain``

CPU reciprocal lattice fitting, reference selection and rotated infinitesimal strain.

Use ``LatticeStrain`` for the pinned reference conventions, or its explicitly named
compatibility methods for intensity-once fits, MAD references and cluster consensus (``LatticeStrain/chooseBasisConsensusResult(_:origin:minimumRadius:isCancelled:)`` and
``LatticeStrain/referenceMADResult(_:)`` also return the scoring diagnostics and the retained candidates).
Detector row is the reference first coordinate; detector column is the second.

For a whole scan, ``StrainMapping`` follows py4DSTEM's `StrainMap`: build the Bragg vector map from centred peaks,
choose the basis from its maxima, fit every scan position, take the reference from a scan region and return flat
row-major exx, eyy, exy and theta maps (NaN where a position could not be fitted), with an optional coordinate rotation.
