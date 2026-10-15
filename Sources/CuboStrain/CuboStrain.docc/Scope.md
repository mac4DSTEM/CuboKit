# Scope

``CuboStrain`` is a placeholder. It holds no processing yet, and ``CuboStrain/scope`` returns this statement as text.
The module is not registered yet, so it has no numbers to validate.

The planned scope is strain maps from Bragg vectors, following py4DSTEM's `StrainMap`:

- choosing and fitting basis vectors, and `get_strain` with its coordinate rotation;
- a reference lattice taken from a region of interest or from the whole scan;
- checks against py4DSTEM at its pinned commit, and against planted lattices with known strain.

The module consumes Bragg vectors. Where disk detection lives is decided when the module is registered.
