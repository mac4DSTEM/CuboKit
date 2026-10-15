# ``CuboCore``

The cube model every other module reads: a stack of diffraction patterns, one per scan position, held in memory or
memory-mapped from a NumPy `.npy` file, read as Float32 and tiled along the scan.

## Topics

### Articles

- <doc:Opening-a-cube>

### The cube

- ``CubeShape``
- ``CubeSource``
- ``MemoryCube``
- ``MappedCube``
- ``ElementType``

### Reads and tiling

- ``DiffractionPattern``
- ``ScanTile``
- ``ScanTiling``

### The file format

- ``NPYFile``
- ``NPYHeader``

### Origin fits

- ``OriginFit``
- ``OriginFitModel``
- ``OriginFitError``

### Errors

- ``CuboError``
