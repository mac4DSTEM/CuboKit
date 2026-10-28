# ``CuboCore``

The cube model every other module reads: a stack of diffraction patterns, one per scan position, held in memory or
memory-mapped from a NumPy `.npy` file, read as Float32 and tiled along the scan.

``Calibration`` carries py4DSTEM's calibration of a cube (pixel sizes, units, origin, ellipse, QR rotation and flip, probe);
``CubeEditing`` is py4DSTEM's reshaping of a `DataCube` (scan binning and thinning, padding, axis swaps, detector resampling including the Fourier resample, the masked-pixel median filter) and
returns an ``EditedCube`` with the calibration that goes with it. ``EllipseFit`` is py4DSTEM's
`process/calibration/ellipse.py`: the algebraic 1D fit, the amorphous-ring fit and the degenerate-ellipse constraint.

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

### Crop and bin views

- ``ViewedCube``
- ``CubeViewSpecification``
- ``CubeViewGeometry``
- ``AxisCrop``

### The file format

- ``NPYFile``
- ``NPYHeader``

### Calibration

- ``Calibration``
- ``CalibrationError``
- ``CalibrationMeasurement``
- ``DetectorUnits``
- ``EllipseCalibration``
- ``ScanField``
- ``CoordinateGrid``

### Editing a cube

- ``CubeEditing``
- ``EditedCube``
- ``FourierResampleSpecification``
- ``FourierResampleOptions``
- ``FrameTransform``

### Ellipse fits

- ``EllipseFit``
- ``EllipseFitPlane``
- ``EllipseParameters``
- ``EllipseFitResult``
- ``EllipseFitError``

### Origin fits

- ``OriginFit``
- ``OriginFitModel``
- ``OriginFitError``
- ``TrimmedOriginFit``

### Errors

- ``CuboError``

### Preprocessing

- ``HotPixelFilter``
- ``HotPixelFilterError``
