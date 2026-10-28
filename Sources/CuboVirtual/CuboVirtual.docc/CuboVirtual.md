# ``CuboVirtual``

Virtual detectors and the reductions over a cube: virtual images, the mean, median and maximum diffraction pattern,
the centre of mass, probe size and origins. Each runs on the CPU or with Metal, behind one API.

``ResidentCube`` holds a cube in one Metal buffer for repeated reductions; ask ``ResidentCube/availability(of:physicalMemory:)``
first, since it is capped at half of the physical memory. ``PolarCube`` is py4DSTEM's polar-elliptical transform of every
pattern and ``RadialBackground`` its radial background, subtraction and local average; both run on the CPU in Double.

## Topics

### Articles

- <doc:Virtual-images-and-diffraction-patterns>

### Detectors

- ``DetectorGeometry``
- ``DetectorMask``
- ``BeamstopMask``

### Images and patterns

- ``VirtualImaging``
- ``VirtualDiffraction``
- ``DiffractionStatistic``
- ``ScanSelection``
- ``PatternShift``
- ``MaskShift``

### Centre of mass and probe

- ``CenterOfMass``
- ``CenterOfMassMaps``
- ``ProbeSize``
- ``ProbeEstimate``
- ``VacuumProbe``
- ``ProbeKernel``
- ``SigmoidProfile``
- ``SyntheticProbe``

### Resident cube

- ``ResidentCube``
- ``ResidentCubeError``

### Polar cube and radial background

- ``PolarCube``
- ``PolarSampling``
- ``PolarPattern``
- ``PolarCubeData``
- ``RadialBackground``

### Origins

- ``OriginMeasurement``
- ``OriginEstimator``
- ``OriginMap``
- ``GaussianFilter``
- ``FriedelOrigin``

### Results and errors

- ``ComputePath``
- ``ScanImage``
- ``DetectorImage``
- ``CuboVirtualError``
