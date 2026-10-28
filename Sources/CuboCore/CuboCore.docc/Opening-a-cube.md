# Opening a cube

A cube holds one diffraction pattern per scan position, in C order: `[scanRow][scanColumn][detectorRow][detectorColumn]`.
This is py4DSTEM's `(R_Nx, R_Ny, Q_Nx, Q_Ny)` and the shape of NumPy's `data`. ``CubeShape`` holds the four extents, scan
rows and columns first, then detector rows and columns. The library says row and column, never x and y: py4DSTEM's first
axis of each pair is the row here.

Export a cube from Python with `np.save("cube.npy", datacube.data)`, then open it with ``MappedCube``. The file is mapped
read-only, and nothing is read until a pass asks for it. A read converts only the elements it needs to Float32. The
file must be a rank-4, C-ordered, little-endian array of one of the ten NumPy element types: float32, float64, the eight
integer types. Big-endian data and Fortran order are refused by name.

```swift
import CuboCore
import Foundation

let lazy = try MappedCube(npy: URL(fileURLWithPath: "cube.npy"))   // maps the file; reads nothing yet
print(lazy.shape)                                                  // the four extents, as CubeShape describes them

// The same values held in memory, loaded tile by tile.
let eager = try await MemoryCube(loading: lazy, tiling: .automatic)
let pattern = try await eager.readPattern(row: 0, column: 0)       // detectorRows × detectorColumns Float32 values
```

Every read returns Float32, whatever the stored element type. Integers convert exactly up to 2^24; a float64 cube is
rounded to the nearest Float32 when it is read.

Whole-cube passes walk the scan in tiles. ``ScanTiling/automatic`` takes the largest number of scan rows whose Float32
tile fits in 64 MiB, and at least one row. It depends on the shape alone, not on the machine. ``ScanTiling/rows(_:)``
fixes the count. On the CPU path a result does not depend on the tiling; the Metal path agrees with it within relative
1e-5.

To build a cube in memory from an array, use ``MemoryCube/init(shape:values:)`` with the values in storage order.
