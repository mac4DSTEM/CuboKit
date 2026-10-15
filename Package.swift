// swift-tools-version: 6.2
// CuboKit — a 4D-STEM library for the Mac: the cube model with an eager and a lazy, memory-mapped path (CuboCore,
// slice 1); the mean and maximum diffraction pattern and virtual images from py4DSTEM's detector geometries, on the CPU
// with Accelerate and on the GPU with Metal behind one API (CuboVirtual, slice 1); centre of mass and DPC (slice 2);
// strain (CuboStrain) and ptychography (CuboPtycho), placeholders with their scope. docs/registration-slice-1.md is
// the first registration. The cube model and reducers were extracted from a GPL-3.0 app (see NOTICE), and the results are
// checked against py4DSTEM f050d207 (GPL-3.0) as the oracle.
//
// Swift 6 language mode, nonisolated by default (no `.defaultIsolation`), the three upcoming features
// enabled below, no unsafe flags (a versioned dependency cannot carry them): a consumer builds the kit optimised
// itself, and tools/test.sh runs this package's tests with `-c release`. One product per module. What one module needs
// from another is `package`, never `public`; what an app reads is `public` and is guarded by tools/consumer-check.
import PackageDescription

let settings: [SwiftSetting] = [
    .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
    .enableUpcomingFeature("InferIsolatedConformances"),
    .enableUpcomingFeature("MemberImportVisibility"),
]

let package = Package(
    name: "CuboKit",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "CuboCore", targets: ["CuboCore"]),
        .library(name: "CuboVirtual", targets: ["CuboVirtual"]),
        .library(name: "CuboPhase", targets: ["CuboPhase"]),
        .library(name: "CuboStrain", targets: ["CuboStrain"]),
        .library(name: "CuboPtycho", targets: ["CuboPtycho"]),
        // A first look at a cube: `swift run cubo-first-look <cube.npy> [output-folder]` (Examples/CuboFirstLook).
        .executable(name: "cubo-first-look", targets: ["CuboFirstLook"]),
    ],
    targets: [
        // Slice 1 (docs/registration-slice-1.md 1.1): the cube's shape, sources, the eager MemoryCube, the lazy
        // memory-mapped MappedCube, the .npy container. Foundation and Accelerate.
        .target(name: "CuboCore", swiftSettings: settings),
        // Slice 1 (1.2): detector geometries and masks, the mean and max pattern, virtual images; CPU (Accelerate) and
        // Metal (kernels compiled at run time from the source this module carries) behind one API.
        .target(name: "CuboVirtual", dependencies: ["CuboCore"], swiftSettings: settings),
        // Slice 2 (docs/registration-slice-2.md): the centre-of-mass reference and plane fit, the R-Q rotation by curl or
        // divergence, py4DSTEM's iterative DPC (Double, Accelerate's FFT with Bluestein for other lengths).
        .target(name: "CuboPhase", dependencies: ["CuboCore"], swiftSettings: settings),
        // Later: strain from Bragg vectors (py4DSTEM StrainMap). Empty until registered.
        .target(name: "CuboStrain", dependencies: ["CuboCore"], swiftSettings: settings),
        // Later: ptychography (py4DSTEM SingleslicePtychography and its family). Empty until registered.
        .target(name: "CuboPtycho", dependencies: ["CuboCore"], swiftSettings: settings),
        // The "first ten minutes" example: CuboCore and CuboVirtual only, ImageIO for the PNGs.
        .executableTarget(name: "CuboFirstLook", dependencies: ["CuboCore", "CuboVirtual"],
                          path: "Examples/CuboFirstLook", swiftSettings: settings),
        .testTarget(name: "CuboCoreTests", dependencies: ["CuboCore"], swiftSettings: settings),
        .testTarget(name: "CuboVirtualTests", dependencies: ["CuboCore", "CuboVirtual"], exclude: ["Fixtures"],
                    swiftSettings: settings),
        .testTarget(name: "CuboPhaseTests", dependencies: ["CuboCore", "CuboPhase", "CuboVirtual"],
                    swiftSettings: settings),
        .testTarget(name: "CuboStrainTests", dependencies: ["CuboStrain"], swiftSettings: settings),
        .testTarget(name: "CuboPtychoTests", dependencies: ["CuboPtycho"], swiftSettings: settings),
    ],
    swiftLanguageModes: [.v6]
)
