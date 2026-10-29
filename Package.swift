// swift-tools-version: 6.2
// SPDX-License-Identifier: GPL-3.0-only
// CuboKit — a 4D-STEM library for the Mac: the cube model with an eager and a lazy, memory-mapped path (CuboCore);
// the mean and maximum diffraction pattern and virtual images from py4DSTEM's detector geometries, on the CPU
// with Accelerate and on the GPU with Metal behind one API (CuboVirtual); centre of mass and DPC (CuboPhase);
// parallax alignment, strain, Bragg disk detection, embedding and single-slice ptychography (CuboParallax,
// CuboStrain, CuboBragg, CuboEmbedding, CuboPtycho). The cube model and reducers were extracted from a GPL-3.0 app (see NOTICE), and the results are
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
        .library(name: "CuboParallax", targets: ["CuboParallax"]),
        .library(name: "CuboStrain", targets: ["CuboStrain"]),
        .library(name: "CuboBragg", targets: ["CuboBragg"]),
        .library(name: "CuboPtycho", targets: ["CuboPtycho"]),
        .library(name: "CuboEmbedding", targets: ["CuboEmbedding"]),
        // A first look at a cube: `swift run cubo-first-look <cube.npy> [output-folder]` (Examples/CuboFirstLook).
        .executable(name: "cubo-first-look", targets: ["CuboFirstLook"]),
    ],
    targets: [
        // the cube's shape, sources, the eager MemoryCube, the lazy
        // memory-mapped MappedCube, the .npy container. Foundation and Accelerate.
        .target(name: "CuboCore", swiftSettings: settings),
        // detector geometries and masks, the mean and max pattern, virtual images; CPU (Accelerate) and
        // Metal (kernels compiled at run time from the source this module carries) behind one API.
        .target(name: "CuboVirtual", dependencies: ["CuboCore"], swiftSettings: settings),
        // the centre-of-mass reference and plane fit, the R-Q rotation by curl or
        // divergence, py4DSTEM's iterative DPC (Double, Accelerate's FFT with Bluestein for other lengths).
        .target(name: "CuboPhase", dependencies: ["CuboCore"], swiftSettings: settings),
        .target(name: "CuboParallax", dependencies: ["CuboCore"], swiftSettings: settings),
        .target(name: "CStrainLAPACK", cSettings: [.define("ACCELERATE_NEW_LAPACK")], linkerSettings: [.linkedFramework("Accelerate")]),
        .target(name: "CuboStrain", dependencies: ["CuboCore", "CuboBragg", "CStrainLAPACK"], swiftSettings: settings),
        .target(name: "CEmbeddingLAPACK", cSettings: [.define("ACCELERATE_NEW_LAPACK")], linkerSettings: [.linkedFramework("Accelerate")]),
        // PCA and k-means: scikit-learn's behaviour as the primary API, mac4DSTEM's diffraction-embedding numerics as
        // named options (docs/embedding.md).
        .target(name: "CuboEmbedding", dependencies: ["CEmbeddingLAPACK"], swiftSettings: settings),

        // Bragg disk detection's numeric core: py4DSTEM's find_Bragg_disks (hybrid cross-correlation with a probe kernel, maxima,
        // the filter cascade, parabolic and multicorr subpixel refinement), Double, CPU, with the mac4DSTEM variants as named options.
        .target(name: "CuboBragg", dependencies: ["CuboCore", "CuboVirtual", "CuboParallax"], swiftSettings: settings),
        // CPU single-slice ptychography after py4DSTEM's SingleslicePtychography (probe construction, gradient descent).
        .target(name: "CuboPtycho", dependencies: ["CuboCore"], swiftSettings: settings),
        // The "first ten minutes" example: CuboCore and CuboVirtual only, ImageIO for the PNGs.
        .executableTarget(name: "CuboFirstLook", dependencies: ["CuboCore", "CuboVirtual"],
                          path: "Examples/CuboFirstLook", swiftSettings: settings),
        // Helpers, pins and fixture loaders the fast tests and the real-data parity suites share (package access; fixtures stay
        // beside the target that owns them, see FixtureDirs).
        .target(name: "CuboKitTestSupport", dependencies: ["CuboCore", "CuboVirtual", "CuboPhase"],
                path: "Tests/CuboKitTestSupport", swiftSettings: settings),
        // The real-data comparisons against the pinned float64 / py4DSTEM exports in $CUBOKIT_REAL_DATA (skipped by name when absent):
        // `swift test -c release --parallel --skip CuboKitParityTests` is the fast gate, the plain run is the full suite.
        .testTarget(name: "CuboKitParityTests", dependencies: ["CuboCore", "CuboVirtual", "CuboPhase", "CuboKitTestSupport"],
                    swiftSettings: settings),
        .testTarget(name: "CuboCoreTests", dependencies: ["CuboCore"], exclude: ["Fixtures"], swiftSettings: settings),
        .testTarget(name: "CuboVirtualTests", dependencies: ["CuboCore", "CuboVirtual", "CuboKitTestSupport"], exclude: ["Fixtures"],
                    swiftSettings: settings),
        .testTarget(name: "CuboPhaseTests", dependencies: ["CuboCore", "CuboPhase", "CuboVirtual", "CuboKitTestSupport"],
                    exclude: ["Fixtures"], swiftSettings: settings),
        .testTarget(name: "CuboParallaxTests", dependencies: ["CuboCore", "CuboParallax"],
                    resources: [.copy("Fixtures")], swiftSettings: settings),
        .testTarget(name: "CuboStrainTests", dependencies: ["CuboStrain", "CuboBragg"], exclude: ["Fixtures"], swiftSettings: settings),
        .testTarget(name: "CuboEmbeddingTests", dependencies: ["CuboEmbedding"], exclude: ["Fixtures"], swiftSettings: settings),

        .testTarget(name: "CuboBraggTests", dependencies: ["CuboCore", "CuboVirtual", "CuboBragg", "CuboEmbedding"],
                    exclude: ["Fixtures"], swiftSettings: settings),
        .testTarget(name: "CuboPtychoTests", dependencies: ["CuboPtycho"], exclude: ["Fixtures"], swiftSettings: settings),
    ],
    swiftLanguageModes: [.v6]
)
