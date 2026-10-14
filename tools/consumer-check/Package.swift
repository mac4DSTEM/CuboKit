// swift-tools-version: 6.2
// CuboKit consumer check: a SEPARATE package that depends on CuboKit by path and uses, from outside the package, the
// public API an app reads. `package` access does not cross a package boundary, so this compiles only while each symbol
// it names is `public`.
import Foundation
import PackageDescription

// A path dependency's identity is its directory's name (lower-cased): "cubokit" in the repository, but not in a scratch
// copy; `.product` needs that name, so it is taken from the directory two levels above this manifest.
let kit = URL(fileURLWithPath: Context.packageDirectory).deletingLastPathComponent().deletingLastPathComponent()
    .lastPathComponent

let package = Package(
    name: "ConsumerCheck",
    platforms: [.macOS("27.0")],
    dependencies: [.package(path: "../..")],
    targets: [
        .executableTarget(name: "consumer-check", dependencies: [
            .product(name: "CuboCore", package: kit),
            .product(name: "CuboVirtual", package: kit),
            .product(name: "CuboStrain", package: kit),
            .product(name: "CuboPtycho", package: kit),
        ]),
    ],
    swiftLanguageModes: [.v6]
)
