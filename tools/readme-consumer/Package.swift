// swift-tools-version: 6.2
// CuboKit README consumer: a SEPARATE package that depends on CuboKit by path and compiles the Swift code blocks of
// README.md verbatim, from outside the package, so a stranger's copy of the README is known to build and run.
import Foundation
import PackageDescription

// A path dependency's identity is its directory's name (lower-cased): "cubokit" in the repository, but not in a scratch
// copy; `.product` needs that name, so it is taken from the directory two levels above this manifest.
let kit = URL(fileURLWithPath: Context.packageDirectory).deletingLastPathComponent().deletingLastPathComponent()
    .lastPathComponent

let package = Package(
    name: "ReadmeConsumer",
    platforms: [.macOS(.v15)],
    dependencies: [.package(path: "../..")],
    targets: [
        .executableTarget(name: "readme-consumer", dependencies: [
            .product(name: "CuboCore", package: kit),
            .product(name: "CuboVirtual", package: kit),
        ]),
    ],
    swiftLanguageModes: [.v6]
)
