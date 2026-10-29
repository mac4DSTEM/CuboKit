import Foundation

/// A fixture beside this test target, found through `#filePath` like the other targets' fixtures (not a bundle resource).
func parallaxFixture(_ name: String, _ ext: String = "json") -> URL? {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Fixtures").appendingPathComponent("\(name).\(ext)")
    return FileManager.default.fileExists(atPath: url.path) ? url : nil
}
