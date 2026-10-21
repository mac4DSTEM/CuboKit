//
//  FixtureDirs.swift
//  Role: Where each test target's `Fixtures/` directory lives, for the helpers in this support target. The fixtures stay beside
//        the target that owns them (read by #filePath, not as resources); nothing is copied.
//

import Foundation

package enum FixtureDirs {
    /// `Tests/` of the package.
    package static let tests: URL = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()

    /// `Tests/<target>/Fixtures`.
    package static func fixtures(of target: String) -> URL {
        tests.appendingPathComponent(target).appendingPathComponent("Fixtures")
    }
}
