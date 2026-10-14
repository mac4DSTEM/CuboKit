import XCTest
import CuboPtycho

final class CuboPtychoPlaceholderTests: XCTestCase {
    func testTheModuleStatesItsScope() { XCTAssertTrue(CuboPtycho.scope.contains("not registered")) }
}
