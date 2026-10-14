import XCTest
import CuboStrain

final class CuboStrainPlaceholderTests: XCTestCase {
    func testTheModuleStatesItsScope() { XCTAssertTrue(CuboStrain.scope.contains("not registered")) }
}
