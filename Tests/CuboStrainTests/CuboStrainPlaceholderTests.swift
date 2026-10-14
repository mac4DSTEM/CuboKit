import Testing
import CuboStrain

@Suite struct CuboStrainPlaceholderTests {
    @Test func testTheModuleStatesItsScope() { #expect(CuboStrain.scope.contains("not registered")) }
}
