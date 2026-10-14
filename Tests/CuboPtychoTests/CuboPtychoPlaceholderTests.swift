import Testing
import CuboPtycho

@Suite struct CuboPtychoPlaceholderTests {
    @Test func testTheModuleStatesItsScope() { #expect(CuboPtycho.scope.contains("not registered")) }
}
