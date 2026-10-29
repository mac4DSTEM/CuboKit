import Testing
import CuboPtycho

@Suite struct CuboPtychoScopeTests {
    @Test func testTheModuleStatesItsScope() { #expect(CuboPtycho.scope.contains("single-slice")) }
}
