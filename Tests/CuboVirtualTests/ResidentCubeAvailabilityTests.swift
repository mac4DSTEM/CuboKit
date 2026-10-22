// ResidentCube availability (suite decision 2026-10-22): a resident cube may take at most half of the physical memory, and the
// refusal says why. Closed forms with the memory injected; the Metal-dependent cases skip where there is no device.
import CuboCore
import CuboVirtual
import Foundation
import Testing

@Suite("Resident cube availability") struct ResidentCubeAvailabilityTests {
    /// A source that claims a shape and has no data: the refusal happens before any pattern is read.
    actor EmptySource: CubeSource {
        nonisolated let shape: CubeShape
        init(_ shape: CubeShape) { self.shape = shape }
        func readPattern(row: Int, column: Int) async throws -> DiffractionPattern { fatalError("never read") }
        func readScanTile(rows: Range<Int>) async throws -> ScanTile { fatalError("never read") }
    }

    private func shape(scan: Int, detector: Int) throws -> CubeShape {
        try CubeShape(scanRows: scan, scanColumns: scan, detectorRows: detector, detectorColumns: detector)
    }

    @Test func aCubeAboveHalfTheMemoryIsRefusedWithTheNumbers() throws {
        let s = try shape(scan: 100, detector: 64)                 // 40 960 000 values, 163 840 000 bytes
        let result = ResidentCube.availability(of: s, physicalMemory: 300_000_000)   // cap 150 000 000 < 167 772 160
        #expect(result == .unavailable(.exceedsMemoryCap(bytes: 163_840_000, cap: 150_000_000, physicalMemory: 300_000_000)))
        guard case .unavailable(let reason) = result else { return }
        #expect(reason.message.contains("half of this Mac's memory") && reason.message.contains("0.2 GB"))
    }

    @Test func theCapIsInclusiveAtExactlyHalf() throws {
        let s = try shape(scan: 10, detector: 8)                   // 6400 values, 25 600 bytes
        let atCap = ResidentCube.availability(of: s, physicalMemory: 51_200)
        let overCap = ResidentCube.availability(of: s, physicalMemory: 51_199)
        #expect(atCap != .unavailable(.exceedsMemoryCap(bytes: 25_600, cap: 25_600, physicalMemory: 51_200)))
        if case .unavailable(.exceedsMemoryCap(let bytes, let cap, _)) = overCap { #expect(bytes == 25_600 && cap == 25_599) } else {
            Issue.record("one byte over the cap must be refused: \(overCap)")
        }
    }

    @Test func aSmallCubeOnThisMachineIsAvailableWhereMetalIs() throws {
        let s = try shape(scan: 8, detector: 16)
        switch ResidentCube.availability(of: s) {
        case .available: #expect(ResidentCube.fits(s))
        case .unavailable(.noMetalDevice): #expect(!ResidentCube.fits(s))
        case .unavailable(let other): Issue.record("a 8 x 8 x 16 x 16 cube must fit this machine: \(other)")
        }
    }

    @Test func constructionThrowsTheReasonNotAGenericMetalError() async throws {
        // Larger than half of this machine's memory, by shape alone (no data is allocated: the refusal precedes the read).
        let physical = ProcessInfo.processInfo.physicalMemory
        let values = Int(physical) / 4 / 2 + 1_000_000
        let side = Int(Double(values).squareRoot()) + 1
        let huge = try CubeShape(scanRows: side, scanColumns: side, detectorRows: 1, detectorColumns: 1)
        if ResidentCube.availability(of: huge) == .unavailable(.noMetalDevice) { return }
        do {
            _ = try await ResidentCube(loading: EmptySource(huge))
            Issue.record("a cube above half of the physical memory must be refused")
        } catch let error as ResidentCubeError {
            guard case .unavailable(.exceedsMemoryCap(let bytes, let cap, let phys)) = error else { Issue.record("wrong reason: \(error)"); return }
            #expect(bytes > cap && phys == Int(physical))
            #expect(error.errorDescription?.contains("half of this Mac's memory") == true)
        } catch CuboVirtualError.noMetalDevice {
            // no device here: nothing to check
        }
    }
}
