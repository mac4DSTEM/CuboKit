import XCTest
import CuboBragg

/// The app preset changes behaviour switches only. Its thresholds are py4DSTEM's (`find_Bragg_disks`, diskdetection.py:22-38):
/// until 0.11.0 it lowered `minimumRelativeIntensity` to 0.0015, a value chosen on one dataset.
final class AppPresetTests: XCTestCase {
    func testTheAppPresetKeepsPy4DSTEMsThresholds() {
        let preset = BraggDiskOptions.mac4DSTEM, reference = BraggDiskOptions()
        XCTAssertEqual(reference.minimumRelativeIntensity, 0.005)
        XCTAssertEqual(preset.minimumRelativeIntensity, reference.minimumRelativeIntensity)
        XCTAssertEqual(preset.minimumAbsoluteIntensity, reference.minimumAbsoluteIntensity)
        XCTAssertEqual(preset.relativeToPeak, reference.relativeToPeak)
        XCTAssertEqual(preset.minimumPeakSpacing, reference.minimumPeakSpacing)
        XCTAssertEqual(preset.edgeBoundary, reference.edgeBoundary)
        XCTAssertEqual(preset.maximumPeaks, reference.maximumPeaks)
        XCTAssertEqual(preset.relativeReferenceMinimumRadius, 0)
        // What the preset does change.
        XCTAssertEqual(preset.nonFinitePixels, .neighbourMedian)
        XCTAssertEqual(preset.multicorrOrder, .correlationIntensity)
    }
}
