import CuboCore
import CuboVirtual
import Foundation
import Testing
@testable import CuboBragg

/// The `.single` (Float32, Accelerate) path against py4DSTEM's pins and against the kit's own `.double` path, the per-stage
/// funnel, and `fillNonFinite`. Fixtures and helpers are BraggTests.swift's.
@Suite struct SinglePrecisionTests {
    let fixture: Fixture
    init() throws { fixture = try Fixture() }

    func single(_ o: BraggDiskOptions) -> BraggDiskOptions { var s = o; s.precision = .single; return s }

    /// Scenes vDSP's DFT supports (64 × 64); S2 (40 × 56) is refused.
    func supported(_ c: [String: Any]) -> Bool {
        let s = fixture.scene(c["scene"] as! String)
        return BraggDetector.supportsSinglePrecision(detectorRows: s.rows, detectorColumns: s.columns)
    }

    /// Every py4DSTEM case on a supported scene: same count and order, positions and intensities within the written tolerance.
    @Test func singleMatchesPy4DSTEM() throws {
        var worstPosition = 0.0, worstIntensity = 0.0, strongPosition = 0.0, strongIntensity = 0.0, checked = 0
        for c in fixture.cases where (c["reference"] as! String) == "py4dstem" && supported(c) && options(c["options"] as! [String: Any]).correlationPower > 0 {
            let id = c["id"] as! String
            let s = fixture.scene(c["scene"] as! String)
            let expected = peaks(fixture.array(c["expected"] as! String))
            let actual = try BraggDisks.detect(s.patterns[c["pattern"] as! Int], kernel: s.kernel,
                                               options: single(options(c["options"] as! [String: Any])))
            #expect(actual.count == expected.count, "\(id): \(actual.count) peaks, py4DSTEM \(expected.count)")
            guard actual.count == expected.count else { continue }
            let g = SingleTolerance.check(actual, expected, id)
            worstPosition = max(worstPosition, g.position); worstIntensity = max(worstIntensity, g.intensity)
            strongPosition = max(strongPosition, g.strongPosition); strongIntensity = max(strongIntensity, g.strongIntensity); checked += 1
        }
        print("CuboBragg .single vs py4DSTEM: \(checked) cases, maximum position gap \(worstPosition) px (peaks above 1e-3 of the brightest: \(strongPosition)), maximum relative intensity gap \(worstIntensity) (those: \(strongIntensity))")
        #expect(checked >= 24)
    }

    /// The same inputs through `.double` and `.single`: peaks agree (py4DSTEM's cases and the app's variants).
    @Test func singleAgreesWithDouble() throws {
        var worstPosition = 0.0, worstIntensity = 0.0, strongPosition = 0.0, strongIntensity = 0.0, checked = 0
        for c in fixture.cases where supported(c) && options(c["options"] as! [String: Any]).correlationPower > 0 {
            let id = c["id"] as! String
            let s = fixture.scene(c["scene"] as! String)
            var o = options(c["options"] as! [String: Any])
            if (c["reference"] as! String) == "app" { o.nonFinitePixels = .neighbourMedian; o.multicorrOrder = .correlationIntensity }
            let d = try BraggDisks.detect(s.patterns[c["pattern"] as! Int], kernel: s.kernel, options: o)
            let f = try BraggDisks.detect(s.patterns[c["pattern"] as! Int], kernel: s.kernel, options: single(o))
            #expect(d.count == f.count, "\(id): double \(d.count) peaks, single \(f.count)")
            guard d.count == f.count else { continue }
            let g = SingleTolerance.check(f, d, id)
            worstPosition = max(worstPosition, g.position); worstIntensity = max(worstIntensity, g.intensity)
            strongPosition = max(strongPosition, g.strongPosition); strongIntensity = max(strongIntensity, g.strongIntensity); checked += 1
        }
        print("CuboBragg .single vs .double: \(checked) cases, maximum position gap \(worstPosition) px (peaks above 1e-3 of the brightest: \(strongPosition)), maximum relative intensity gap \(worstIntensity) (those: \(strongIntensity))")
        #expect(checked >= 30)
    }

    /// The correlation itself: the real map and the Fourier product, as a fraction of their largest magnitude.
    @Test func correlationMapAgreesWithDouble() throws {
        var worstReal = 0.0, worstFourier = 0.0
        for (name, power, sigma) in [("S1", 1.0, 0.0), ("S1", 0.5, 1.5), ("S1", 0.75, 0.0), ("S3", 1.0, 0.0), ("S3", 0.5, 0.0)] {
            let s = fixture.scene(name)
            let detector = try BraggDetector(kernel: s.kernel)
            var o = BraggDiskOptions(); o.correlationPower = power; o.patternSigma = sigma
            for pattern in s.patterns {
                let d = try detector.correlate(pattern, options: o)
                let f = try detector.correlate(pattern, options: single(o))
                let realScale = d.real.map(abs).max()!
                let fourierScale = zip(d.fourierRe, d.fourierIm).map { hypot($0, $1) }.max()!
                for i in d.real.indices {
                    worstReal = max(worstReal, abs(f.real[i] - d.real[i]) / realScale)
                    worstFourier = max(worstFourier, hypot(f.fourierRe[i] - d.fourierRe[i], f.fourierIm[i] - d.fourierIm[i]) / fourierScale)
                }
            }
        }
        print("CuboBragg .single correlation map: maximum gap \(worstReal) of the map's maximum, Fourier product \(worstFourier) of its largest magnitude")
        #expect(worstReal <= SingleTolerance.correlation, "map gap \(worstReal)")
        #expect(worstFourier <= SingleTolerance.fourier, "Fourier gap \(worstFourier)")
    }

    @Test func refusesUnsupportedSizesAndKeepsDouble() async throws {
        let s2 = fixture.scene("S2")   // 40 × 56: vDSP has no DFT of length 56
        #expect(!BraggDetector.supportsSinglePrecision(detectorRows: 40, detectorColumns: 56))
        #expect(BraggDetector.supportsSinglePrecision(detectorRows: 64, detectorColumns: 64))
        #expect(BraggDetector.supportsSinglePrecision(detectorRows: 256, detectorColumns: 256))
        #expect(!BraggDetector.supportsSinglePrecision(detectorRows: 56, detectorColumns: 56))
        #expect(!BraggDetector.supportsSinglePrecision(detectorRows: 127, detectorColumns: 128))
        #expect(!BraggDetector.supportsSinglePrecision(detectorRows: 0, detectorColumns: 64))
        var o = BraggDiskOptions(); o.precision = .single
        #expect(throws: CuboBraggError.self) { try BraggDisks.detect(s2.patterns[0], kernel: s2.kernel, options: o) }
        #expect(throws: CuboBraggError.self) { try BraggDetector(kernel: s2.kernel).detectWithFunnel(s2.patterns[0], options: o) }
        #expect(throws: CuboBraggError.self) { try BraggDetector(kernel: s2.kernel).correlate(s2.patterns[0], options: o) }
        do { _ = try BraggDisks.detect(s2.patterns[0], kernel: s2.kernel, options: o) } catch let e as CuboBraggError {
            guard case .invalidOption(let message) = e else { Issue.record("wrong error \(e)"); return }
            #expect(message.contains("40 × 56"))
        }
        // The whole-cube pass refuses before reading a tile.
        let shape = try CubeShape(scanRows: 1, scanColumns: 1, detectorRows: 40, detectorColumns: 56)
        let cube = try MemoryCube(shape: shape, values: [Float](repeating: 0, count: 40 * 56))
        await #expect(throws: CuboBraggError.self) { _ = try await BraggDisks.detectAll(in: cube, kernel: s2.kernel, options: o) }
        // .double is untouched on that size, and is the default.
        #expect(BraggDiskOptions().precision == .double)
        _ = try BraggDisks.detect(s2.patterns[0], kernel: s2.kernel)
        // Options are validated first, as in .double.
        o.upsampleFactor = 2
        #expect(throws: CuboBraggError.self) { try BraggDisks.detect(fixture.scene("S1").patterns[0], kernel: fixture.scene("S1").kernel, options: o) }
    }

    /// Non-square supported grids (the transposed spectrum must be undone correctly) and a Gaussian wider than the image
    /// (the reflection folds repeatedly): the correlation map and its Fourier product agree with `.double`.
    @Test func nonSquareAndWideFilter() throws {
        let s = fixture.scene("S1")
        var worstMap = 0.0, worstFourier = 0.0
        for (rows, columns, sigma) in [(64, 48, 0.0), (48, 64, 1.5), (16, 64, 0.0), (16, 16, 5.0), (32, 80, 2.0), (32, 32, 0.1)] {   // σ 0.1: radius 0
            // The S1 data tiled onto the grid: any values do, both precisions see the same ones.
            let kernel = DetectorImage(detectorRows: rows, detectorColumns: columns,
                                       values: (0..<(rows * columns)).map { s.kernel.values[$0 % 4096] }, path: .cpu)
            #expect(BraggDetector.supportsSinglePrecision(detectorRows: rows, detectorColumns: columns), "\(rows) × \(columns)")
            let detector = try BraggDetector(kernel: kernel)
            var o = BraggDiskOptions(); o.patternSigma = sigma
            for pattern in s.patterns {
                let values = (0..<(rows * columns)).map { pattern.values[$0 % 4096] }
                let image = DetectorImage(detectorRows: rows, detectorColumns: columns, values: values, path: .cpu)
                let d = try detector.correlate(image, options: o)
                var os = o; os.precision = .single
                let f = try detector.correlate(image, options: os)
                let mapScale = d.real.max()!
                let fourierScale = zip(d.fourierRe, d.fourierIm).map { hypot($0, $1) }.max()!
                for i in d.real.indices {
                    worstMap = max(worstMap, abs(f.real[i] - d.real[i]) / mapScale)
                    worstFourier = max(worstFourier, hypot(f.fourierRe[i] - d.fourierRe[i], f.fourierIm[i] - d.fourierIm[i]) / fourierScale)
                }
            }
        }
        print("CuboBragg .single non-square and wide-filter grids: map gap \(worstMap), Fourier gap \(worstFourier)")
        #expect(worstMap <= SingleTolerance.correlation && worstFourier <= SingleTolerance.fourier)
    }

    /// The Float entry and the DetectorImage entry are the same computation; `.double` widens.
    @Test func floatEntryEqualsImageEntry() throws {
        let s = fixture.scene("S1")
        let detector = try BraggDetector(kernel: s.kernel)
        var o = BraggDiskOptions(); o.minimumPeakSpacing = 8; o.edgeBoundary = 5
        for precision in BraggPrecision.allCases {
            o.precision = precision
            for pattern in s.patterns {
                let floats = pattern.values.map { Float($0) }
                let widened = DetectorImage(detectorRows: 64, detectorColumns: 64, values: floats.map { Double($0) }, path: .cpu)
                let viaFloat = try floats.withUnsafeBufferPointer { try detector.detectWithFunnel($0, options: o) }
                let viaImage = try detector.detectWithFunnel(widened, options: o)
                #expect(viaFloat.peaks == viaImage.peaks && viaFloat.funnel == viaImage.funnel, "\(precision)")
                #expect(!viaFloat.peaks.isEmpty)
                #expect(try floats.withUnsafeBufferPointer { try detector.detect($0, options: o) } == viaFloat.peaks)
            }
        }
        let short = [Float](repeating: 0, count: 10)
        #expect(throws: CuboBraggError.self) { try short.withUnsafeBufferPointer { try detector.detect($0) } }
    }

    /// The whole-cube pass in `.single` is the per-pattern pass, position by position, and reuses a handful of scratches.
    @Test func wholeCubeSingleEqualsPerPattern() async throws {
        let s = fixture.scene("S1")
        let count = 40
        let shape = try CubeShape(scanRows: count, scanColumns: 1, detectorRows: 64, detectorColumns: 64)
        var floats: [Float] = []
        for k in 0..<count { floats += s.patterns[k % s.patterns.count].values.map { Float($0) } }
        let cube = try MemoryCube(shape: shape, values: floats)
        var o = BraggDiskOptions(); o.minimumPeakSpacing = 8; o.edgeBoundary = 5; o.precision = .single
        let all = try await BraggDisks.detectAll(in: cube, kernel: s.kernel, options: o, tiling: .rows(16))
        let detector = try BraggDetector(kernel: s.kernel)
        for k in 0..<count {
            let one = try floats[(k * 4096)..<((k + 1) * 4096)].withUnsafeBufferPointer { try detector.detect($0, options: o) }
            #expect(all[k, 0] == one)
        }
        #expect(all.peaks.map(\.count).reduce(0, +) > 100)
        // Scratches: one per concurrent worker, not one per pattern.
        let state = try detector.singleState()
        let pattern = Array(floats[0..<4096])
        pattern.withUnsafeBufferPointer { p in
            DispatchQueue.concurrentPerform(iterations: 200) { _ in _ = try? state.detect(p.baseAddress!, options: o) }
        }
        #expect(state.scratchCount >= 1 && state.scratchCount <= ProcessInfo.processInfo.activeProcessorCount,
                "\(state.scratchCount) scratches for \(ProcessInfo.processInfo.activeProcessorCount) cores")
    }

    // MARK: Funnel

    /// A hand-built correlation: five maxima of known intensity, one inside the edge boundary.
    func handMap() -> [Double] {
        var m = [Double](repeating: 0, count: 30 * 30)
        func put(_ r: Int, _ c: Int, _ v: Double) { m[r * 30 + c] = v }
        put(8, 8, 10)      // A
        put(8, 20, 5)      // B
        put(9, 10, 2)      // E: 2.2 px from A
        put(20, 8, 1)      // C: 0.1 of A
        put(20, 20, 0.03)  // D
        put(2, 2, 7)       // F: inside the edge boundary of 3
        return m
    }

    @Test func funnelHandBuilt() throws {
        var o = BraggDiskOptions()
        o.correlationSigma = 0; o.subpixel = .pixel; o.edgeBoundary = 3
        o.minimumAbsoluteIntensity = 0; o.minimumRelativeIntensity = 0; o.minimumPeakSpacing = 0; o.maximumPeaks = 70
        func run(_ edit: (inout BraggDiskOptions) -> Void) throws -> (peaks: [BraggPeak], funnel: BraggFunnel) {
            var p = o; edit(&p)
            return try BraggMaxima.findWithFunnel(in: handMap(), rows: 30, columns: 30, options: p)
        }
        // No cut: five maxima (F is inside the edge), all accepted.
        var r = try run { _ in }
        #expect(r.funnel == BraggFunnel(localMaximumCount: 5, afterAbsoluteCutCount: 5, afterRelativeCutCount: 5, afterSpacingCount: 5,
                                        acceptedCount: 5, wasCountLimited: false, relativeReferenceIntensity: nil,
                                        relativeReferenceWasAvailable: true, correlationMaximum: 10))
        #expect(r.peaks.map(\.intensity) == [10, 5, 2, 1, 0.03])
        // Every cut in turn: absolute removes D, relative (0.15 of A) removes C, spacing 5 removes E, the cap keeps one.
        r = try run { $0.minimumAbsoluteIntensity = 0.5 }
        #expect(r.funnel == BraggFunnel(localMaximumCount: 5, afterAbsoluteCutCount: 4, afterRelativeCutCount: 4, afterSpacingCount: 4,
                                        acceptedCount: 4, wasCountLimited: false, relativeReferenceIntensity: nil,
                                        relativeReferenceWasAvailable: true, correlationMaximum: 10))
        r = try run { $0.minimumAbsoluteIntensity = 0.5; $0.minimumRelativeIntensity = 0.15 }
        #expect(r.funnel == BraggFunnel(localMaximumCount: 5, afterAbsoluteCutCount: 4, afterRelativeCutCount: 3, afterSpacingCount: 3,
                                        acceptedCount: 3, wasCountLimited: false, relativeReferenceIntensity: 10,
                                        relativeReferenceWasAvailable: true, correlationMaximum: 10))
        r = try run { $0.minimumAbsoluteIntensity = 0.5; $0.minimumRelativeIntensity = 0.15; $0.minimumPeakSpacing = 5 }
        #expect(r.funnel == BraggFunnel(localMaximumCount: 5, afterAbsoluteCutCount: 4, afterRelativeCutCount: 3, afterSpacingCount: 2,
                                        acceptedCount: 2, wasCountLimited: false, relativeReferenceIntensity: 10,
                                        relativeReferenceWasAvailable: true, correlationMaximum: 10))
        #expect(r.peaks.map(\.intensity) == [10, 5])
        r = try run { $0.minimumAbsoluteIntensity = 0.5; $0.minimumRelativeIntensity = 0.15; $0.minimumPeakSpacing = 5; $0.maximumPeaks = 1 }
        #expect(r.funnel == BraggFunnel(localMaximumCount: 5, afterAbsoluteCutCount: 4, afterRelativeCutCount: 3, afterSpacingCount: 2,
                                        acceptedCount: 1, wasCountLimited: true, relativeReferenceIntensity: 10,
                                        relativeReferenceWasAvailable: true, correlationMaximum: 10))
        #expect(r.peaks.count == 1)
        // The reference is the maximum of rank relativeToPeak (B = 5: 0.3 of it removes C and D and keeps E? 2/5 = 0.4 kept).
        r = try run { $0.minimumRelativeIntensity = 0.3; $0.relativeToPeak = 1 }
        #expect(r.funnel.relativeReferenceIntensity == 5 && r.funnel.afterRelativeCutCount == 3 && r.peaks.map(\.intensity) == [10, 5, 2])
        // A relative cut with no maximum of that rank: the cut does nothing and says it had no reference.
        r = try run { $0.minimumRelativeIntensity = 0.3; $0.relativeToPeak = 9 }
        #expect(r.funnel.relativeReferenceIntensity == nil && !r.funnel.relativeReferenceWasAvailable
                && r.funnel.afterRelativeCutCount == 5 && r.funnel.acceptedCount == 5)
        // The mac4DSTEM reference radius: C and D lie far from A, E near; with radius 10 the reference is B (the brightest
        // beyond 10 px of A), so E (2) is kept at 0.3 of 5 and D (0.03) is cut.
        r = try run { $0.minimumRelativeIntensity = 0.3; $0.relativeReferenceMinimumRadius = 10 }
        #expect(r.funnel.relativeReferenceIntensity == 5 && r.funnel.afterRelativeCutCount == 3)
        // Nothing found: edge boundary larger than the map, and an empty map.
        r = try run { $0.edgeBoundary = 15 }
        #expect(r.funnel == BraggFunnel(localMaximumCount: 0, afterAbsoluteCutCount: 0, afterRelativeCutCount: 0, afterSpacingCount: 0,
                                        acceptedCount: 0, wasCountLimited: false, relativeReferenceIntensity: nil,
                                        relativeReferenceWasAvailable: true, correlationMaximum: 10))
        let empty = try BraggMaxima.findWithFunnel(in: [Double](repeating: 0, count: 900), rows: 30, columns: 30, options: o)
        #expect(empty.peaks.isEmpty && empty.funnel.localMaximumCount == 0 && empty.funnel.correlationMaximum == 0)
        // The relative cut asked for but nothing there to measure: not available.
        var cut = o; cut.minimumRelativeIntensity = 0.1
        let none = try BraggMaxima.findWithFunnel(in: [Double](repeating: 0, count: 900), rows: 30, columns: 30, options: cut)
        #expect(!none.funnel.relativeReferenceWasAvailable)
        // find is findWithFunnel's peaks.
        #expect(try BraggMaxima.find(in: handMap(), rows: 30, columns: 30, options: o) == run { _ in }.peaks)
    }

    /// On every fixture case the funnel is consistent with the peaks and `detectWithFunnel` returns `detect`'s peaks, in both
    /// precisions; `.single` and `.double` count the same maxima.
    @Test func funnelMatchesPeaksOnFixtures() throws {
        for c in fixture.cases {
            let id = c["id"] as! String
            let s = fixture.scene(c["scene"] as! String)
            let pattern = s.patterns[c["pattern"] as! Int]
            let o = options(c["options"] as! [String: Any])
            let detector = try BraggDetector(kernel: s.kernel)
            let precisions: [BraggPrecision] = supported(c) ? [.double, .single] : [.double]
            var funnels: [BraggFunnel] = []
            for precision in precisions {
                var p = o; p.precision = precision
                let (peaks, funnel) = try detector.detectWithFunnel(pattern, options: p)
                #expect(peaks == (try detector.detect(pattern, options: p)), "\(id) \(precision)")
                #expect(funnel.acceptedCount == peaks.count, "\(id) \(precision)")
                #expect(funnel.localMaximumCount >= funnel.afterAbsoluteCutCount && funnel.afterAbsoluteCutCount >= funnel.afterRelativeCutCount
                        && funnel.afterRelativeCutCount >= funnel.afterSpacingCount && funnel.afterSpacingCount >= funnel.acceptedCount, "\(id)")
                #expect(funnel.wasCountLimited == (funnel.afterSpacingCount > funnel.acceptedCount), "\(id)")
                #expect(funnel.acceptedCount == min(funnel.afterSpacingCount, p.maximumPeaks), "\(id)")
                if p.minimumAbsoluteIntensity == 0 { #expect(funnel.afterAbsoluteCutCount == funnel.localMaximumCount, "\(id)") }
                if p.minimumPeakSpacing == 0 { #expect(funnel.afterSpacingCount == funnel.afterRelativeCutCount, "\(id)") }
                if funnel.localMaximumCount > 0, (c["id"] as! String) != "s3p2_nonfinite" { #expect(funnel.correlationMaximum > 0, "\(id)") }
                if p.minimumRelativeIntensity > 0, funnel.relativeReferenceWasAvailable {
                    #expect(funnel.relativeReferenceIntensity != nil, "\(id)")
                }
                funnels.append(funnel)
            }
            if funnels.count == 2 {
                // Counts agree after the cuts; the raw maxima count can differ by the maxima that are rounding noise of the
                // zero-valued parts of the map (Float against Double), which the relative cut removes.
                let (d, f) = (funnels[0], funnels[1])
                if o.minimumRelativeIntensity > 0, o.correlationPower > 0 {
                    #expect(d.acceptedCount == f.acceptedCount && d.afterRelativeCutCount == f.afterRelativeCutCount
                            && d.afterSpacingCount == f.afterSpacingCount, "\(id): \(d) vs \(f)")
                }
                if o.correlationPower > 0 {
                    #expect(abs(d.correlationMaximum - f.correlationMaximum) <= SingleTolerance.correlation * max(d.correlationMaximum, 1e-300), "\(id)")
                }
            }
        }
    }

    /// Funnel's non-finite count: the three bad pixels of S3 pattern 2, filled, in both precisions.
    @Test func funnelCountsFilledPixels() throws {
        let s = fixture.scene("S3")
        let detector = try BraggDetector(kernel: s.kernel)
        for precision in BraggPrecision.allCases {
            var o = BraggDiskOptions(); o.nonFinitePixels = .neighbourMedian; o.precision = precision
            o.minimumPeakSpacing = 8; o.edgeBoundary = 5
            let withBad = try detector.detectWithFunnel(s.patterns[2], options: o)
            #expect(withBad.funnel.nonFinitePixelCount == 3, "\(precision)")
            #expect(withBad.funnel.acceptedCount >= 3)
            let clean = try detector.detectWithFunnel(s.patterns[0], options: o)
            #expect(clean.funnel.nonFinitePixelCount == 0)
            o.nonFinitePixels = .propagate
            let propagated = try detector.detectWithFunnel(s.patterns[2], options: o)
            #expect(propagated.peaks.isEmpty && propagated.funnel.nonFinitePixelCount == 0, "\(precision)")
        }
    }

    // MARK: fillNonFinite

    @Test func fillNonFiniteExact() {
        // 3 × 4. NaN at (1,1): finite neighbours (8) 1,2,3,5,7,9,10,11 -> sorted, even count 8 -> mean of the 4th and 5th.
        // inf at (0,3) corner: neighbours (0,2)=4, (1,2)=8, (1,3)=12 -> odd count 3 -> 8. -inf at (2,0): neighbours
        // (1,0)=5, (1,1)=NaN(excluded), (2,1)=10 -> even count 2 -> mean 7.5.
        let nan = Double.nan
        let image: [Double] = [1, 2, 4, .infinity,
                               5, nan, 8, 12,
                               -.infinity, 10, 11, 13]
        // (1,1)'s finite neighbours: (0,0)=1,(0,1)=2,(0,2)=4,(1,0)=5,(1,2)=8,(2,0)=-inf excluded,(2,1)=10,(2,2)=11 -> 1,2,4,5,8,10,11: 7, median 5.
        var d = image
        #expect(BraggDetector.fillNonFinite(&d, rows: 3, columns: 4) == 3)
        #expect(d == [1, 2, 4, 8, 5, 5, 8, 12, 7.5, 10, 11, 13])
        var f = image.map { Float($0) }
        #expect(BraggDetector.fillNonFinite(&f, rows: 3, columns: 4) == 3)
        #expect(f == [1, 2, 4, 8, 5, 5, 8, 12, 7.5, 10, 11, 13])
        // Neighbours are read before any replacement: two adjacent NaNs do not feed each other.
        var pair: [Double] = [1, nan, nan, 4, 5, 6, 7, 8]   // 2 × 4
        #expect(BraggDetector.fillNonFinite(&pair, rows: 2, columns: 4) == 2)
        // (0,1): finite neighbours (0,0)=1,(1,0)=5,(1,1)=6,(1,2)=7 -> 1,5,6,7 -> 5.5. (0,2): (1,1)=6,(1,2)=7,(1,3)=8,(0,3)=4 -> 4,6,7,8 -> 6.5.
        #expect(pair == [1, 5.5, 6.5, 4, 5, 6, 7, 8])
        // No finite neighbour: 0.
        var alone: [Double] = [nan, nan, nan, nan, nan, nan, nan, nan, nan]
        #expect(BraggDetector.fillNonFinite(&alone, rows: 3, columns: 3) == 9)
        #expect(alone == [Double](repeating: 0, count: 9))
        var aloneFloat = [Float](repeating: .infinity, count: 9)
        #expect(BraggDetector.fillNonFinite(&aloneFloat, rows: 3, columns: 3) == 9)
        #expect(aloneFloat == [Float](repeating: 0, count: 9))
        // A finite image is untouched and counts 0 (including a sum that overflows).
        var big = [Float](repeating: Float.greatestFiniteMagnitude, count: 12)
        #expect(BraggDetector.fillNonFinite(&big, rows: 3, columns: 4) == 0)
        #expect(big == [Float](repeating: Float.greatestFiniteMagnitude, count: 12))
        var clean: [Double] = (0..<12).map(Double.init)
        #expect(BraggDetector.fillNonFinite(&clean, rows: 3, columns: 4) == 0)
        #expect(clean == (0..<12).map(Double.init))
        // The result is what the correlation uses: neighbourMedian reproduces the explicit fill.
        var empty: [Double] = []
        #expect(BraggDetector.fillNonFinite(&empty, rows: 0, columns: 0) == 0)
    }
}

/// The tolerances of the `.single` path, written per quantity from the maxima measured in this suite (docs/bragg.md).
/// The Float32 transform has an absolute error of about 1e-6 of the correlation maximum, so a peak's accuracy falls with its
/// height: peaks above 1e-3 of the brightest (every peak of the pins' default cascade) are held to 5e-5 px and 5e-6 (the app's
/// own tolerance against py4DSTEM is 2e-4 px); fainter ones (the pins with the spacing filter off reach 4e-5 of the brightest) to 1e-3 px and 5e-4. Phase correlation
/// (`correlationPower` 0) is not held: bins at rounding-noise level get unit weight and a random phase in Float32 (docs/bragg.md).
enum SingleTolerance {
    /// Peaks above `faintBelow` of the brightest: measured 1.3e-5 px and 1.8e-6 (64 × 64 pins).
    static let position = 5e-5, intensity = 5e-6
    /// Fainter peaks: measured 4.4e-4 px and 2.4e-4.
    static let faintPosition = 1e-3, faintIntensity = 5e-4
    /// A peak is faint below this fraction of the brightest peak of its list.
    static let faintBelow = 1e-3
    /// The real correlation map, as a fraction of its maximum.
    static let correlation = 1e-5
    /// The complex Fourier product, as a fraction of its largest magnitude.
    static let fourier = 3e-5

    /// Compares two peak lists of equal length; records an issue per peak outside its tolerance; returns the worst gaps.
    @discardableResult
    static func check(_ actual: [BraggPeak], _ expected: [BraggPeak], _ id: String)
        -> (position: Double, intensity: Double, strongPosition: Double, strongIntensity: Double) {
        let brightest = expected.map(\.intensity).max() ?? 0
        var worstPosition = 0.0, worstIntensity = 0.0, strongPosition = 0.0, strongIntensity = 0.0
        for (a, e) in zip(actual, expected) {
            let position = max(abs(a.row - e.row), abs(a.column - e.column))
            let intensity = abs(a.intensity - e.intensity) / max(abs(e.intensity), 1e-300)
            let faint = e.intensity < faintBelow * brightest
            #expect(position <= (faint ? faintPosition : self.position) && intensity <= (faint ? faintIntensity : self.intensity),
                    "\(id): peak at \(e.row), \(e.column): position gap \(position), intensity gap \(intensity)")
            worstPosition = max(worstPosition, position); worstIntensity = max(worstIntensity, intensity)
            if !faint { strongPosition = max(strongPosition, position); strongIntensity = max(strongIntensity, intensity) }
        }
        return (worstPosition, worstIntensity, strongPosition, strongIntensity)
    }
}

/// Timing, 256 × 256 patterns: `CUBOKIT_BENCH=1 swift test -c release --filter CuboBraggTests.BraggBenchmark`. Skipped otherwise.
@Suite struct BraggBenchmark {
    static let size = 256

    /// A direct beam and a lattice of disks with Poisson-like noise, seeded.
    static func pattern(seed: UInt64) -> [Float] {
        var state = seed &* 6364136223846793005 &+ 1442695040888963407
        func uniform() -> Double {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Double(state >> 11) / Double(1 << 53)
        }
        let n = size
        var a = [Float](repeating: 0, count: n * n)
        let disks: [(Double, Double, Double)] = [(128, 128, 4000), (128 + 40.3, 128 + 10.7, 700), (128 - 40.3, 128 - 10.7, 650), (128 - 10.7, 128 + 40.3, 500),
                                                 (128 + 10.7, 128 - 40.3, 520), (128 + 29.6, 128 + 51, 160), (128 - 29.6, 128 - 51, 150),
                                                 (128 + 80.6, 128 + 21.4, 90), (128 - 80.6, 128 - 21.4, 85)]
        for r in 0..<n { for c in 0..<n {
            var v = 3.0
            for (dr, dc, height) in disks {
                let d = hypot(Double(r) - dr, Double(c) - dc)
                v += height / (1 + exp((d - 6) / 0.8))
            }
            a[r * n + c] = Float(v + (uniform() - 0.5) * 2 * v.squareRoot() * 0.3)
        } }
        return a
    }

    static func kernel() -> DetectorImage {
        let n = size
        var k = [Double](repeating: 0, count: n * n)
        for r in 0..<n { for c in 0..<n {
            let dr = Double(min(r, n - r)), dc = Double(min(c, n - c))
            k[r * n + c] = 1 / (1 + exp((hypot(dr, dc) - 6) / 0.8))
        } }
        return DetectorImage(detectorRows: n, detectorColumns: n, values: k, path: .cpu)
    }

    @Test func timings() async throws {
        guard ProcessInfo.processInfo.environment["CUBOKIT_BENCH"] != nil else { return }
        let n = Self.size
        let kernel = Self.kernel()
        let patterns = (0..<16).map { Self.pattern(seed: UInt64($0 + 1)) }
        let detector = try BraggDetector(kernel: kernel)
        let wideImages = patterns.map { p in DetectorImage(detectorRows: n, detectorColumns: n, values: p.map { Double($0) }, path: .cpu) }
        for (label, subpixel) in [("poly", BraggSubpixel.parabolic), ("multicorr", .multicorr)] {
            var o = BraggDiskOptions.mac4DSTEM
            o.subpixel = subpixel; o.edgeBoundary = 10; o.minimumPeakSpacing = 12
            // Agreement on this scene first.
            var worst = 0.0, worstIntensity = 0.0
            for (p, w) in zip(patterns, wideImages) {
                var os = o; os.precision = .single
                let d = try detector.detect(w, options: o)
                let f = try p.withUnsafeBufferPointer { try detector.detect($0, options: os) }
                #expect(d.count == f.count && d.count >= 5)
                let g = gaps(f, d); worst = max(worst, g.position); worstIntensity = max(worstIntensity, g.intensity)
            }
            print("BENCH 256x256 \(label): single vs double, max position gap \(worst) px, relative intensity gap \(worstIntensity)")
            // Serial ms per pattern.
            for precision in BraggPrecision.allCases {
                var p = o; p.precision = precision
                let reps = subpixel == .multicorr ? 6 : 40
                for k in 0..<4 { _ = try patterns[k].withUnsafeBufferPointer { try detector.detect($0, options: p) } }   // warm up
                let start = DispatchTime.now().uptimeNanoseconds
                for k in 0..<reps { _ = try patterns[k % 16].withUnsafeBufferPointer { try detector.detect($0, options: p) } }
                let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 / Double(reps)
                print("BENCH 256x256 \(label) \(precision) serial: \(ms) ms per pattern")
            }
            // Wall ms per pattern over a cube on every core.
            let positions = subpixel == .multicorr ? 256 : 2048
            let shape = try CubeShape(scanRows: positions, scanColumns: 1, detectorRows: n, detectorColumns: n)
            var values: [Float] = []
            values.reserveCapacity(positions * n * n)
            for k in 0..<positions { values += patterns[k % 16] }
            let cube = try MemoryCube(shape: shape, values: values)
            for precision in BraggPrecision.allCases {
                var p = o; p.precision = precision
                _ = try await BraggDisks.detectAll(in: cube, kernel: kernel, options: p, tiling: .rows(64))   // warm up
                let start = DispatchTime.now().uptimeNanoseconds
                _ = try await BraggDisks.detectAll(in: cube, kernel: kernel, options: p, tiling: .rows(64))
                let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 / Double(positions)
                print("BENCH 256x256 \(label) \(precision) detectAll, \(ProcessInfo.processInfo.activeProcessorCount) cores: \(ms) ms per pattern wall")
            }
        }
    }
}
