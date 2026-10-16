import Foundation
import Testing
@testable import CuboCore

@Suite struct FrameTransformTests {
    private func fixture() throws -> [String: Any] {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/frametransform.json")
        return try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    }
    private func f(_ x: Any) -> Float { Float((x as! NSNumber).doubleValue) }
    private func fa(_ x: Any) -> [Float] { (x as! [Any]).map(f) }

    @Test func coordinatesBitForBit() throws {
        for c in try fixture()["coords"] as! [[String: Any]] {
            let v = f(c["v"]!), b = (c["bin"] as! NSNumber).intValue
            #expect(FrameTransform.binnedCoordinate(v, bin: b) == f(c["binned"]!))
            #expect(FrameTransform.sourceCoordinate(v, bin: b) == f(c["source"]!))
        }
        #expect(abs(FrameTransform.sourceCoordinate(FrameTransform.binnedCoordinate(37, bin: 4), bin: 4) - 37) < 1e-4)
    }

    @Test func positionsAndBounds() throws {
        for p in try fixture()["positions"] as! [[String: Any]] {
            let c: Float = p["c"] is NSNull ? .nan : f(p["c"]!)
            let r = FrameTransform.movePosition(column: c, row: f(p["r"]!), columnOffset: (p["co"] as! NSNumber).intValue,
                rowOffset: (p["ro"] as! NSNumber).intValue, bin: (p["bin"] as! NSNumber).intValue,
                width: (p["w"] as! NSNumber).intValue, height: (p["h"] as! NSNumber).intValue)
            #expect((r != nil) == (p["inside"] as! Bool))
            if let r { #expect(r.column == f(p["mc"]!) && r.row == f(p["mr"]!)) }
        }
    }

    @Test func scanCropSelectsExactly() throws {
        let c = try fixture()["crop"] as! [String: Any]
        let n = { (k: String) in (c[k] as! NSNumber).intValue }
        let out = FrameTransform.cropMap(fa(c["values"]!), mapWidth: n("w"), mapHeight: n("h"), columnOffset: n("co"),
                                         rowOffset: n("ro"), width: n("cw"), height: n("ch"))
        #expect(out == fa(c["expected"]!))
        #expect(FrameTransform.cropMap(fa(c["values"]!), mapWidth: 6, mapHeight: 5, columnOffset: 4, rowOffset: 0, width: 4, height: 3) == nil)
    }

    @Test func originMaps() throws {
        let j = try fixture()
        for k in ["map_a", "map_b"] {
            let m = j[k] as! [String: Any]
            let n = { (s: String) in (m[s] as! NSNumber).intValue }
            let r = try #require(FrameTransform.moveOriginMap(columns: fa(m["cols"]!), rows: fa(m["rows"]!), columnOffset: n("co"),
                rowOffset: n("ro"), bin: n("bin"), width: n("w"), height: n("h")))
            #expect(r.columns == fa(m["mc"]!) && r.rows == fa(m["mr"]!))
            #expect(r.firstOutside == (m["first"] as? NSNumber)?.intValue)
        }
    }

    @Test func scalars() {
        #expect(FrameTransform.binnedLength(8, bin: 2) == 4)
        #expect(FrameTransform.binnedPixelSize(0.5, bin: 4) == 2)
    }
}
