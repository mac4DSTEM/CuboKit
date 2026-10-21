"""Generate CuboBragg pins. Run with memcap and the pinned venv, from the package root:
../../tools/memcap.sh -- ../../References/python/py4dstem/bin/python -I -B tools/bragg/pins.py

Two independent oracles, neither is Swift written for the kit:
  1. py4DSTEM 0.14.19 (f050d207) `find_Bragg_disks`, called as a public function, is the primary reference.
  2. The mac4DSTEM source at APP (Core/Analysis/DiskDetection.swift, Core/Compute/MatrixDFTCorrelation.swift,
     Core/Analysis/ProbeRingHint.swift, with ImagoKit 0.1.1's Float32 FFT2D) is compiled UNCHANGED apart from
     access qualifiers and four minimal shims (ProbeKernel's fields, a probe-size stub fed from py4DSTEM) and run.
Scenes are synthetic (logistic-edge disks at fractional positions, seeded noise); no owner data is used.
"""
import hashlib, json, platform, subprocess, tempfile
from pathlib import Path
import numpy as np
import scipy
import py4DSTEM
from py4DSTEM import Probe
from py4DSTEM.braggvectors.diskdetection import find_Bragg_disks
from py4DSTEM.process.calibration import get_probe_size

ROOT = Path(__file__).resolve().parents[2]
SUITE = next(p for p in ROOT.parents if (p / 'References/CuboKit').is_dir())
REF = SUITE / 'References/CuboKit'
PIN = 'f050d207851f8accd3aa550365baaf1f38230428'
assert subprocess.check_output(['git', '-C', str(REF / 'py4DSTEM-f050d207'), 'rev-parse', 'HEAD'], text=True).strip() == PIN
assert py4DSTEM.__version__ == '0.14.19' and np.__version__ == '2.5.4'
assert platform.python_version() == '3.12.14' and scipy.__version__ == '1.18.1'
for rel in ['braggvectors/diskdetection.py', 'preprocess/utils.py', 'process/utils/cross_correlate.py',
            'process/utils/multicorr.py', 'braggvectors/probe.py', 'process/calibration/probe.py']:
    assert (Path(py4DSTEM.__file__).parent / rel).read_bytes() == (REF / 'py4DSTEM-f050d207/py4DSTEM' / rel).read_bytes(), rel

APP = 'c809679c8ee23a355da6af4f63665aaf64f90d0d'
IMAGO = '0.1.1'
APPREPO = SUITE / 'mac4DSTEM'
def app_file(path):
    return subprocess.check_output(['git', '-C', str(APPREPO), '--no-optional-locks', 'show', f'{APP}:{path}'], text=True)

# ---------------------------------------------------------------- scenes
rng = np.random.default_rng(20261016)
def disk(qy, qx, cy, cx, amplitude, radius=4.0, width=2.0):
    yy, xx = np.mgrid[0:qy, 0:qx]
    return amplitude / (1 + np.exp(4 * (np.hypot(yy - cy, xx - cx) - radius) / width))
def pattern(qy, qx, beam, disks, noise=0.5, radius=4.0, plateau=None):
    dp = disk(qy, qx, beam[0], beam[1], beam[2], radius)
    if plateau is not None: dp = np.minimum(dp, plateau)
    for cy, cx, a in disks: dp = dp + disk(qy, qx, cy, cx, a, radius)
    return dp + noise * rng.standard_normal((qy, qx))
def kernel(qy, qx, radius):
    probe = Probe.generate_synthetic_probe(radius, 2.0, (qy, qx))
    return Probe.get_probe_kernel_edge_sigmoid(probe.probe, (radius, 2 * radius), origin=(qy / 2, qx / 2))

arrays = {}
def put(name, a): arrays[name] = np.ascontiguousarray(a, dtype='<f8')
scenes = {}

# S1: 64 x 64, three patterns. p0 has a bright beam and five disks; p1 has two weak disks (8, 12) next to strong ones;
# p2 has two disks 6 px apart (the spacing filter) and a disk near the edge.
k1 = kernel(64, 64, 4.0)
s1 = [pattern(64, 64, (31.6, 32.3, 1000), [(10.3, 12.7, 200), (50.2, 13.1, 150), (14.8, 50.4, 120), (48.9, 49.6, 80), (31.7, 8.2, 60)]),
      pattern(64, 64, (32.4, 31.7, 1000), [(12.6, 31.4, 140), (51.3, 30.2, 90), (30.2, 52.7, 40), (20.5, 20.5, 8), (44.1, 44.6, 12)]),
      pattern(64, 64, (32.0, 32.0, 800), [(14.0, 14.0, 100), (14.0, 20.0, 90), (3.4, 40.2, 70), (52.2, 22.4, 50)])]
put('k1', k1); put('p1', np.array(s1)); scenes['S1'] = dict(rows=64, columns=64, kernel='k1', patterns='p1')
# S2: 40 x 56, a grid that is neither square nor a power of two (Bluestein on both axes).
k2 = kernel(40, 56, 3.0)
s2 = [pattern(40, 56, (19.7, 28.4, 600), [(8.3, 10.6, 90), (30.6, 44.2, 70), (9.1, 46.5, 55)], radius=3.0),
      pattern(40, 56, (20.2, 27.9, 600), [(7.7, 28.1, 100), (33.4, 12.8, 60)], radius=3.0)]
put('k2', k2); put('p2', np.array(s2)); scenes['S2'] = dict(rows=40, columns=56, kernel='k2', patterns='p2')

# S3: the mac4DSTEM-variant scene. p0: a saturated direct beam (a plateau, so correlation maxima crowd it) and disks of
# 2 to 7 % of the plateau; p1: reorder (below); p2: three non-finite pixels, one on a disk.
def reorder_pattern(ratio):
    return pattern(64, 64, (32.5, 32.5, 1000), [(20.0, 20.0, 100), (40.5, 44.5, 100 * ratio), (14.4, 48.7, 40)], noise=0.0)
def poly_ordered(dp):
    r = find_Bragg_disks(dp, k1, minRelativeIntensity=0.005, minPeakSpacing=8, edgeBoundary=5, subpixel='poly')
    v = r.data['intensity']
    return bool(np.all(np.diff(v) <= 0))
ratio = next(r for r in np.arange(1.0, 1.4, 0.005) if not poly_ordered(reorder_pattern(r)))
s3 = [pattern(64, 64, (31.8, 32.2, 1000), [(11.2, 13.4, 70), (52.6, 14.9, 45), (13.1, 51.3, 30), (49.7, 50.8, 20), (32.4, 9.1, 25)], plateau=300),
      reorder_pattern(ratio),
      pattern(64, 64, (31.8, 32.2, 1000), [(11.2, 13.4, 200), (52.6, 14.9, 150), (13.1, 51.3, 100), (49.7, 50.8, 60)])]
nonfinite = [[2, 11 * 64 + 13, 'nan'], [2, 52 * 64 + 15, 'inf'], [2, 40 * 64 + 5, 'nan']]
s3_nonfinite = [p.copy() for p in s3]
for pi, idx, kind in nonfinite: s3_nonfinite[pi].flat[idx] = np.nan if kind == 'nan' else np.inf
put('p3', np.array(s3_nonfinite)); scenes['S3'] = dict(rows=64, columns=64, kernel='k1', patterns='p3')
put('p3finite', np.array(s3))

# ---------------------------------------------------------------- py4DSTEM cases (primary)
DEFAULT = dict(corrPower=1, sigma_dp=0, sigma_cc=2, subpixel='poly', upsample_factor=16, minAbsoluteIntensity=0,
               minRelativeIntensity=0.005, relativeToPeak=0, minPeakSpacing=8, edgeBoundary=5, maxNumPeaks=70)
def variants(**kw):
    d = dict(DEFAULT); d.update(kw); return d
cases = []
def py4_run(dp, k, opt):
    r = find_Bragg_disks(dp, k, **opt)
    return np.column_stack([r.data['qx'], r.data['qy'], r.data['intensity']])
def add_py4(cid, scene, p, opt, patterns_name=None):
    s = scenes[scene]
    dp = arrays[s['patterns']][p].astype(float)
    if np.isnan(dp).any() or np.isinf(dp).any():
        pass
    out = py4_run(dp, arrays[s['kernel']], opt)
    if len(out) > 1:
        assert len(set(np.round(out[:, 2], 12))) == len(out), cid   # no tied intensities: tie order is unspecified
    put('e_' + cid, out.reshape(-1, 3))
    jopt = dict(opt); jopt['subpixel'] = ['pixel', 'poly', 'multicorr'].index(opt['subpixel']); jopt['relativeReferenceMinimumRadiusPx'] = 0
    cases.append(dict(id=cid, scene=scene, pattern=p, reference='py4dstem', options=jopt, expected='e_' + cid))
    return out
for p in range(3):
    add_py4(f's1p{p}_pixel', 'S1', p, variants(subpixel='pixel'))
    add_py4(f's1p{p}_poly', 'S1', p, variants(subpixel='poly'))
    add_py4(f's1p{p}_multicorr', 'S1', p, variants(subpixel='multicorr'))
add_py4('s1p0_power0.5', 'S1', 0, variants(corrPower=0.5))
add_py4('s1p0_power0', 'S1', 0, variants(corrPower=0.0))
add_py4('s1p1_sigmaDP1', 'S1', 1, variants(sigma_dp=1.0, sigma_cc=1.0, subpixel='multicorr'))
add_py4('s1p1_sigma0_multicorr_u8', 'S1', 1, variants(sigma_cc=0, subpixel='multicorr', upsample_factor=8))
add_py4('s1p1_sigma0_multicorr_u3', 'S1', 1, variants(sigma_cc=0, subpixel='multicorr', upsample_factor=3))
add_py4('s1p1_absolute', 'S1', 1, variants(minAbsoluteIntensity=5.0, minRelativeIntensity=0.0))
add_py4('s1p1_relative_rank1', 'S1', 1, variants(relativeToPeak=1, minRelativeIntensity=0.25))
add_py4('s1p0_maxPeaks3', 'S1', 0, variants(maxNumPeaks=3, minRelativeIntensity=0.0))
add_py4('s1p2_spacing_off', 'S1', 2, variants(minPeakSpacing=0, minRelativeIntensity=0.0))
add_py4('s1p2_spacing8', 'S1', 2, variants(minPeakSpacing=8))
add_py4('s1p2_edge1', 'S1', 2, variants(edgeBoundary=1, minRelativeIntensity=0.0))
add_py4('s1p2_edge0', 'S1', 2, variants(edgeBoundary=0, minRelativeIntensity=0.0))
add_py4('s1p0_edge30', 'S1', 0, variants(edgeBoundary=30))
for p in range(2):
    add_py4(f's2p{p}_poly', 'S2', p, variants(minPeakSpacing=6, edgeBoundary=4))
    add_py4(f's2p{p}_multicorr', 'S2', p, variants(minPeakSpacing=6, edgeBoundary=4, subpixel='multicorr'))
# S3 through py4DSTEM: the saturated beam as the relative reference; the reorder pattern in multicorr; NaN / inf (empty).
add_py4('s3p0_poly', 'S3', 0, variants(minRelativeIntensity=0.05))
add_py4('s3p0_rel0.5', 'S3', 0, variants(minRelativeIntensity=0.5))
add_py4('s3p1_multicorr', 'S3', 1, variants(subpixel='multicorr'))
nan_result = add_py4('s3p2_nonfinite', 'S3', 2, variants())
assert len(nan_result) == 0
assert not poly_ordered(s3[1])

# ---------------------------------------------------------------- probe-ring hint inputs (py4DSTEM get_probe_size feeds the app stub)
def bullseye(cy, cx, inner, gap, outer, smooth=0.0):
    yy, xx = np.mgrid[0:64, 0:64]; r = np.hypot(yy - cy, xx - cx)
    prof = np.where(r < inner, 1.0, np.where(r < gap, 0.04, np.where(r < outer, 0.6, 0.0)))
    return prof
hints_in = {
    'bullseye': bullseye(31.37, 32.61, 3.0, 5.0, 8.0) + 0.002 * np.abs(rng.standard_normal((64, 64))),
    'disk': disk(64, 64, 31.37, 32.61, 1.0, radius=6.0, width=1.0),
    'wide_ring': bullseye(31.37, 32.61, 3.0, 6.0, 9.4),
    'no_peak': np.zeros((64, 64)),
}
for n, a in hints_in.items(): put('h_' + n, a)

# ---------------------------------------------------------------- the app oracle (Float32, compiled from the app source)
def build_app_oracle(tmp):
    dd = app_file('mac4DSTEM/Core/Analysis/DiskDetection.swift').split('\n')
    # 1-indexed ranges, located by their anchors rather than by number
    def find(prefix, start=0):
        return next(i for i in range(start, len(dd)) if dd[i].startswith(prefix))
    a0 = find('package nonisolated struct BraggPeak') - 3           # doc comment of BraggPeak
    a1 = find('/// Compares the provenance a')                       # types up to the record match
    c0 = find('package nonisolated struct DiskDetectionContext')
    c1 = find('package nonisolated struct DiskDetectionScanSummary')
    p0 = find('/// Detection parameters, mirroring')
    p1 = find('/// Detected peaks for every scan position')
    d0 = find('// MARK: - DiskDetector')
    d1 = find('// MARK: - Full-scan detection')
    body = dd[a0:a1] + dd[c0:c1] + dd[p0:p1] + dd[d0:d1]
    head = '''import Foundation
import Accelerate
package enum ProbeKernelSource { case synthetic; package var provenanceID: String { "s" } }
package enum ProbeKernelMode { case flat; package var provenanceID: String { "m" } }
package struct ProbeKernel { package let px: Int; package let py: Int; package let qx: Int; package let qy: Int
  package let probeRadius: Float; package let trenchRadii: (inner: Float, outer: Float)
  package let source: ProbeKernelSource; package let mode: ProbeKernelMode; package let probePath: String?
  package let ftRe: [Float]; package let ftIm: [Float] }
package protocol FFT2DTransform {}
'''
    text = head + '\n'.join(body)
    (tmp / 'app_dd.swift').write_text(text)
    mdc = app_file('mac4DSTEM/Core/Compute/MatrixDFTCorrelation.swift').replace('import ImagoCore', '')
    (tmp / 'mdc.swift').write_text(mdc)
    fft = subprocess.check_output(['git', '-C', str(SUITE / 'ImagoKit'), 'show', f'{IMAGO}:Sources/ImagoCore/FFT2D.swift'], text=True)
    (tmp / 'fft2d.swift').write_text(fft)
    ring = app_file('mac4DSTEM/Core/Analysis/ProbeRingHint.swift')
    ring_shim = '''
enum OriginCalibration {
    nonisolated(unsafe) static var stub: (r: Float, x0: Float, y0: Float)? = nil
    static func probeSize(dp: [Float], qy: Int, qx: Int) -> (r: Float, x0: Float, y0: Float)? { stub }
}
'''
    (tmp / 'ring.swift').write_text(ring + ring_shim)
    return hashlib.sha256(text.encode()).hexdigest(), hashlib.sha256(mdc.encode()).hexdigest(), hashlib.sha256(ring.encode()).hexdigest(), hashlib.sha256(fft.encode()).hexdigest()

RUNNER = r'''
import Foundation
let input = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))) as! [String: Any]
var out = [String: Any]()
let scenes = input["scenes"] as! [[String: Any]]
var runs = [[[Double]]]()
for r in input["runs"] as! [[String: Any]] {
    let s = scenes[r["scene"] as! Int]
    let qy = s["rows"] as! Int, qx = s["columns"] as! Int
    let k = (s["kernel"] as! [Double]).map { Float($0) }
    var re = k, im = [Float](repeating: 0, count: k.count)
    let fft = FFT2D(nx: qx, ny: qy)!
    fft.transform(re: &re, im: &im, forward: true)
    for i in im.indices { im[i] = -im[i] }
    let kernel = ProbeKernel(px: qx, py: qy, qx: qx, qy: qy, probeRadius: 4, trenchRadii: (0, 0), source: .synthetic, mode: .flat,
                             probePath: nil, ftRe: re, ftIm: im)
    var pat = ((s["patterns"] as! [[Double]])[r["pattern"] as! Int]).map { Float($0) }
    for nf in (s["nonfinite"] as! [[Any]]) where (nf[0] as! Int) == (r["pattern"] as! Int) {
        pat[nf[1] as! Int] = (nf[2] as! String) == "nan" ? Float.nan : Float.infinity
    }
    let o = r["options"] as! [String: Double]
    var p = DiskDetectionParams()
    p.corrPower = Float(o["corrPower"]!); p.sigmaDP = Float(o["sigma_dp"]!); p.sigmaCC = Float(o["sigma_cc"]!)
    p.subpixel = o["subpixel"]! == 0 ? .pixel : (o["subpixel"]! == 1 ? .poly : .multicorr)
    p.upsampleFactor = Int(o["upsample_factor"]!)
    p.minAbsoluteIntensity = Float(o["minAbsoluteIntensity"]!); p.minRelativeIntensity = Float(o["minRelativeIntensity"]!)
    p.relativeToPeak = Int(o["relativeToPeak"]!); p.relativeReferenceMinimumRadiusPx = Float(o["relativeReferenceMinimumRadiusPx"]!)
    p.minPeakSpacing = Float(o["minPeakSpacing"]!); p.edgeBoundary = Int(o["edgeBoundary"]!); p.maxNumPeaks = Int(o["maxNumPeaks"]!)
    let det = DiskDetector(kernel: kernel)!
    runs.append(det.detect(pattern: pat, params: p).map { [Double($0.y), Double($0.x), Double($0.intensity)] })   // row, column, intensity
}
out["runs"] = runs
out["adapted"] = (input["adapted"] as! [[Double]]).map { a -> [Double] in
    let p = DiskDetectionParams.detectorAdapted(qy: Int(a[0]), qx: Int(a[1]), probeRadius: a[2] < 0 ? nil : Float(a[2]))
    return [Double(p.minPeakSpacing), Double(p.edgeBoundary)]
}
var hints = [[Double]]()
for h in input["hints"] as! [[String: Any]] {
    let pat = (h["pattern"] as! [Double]).map { Float($0) }
    if let probe = h["probe"] as? [Double] { OriginCalibration.stub = (Float(probe[0]), Float(probe[1]), Float(probe[2])) } else { OriginCalibration.stub = nil }
    if let r = ProbeRingHint.outerEdge(meanDP: pat, qy: 64, qx: 64) { hints.append([Double(r.outerRadius), Double(r.dip)]) } else { hints.append([]) }
}
out["hints"] = hints
print(String(data: try! JSONSerialization.data(withJSONObject: out, options: [.sortedKeys]), encoding: .utf8)!)
'''

APP_DEFAULT = dict(corrPower=1, sigma_dp=0, sigma_cc=2, subpixel=1, upsample_factor=16, minAbsoluteIntensity=0,
                   minRelativeIntensity=0.005, relativeToPeak=0, relativeReferenceMinimumRadiusPx=0, minPeakSpacing=8,
                   edgeBoundary=5, maxNumPeaks=70)
def app_opts(**kw):
    d = dict(APP_DEFAULT); d.update(kw); return d
app_runs = [  # (id, scene, pattern, options)
    ('app_s1p0_poly', 'S1', 0, app_opts()),
    ('app_s1p0_multicorr', 'S1', 0, app_opts(subpixel=2)),
    ('app_s1p1_multicorr', 'S1', 1, app_opts(subpixel=2)),
    ('app_s1p0_power0.5_sigmaDP1', 'S1', 0, app_opts(corrPower=0.5, sigma_dp=1.0)),
    ('app_s2p0_multicorr', 'S2', 0, app_opts(subpixel=2, minPeakSpacing=6, edgeBoundary=4)),
    ('app_s3p0_refradius0', 'S3', 0, app_opts(minRelativeIntensity=0.5)),
    ('app_s3p0_refradius10', 'S3', 0, app_opts(minRelativeIntensity=0.5, relativeReferenceMinimumRadiusPx=10)),
    ('app_s3p1_multicorr', 'S3', 1, app_opts(subpixel=2)),
    ('app_s3p2_nonfinite_fill', 'S3', 2, app_opts()),
    ('app_s3p2_nonfinite_fill_multicorr', 'S3', 2, app_opts(subpixel=2, corrPower=0.5)),
]
adapted_in = [[64, 64, -1.0], [64, 64, 4.0], [128, 96, 10.6], [256, 256, 5.03], [512, 512, 60.0], [100, 100, 40.0], [48, 48, 0.0]]
hint_names = list(hints_in)
hint_probe = {}
for n in hint_names:
    a = hints_in[n]
    if a.max() > 0:
        r, x0, y0 = get_probe_size(a)          # py4DSTEM: x0 is the row, y0 the column
        hint_probe[n] = [float(r), float(y0), float(x0)]   # the app's (r, x0 = column, y0 = row)

with tempfile.TemporaryDirectory() as t:
    t = Path(t)
    shas = build_app_oracle(t)
    scene_payload = []
    scene_index = {}
    for i, (name, s) in enumerate(scenes.items()):
        scene_index[name] = i
        pats = arrays[s['patterns']]
        entry = dict(rows=s['rows'], columns=s['columns'], kernel=arrays[s['kernel']].ravel().tolist(),
                     patterns=[np.nan_to_num(p, nan=0.0, posinf=0.0).ravel().tolist() for p in
                               (arrays['p3finite'] if name == 'S3' else pats)],
                     nonfinite=nonfinite if name == 'S3' else [])
        scene_payload.append(entry)
    payload = dict(scenes=scene_payload,
                   runs=[dict(scene=scene_index[s], pattern=p, options=o) for _, s, p, o in app_runs],
                   adapted=adapted_in,
                   hints=[dict(pattern=hints_in[n].ravel().tolist(), **({'probe': hint_probe[n]} if n in hint_probe else {})) for n in hint_names])
    (t / 'main.swift').write_text(RUNNER)
    (t / 'input.json').write_text(json.dumps(payload))
    subprocess.run(['swiftc', '-package-name', 'oracle', '-O', '-o', str(t / 'oracle'), str(t / 'main.swift'),
                    str(t / 'app_dd.swift'), str(t / 'mdc.swift'), str(t / 'fft2d.swift'), str(t / 'ring.swift')], check=True)
    app = json.loads(subprocess.check_output([str(t / 'oracle'), str(t / 'input.json')], text=True))

for (cid, scene, p, opt), peaks in zip(app_runs, app['runs']):
    a = np.array(peaks).reshape(-1, 3)
    put('e_' + cid, a)
    cases.append(dict(id=cid, scene=scene, pattern=p, reference='app', options=opt, expected='e_' + cid))
byid = {c['id']: c for c in cases}
# The variants must be visible: each app-only option changes the answer, or the case proves nothing.
e = lambda i: arrays['e_' + i]
for i in ['app_s3p0_refradius0','app_s3p0_refradius10','s3p0_poly']: print(i, np.round(e(i),2).tolist())
assert not (e('app_s3p0_refradius0').shape == e('app_s3p0_refradius10').shape and np.allclose(e('app_s3p0_refradius0'), e('app_s3p0_refradius10'))), 'reference radius changes nothing'
assert len(e('app_s3p2_nonfinite_fill')) > 0 and len(e('s3p2_nonfinite')) == 0
assert not np.all(np.diff(e('app_s3p1_multicorr')[:, 2]) <= 0), 'app multicorr order is not out of intensity order'
assert np.all(np.diff(e('s3p1_multicorr')[:, 2]) <= 0), 'py4DSTEM multicorr order is sorted by refined intensity'
hints = []
for n, h in zip(hint_names, app['hints']):
    hints.append(dict(name=n, pattern='h_' + n, expected=h, probe=hint_probe.get(n)))
print('hints (app):', dict(zip(hint_names, app['hints'])))
assert sum(1 for h in app['hints'] if h) >= 1 and any(not h for h in app['hints'])

# ---------------------------------------------------------------- write
out = ROOT / 'Tests/CuboBraggTests/Fixtures'; out.mkdir(exist_ok=True, parents=True)
blob = bytearray(); table = {}
for name, a in arrays.items():
    table[name] = dict(offset=len(blob) // 8, shape=list(a.shape)); blob += a.tobytes()
(out / 'bragg.f64').write_bytes(bytes(blob))
meta = dict(py4dstem=PIN, numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version(), app=APP, imagoKit=IMAGO,
            appSources=dict(zip(['DiskDetection(sliced)', 'MatrixDFTCorrelation', 'ProbeRingHint', 'FFT2D'], shas)),
            reorderRatio=float(ratio), nonfinite=nonfinite, arrays=table, scenes=scenes, cases=cases, hints=hints,
            adapted=[dict(input=i, expected=o) for i, o in zip(adapted_in, app['adapted'])])
(out / 'bragg.json').write_text(json.dumps(meta, indent=1) + '\n')
print('cases', len(cases), 'bytes', len(blob))

# Tests/CuboVirtualTests/GaussianEdgeTests.swift pins scipy's two edge modes on a 3 x 4 image (sigma 1.7, radius beyond both
# extents); its numbers are `scipy.ndimage.gaussian_filter(a, 1.7, mode=m).ravel().tolist()` for m in reflect, nearest with
# a = [[1,5,2,0],[7,3,9,4],[0,8,6,1]], run with the same pinned venv.
