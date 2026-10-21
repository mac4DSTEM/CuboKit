"""Pins for the scan-level strain API (CuboStrain `StrainMapping`): py4DSTEM's real BraggVectors.histogram and
StrainMap (choose_basis_vectors, fit_basis_vectors, get_strain) run on planted lattices and on peaks measured on the
public simulated Au dataset. Sibling of pins.py (which is untouched); writes only `map_*.f64` and `strain_mapping.json`
under Tests/CuboStrainTests/Fixtures. Run from the clone root:

  ../../tools/memcap.sh -- ../References/python/py4dstem/bin/python -I -B tools/strain/mapping_pins.py

Peaks are stored as py4DSTEM sees them (centred, calibrated vectors from `get_vectors(center=True)`), so the Swift side
starts from the same doubles. Rows are py4DSTEM x, columns y.
"""
import os
os.environ['MPLBACKEND'] = 'Agg'
import hashlib, json, platform, subprocess, warnings
from pathlib import Path
import numpy as np
import h5py
import py4DSTEM
from py4DSTEM.braggvectors import BraggVectors
from py4DSTEM.preprocess.utils import get_maxima_2D
from py4DSTEM.process.strain.strain import StrainMap
from py4DSTEM.process.strain.latticevectors import get_reference_g1g2

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
SUITE = next(p for p in ROOT.parents if (p/'References/CuboKit').is_dir())
REF = SUITE/'References/CuboKit'
PIN = 'f050d207851f8accd3aa550365baaf1f38230428'
assert subprocess.check_output(['git','-C',str(REF/'py4DSTEM-f050d207'),'rev-parse','HEAD'],text=True).strip() == PIN
assert py4DSTEM.__version__ == '0.14.19' and np.__version__ == '2.5.4' and platform.python_version() == '3.12.14'
for rel in ['process/strain/latticevectors.py','process/strain/strain.py','braggvectors/braggvector_methods.py','preprocess/utils.py']:
    assert (Path(py4DSTEM.__file__).parent/rel).read_bytes() == (REF/'py4DSTEM-f050d207/py4DSTEM'/rel).read_bytes()

OUT = ROOT/'Tests/CuboStrainTests/Fixtures'
OUT.mkdir(parents=True, exist_ok=True)
arrays, meta = {}, dict(py4dstem=PIN, numpy=np.__version__, python=platform.python_version(), datasets={}, runs={}, histograms={})

def build_bv(raw_lists, scan, det, origin):
    """raw_lists: scan[0]*scan[1] arrays of (qx, qy, intensity) in uncentred detector pixels."""
    bv = BraggVectors(scan, det)
    for n, pts in enumerate(raw_lists):
        i, j = divmod(n, scan[1])
        if len(pts):
            bv._v_uncal[i, j].add_data_by_field(tuple(np.ascontiguousarray(pts[:, c]) for c in range(3)))
    bv.calibration.set_origin(tuple(origin))
    bv.calibrate()
    return bv

def centred_peaks(bv, scan):
    counts, rows = [], []
    for i in range(scan[0]):
        for j in range(scan[1]):
            d = bv.get_vectors(i, j, center=True, ellipse=False, rotate=False, pixel=False).data
            counts.append(len(d))
            rows.append(np.column_stack([d['qx'], d['qy'], d['intensity']]))
    return np.array(counts, float), (np.concatenate(rows) if rows else np.zeros((0, 3)))

def add_dataset(name, bv, scan, det, origin, truth=None, extra=None):
    counts, peaks = centred_peaks(bv, scan)
    arrays[f'map_{name}_counts'] = counts
    arrays[f'map_{name}_peaks'] = peaks
    if truth is not None:
        arrays[f'map_{name}_truth'] = truth
    meta['datasets'][name] = dict(scan=list(scan), detector=list(det), origin=[float(origin[0]), float(origin[1])], **(extra or {}))

def planted(seed, scan, det, origin, g_ref, noise, drop, amp, thin):
    rng = np.random.default_rng(seed)
    R, C = scan
    exx = np.zeros(scan); eyy = np.zeros(scan); exy = np.zeros(scan); th = np.zeros(scan)
    for r in range(R):
        for c in range(C):
            if c >= 3:                      # columns 0..2 are exactly unstrained: the reference region
                exx[r, c] = amp * 0.0030 * (c - 2)
                eyy[r, c] = -amp * 0.0015 * (r + 1)
                exy[r, c] = amp * 0.0020 * (r - 3)
                th[r, c] = amp * 0.0010 * (r - 2)
    raw = []
    for r in range(R):
        for c in range(C):
            beta = np.array([[1 - exx[r, c], -exy[r, c] + th[r, c]], [-exy[r, c] - th[r, c], 1 - eyy[r, c]]])
            g = (beta @ g_ref.T).T                       # rows are the local g1, g2
            pts = []
            for h in range(-2, 3):
                for k in range(-2, 3):
                    if (h, k) != (0, 0) and rng.random() < drop:
                        continue
                    p = h * g[0] + k * g[1] + noise * rng.standard_normal(2) * (1 if (h, k) != (0, 0) else 0)
                    inten = 200 * np.exp(-0.3 * (h * h + k * k)) * (1 + 0.15 * h - 0.1 * k) * (1 + (0.1 * rng.random() if noise else 0))
                    q = p + origin
                    if 1 <= q[0] <= det[0] - 2 and 1 <= q[1] <= det[1] - 2:
                        pts.append((q[0], q[1], inten))
            pts = np.array(pts)
            if (r, c) in thin:                           # too few peaks to fit
                pts = pts[:3]
            raw.append(pts)
    return raw, np.stack([exx, eyy, exy, th])

def flatmaps(sm):
    return np.asarray(sm.g1g2_map.data, float)

def run_strainmap(name, dataset, bv, scan, choose, max_peak_spacing, mask, gvects, rotation, ref_kind):
    sm = StrainMap(bv)
    g0, g1, g2, bd = sm.choose_basis_vectors(returncalc=True, **choose)
    sm.fit_basis_vectors(mask=mask, max_peak_spacing=max_peak_spacing)
    sm.get_strain(gvects=gvects, coordinate_rotation=rotation)
    g = sm.g
    pre = f'map_{name}'
    arrays[f'{pre}_maxima'] = np.column_stack([g['x'], g['y'], g['intensity']])
    arrays[f'{pre}_basis'] = np.array([g0[0], g0[1], g1[0], g1[1], g2[0], g2[1]], float)
    arrays[f'{pre}_directions'] = np.column_stack([bd.data['qx'], bd.data['qy'], bd.data['g1_ind'], bd.data['g2_ind']]).astype(float)
    arrays[f'{pre}_fits'] = flatmaps(sm)                              # x0 y0 g1x g1y g2x g2y error mask
    arrays[f'{pre}_mask'] = sm.mask.astype(float)
    arrays[f'{pre}_strain'] = np.asarray(sm.data[:5], float)           # exx eyy exy theta mask
    if ref_kind == 'median':
        ref = get_reference_g1g2(sm.g1g2_map, sm.mask)
    elif ref_kind == 'roi':
        ref = get_reference_g1g2(sm.g1g2_map, np.logical_and(gvects, sm.mask))
    else:
        ref = (tuple(gvects[0]), tuple(gvects[1]))
    arrays[f'{pre}_reference'] = np.array([ref[0][0], ref[0][1], ref[1][0], ref[1][1]], float)
    meta['runs'][name] = dict(dataset=dataset, choose=choose, maxPeakSpacing=max_peak_spacing, rotation=rotation, reference=ref_kind,
                              scanMask=None if mask is None else 'map_%s_scanmask' % name)
    if mask is not None:
        arrays[f'map_{name}_scanmask'] = mask.astype(float)
    if ref_kind == 'roi':
        arrays[f'map_{name}_roi'] = np.asarray(gvects, float)
    return sm

# ---- D1: noise-free planted lattice, 7x8 scan, columns 0..2 unstrained -------------------------------------------------
scan1, det1, org1 = (7, 8), (48, 48), np.array([24.4, 23.6])
g_ref = np.array([[9.0, 1.5], [-1.2, 10.5]])
raw, truth = planted(11, scan1, det1, org1, g_ref, 0.0, 0.0, 1.0, set())
bv1 = build_bv(raw, scan1, det1, org1)
add_dataset('d1', bv1, scan1, det1, org1, truth, dict(refG1=g_ref[0].tolist(), refG2=g_ref[1].tolist()))
arrays['map_d1_hist'] = bv1.histogram(mode='cal').data
meta['histograms']['d1'] = dict(dataset='d1', sampling=1)
run_strainmap('r1', 'd1', bv1, scan1, dict(), 3.0, None, None, 0, 'median')
roi_ref = (tuple(g_ref[0]), tuple(g_ref[1]))
run_strainmap('r4', 'd1', bv1, scan1, dict(subpixel='poly', sigma=1, minSpacing=4, maxNumPeaks=12), 3.0, None, roi_ref, 0, 'supplied')

# ---- D2: noisy, dropped peaks, thin positions, scan mask, ROI, rotation, histogram variants ------------------------------
scan2, det2, org2 = (7, 8), (48, 48), np.array([24.35, 23.8])
raw, truth = planted(22, scan2, det2, org2, g_ref, 0.04, 0.12, 6.0, {(1, 5), (4, 4), (6, 7), (2, 0)})
bv2 = build_bv(raw, scan2, det2, org2)
add_dataset('d2', bv2, scan2, det2, org2, truth)
rng = np.random.default_rng(5)
bmask = rng.random(scan2) > 0.2
wnum = np.where(rng.random(scan2) > 0.15, rng.random(scan2) * 2.0, 0.001)
arrays['map_d2_scanmask'] = bmask.astype(float)
arrays['map_d2_weights'] = wnum
hist_cases = {'default': dict(), 's15': dict(sampling=1.5), 'bool': dict(weights=bmask), 'num': dict(weights=wnum)}
for key, kw in hist_cases.items():
    arrays[f'map_d2_hist_{key}'] = bv2.histogram(mode='cal', **kw).data
    meta['histograms'][f'd2_{key}'] = dict(dataset='d2', sampling=kw.get('sampling', 1), weights=key if key in ('bool', 'num') else None)
meta['weightsThreshold'] = 0.005
roi = np.zeros(scan2, bool); roi[:, :3] = True; roi[0, :] = False
run_strainmap('r2', 'd2', bv2, scan2, dict(subpixel='poly', sigma=1, minSpacing=4, maxNumPeaks=12), 2.5, None, roi, 20.0, 'roi')
smask = np.ones(scan2, bool); smask[3, 3] = False; smask[5, 6] = False
sm2 = run_strainmap('r3', 'd2', bv2, scan2, dict(subpixel='pixel', sigma=0, minSpacing=3, minRelativeIntensity=0.02, maxNumPeaks=14), 2.5, smask, None, -35.0, 'median')
# explicit indices (g1 given, origin and g2 guessed) and the default multicorr
run_strainmap('r5', 'd2', bv2, scan2, dict(index_g1=1, sigma=1, minSpacing=2, maxNumPeaks=10), 2.5, None, None, 0, 'median')

# ---- D3: peaks measured on the public simulated Au dataset (py4DSTEM test data Au_sim.h5, simulation_4DSTEM) --------------
au = REF/'py4dstem-test-data/Au_sim.h5'
f = h5py.File(au, 'r')
cube = f['4DSTEM_experiment/data/datacubes/simulation_4DSTEM/data']
r0, c0, scan3, det3 = 48, 38, (8, 8), (125, 125)
raw, beams = [], []
for i in range(scan3[0]):
    for j in range(scan3[1]):
        p = np.asarray(cube[r0 + i, c0 + j], float)
        m = get_maxima_2D(p, subpixel='poly', sigma=1, minSpacing=4, edgeBoundary=3, maxNumPeaks=40, minRelativeIntensity=0.02)
        raw.append(np.column_stack([m['x'], m['y'], m['intensity']])); beams.append(raw[-1][0, :2])
org3 = np.mean(beams, axis=0)
bv3 = build_bv(raw, scan3, det3, org3)
add_dataset('d3', bv3, scan3, det3, org3, None, dict(file='Au_sim.h5', sha256=hashlib.sha256(au.read_bytes()).hexdigest(),
            selection=f'simulation_4DSTEM[{r0}:{r0+scan3[0]},{c0}:{c0+scan3[1]}], get_maxima_2D(poly, sigma 1, minSpacing 4, edgeBoundary 3, maxNumPeaks 40, minRelativeIntensity 0.02)'))
run_strainmap('r6', 'd3', bv3, scan3, dict(subpixel='poly', sigma=1, minSpacing=6, maxNumPeaks=8), 4.0, None, None, 0, 'median')

# ---- ties: py4DSTEM's order of equal maxima (np.sort(order='intensity')[::-1]) on an array with exact ties ---------------
tie = np.zeros((14, 14))
for (r, c, v) in [(3, 3, 5), (3, 9, 5), (9, 3, 5), (9, 9, 5), (6, 6, 5), (11, 6, 3), (6, 11, 3)]:
    tie[r, c] = v
mt = get_maxima_2D(tie, subpixel='pixel', sigma=0, minSpacing=0, maxNumPeaks=10)
arrays['map_ties_ar'] = tie
arrays['map_ties_maxima'] = np.column_stack([mt['x'], mt['y'], mt['intensity']])

# ---- write -----------------------------------------------------------------------------------------------------------
meta['arrays'] = {}
total = 0
for name, a in arrays.items():
    a = np.asarray(a, dtype='<f8')
    (OUT/(name + '.f64')).write_bytes(a.tobytes()); meta['arrays'][name] = list(a.shape); total += a.nbytes
(OUT/'strain_mapping.json').write_text(json.dumps(meta, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)) + '\n')
print('Wrote', total, 'bytes in', len(arrays), 'arrays')
for r, m in meta['runs'].items():
    print(r, 'valid', int(arrays[f'map_{r}_mask'].sum()), 'maxima', arrays[f'map_{r}_maxima'].shape[0])
