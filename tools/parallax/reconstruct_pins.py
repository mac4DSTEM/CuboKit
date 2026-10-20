"""Pinned Parallax.reconstruct oracle (alignment schedule, regularised shifts, running average) and one end-to-end
subpixel case. Real py4DSTEM `Parallax(...).preprocess().reconstruct()` on small synthetic bright-field cubes.
From the clone root:
  ../../tools/memcap.sh -- References/py4dstem-venv/bin/python -I -B tools/parallax/reconstruct_pins.py
Bright-field disks are small (at most 16 groups per bin): NumPy sorts up to 16 elements with a stable insertion sort.
Planted data: one smooth random object seen through every bright-field pixel with a linear shift of the pixel offset
(a parallax of defocus type), so correlation maxima are unique (no float32 ties). For every bin the script asserts
that NumPy's default argsort of the radial order equals a stable sort (the Swift port is stable).
"""
import json, os, warnings
from pathlib import Path
import runpy
common = runpy.run_path(str(Path(__file__).with_name('preprocess_pins.py')))
ROOT, VERSIONS, np = common['ROOT'], common['VERSIONS'], common['np']
import py4DSTEM
warnings.simplefilter('ignore')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.build/mplconfig'))
SCAN_PX = 1.25
rng = np.random.default_rng(20261020)

def reconstruct(p, **kw):
    """py4DSTEM 0.14.19 `reconstruct` ends each bin with float(<1-element array>), which NumPy 2.5.0 refuses (TypeError).
    The only change here: np.atleast_1d returns the 0-d input unchanged for the duration of the call; no arithmetic differs."""
    saved = np.atleast_1d
    np.atleast_1d = lambda a: np.asarray(a)
    try:
        return p.reconstruct(plot_aligned_bf=False, plot_convergence=False, progress_bar=False, **kw)
    finally:
        np.atleast_1d = saved

def sig(a):
    """Shortest decimal that round-trips float32 (9 significant digits)."""
    return [float('%.9g' % v) for v in np.asarray(a, dtype=np.float32).ravel()]

def make_cube(scan, det, centre, radius, matrix, seed):
    g = np.random.default_rng(seed)
    sr, sc = scan
    obj = g.normal(size=scan)
    # band-limit so Fourier shifts are smooth
    f = np.fft.fft2(obj)
    qr = np.fft.fftfreq(sr)[:, None]; qc = np.fft.fftfreq(sc)[None, :]
    f *= np.exp(-(qr ** 2 + qc ** 2) / (2 * 0.18 ** 2))
    obj = np.fft.ifft2(f).real
    obj /= np.abs(obj).max()
    cube = np.full(scan + det, 0.01)
    for a in range(det[0]):
        for b in range(det[1]):
            off = np.array([a - centre[0], b - centre[1]])
            if (off ** 2).sum() <= radius ** 2:
                s = matrix @ off
                sh = np.fft.ifft2(np.fft.fft2(obj) * np.exp(-2j * np.pi * (s[0] * qr + s[1] * qc))).real
                cube[:, :, a, b] = 1.0 + 0.4 * sh + 0.02 * g.normal(size=scan)
    return np.round(cube, 4).astype(np.float32)

def new(cube, padding, q_sampling):
    dc = py4DSTEM.DataCube(cube.copy())
    dc.calibration.set_R_pixel_size(SCAN_PX); dc.calibration.set_R_pixel_units('A')
    dc.calibration.set_Q_pixel_size(q_sampling); dc.calibration.set_Q_pixel_units('A^-1')
    return py4DSTEM.process.phase.Parallax(energy=200000, datacube=dc, object_padding_px=padding, verbose=False, device='cpu')

def schedule(p, o):
    inds = np.asarray(p._xy_inds)
    diameter = int(max(inds[:, 0].max() - inds[:, 0].min(), inds[:, 1].max() - inds[:, 1].min()) + 1)
    mx = diameter if o.get('max_alignment_bin') is None else min(diameter, o['max_alignment_bin'])
    if o.get('alignment_bin_values') is not None:
        return np.array(o['alignment_bin_values']).clip(1, mx).astype(float)
    mn = o.get('min_alignment_bin', 1)
    bmin = np.ceil(np.log(mn) / np.log(2)); bmax = np.ceil(np.log(mx) / np.log(2))
    b = 2 ** np.arange(bmin, bmax)[::-1]
    if o.get('num_iter_at_min_bin', 2) > 1:
        b = np.hstack((b, np.repeat(b[-1], o.get('num_iter_at_min_bin', 2) - 1)))
    return b

def check_ties(p, bins, centered):
    xy = np.asarray(p._xy_inds); c = (xy - np.median(xy, axis=0)).astype(float)
    for b in bins:
        v = np.unique(np.round(c / b + (0 if centered else .5)).astype(int), axis=0)
        r = (v ** 2).sum(1)
        assert (np.argsort(r) == np.argsort(r, kind='stable')).all(), 'argsort tie order differs from stable'

CASES = [
    # name, scan, det, centre, radius, matrix, padding, q, preprocess kw, reconstruct kw, full
    ('default', (8, 8), (8, 8), (3.5, 3.5), 2.4, [[1.1, .3], [-.15, .9]], (4, 4), .08, {}, {}, True),
    ('regularized-2x1', (8, 8), (8, 8), (3.5, 3.5), 2.4, [[1.1, .3], [-.15, .9]], (4, 4), .08, {}, dict(regularize_shifts=True, regularizer_matrix_size=(2, 1)), False),
    ('no-running-centered', (8, 8), (8, 8), (3.5, 3.5), 2.4, [[1.1, .3], [-.15, .9]], (4, 4), .08, {}, dict(running_average=False, centered_alignment_bins=True), True),
    ('explicit-bins-nonsquare', (6, 9), (8, 9), (3.5, 4.0), 2.2, [[.8, -.25], [.3, 1.05]], (4, 6), .08, {}, dict(alignment_bin_values=[3, 2, 1], cross_correlation_upsample_factor=16, regularize_shifts=True), False),
    ('capped-single-pass', (8, 8), (8, 8), (3.5, 3.5), 2.4, [[1.1, .3], [-.15, .9]], (4, 4), .08, {}, dict(max_alignment_bin=2, num_iter_at_min_bin=1, cross_correlation_upsample_factor=4), False),
]
out_cases = []
for name, scan, det, centre, radius, matrix, padding, q, pkw, rkw, full in CASES:
    cube = make_cube(scan, det, centre, radius, np.array(matrix), seed=int(rng.integers(1 << 30)))
    p = new(cube, padding, q).preprocess(threshold_intensity=.5, edge_blend=2., plot_average_bf=False, **pkw)
    bins = schedule(p, rkw)
    check_ties(p, bins, rkw.get('centered_alignment_bins', False))
    reconstruct(p, **rkw)
    assert len(p.error_iterations) == len(bins), (len(p.error_iterations), len(bins))
    st = np.asarray(p._stack_BF_shifted)
    case = dict(name=name, scan=list(scan), detector=list(det), padding=list(padding), reciprocalSampling=q, threshold=.5, edgeBlend=2.,
                cube=[round(float(v), 4) for v in cube.ravel()], options={k: (list(v) if isinstance(v, tuple) else v) for k, v in rkw.items()},
                bins=[float(b) for b in bins], nImages=int(st.shape[0]), indices=np.asarray(p._xy_inds).ravel().tolist(),
                shifts=sig(p._xy_shifts), image=sig(p._recon_BF), mask=sig(p._recon_mask), errors=[float(e) for e in p.error_iterations],
                stackNorms=[float(np.linalg.norm(s.astype(np.float64))) for s in st], maskSum=float(np.asarray(p._stack_mask, np.float64).sum()),
                stack=sig(st) if full else None)
    out_cases.append(case)
    print(name, 'images', st.shape[0], 'bins', bins.tolist(), 'max |shift|', float(np.abs(p._xy_shifts).max()), 'errors', [round(e, 5) for e in p.error_iterations])

# End to end through subpixel_alignment: wider Q sampling so the factor heuristic and the limits are in range.
cube = make_cube((8, 8), (8, 8), (3.5, 3.5), 2.4, np.array([[1.1, .3], [-.15, .9]]), seed=int(rng.integers(1 << 30)))
e2e = []
for q, sub in [(.2, dict()), (.2, dict(kde_upsample_factor=2.0, kde_sigma_px=.3, position_correction_num_iter=2, position_correction_checkerboard_steps=True))]:
    p = new(cube, (4, 4), q).preprocess(threshold_intensity=.5, edge_blend=2., plot_average_bf=False, apply_realspace_mask_to_stack=True)
    reconstruct(p)
    p.subpixel_alignment(plot_upsampled_BF_comparison=False, plot_position_correction_convergence=False, progress_bar=False, **sub)
    img = np.asarray(p._recon_BF_subpixel_aligned)
    e2e.append(dict(reciprocalSampling=q, scanSampling=SCAN_PX, subpixel={k: v for k, v in sub.items()}, shifts=sig(p._xy_shifts),
                    usedFactor=float(p._kde_upsample_factor), bfLimit=float(p._BF_upsample_limit), dfLimit=float(p._DF_upsample_limit),
                    shape=list(img.shape), image=sig(img),
                    dr=sig(p._probe_dx) if 'position_correction_num_iter' in sub else None,
                    dc=sig(p._probe_dy) if 'position_correction_num_iter' in sub else None))
    print('e2e', q, sub, 'factor', p._kde_upsample_factor, 'limits', p._BF_upsample_limit, p._DF_upsample_limit)
out = dict(versions=VERSIONS, cases=out_cases, endToEnd=dict(cube=[round(float(v), 4) for v in cube.ravel()], scan=[8, 8], detector=[8, 8], padding=[4, 4], cases=e2e))
dest = ROOT / 'Tests/CuboParallaxTests/Fixtures/reconstruct.json'
dest.write_text(json.dumps(out, separators=(',', ':')) + '\n')
print(dest, dest.stat().st_size)
