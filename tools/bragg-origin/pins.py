"""Pins for py4DSTEM's origin from the Bragg peaks: BraggVectors.measure_origin (every score method, findcenter "max" and
"CoM", a given centre guess) and BraggVectors.fit_origin (plane, parabola, constant, robust), on planted peaks with a
descan plane, Bragg reflections brighter than the direct beam at a few positions, an empty position and a position
whose chosen peak sits at qx = 0 (fit_origin's mask_check_data leaves it out). Everything is produced by py4DSTEM itself;
no CuboKit Swift is used. Run from the clone root with the environment of tools/requirements.txt (CONTRIBUTING.md):

    python -I -B tools/bragg-origin/pins.py

Output (floats as repr, so they round-trip exactly): Tests/CuboBraggTests/Fixtures/braggorigin.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pinenv  # noqa: E402

import py4DSTEM  # noqa: E402
from emdfile import PointListArray  # noqa: E402
from py4DSTEM.braggvectors import BraggVectors  # noqa: E402

assert py4DSTEM.__version__ == "0.14.19"
warnings.filterwarnings("ignore")

R = (7, 9)
Q = (48, 56)
rng = np.random.default_rng(20261105)
rr, cc = np.mgrid[0 : R[0], 0 : R[1]]
true_row = 22.0 + 0.30 * rr - 0.20 * cc
true_col = 30.0 - 0.15 * rr + 0.25 * cc
g1, g2 = np.array([9.5, 1.2]), np.array([-1.0, 11.0])

peaks = []
for r in range(R[0]):
    for c in range(R[1]):
        k = r * R[1] + c
        if k == 13:
            peaks.append([])                                        # an empty position
            continue
        o = np.array([true_row[r, c], true_col[r, c]]) + rng.normal(0, 0.15, 2)
        pts = [(o[0], o[1], 100.0 + rng.uniform(-5, 5))]
        for m, g in enumerate((g1, -g1, g2, -g2, g1 + g2)):
            p = o + g
            pts.append((p[0], p[1], float(rng.uniform(15, 60))))
        if k % 11 == 4:                                             # a reflection brighter than the direct beam
            pts[2] = (pts[2][0], pts[2][1], 180.0)
        if k == 40:                                                 # the chosen (brightest) peak at qx = 0 exactly
            pts = [(0.0, 31.5, 300.0)] + pts
        rng.shuffle(pts)
        peaks.append(pts)

dtype = [("qx", np.float64), ("qy", np.float64), ("intensity", np.float64)]
pla = PointListArray(dtype=dtype, shape=R)
for k, pts in enumerate(peaks):
    if len(pts):
        pla.get_pointlist(k // R[1], k % R[1]).add(np.array([tuple(p) for p in pts], dtype=dtype))


def flat(a):
    return [float(v) for v in np.asarray(a, dtype=np.float64).ravel()]


def fresh():
    bv = BraggVectors(R, Q)
    bv.set_raw_vectors(pla)
    return bv


# The guess of findcenter="max": argmax of gaussian_filter(raw histogram, 10). measure_origin passes the BraggVectorMap
# object itself to gaussian_filter, which returns a 0-d object array, so upstream's guess is always (0, 0) at f050d207;
# the kit computes the intended guess on the map's data (a DEVIATION, docs/parity.md). The distance-scored cases below pass
# that intended guess explicitly, so the pins hold py4DSTEM's arithmetic for what the kit computes.
from scipy.ndimage import gaussian_filter  # noqa: E402

bvm = fresh().histogram(mode="raw")
upstream_guess = np.unravel_index(np.argmax(gaussian_filter(bvm, 10)), Q)
assert tuple(int(v) for v in upstream_guess) == (0, 0), upstream_guess
guess = tuple(int(v) for v in np.unravel_index(np.argmax(gaussian_filter(np.asarray(bvm.data), 10)), Q))
found = np.array([[len(peaks[r * R[1] + c]) > 0 for c in range(R[1])] for r in range(R[0])])

measures = []
for label, kwargs, expected_guess in [
    ("default", {}, None),
    ("max-distance", dict(center_guess=guess, score_method="distance"), "max"),
    ("max-iwd", dict(center_guess=guess, score_method="intensity weighted distance"), "max"),
    ("guess-default", dict(center_guess=(24.0, 31.0)), None),
    ("guess-intensity", dict(center_guess=(24.0, 31.0), score_method="intensity"), None),
    ("guess-iwd", dict(center_guess=(24.0, 31.0), score_method="intensity weighted distance"), None),
]:
    bv = fresh()
    qx0, qy0, mask = bv.measure_origin(**kwargs)
    if "center_guess" not in kwargs:
        # upstream wrote its always-(0, 0) guess at the empty position; the kit writes the intended guess, so the pins
        # (and the robust fits below, which score every position) are py4DSTEM's arithmetic on that value
        qx0 = np.where(found, qx0, float(guess[0]))
        qy0 = np.where(found, qy0, float(guess[1]))
        bv.calibration.set_origin_meas((qx0, qy0))
    case = dict(label=label, row=flat(qx0), column=flat(qy0), maskScalar=bool(np.ndim(mask) == 0))
    # "max" cases are run by the kit with findCenter .maximum and no guess; the others with the guess given
    case["centerGuess"] = None if expected_guess == "max" or "center_guess" not in kwargs else [float(v) for v in kwargs["center_guess"]]
    case["score"] = kwargs.get("score_method")
    fits = []
    for fitlabel, fk in [
        ("plane", dict(fitfunction="plane")),
        ("parabola", dict(fitfunction="parabola")),
        ("plane-robust", dict(fitfunction="plane", robust=True, robust_steps=3, robust_thresh=2)),
    ]:
        if label not in ("default", "guess-default") and fitlabel != "plane":
            continue
        # mask = the positions that had peaks: the kit's fit leaves an empty position out (upstream would fit the guess it
        # wrote there, unless its row is 0, as the (0, 0) guess of the default path is)
        xf, yf, xr, yr = bv.fit_origin(plot=False, mask=found.copy(), **fk)
        fits.append(dict(label=fitlabel, fittedRow=flat(xf), fittedColumn=flat(yf), residualRow=flat(xr), residualColumn=flat(yr)))
    if label == "default":
        bv2 = fresh()
        bv2.measure_origin()
        xf, yf, _, _ = bv2.fit_origin(plot=False)
        assert np.allclose(xf, np.asarray(fits[0]["fittedRow"]).reshape(R), rtol=1e-12, atol=1e-12), "default fit without a mask differs"
    case["fits"] = fits
    measures.append(case)

# The mac4DSTEM symmetry guess (BraggOrigin.symmetryCenterGuess), transcribed here in NumPy/SciPy: not in py4DSTEM.
from scipy.ndimage import maximum_filter  # noqa: E402


def symmetry_guess(peak_lists, shape, candidates=None):
    counted = PointListArray(dtype=dtype, shape=(1, len(peak_lists)))
    for k, pts in enumerate(peak_lists):
        if len(pts):
            counted.get_pointlist(0, k).add(np.array([(p[0], p[1], 1.0) for p in pts], dtype=dtype))
    bvc = BraggVectors((1, len(peak_lists)), shape)
    bvc.set_raw_vectors(counted)
    m = gaussian_filter(np.asarray(bvc.histogram(mode="raw").data, dtype=np.float64), 1, mode="reflect")
    loc = (m == maximum_filter(m, size=5)) & (m > 0.05 * m.max())
    idx = np.argwhere(loc)
    order = np.argsort(-m[loc], kind="stable")[:candidates]  # None: every maximum
    rows, cols = shape
    best = None
    for k in order:
        r, c = (int(v) for v in idx[k])
        i0, i1 = max(0, 2 * r - (rows - 1)), min(rows - 1, 2 * r)
        j0, j1 = max(0, 2 * c - (cols - 1)), min(cols - 1, 2 * c)
        a = m[i0 : i1 + 1, j0 : j1 + 1]
        b = m[2 * r - i1 : 2 * r - i0 + 1, 2 * c - j1 : 2 * c - j0 + 1][::-1, ::-1]
        score = float(np.minimum(a, b).sum())
        if best is None or score > best[0]:
            best = (score, (r, c))
    return [float(v) for v in best[1]]


# A pattern whose first-order reflection g1 is brighter than the direct beam at every position (Si/SiGe-like), no descan:
# py4DSTEM's guess and its intensity score pick the reflection; the symmetry guess and a distance score pick the beam.
bright = []
beam = np.array([25.0, 27.0])
for k in range(R[0] * R[1]):
    o = beam + rng.normal(0, 0.1, 2)
    pts = [(o[0], o[1], 60.0 + rng.uniform(-5, 5))]
    for i in (-2, -1, 0, 1, 2):
        for j in (-2, -1, 0, 1, 2):
            if (i, j) == (0, 0) or abs(i) + abs(j) > 3:
                continue
            p = o + i * g1 + j * g2
            if 0 < p[0] < Q[0] - 1 and 0 < p[1] < Q[1] - 1:
                pts.append((p[0], p[1], 200.0 if (i, j) == (1, 0) else float(rng.uniform(5, 40))))
    rng.shuffle(pts)
    bright.append(pts)
bright_pla = PointListArray(dtype=dtype, shape=R)
for k, pts in enumerate(bright):
    bright_pla.get_pointlist(k // R[1], k % R[1]).add(np.array([tuple(p) for p in pts], dtype=dtype))
bvb = BraggVectors(R, Q)
bvb.set_raw_vectors(bright_pla)
sym = symmetry_guess(bright, Q)
assert np.hypot(sym[0] - beam[0], sym[1] - beam[1]) < 1, sym
bqx, bqy, _ = bvb.measure_origin(center_guess=tuple(sym))
symmetry = dict(
    peaks=[[list(map(float, p)) for p in pts] for pts in bright],
    guess=sym,
    row=flat(bqx),
    column=flat(bqy),
    planedGuess=symmetry_guess(peaks, Q),
)

out = dict(
    reference="py4DSTEM f050d207 / 0.14.19",
    symmetry=symmetry,
    scan=list(R),
    detector=list(Q),
    peaks=[[list(map(float, p)) for p in pts] for pts in peaks],
    histogram=flat(np.asarray(bvm.data)),
    maxGuess=list(guess),
    measures=measures,
)
dest = pinenv.fixtures("CuboBraggTests") / "braggorigin.json"
dest.write_text(json.dumps(out, separators=(",", ":"), allow_nan=True) + "\n")
print("fixture bytes", dest.stat().st_size, "guess", guess)
_bv = BraggVectors(R, Q)
_bv.set_raw_vectors(bright_pla)
_qx, _qy, _ = _bv.measure_origin()
assert np.all(np.hypot(_qx - (beam[0] + g1[0]), _qy - (beam[1] + g1[1])) < 1), "py4DSTEM's default should pick g1 here"
