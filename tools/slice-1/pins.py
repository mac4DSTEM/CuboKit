"""CuboKit slice 1: the pin generator (registration docs/registration-slice-1.md, sections 2, 3, 4, 5).

Run with the pinned oracle, from any directory:

    References/py4dstem-venv/bin/python -I -B tools/slice-1/pins.py synthetic
    References/py4dstem-venv/bin/python -I -B tools/slice-1/pins.py real [name ...]

`synthetic` writes Tests/CuboVirtualTests/Fixtures/<cube>.npy and slice-1-pins.json (committed) BEFORE any Swift exists.
`real` exports the owner's cubes (section 5.4) and py4DSTEM's test datacube (5.3) as .npy into the real-data
directory, $CUBOKIT_REAL_DATA (default ~/Library/Caches/CuboKit/real-data), outside the repository, and writes beside
each <name>-pins.json plus its float64 arrays as <name>-<quantity>.npy.

py4DSTEM f050d207 (0.14.19) is the oracle: DataCube.get_virtual_image, make_detector, get_dp_mean, get_dp_max, called
as a user calls them (verbose=False). Beside each py4DSTEM number the generator records the float64 evaluation of the
same definition (section 2); the kit is held to the float64 number and the distance between the two is recorded.

File:line pins into References/py4DSTEM-f050d207/py4DSTEM/datacube/:
  virtualimage.py       get_virtual_image 67 (np.sum(self.data[rx, ry] * mask)), make_detector 588
  virtualdiffraction.py get_virtual_diffraction 63 (np.mean / np.max over axes (0, 1)), get_dp_max 308, get_dp_mean 337
  datacube.py           DataCube.__init__ 33 (dims Rx, Ry, Qx, Qy), R_Nx/Q_Nx 172-194
Index map (registration 1.1): py4DSTEM's qx (first detector axis) is the kit's ROW, qy its COLUMN; geometry tuples
pass through unchanged: circle ((row, column), r), annulus ((row, column), (r_in, r_out)), rectangle (rowMin, rowMax,
columnMin, columnMax), point (row, column).
"""
import datetime
import hashlib
import inspect
import json
import os
import platform
import sys
import warnings
from importlib.metadata import version as dist_version

warnings.filterwarnings("ignore")

import numpy as np
import h5py
import py4DSTEM
from py4DSTEM.datacube import DataCube

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIXTURES = os.path.join(KIT, "Tests", "CuboVirtualTests", "Fixtures")
REAL = os.environ.get("CUBOKIT_REAL_DATA", os.path.expanduser("~/Library/Caches/CuboKit/real-data"))
MACEM_DATA = os.environ.get("MACEM_DATA", os.path.abspath(os.path.join(KIT, "..", "data")))
SEED = 20261014
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append(name)
    assert ok, f"SELF-CHECK FAILED: {name} {detail}"


def g17(x):
    """Every number as a float64 string with 17 significant digits (registration 4); parsed by Double(String)."""
    return "%.17g" % float(x)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def versions():
    return {
        "py4dstem": py4DSTEM.__version__,
        "py4dstem_commit": "f050d207851f8accd3aa550365baaf1f38230428",
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "emdfile": dist_version("emdfile"),
        "python": platform.python_version(),
    }


# ----------------------------------------------------------------------------------------------------------------
# The detector (registration 3.1), written from the text, standard library only, to cross-check make_detector
# ----------------------------------------------------------------------------------------------------------------
def mask_by_definition(shape, g):
    qy, qx = shape
    m = np.zeros(shape, dtype=np.float64)
    kind = g["kind"]
    for i in range(qy):
        for j in range(qx):
            if kind == "circle":
                d2 = (i - g["centerRow"]) * (i - g["centerRow"]) + (j - g["centerColumn"]) * (j - g["centerColumn"])
                v = d2 < g["radius"] * g["radius"]
            elif kind == "annulus":
                d2 = (i - g["centerRow"]) * (i - g["centerRow"]) + (j - g["centerColumn"]) * (j - g["centerColumn"])
                v = d2 > g["inner"] * g["inner"] and d2 < g["outer"] * g["outer"]
            elif kind == "rectangle":
                r0, r1 = round(g["rowMin"]), round(g["rowMax"])          # Python's round: half to even, as np.round
                c0, c1 = round(g["columnMin"]), round(g["columnMax"])
                v = r0 <= i < r1 and c0 <= j < c1
            elif kind == "point":
                v = i == g["row"] and j == g["column"]
            elif kind == "mask":
                v = g["weights"][i * qx + j]
            else:
                raise ValueError(kind)
            m[i, j] = float(v)
    return m


def py4dstem_call(g):
    """The (mode, geometry) a py4DSTEM user passes for the kit's geometry (identity map, see the module docstring)."""
    k = g["kind"]
    if k == "circle":
        return "circle", ((g["centerRow"], g["centerColumn"]), g["radius"])
    if k == "annulus":
        return "annulus", ((g["centerRow"], g["centerColumn"]), (g["inner"], g["outer"]))
    if k == "rectangle":
        return "rectangle", (g["rowMin"], g["rowMax"], g["columnMin"], g["columnMax"])
    if k == "point":
        return "point", (g["row"], g["column"])
    if k == "mask":
        return "mask", None
    raise ValueError(k)


def py4dstem_mask(shape, g):
    mode, geo = py4dstem_call(g)
    if mode == "mask":
        geo = np.asarray(g["weights"], dtype=np.float64).reshape(shape)
    return DataCube.make_detector(shape, mode, geo), mode, geo


def float32_predicate_mask(shape, g):
    """mac4DSTEM's Float32 evaluation (VirtualDetector.fillRadial at 7448401b), for the boundary search of 5.1 (g)."""
    qy, qx = shape
    f = np.float32
    rows = np.arange(qy, dtype=np.float32)[:, None]
    cols = np.arange(qx, dtype=np.float32)[None, :]
    dy = rows - f(g["centerRow"])
    dx = cols - f(g["centerColumn"])
    r2 = dx * dx + dy * dy
    if g["kind"] == "circle":
        return r2 < f(g["radius"]) * f(g["radius"])
    return (r2 > f(g["inner"]) * f(g["inner"])) & (r2 < f(g["outer"]) * f(g["outer"]))


# ----------------------------------------------------------------------------------------------------------------
# What py4DSTEM computes, and the float64 evaluation of the same definition
# ----------------------------------------------------------------------------------------------------------------
def virtual_image_both(data, g):
    """(py4DSTEM's get_virtual_image on `data` as given, the float64 evaluation on `data`'s values, mask)."""
    shape = data.shape[2:]
    mask, mode, geo = py4dstem_mask(shape, g)
    dc = DataCube(data=data)
    own = dc.get_virtual_image(mode=mode, geometry=geo, verbose=False).data
    m64 = np.asarray(mask, dtype=np.float64).reshape(-1)
    f64 = np.empty(data.shape[:2], dtype=np.float64)
    for r in range(data.shape[0]):   # row by row: a float64 copy of the whole cube is never held
        f64[r] = data[r].reshape(data.shape[1], -1).astype(np.float64) @ m64
    return np.asarray(own, dtype=np.float64), f64, mask


def mean_max_both(data):
    dc = DataCube(data=data)
    own_mean = np.asarray(dc.get_dp_mean().data, dtype=np.float64)
    own_max = np.asarray(dc.get_dp_max().data, dtype=np.float64)
    f64_mean = np.mean(data, axis=(0, 1), dtype=np.float64)
    return own_mean, f64_mean, own_max


def rel_distance(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    floor = 1e-6 * max(np.max(np.abs(b)), 1e-300)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), floor)))


# ----------------------------------------------------------------------------------------------------------------
# Section 5.1: the synthetic cubes
# ----------------------------------------------------------------------------------------------------------------
def synthetic_cubes():
    rng = np.random.default_rng(SEED)
    cubes = {}
    ry, rx, qy, qx = 2, 3, 5, 7
    s = np.arange(ry * rx).reshape(ry, rx, 1, 1)
    i = np.arange(qy).reshape(1, 1, qy, 1)
    j = np.arange(qx).reshape(1, 1, 1, qx)
    cubes["digits"] = (s * 10000 + i * 100 + j).astype(np.float32)

    ry, rx, qy, qx = 6, 5, 16, 16
    bg = rng.normal(1000.0, 300.0, size=(ry, rx, qy, qx))
    ii, jj = np.indices((qy, qx))
    disk = ((ii - 7.5) ** 2 + (jj - 8.5) ** 2 < 16.0).astype(np.float64) * 50000.0
    g = bg + disk[None, None] * rng.uniform(0.8, 1.2, size=(ry, rx, 1, 1)) - 1500.0
    cubes["gaussian"] = g.astype(np.float32)

    cubes["counts16"] = rng.poisson(3000, size=(4, 6, 12, 12)).astype(np.uint16)
    cubes["counts32"] = rng.integers(0, 60001, size=(3, 4, 9, 9)).astype(np.uint32)
    cubes["thirds"] = (rng.integers(-300, 301, size=(2, 2, 4, 4)) / 3.0).astype(np.float64)
    cubes["negative"] = -rng.uniform(1.0, 100.0, size=(2, 2, 3, 3)).astype(np.float32)
    cubes["boundary"] = np.ones((1, 1, 32, 32), dtype=np.float32)
    return cubes


def find_boundary_geometry():
    """5.1 (g): a circle on 32 x 32 where the Float32 and the Double predicates disagree on at least one pixel."""
    rng = np.random.default_rng(SEED + 7)
    for _ in range(200000):
        g = {"kind": "circle", "centerRow": float(rng.integers(4, 28)) + float(rng.choice([0.1, 0.2, 0.3, 0.7, 0.9])),
             "centerColumn": float(rng.integers(4, 28)) + float(rng.choice([0.1, 0.3, 0.6, 0.7])),
             "radius": float(rng.uniform(2.0, 12.0))}
        # Put the radius exactly on a pixel distance computed in float64, then nudge it by less than a Float32 ulp.
        di, dj = int(rng.integers(-8, 9)), int(rng.integers(-8, 9))
        pi, pj = int(round(g["centerRow"])) + di, int(round(g["centerColumn"])) + dj
        if not (0 <= pi < 32 and 0 <= pj < 32):
            continue
        d = np.sqrt((pi - g["centerRow"]) ** 2 + (pj - g["centerColumn"]) ** 2)
        if d < 1.5:
            continue
        g["radius"] = float(np.nextafter(d, np.inf))
        m64 = mask_by_definition((32, 32), g).astype(bool)
        m32 = float32_predicate_mask((32, 32), g)
        if np.any(m64 != m32):
            return g, int(np.sum(m64 != m32))
    raise SystemExit("5.1 (g): no geometry found where the Float32 and Double predicates disagree")


def synthetic_geometries(shape, boundary=None):
    qy, qx = shape
    m = min(qy, qx)
    ci, cj = qy // 2, qx // 2
    geos = [
        ("bf_centre", {"kind": "circle", "centerRow": (qy - 1) / 2, "centerColumn": (qx - 1) / 2, "radius": 0.3 * m}),
        ("circle_off_centre", {"kind": "circle", "centerRow": 2.3, "centerColumn": 4.7, "radius": 2.9}),
        ("annulus_fractional", {"kind": "annulus", "centerRow": 1.5, "centerColumn": 5.25, "inner": 1.4, "outer": 3.6}),
        ("circle_r2_integer_centre", {"kind": "circle", "centerRow": float(ci), "centerColumn": float(cj), "radius": 2.0}),
        ("circle_r3_integer_centre", {"kind": "circle", "centerRow": float(ci), "centerColumn": float(cj), "radius": 3.0}),
        ("annulus_1_2p5_integer_centre", {"kind": "annulus", "centerRow": float(ci), "centerColumn": float(cj),
                                          "inner": 1.0, "outer": 2.5}),
        ("point_origin", {"kind": "point", "row": 0, "column": 0}),
        ("point_last", {"kind": "point", "row": qy - 1, "column": qx - 1}),
        ("rectangle_half_integer", {"kind": "rectangle", "rowMin": 0.5, "rowMax": 2.5, "columnMin": 1.5, "columnMax": 3.5}),
    ]
    pattern = [0.0, 0.5, 1.0, 2.0, -1.0]
    geos.append(("mask_weighted", {"kind": "mask", "weights": [pattern[(k * 7 + k // qx) % 5] for k in range(qy * qx)]}))
    if boundary is not None:
        geos.append(("boundary_float32_disagrees", boundary))
    return geos


def run_synthetic():
    os.makedirs(FIXTURES, exist_ok=True)
    cubes = synthetic_cubes()
    boundary, disagreeing = find_boundary_geometry()
    out = {"versions": versions(), "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "seed": SEED, "boundary_pixels_float32_disagrees": disagreeing, "cubes": []}
    for name, cube in cubes.items():
        path = os.path.join(FIXTURES, name + ".npy")
        np.save(path, cube)
        back = np.load(path)
        check(f"{name}: npy round trip", back.dtype == cube.dtype and np.array_equal(back, cube))
        cube32 = cube.astype(np.float32)
        exact = bool(np.array_equal(cube32.astype(np.float64), cube.astype(np.float64)))
        own_mean32, f64_mean, own_max32 = mean_max_both(cube32)
        own_mean_native, _, own_max_native = mean_max_both(cube)
        entry = {
            "name": name, "file": name + ".npy", "sha256": sha256_file(path), "shape": list(cube.shape),
            "dtype": str(cube.dtype), "exact_in_float32": exact,
            "mean_f64": [g17(v) for v in f64_mean.reshape(-1)],
            "mean_py4dstem_float32_cube": [g17(v) for v in own_mean32.reshape(-1)],
            "mean_py4dstem_native": [g17(v) for v in own_mean_native.reshape(-1)],
            "max": [g17(v) for v in own_max32.reshape(-1)],
            "mean_distance_py4dstem_own_vs_f64": rel_distance(own_mean32, f64_mean),
            "geometries": [],
        }
        # 3.3 truth on the max: exact against the native max for every cube converted exactly
        if exact:
            check(f"{name}: max float32 = native", np.array_equal(own_max32, own_max_native))
        shape = cube.shape[2:]
        geos = synthetic_geometries(shape, boundary if name == "boundary" else None)
        for gname, g in geos:
            mask_py, _, _ = py4dstem_mask(shape, g)
            mask_def = mask_by_definition(shape, g)
            check(f"{name}/{gname}: make_detector = registration 3.1",
                  np.array_equal(np.asarray(mask_py, dtype=np.float64), mask_def))
            own32, f64, _ = virtual_image_both(cube32, g)
            own_native, _, _ = virtual_image_both(cube, g)
            flat = np.asarray(mask_py, dtype=np.float64).reshape(-1)
            entry["geometries"].append({
                "name": gname, "geometry": {k: (v if k != "weights" else [g17(w) for w in v]) for k, v in g.items()},
                "mask_count": int(np.count_nonzero(flat)),
                "mask_pixels": [int(k) for k in np.flatnonzero(flat)],
                "image_f64": [g17(v) for v in f64.reshape(-1)],
                "image_py4dstem_float32_cube": [g17(v) for v in own32.reshape(-1)],
                "image_py4dstem_native": [g17(v) for v in own_native.reshape(-1)],
                "image_distance_py4dstem_own_vs_f64": rel_distance(own32, f64),
            })
        out["cubes"].append(entry)

    # 5.5 planted truth, checked here against py4DSTEM so a wrong closed form fails before any Swift exists
    d = cubes["digits"]; N = 6
    i = np.arange(5)[:, None]; j = np.arange(7)[None, :]
    check("digits: mean closed form", np.array_equal(np.mean(d, axis=(0, 1), dtype=np.float64),
                                                     ((N - 1) / 2) * 10000 + i * 100 + j))
    check("digits: max closed form", np.array_equal(np.max(d, axis=(0, 1)).astype(np.float64),
                                                    (N - 1) * 10000.0 + i * 100 + j))
    for r, want in ((2.0, 9), (3.0, 25)):
        m = mask_by_definition((16, 16), {"kind": "circle", "centerRow": 8.0, "centerColumn": 8.0, "radius": r})
        check(f"circle r={r} holds {want} pixels", int(m.sum()) == want)
    m = mask_by_definition((16, 16), {"kind": "annulus", "centerRow": 8.0, "centerColumn": 8.0, "inner": 1.0, "outer": 2.5})
    check("annulus (1, 2.5) holds 16 pixels", int(m.sum()) == 16)
    m = mask_by_definition((5, 7), {"kind": "rectangle", "rowMin": 0.5, "rowMax": 2.5, "columnMin": 1.5, "columnMax": 3.5})
    check("rectangle half to even: rows 0..<2, columns 2..<4",
          set(map(tuple, np.argwhere(m))) == {(0, 2), (0, 3), (1, 2), (1, 3)})
    check("boundary cube: the Float32 predicate disagrees", disagreeing >= 1)

    pins = os.path.join(FIXTURES, "slice-1-pins.json")
    with open(pins, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"pins.py synthetic: {len(out['cubes'])} cubes, {sum(len(c['geometries']) for c in out['cubes'])} geometries,"
          f" {len(CHECKS)} self-checks passed; boundary geometry {boundary} ({disagreeing} pixel(s) differ in Float32)")
    for c in out["cubes"]:
        worst = max(gm["image_distance_py4dstem_own_vs_f64"] for gm in c["geometries"])
        print(f"  {c['name']:9s} {c['dtype']:8s} {c['shape']} exact_in_float32={c['exact_in_float32']}"
              f"  py4DSTEM own vs f64: mean {c['mean_distance_py4dstem_own_vs_f64']:.3g}, images {worst:.3g}")


# ----------------------------------------------------------------------------------------------------------------
# Sections 5.3 and 5.4: real data
# ----------------------------------------------------------------------------------------------------------------
REAL_CUBES = {
    "almgsi": ("4dstem/training_dataset/Al_Mg_Si_060_STEM SI_preprocessed_unfiltered_bin_4_20260712.h5",
               "dm_dataset_root/dm_dataset/data"),
    "auref": ("4dstem/training_dataset/Au_ref_ROI15_preprocessed_unfiltered_bin_4_20241214.h5",
              "dm_dataset_root/dm_dataset/data"),
    "simau": ("4dstem/training_dataset/sim_Au_data_all_binned.h5", "4DSTEM_simulation/4DSTEM_polyAu/data"),
    "particle": ("4dstem/training_dataset/Particle_1_Stack_1_45x90_ss30nm_0p09s_spot8_alpha=0p48_bin2_cl-600mm_300kV_bin8.h5",
                 "datacube_root/datacube/data"),
    "py4dstem_small": (None, None),   # py4DSTEM's test datacube, read with its own import_file
}


def load_real(name):
    rel, dpath = REAL_CUBES[name]
    if rel is None:
        path = os.path.join(KIT, "References", "py4dstem-test-data", "small_datacube.dm4")
        dc = py4DSTEM.import_file(path)
        return np.ascontiguousarray(dc.data), path, "py4DSTEM.import_file"
    path = os.path.join(MACEM_DATA, rel)
    with h5py.File(path, "r") as h:
        data = h[dpath][...]
    return np.ascontiguousarray(data), path, dpath


def real_geometries(mean):
    qy, qx = mean.shape
    m = min(qy, qx)
    ci, cj = np.unravel_index(int(np.argmax(mean)), mean.shape)
    ii, jj = np.indices(mean.shape)
    off = np.where((ii - ci) ** 2 + (jj - cj) ** 2 >= (0.2 * m) ** 2, mean, -np.inf)
    di, dj = np.unravel_index(int(np.argmax(off)), mean.shape)
    return [
        ("bf", {"kind": "circle", "centerRow": float(ci), "centerColumn": float(cj), "radius": 0.1 * m}),
        ("adf", {"kind": "annulus", "centerRow": float(ci), "centerColumn": float(cj), "inner": 0.28 * m, "outer": 0.47 * m}),
        ("df_strongest_offaxis", {"kind": "circle", "centerRow": float(di), "centerColumn": float(dj), "radius": 3.0}),
        ("circle_r2_centre", {"kind": "circle", "centerRow": float(ci), "centerColumn": float(cj), "radius": 2.0}),
    ]


def run_real(names):
    os.makedirs(REAL, exist_ok=True)
    for name in names:
        t0 = datetime.datetime.now()
        data, source, dpath = load_real(name)
        npy = os.path.join(REAL, name + ".npy")
        np.save(npy, data)
        cube32 = data.astype(np.float32)
        exact = bool(np.array_equal(cube32, data)) if data.dtype.kind in "ui" else data.dtype == np.float32
        if data.dtype.kind in "ui":
            exact = bool(int(data.max()) < 2 ** 24)
        own_mean32, f64_mean, own_max32 = mean_max_both(cube32)
        if data.dtype != np.float32:
            own_mean_native, _, _ = mean_max_both(data)
        else:
            own_mean_native = own_mean32
        arrays = {"mean_f64": f64_mean, "mean_py4dstem_float32_cube": own_mean32,
                  "mean_py4dstem_native": own_mean_native, "max": own_max32}
        geos = real_geometries(f64_mean)
        meta = {"name": name, "source": os.path.relpath(source, MACEM_DATA) if source.startswith(MACEM_DATA) else
                os.path.basename(source), "dataset_path": dpath, "npy": name + ".npy", "npy_sha256": sha256_file(npy),
                "shape": list(data.shape), "dtype": str(data.dtype), "exact_in_float32": exact,
                "versions": versions(), "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "mean_distance_py4dstem_own_vs_f64": rel_distance(own_mean32, f64_mean), "geometries": []}
        for gname, g in geos:
            own32, f64, mask = virtual_image_both(cube32, g)
            if data.dtype != np.float32:
                own_native, _, _ = virtual_image_both(data, g)
            else:
                own_native = own32
            check(f"{name}/{gname}: make_detector = registration 3.1",
                  np.array_equal(np.asarray(mask, dtype=np.float64), mask_by_definition(mask.shape, g)))
            arrays[f"{gname}-image_f64"] = f64
            arrays[f"{gname}-image_py4dstem_float32_cube"] = own32
            arrays[f"{gname}-image_py4dstem_native"] = own_native
            meta["geometries"].append({"name": gname, "geometry": g, "mask_count": int(np.count_nonzero(mask)),
                                       "image_distance_py4dstem_own_vs_f64": rel_distance(own32, f64)})
        for key, arr in arrays.items():
            np.save(os.path.join(REAL, f"{name}-{key}.npy"), np.ascontiguousarray(arr, dtype=np.float64))
        with open(os.path.join(REAL, name + "-pins.json"), "w") as f:
            json.dump(meta, f, indent=1)
            f.write("\n")
        dt = (datetime.datetime.now() - t0).total_seconds()
        print(f"pins.py real {name}: {data.dtype} {data.shape} exact_in_float32={exact} "
              f"own-vs-f64 mean {meta['mean_distance_py4dstem_own_vs_f64']:.3g}, images "
              + ", ".join(f"{gm['name']} {gm['image_distance_py4dstem_own_vs_f64']:.3g} ({gm['mask_count']} px)"
                          for gm in meta["geometries"]) + f"; {dt:.0f} s; sha256 {meta['npy_sha256'][:12]}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("synthetic", "real"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "synthetic":
        run_synthetic()
    else:
        run_real(sys.argv[2:] or list(REAL_CUBES))
