"""py4DSTEM side of the CuboKit benchmark, and the merge into docs/benchmarks.md.
Run from the clone root with the environment of tools/requirements.txt; one benchmark at a time:
  python -I -B tools/benchmarks/bench.py make   <dir>    # writes <dir>/{small,mid,typical}.npy
  python -I -B tools/benchmarks/bench.py python <dir>    # writes <dir>/python.json
  CUBOKIT_BENCH_DIR=<dir> swift test -c release --filter BenchmarkHarness                            # writes <dir>/swift.json
  python -I -B tools/benchmarks/bench.py merge <dir>                       # prints the markdown table
Speed round (docs/benchmarks.md): `bench.py speed <dir> [calls]` prints one `SPEED name median_ms= best_ms=` line per case for the Python side (default 15 calls);
the Swift side is `CUBOKIT_SPEED_DIR=<dir> swift test -c release --filter SpeedBench` (Tests/CuboVirtualTests/SpeedBench.swift, same cases and line format, plus a
hash of each result's bits so two builds can be compared). Run the old sources, the new sources and the Python side in turn, three rounds, and take medians.
Sizes (scan rows x columns x detector rows x columns, Float32): small 32x32x32x32, mid 64x64x32x32, typical 100x100x64x64; the large size
(330x330x64x64) is the nightly review's. Best of five after one warm-up; the table reports best, median and spread.
"""
import json, sys, time, platform, subprocess
from pathlib import Path
import numpy as np

SIZES = {'small': (32, 32, 32, 32), 'mid': (64, 64, 32, 32), 'typical': (100, 100, 64, 64)}

def make(d):
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261021)
    for name, shape in SIZES.items():
        qr, qc = shape[2:]
        rr, cc = np.meshgrid(np.arange(qr), np.arange(qc), indexing='ij')
        blob = 300 * np.exp(-((rr - qr / 2) ** 2 + (cc - qc / 2) ** 2) / (qr * 1.5))
        a = (blob[None, None] * (1 + 0.2 * rng.standard_normal(shape[:2] + (1, 1))) + 10 * rng.standard_normal(shape) - 5).astype(np.float32)
        np.save(d / f'{name}.npy', a)

def timeit(f, runs=5):
    f()
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    ts.sort()
    return dict(best=ts[0], median=ts[len(ts) // 2], spread=ts[-1] - ts[0])

def python(d):
    import py4DSTEM
    from py4DSTEM import DataCube
    from py4DSTEM.preprocess.preprocess import (bin_data_diffraction, bin_data_real, thin_data_real, pad_data_diffraction,
                                                resample_data_diffraction, median_filter_masked_pixels)
    d = Path(d); out = dict(py4DSTEM=py4DSTEM.__version__, numpy=np.__version__, python=platform.python_version(), results={})
    for name in SIZES:
        data = np.load(d / f'{name}.npy')
        R = data.shape[:2]; Q = data.shape[2:]
        def fresh(): return DataCube(data=data.copy()) if name == 'small' else DataCube(data=data)
        dc = DataCube(data=data)
        rng = np.random.default_rng(3)
        mask = rng.random(R) > 0.2
        weights = np.where(mask, rng.random(R), 0.0)
        pmask = np.zeros(Q, bool); pmask[5, 7] = pmask[20, 30 % Q[1]] = True
        cases = {
            'dp_mean': lambda: dc.get_dp_mean(),
            'dp_max': lambda: dc.get_dp_max(),
            'dp_median': lambda: dc.get_dp_median(),
            'virtual_image_circle': lambda: dc.get_virtual_image(mode='circle', geometry=((Q[0] / 2, Q[1] / 2), Q[0] / 4), name='bench'),
            'selected_area_mean_mask': lambda: dc.get_virtual_diffraction(method='mean', mask=mask, name='bench'),
            'selected_area_mean_weights': lambda: dc.get_virtual_diffraction(method='mean', mask=weights, name='bench'),
            'bin_Q_2': lambda: bin_data_diffraction(fresh(), 2),
            'bin_R_2': lambda: bin_data_real(fresh(), 2),
            'thin_R_2': lambda: thin_data_real(fresh(), 2),
            'pad_Q_2': lambda: pad_data_diffraction(fresh(), pad_factor=2),
        }
        if name != 'typical':   # per-pattern scipy loops: the typical size takes minutes in Python
            cases['resample_Q_half'] = lambda: resample_data_diffraction(fresh(), resampling_factor=0.5, method='bilinear')
            cases['median_masked_pixels'] = lambda: median_filter_masked_pixels(fresh(), pmask, 3)
        for k, f in cases.items():
            out['results'][f'{k}|{name}'] = timeit(f)
            print(k, name, out['results'][f'{k}|{name}'], flush=True)
    (d / 'python.json').write_text(json.dumps(out, indent=1))

def speed(d, calls=15):
    """Alternating-rounds side of the speed round: one `SPEED name median_ms= best_ms=` line per case (the Swift SpeedBench prints the same)."""
    from py4DSTEM import DataCube
    from py4DSTEM.preprocess.preprocess import bin_data_real, median_filter_masked_pixels
    d = Path(d)
    def report(name, f):
        f(); ts = []
        for _ in range(int(calls)):
            t = time.perf_counter(); f(); ts.append((time.perf_counter() - t) * 1e3)
        ts.sort()
        print(f"SPEED {name} median_ms={ts[len(ts) // 2]:.2f} best_ms={ts[0]:.2f}", flush=True)
    for name in ('typical', 'mid'):
        data = np.load(d / f'{name}.npy')
        R = data.shape[:2]; Q = data.shape[2:]
        dc = DataCube(data=data)
        if name == 'typical':
            mask = np.random.default_rng(3).random(R) > 0.2
            report(f'dp_mean|{name}', lambda: dc.get_dp_mean())
            report(f'mask_mean|{name}', lambda: dc.get_virtual_diffraction(method='mean', mask=mask, name='bench'))
            weights = np.where(mask, np.random.default_rng(4).random(R), 0.0)
            report(f'weights_mean|{name}', lambda: dc.get_virtual_diffraction(method='mean', mask=weights, name='bench'))
            report(f'bin_R_2|{name}', lambda: bin_data_real(DataCube(data=data), 2))
        else:
            pmask = np.zeros(Q, bool); pmask[5, 7] = pmask[20, 30 % Q[1]] = True
            report(f'median_masked_pixels|{name}', lambda: median_filter_masked_pixels(DataCube(data=data), pmask, 3))
            report(f'bin_R_2|{name}', lambda: bin_data_real(DataCube(data=data), 2))

def compare(d, dumps):
    """Accuracy of the speed-round results: `CUBOKIT_SPEED_DUMP=<dumps> swift test ... --filter SpeedBench` writes the kit's results; this prints their
    largest difference from a float64 truth and from py4DSTEM/NumPy (float32 accumulation), relative to the largest value, for the same cube, mask and weights."""
    import py4DSTEM
    from py4DSTEM import DataCube
    from py4DSTEM.preprocess.preprocess import bin_data_real, median_filter_masked_pixels
    d = Path(d); dumps = Path(dumps)
    data = np.load(d / 'typical.npy'); dc = DataCube(data=data)
    R = data.shape[:2]
    mask = np.fromfile(dumps / 'mask.u8', np.uint8).astype(bool).reshape(R)
    weights = np.fromfile(dumps / 'weights.f64', np.float64).reshape(R)
    def rel(a, b): return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))) / np.max(np.abs(b)))
    d64 = data.astype(np.float64)
    cases = {
        'dp_mean': (np.fromfile(dumps / 'dp_mean.f64'), dc.get_dp_mean().data, d64.mean(axis=(0, 1))),
        'mask_mean': (np.fromfile(dumps / 'mask_mean.f64'), dc.get_virtual_diffraction(method='mean', mask=mask, name='b').data,
                      d64[mask].mean(axis=0)),
        'weights_mean': (np.fromfile(dumps / 'weights_mean.f64'), dc.get_virtual_diffraction(method='mean', mask=weights, name='b').data,
                         (d64 * weights[:, :, None, None]).sum(axis=(0, 1)) / weights.sum()),
    }
    for k, (kit, py, truth) in cases.items():
        kit = kit.reshape(truth.shape); py = np.asarray(py).reshape(truth.shape)
        print(f"{k}: kit vs float64 truth {rel(kit, truth):.2e}; py4DSTEM/NumPy float32 vs truth {rel(py, truth):.2e}; kit vs py4DSTEM {rel(kit, py):.2e}")
    ref = bin_data_real(DataCube(data=data), 2).data
    kit = np.fromfile(dumps / 'bin_R_2_typical.f32', np.float32).reshape(ref.shape)
    t64 = d64[:R[0] // 2 * 2, :R[1] // 2 * 2].reshape(R[0] // 2, 2, R[1] // 2, 2, *data.shape[2:]).sum(axis=(1, 3))
    print(f"bin_R_2: kit vs float64 truth {rel(kit, t64):.2e}; py4DSTEM float32 vs truth {rel(ref, t64):.2e}; kit vs py4DSTEM {rel(kit, ref):.2e}")
    mid = np.load(d / 'mid.npy'); Q = mid.shape[2:]
    pmask = np.zeros(Q, bool); pmask[5, 7] = pmask[20, 30 % Q[1]] = True
    ref = median_filter_masked_pixels(DataCube(data=mid.copy()), pmask, 3).data
    kit = np.fromfile(dumps / 'median_masked_pixels_mid.f32', np.float32).reshape(ref.shape)
    print(f"median_masked_pixels: kit == py4DSTEM bit for bit: {bool(np.array_equal(kit.view(np.uint32), np.asarray(ref, np.float32).view(np.uint32)))}")

def merge(d):
    d = Path(d)
    py = json.loads((d / 'python.json').read_text()); sw = json.loads((d / 'swift.json').read_text())
    machine = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    print(f"Machine: {machine}, macOS {platform.mac_ver()[0]}; {sw['toolchain']}; py4DSTEM {py['py4DSTEM']}, NumPy {py['numpy']}, Python {py['python']}.")
    print("Float32 cubes, best of five after one warm-up; cells are best / median (spread) in ms.\n")
    print('| function | size | kit CPU | kit Metal | py4DSTEM | CPU vs py4DSTEM |\n|---|---|---|---|---|---|')
    def cell(r): return '—' if r is None else f"{r['best']*1e3:.1f} / {r['median']*1e3:.1f} ({r['spread']*1e3:.1f})"
    names = sorted({k.split('|')[0] for k in list(py['results']) + list(sw['results'])})
    for n in names:
        for size in SIZES:
            key = f'{n}|{size}'
            c, m, p = sw['results'].get(f'{key}|cpu'), sw['results'].get(f'{key}|metal'), py['results'].get(key)
            if c is None and m is None and p is None: continue
            ratio = '—' if (c is None or p is None) else f"{p['best'] / c['best']:.2f}x"
            print(f"| {n} | {size} | {cell(c)} | {cell(m)} | {cell(p)} | {ratio} |")

if __name__ == '__main__':
    {'make': make, 'python': python, 'merge': merge, 'speed': speed, 'compare': compare}[sys.argv[1]](*sys.argv[2:])
