"""py4DSTEM side of the CuboKit benchmark, and the merge into docs/benchmarks.md.
Run from the suite folder (one benchmark at a time machine-wide, under memcap):
  tools/memcap.sh -- References/python/py4dstem/bin/python -I -B CuboKit/tools/benchmarks/bench.py make   <dir>    # writes <dir>/{small,mid,typical}.npy
  tools/memcap.sh -- References/python/py4dstem/bin/python -I -B CuboKit/tools/benchmarks/bench.py python <dir>    # writes <dir>/python.json
  CUBOKIT_BENCH_DIR=<dir> swift test -c release --filter BenchmarkHarness                            # writes <dir>/swift.json
  References/python/py4dstem/bin/python -I -B CuboKit/tools/benchmarks/bench.py merge <dir>                       # prints the markdown table
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
    {'make': make, 'python': python, 'merge': merge}[sys.argv[1]](sys.argv[2])
