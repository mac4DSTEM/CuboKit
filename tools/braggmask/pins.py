"""Pinned py4DSTEM DataCube.get_braggmask and get_local_ave_dp (braggmask and plain). Run from the clone root with the environment of tools/requirements.txt (tools/pinenv.py finds the inputs):
  python -I -B tools/braggmask/pins.py
Output: Tests/CuboBraggTests/Fixtures/braggmask.json
"""
from pathlib import Path
import json
import numpy as np
import py4DSTEM
from py4DSTEM import DataCube
KIT = Path(__file__).resolve().parents[2]
assert py4DSTEM.__version__ == '0.14.19'
rng = np.random.default_rng(20261021)
shape = (4, 5, 12, 14)
data = (rng.gamma(2.0, 30.0, shape)).astype(np.float32)
dc = DataCube(data=data.copy())
class V:   # the .raw[rx, ry] -> .data, .qx, .qy surface get_braggmask reads
    def __init__(self, pts): self.qx = np.array([p[0] for p in pts]); self.qy = np.array([p[1] for p in pts]); self.data = pts
peaks = {}
for r in range(4):
    for c in range(5):
        k = (r * 5 + c) % 4
        peaks[(r, c)] = [(2.5 + r, 3.0 + c), (8.2, 9.7 - 0.5 * c), (5.0, 5.0)][: (k if k else 1) + 0] if k < 3 else []
class B:
    class raw:
        @staticmethod
        def __class_getitem__(k): return None
B.raw = type('raw', (), {'__getitem__': lambda self, k: V(peaks[tuple(k)])})()
def flat(a): return np.asarray(a, dtype=np.float64).ravel().tolist()
masks = [dict(r=r, c=c, radius=rad, peaks=[list(p) for p in peaks[(r, c)]], mask=np.asarray(dc.get_braggmask(B, r, c, rad)).astype(int).ravel().tolist())
         for (r, c, rad) in [(0, 0, 2.0), (1, 2, 1.5), (3, 4, 3.0), (2, 1, 2.25)]]
ave = []
for (r, c) in [(1, 1), (0, 0), (0, 4), (3, 0), (3, 4), (0, 2), (2, 4), (3, 2), (2, 0)]:
    plain = np.asarray(dc.get_local_ave_dp(r, c))
    ave.append(dict(r=r, c=c, plain=flat(plain)))
for (r, c, rad) in [(1, 1, 2.0), (0, 0, 2.5), (3, 4, 1.5), (2, 2, 4.0)]:
    m = dc.get_local_ave_dp(r, c, braggmask=True, braggvectors=B, braggmask_radius=rad)
    ave.append(dict(r=r, c=c, radius=rad, masked=np.nan_to_num(np.ma.getdata(m).astype(np.float64)).ravel().tolist(), valid=(~np.ma.getmaskarray(m)).astype(int).ravel().tolist()))
lattice = []
for (g1, g2, rad, org, mq, inc, rot) in [((3.0, 0.5), (-0.5, 3.0), 1.2, (5.5, 6.5), 9.0, True, 0), ((3.0, 0.5), (-0.5, 3.0), 1.2, (5.5, 6.5), 9.0, False, 0), ((4.0, 0.0), (1.5, 3.5), 1.6, (6.0, 7.0), 12.0, True, 20.0)]:
    m = dc.make_bragg_mask((12, 14), g1, g2, rad, org, mq, return_sum=True, include_origin=inc, rotation_deg=rot)
    lattice.append(dict(g1=list(g1), g2=list(g2), radius=rad, origin=list(org), maxQ=mq, includeOrigin=inc, rotation=rot, mask=np.asarray(m).astype(int).ravel().tolist()))
out = dict(lattice=lattice, reference='py4DSTEM f050d207 / 0.14.19', shape=list(shape), data=flat(data),
           peaks={f'{k[0]},{k[1]}': [list(p) for p in v] for k, v in peaks.items()}, masks=masks, ave=ave)
dest = KIT / 'Tests/CuboBraggTests/Fixtures/braggmask.json'
dest.write_text(json.dumps(out, separators=(',', ':'), allow_nan=True) + '\n')
print('fixture bytes', dest.stat().st_size)
