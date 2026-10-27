"""Pinned py4DSTEM crop and bin: preprocess.crop_data_real, crop_data_diffraction, bin_data_diffraction on a Float32 DataCube.
Run from the kit root:
  ../References/python/py4dstem/bin/python -I -B tools/cubeview/pins.py
Output: Tests/CuboCoreTests/Fixtures/cubeview.json
"""
from pathlib import Path
import json, subprocess, platform
import numpy as np
import py4DSTEM
from py4DSTEM import DataCube
from py4DSTEM.preprocess.preprocess import crop_data_real, crop_data_diffraction, bin_data_diffraction
KIT = Path(__file__).resolve().parents[2]
REF = KIT.parents[0]/'References/CuboKit/py4DSTEM-f050d207'
assert subprocess.check_output(['git','-C',str(REF),'rev-parse','HEAD'],text=True).strip().startswith('f050d207')
assert py4DSTEM.__version__ == '0.14.19' and np.__version__ == '2.5.4' and platform.python_version() == '3.12.14'
assert (Path(py4DSTEM.__file__).parent/'preprocess/preprocess.py').read_bytes() == subprocess.check_output(['git','-C',str(REF),'show','HEAD:py4DSTEM/preprocess/preprocess.py'])
rng = np.random.default_rng(20261020)
shape = (5, 6, 9, 11)
floats = (rng.gamma(2.0, 40.0, shape) - 5).astype(np.float32)
ints = rng.integers(0, 200, shape).astype(np.float32)
cases = []
# (name, data, scanCrop (r0, rows, c0, cols) | None, detectorCrop likewise | None, bin)
specs = [('cropScan', 'f', (1,3,2,3), None, 1), ('cropDetector', 'f', None, (1,7,0,10), 1), ('binOnly', 'f', None, None, 2),
         ('binRemainder', 'i', None, None, 4), ('cropAndBin', 'f', (1,3,2,3), (1,7,0,10), 3), ('cropAndBinInt', 'i', (0,5,1,4), (2,6,3,8), 2),
         ('binThree', 'f', None, None, 3), ('identity', 'f', None, None, 1), ('wholeDetectorBin', 'i', None, (0,8,0,8), 8)]
for name, kind, scan, det, bin_ in specs:
    data = (floats if kind == 'f' else ints).copy()
    dc = DataCube(data=data.copy())
    if scan is not None: dc = crop_data_real(dc, scan[0], scan[0]+scan[1], scan[2], scan[2]+scan[3])
    if det is not None: dc = crop_data_diffraction(dc, det[0], det[0]+det[1], det[2], det[2]+det[3])
    before = dc.data.shape[-2:]
    dc = bin_data_diffraction(dc, bin_)
    out = np.asarray(dc.data)
    cases.append(dict(name=name, integer=(kind == 'i'), scanCrop=scan, detectorCrop=det, bin=bin_,
                      shape=list(out.shape), values=out.ravel().astype(np.float64).tolist(),
                      discardedRows=int(before[0] % bin_), discardedColumns=int(before[1] % bin_)))
out = dict(reference='py4DSTEM f050d207 / 0.14.19', sourceShape=list(shape), floats=floats.ravel().astype(np.float64).tolist(),
           ints=ints.ravel().astype(np.float64).tolist(), cases=cases)
dest = KIT/'Tests/CuboCoreTests/Fixtures/cubeview.json'
dest.write_text(json.dumps(out,separators=(',',':'))+'\n')
print('fixture bytes', dest.stat().st_size)
