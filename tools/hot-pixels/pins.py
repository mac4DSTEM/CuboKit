"""Actual pinned py4DSTEM DataCube.filter_hot_pixels; Float32 cube semantics.
Run under ../../tools/memcap.sh using the pinned isolated Python venv.
"""
from pathlib import Path
import json, subprocess, platform
import numpy as np
import py4DSTEM
from py4DSTEM import DataCube
KIT = Path(__file__).resolve().parents[2]
REF = KIT.parents[0]/'References/CuboKit/py4DSTEM-f050d207'
assert subprocess.check_output(['git','-C',str(REF),'rev-parse','HEAD'],text=True).strip().startswith('f050d207')
assert py4DSTEM.__version__ == '0.14.19' and np.__version__ == '2.5.4' and platform.python_version() == '3.12.14'
for file in ('preprocess/preprocess.py','datacube/datacube.py'):
    assert (Path(py4DSTEM.__file__).parent/file).read_bytes() == subprocess.check_output(['git','-C',str(REF),'show',f'HEAD:py4DSTEM/{file}'])
rng = np.random.default_rng(501)
cases = []
for shape,threshold,index in [((2,3,7,8),8,1),((2,2,3,4),2,8),((1,1,1,5),0,20),((1,2,4,5),3,0),((3,2,5,6),8,1),((1,256,1,1),8,1),((1,1,1,1),-1,1),((1,1,2,2),-1,1)]:
    data = rng.uniform(0,5,shape).astype(np.float32)
    if shape[-2] > 1 and shape[-1] > 2:
        data[:,:,0,0] += 100
        data[:,:,-1,-1] += 120
        data[:,:,1,1] += 90
        data[:,:,1,2] += 60
    elif shape[-1] > 2: data[:,:,0,2] += 30
    if shape == (1,256,1,1): data = np.tile(np.array([1e8,1,-1e8,1],np.float32),64).reshape(shape)
    if shape == (3,2,5,6):
        # Noninteger means and threshold-edge arithmetic; contrast against Float64
        # is recorded for diagnostics, but the supported input contract is Float32.
        data *= np.float32(123.456)
    if shape in ((1,1,1,1),(1,1,2,2)): data = np.full(shape,-0.0,np.float32)
    before = data.copy()
    mean = data.mean(axis=(0,1))
    if shape == (1,256,1,1):
        sequential = np.float32(0)
        for value in data.ravel(): sequential += value
        sequential /= np.float32(data.size)
        assert sequential != mean[0,0]
        print('1x1 detector pairwise mean',float(mean[0,0]),'sequential mean',float(sequential))
    cube,mask = DataCube(data=data).filter_hot_pixels(thresh=threshold,ind_compare=index,return_mask=True)
    cases.append(dict(shape=shape,threshold=threshold,index=index,values=before.ravel().tolist(),
                      mean=mean.ravel().tolist(),mask=mask.ravel().tolist(),filtered=cube.data.ravel().tolist()))
(KIT/'Tests/CuboCoreTests/Fixtures/hot-pixels.json').write_text(json.dumps(dict(reference='py4DSTEM f050d207 / 0.14.19',cases=cases),separators=(',',':'))+'\n')
