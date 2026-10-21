"""NumPy reference for the DPC views: magnitude np.hypot and the CoM angle np.angle(x + 1j y), as py4DSTEM's
show_complex_CoM builds the complex CoM (`com_x + 1j*com_y`, x = first axis). Wrapped to [0, 2 pi) with +0 canonical.
Run from the kit root:  ../tools/memcap.sh -- ../References/python/py4dstem/bin/python -I -B tools/dpcviews/pins.py
Output: Tests/CuboPhaseTests/Fixtures/dpcviews.json
"""
from pathlib import Path
import inspect, json
import numpy as np
import py4DSTEM
from py4DSTEM.process.phase import phase_base_class
assert py4DSTEM.__version__ == '0.14.19' and np.__version__ == '2.5.4'
src = inspect.getsource(phase_base_class.PhaseReconstruction.show_complex_CoM)
assert 'complex_com = com[0] + 1j * com[1]' in src
KIT = Path(__file__).resolve().parents[2]
rng = np.random.default_rng(20261020)
R, C = 4, 5
row = rng.normal(0, 1.3, R*C); col = rng.normal(0, 0.7, R*C)
# axis-aligned, zero and signed-zero vectors
row[:9] = [1, -1, 0, 0, 0.0, -0.0, -1, 0, 1]; col[:9] = [0, 0, 1, -1, 0.0, 0.0, -0.0, -0.0, -1e-20]
# py4DSTEM: x (row) + 1j y (column). Built with .real/.imag so signed zeros survive: `row + 1j*col` in NumPy drops the sign
# of a zero real part (complex multiplication adds +0), which would change atan2(+0, -0) from pi to 0 at exact zeros.
complex_com = np.empty(R*C, dtype=complex); complex_com.real = row; complex_com.imag = col
mag = np.hypot(row, col)
def wrap(a):
    a = np.where(a < 0, a + 2*np.pi, a)
    a = np.where(a >= 2*np.pi, 0.0, a)           # a tiny negative angle rounds to 2 pi
    return np.where(a == 0, 0.0, a)
# only the unwrapped column is py4DSTEM's own value; the wrapping is this kit's definition
py_angle = wrap(np.angle(complex_com))           # py4DSTEM: atan2(column, row)
app_angle = wrap(np.arctan2(row, col))           # row over column
out = dict(rows=R, columns=C, row=row.tolist(), column=col.tolist(), magnitude=mag.tolist(),
           py4dstemAngle=py_angle.tolist(), rowOverColumnAngle=app_angle.tolist(),
           unwrappedPy4dstem=np.angle(complex_com).tolist())
dest = KIT/'Tests/CuboPhaseTests/Fixtures/dpcviews.json'
dest.write_text(json.dumps(out)+'\n'); print('bytes', dest.stat().st_size)
