"""NumPy float64 transcription of the app's Fourier-integration iDPC.
This is NOT py4DSTEM iterative DPC. Periodic/centered zero-padded boundary,
mean subtraction before padding and after cropping, regularized inverse gradient.
Run with the pinned venv under ../../tools/memcap.sh, from this worktree.
"""
from pathlib import Path
import json, platform, hashlib, subprocess
import numpy as np
assert np.__version__ == '2.5.4' and platform.python_version() == '3.12.14'
KIT = Path(__file__).resolve().parents[2]
app_commit = 'c809679c8ee23a355da6af4f63665aaf64f90d0d'
source = subprocess.check_output(['git','-C',str(KIT.parents[1]/'mac4DSTEM'),'show',
                                  app_commit+':mac4DSTEM/Core/Analysis/DPC.swift'])
source_sha256 = hashlib.sha256(source).hexdigest()
cases = []
for rows,cols,factor,reg,dr,dc,scale in [(7,10,1,0,0.8,1.1,1),(8,8,2,1e-4,1,1,1),(7,10,3,0.03,0.8,1.1,2*np.pi*0.17)]:
    r,c = np.indices((rows,cols))
    phase = 0.7*np.sin(2*np.pi*r/rows)+0.4*np.cos(4*np.pi*c/cols)+0.3*np.sin(2*np.pi*(r/rows+c/cols))
    row_grad = 0.7*2*np.pi/(rows*dr)*np.cos(2*np.pi*r/rows)+0.3*2*np.pi/(rows*dr)*np.cos(2*np.pi*(r/rows+c/cols))
    col_grad = -0.4*4*np.pi/(cols*dc)*np.sin(4*np.pi*c/cols)+0.3*2*np.pi/(cols*dc)*np.cos(2*np.pi*(r/rows+c/cols))
    row_com = row_grad/scale+2; col_com = col_grad/scale-3
    a,b = row_com*scale,col_com*scale
    a-=a.mean(); b-=b.mean()
    nr,nc = rows*factor,cols*factor
    ro,co = (nr-rows)//2,(nc-cols)//2
    pad_a,pad_b = np.zeros((nr,nc)),np.zeros((nr,nc))
    pad_a[ro:ro+rows,co:co+cols] = a; pad_b[ro:ro+rows,co:co+cols] = b
    qr,qc = np.fft.fftfreq(nr,dr)[:,None],np.fft.fftfreq(nc,dc)[None,:]
    q2 = qr**2+qc**2
    denominator = 2*np.pi*(q2+reg*((0.5/dr)**2+(0.5/dc)**2))
    denominator[0,0] = np.inf
    spectrum = -1j*(qr*np.fft.fft2(pad_a)+qc*np.fft.fft2(pad_b))/denominator
    integrated = np.fft.ifft2(spectrum).real[ro:ro+rows,co:co+cols].copy()
    integrated-=integrated.mean()
    if factor == 1: assert np.max(np.abs(integrated-(phase-phase.mean()))) < 3e-15
    cases.append(dict(rows=rows,columns=cols,factor=factor,regularization=reg,rowSampling=dr,columnSampling=dc,
                      scale=scale,row=row_com.ravel().tolist(),column=col_com.ravel().tolist(),phase=integrated.ravel().tolist(),truth=(phase-phase.mean()).ravel().tolist()))
output = dict(oracle='NumPy transcription of app Fourier iDPC, float64; not py4DSTEM DPC',numpy=np.__version__,appCommit=app_commit,sourceSHA256=source_sha256,cases=cases)
(KIT/'Tests/CuboPhaseTests/Fixtures/idpc.json').write_text(json.dumps(output,separators=(',',':'))+'\n')
