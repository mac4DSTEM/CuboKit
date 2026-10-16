"""Pinned depth_section oracle. Reference path: use_CTF_fit=False (the reference's use_CTF_fit=True raises TypeError at
0.14.19: `_calculate_CTF(*self._aberrations_coefs)`). Full-fit path: NumPy transcription of the app source (commit
APP_PIN, ParallaxDepthSectioning.swift), never generated from the Swift port."""
import json, hashlib
from pathlib import Path
import runpy
src=Path(__file__).with_name('aberration_pins.py').read_text().split("cases=[]")[0]
ns={'__file__':str(Path(__file__).with_name('aberration_pins.py'))}
exec(compile(src,'aberration_pins.py','exec'),ns)
np,p,terms,shifts,angles,VERSIONS,ROOT,APP_PIN,fingerprint,flat=[ns[k] for k in 'np p terms shifts angles VERSIONS ROOT APP_PIN fingerprint flat'.split()]
p._xy_shifts=shifts.copy()
p.aberration_fit(max_radial_order=4)
stack=np.asarray(p._stack_BF_shifted); n=stack.shape[0]; nx,ny=stack.shape[1:]
depths=[-60.,0.,35.]
cases=[]
for limit,power in [(None,1.),(0.35,1.5)]:
    ref=p.depth_section(depth_angstroms=depths,use_CTF_fit=False,plot_depth_sections=False,progress_bar=False,k_info_limit=limit,k_info_power=power)
    # app transcription (full fit)
    s=1.25; wl=p._wavelength
    kx=np.fft.fftfreq(nx,s);ky=np.fft.fftfreq(ny,s);k2=kx[:,None]**2+ky[None,:]**2
    alpha=np.sqrt(k2)*wl;theta=np.arctan2(ky[None,:],kx[:,None]);chi=np.zeros_like(k2)
    for (m,nn,a),c in zip(terms,p._aberrations_coefs):
        r=alpha**(m+1)/(m+1); chi+=r*c*(1 if nn==0 else np.cos(nn*theta) if a==0 else np.sin(nn*theta))
    ctf=np.sign(np.sin(chi*2*np.pi/wl));ctf[0,0]=0
    env=np.ones_like(k2) if limit is None else 1/(1+k2**power/limit**(2*power))
    fr=np.fft.fftfreq(nx)[:,None];fc=np.fft.fftfreq(ny)[None,:]
    app=[]
    for d in depths:
        acc=np.zeros((nx,ny))
        for b in range(n):
            sr=-angles[b,0]*d/s;sc=-angles[b,1]*d/s
            ph=np.exp(-2j*np.pi*(fr*sr+fc*sc))
            acc+=np.fft.ifft2(np.fft.fft2(stack[b])*ph*ctf*env).real
        app.append(acc/n)
    cases.append(dict(limit=limit,power=power,reference=flat(ref),app=flat(np.array(app))))
out=dict(versions=VERSIONS,app=dict(commit=APP_PIN,sha256=fingerprint('ParallaxDepthSectioning')),rows=nx,columns=ny,count=n,
  stack=flat(stack),angles=flat(angles),terms=terms.tolist(),coefs=flat(p._aberrations_coefs),c1=float(p.aberrations_C1),wavelength=wl,depths=depths,cases=cases)
dest=ROOT/'Tests/CuboParallaxTests/Fixtures/depth.json';dest.write_text(json.dumps(out,separators=(',',':'))+'\n');print(dest,dest.stat().st_size)
