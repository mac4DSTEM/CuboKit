"""Direct pinned aberration_fit/correct calls plus independent application transcriptions."""
import sys, runpy, json, hashlib
from pathlib import Path
common=runpy.run_path(str(Path(__file__).with_name('preprocess_pins.py')))
globals().update({k:common[k] for k in ['ROOT','MAC4DSTEM_REPO','PIN','VERSIONS','flat','make','np','subprocess']})
from py4DSTEM.process.phase.utils import calculate_aberration_gradient_basis
from scipy.linalg import polar
APP_PIN='c809679c8ee23a355da6af4f63665aaf64f90d0d'
def fingerprint(name):
    source=subprocess.check_output(['git','--no-optional-locks','-C',str(MAC4DSTEM_REPO),'show',APP_PIN+':mac4DSTEM/Core/Analysis/'+name+'.swift'])
    return hashlib.sha256(source).hexdigest()
rng=np.random.default_rng(20261017)
# Non-centrosymmetric selected detector geometry makes the two basis origins distinguishable.
cube=rng.uniform(1,2,(6,8,9,10)).astype(np.float32)
mask=np.zeros((9,10),bool); mask[1:8,2:9]=True;mask[1:4,2:4]=False
p=make(cube,(2,2)).preprocess(dp_mask=mask,edge_blend=1.5,plot_average_bf=False)
inds=p._xy_inds; angles=p._probe_angles.astype(float)
m=np.array([[1400.,220.],[-150.,1000.]])
shifts=angles@m + np.stack([angles[:,0]**2*140000,angles[:,1]**2*-90000],1)
shifts/=1.25
terms=np.array([[m,n,a] for m in range(1,4) for n in range(min(4,m+1)+1) if (m+n)%2 for a in range(1 if n==0 else 2)])
def app_fit(transpose,forced,method):
    measured,rot,c1,c12a,c12b=p._aberration_fit_polar_decomposition(shifts,(1.25,1.25),angles,force_transpose=transpose,force_rotation_angle_deg=forced)
    u0,v0=angles.T;u=u0*np.cos(-rot)+v0*np.sin(-rot);v=-u0*np.sin(-rot)+v0*np.cos(-rot)
    alpha=np.hypot(u,v);theta=np.arctan2(v,u)
    du=[];dv=[]
    for m,n,a in terms:
        radial=alpha**(m-1);cn=np.cos(n*theta);sn=np.sin(n*theta)
        if n==0: x,y=u*radial,v*radial
        elif a==0: x,y=radial*((m+1)*u*cn+n*v*sn)/(m+1),radial*((m+1)*v*cn-n*u*sn)/(m+1)
        else: x,y=radial*((m+1)*u*sn-n*v*cn)/(m+1),radial*((m+1)*v*sn+n*u*cn)/(m+1)
        du.append(x);dv.append(y)
    g=np.vstack((np.array(du).T,np.array(dv).T))
    coefs=np.zeros(len(terms));coefs[:3]=c1,c12a,c12b
    fitted=g@coefs
    groups=[np.flatnonzero(terms[:,0]==m) for m in np.unique(terms[:,0])]
    passes=[np.concatenate(groups)] if method=='global' else [np.concatenate(groups[:i+1]) for i in range(len(groups))] if method=='recursive' else groups
    for group in passes:
        # Column-normalized normal equations reproduce the app's solver on this well-conditioned geometry.
        a=g[:,group];sc=np.linalg.norm(a,axis=0);a=a/sc
        increment=np.linalg.solve(a.T@a,a.T@(measured.T.ravel()-fitted))/sc
        coefs[group]+=increment;fitted+=g[:,group]@increment
    if transpose: coefs[(terms[:,1]>0)&(terms[:,2]==0)]*=-1
    return coefs
cases=[]
for transpose,forced,method in [(False,None,'recursive'),(False,17.,'global'),(True,None,'recursive-exclusive')]:
    p._xy_shifts=shifts.copy()
    p.aberration_fit(max_radial_order=4,force_transpose=transpose,force_rotation_angle_deg=forced,fit_method=method)
    p.aberration_correct(use_CTF_fit=True,plot_corrected_phase=False,upsampled=False,q_lowpass=.3,q_highpass=.12)
    # Application explicitly applies q_highpass while reference signature ignores it.
    kx=np.fft.fftfreq(p._recon_BF.shape[0],1.25);ky=np.fft.fftfreq(p._recon_BF.shape[1],1.25)
    k2=kx[:,None]**2+ky[None,:]**2
    even=terms[:,0]%2==1
    # App source calculates CTF in Double then stores Float.
    alpha=np.sqrt(k2)*p._wavelength;theta=np.arctan2(ky[None,:],kx[:,None])
    appchi=[]
    for subset in [even,~even]:
        chi=np.zeros_like(k2)
        for (m,n,a),c in zip(terms[subset],p._aberrations_coefs[subset]):
            radial=alpha**(m+1)/(m+1)
            chi+=radial*c*(1 if n==0 else np.cos(n*theta) if a==0 else np.sin(n*theta))
        appchi.append((chi*2*np.pi/p._wavelength).astype(np.float32))
    transfer=np.sign(np.sin(appchi[0].astype(float)))*np.exp(-1j*appchi[1].astype(float))
    transfer/=1+(np.sqrt(k2)/.3)**4;transfer*=1-1/(1+(np.sqrt(k2)/.12)**4);transfer[0,0]=0
    corrected=np.fft.ifft2(np.fft.fft2(p._recon_BF)*transfer).real
    cases.append(dict(transpose=transpose,forced=forced,method=method,rotation=float(p.rotation_Q_to_R_rads),low=[float(p.aberrations_C1),float(p.aberrations_C12a),float(p.aberrations_C12b)],coefs=flat(p._aberrations_coefs),appCoefs=flat(app_fit(transpose,forced,method)),corrected=flat(p._recon_phase_corrected),appCorrected=flat(corrected)))
out=dict(versions=VERSIONS,app=dict(commit=APP_PIN,fitSHA256=fingerprint('ParallaxAberrationFitting'),correctionSHA256=fingerprint('ParallaxAberrationCorrection')),detectorShape=[9,10],indices=flat(inds),angles=flat(angles),shifts=flat(shifts),terms=terms.tolist(),rows=p._recon_BF.shape[0],columns=p._recon_BF.shape[1],image=flat(p._recon_BF),wavelength=p._wavelength,cases=cases)
dest=ROOT/'Tests/CuboParallaxTests/Fixtures/aberrations.json';dest.write_text(json.dumps(out,separators=(',',':'))+'\n');print(dest,dest.stat().st_size)
