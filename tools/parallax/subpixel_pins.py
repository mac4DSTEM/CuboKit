"""Pinned subpixel_alignment KDE oracle (bilinear and Lanczos deposition, Gaussian blur, sinc low-pass, default factor
heuristic). Position correction is not pinned here. Run with the pinned venv, python -I -B."""
import json, warnings
from pathlib import Path
src=Path(__file__).with_name('aberration_pins.py').read_text().split("cases=[]")[0]
ns={'__file__':str(Path(__file__).with_name('aberration_pins.py'))}
exec(compile(src,'aberration_pins.py','exec'),ns)
np,p,VERSIONS,ROOT,flat=[ns[k] for k in 'np p VERSIONS ROOT flat'.split()]
warnings.simplefilter('ignore')
rng=np.random.default_rng(20261018)
n=p._stack_BF_unshifted.shape[0]
sh=rng.uniform(-1.3,1.3,(n,2)).astype(np.float32)
p._xy_shifts=sh.copy()
cases=[]
for factor,sigma,lanczos,low in [(2.5,.125,None,False),(2.0,.4,None,True),(3.0,.2,2,False),(None,.125,None,False)]:
    p.subpixel_alignment(kde_upsample_factor=factor,kde_sigma_px=sigma,lanczos_interpolation_order=lanczos,kde_lowpass_filter=low,plot_upsampled_BF_comparison=False,progress_bar=False)
    out=np.asarray(p._recon_BF_subpixel_aligned)
    cases.append(dict(factor=factor,sigma=sigma,lanczos=lanczos,lowpass=low,usedFactor=float(p._kde_upsample_factor),bfLimit=float(p._BF_upsample_limit),dfLimit=float(p._DF_upsample_limit),shape=list(out.shape),output=flat(out)))
pos=[]
for kw in [dict(), dict(position_correction_checkerboard_steps=True), dict(position_correction_gaussian_filter_sigma=(1.5,1.5),position_correction_butterworth_q_lowpass=(.5,.5),position_correction_butterworth_q_highpass=(.02,.02))]:
    p._xy_shifts=sh.copy()
    p.subpixel_alignment(kde_upsample_factor=2.0,kde_sigma_px=.3,position_correction_num_iter=3,plot_upsampled_BF_comparison=False,plot_position_correction_convergence=False,progress_bar=False,**kw)
    pos.append(dict(options={k:(list(v) if isinstance(v,tuple) else v) for k,v in kw.items()},dx=flat(np.asarray(p._probe_dx)),dy=flat(np.asarray(p._probe_dy)),output=flat(np.asarray(p._recon_BF_subpixel_aligned))))
st=np.asarray(p._stack_BF_unshifted)
out=dict(versions=VERSIONS,rows=st.shape[1],columns=st.shape[2],count=n,stack=flat(st),shifts=flat(sh),kr=flat(np.asarray(p._kr)),reciprocalSampling=float(p._reciprocal_sampling[0]),detectorRows=int(p._region_of_interest_shape[0]),scanSampling=float(p._scan_sampling[0]),cases=cases,window=flat(np.asarray(p._window_pad)),positionCases=pos)
dest=ROOT/'Tests/CuboParallaxTests/Fixtures/subpixel.json';dest.write_text(json.dumps(out,separators=(',',':'))+'\n');print(dest,dest.stat().st_size,[(c['usedFactor'],c['bfLimit'],c['dfLimit']) for c in cases])
