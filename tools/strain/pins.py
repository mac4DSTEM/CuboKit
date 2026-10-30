"""Generate reference and original-app pins. Run with the environment of tools/requirements.txt (CONTRIBUTING.md).
Paths resolve from either the primary checkout or a worktree. No new Swift implementation is used.
"""
import hashlib, json, platform, subprocess, tempfile
from pathlib import Path
import numpy as np
import py4DSTEM
from emdfile import PointList
from py4DSTEM.data import RealSlice
from py4DSTEM.process.strain.latticevectors import (fit_lattice_vectors, get_strain_from_reference_g1g2,
    get_reference_g1g2, get_rotated_strain_map, index_bragg_directions)
ROOT = Path(__file__).resolve().parents[2]
PIN='f050d207851f8accd3aa550365baaf1f38230428'
import sys as _sys, pathlib as _pathlib; _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))  # tools/pinenv.py
from pinenv import PY4DSTEM_SRC, PY4DSTEM_TEST_DATA, MAC4DSTEM_REPO
assert subprocess.check_output(['git','-C',str(PY4DSTEM_SRC),'rev-parse','HEAD'],text=True).strip()==PIN
assert py4DSTEM.__version__=='0.14.19' and np.__version__=='2.5.4'
assert platform.python_version()=='3.12.14'
for rel in ['process/strain/latticevectors.py','process/strain/strain.py']:
    assert (Path(py4DSTEM.__file__).parent/rel).read_bytes()==(PY4DSTEM_SRC / 'py4DSTEM'/rel).read_bytes()
def flat(a): return np.asarray(a,dtype=float).ravel().tolist()
# Each peak is [row,column,intensity,h,k]. Planted reciprocal transforms include shear and rotation.
indices=np.array([(h,k) for h in [-1,0,1] for k in [-1,0,1]],dtype=float)
reference=np.array([[8.,1.],[-.5,10.]])
inputs=[]
for i, transform in enumerate([np.eye(2),np.array([[.98,.025],[-.015,1.03]]),np.array([[1.04,-.01],[.03,.99]])]):
    xy=indices@reference@transform+np.array([.3,-.2])
    if i==2: xy[0]+=[.21,-.13]
    inputs.append(np.column_stack([xy,np.arange(1,10),indices]))
# A small crop of a public DM4 pattern: find actual peaks, index against explicitly chosen vectors.
path=PY4DSTEM_TEST_DATA / 'small_datacube.dm4'
cube=py4DSTEM.import_file(str(path))
crop=np.array(cube.data[0,0,200:312,200:312],dtype=float)
from py4DSTEM.preprocess.utils import get_maxima_2D
maxima=get_maxima_2D(crop,subpixel='poly',sigma=1,minSpacing=5,maxNumPeaks=12)
xy=np.column_stack([maxima['x']-56,maxima['y']-56]); basis=reference
hk=np.round(np.linalg.lstsq(basis.T,xy.T,rcond=None)[0].T)
inputs.append(np.column_stack([xy,maxima['intensity'],hk]))
fits=[]
for p in inputs:
    pl=PointList(data=np.array([tuple(v) for v in p],dtype=[('qx',float),('qy',float),('intensity',float),('g1_ind',int),('g2_ind',int)]))
    fits.append(flat(fit_lattice_vectors(pl)))
labels=['x0','y0','g1x','g1y','g2x','g2y','error','mask']
data=np.column_stack([np.array(fits),np.ones(len(fits))]).T.reshape(8,1,len(fits))
rs=RealSlice(data=data,slicelabels=labels)
ref=get_reference_g1g2(rs,np.array([[True,True,True,False]]))
strains=[]
for angle in [0.,31.]:
    sm=get_strain_from_reference_g1g2(rs,*ref)
    sm=get_rotated_strain_map(sm,np.cos(np.radians(angle)),np.sin(np.radians(angle)),False)
    strains.append(dict(angle=angle,values=[flat(sm.get_slice(k).data) for k in ['e_xx','e_yy','e_xy','theta']]))
# Compile ORIGINAL app helpers. The numeric function text is untouched; only access/isolation qualifiers removed.
APP='c809679c8ee23a355da6af4f63665aaf64f90d0d'
src=subprocess.check_output(['git','-C',str(MAC4DSTEM_REPO),'--no-optional-locks','show',APP+':mac4DSTEM/Core/Analysis/StrainMapping.swift'],text=True)
helper='import Foundation\nstruct BraggPeak { let x:Float; let y:Float; let intensity:Float }\nstruct BraggVectors { let peaks:[[BraggPeak]] }\nstruct AnalysisCancellationToken { let isCancelled=false }\nenum StrainMapping {\n'
helper+=src[src.index('    private struct BasisEstimate'):src.index('    /// Full strain-map computation')]
helper+=src[src.index('    // MARK: - Lattice-vector selection'):]
helper=helper.replace('private ','').replace('package ','').replace('nonisolated ','')
# Reference population deliberately differentiates robust and plain medians.
app_bases=[[8+x,.8+x/10,-.4,10+x] for x in [0,.01,.02,.03,1,2]]
consensus=[[[8.,1.,1.],[-.5,10.,1.],[7.5,11.,1.],[-8.,-1.,1.],[.5,-10.,1.]] for _ in range(4)]
runner='''
let input = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))) as! [String:Any]
let fits = (input["inputs"] as! [[[Double]]]).map { p -> [Double] in
 let a = p.map { (h:Float($0[3]),k:Float($0[4]),x:Float($0[0]),y:Float($0[1]),wgt:max(Float($0[2]),1e-6)) }
 let f = StrainMapping.fitLattice(a)!
 return [f.originX,f.originY,f.g1x,f.g1y,f.g2x,f.g2y,f.residual].map(Double.init)
}
let b = input["appBases"] as! [[Double]]
let a = b.map { Float($0[0]) }, c = b.map { Float($0[1]) }, d = b.map { Float($0[2]) }, e = b.map { Float($0[3]) }
let ids = StrainMapping.robustReferenceIndices(candidates:Array(b.indices),g1x:a,g1y:c,g2x:d,g2y:e)
let ref = [a,c,d,e].map { Double(StrainMapping.median(ids.map { $0 < b.count ? $0 : 0 }.map { $0 }).map { _ in Float(0) } ?? 0) }
let med = [a,c,d,e].map { v in Double(StrainMapping.median(ids.map { v[$0] })!) }
let positions = input["consensus"] as! [[[Double]]]
let grid = BraggVectors(peaks:positions.map { $0.map { BraggPeak(x:Float($0[0]),y:Float($0[1]),intensity:Float($0[2])) } })
let chosen = StrainMapping.chooseLatticeVectors(bragg:grid,x0:0,y0:0,cancellation:nil)!
let out:[String:Any] = ["fits":fits,"reference":med,"basis":[chosen.g1.x,chosen.g1.y,chosen.g2.x,chosen.g2.y].map(Double.init)]
print(String(data:try! JSONSerialization.data(withJSONObject:out,options:[.sortedKeys]),encoding:.utf8)!)
'''
runner='\n'.join(l for l in runner.splitlines() if not l.startswith('let ref ='))
with tempfile.TemporaryDirectory() as t:
    t=Path(t); (t/'main.swift').write_text(helper+runner)
    (t/'input.json').write_text(json.dumps(dict(inputs=[flat(p.reshape(-1)) for p in []]))) if False else None
    (t/'input.json').write_text(json.dumps(dict(inputs=[p.tolist() for p in inputs],appBases=app_bases,consensus=consensus)))
    subprocess.run(['swiftc',str(t/'main.swift'),'-o',str(t/'oracle')],check=True)
    app=json.loads(subprocess.check_output([str(t/'oracle'),str(t/'input.json')],text=True))
# Call the actual StrainMap basis selector with plotting disabled at the boundary.
from types import SimpleNamespace
from py4DSTEM.process.strain.strain import StrainMap
mock=SimpleNamespace(calstate={},braggvectors=SimpleNamespace(calstate={}),bvm=SimpleNamespace(data=crop),
                     origin=(56,56),show_bragg_indexing=lambda *a,**k:None)
StrainMap.choose_basis_vectors(mock,subpixel='poly',sigma=1,minSpacing=5,maxNumPeaks=12)
# Arrays stored as little-endian f64; JSON contains only metadata/shape.
out=ROOT/'Tests/CuboStrainTests/Fixtures';out.mkdir(exist_ok=True,parents=True)
index_inputs=[[1e-8,0,0,1e-8,2e-8,-3e-8],[2,0,0,0,4,99],[1,0,0,1,.5,1.5]]
index_outputs=[]
for a,b,c,d,r,q in index_inputs:
    h,k,_=index_bragg_directions(0,0,np.array([r]),np.array([q]),(a,b),(c,d))
    index_outputs.append([int(h[0]),int(k[0])])
arrays=dict(indexInputs=index_inputs,indexOutputs=index_outputs,maxima=np.column_stack([maxima['x'],maxima['y']]),selectedBasis=flat([mock.g1,mock.g2]),fits=fits,reference=flat(ref),appFits=app['fits'],appBases=app_bases,appReference=app['reference'],consensus=consensus,appBasis=app['basis'])
for i,p in enumerate(inputs): arrays[f'input{i}']=p
for i,s in enumerate(strains): arrays[f'strain{i}']=s['values']
meta=dict(py4dstem=PIN,numpy=np.__version__,python=platform.python_version(),app=APP,appSHA256=hashlib.sha256(src.encode()).hexdigest(),crop=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),selection='[0,0,200:312,200:312]'),arrays={})
for name,a in arrays.items():
    a=np.asarray(a,dtype='<f8'); (out/(name+'.f64')).write_bytes(a.tobytes()); meta['arrays'][name]=list(a.shape)
(out/'strain.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Wrote',sum(p.stat().st_size for p in out.iterdir()),'bytes')
