"""Generate CuboEmbedding pins. Run with the OLD pinned venv (scikit-learn 1.4.2; References/python/py4dstem has 1.9.1,
whose PCA fixtures differ, so a regeneration there is a fixture move of its own):

    ../../References/CuboKit/py4dstem-venv/bin/python -I -B tools/embedding/pins.py

Primary pins come from scikit-learn (PCA svd_solver="full", KMeans init=<array>, n_init=1, algorithm="lloyd").
App pins come from the ORIGINAL mac4DSTEM DiffractionEmbedding.swift at APP: its numeric helpers are sliced out of the
git object, access qualifiers removed, compiled with swiftc and run. No CuboEmbedding Swift is used to generate pins.
"""
import hashlib, json, platform, subprocess, tempfile, struct
from pathlib import Path
import numpy as np
import scipy, sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sklearn.utils.extmath as extmath

ROOT = Path(__file__).resolve().parents[2]
SUITE = next(p for p in ROOT.parents if (p / 'References/CuboKit').is_dir())
OUT = ROOT / 'Tests/CuboEmbeddingTests/Fixtures'
APP = 'c809679c8ee23a355da6af4f63665aaf64f90d0d'
APPFILE = 'mac4DSTEM/Core/Analysis/DiffractionEmbedding.swift'
assert np.__version__ == '2.5.4' and platform.python_version() == '3.12.14'
assert sklearn.__version__ == '1.4.2'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name, arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype='<f8')).ravel()
    (OUT / (name + '.f64')).write_bytes(a.tobytes())
    return a.size
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data
rng = np.random.default_rng(20261016)
def planted(n, d, scales, offset):
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return (rng.standard_normal((n, len(scales))) * np.array(scales)) @ q[:len(scales)] + offset
pcaA = planted(60, 8, [5, 3, 2, 1, .5, .3, .2, .1], rng.standard_normal(8) * 3)
pcaB = planted(6, 10, [4, 2.5, 1.5, 1, .6], rng.standard_normal(10))   # fewer rows than columns, rank 5
blobs = np.vstack([rng.standard_normal((30, 3)) * .6 + c for c in ([0, 0, 0], [5, 4, -3], [-4, 6, 3])])
rng.shuffle(blobs)

meta = dict(
    versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  scikit_learn=sklearn.__version__),
    source_sha256=dict(sklearn_pca=sha(sklearn.decomposition._pca.__file__),
                       sklearn_kmeans=sha(sklearn.cluster._kmeans.__file__),
                       sklearn_svd_flip=sha(extmath.__file__)),
    app=dict(commit=APP, file=APPFILE),
    sign_convention='sklearn svd_flip(u_based_decision=True): the largest-|score| entry of each component (first index on ties) is positive',
    cases={})
cases = meta['cases']

# ---------------------------------------------------------------- scikit-learn PCA
for name, X, k in [('pcaA', pcaA, 4), ('pcaB', pcaB, 5)]:
    p = PCA(n_components=k, svd_solver='full').fit(X)
    scores = p.transform(X)
    write(name + '.in', X)
    n = write(name + '.sk', np.concatenate([p.mean_, p.components_.ravel(), p.explained_variance_,
                                            p.explained_variance_ratio_, p.singular_values_, scores.ravel()]))
    cases[name] = dict(rows=X.shape[0], columns=X.shape[1], components=k,
                       layout='mean(d), components(k*d), explainedVariance(k), ratio(k), singularValues(k), scores(n*k)')

# ---------------------------------------------------------------- scikit-learn KMeans
def kmeans_case(name, X, init, **kw):
    km = KMeans(n_clusters=len(init), init=init.copy(), n_init=1, algorithm='lloyd', **kw).fit(X)
    write(name + '.in', np.concatenate([X.ravel(), init.ravel()]))
    write(name + '.sk', np.concatenate([km.cluster_centers_.ravel(), km.labels_.astype(float), [km.inertia_, km.n_iter_]]))
    cases[name] = dict(rows=X.shape[0], columns=X.shape[1], clusters=len(init), maxIterations=km.max_iter,
                       tolerance=km.tol, nIter=int(km.n_iter_),
                       layout='input: data(n*d), initialCentroids(k*d); output: centroids(k*d), labels(n), inertia, nIter')
    return km
init = blobs[[0, 1, 2]] + .3
kmeans_case('kmBlobs', blobs, init)
far = init.copy(); far[2] = [60., 60., 60.]                       # an empty cluster in iteration 1
kmeans_case('kmEmpty', blobs, far)
kmeans_case('kmCapped', blobs, blobs[[3, 4, 5, 6]] + .1, max_iter=2, tol=0.0)   # stops at the iteration cap, relabelled
km_tol = kmeans_case('kmTol', blobs, blobs[[3, 4, 5, 6]] + .1, tol=0.5)         # stops on the centre-shift tolerance
cases['kmTol']['tolerance'] = 0.5

# ---------------------------------------------------------------- the original app source
src = subprocess.check_output(['git', '-C', str(SUITE / 'mac4DSTEM'), '--no-optional-locks', 'show', f'{APP}:{APPFILE}'], text=True)
meta['app']['file_sha256'] = hashlib.sha256(src.encode()).hexdigest()
def between(a, b):
    i = src.index(a); j = src.index(b, i) if b else len(src)
    return src[i:j]
def clean(t):
    return t.replace('package ', '').replace('private ', '').replace('nonisolated ', '')
helpers = clean(between('    /// The box-binning geometry', '    /// The running (double precision)')
                + between('    /// The running (double precision)', '    // MARK: - Deterministic RNG')
                + between('    // MARK: - Deterministic RNG', '    // MARK: - k-means')
                + between('    /// k-means++ seeded Lloyd', None).rstrip().rstrip('}')
                + between('    /// One binned vector\'s coordinates', '    // MARK: - similarity / groupMap'))
stats = between('        // Mean and covariance in double precision.', '        // Non-finite guard')
eig = between('        let (eigenvectors, eigenvalues) = symmetricEigenTop(', '        progress?(statsWeight + eigenWeight)')
comp_line = next(l for l in src.splitlines() if l.strip().startswith('let componentCount = max(1, min(settings.components'))
comp_line = comp_line.replace('settings.components', 'requestedComponents')
swift = '''import Accelerate
import Foundation
struct AnalysisCancellationToken { var isCancelled: Bool { false } }
enum EmbeddingError: Error { case invalidDataset(String) }
enum DiffractionEmbedding {
''' + helpers + '''
    static func appPCA(vectors: [[Float]], dims: Int, requestedComponents: Int) throws -> ([Float], [Float], [Double], [Float], [Float])? {
        let totalPositions = vectors.count
        let cancellation: AnalysisCancellationToken? = nil
        let progress: (@Sendable (Double) -> Void)? = nil
        let statsWeight = 0.5, eigenWeight = 0.05
''' + comp_line + '''
        var accumulator = OuterProductAccumulator(dims: dims)
        for v in vectors { accumulator.add(v) }
        let (sumVec, sumOuter) = accumulator.finish()
''' + stats + eig + '''
        var coordinates = [Float](repeating: 0, count: totalPositions * actualComponents)
        var centred = [Float](repeating: 0, count: dims)
        for pos in 0..<totalPositions {
            vectors[pos].withUnsafeBufferPointer { v in basis.withUnsafeBufferPointer { b in meanFloat.withUnsafeBufferPointer { m in
                centred.withUnsafeMutableBufferPointer { cen in coordinates.withUnsafeMutableBufferPointer { out in
                    project(v.baseAddress!, mean: m.baseAddress!, basis: b.baseAddress!, dims: dims, components: actualComponents,
                            centred: cen.baseAddress!, into: out.baseAddress! + pos * actualComponents)
                } } } } }
        }
        return (meanFloat, basis, Array(eigenvalues.prefix(actualComponents)), explainedVariance, coordinates)
    }
}
func floats(_ a: Any) -> [Float] { (a as! [Double]).map { Float($0) } }
let input = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))) as! [String: Any]
var out: [String: Any] = [:]
let pca = input["pca"] as! [String: Any]
let n = pca["rows"] as! Int, d = pca["columns"] as! Int, k = pca["components"] as! Int
let flat = floats(pca["data"]!)
let vectors = (0..<n).map { Array(flat[($0 * d)..<(($0 + 1) * d)]) }
let r = try! DiffractionEmbedding.appPCA(vectors: vectors, dims: d, requestedComponents: k)!
out["pca"] = ["mean": r.0.map(Double.init), "basis": r.1.map(Double.init), "eigenvalues": r.2,
              "ratio": r.3.map(Double.init), "coordinates": r.4.map(Double.init)]
var km: [[String: Any]] = []
for c in input["kmeans"] as! [[String: Any]] {
    let x = floats(c["data"]!), rows = c["rows"] as! Int, dim = c["columns"] as! Int
    let (a, cen) = DiffractionEmbedding.kMeans(coordinates: x, positions: rows, dimension: dim, requestedGroups: c["groups"] as! Int,
                                               seed: UInt64(c["seed"] as! Int), cancellation: nil)
    km.append(["labels": a, "centroids": cen.map(Double.init)])
}
out["kmeans"] = km
var em: [[String: Any]] = []
for c in input["embed"] as! [[String: Any]] {
    let qy = c["rows"] as! Int, qx = c["columns"] as! Int, b = c["binnedSize"] as! Int
    let px = floats(c["pattern"]!)
    let g = DiffractionEmbedding.BinGeometry.compute(qy: qy, qx: qx, binnedSize: b)
    em.append(["vector": DiffractionEmbedding.embed(pixels: px, base: 0, qy: qy, qx: qx, geometry: g, binnedSize: b).map(Double.init)])
}
out["embed"] = em
print(String(data: try! JSONSerialization.data(withJSONObject: out, options: [.sortedKeys]), encoding: .utf8)!)
'''
# ---------------------------------------------------------------- app inputs
km_cases = [('appKm3', blobs, 3, 1), ('appKm4', blobs, 4, 7), ('appKm5', pcaA, 5, 12345)]
pats = []
for name, qy, qx, b in [('appEmbedA', 12, 10, 4), ('appEmbedB', 5, 6, 8), ('appEmbedC', 16, 16, 16)]:
    P = rng.poisson(3.0, size=(qy, qx)).astype(float) ** 1.5 + np.where(rng.random((qy, qx)) < .1, 200., 0.)
    if name == 'appEmbedC': P[:] = 0                                     # all-zero pattern: max 0, no division
    P = P.astype(np.float32).astype(float)
    if name == 'appEmbedA': P[0, 0] = -3.0                               # negative counts are clamped to zero
    pats.append((name, qy, qx, b, P))
payload = dict(pca=dict(rows=60, columns=8, components=4, data=pcaA.ravel().tolist()),
               kmeans=[dict(rows=X.shape[0], columns=X.shape[1], groups=g, seed=s, data=X.ravel().tolist()) for _, X, g, s in km_cases],
               embed=[dict(rows=qy, columns=qx, binnedSize=b, pattern=P.ravel().tolist()) for _, qy, qx, b, P in pats])
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    (td / 'oracle.swift').write_text(swift)
    (td / 'in.json').write_text(json.dumps(payload))
    subprocess.check_call(['swiftc', '-O', '-Xcc', '-DACCELERATE_NEW_LAPACK', '-o', str(td / 'oracle'), str(td / 'oracle.swift')],
                          stderr=subprocess.DEVNULL)
    res = json.loads(subprocess.check_output([str(td / 'oracle'), str(td / 'in.json')], text=True))
# app PCA: inputs share pcaA.in. Layout: mean, basis(k*d), eigenvalues(k, divisor n), ratio(k), coordinates(n*k)
a = res['pca']
write('pcaA.app', np.concatenate([a['mean'], a['basis'], a['eigenvalues'], a['ratio'], a['coordinates']]))
cases['pcaA']['appLayout'] = 'mean(d), components(k*d), eigenvalues divisor n (k), ratio(k), coordinates(n*k); all Float32-rounded except eigenvalues'
cases['pcaA']['appComponents'] = 4
for (name, X, g, s), r in zip(km_cases, res['kmeans']):
    write(name + '.in', X)
    write(name + '.app', np.concatenate([r['centroids'], np.array(r['labels'], dtype=float)]))
    cases[name] = dict(rows=X.shape[0], columns=X.shape[1], groups=g, seed=s, layout='input: data(n*d); output: centroids(k*d), labels(n)')
for (name, qy, qx, b, P), r in zip(pats, res['embed']):
    write(name + '.in', P)
    write(name + '.app', r['vector'])
    cases[name] = dict(rows=qy, columns=qx, binnedSize=b)
(OUT / 'embedding.json').write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
print('fixture bytes:', sum(p.stat().st_size for p in OUT.iterdir()))
