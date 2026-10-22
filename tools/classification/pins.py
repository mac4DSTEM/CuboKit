"""Pins for the classification slice: py4DSTEM's Featurization (PCA, NMF, GMM, class patterns), the NMF and Gaussian mixture of
scikit-learn they call, NumPy's RandomState draws, BraggVectorClassification and BraggVectors.get_virtual_image. Everything is
produced by py4DSTEM itself on small planted inputs; no CuboKit Swift is used. Run from the CuboKit folder with the suite's
reference environment under the memory cap:

    ../tools/memcap.sh -- ../References/python/py4dstem/bin/python -I -B tools/classification/pins.py [featurization|classification|virtualimage|all]

Outputs (floats as repr, so they round-trip exactly):
    Tests/CuboEmbeddingTests/Fixtures/featurization.json
    Tests/CuboBraggTests/Fixtures/classification.json
    Tests/CuboBraggTests/Fixtures/virtualimage.json
"""
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy
import skimage
import sklearn

import py4DSTEM

KIT = Path(__file__).resolve().parents[2]
assert py4DSTEM.__version__ == '0.14.19'
assert np.__version__ == '2.5.4' and platform.python_version() == '3.12.14'
assert sklearn.__version__ == '1.9.1' and scipy.__version__ == '1.18.1' and skimage.__version__ == '0.26.0'
VERSIONS = dict(py4DSTEM=py4DSTEM.__version__, numpy=np.__version__, python=platform.python_version(),
                scikit_learn=sklearn.__version__, scipy=scipy.__version__, scikit_image=skimage.__version__)
warnings.simplefilter('ignore')


def flat(a):
    return np.asarray(a, dtype=np.float64).ravel().tolist()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(',', ':'), allow_nan=True) + '\n')
    print(path.relative_to(KIT), path.stat().st_size, 'bytes')


# ======================================================================================================== featurization
def featurization():
    from sklearn.decomposition import NMF, PCA
    from sklearn.mixture import GaussianMixture
    from py4DSTEM.process.classification.featurization import Featurization
    from py4DSTEM import DataCube

    out = dict(versions=VERSIONS)
    rng = np.random.default_rng(20261022)

    # --- NumPy RandomState draws
    rs = dict(seed=12345, uint32=[], sample=[], randint=[], normal=[], choice=[])
    r = np.random.RandomState(12345)
    rs['sample'] = [float(r.random_sample()) for _ in range(5)]
    r = np.random.RandomState(42)
    rs['randint'] = [int(r.randint(5000)) for _ in range(12)]
    r = np.random.RandomState(7)
    rs['normal'] = [float(r.standard_normal()) for _ in range(9)]
    p = np.ones(17) / 17
    r = np.random.RandomState(99)
    rs['choice'] = [int(r.choice(17, p=p)) for _ in range(6)]
    r = np.random.RandomState(2 ** 32 - 1)
    rs['highSeed'] = [float(r.random_sample()) for _ in range(3)]
    r = np.random.RandomState(0)
    rs['randint2'] = [int(r.randint(1)) for _ in range(2)] + [int(r.randint(2)) for _ in range(4)] + [int(r.randint(2 ** 32)) for _ in range(2)]
    # a draw long enough to cross the 624-word refill
    r = np.random.RandomState(31337)
    rs['long'] = [float(r.random_sample()) for _ in range(400)][-3:]
    out['randomState'] = rs

    # --- scikit-learn NMF (cd), py4DSTEM's calls
    def planted_nonneg(n, d, k, noise=0.02):
        W = rng.gamma(1.5, 1.0, (n, k)) * (rng.random((n, k)) < 0.6)
        H = rng.gamma(1.5, 1.0, (k, d)) * (rng.random((k, d)) < 0.5)
        return W @ H + noise * rng.random((n, d))

    nmf_cases = []

    def nmf_case(name, X, k, init=None, random_state=None, max_iter=200, W0=None, H0=None, tol=1e-4):
        m = NMF(n_components=k, init=init, random_state=random_state, max_iter=max_iter, tol=tol)
        if init == 'custom':
            W = m.fit_transform(X, W=W0.copy(), H=H0.copy())
        else:
            W = m.fit_transform(X)
        init_spec = {None: dict(kind='automatic', seed=random_state if random_state is not None else 0),
                     'custom': dict(kind='custom', w=flat(W0) if W0 is not None else None, h=flat(H0) if H0 is not None else None),
                     'nndsvd': dict(kind='nndsvd'), 'nndsvda': dict(kind='nndsvda'),
                     'nndsvdar': dict(kind='nndsvdar', seed=random_state), 'random': dict(kind='random', seed=random_state)}[init]
        nmf_cases.append(dict(name=name, rows=X.shape[0], columns=X.shape[1], components=k, x=flat(X), initialization=init_spec,
                              maxIter=max_iter, tol=tol, w=flat(W), h=flat(m.components_), error=float(m.reconstruction_err_),
                              nIter=int(m.n_iter_), converged=bool(m.n_iter_ < max_iter)))

    X1 = planted_nonneg(30, 10, 3)
    nmf_case('nndsvdaAutomatic', X1, 3)
    nmf_case('nndsvdaFourComponents', X1, 4)
    nmf_case('nndsvd', X1, 3, init='nndsvd')
    nmf_case('nndsvdar', X1, 3, init='nndsvdar', random_state=5)
    nmf_case('random', X1, 3, init='random', random_state=3)
    nmf_case('randomAutomaticFallback', planted_nonneg(6, 5, 2), 7, random_state=21)  # k > min(n, d): scikit-learn picks 'random'
    X2 = planted_nonneg(8, 20, 3)
    nmf_case('wideNndsvda', X2, 3)
    W0 = rng.random((30, 3)) + 0.1
    H0 = rng.random((3, 10)) + 0.1
    for it in (1, 2, 7, 200):
        nmf_case(f'custom{it}', X1, 3, init='custom', max_iter=it, W0=W0, H0=H0)
    nmf_case('customTight', X1, 3, init='custom', max_iter=500, W0=W0, H0=H0, tol=1e-9)
    # a column that is all zero: zero denominators in the coordinate update
    Xz = X1.copy(); Xz[:, 4] = 0
    nmf_case('zeroColumn', Xz, 3, init='custom', max_iter=50, W0=W0, H0=H0)
    out['nmf'] = nmf_cases

    # --- Featurization.PCA / scalers / feature reductions
    Xp = rng.standard_normal((40, 6)) @ np.diag([4, 2.5, 1.5, 1, 0.6, 0.3]) + rng.standard_normal(6)
    f = Featurization(Xp.copy(), 5, 8, 'p')
    f.PCA(3)
    out['pca'] = dict(rows=40, columns=6, components=3, x=flat(Xp), scores=flat(f.pca))
    pcaw = Featurization(rng.standard_normal((6, 9)) * 3 + 1, 2, 3, 'wide')   # fewer samples than features
    pcaw_x = pcaw.features.copy(); pcaw.PCA(4)
    out['pcaWide'] = dict(rows=6, columns=9, components=4, x=flat(pcaw_x), scores=flat(pcaw.pca))
    Xs = rng.random((12, 5)) * 7 - 2
    Xs[:, 2] = 3.5                                                         # constant column
    f = Featurization(Xs.copy(), 3, 4, 's')
    mm = f.MinMaxScaler(return_scaled=True).copy()
    f = Featurization(Xs.copy(), 3, 4, 's'); f.shift_positive(); sp = f.features.copy()
    red = {}
    for nm in ('mean_feature', 'median_feature', 'max_feature'):
        f = Featurization(Xs.copy(), 3, 4, 's')
        getattr(f, nm)([0, 3, 4])
        red[nm] = flat(f.features)
    f = Featurization(Xs.copy(), 3, 4, 's'); f.median_feature([1, 2])
    red['median_two'] = flat(f.features)
    out['preprocessing'] = dict(rows=12, columns=5, x=flat(Xs), minMax=flat(mm), shiftPositive=flat(sp), reductions=red)

    # --- Featurization.NMF (iterative _nmf_single)
    nmf_runs = []

    def feat_nmf(name, X, scan, **kw):
        f = Featurization(X.copy(), scan[0], scan[1], name)
        try:
            f.NMF(**kw)
            res = f.W if isinstance(f.W, list) else [f.W]
            nmf_runs.append(dict(name=name, rows=X.shape[0], columns=X.shape[1], scan=list(scan), x=flat(X), args={k: v for k, v in kw.items()},
                                 models=[dict(rows=w.shape[0], columns=w.shape[1], values=flat(w)) for w in res]))
        except Exception as e:  # noqa
            nmf_runs.append(dict(name=name, error=type(e).__name__, rows=X.shape[0], columns=X.shape[1], scan=list(scan), x=flat(X),
                                 args={k: v for k, v in kw.items()}))

    Xn = planted_nonneg(36, 14, 4)
    feat_nmf('plain', Xn, (6, 6), max_components=4, num_models=1)
    feat_nmf('threeModelsDefaultSeeds', Xn, (6, 6), max_components=3, num_models=3)
    feat_nmf('seeded', Xn, (6, 6), max_components=3, num_models=2, random_seed=17)
    feat_nmf('bestOfModels', Xn, (6, 6), max_components=3, num_models=3, save_all_models=False)
    # duplicated sources make strongly correlated weight columns, so the merge fires
    Wd = rng.gamma(1.5, 1.0, (36, 3)) * (rng.random((36, 3)) < 0.55)
    Hd = rng.gamma(1.5, 1.0, (3, 14)) * (rng.random((3, 14)) < 0.6)
    Xd = Wd @ Hd + 0.01 * rng.random((36, 14))
    for thr in (0.99, 0.9, 0.6, 0.3):
        feat_nmf(f'iterative{thr}', Xd, (6, 6), max_components=8, num_models=2, merge_thresh=thr, max_iterations=6, random_seed=4)
    feat_nmf('iterativeDefaultSeeds', Xd, (6, 6), max_components=7, num_models=2, merge_thresh=0.5, max_iterations=5)
    feat_nmf('iterativeBest', Xd, (6, 6), max_components=7, num_models=3, merge_thresh=0.5, max_iterations=5, save_all_models=False)
    feat_nmf('oneColumnCorrelationErrors', Xd, (6, 6), max_components=1, num_models=1, max_iterations=3)
    out['featurizationNMF'] = nmf_runs

    # --- Featurization.GMM
    centres = np.array([[0, 0, 0, 0], [4, 3, -2, 1], [-3, 4, 3, -2.]])
    Xg = np.vstack([rng.standard_normal((25, 4)) * [0.7, 0.5, 0.9, 0.6] + c for c in centres])
    perm = rng.permutation(len(Xg)); Xg = Xg[perm]
    gmm_runs = []

    def feat_gmm(name, X, scan, cv, comps, num_models, seed):
        f = Featurization(X.copy(), scan[0], scan[1], name)
        f.GMM(cv=cv, components=comps, num_models=num_models, random_seed=seed)
        fits = []
        k = 0
        for n in range(num_models):
            for c in comps:
                for t in cv:
                    g = f.gmm[k]
                    fits.append(dict(covarianceType=t, components=c, labels=f.gmm_labels[k].tolist(), proba=flat(f.gmm_proba[k]),
                                     weights=flat(g.weights_), means=flat(g.means_), covariances=flat(g.covariances_),
                                     precisionsCholesky=flat(g.precisions_cholesky_), bic=float(g.bic(X)), nIter=int(g.n_iter_),
                                     lowerBound=float(g.lower_bound_), converged=bool(g.converged_)))
                    k += 1
        gmm_runs.append(dict(name=name, rows=X.shape[0], columns=X.shape[1], scan=list(scan), x=flat(X), numModels=num_models,
                             seed=seed, covarianceTypes=cv, components=comps, fits=fits))

    feat_gmm('seeded', Xg, (5, 15), ['full', 'tied', 'diag', 'spherical'], [3], 1, 11)
    feat_gmm('twoComponentCounts', Xg, (5, 15), ['full', 'diag'], [2, 4], 2, 3)
    feat_gmm('defaultSeeds', Xg, (5, 15), ['full', 'spherical'], [3], 3, None)
    Xh = rng.random((50, 7)) * 3                      # unstructured, higher dimension, slower to converge
    feat_gmm('unstructured', Xh, (5, 10), ['full', 'tied', 'diag', 'spherical'], [4], 1, 8)
    # five seeds drawn from RandomState(42) on data without clusters: the k-means++ draws decide the fits
    feat_gmm('unstructuredSeeds', Xh, (5, 10), ['full'], [5], 5, None)
    feat_gmm('unstructuredDiagSix', Xh, (5, 10), ['diag', 'spherical'], [6], 4, None)
    out['gmm'] = gmm_runs

    # --- class patterns (get_class_DPs, the 'pca' branch uses the pattern at x_[x])
    Q = (3, 4)
    R = (4, 5)
    pats = (rng.gamma(2.0, 20.0, (R[0] * R[1], 1) + Q)).astype(np.float32)
    dc = DataCube(data=pats.copy())
    wts = rng.random((R[0] * R[1], 3)) * (rng.random((R[0] * R[1], 3)) < 0.5)
    f = Featurization(wts.copy(), R[0], R[1], 'cls')
    f.pca = wts.copy()
    dc2 = DataCube(data=pats.copy())
    f.get_class_DPs(dc2, 'pca', 0.25)
    cp = [flat(a) for a in f.class_DPs[0]]
    out['classPatterns'] = dict(positions=R[0] * R[1], pixels=Q[0] * Q[1], patterns=flat(pats), weights=flat(wts), classes=3, threshold=0.25, expected=cp)
    return out


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('featurization', 'all'):
        write(KIT / 'Tests/CuboEmbeddingTests/Fixtures/featurization.json', featurization())
