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


# ====================================================================================================== classification
def make_pla(peaks_by_position, shape):
    from emdfile import PointListArray
    dtype = [('qx', np.float64), ('qy', np.float64), ('intensity', np.float64)]
    pla = PointListArray(dtype=dtype, shape=shape)
    for k, pts in enumerate(peaks_by_position):
        if len(pts):
            pla.get_pointlist(k // shape[1], k % shape[1]).add(np.array([tuple(p) for p in pts], dtype=dtype))
    return pla


def blob(shape, centre, radius, value=1.0):
    r, c = np.mgrid[0:shape[0], 0:shape[1]]
    return value * ((r - centre[0]) ** 2 + (c - centre[1]) ** 2 <= radius ** 2)


def classification():
    from py4DSTEM.process.classification import BraggVectorClassification
    from py4DSTEM.process.classification.braggvectorclassification import get_initial_classes, get_braggpeak_labels_by_scan_position
    from py4DSTEM.process.classification.classutils import get_class_DP, get_class_DP_without_Bragg_scattering
    from py4DSTEM import DataCube

    rng = np.random.default_rng(20261022)
    out = dict(versions=VERSIONS)
    R = (12, 14)
    n = R[0] * R[1]
    # nine reference points in a 64 x 64 detector: the direct beam and eight reflections
    Qx = np.array([32.0, 20.0, 44.0, 32.0, 12.0, 52.0, 20.0, 44.0, 32.0])
    Qy = np.array([32.0, 32.0, 32.0, 14.0, 20.0, 44.0, 48.0, 14.0, 52.0])
    # three phases: A (left blob), B (lower right blob), C (two separate patches); direct beam everywhere
    truth = {'A': blob(R, (4, 3), 4.2), 'B': blob(R, (8, 10), 4.0), 'C': np.maximum(blob(R, (2, 11), 2.6), blob(R, (10, 2), 2.2))}
    sets = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8]}
    peaks = []
    for r in range(R[0]):
        for c in range(R[1]):
            pts = [(Qx[0] + rng.normal(0, 0.25), Qy[0] + rng.normal(0, 0.25), 10 + rng.gamma(4, 1.0))]
            for ph in 'ABC':
                if truth[ph][r, c]:
                    for i in sets[ph]:
                        if rng.random() < 0.9:                                   # a reflection is missed now and then
                            pts.append((Qx[i] + rng.normal(0, 0.35), Qy[i] + rng.normal(0, 0.35), rng.gamma(3, 1.0) + 0.5))
            if rng.random() < 0.08:
                pts.append((rng.uniform(0, 64), rng.uniform(0, 64), rng.gamma(2, 0.5) + 0.2))   # a stray peak
            peaks.append(pts)
    pla = make_pla(peaks, R)
    out['scan'] = list(R)
    out['peaks'] = [[list(map(float, p)) for p in pts] for pts in peaks]
    out['Qx'] = Qx.tolist(); out['Qy'] = Qy.tolist()
    out['truth'] = {k: flat(v) for k, v in truth.items()}

    cases = []

    # --- labels and X
    lab = []
    for max_dist in (None, 1.0):
        labels = get_braggpeak_labels_by_scan_position(pla, Qx, Qy, max_dist)
        bv = BraggVectorClassification(pla, Qx, Qy, X_is_boolean=True, max_dist=max_dist)
        bv_i = BraggVectorClassification(pla, Qx, Qy, X_is_boolean=False, max_dist=max_dist)
        lab.append(dict(maxDist=max_dist, labels=[sorted(int(i) for i in labels[r][c]) for r in range(R[0]) for c in range(R[1])],
                        xBoolean=flat(bv.X), xIntensity=flat(bv_i.X)))
    out['labels'] = lab

    # --- get_initial_classes
    bv = BraggVectorClassification(pla, Qx, Qy)
    ic = []
    for kw in (dict(), dict(thresh=0.5), dict(thresh=0.1, BP_fraction_thresh=0.0), dict(BP_fraction_thresh=0.5),
               dict(max_iterations=1), dict(n_corr_init=3), dict(n_corr_init=3, thresh=0.6)):
        sets_ = get_initial_classes(bv.braggpeak_labels, len(Qx), **kw)
        ic.append(dict(args=kw, classes=[sorted(int(i) for i in s) for s in sets_]))
    out['initialClasses'] = ic

    # --- the state after get_initial_classes_by_cooccurrence
    def state(b):
        return dict(classCount=int(b.N_c), w=flat(b.W), h=flat(b.H))
    for kw in (dict(), dict(thresh=0.5), dict(n_corr_init=3)):
        b = BraggVectorClassification(pla, Qx, Qy)
        b.get_initial_classes_by_cooccurrence(**kw)
        cases.append(dict(name='cooccurrence', args=kw, state=state(b)))
    bI = BraggVectorClassification(pla, Qx, Qy, X_is_boolean=False)
    bI.get_initial_classes_by_cooccurrence()
    cases.append(dict(name='cooccurrenceIntensity', args=dict(booleanData=False), state=state(bI)))
    # from images (H given; W by least squares)
    imgs = np.stack([truth['A'], truth['B'], truth['C']], axis=2).astype(float)
    b = BraggVectorClassification(pla, Qx, Qy)
    b.get_initial_classes_from_images(imgs)
    cases.append(dict(name='fromImages', images=[flat(imgs[:, :, i]) for i in range(3)], state=state(b)))
    # rank-deficient images (a repeated class): the minimum-norm solution
    imgs2 = np.stack([truth['A'], truth['B'], truth['A'], truth['C']], axis=2).astype(float)
    b = BraggVectorClassification(pla, Qx, Qy)
    b.get_initial_classes_from_images(imgs2)
    cases.append(dict(name='fromImagesRankDeficient', images=[flat(imgs2[:, :, i]) for i in range(4)], state=state(b)))
    # more classes than reference points
    imgs3 = np.stack([blob(R, (r, c), 1.5) for r in (2, 6, 10) for c in (2, 7, 11)] + [truth['A'], truth['B'], truth['C']], axis=2).astype(float)
    b = BraggVectorClassification(pla, Qx, Qy)
    b.get_initial_classes_from_images(imgs3)
    cases.append(dict(name='fromImagesWide', images=[flat(imgs3[:, :, i]) for i in range(imgs3.shape[2])], state=state(b)))
    out['cases'] = cases

    # --- refinement from a given state: the candidates of nmf, split, merge, remove, split/merge by index
    def run_ops(tag, W, H, ops):
        res = []
        for op in ops:
            b = BraggVectorClassification(pla, Qx, Qy)
            b.N_c = W.shape[1]; b.W = W.copy(); b.H = H.copy(); b.W_next = b.H_next = b.N_c_next = None
            kind = op['op']
            try:
                if kind == 'nmf': b.nmf(max_iterations=op['maxIterations'])
                elif kind == 'split': b.split(sigma=op['sigma'], threshold_split=op['thresholdSplit'], expand_mask=op['expandMask'], minimum_pixels=op['minimumPixels'])
                elif kind == 'splitIndex': b.split_by_class_index(op['index'], sigma=op['sigma'], threshold_split=op['thresholdSplit'], expand_mask=op['expandMask'], minimum_pixels=op['minimumPixels'])
                elif kind == 'merge': b.merge_iterative(threshBPs=op['thresholdPeaks'], threshScanPosition=op['thresholdScan'])
                elif kind == 'mergeIndex': b.merge_by_class_index(op['i'], op['j'])
                elif kind == 'remove': b.remove_class(op['index'])
                cand = dict(classCount=int(b.N_c_next), w=flat(b.W_next), h=flat(b.H_next))
                b.accept()
                accepted = dict(classCount=int(b.N_c), w=flat(b.W), h=flat(b.H))
                res.append(dict(op=op, candidate=cand, accepted=accepted))
            except Exception as e:  # noqa
                res.append(dict(op=op, error=type(e).__name__))
        return dict(name=tag, classCount=int(W.shape[1]), w=flat(W), h=flat(H), results=res)

    states = []
    b = BraggVectorClassification(pla, Qx, Qy)
    b.get_initial_classes_by_cooccurrence()
    base_ops = [dict(op='nmf', maxIterations=1), dict(op='nmf', maxIterations=4), dict(op='nmf', maxIterations=200),
                dict(op='split', sigma=2, thresholdSplit=0.25, expandMask=1, minimumPixels=1),
                dict(op='split', sigma=1, thresholdSplit=0.4, expandMask=2, minimumPixels=6),
                dict(op='split', sigma=0.5, thresholdSplit=0.1, expandMask=0, minimumPixels=1),
                dict(op='merge', thresholdPeaks=0.1, thresholdScan=0.1),
                dict(op='merge', thresholdPeaks=0.5, thresholdScan=0.5),
                dict(op='remove', index=0), dict(op='splitIndex', index=0, sigma=2, thresholdSplit=0.25, expandMask=1, minimumPixels=1),
                dict(op='mergeIndex', i=0, j=1)]
    states.append(run_ops('cooccurrenceState', b.W, b.H, base_ops))
    # a crafted state: 0 and 1 alike, 2 in two patches, 3 apart, 4 a speck; and 5-7 three alike classes (the dropped-partner quirk)
    G = R
    imgs_c = [blob(G, (4, 3), 3.5), blob(G, (4, 3), 3.5) * (0.7 + 0.3 * rng.random(G)),
              np.maximum(blob(G, (2, 11), 2.4), blob(G, (10, 2), 2.4)) * (0.5 + 0.5 * rng.random(G)),
              blob(G, (9, 10), 3.0) * (0.6 + 0.4 * rng.random(G)), blob(G, (6, 6), 0.6),
              blob(G, (5, 12), 2.5), blob(G, (5, 12), 2.5) * (0.8 + 0.2 * rng.random(G)), blob(G, (5, 12), 2.6) * (0.8 + 0.2 * rng.random(G))]
    Hc = np.stack([im.ravel() for im in imgs_c]).astype(float)
    Wc = rng.random((9, 8)) * (rng.random((9, 8)) < 0.5)
    Wc[:, 1] = Wc[:, 0] * (0.9 + 0.2 * rng.random(9)); Wc[:, 6] = Wc[:, 5] * (0.9 + 0.2 * rng.random(9)); Wc[:, 7] = Wc[:, 5] * (0.9 + 0.2 * rng.random(9))
    Wc[0, :] += 0.05
    crafted_ops = [dict(op='split', sigma=2, thresholdSplit=0.25, expandMask=1, minimumPixels=1),
                   dict(op='split', sigma=2, thresholdSplit=0.25, expandMask=1, minimumPixels=4),
                   dict(op='split', sigma=1.5, thresholdSplit=0.5, expandMask=2, minimumPixels=3),
                   dict(op='split', sigma=3.5, thresholdSplit=0.05, expandMask=1, minimumPixels=1),
                   dict(op='splitIndex', index=2, sigma=2, thresholdSplit=0.25, expandMask=1, minimumPixels=1),
                   dict(op='splitIndex', index=4, sigma=2, thresholdSplit=0.25, expandMask=1, minimumPixels=1),
                   dict(op='merge', thresholdPeaks=0.1, thresholdScan=0.1), dict(op='merge', thresholdPeaks=0.9, thresholdScan=0.9),
                   dict(op='merge', thresholdPeaks=0.6, thresholdScan=0.4), dict(op='merge', thresholdPeaks=-1.0, thresholdScan=-1.0),
                   dict(op='mergeIndex', i=0, j=1), dict(op='mergeIndex', i=1, j=0), dict(op='mergeIndex', i=2, j=7), dict(op='mergeIndex', i=7, j=2),
                   dict(op='remove', index=3), dict(op='remove', index=7), dict(op='nmf', maxIterations=3)]
    states.append(run_ops('craftedState', Wc, Hc, crafted_ops))
    out['states'] = states

    # --- class patterns: get_class_DP and the Bragg-masked variant
    Qn = (8, 9)
    data = rng.gamma(2.0, 25.0, R + Qn).astype(np.float32)
    dc = DataCube(data=data.copy())
    img = truth['A'] * (0.4 + 0.6 * rng.random(R)) + 0.005 * rng.random(R)
    dark = rng.random(Qn) * 5
    xs = rng.normal(0, 1.4, R); ys = rng.normal(0, 1.4, R)
    dps = []
    dps.append(dict(args=dict(), dp=flat(get_class_DP(dc, img))))
    dps.append(dict(args=dict(threshold=0.3), dp=flat(get_class_DP(dc, img, thresh=0.3))))
    dps.append(dict(args=dict(dark=True), dp=flat(get_class_DP(dc, img, darkref=dark))))
    dps.append(dict(args=dict(shifts=True, integerShifts=True), dp=flat(get_class_DP(dc, img, xshifts=xs, yshifts=ys, intshifts=True))))
    dps.append(dict(args=dict(shifts=True, integerShifts=False), dp=flat(get_class_DP(dc, img, xshifts=xs, yshifts=ys, intshifts=False))))
    dps.append(dict(args=dict(shifts=True, integerShifts=True, dark=True), dp=flat(get_class_DP(dc, img, xshifts=xs, yshifts=ys, darkref=dark))))
    # shifts with exact halves, to pin round half to even
    xh = np.full(R, 0.5); xh[::2] = 1.5; yh = np.full(R, -2.5)
    dps.append(dict(args=dict(shifts='halves'), dp=flat(get_class_DP(dc, img, xshifts=xh, yshifts=yh))))
    out['classPattern'] = dict(patterns=flat(data), detector=list(Qn), image=flat(img), dark=flat(dark), rowShifts=flat(xs), columnShifts=flat(ys),
                               rowHalves=flat(xh), columnHalves=flat(yh), cases=dps)
    peaks_q = []
    for r in range(R[0]):
        for c in range(R[1]):
            k = int(rng.integers(0, 4))
            peaks_q.append([(rng.uniform(0, 8), rng.uniform(0, 9), 1.0) for _ in range(k)])
    plaq = make_pla(peaks_q, R)
    nb = []
    nb.append(dict(args=dict(radius=1.5), dp=flat(get_class_DP_without_Bragg_scattering(dc, img, plaq, 1.5, 4.0, 4.5))))
    nb.append(dict(args=dict(radius=2.5, threshold=0.2, origin=[3.5, 4.0]), dp=flat(get_class_DP_without_Bragg_scattering(dc, img, plaq, 2.5, 3.5, 4.0, thresh=0.2))))
    nb.append(dict(args=dict(radius=1.5, shifts=True, integerShifts=True, dark=True),
                   dp=flat(get_class_DP_without_Bragg_scattering(dc, img, plaq, 1.5, 4.0, 4.5, xshifts=xs, yshifts=ys, darkref=dark))))
    nb.append(dict(args=dict(radius=2.0, shifts=True, integerShifts=False),
                   dp=flat(get_class_DP_without_Bragg_scattering(dc, img, plaq, 2.0, 4.0, 4.5, xshifts=xs, yshifts=ys, intshifts=False))))
    out['classPatternMasked'] = dict(peaks=[[list(map(float, p)) for p in pts] for pts in peaks_q], origin=[4.0, 4.5], cases=nb)
    return out


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('featurization', 'all'):
        write(KIT / 'Tests/CuboEmbeddingTests/Fixtures/featurization.json', featurization())
    if which in ('classification', 'all'):
        write(KIT / 'Tests/CuboBraggTests/Fixtures/classification.json', classification())
