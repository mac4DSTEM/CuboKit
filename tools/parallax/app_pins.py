"""NumPy transcriptions of the source application's parallax pieces that CuboParallax 0.6 exposes: the spectrum-level
shift estimate (app `ParallaxAlignment.correlationShift` + `MatrixDFTCorrelation`, Double instead of Float), the CTF
transfer function (app `ParallaxAberrationCorrection`, Double), and the position-correction loop of
`ParallaxSubpixelReconstruction` (Float32 arithmetic as in the app; the deposition in Float64 as the kit's).
App commit c809679c (the pin of aberration_pins.py). Writes Tests/CuboParallaxTests/Fixtures/appvariants.json.
Run: python -I -B tools/parallax/app_pins.py (needs only NumPy and SciPy)."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / 'Tests/CuboParallaxTests/Fixtures'
flat = lambda a: np.asarray(a, dtype=float).ravel().tolist()

# ---- spectrum-level shift estimate (Float64 transcription of the app's correlationShift) ----
def wrap(v, size):
    half = size / 2
    return np.mod(v + half, size) - half
def parabolic(prev, center, nxt):
    d = 4 * center - 2 * nxt - 2 * prev
    with np.errstate(all='ignore'):
        o = (nxt - prev) / d
    return float(o) if np.isfinite(o) else 0.0
def refine(C, init, u):
    H, W = C.shape
    row = np.round(init[0] * 2) / 2; col = np.round(init[1] * 2) / 2
    row = np.round(row * u) / u; col = np.round(col * u) / u
    P = int(np.ceil(1.5 * u)); g = np.floor(P / 2)
    cr, cc = g - u * row, g - u * col
    wfc = np.where(np.arange(W) < (W + 1) // 2, np.arange(W), np.arange(W) - W).astype(float)
    wfr = np.where(np.arange(H) < (H + 1) // 2, np.arange(H), np.arange(H) - H).astype(float)
    colK = np.exp(1j * (-2 * np.pi / (W * u)) * wfc[:, None] * (np.arange(P)[None, :] - cc))
    rowK = np.exp(1j * (-2 * np.pi / (H * u)) * (np.arange(P)[:, None] - cr) * wfr[None, :])
    patch = np.real(rowK @ (np.conj(C) @ colK))
    m = int(np.argmax(patch)); pr, pc = divmod(m, P)
    sr, sc = float(pr), float(pc)
    if 0 < pr < P - 1 and 0 < pc < P - 1:
        sr += parabolic(patch[pr - 1, pc], patch[pr, pc], patch[pr + 1, pc])
        sc += parabolic(patch[pr, pc - 1], patch[pr, pc], patch[pr, pc + 1])
    return row + (sr - g) / u, col + (sc - g) / u
def correlation_shift(R, M, factor):
    H, W = R.shape
    C = R * np.conj(M)
    if factor > 2:
        r = np.fft.ifft2(C).real
        m = int(np.argmax(r)); row, col = divmod(m, W)
        ro = parabolic(r[(row - 1) % H, col], r[row, col], r[(row + 1) % H, col])
        co = parabolic(r[row, (col - 1) % W], r[row, col], r[row, (col + 1) % W])
        pr, pc = refine(C, (row + ro, col + co), factor)
        return [float(wrap(pr, H)), float(wrap(pc, W))]
    r = np.fft.ifft2(C).real
    m = int(np.argmax(r)); row, col = divmod(m, W)
    return [float(wrap(row, H)), float(wrap(col, W))]
def shift_spectrum(S, sh):
    H, W = S.shape
    fy = np.fft.fftfreq(H); fx = np.fft.fftfreq(W)
    return S * np.exp(-2j * np.pi * (fy[:, None] * sh[0] + fx[None, :] * sh[1]))

rng = np.random.default_rng(20261021)
corr = []
for (H, W) in [(8, 9), (12, 12), (9, 8)]:
    base = rng.uniform(0.5, 2, (H, W))
    # the running-average reference of three shifted spectra, then a moving image planted at a subpixel shift
    spectra = [shift_spectrum(np.fft.fft2(np.roll(base, (k, -k), (0, 1)) + 0.05 * rng.standard_normal((H, W))), (-k, k)) for k in range(3)]
    ref = sum(spectra) / 3
    cases = []
    for planted in [(1.3, -2.6), (-0.4, 0.7)]:
        mov = np.fft.fft2(np.real(np.fft.ifft2(shift_spectrum(np.fft.fft2(base), planted))) + 0.05 * rng.standard_normal((H, W)))
        for factor in [1, 2, 3, 8, 16]:
            cases.append(dict(factor=factor, planted=list(planted), moving=[flat(mov.real), flat(mov.imag)],
                              shift=correlation_shift(ref, mov, factor)))
    corr.append(dict(rows=H, columns=W, reference=[flat(ref.real), flat(ref.imag)], cases=cases))

# ---- transfer function (Float64 transcription of the app's ParallaxAberrationCorrection) ----
ab = json.load(open(FIX / 'aberrations.json'))
rows, columns, wl = ab['rows'], ab['columns'], ab['wavelength']
terms = ab['terms']
def transfer(coefs, low, high, order, sampling=1.25):
    kr = np.fft.fftfreq(rows)[:, None] / sampling; kc = np.fft.fftfreq(columns)[None, :] / sampling
    q = np.hypot(kr, kc); alpha = q * wl; theta = np.arctan2(kc, kr)
    even = np.zeros((rows, columns)); odd = np.zeros((rows, columns))
    for (m, n, a), c in zip(terms, coefs):
        ang = 1.0 if n == 0 else (np.cos(n * theta) if a == 0 else np.sin(n * theta))
        ph = alpha ** (m + 1) / (m + 1) * ang * c * 2 * np.pi / wl
        if m % 2 == 1: even = even + ph
        else: odd = odd + ph
    s = np.sin(even); sign = np.sign(s)
    env = 1 / (1 + (q / low) ** (2 * order)) * (1 - 1 / (1 + (q / high) ** (2 * order)))
    tr = sign * np.cos(odd) * env; ti = -sign * np.sin(odd) * env
    tr[0, 0] = 0; ti[0, 0] = 0
    return tr, ti, even, odd
tf = []
for c in ab['cases']:
    tr, ti, even, odd = transfer(c['coefs'], .3, .12, 2)
    img = np.array(ab['image']).reshape(rows, columns)
    out = np.fft.ifft2(np.fft.fft2(img) * (tr + 1j * ti)).real
    tf.append(dict(coefs=c['coefs'], real=flat(tr), imag=flat(ti), image=flat(out), minAbsSin=float(np.min(np.abs(np.sin(even)[np.abs(tr) + np.abs(ti) > 0])))))

# ---- the application's position-correction loop (Float32 arithmetic; Float64 deposition) ----
f32 = np.float32
def kde(images, shifts, dr, dc, factor, sigma):
    n, R, C = images.shape
    oR, oC = int(np.round(R * factor)), int(np.round(C * factor))
    cnt = np.zeros((oR, oC)); out = np.zeros((oR, oC))
    rr, cc = np.meshgrid(np.arange(R), np.arange(C), indexing='ij')
    for b in range(n):
        xr = (rr + dr.astype(float).reshape(R, C) + shifts[b, 0]) * factor
        xc = (cc + dc.astype(float).reshape(R, C) + shifts[b, 1]) * factor
        fr = np.floor(xr).astype(int); fc = np.floor(xc).astype(int); a = xr - fr; q = xc - fc
        v = images[b]
        for i, j, w in [(0, 0, (1 - a) * (1 - q)), (1, 0, a * (1 - q)), (0, 1, (1 - a) * q), (1, 1, a * q)]:
            np.add.at(cnt, ((fr + i) % oR, (fc + j) % oC), w)
            np.add.at(out, ((fr + i) % oR, (fc + j) % oC), w * v)
    s = sigma * factor
    cnt = gaussian_filter(cnt, s, mode='reflect', truncate=4.0); out = gaussian_filter(out, s, mode='reflect', truncate=4.0)
    return np.where(cnt > 1e-3, out / np.where(cnt > 1e-3, cnt, 1), 1.0)
def score_map(img, images32, sh32, dr, dc, factor, window):
    R, C = images32.shape[1:]
    oR, oC = img.shape
    rr = np.arange(R, dtype=f32)[:, None] * np.ones((1, C), f32); cc = np.arange(C, dtype=f32)[None, :] * np.ones((R, 1), f32)
    drr = dr.reshape(R, C); dcc = dc.reshape(R, C)
    score = np.zeros((R, C), f32)
    for b in range(images32.shape[0]):
        xr = ((rr + drr) + sh32[b, 0]) * factor; xc = ((cc + dcc) + sh32[b, 1]) * factor
        fr = np.floor(xr).astype(int); fc = np.floor(xc).astype(int)
        a = (xr - fr.astype(f32)).astype(f32); q = (xc - fc.astype(f32)).astype(f32)
        r0 = fr % oR; r1 = (fr + 1) % oR; c0 = fc % oC; c1 = (fc + 1) % oC
        one = f32(1)
        v = (((img[r0, c0] * (one - a)) * (one - q)) + ((img[r1, c0] * a) * (one - q))
             + ((img[r0, c1] * (one - a)) * q) + ((img[r1, c1] * a) * q)).astype(f32)
        score = (score + np.abs(v - images32[b])).astype(f32)
    score = (score / f32(images32.shape[0])).astype(f32)
    return (score * (window.reshape(R, C) if window is not None else f32(1))).astype(f32).ravel()
def fmean(x):
    return float(np.cumsum(x, dtype=f32)[-1] / f32(x.size))
def app_loop(images, shifts, window, factor, sigma, iters, checker, init_step, min_step, step_factor):
    n, R, C = images.shape; N = R * C
    sh32 = shifts.astype(f32); img32 = images.astype(f32); win = window.astype(f32).ravel()
    fac = f32(factor)
    dr = np.zeros(N, f32); dc = np.zeros(N, f32); step = np.full(N, f32(init_step), f32)
    mn = f32(min_step); sf = f32(step_factor)
    def deposit():
        return kde(images, shifts.astype(f32).astype(float), dr, dc, factor, sigma).astype(f32)
    img = deposit()
    scores = score_map(img, img32, sh32, dr, dc, fac, win)
    hist = [fmean(scores)]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)] if checker else [(-.5, 0), (.5, 0), (0, -.5), (0, .5)]
    for _ in range(iters):
        trials = [score_map(img, img32, sh32, (dr + f32(d[0]) * step).astype(f32), (dc + f32(d[1]) * step).astype(f32), fac, None) for d in dirs]
        update = np.zeros(N, bool)
        if checker:
            for i in range(N):
                best = 0; bs = f32(trials[0][i] * win[i])
                for k in range(1, 4):
                    v = f32(trials[k][i] * win[i])
                    if v < bs: best = k; bs = v
                if bs < scores[i]:
                    update[i] = True
                    dr[i] = f32(dr[i] + f32(f32(f32(dirs[best][0]) * step[i]) * win[i]))
                    dc[i] = f32(dc[i] + f32(f32(f32(dirs[best][1]) * step[i]) * win[i]))
        else:
            ddr = np.zeros(N, f32); ddc = np.zeros(N, f32)
            for i in range(N):
                gr = f32(trials[0][i] - trials[1][i]); gc = f32(trials[2][i] - trials[3][i])
                den = f32(np.hypot(gr, gc) / step[i])
                if np.isfinite(den) and den > 0:
                    ddr[i] = f32(f32(gr * win[i]) / den); ddc[i] = f32(f32(gc * win[i]) / den)
            fixed = score_map(img, img32, sh32, (dr + ddr).astype(f32), (dc + ddc).astype(f32), fac, win)
            for i in range(N):
                if fixed[i] < scores[i]:
                    update[i] = True; dr[i] = f32(dr[i] + ddr[i]); dc[i] = f32(dc[i] + ddc[i])
        for i in range(N):
            if not update[i]: step[i] = max(f32(step[i] * sf), mn)
        img = deposit()
        scores = score_map(img, img32, sh32, dr, dc, fac, win)
        hist.append(fmean(scores))
    return dict(dr=flat(dr), dc=flat(dc), scores=hist, image=flat(img.astype(float)))
pos = []
R, C, n = 7, 6, 5
rs = np.random.default_rng(7)
images = rs.uniform(0.5, 2, (n, R, C)); shifts = rs.uniform(-1, 1, (n, 2)).astype(f32).astype(float)
window = rs.uniform(0.2, 1, (R, C))
for name, kw in [('plain', dict(checker=False, init_step=1.0, min_step=0.1)),
                 ('checkerboard', dict(checker=True, init_step=1.0, min_step=0.1)),
                 ('floor-above-initial', dict(checker=False, init_step=0.05, min_step=0.1)),
                 ('checkerboard-floor', dict(checker=True, init_step=0.05, min_step=0.1))]:
    r = app_loop(images, shifts, window, 2.0, 0.3, 3, step_factor=0.75, **kw)
    pos.append(dict(name=name, iterations=3, factor=2.0, sigma=0.3, checkerboard=kw['checker'], initialStep=kw['init_step'],
                    minimumStep=kw['min_step'], stepFactor=0.75, **r))
flat_images = np.ones((3, 6, 6)); flat_shifts = np.zeros((3, 2)); flat_window = np.ones((6, 6))
r = app_loop(flat_images, flat_shifts, flat_window, 2.0, 0.3, 2, False, 1.0, 0.1, 0.75)
zero = dict(name='zero-gradient', iterations=2, factor=2.0, sigma=0.3, checkerboard=False, initialStep=1.0, minimumStep=0.1, stepFactor=0.75, **r)
out = dict(note='NumPy transcriptions of the source application (see tools/parallax/app_pins.py)',
           correlation=corr, transfer=tf,
           position=dict(rows=R, columns=C, count=n, images=flat(images), shifts=flat(shifts), window=flat(window), cases=pos,
                         flat=zero))
dest = FIX / 'appvariants.json'
dest.write_text(json.dumps(out, separators=(',', ':')) + '\n')
print(dest, dest.stat().st_size, 'min|sin(chi)| on active pixels', [t['minAbsSin'] for t in tf], 'flat offsets max', max(map(abs, zero['dr'] + zero['dc'])))
