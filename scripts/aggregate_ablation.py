"""Tong hop bang ablation tu <dir>/<variant>_seed<seed>.npz. python aggregate_ablation.py <dir> <out_csv>
Voi >=3 seed, them hieu so voi 'full' (bien the 'final') kem CI N-B."""
import sys, glob, re, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
d, outp = sys.argv[1], sys.argv[2]
def cal(y, p, nb=10):
    br = float(np.mean(np.sum((p-np.eye(3)[y])**2, 1))); c = p.max(1); a = (p.argmax(1) == y).astype(float); e = np.linspace(0, 1, nb+1); r = 0
    for i in range(nb):
        m = (c >= e[i]) & (c <= e[i+1]) if i == 0 else (c > e[i]) & (c <= e[i+1])
        if m.any(): r += m.mean()*abs(a[m].mean()-c[m].mean())
    return r, br
def ts_probs(pc, yc, pe):   # temperature scaling tren log-xac-suat (fit tren cal)
    from scipy.optimize import minimize_scalar
    from scipy.special import log_softmax
    lc, le = np.log(np.clip(pc, 1e-12, 1)), np.log(np.clip(pe, 1e-12, 1))
    t = np.exp(minimize_scalar(lambda z: -log_softmax(lc/np.exp(z), 1)[np.arange(len(yc)), yc].mean(), bounds=(-3, 3), method='bounded').x)
    return np.exp(log_softmax(le/t, 1))
rows = []
for f in sorted(glob.glob(f'{d}/*_seed*.npz')):
    var, seed = re.match(r'(.+)_seed(\d+)\.npz', f.split('/')[-1]).groups(); z = np.load(f); y = z['y_te']; yc = z['y_cal']
    sets = {'LR (base, fused feats)': z['base_lr'], 'LinearSVC (base)': z['base_svm'], 'LightGBM (base)': z['base_lgbm'], 'avg of 3 base': z['avg3'],
            'stacking raw': z['meta_raw'], 'stacking + Platt3x3 (always)': z['meta_platt'], 'stacking + TS': ts_probs(z['meta_raw_cal'], yc, z['meta_raw']), 'final (conditional rule)': z['final']}
    for k, p in sets.items():
        e, b = cal(y, p); rows.append(dict(variant=var, seed=int(seed), model=k, f1=f1_score(y, p.argmax(1), average='macro'), acc=(p.argmax(1) == y).mean(), ece=e, brier=b,
                                          neutral_f1=f1_score(y, p.argmax(1), average=None)[1], kept=bool(z['keep'])))
R = pd.DataFrame(rows); R.to_csv(outp.replace('.csv', '_per_seed.csv'), index=False)
S = R.groupby(['variant', 'model'])[['f1', 'acc', 'ece', 'brier', 'neutral_f1']].mean().round(4); S['n_seeds'] = R.groupby(['variant', 'model']).seed.nunique()
S.to_csv(outp); print(S.to_string())
