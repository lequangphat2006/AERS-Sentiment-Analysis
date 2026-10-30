"""Thong ke hieu chinh Nadeau-Bengio, TOST, Holm cho AERS-Pro vs PhoBERT.
Doc truc tiep results_*_multiseed.json (nguon duy nhat). Chay: python stats.py <repo_dir>"""
import json, sys, numpy as np
from scipy import stats
R = sys.argv[1] if len(sys.argv) > 1 else '.'
A = json.load(open(f'{R}/results_v40_multiseed.json'))['metrics_by_seed']
P = json.load(open(f'{R}/results_phobert_multiseed.json'))['metrics_by_seed']
seeds = sorted(A, key=int)
N_TR, N_TE = 14849, 5315   # seed 42 (xem split_audit); ty le ~0.358
def nb(d, ratio=N_TE/N_TR, alpha=0.05):
    d = np.asarray(d, float); J = len(d)
    se = np.sqrt((1/J + ratio) * d.var(ddof=1)); se0 = d.std(ddof=1)/np.sqrt(J)
    t = d.mean()/se; p = 2*stats.t.sf(abs(t), J-1); h = stats.t.ppf(1-alpha/2, J-1)*se
    return dict(mean=d.mean(), se_naive=se0, se_nb=se, p=p, lo=d.mean()-h, hi=d.mean()+h)
def tost(d, margin, ratio=N_TE/N_TR):
    d = np.asarray(d, float); J = len(d); se = np.sqrt((1/J+ratio)*d.var(ddof=1))
    return max(stats.t.sf((d.mean()+margin)/se, J-1), stats.t.cdf((d.mean()-margin)/se, J-1))
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m-r)*p[i]); adj[i] = min(1, run)
    return adj
out = {}
print(f'{"metric":9s} {"mean diff":>10s} {"naive CI":>20s} {"p_naive":>8s} {"NB CI":>20s} {"p_NB":>8s} {"p_NB_holm":>9s}')
ms = ['f1', 'accuracy', 'ece', 'brier']; rows = []
for m in ms:
    d = np.array([A[s][m]-P[s][m] for s in seeds])
    r = nb(d, 0.0); rn = nb(d)
    pn = stats.ttest_1samp(d, 0).pvalue
    rows.append((m, d, r, rn, pn))
ph = holm([x[3]['p'] for x in rows])
for (m, d, r, rn, pn), h in zip(rows, ph):
    print(f'{m:9s} {d.mean():+10.4f} [{r["lo"]:+.4f},{r["hi"]:+.4f}] {pn:8.4f} [{rn["lo"]:+.4f},{rn["hi"]:+.4f}] {rn["p"]:8.4f} {h:9.4f}')
    out[m] = dict(mean=d.mean(), naive_ci=[r['lo'], r['hi']], p_naive=pn, nb_ci=[rn['lo'], rn['hi']], p_nb=rn['p'], p_nb_holm=h, sd=d.std(ddof=1))
d = np.array([A[s]['f1']-P[s]['f1'] for s in seeds])
print('\nTOST F1-macro (hieu chinh N-B) vs bien:')
out['tost_f1'] = {}
for mg in (0.005, 0.01, 0.015, 0.02):
    pn, pb = tost(d, mg, 0.0), tost(d, mg)
    print(f'  +-{mg:.3f}: p_naive={pn:.4f}  p_NB={pb:.4f}  ->', 'tuong duong' if pb < .05 else 'KHONG dat')
    out['tost_f1'][mg] = dict(p_naive=pn, p_nb=pb)
print('\nSD F1 theo seed: AERS=%.4f PhoBERT=%.4f' % (np.std([A[s]['f1'] for s in seeds], ddof=1), np.std([P[s]['f1'] for s in seeds], ddof=1)))
print('Wilcoxon F1: p=%.4f' % stats.wilcoxon(d).pvalue)
print('So seed giu calibrator:', sum(A[s]['calibrator_applied'] for s in seeds), '/10', [s for s in seeds if A[s]['calibrator_applied']])
print('mean(ECE) AERS=%.4f PhoBERT=%.4f' % (np.mean([A[s]['ece'] for s in seeds]), np.mean([P[s]['ece'] for s in seeds])))
json.dump(out, open('stats_out.json', 'w'), indent=1, default=float)
