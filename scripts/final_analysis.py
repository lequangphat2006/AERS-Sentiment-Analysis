"""Phan tich cuoi: AERS-Pro (V40, lan chay goc) vs PhoBERT v2 (lr 3e-5, 5 epoch). Chay: python final_analysis.py <w6_dir> <repo_dir> <out_dir>"""
import sys, json, numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
W, R, O = sys.argv[1:4]; SEEDS = [42, 7, 123, 2024, 99, 555, 2026, 31337, 8080, 12321]; cls = ['negative', 'neutral', 'positive']
df = pd.read_csv(f'{R}/input_data/dataset_cleaned.csv', encoding='utf-8-sig'); y = df.sentiment.map({c: i for i, c in enumerate(cls)}).values; g = df.near_dup_group_id.values; ia = np.arange(len(df))
PJ = json.load(open(f'{W}/results_phobert_v2.json'))['metrics_by_seed']; AJ = json.load(open(f'{W}/results_v40_multiseed.json'))['metrics_by_seed']
def calib(yt, p, nb=10):
    br = float(np.mean(np.sum((p-np.eye(3)[yt])**2, 1))); c = p.max(1); a = (p.argmax(1) == yt).astype(float); e = np.linspace(0, 1, nb+1); r = 0
    for i in range(nb):
        m = (c >= e[i]) & (c <= e[i+1]) if i == 0 else (c > e[i]) & (c <= e[i+1])
        if m.any(): r += m.mean()*abs(a[m].mean()-c[m].mean())
    return r, br
def nb(d, ratio):
    d = np.asarray(d, float); J = len(d); se = np.sqrt((1/J+ratio)*d.var(ddof=1)); h = stats.t.ppf(.975, J-1)*se
    return dict(mean=d.mean(), lo=d.mean()-h, hi=d.mean()+h, p=2*stats.t.sf(abs(d.mean()/se), J-1), p_naive=stats.ttest_1samp(d, 0).pvalue, dz=d.mean()/d.std(ddof=1))
def tost(d, m, ratio):
    d = np.asarray(d, float); J = len(d); se = np.sqrt((1/J+ratio)*d.var(ddof=1)); return max(stats.t.sf((d.mean()+m)/se, J-1), stats.t.cdf((d.mean()-m)/se, J-1))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); a = np.empty(len(p)); r = 0
    for i, j in enumerate(o): r = max(r, (len(p)-i)*p[j]); a[j] = min(1, r)
    return a
def fit_T(lg, yc): return float(np.exp(minimize_scalar(lambda z: -log_softmax(lg/np.exp(z), 1)[np.arange(len(yc)), yc].mean(), bounds=(-3, 3), method='bounded').x))
def platt(pc, yc, pe):
    f = lambda p: np.log(np.clip(p, 1e-6, 1-1e-6)/(1-np.clip(p, 1e-6, 1-1e-6))); m = LogisticRegression(max_iter=2000).fit(f(pc), yc); return m.predict_proba(f(pe))
NEG = {'không', 'chẳng', 'chả', 'đâu', 'chưa', 'k', 'ko', 'chớ', 'hok', 'hem'}; HED = ['tạm', 'tạm được', 'cũng được', 'bình thường', 'cũng ổn', 'tàm tạm', 'không đến nỗi', 'cũng tạm']
def tags(texts):
    r = []
    for t in texts:
        t = str(t).lower(); tk = t.split()
        r.append((any(w in NEG for w in tk), any(h in t for h in HED), any(m in t for m in ('nhưng', 'mà', 'tuy nhiên', 'ai ngờ', 'vậy mà', 'thế mà')), len(tk) < 5))
    return pd.DataFrame(r, columns=['phu_dinh', 'hedge', 'trai_chieu', 'cau_ngan'])
from sklearn.feature_extraction.text import TfidfVectorizer
import re
def norm(s): return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', s.lower())).strip()
rows = []; check = []; per = {}
for s in SEEDS:
    a_, te = next(iter(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(ia, y, g)))
    r_, c_ = next(iter(StratifiedGroupKFold(4, shuffle=True, random_state=s).split(a_, y[a_], g[a_]))); tr = a_[r_]; cal_idx = a_[c_]
    pt = pd.read_csv(f'{W}/phobert_logits/phobert_test_seed_{s}.csv'); pc = pd.read_csv(f'{W}/phobert_logits/phobert_calib_seed_{s}.csv')
    assert (pt.row_index.values == te).all() or set(pt.row_index) == set(te), 'test partition khac'
    order = np.argsort(pt.row_index.values); assert (np.sort(pt.row_index.values) == np.sort(te)).all() and (np.sort(pc.row_index.values) == np.sort(cal_idx)).all()
    Lt = pt[['logit_0', 'logit_1', 'logit_2']].values; yt_p = pt.y_true.values; assert (yt_p == y[pt.row_index.values]).all()
    Lc = pc[['logit_0', 'logit_1', 'logit_2']].values; ycal = pc.y_true.values
    av = pd.read_csv(f'{W}/v40_probs/v40_test_seed_{s}.csv', encoding='utf-8-sig'); Pa = av[['p_negative', 'p_neutral', 'p_positive']].values
    assert (av.text.astype(str).values == df.comment.values[te].astype(str)).all()   # V40 theo thu tu te
    # dua PhoBERT ve cung thu tu te
    pos = {r: i for i, r in enumerate(pt.row_index.values)}; idx = np.array([pos[r] for r in te]); Lt = Lt[idx]; yt = y[te]
    Pp = softmax(Lt, 1); fa, fp = f1_score(yt, Pa.argmax(1), average='macro'), f1_score(yt, Pp.argmax(1), average='macro')
    check.append(dict(seed=s, f1_phobert_logits=fp, f1_phobert_json=PJ[str(s)]['f1'], f1_aers_probs=fa, f1_aers_json=AJ[str(s)]['f1'], n_test=len(te), n_train=len(tr)))
    T = fit_T(Lc, ycal); Pts = softmax(Lt/T, 1); Ppl = platt(softmax(Lc, 1), ycal, Pp)
    ea, ba = calib(yt, Pa); ep, bp = calib(yt, Pp); et, bt = calib(yt, Pts); epl, bpl = calib(yt, Ppl)
    prfa = precision_recall_fscore_support(yt, Pa.argmax(1)); prfp = precision_recall_fscore_support(yt, Pp.argmax(1))
    row = dict(seed=s, T=T, f1_a=fa, f1_p=fp, acc_a=(Pa.argmax(1) == yt).mean(), acc_p=(Pp.argmax(1) == yt).mean(), ece_a=ea, ece_p=ep, ece_ts=et, ece_platt=epl, brier_a=ba, brier_p=bp, brier_ts=bt, brier_platt=bpl,
               f1_ts=f1_score(yt, Pts.argmax(1), average='macro'), f1_platt=f1_score(yt, Ppl.argmax(1), average='macro'), ratio=len(te)/len(tr))
    for i, c in enumerate(cls):
        for nm, prf in (('a', prfa), ('p', prfp)): row[f'P_{c}_{nm}'], row[f'R_{c}_{nm}'], row[f'F_{c}_{nm}'] = prf[0][i], prf[1][i], prf[2][i]
    for nb_ in (15, 20): row[f'ece{nb_}_a'] = calib(yt, Pa, nb_)[0]; row[f'ece{nb_}_p'] = calib(yt, Pp, nb_)[0]; row[f'ece{nb_}_ts'] = calib(yt, Pts, nb_)[0]
    # subgroup
    tg = tags(df.comment.values[te])
    for k in tg:
        m = tg[k].values; row[f'n_{k}'] = int(m.sum()); row[f'sf_a_{k}'] = f1_score(yt[m], Pa.argmax(1)[m], average='macro'); row[f'sf_p_{k}'] = f1_score(yt[m], Pp.argmax(1)[m], average='macro')
        for i, c in enumerate(cls): row[f'n_{k}_{c}'] = int((yt[m] == i).sum())
    # near-dup
    bp_ = AJ[str(s)]['best_params']; tv = TfidfVectorizer(max_features=bp_['tfidf_max_feat'], ngram_range=tuple(bp_['ngram_range']), sublinear_tf=bp_['sublinear_tf'])
    Xtr = tv.fit_transform([norm(t) for t in df.comment.values[tr]]); Xte = tv.transform([norm(t) for t in df.comment.values[te]])
    ms = np.concatenate([(Xte[i:i+1000] @ Xtr.T).max(1).toarray().ravel() for i in range(0, len(te), 1000)])
    for th in (0.99, 0.95, 0.90):
        k_ = ms < th; row[f'nd{th}_frac'] = 1-k_.mean(); row[f'nd{th}_f1a'] = f1_score(yt[k_], Pa.argmax(1)[k_], average='macro'); row[f'nd{th}_f1p'] = f1_score(yt[k_], Pp.argmax(1)[k_], average='macro')
        row[f'nd{th}_ecea'] = calib(yt[k_], Pa[k_])[0]; row[f'nd{th}_ecep'] = calib(yt[k_], Pp[k_])[0]
    rows.append(row); print(s, 'ok', flush=True)
D = pd.DataFrame(rows); D.to_csv(f'{O}/per_seed_final.csv', index=False); pd.DataFrame(check).to_csv(f'{O}/consistency_check.csv', index=False)
ck = pd.DataFrame(check); print('\nCHECK max|dF1| phobert logits vs json = %.2e ; aers probs vs json = %.2e' % ((ck.f1_phobert_logits-ck.f1_phobert_json).abs().max(), (ck.f1_aers_probs-ck.f1_aers_json).abs().max()))
ratio = float(D.ratio.mean()); print('mean n_test/n_train = %.4f' % ratio)
out = []
def rep(name, d):
    r = nb(d, ratio); r['name'] = name; out.append(r); return r
print('\n== MAIN (AERS - PhoBERT v2), N-B ==')
fam = [rep(m, (D[f'{m}_a']-D[f'{m}_p']).values) for m in ('f1', 'acc', 'ece', 'brier')]
for r, h in zip(fam, holm([r['p'] for r in fam])): print('%-6s mean %+.4f  NB-CI[%+.4f,%+.4f] p_NB=%.4f holm=%.4f | naive p=%.4f dz=%.2f' % (r['name'], r['mean'], r['lo'], r['hi'], r['p'], h, r['p_naive'], r['dz']))
d = (D.f1_a-D.f1_p).values; sd_p = D.f1_p.std(ddof=1); print('SD F1 phobert=%.4f aers=%.4f' % (sd_p, D.f1_a.std(ddof=1)), '| mean F1 aers %.4f phobert %.4f' % (D.f1_a.mean(), D.f1_p.mean()))
for m in (0.005, 0.01, 0.015, 0.02, 0.03): print('  TOST +-%.3f p_NB=%.4f' % (m, tost(d, m, ratio)))
print('wins (AERS>PhoBERT F1):', int((d > 0).sum()), '/10; Wilcoxon p=%.4f' % stats.wilcoxon(d).pvalue)
print('\n== Per class (AERS - PhoBERT), F1 / precision / recall; Holm over 3 classes (F1) ==')
pcl = [rep(f'F1_{c}', (D[f'F_{c}_a']-D[f'F_{c}_p']).values) for c in cls]
for c, r, h in zip(cls, pcl, holm([r['p'] for r in pcl])): print('%-9s F1 aers %.4f pho %.4f diff %+.4f NB-CI[%+.4f,%+.4f] p=%.4f holm=%.4f | R aers %.3f pho %.3f | P aers %.3f pho %.3f' % (c, D[f'F_{c}_a'].mean(), D[f'F_{c}_p'].mean(), r['mean'], r['lo'], r['hi'], r['p'], h, D[f'R_{c}_a'].mean(), D[f'R_{c}_p'].mean(), D[f'P_{c}_a'].mean(), D[f'P_{c}_p'].mean()))
print('\n== Calibration family (PhoBERT v2) vs AERS final ==  mean over seeds')
print('PhoBERT raw ECE %.4f Brier %.4f | +TS ECE %.4f Brier %.4f F1 %.4f (T mean %.2f) | +Platt3x3 ECE %.4f Brier %.4f F1 %.4f | AERS final ECE %.4f Brier %.4f' % (D.ece_p.mean(), D.brier_p.mean(), D.ece_ts.mean(), D.brier_ts.mean(), D.f1_ts.mean(), D['T'].mean(), D.ece_platt.mean(), D.brier_platt.mean(), D.f1_platt.mean(), D.ece_a.mean(), D.brier_a.mean()))
for nm, col_e, col_b in (('vs TS', 'ece_ts', 'brier_ts'), ('vs Platt', 'ece_platt', 'brier_platt')):
    re_, rb_ = rep('ECE '+nm, (D.ece_a-D[col_e]).values), rep('Brier '+nm, (D.brier_a-D[col_b]).values)
    print('AERS final %s: ECE diff %+.4f NB-CI[%+.4f,%+.4f] p=%.4f | Brier diff %+.4f NB-CI[%+.4f,%+.4f] p=%.4f' % (nm, re_['mean'], re_['lo'], re_['hi'], re_['p'], rb_['mean'], rb_['lo'], rb_['hi'], rb_['p']))
print('ECE by #bins (AERS final / PhoBERT raw / PhoBERT TS): 10: %.4f/%.4f/%.4f  15: %.4f/%.4f/%.4f  20: %.4f/%.4f/%.4f' % (D.ece_a.mean(), D.ece_p.mean(), D.ece_ts.mean(), D.ece15_a.mean(), D.ece15_p.mean(), D.ece15_ts.mean(), D.ece20_a.mean(), D.ece20_p.mean(), D.ece20_ts.mean()))
print('\n== Subgroups (AERS - PhoBERT F1), Holm over 4 ==')
sg = [rep('sub_'+k, (D[f'sf_a_{k}']-D[f'sf_p_{k}']).values) for k in ('phu_dinh', 'hedge', 'trai_chieu', 'cau_ngan')]
for k, r, h in zip(('phu_dinh', 'hedge', 'trai_chieu', 'cau_ngan'), sg, holm([r['p'] for r in sg])): print('%-10s n~%.0f dF1 %+.4f NB-CI[%+.4f,%+.4f] p=%.4f holm=%.4f (F1 aers %.3f pho %.3f)' % (k, D[f'n_{k}'].mean(), r['mean'], r['lo'], r['hi'], r['p'], h, D[f'sf_a_{k}'].mean(), D[f'sf_p_{k}'].mean()))
print('\n== Near-duplicate sensitivity (proxy TF-IDF space): F1 aers / pho / diff ; ECE aers / pho ==')
print('all: %.4f / %.4f / %+.4f ; ECE %.4f / %.4f' % (D.f1_a.mean(), D.f1_p.mean(), (D.f1_a-D.f1_p).mean(), D.ece_a.mean(), D.ece_p.mean()))
for th in (0.99, 0.95, 0.90): print('drop sim>=%.2f (%.1f%% removed): %.4f / %.4f / %+.4f ; ECE %.4f / %.4f' % (th, 100*D[f'nd{th}_frac'].mean(), D[f'nd{th}_f1a'].mean(), D[f'nd{th}_f1p'].mean(), (D[f'nd{th}_f1a']-D[f'nd{th}_f1p']).mean(), D[f'nd{th}_ecea'].mean(), D[f'nd{th}_ecep'].mean()))
pd.DataFrame(out).to_csv(f'{O}/tests_final.csv', index=False)
