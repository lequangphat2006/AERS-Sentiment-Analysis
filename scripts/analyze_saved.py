"""Phan tich tu xac suat AERS-Pro da luu (reports/v40_probs) + JSON PhoBERT. Chay duoi sklearn 1.6.1.
python analyze_saved.py <repo_dir> <out_dir>"""
import sys, json, re, numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0, sys.argv[1]); R, O = sys.argv[1], sys.argv[2]
SEEDS = [42, 7, 123, 2024, 99, 555, 2026, 31337, 8080, 12321]
df = pd.read_csv(f'{R}/input_data/dataset_cleaned.csv', encoding='utf-8-sig')
cls = ['negative', 'neutral', 'positive']; y = df.sentiment.map({c: i for i, c in enumerate(cls)}).values
g = df.near_dup_group_id.values; ia = np.arange(len(df))
A = json.load(open(f'{R}/results_v40_multiseed.json'))['metrics_by_seed']
P = json.load(open(f'{R}/results_phobert_multiseed.json'))['metrics_by_seed']
# --- ham dung chung (copy tu train_v40.compute_calibration_metrics, n_bins tham so)
def calib(yt, p, nb=10):
    br = float(np.mean(np.sum((p-np.eye(3)[yt])**2, 1))); conf = p.max(1); acc = (p.argmax(1) == yt).astype(float)
    e = np.linspace(0, 1, nb+1); ece = 0
    for i in range(nb):
        m = (conf >= e[i]) & (conf <= e[i+1]) if i == 0 else (conf > e[i]) & (conf <= e[i+1])
        if m.any(): ece += m.mean()*abs(acc[m].mean()-conf[m].mean())
    return ece, br
def nbci(d, ratio=0.358):
    d = np.asarray(d, float); J = len(d); se = np.sqrt((1/J+ratio)*d.var(ddof=1)); h = stats.t.ppf(.975, J-1)*se
    return d.mean(), d.mean()-h, d.mean()+h, 2*stats.t.sf(abs(d.mean()/se), J-1)
# --- subgroup tags (finish_analysis.py)
NEG = ["không", "ko", "k", "chẳng", "chả", "chưa", "đâu", "hem", "hok", "chớ"]
CON = ["nhưng", "tuy nhiên", "mà", "thế mà", "vậy mà", "ai ngờ"]
HED = ["hơi", "khá", "có lẽ", "có thể", "hình như", "dường như", "chắc là"]
def tags(texts):
    t = pd.Series(texts).fillna('').astype(str).str.lower()
    def has(w):
        pat = r"(?<!\w)(?:"+"|".join(x.replace(' ', r'\s+') for x in sorted(w, key=len, reverse=True))+r")(?!\w)"
        return t.str.contains(pat, regex=True).to_numpy()
    sub_con = t.str.contains("|".join(re.escape(w) for w in CON), regex=True).to_numpy()  # substring: khop JSON PhoBERT chinh xac
    return {'phu_dinh': has(NEG), 'hedge': has(HED), 'trai_chieu': sub_con, 'cau_ngan': t.str.split().apply(len).to_numpy() <= 4}
def norm(s): return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', s.lower())).strip()
res = []; cm_total = None; sub_rows = []; nd_rows = []; ece_rows = []
for s in SEEDS:
    a, te = next(iter(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(ia, y, g)))
    r, c = next(iter(StratifiedGroupKFold(4, shuffle=True, random_state=s).split(a, y[a], g[a]))); tr = a[r]
    p = pd.read_csv(f'{R}/reports/v40_probs/v40_test_seed_{s}.csv', encoding='utf-8-sig')
    # khop thu tu hang voi te
    assert (p.text.astype(str).values == df.comment.values[te].astype(str)).all(), 'thu tu khac'
    yt = y[te]; pr = p[['p_negative', 'p_neutral', 'p_positive']].values; pred = pr.argmax(1)
    f1 = f1_score(yt, pred, average='macro'); assert abs(f1-A[str(s)]['f1']) < 1e-9, (s, f1, A[str(s)]['f1'])
    e, b = calib(yt, pr); assert abs(e-A[str(s)]['ece']) < 1e-9 and abs(b-A[str(s)]['brier']) < 1e-9
    if s == 42: cm42 = confusion_matrix(yt, pred)
    for nb_ in (10, 15, 20): ece_rows.append(dict(seed=s, bins=nb_, ece_aers=calib(yt, pr, nb_)[0]))
    # tag counts + AERS subgroup F1
    tg = tags(df.comment.values[te]); sg = P[str(s)]['subgroup_results']
    for k, m in tg.items():
        sub_rows.append(dict(seed=s, group=k, n=int(m.sum()), n_phobert_json=sg[k]['n'], f1_aers=f1_score(yt[m], pred[m], average='macro'),
                             f1_pho=sg[k]['f1_macro'], **{f'n_{cls[i]}': int((yt[m] == i).sum()) for i in range(3)}))
    # near-duplicate sensitivity (cosine tren TF-IDF tu da chuan hoa, fit tren train)
    bp = A[str(s)]['best_params']; tv = TfidfVectorizer(max_features=bp['tfidf_max_feat'], ngram_range=tuple(bp['ngram_range']), sublinear_tf=bp['sublinear_tf']); Xtr = tv.fit_transform([norm(t) for t in df.comment.values[tr]])
    Xte = tv.transform([norm(t) for t in df.comment.values[te]]); ms = np.zeros(len(te))
    for i in range(0, len(te), 1000): ms[i:i+1000] = (Xte[i:i+1000] @ Xtr.T).max(1).toarray().ravel()
    for th in (0.0, 0.99, 0.95, 0.90):
        keep = ms < th if th > 0 else np.ones(len(te), bool)
        e2, b2 = calib(yt[keep], pr[keep])
        nd_rows.append(dict(seed=s, thr=th, n_keep=int(keep.sum()), frac_removed=1-keep.mean(), f1=f1_score(yt[keep], pred[keep], average='macro'),
                            acc=(pred[keep] == yt[keep]).mean(), ece=e2, brier=b2,
                            f1_removed=(f1_score(yt[~keep], pred[~keep], average='macro') if (~keep).sum() > 30 else np.nan)))
    print('seed', s, 'ok', flush=True)
pd.DataFrame(sub_rows).to_csv(f'{O}/subgroup_audit.csv', index=False); nd = pd.DataFrame(nd_rows); nd.to_csv(f'{O}/neardup_sensitivity.csv', index=False)
pd.DataFrame(ece_rows).to_csv(f'{O}/ece_bins_aers.csv', index=False)
print('\nConfusion seed 42 (rows=true neg/neu/pos):\n', cm42)
print('\nNear-dup sensitivity (AERS-Pro, mean over 10 seeds):')
print(nd.groupby('thr')[['frac_removed', 'n_keep', 'f1', 'acc', 'ece', 'brier']].mean().round(4).to_string())
base = nd[nd.thr == 0].set_index('seed')
for th in (0.99, 0.95, 0.90):
    d = (nd[nd.thr == th].set_index('seed').f1-base.f1).values; print(f'  dF1(thr<{th}) mean={d.mean():+.4f} sd={d.std(ddof=1):.4f}')
sub = pd.DataFrame(sub_rows)
print('\nTag counts khop JSON PhoBERT theo nhom:', {k: int((x.n == x.n_phobert_json).sum()) for k, x in sub.groupby('group')}, '(so seed khop /10)')
print('\nSubgroup dF1 (AERS-PhoBERT), N-B CI, Holm over 4 groups:')
ps = []; rr = []
for k in ('phu_dinh', 'hedge', 'trai_chieu', 'cau_ngan'):
    x = sub[sub.group == k]; d = (x.f1_aers.values-x.f1_pho.values); m, lo, hi, pv = nbci(d); ps.append(pv); rr.append((k, x.n.mean(), x.n.min(), x.n.max(), m, lo, hi, pv))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for i, j in enumerate(o): run = max(run, (len(p)-i)*p[j]); adj[j] = min(1, run)
    return adj
for t, h in zip(rr, holm(ps)): print(f'  {t[0]:11s} n~{t[1]:.0f} [{t[2]},{t[3]}] dF1={t[4]:+.4f} NB-CI[{t[5]:+.4f},{t[6]:+.4f}] p_NB={t[7]:.4f} holm={h:.4f}')
x = sub[sub.group == 'cau_ngan']; print('  cau_ngan class counts (mean/seed):', x[['n_negative', 'n_neutral', 'n_positive']].mean().round(1).to_dict(), ' share of test=%.3f' % (x.n.mean()/np.mean([len(pd.read_csv(f'{R}/reports/v40_probs/v40_test_seed_{s}.csv', encoding="utf-8-sig")) for s in SEEDS])))
er = pd.DataFrame(ece_rows).groupby('bins').ece_aers.mean(); print('\nAERS ECE theo so bin:', er.round(4).to_dict())
