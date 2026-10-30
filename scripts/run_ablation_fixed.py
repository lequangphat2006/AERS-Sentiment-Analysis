"""Ablation/baseline voi best_params CO DINH (khong chay lai Optuna), cung phan chia nhu bai (sklearn 1.6.1).
Moi (seed, bien the dac trung) luu xac suat cal/test cua: lr, svm, lgbm, avg3, meta_raw, meta_calibrated(luon ap), final(theo quy tac giu/bo).
Chay tu thu muc repo:  python run_ablation_fixed.py <cache.pkl> <out_dir> <seeds csv> <variants csv>
variants: full,no_pos,no_lex,no_both,no_cw"""
import sys, os, json, time, joblib, numpy as np, pandas as pd
sys.path.insert(0, '.')
import train_v40 as T
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
cache_p, out_dir, seeds_s, var_s = sys.argv[1:5]; os.makedirs(out_dir, exist_ok=True)
C = joblib.load(cache_p); df = pd.read_csv('input_data/dataset_cleaned.csv', encoding='utf-8-sig')
raw = df.comment.astype(str).tolist(); clean = C['clean']; manual_all = C['manual']; pos_dicts = C['pos'][0]
le = T.LabelEncoder(); y = le.fit_transform(df.sentiment); g = df.near_dup_group_id.values
BP = json.load(open('results_v40_multiseed.json'))['metrics_by_seed']
class Pool(T.BaseLearnersPool):
    def __init__(self, *a, cw='balanced', **k): super().__init__(*a, **k); self.cw = cw
    def build_fresh_learner(self, name, yf, gf=None):
        if name == 'lr': return LogisticRegression(C=self.lr_C, class_weight=self.cw, max_iter=1000, random_state=self.seed)
        if name == 'lgbm': return LGBMClassifier(class_weight=self.cw, random_state=self.seed, verbose=-1)
        base = LinearSVC(C=self.svm_C, class_weight=self.cw, random_state=self.seed, max_iter=2000)
        folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=self.seed).split(np.zeros(len(yf)), yf, gf))
        return CalibratedClassifierCV(base, method='sigmoid', cv=folds)
def run(seed, var):
    t0 = time.time(); bp = BP[str(seed)]['best_params']; ng = tuple(bp['ngram_range'])
    use_pos = var not in ('no_pos', 'no_both'); use_lex = var not in ('no_lex', 'no_both'); cw = None if var == 'no_cw' else 'balanced'
    itr, ical, ite = T.split_data_v40(df, y, 'near_dup_group_id', seed)
    sl = lambda ix: ([raw[i] for i in ix], [clean[i] for i in ix])
    (r_tr, c_tr), (r_cal, c_cal), (r_te, c_te) = sl(itr), sl(ical), sl(ite); y_tr, y_cal, y_te = y[itr], y[ical], y[ite]; g_tr = g[itr]
    pos = (lambda ix: ([pos_dicts[i] for i in ix], [])) if use_pos else (lambda ix: None)
    m_tr, m_cal, m_te = (manual_all[i].copy() for i in (itr, ical, ite)); ipmn, ineu = T.TextPreprocessor.IDX_POS_MINUS_NEG, T.TextPreprocessor.IDX_IS_NEUTRAL
    if use_lex:
        bT, _ = T.tune_neutral_threshold(m_cal[:, ipmn], y_cal)
        for m in (m_tr, m_cal, m_te): m[:, ineu] = (np.abs(m[:, ipmn]) < bT).astype(np.float32)
    else:
        bT = float('nan')
        for m in (m_tr, m_cal, m_te): m[:] = 0.0
    mk = lambda: T.FeaturePipeline(tfidf_max_features=bp['tfidf_max_feat'], tfidf_ngram_range=ng, hashing_n_features=bp['hash_features'], use_pos=use_pos, use_hashing=True, sublinear_tf=bp['sublinear_tf'])
    pipe = mk(); X_tr = pipe.fit_transform(r_tr, c_tr, y_tr, pos_cache=pos(itr), manual_features=m_tr)
    X_cal = pipe.transform(r_cal, c_cal, pos_cache=pos(ical), manual_features=m_cal); X_te = pipe.transform(r_te, c_te, pos_cache=pos(ite), manual_features=m_te)
    cl = T.MetaClusterer(4, seed); oh_tr = cl.to_onehot(cl.fit_predict(T.MetaFeatureExtractor.extract_batch(r_tr)))
    oh_cal = cl.to_onehot(cl.transform(T.MetaFeatureExtractor.extract_batch(r_cal))); oh_te = cl.to_onehot(cl.transform(T.MetaFeatureExtractor.extract_batch(r_te)))
    pool = Pool(lr_C=bp['lr_C'], seed=seed, cw=cw); names = pool.learner_names
    folds = list(StratifiedGroupKFold(n_splits=T.N_OOF_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(y_tr)), y_tr, g_tr))
    oof = {n: np.zeros((len(y_tr), 3)) for n in names}
    for tr_i, va_i in folds:
        pf = mk(); Xf = pf.fit_transform([r_tr[i] for i in tr_i], [c_tr[i] for i in tr_i], y_tr[tr_i], pos_cache=(T._slice_pos_cache(pos(itr), tr_i) if use_pos else None), manual_features=m_tr[tr_i])
        Xv = pf.transform([r_tr[i] for i in va_i], [c_tr[i] for i in va_i], pos_cache=(T._slice_pos_cache(pos(itr), va_i) if use_pos else None), manual_features=m_tr[va_i])
        for n in names:
            mdl = pool.build_fresh_learner(n, y_tr[tr_i], g_tr[tr_i]); mdl.fit(Xf, y_tr[tr_i]); oof[n][va_i] = mdl.predict_proba(Xv)
    pool.fit(X_tr, y_tr, groups=g_tr); pc, pt = pool.predict_proba_dict(X_cal), pool.predict_proba_dict(X_te)
    mt = np.hstack([oof[n] for n in names] + [oh_tr, pipe.scale_manual_features(m_tr)])
    mc = np.hstack([pc[n] for n in names] + [oh_cal, pipe.scale_manual_features(m_cal)]); me = np.hstack([pt[n] for n in names] + [oh_te, pipe.scale_manual_features(m_te)])
    best_c, bf = None, -1
    for Cc in [0.01, 0.1, 1.0, 5.0, 10.0]:
        mm = T.StackingMetaLearner(C=Cc, seed=seed); mm.fit(mt, y_tr); f = f1_score(y_cal, mm.predict_proba(mc).argmax(1), average='macro')
        if f > bf: bf, best_c = f, Cc
    meta = T.StackingMetaLearner(C=best_c, seed=seed)
    if cw is None: meta.model.set_params(class_weight=None)
    meta.fit(mt, y_tr); raw_c, raw_e = meta.predict_proba(mc), meta.predict_proba(me)
    cal = T.ProbabilityCalibrator(seed=seed).fit(raw_c, y_cal); cal_c, cal_e = cal.predict_proba(raw_c), cal.predict_proba(raw_e)
    mr, mc2 = T.compute_calibration_metrics(y_cal, raw_c), T.compute_calibration_metrics(y_cal, cal_c)
    keep = mc2['ece'] < mr['ece'] and f1_score(y_cal, cal_c.argmax(1), average='macro') >= f1_score(y_cal, raw_c.argmax(1), average='macro') - 0.005
    out = dict(y_te=y_te, idx_te=ite, y_cal=y_cal, keep=np.array(keep), best_c=np.array(best_c), T=np.array(bT), meta_raw_cal=raw_c, meta_raw=raw_e, meta_platt=cal_e, meta_platt_cal=cal_c,
               final=(cal_e if keep else raw_e), avg3=np.mean([pt[n] for n in names], 0), avg3_cal=np.mean([pc[n] for n in names], 0), secs=np.array(time.time()-t0), **{f'base_{n}': pt[n] for n in names}, **{f'base_{n}_cal': pc[n] for n in names})
    np.savez_compressed(f'{out_dir}/{var}_seed{seed}.npz', **out)
    print(seed, var, 'F1 final=%.4f raw=%.4f lr=%.4f avg3=%.4f' % (f1_score(y_te, out['final'].argmax(1), average='macro'), f1_score(y_te, raw_e.argmax(1), average='macro'),
          f1_score(y_te, pt['lr'].argmax(1), average='macro'), f1_score(y_te, out['avg3'].argmax(1), average='macro')), 'json=%.4f' % BP[str(seed)]['f1'], 'keep', keep, '%.0fs' % (time.time()-t0), flush=True)
for s in map(int, seeds_s.split(',')):
    for v in var_s.split(','):
        if not os.path.exists(f'{out_dir}/{v}_seed{s}.npz'): run(s, v)
