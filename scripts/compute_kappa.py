"""Tính độ đồng thuận giữa người gán nhãn 2 và nhãn gốc (T01).

    python compute_kappa.py --key label_round2/annotation_key_PRIVATE.csv \
                            --ann label_round2/annotation_done.csv --out label_round2/agreement

File --ann: cột item_id, label (negative/neutral/positive; chấp nhận cả 0/1/2 = neg/neu/pos). Cột label rỗng bị bỏ qua.
Xuất: agreement_summary.json, confusion_matrix.csv, per_class.csv, disagreements.csv (để họp xử lý bất đồng).
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

LAB = ['negative', 'neutral', 'positive']
MAP = {'0': 'negative', '1': 'neutral', '2': 'positive', 'neg': 'negative', 'neu': 'neutral', 'pos': 'positive',
       'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'}

def boot_ci(y1, y2, fn, n=2000, seed=0):
    rng = np.random.RandomState(seed); idx = np.arange(len(y1)); v = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        try: v.append(fn(y1[s], y2[s]))
        except Exception: pass
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

def k(y1, y2, **kw): return cohen_kappa_score(y1, y2, labels=LAB, **kw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True); ap.add_argument('--ann', required=True); ap.add_argument('--out', default='agreement')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    key = pd.read_csv(a.key); ann = pd.read_csv(a.ann)
    ann['label'] = ann['label'].astype(str).str.strip().str.lower().map(MAP)
    d = key.merge(ann[['item_id', 'label']], on='item_id', how='left')
    miss = int(d.label.isna().sum()); d = d.dropna(subset=['label'])
    y1, y2 = d.label_orig.values, d.label.values
    res = {'n_labelled': int(len(d)), 'n_missing_or_invalid': miss,
           'raw_agreement': float((y1 == y2).mean()),
           'kappa': float(k(y1, y2)), 'kappa_ci95_bootstrap': boot_ci(y1, y2, k),
           'kappa_linear_weighted': float(k(y1, y2, weights='linear')),
           'kappa_quadratic_weighted': float(k(y1, y2, weights='quadratic'))}
    res['by_stratum'] = {s: {'n': int(len(g)), 'agreement': float((g.label_orig == g.label).mean()),
                             'kappa': float(k(g.label_orig.values, g.label.values))} for s, g in d.groupby('stratum')}
    rows = []
    for c in LAB:   # một-đối-tất-cả: κ nhị phân từng lớp + tỷ lệ đồng ý theo lớp gốc
        b1, b2 = (y1 == c).astype(int), (y2 == c).astype(int)
        g = d[d.label_orig == c]
        rows.append({'class': c, 'n_orig': int(len(g)), 'agree_rate_given_orig': float((g.label == c).mean()),
                     'kappa_one_vs_rest': float(cohen_kappa_score(b1, b2)), 'n_annotator2': int((y2 == c).sum())})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, 'per_class.csv'), index=False)
    cm = pd.DataFrame(confusion_matrix(y1, y2, labels=LAB), index=['orig_' + c for c in LAB], columns=['ann2_' + c for c in LAB])
    cm.to_csv(os.path.join(a.out, 'confusion_matrix.csv'))
    d[d.label_orig != d.label].merge(pd.read_csv(a.key)[['item_id']], on='item_id') \
        .to_csv(os.path.join(a.out, 'disagreements.csv'), index=False)
    json.dump(res, open(os.path.join(a.out, 'agreement_summary.json'), 'w'), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False)); print(cm); print(pd.DataFrame(rows))

if __name__ == '__main__':
    main()
