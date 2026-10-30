"""Audit phan chia train/cal/test (copy logic split_data_v40). Chay duoi tung phien ban sklearn.
python split_audit.py <repo_dir> <out_dir>"""
import sys, hashlib, numpy as np, pandas as pd, sklearn
from sklearn.model_selection import StratifiedGroupKFold
R, O = sys.argv[1], sys.argv[2]
SEEDS = [42, 7, 123, 2024, 99, 555, 2026, 31337, 8080, 12321]
df = pd.read_csv(f'{R}/input_data/dataset_cleaned.csv', encoding='utf-8-sig')
cls = sorted(df.sentiment.unique()); y = df.sentiment.map({c: i for i, c in enumerate(cls)}).values
g = df.near_dup_group_id.values; ia = np.arange(len(df))
def split(seed):
    a, te = next(iter(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(ia, y, g)))
    r, c = next(iter(StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(a, y[a], g[a])))
    return a[r], a[c], te
rows = []
for s in SEEDS:
    parts = dict(zip(['train', 'cal', 'test'], split(s)))
    for k, idx in parts.items():
        pr = np.bincount(y[idx], minlength=3)/len(idx)
        rows.append(dict(seed=s, part=k, n=len(idx), n_groups=len(set(g[idx])), **{cls[i]: round(pr[i], 4) for i in range(3)},
                         sha256=hashlib.sha256(np.sort(idx).astype('<i8').tobytes()).hexdigest()[:16]))
    assert not (set(g[parts['train']]) & set(g[parts['test']]))
out = pd.DataFrame(rows); out.to_csv(f'{O}/split_audit_sklearn_{sklearn.__version__}.csv', index=False)
print(sklearn.__version__, 'corpus', len(df), dict(zip(cls, np.round(np.bincount(y)/len(y), 4))))
print(out[out.part == 'test'][['seed', 'n', 'n_groups'] + cls].to_string(index=False))
