"""Holm-Bonferroni theo từng 'family' từ bảng Nadeau-Bengio đã có.

Chạy từ thư mục gốc repo:
    py scripts\\holm_from_tables.py --csv tables/tests_nadeau_bengio_8ep.csv --out tables/tests_nadeau_bengio_8ep_holm.csv
"""
import argparse
import numpy as np, pandas as pd

FAMILIES = {
    'primary (F1, Acc, ECE, Brier)': ['f1', 'acc', 'ece', 'brier'],
    'per-class F1': ['F1_negative', 'F1_neutral', 'F1_positive'],
    'vs calibrated PhoBERT': ['ECE vs TS', 'Brier vs TS', 'ECE vs Platt', 'Brier vs Platt'],
    'subgroups': ['sub_phu_dinh', 'sub_hedge', 'sub_trai_chieu', 'sub_cau_ngan'],
}

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj

ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='tables/tests_nadeau_bengio_8ep.csv')
ap.add_argument('--out', default='tables/tests_nadeau_bengio_8ep_holm.csv')
a = ap.parse_args()
df = pd.read_csv(a.csv).set_index('name')
df['family'] = ''; df['p_holm'] = np.nan
for fam, names in FAMILIES.items():
    names = [n for n in names if n in df.index]
    if not names: continue
    df.loc[names, 'p_holm'] = holm(df.loc[names, 'p'].values)
    df.loc[names, 'family'] = fam
df = df.reset_index()
pd.set_option('display.width', 200)
print(df[['name', 'mean', 'lo', 'hi', 'p', 'p_holm', 'family']].round(5).to_string(index=False))
df.to_csv(a.out, index=False)
print('Đã lưu', a.out)
