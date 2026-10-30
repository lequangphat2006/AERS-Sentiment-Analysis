import sys
sys.path.insert(0, 'src')
import pandas as pd
from seeds_config import SEEDS

df = pd.read_csv('data/dataset_cleaned.csv', encoding='utf-8-sig')
assert not df['comment'].duplicated().any(), 'Corpus co comment trung nhau'
pos = {t: i for i, t in enumerate(df['comment'])}
lab = df['sentiment'].values
for s in SEEDS:
    f = f'results/v40_probs/v40_test_seed_{s}.csv'
    p = pd.read_csv(f, encoding='utf-8-sig')
    if 'text' not in p.columns:
        print(s, 'da xu ly, bo qua'); continue
    miss = [t for t in p['text'] if t not in pos]
    assert not miss, f'seed {s}: {len(miss)} dong khong khop van ban goc'
    ridx = p['text'].map(pos).values
    assert (lab[ridx] == p['y_true'].values).all(), f'seed {s}: nhan khong khop'
    ph = pd.read_csv(f'results/phobert_logits/phobert_test_seed_{s}.csv')
    assert set(ridx) == set(ph['row_index']), f'seed {s}: tap test khac PhoBERT'
    p.insert(0, 'row_index', ridx)
    p.drop(columns=['text']).to_csv(f, index=False)
    print(s, 'OK', len(p))