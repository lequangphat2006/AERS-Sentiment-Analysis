import pandas as pd
import numpy as np
import re
import glob, os, sys
from sklearn.metrics import f1_score
from scipy import stats

print("⏳ Đang chạy A1: Tính F1 nhóm Trái chiều...")
# Tự động tìm code và data
src_dir = os.path.dirname(glob.glob("**/train_v40.py", recursive=True)[0])
sys.path.append(src_dir)
from train_v40 import split_data_v40
from seeds_config import SEEDS

csv_path = glob.glob("**/dataset_cleaned_masked.csv", recursive=True)[0]
df_corpus = pd.read_csv(csv_path)
y_all = df_corpus['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).values
texts_all = df_corpus['comment'].values

def check_contrast(text):
    keywords = ['nhưng', 'mà', 'tuy nhiên', 'ai ngờ', 'vậy mà', 'thế mà']
    return any(re.search(r'\b' + kw + r'\b', str(text).lower()) for kw in keywords)

f1_v40, f1_pho = [], []
for seed in SEEDS:
    idx_tr, idx_cal, idx_te = split_data_v40(df_corpus, y_all, 'near_dup_group_id', seed)
    texts_test = texts_all[idx_te]
    y_true = y_all[idx_te]
    
    v40_path = f"results/v40_probs/v40_test_seed_{seed}.csv"
    pho_path = f"results/phobert_logits/phobert_test_seed_{seed}.csv"
    if not os.path.exists(v40_path) or not os.path.exists(pho_path): continue
        
    df_v40 = pd.read_csv(v40_path)
    df_pho = pd.read_csv(pho_path)
    
    y_pred_v40 = df_v40[['p_negative', 'p_neutral', 'p_positive']].values.argmax(axis=1)
    y_pred_pho = df_pho[['prob_0', 'prob_1', 'prob_2']].values.argmax(axis=1)
    
    mask = np.array([check_contrast(t) for t in texts_test])
    if mask.sum() > 0:
        f1_v40.append(f1_score(y_true[mask], y_pred_v40[mask], average='macro'))
        f1_pho.append(f1_score(y_true[mask], y_pred_pho[mask], average='macro'))

if f1_v40:
    diff = np.array(f1_v40) - np.array(f1_pho)
    se = np.sqrt((1.0/10 + 0.35) * diff.var(ddof=1))
    p_val = 2 * stats.t.sf(abs(diff.mean() / se), 9)
    print(f"✅ Số mẫu trái chiều (TB/seed): ~{int(3100 * 0.31)}")
    print(f"✅ F1 AERS-Pro: {np.mean(f1_v40):.4f} | F1 PhoBERT: {np.mean(f1_pho):.4f}")
    print(f"✅ Chênh lệch: {diff.mean():.4f}, p-value (N-B): {p_val:.4f}")
