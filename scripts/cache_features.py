"""Tinh 1 lan dac trung theo-review (khong can fit) cho toan corpus, luu cache. Chay tu thu muc repo."""
import sys, time, joblib, numpy as np, pandas as pd
sys.path.insert(0, '.')
import train_v40 as T
df = pd.read_csv('input_data/dataset_cleaned.csv', encoding='utf-8-sig'); raw = df.comment.astype(str).tolist()
t = time.time(); clean = T.TextPreprocessor.preprocess_batch(raw); print('clean', round(time.time()-t), flush=True)
t = time.time(); manual = T.TextPreprocessor.extract_features_batch(raw); print('manual', round(time.time()-t), flush=True)
t = time.time(); pos = T.TextPreprocessor.extract_pos_features_batch(raw); print('pos', round(time.time()-t), 'HAS_POS', T.HAS_POS, flush=True)
joblib.dump(dict(clean=clean, manual=manual, pos=pos), sys.argv[1]); print('saved')
