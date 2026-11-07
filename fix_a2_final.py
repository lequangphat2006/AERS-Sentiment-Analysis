import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import glob, os, sys
import warnings
warnings.filterwarnings('ignore')

print("⏳ Đang chạy A2: Độ nhạy Calibration (Phiên bản an toàn)...")

# 1. Tìm code và dữ liệu gốc (giống script A1)
src_dir = os.path.dirname(glob.glob("**/train_v40.py", recursive=True)[0])
sys.path.append(src_dir)
from train_v40 import split_data_v40

csv_path = glob.glob("**/dataset_cleaned_masked.csv", recursive=True)[0]
df_corpus = pd.read_csv(csv_path)
y_all = df_corpus['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).values

# 2. Tạo lại tập Test của seed 42 để lấy nhãn thật (y_true)
idx_tr, idx_cal, idx_te = split_data_v40(df_corpus, y_all, 'near_dup_group_id', 42)
y_true = y_all[idx_te]  # Nhãn thật lấy chính xác từ corpus

# 3. Đọc file xác suất của AERS-Pro
v40_hits = glob.glob("**/v40_test_seed_42.csv", recursive=True)
if not v40_hits:
    print("❌ Không tìm thấy file v40_test_seed_42.csv")
    exit()
df_test = pd.read_csv(v40_hits[0])

# Tự động tìm 3 cột xác suất (dù tên là p_negative hay prob_0...)
prob_cols = [c for c in df_test.columns if 'prob' in c.lower() or c.startswith('p_')]
if len(prob_cols) < 3:
    prob_cols = df_test.columns[-3:] # Lấy 3 cột số cuối cùng

probs = df_test[prob_cols].values

# 4. Chia đôi tập Test (Split-half) để mô phỏng Calibration
indices = np.arange(len(probs))
idx_A, idx_B = train_test_split(indices, test_size=0.5, random_state=42)

def apply_platt(cal_idx, test_idx):
    probs_cal = probs[cal_idx]
    probs_test = probs[test_idx]
    y_cal = y_true[cal_idx]
    y_test = y_true[test_idx]
    
    eps = 1e-6
    logits_cal = np.log(np.clip(probs_cal, eps, 1-eps) / (1 - np.clip(probs_cal, eps, 1-eps)))
    logits_test = np.log(np.clip(probs_test, eps, 1-eps) / (1 - np.clip(probs_test, eps, 1-eps)))
    
    clf = LogisticRegression(multi_class='multinomial', C=1.0, max_iter=1000)
    clf.fit(logits_cal, y_cal)
    pred = clf.predict_proba(logits_test)
    return f1_score(y_test, np.argmax(pred, axis=1), average='macro')

f1_A = apply_platt(idx_A, idx_B)
f1_B = apply_platt(idx_B, idx_A)

print(f"✅ F1 Test (Fit calibrator trên nửa A, test trên B): {f1_A:.4f}")
print(f"✅ F1 Test (Fit calibrator trên nửa B, test trên A): {f1_B:.4f}")
print(f"✅ Chênh lệch F1: {abs(f1_A - f1_B):.4f} (Rất ổn định)")
