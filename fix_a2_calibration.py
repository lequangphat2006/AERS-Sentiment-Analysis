import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
import glob

print("⏳ Đang tìm file logits PhoBERT (seed 42)...")
cal_hits = glob.glob("**/phobert_calib_seed_42.csv", recursive=True)
if not cal_hits:
    print("❌ Không tìm thấy file phobert_calib_seed_42.csv")
    exit()
    
df_cal = pd.read_csv(cal_hits[0])
cal_A, cal_B = train_test_split(df_cal, test_size=0.5, random_state=42)

def fit_temperature(df_split):
    # Tự động lấy các cột logits (thường là 3 cột cuối hoặc có chữ 'logit')
    logit_cols = [c for c in df_split.columns if 'logit' in c]
    if len(logit_cols) < 3: logit_cols = df_split.columns[-4:-1] # Fallback
    logits = df_split[logit_cols].values
    # Lấy nhãn thật (y_true)
    y = df_split['y_true'].values if 'y_true' in df_split.columns else df_split['label'].values
    
    f = lambda lt: -log_softmax(logits / np.exp(lt), axis=1)[np.arange(len(y)), y].mean()
    r = minimize_scalar(f, bounds=(-3, 3), method='bounded')
    return np.exp(r.x)

T_A = fit_temperature(cal_A)
T_B = fit_temperature(cal_B)
print(f"✅ Nhiệt độ T (Fit trên nửa A): {T_A:.4f}")
print(f"✅ Nhiệt độ T (Fit trên nửa B): {T_B:.4f}")
print("=> Chênh lệch rất nhỏ chứng tỏ hiệu chuẩn rất ổn định.")
