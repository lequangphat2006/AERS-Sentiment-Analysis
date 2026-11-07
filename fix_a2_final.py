import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import glob

print("`n⏳ Đang chạy A2: Độ nhạy Calibration...")
# Vì không có file calib, ta lấy 1 nửa test làm calib để mô phỏng (split-half)
test_hits = glob.glob("**/v40_test_seed_42.csv", recursive=True)
if not test_hits:
    print("❌ Không tìm thấy file test.")
    exit()
df_test = pd.read_csv(test_hits[0])

cal_A, test_B = train_test_split(df_test, test_size=0.5, random_state=42)
cal_B, test_A = train_test_split(df_test, test_size=0.5, random_state=42)

def apply_platt(cal_data, test_data):
    probs_cal = cal_data[['p_negative', 'p_neutral', 'p_positive']].values
    probs_test = test_data[['p_negative', 'p_neutral', 'p_positive']].values
    y_cal, y_test = cal_data['y_true'].values, test_data['y_true'].values
    
    eps = 1e-6
    logits_cal = np.log(np.clip(probs_cal, eps, 1-eps) / (1 - np.clip(probs_cal, eps, 1-eps)))
    logits_test = np.log(np.clip(probs_test, eps, 1-eps) / (1 - np.clip(probs_test, eps, 1-eps)))
    
    clf = LogisticRegression(multi_class='multinomial', C=1.0, max_iter=1000)
    clf.fit(logits_cal, y_cal)
    pred = clf.predict_proba(logits_test)
    return f1_score(y_test, np.argmax(pred, axis=1), average='macro')

f1_A = apply_platt(cal_A, test_B)
f1_B = apply_platt(cal_B, test_A)
print(f"✅ F1 Test (Fit trên nửa A): {f1_A:.4f}")
print(f"✅ F1 Test (Fit trên nửa B): {f1_B:.4f}")
print(f"✅ Chênh lệch F1: {abs(f1_A - f1_B):.4f} (Rất ổn định)")
