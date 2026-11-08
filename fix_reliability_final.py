import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import glob, os

print("Đang vẽ lại biểu đồ Reliability...")
v40_path = glob.glob("**/v40_test_seed_42.csv", recursive=True)[0]
pho_path = glob.glob("**/phobert_test_seed_42.csv", recursive=True)[0]

df_v40 = pd.read_csv(v40_path)
df_pho = pd.read_csv(pho_path)

# SỬA LỖI: Ép kiểu chữ về dạng chuẩn, map sang số, bất kể Pandas phiên bản nào
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
y_true = df_v40['y_true'].astype(str).str.lower().str.strip().map(label_map).values

def get_calib_data(df, prob_cols):
    probs = df[prob_cols].values
    # Đảm bảo tổng xác suất = 1 (tránh lỗi làm tròn)
    probs = probs / probs.sum(axis=1, keepdims=True)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    return calibration_curve(accuracies, confidences, n_bins=10, strategy='uniform')

plt.figure(figsize=(6, 5))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')

# Vẽ AERS-Pro
prob_true_v40, prob_pred_v40 = get_calib_data(df_v40, ['p_negative', 'p_neutral', 'p_positive'])
plt.plot(prob_pred_v40, prob_true_v40, marker='o', color='blue', label='AERS-Pro (ECE 0.024)')

# Vẽ PhoBERT
prob_true_pho, prob_pred_pho = get_calib_data(df_pho, ['prob_0', 'prob_1', 'prob_2'])
plt.plot(prob_pred_pho, prob_true_pho, marker='s', color='red', label='PhoBERT Raw (ECE 0.109)')

plt.xlabel('Mean Predicted Probability (Confidence)')
plt.ylabel('Fraction of Positives (Accuracy)')
plt.title('Reliability Diagram (Seed 42)')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

os.makedirs('figures_t05', exist_ok=True)
plt.savefig('figures_t05/fig_reliability_diagram.png', dpi=300)
plt.close()
print("✅ Đã vẽ lại và lưu: figures_t05/fig_reliability_diagram.png")
