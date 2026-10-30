import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
import os

plt.rcParams['figure.dpi'] = 300
os.makedirs("figures_t05", exist_ok=True)

print("--- ĐANG VẼ LẠI HÌNH CHO BÀI BÁO (T05) ---")

# 1. CONFUSION MATRIX
cm = np.array([[1803, 96, 147], [144, 1147, 111], [127, 64, 1676]])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'], 
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('AERS-Pro Confusion Matrix (Seed 42)')
plt.tight_layout()
plt.savefig('figures_t05/fig_confusion_matrix.png')
plt.close()
print("✅ Đã lưu: figures_t05/fig_confusion_matrix.png")

# 2. F1 THEO TỪNG SEED
csv_path = "tables/per_seed_all_metrics_8ep.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, df['f1_a'], width, label='AERS-Pro', color='tab:blue')
    plt.bar(x + width/2, df['f1_p'], width, label='PhoBERT (8ep)', color='tab:orange')
    plt.xticks(x, df['seed'].astype(str), rotation=45)
    plt.ylabel('F1-Macro')
    plt.title('Paired F1-Macro Comparison across 10 Seeds')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('figures_t05/fig_f1_per_seed.png', bbox_inches='tight')
    plt.close()
    print("✅ Đã lưu: figures_t05/fig_f1_per_seed.png")

# 3. RELIABILITY DIAGRAM (Đã sửa Legend và Layout)
v40_path = "results/v40_probs/v40_test_seed_42.csv"
phobert_path = "results/phobert_logits/phobert_test_seed_42.csv"

if os.path.exists(v40_path) and os.path.exists(phobert_path):
    df_v40 = pd.read_csv(v40_path)
    df_pho = pd.read_csv(phobert_path)

    label_col = next((c for c in df_v40.columns if c.lower() in ['label', 'y_true', 'true_label', 'target', 'sentiment', 'y']), None)
    if label_col:
        y_true = df_v40[label_col].values
        
        prob_cols_v40 = [c for c in df_v40.columns if 'prob' in c.lower() or c.startswith('p_')]
        if len(prob_cols_v40) < 3:
            prob_cols_v40 = df_v40.select_dtypes(include=[np.number]).columns.tolist()[-3:]
        prob_cols_pho = [c for c in df_pho.columns if 'prob' in c.lower() or c.startswith('p_')]
        if len(prob_cols_pho) < 3:
            prob_cols_pho = df_pho.select_dtypes(include=[np.number]).columns.tolist()[-3:]

        def get_calibration_data(probs_df, prob_cols):
            probs = probs_df[prob_cols].values
            confidences = np.max(probs, axis=1)
            predictions = np.argmax(probs, axis=1)
            accuracies = (predictions == y_true).astype(float)
            # Sử dụng n_bins=15 để chia nhỏ hơn, dễ nhìn hơn
            return calibration_curve(accuracies, confidences, n_bins=15, strategy='uniform')

        plt.figure(figsize=(7, 5)) # Tăng nhẹ chiều rộng để chứa legend
        
        # Đường chéo hoàn hảo
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        
        # Vẽ AERS-Pro
        prob_true_v40, prob_pred_v40 = get_calibration_data(df_v40, prob_cols_v40)
        plt.plot(prob_pred_v40, prob_true_v40, marker='o', color='blue', label='AERS-Pro (ECE 0.0241)')
        
        # Vẽ PhoBERT Raw
        prob_true_pho, prob_pred_pho = get_calibration_data(df_pho, prob_cols_pho)
        plt.plot(prob_pred_pho, prob_true_pho, marker='s', color='red', label='PhoBERT Raw (ECE 0.1069)')

        plt.xlabel('Mean Predicted Probability (Confidence)')
        plt.ylabel('Fraction of Positives (Accuracy)')
        plt.title('Reliability Diagram (Seed 42)')
        
        # Đẩy legend ra ngoài
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6) # Thêm lưới
        
        plt.tight_layout()
        # Lưu với bbox_inches='tight' để không bị cắt legend
        plt.savefig('figures_t05/fig_reliability_diagram.png', bbox_inches='tight')
        plt.close()
        print("✅ Đã lưu: figures_t05/fig_reliability_diagram.png")
