import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
import numpy as np
import os

print("⏳ Đang load dữ liệu...")
# 1. Load file gốc và file kết quả
df_corpus = pd.read_csv("data/dataset_cleaned_masked.csv") 
df_truth = pd.read_csv("t01_annotation/T01_Truth_500.csv")
df_blind = pd.read_excel("t01_annotation/T01_Blind_500.xlsx")

# 2. Lấy nhãn GỐC từ corpus
# Vì file T01_Truth đã bị sửa 24 nhãn, ta phải map lại theo text để lấy nhãn gốc thực sự
print("Đang đối chiếu nhãn gốc...")
corpus_dict = dict(zip(df_corpus['comment'], df_corpus['sentiment']))
df_truth['True_Label_Original'] = df_truth['Review_Text'].map(corpus_dict)

# Đếm xem có bao nhiêu nhãn trong file Truth bị lệch so với nhãn gốc
mismatch_count = (df_truth['True_Label'] != df_truth['True_Label_Original']).sum()
print(f"-> CẢNH BÁO: Có {mismatch_count} nhãn trong file Truth bị lệch so với corpus gốc (bạn đã phân xử).")

# 3. Chuẩn bị dữ liệu tính Kappa
y1 = df_truth['True_Label_Original'].astype(str).str.lower().str.strip()
y2 = df_blind['Label (negative/neutral/positive)'].astype(str).str.lower().str.strip()

valid_mask = y2.isin(['negative', 'neutral', 'positive'])
y1_clean = y1[valid_mask]
y2_clean = y2[valid_mask]

# 4. Tính Kappa tổng thể và Kappa từng lớp
kappa_overall = cohen_kappa_score(y1_clean, y2_clean)
kappa_weighted = cohen_kappa_score(y1_clean, y2_clean, weights='linear')

print("\n" + "="*50)
print(f"🎯 KAPPA TỔNG THỂ (so với nhãn gốc): {kappa_overall:.4f}")
print(f"🎯 KAPPA CÓ TRỌNG SỐ (linear): {kappa_weighted:.4f}")
print("="*50 + "\n")

labels = ['negative', 'neutral', 'positive']
print("Kappa từng lớp (One-vs-Rest):")
for label in labels:
    y1_bin = (y1_clean == label).astype(int)
    y2_bin = (y2_clean == label).astype(int)
    kappa_class = cohen_kappa_score(y1_bin, y2_bin)
    print(f"  - {label.capitalize()}: {kappa_class:.4f}")

# 5. Ma trận bất đồng
cm = confusion_matrix(y1_clean, y2_clean, labels=labels)
print("\nMa trận bất đồng (Hàng: Nhãn gốc, Cột: Người 2):")
print(pd.DataFrame(cm, index=[f"Gốc_{l}" for l in labels], columns=[f"Ng2_{l}" for l in labels]))

# 6. Xuất báo cáo
report_file = "t01_annotation/T01_True_Kappa_Report.md"
with open(report_file, "w", encoding="utf-8") as f:
    f.write(f"# T01: True Inter-Annotator Agreement Report\n\n")
    f.write(f"**Samples:** 500\n")
    f.write(f"**Mismatches in Truth File (Arbitration):** {mismatch_count}\n\n")
    f.write(f"## Metrics\n")
    f.write(f"*   **Cohen's Kappa (Overall):** `{kappa_overall:.4f}`\n")
    f.write(f"*   **Cohen's Kappa (Linear Weighted):** `{kappa_weighted:.4f}`\n\n")
    f.write(f"## Per-class Kappa\n")
    for label in labels:
        y1_bin = (y1_clean == label).astype(int)
        y2_bin = (y2_clean == label).astype(int)
        kappa_class = cohen_kappa_score(y1_bin, y2_bin)
        f.write(f"*   **{label.capitalize()}:** {kappa_class:.4f}\n")
    f.write(f"\n## Disagreement Matrix (Original Label vs Annotator 2)\n\n")
    f.write("| True Label \\ Predicted | negative | neutral | positive |\n")
    f.write("| :--- | :--- | :--- | :--- |\n")
    for i, label in enumerate(labels):
        f.write(f"| **{label}** | {cm[i][0]} | {cm[i][1]} | {cm[i][2]} |\n")

print(f"\n✅ Đã lưu báo cáo chi tiết tại: {report_file}")
