import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

print("⏳ Đang tìm file dữ liệu gán nhãn...")

# Tự động tìm file trong thư mục t01_annotation
truth_hits = glob.glob("t01_annotation/T01_Truth_500.csv")
blind_hits = glob.glob("t01_annotation/T01_Blind_500.xlsx")

if not truth_hits or not blind_hits:
    print("❌ LỖI: Không tìm thấy file. Hãy chắc chắn bạn đã tạo file bằng script 'extract_clean_500_for_iaa.py' và người thứ 2 đã điền vào file Blind.")
    exit()

truth_file = truth_hits[0]
blind_file = blind_hits[0]
print(f"✅ Đã tìm thấy file Đáp án: {truth_file}")
print(f"✅ Đã tìm thấy file Người 2: {blind_file}")

df_truth = pd.read_csv(truth_file)
df_blind = pd.read_excel(blind_file)

# Gộp 2 file theo ID
try:
    df = pd.merge(df_truth, df_blind, on='ID', suffixes=('_truth', '_blind'))
except KeyError:
    print("❌ LỖI: File Excel của người thứ 2 bị mất cột 'ID'. Hãy kiểm tra lại!")
    exit()

# Lấy cột nhãn
col_truth = 'True_Label'
col_blind = [c for c in df.columns if 'Label' in c and c != col_truth][0]

# Làm sạch dữ liệu (Chữ thường, xóa khoảng trắng)
y1 = df[col_truth].astype(str).str.lower().str.strip()
y2 = df[col_blind].astype(str).str.lower().str.strip()

# Kiểm tra xem có câu nào người 2 chưa điền không
valid_labels = ['negative', 'neutral', 'positive']
valid_mask = y2.isin(valid_labels)
y1_clean = y1[valid_mask]
y2_clean = y2[valid_mask]

dropped_count = len(y2) - len(y2_clean)
if dropped_count > 0:
    print(f"⚠️ CẢNH BÁO: Có {dropped_count} câu người thứ 2 bỏ trống hoặc điền sai chính tả (đã bị loại bỏ).")

print(f"-> Số câu đưa vào tính toán: {len(y1_clean)}")
print("⏳ Đang tính Kappa...\n")

# --- TÍNH KAPPA ---
kappa = cohen_kappa_score(y1_clean, y2_clean)

if kappa > 0.8: agreement = "Almost Perfect (Rất tốt)"
elif kappa > 0.6: agreement = "Substantial (Tốt)"
elif kappa > 0.4: agreement = "Moderate (Trung bình)"
else: agreement = "Fair/Poor (Kém)"

print("="*50)
print(f"🎯 COHEN'S KAPPA: {kappa:.4f}")
print(f"🎯 ĐÁNH GIÁ: {agreement}")
print("="*50 + "\n")

# --- VẼ MA TRẬN ---
cm = confusion_matrix(y1_clean, y2_clean, labels=valid_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=valid_labels, yticklabels=valid_labels)
plt.xlabel('Annotator 2 (Blind)')
plt.ylabel('Annotator 1 (Ground Truth)')
plt.title(f'Annotation Disagreement (Kappa = {kappa:.3f})')
plt.tight_layout()
img_file = "t01_annotation/T01_Disagreement_Matrix.png"
plt.savefig(img_file, dpi=300)
plt.close()

# --- XUẤT BÁO CÁO ---
report_file = "t01_annotation/T01_Kappa_Report.md"
with open(report_file, "w", encoding="utf-8") as f:
    f.write(f"# T01: Inter-Annotator Agreement Report\n\n")
    f.write(f"**Total Valid Samples:** {len(y1_clean)}\n\n")
    f.write(f"## Metrics\n")
    f.write(f"*   **Cohen's Kappa:** `{kappa:.4f}`\n")
    f.write(f"*   **Interpretation:** {agreement}\n\n")
    f.write(f"## Disagreement Matrix (Annotator 1 vs Annotator 2)\n\n")
    f.write("| True Label \\ Predicted | negative | neutral | positive |\n")
    f.write("| :--- | :--- | :--- | :--- |\n")
    for i, label in enumerate(valid_labels):
        f.write(f"| **{label}** | {cm[i][0]} | {cm[i][1]} | {cm[i][2]} |\n")
    f.write(f"\n*See `T01_Disagreement_Matrix.png` for the visual heatmap.*\n")

print(f"✅ Đã lưu ảnh Ma trận: {img_file}")
print(f"✅ Đã lưu Báo cáo Markdown: {report_file}")
print("\n🎉 HOÀN TẤT! Hãy mở thư mục t01_annotation để xem kết quả.")
