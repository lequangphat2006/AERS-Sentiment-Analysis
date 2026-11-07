import pandas as pd
import glob
import re
import os

print("⏳ Đang tìm file dữ liệu MASKED...")
csv_hits = glob.glob("**/dataset_cleaned_masked.csv", recursive=True)
if not csv_hits:
    csv_hits = glob.glob("**/dataset_cleaned.csv", recursive=True)

if not csv_hits:
    print("❌ Không tìm thấy file dữ liệu nào.")
    exit()

csv_path = csv_hits[0]
print(f"✅ Đang đọc dữ liệu từ: {csv_path}")

df = pd.read_csv(csv_path)
df.columns = df.columns.str.replace('\ufeff', '').str.strip()

label_col = next((c for c in df.columns if c.lower() in ['sentiment', 'label', 'target', 'y']), df.columns[1])
text_col = next((c for c in df.columns if c.lower() in ['comment', 'text', 'review', 'content']), df.columns[0])

total_before = len(df)
df['_clean_text'] = df[text_col].astype(str).str.lower().str.strip()
df = df.drop_duplicates(subset=['_clean_text'])
total_after = len(df)

print(f"-> Tổng số câu ban đầu: {total_before}")
print(f"-> Đã loại bỏ {total_before - total_after} câu trùng lặp.")
print(f"-> Số câu còn lại (độc nhất): {total_after}")

df['_temp_label'] = df[label_col].astype(str).str.lower().str.strip()

neutral_df = df[df['_temp_label'] == 'neutral'].sample(n=200, random_state=42)
neg_df = df[df['_temp_label'] == 'negative'].sample(n=150, random_state=42)
pos_df = df[df['_temp_label'] == 'positive'].sample(n=150, random_state=42)

sample_df = pd.concat([neutral_df, neg_df, pos_df]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"-> Đã lấy tổng cộng {len(sample_df)} câu (đảm bảo 100% không trùng lặp).")

def mask_pii(text):
    if not isinstance(text, str): return text
    text = re.sub(r'(?:(?:\+?84)|0)(?:\s|\.|-)?(?:9|8|7|5|3|2)\d(?:\s|\.|-)?\d{3}(?:\s|\.|-)?\d{3}', '[PHONE]', text)
    text = re.sub(r'https?://\S+|www\.\S+|\b\S+\.(?:com|vn|net|org)\b', '[URL]', text)
    return text

blind_df = pd.DataFrame({
    'ID': range(1, len(sample_df) + 1),
    'Review_Text': sample_df[text_col].apply(mask_pii),
    'Label (negative/neutral/positive)': '' 
})

truth_df = pd.DataFrame({
    'ID': range(1, len(sample_df) + 1),
    'Review_Text': sample_df[text_col],
    'True_Label': sample_df[label_col]
})

os.makedirs("t01_annotation", exist_ok=True)
blind_file = "t01_annotation/T01_Blind_500.xlsx"
truth_file = "t01_annotation/T01_Truth_500.csv"

blind_df.to_excel(blind_file, index=False)
truth_df.to_csv(truth_file, index=False, encoding='utf-8-sig')

print(f"\n🎉 ĐÃ TẠO XONG 2 FILE MỚI TINH (Không trùng lặp):")
print(f"1. {blind_file}  -> GỬI FILE NÀY cho người thứ 2.")
print(f"2. {truth_file}  -> BẠN GIỮ file này làm đáp án.")
