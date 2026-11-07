import pandas as pd
import re

print("⏳ Đang load dữ liệu để sửa lỗi nhóm trái chiều...")
df = pd.read_csv("data/dataset_cleaned_masked.csv")

def check_contrast(text):
    keywords = ['nhưng', 'mà', 'tuy nhiên', 'ai ngờ', 'vậy mà', 'thế mà']
    text_lower = str(text).lower()
    # Dùng \b để khớp chính xác nguyên từ (tránh 'mà' khớp với 'màu')
    return any(re.search(r'\b' + kw + r'\b', text_lower) for kw in keywords)

df['is_trai_chieu'] = df['comment'].apply(check_contrast)
print(f"✅ Tỷ lệ trái chiều CHUẨN (dùng token): {df['is_trai_chieu'].mean() * 100:.2f}% (Kỳ vọng ~31.0%)")
