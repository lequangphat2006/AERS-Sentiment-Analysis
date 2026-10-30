import re, sys, pandas as pd
df = pd.read_csv(sys.argv[1] + '/input_data/dataset_cleaned.csv', encoding='utf-8-sig'); t = df.comment.fillna('').astype(str)
pats = {'phone_vn': r'(?<!\d)(?:\+?84|0)(?:[\s.\-]?\d){9}(?!\d)', 'email': r'[\w.+-]+@[\w-]+\.[\w.-]+', 'url': r'https?://\S+|www\.\S+', 'zalo/fb/ig handle': r'(?i)\b(?:zalo|facebook|fb|insta(?:gram)?|tiktok)\b[:\s]*[\w./@-]{3,}',
        'digits10+': r'\d{10,}', 'address_hint': r'(?i)\b(?:số nhà|đường|phường|quận|huyện|tp\.?\s?hcm)\b'}
rows = []
for k, p in pats.items():
    m = t.str.contains(p, regex=True); rows.append((k, int(m.sum()))); 
    if 0 < m.sum() <= 6: print(k, [x[:100] for x in t[m].tolist()])
print(pd.DataFrame(rows, columns=['pattern', 'n_reviews']).to_string(index=False)); print('rows', len(df), 'dup comment texts', int(t.duplicated().sum()), 'cols', list(df.columns))
