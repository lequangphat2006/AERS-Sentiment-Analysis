"""Che thong tin ca nhan (PII) trong corpus. Chay tu thu muc goc repo.

    py scripts/mask_pii.py --src data/dataset_cleaned.csv --dst data/dataset_cleaned_masked.csv

Ban v3: them @handle (ten tai khoan MXH) va ma gioi thieu (referral code).
Sau khi che, script tu quet lai va dung (exit 1) neu con sot SDT/handle/ma gioi thieu/email.
Luu y: ten rieng va dia chi KHONG duoc ra soat boi script nay.
"""
import argparse, re, sys
import pandas as pd

PHONE = re.compile(r'(?<![\w])(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?![\w])')
URL = re.compile(r'https?://\S+|www\.\S+|\b[\w\-]+(?:\.[\w\-]+)*\.(?:com|vn|net|org|me|ly|io)(?:/\S*)?\b', re.I)
EMAIL = re.compile(r'[\w.+\-]+@[\w\-]+\.[\w.\-]+')
SOCIAL = re.compile(r'\b(?:fb|facebook|ins|instagram|zalo|zl|telegram|tiktok)\s*[:：]\s*[\w.@\-]+|\b[\w\-]+\.zalo\.[\w]+', re.I)
HANDLE = re.compile(r'(?<![\w.])@[A-Za-z0-9_.]{3,}')
REFCODE = re.compile(r'((?i:mã\s+(?:người\s+)?giới\s+thiệu|mã\s+mời)\s*[:(]?\s*)((?=[A-Z0-9]*\d)[A-Z0-9]{5,})')

def mask_text(t):
    if not isinstance(t, str):
        return t
    t = EMAIL.sub('[EMAIL]', t)
    t = SOCIAL.sub('[SOCIAL]', t)
    t = HANDLE.sub('[SOCIAL]', t)
    t = URL.sub('[URL]', t)
    t = PHONE.sub('[PHONE]', t)
    t = REFCODE.sub(lambda m: m.group(1) + '[REFCODE]', t)
    return t

RESIDUAL = re.compile(r'(?<!\d)(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\d)')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/dataset_cleaned.csv')
    ap.add_argument('--dst', default='data/dataset_cleaned_masked.csv')
    a = ap.parse_args()
    df = pd.read_csv(a.src, encoding='utf-8-sig')
    s = df['comment'].astype(str)
    counts = {n: int(s.str.count(p).sum()) for n, p in
              [('PHONE', PHONE), ('URL', URL), ('EMAIL', EMAIL), ('SOCIAL', SOCIAL),
               ('HANDLE', HANDLE), ('REFCODE', REFCODE)]}
    df['comment'] = df['comment'].apply(mask_text)
    c = df['comment'].astype(str)
    left = {'PHONE': int(c.str.contains(RESIDUAL).sum()),
            'HANDLE': int(c.str.contains(HANDLE).sum()),
            'REFCODE': int(c.str.contains(REFCODE).sum()),
            'EMAIL': int(c.str.contains(EMAIL).sum())}
    print('So luot khop mau (truoc khi che, co the chong lan):', counts)
    print('Con sot sau khi che:', left)
    print('So dong bi doi:', int((c != s).sum()))
    if any(left.values()):
        sys.exit(1)
    df.to_csv(a.dst, index=False)
    print('Da luu', a.dst)

if __name__ == '__main__':
    main()