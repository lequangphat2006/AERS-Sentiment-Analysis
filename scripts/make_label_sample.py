"""Rút mẫu gán nhãn lần hai (T01). Chạy từ thư mục gốc repo, trên file ĐÃ CHE PII.

    python make_label_sample.py --src data/dataset_cleaned_masked.csv --out label_round2 --seed 20261101

Tạo trong thư mục --out:
  annotation_blind.csv       -> GỬI cho người gán nhãn 2 (chỉ id + comment, đã trộn)
  annotation_key_PRIVATE.csv -> GIỮ RIÊNG (nhãn gốc, nhóm mẫu). KHÔNG commit, KHÔNG gửi.
  sampling_manifest.json     -> seed, tiêu chí, số lượng (đưa vào bài/phụ lục)

Thiết kế mẫu (mô tả được trong bài):
  Phần A (ngẫu nhiên phân tầng theo lớp): đo độ đồng thuận 'chung', không thiên lệch theo độ khó.
  Phần B (ca khó: hedge/contrast/negation, ưu tiên neutral): đo độ đồng thuận ở vùng biên.
  Mỗi mẫu lấy tối đa 1 review / near_dup_group_id để tránh gán nhãn trùng văn bản gần giống.
  KHÔNG chọn theo lỗi mô hình để κ không bị thiên lệch; nếu muốn phân tích lỗi, làm riêng sau.
"""
import argparse, json, os, re
import numpy as np, pandas as pd

HEDGE = ['tạm được', 'tạm', 'cũng được', 'bình thường', 'cũng ổn', 'tàm tạm', 'không đến nỗi', 'cũng tạm']
CONTRAST = ['nhưng', 'tuy nhiên', 'ai ngờ', 'vậy mà', 'thế mà']   # bỏ 'mà' đơn lẻ (dễ khớp nhầm 'màu')
NEG_TOK = {'không', 'chẳng', 'chả', 'chưa', 'k', 'ko', 'chớ', 'hok', 'hem'}  # bỏ 'đâu' (nhập nhằng)

def flags(text):
    t = str(text).lower()
    toks = set(re.findall(r'\w+', t))
    return {'hedge': any(h in t for h in HEDGE),
            'contrast': any(re.search(r'\b' + c + r'\b', t) for c in CONTRAST),
            'negation': bool(toks & NEG_TOK)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/dataset_cleaned_masked.csv')
    ap.add_argument('--out', default='label_round2')
    ap.add_argument('--seed', type=int, default=20261101)
    ap.add_argument('--a_neu', type=int, default=100); ap.add_argument('--a_neg', type=int, default=85); ap.add_argument('--a_pos', type=int, default=85)
    ap.add_argument('--b_neu', type=int, default=90);  ap.add_argument('--b_neg', type=int, default=45); ap.add_argument('--b_pos', type=int, default=45)
    ap.add_argument('--min_len', type=int, default=3, help='bỏ review ngắn hơn n ký tự')
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    df = pd.read_csv(a.src).reset_index().rename(columns={'index': 'row_index'})
    df = df[df.comment.astype(str).str.len() >= a.min_len]
    # 1 đại diện / nhóm near-dup
    df = df.sample(frac=1, random_state=a.seed).drop_duplicates('near_dup_group_id')
    fl = pd.DataFrame([flags(t) for t in df.comment], index=df.index)
    df = pd.concat([df, fl], axis=1)
    df['hard'] = df[['hedge', 'contrast', 'negation']].any(axis=1)
    # ca khó chặt hơn: hedge hoặc contrast (negation đơn thuần quá phổ biến)
    df['hard_core'] = df.hedge | df.contrast

    def take(pool, label, n):
        p = pool[pool.sentiment == label]
        return p.sample(n=min(n, len(p)), random_state=rng)

    B = pd.concat([take(df[df.hard_core], 'neutral', a.b_neu), take(df[df.hard_core], 'negative', a.b_neg),
                   take(df[df.hard_core], 'positive', a.b_pos)]).assign(stratum='B_hard')
    rest = df[~df.row_index.isin(B.row_index)]
    A = pd.concat([take(rest, 'neutral', a.a_neu), take(rest, 'negative', a.a_neg),
                   take(rest, 'positive', a.a_pos)]).assign(stratum='A_random')
    S = pd.concat([A, B]).sample(frac=1, random_state=a.seed + 1).reset_index(drop=True)
    S['item_id'] = ['R2_%04d' % i for i in range(1, len(S) + 1)]

    os.makedirs(a.out, exist_ok=True)
    S[['item_id', 'comment']].to_csv(os.path.join(a.out, 'annotation_blind.csv'), index=False, encoding='utf-8-sig')
    S[['item_id', 'row_index', 'near_dup_group_id', 'sentiment', 'stratum', 'hedge', 'contrast', 'negation']] \
        .rename(columns={'sentiment': 'label_orig'}).to_csv(os.path.join(a.out, 'annotation_key_PRIVATE.csv'), index=False)
    man = {'seed': a.seed, 'source': a.src, 'n_total': len(S),
           'by_stratum_label': S.groupby(['stratum', 'sentiment']).size().rename('n').reset_index().to_dict('records'),
           'rules': {'one_review_per_near_dup_group': True, 'min_len_chars': a.min_len,
                     'hard_core': 'contains hedge phrase or contrast marker',
                     'hedge': HEDGE, 'contrast': CONTRAST, 'negation_tokens': sorted(NEG_TOK)}}
    json.dump(man, open(os.path.join(a.out, 'sampling_manifest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('Tổng', len(S)); print(S.groupby(['stratum', 'sentiment']).size())
    print('Gửi file:', os.path.join(a.out, 'annotation_blind.csv'))

if __name__ == '__main__':
    main()
