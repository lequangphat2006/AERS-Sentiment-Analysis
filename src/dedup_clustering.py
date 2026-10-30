#!/usr/bin/env python
# ═══════════════════════════════════════════════════════════════════
# dedup_clustering.py — Bước 0.2 của lộ trình V42
# ───────────────────────────────────────────────────────────────────
# Sinh cột `near_dup_group_id` mà split_data_v41() trong train_v41.py
# đang chờ sẵn (GROUP_COL = 'near_dup_group_id', xem phần __main__ của
# train_v41.py) nhưng CHƯA có script nào tạo ra nó trong bộ file hiện
# có — đây chính là mảnh còn thiếu.
#
# Phương pháp: TF-IDF char n-gram (2-4 ký tự) trên TOÀN BỘ dữ liệu
# (gộp mọi file input, không phân biệt train/test — group PHẢI được
# gán trước khi chia, nếu không việc chia theo group sẽ vô nghĩa),
# sau đó nối 2 câu bằng 1 cạnh nếu cosine similarity >= threshold
# (mặc định 0.95, khớp với ngưỡng "0.99/0.95" đã audit trong docx).
# Các câu nối với nhau qua connected components sẽ dùng CHUNG 1
# near_dup_group_id — đảm bảo GroupShuffleSplit không bao giờ tách
# 2 biến thể gần giống nhau ra 2 tập khác nhau.
#
# AN TOÀN DỮ LIỆU: script KHÔNG ghi đè lên input_dir gốc. Nó đọc từ
# --input_dir và ghi ra --output_dir (mặc định input_data_deduped/),
# giữ nguyên tên file + mọi cột cũ, chỉ thêm cột near_dup_group_id.
# Lý do: nếu ghi thêm file mới vào chính input_dir, load_data() trong
# train_v41.py (glob *.csv + *.xlsx) sẽ đọc luôn file đó và nhân đôi
# dữ liệu — một lỗi âm thầm, khó phát hiện.
# ═══════════════════════════════════════════════════════════════════
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def _load_all_files(input_dir: str, text_col: str) -> list:
    """
    Đọc mọi file csv/xlsx trong input_dir, GIỮ NGUYÊN cấu trúc từng
    file riêng (không concat mất dấu vết) để sau này ghi group_id
    ngược lại đúng file/đúng dòng gốc.
    Trả về list[(filepath, DataFrame)].
    """
    files = sorted(glob.glob(os.path.join(input_dir, '*.csv')) +
                    glob.glob(os.path.join(input_dir, '*.xlsx')))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file .csv/.xlsx nào trong '{input_dir}'.")

    loaded = []
    for f in files:
        df = pd.read_csv(f) if f.endswith('.csv') else pd.read_excel(f)
        df.rename(columns=lambda c: 'comment' if c.lower() in ('comment', 'text') else
                                     ('sentiment' if c.lower() in ('sentiment', 'label') else c),
                  inplace=True)
        if text_col not in df.columns:
            print(f"  ⚠️  Bỏ qua '{f}' — không có cột '{text_col}' sau khi chuẩn hoá tên cột.")
            continue
        loaded.append((f, df))
    if not loaded:
        raise ValueError("Không có file nào hợp lệ (thiếu cột text) để dedup.")
    return loaded


def build_near_dup_groups(texts: list,
                           threshold: float = 0.95,
                           ngram_range: tuple = (2, 4),
                           analyzer: str = 'char_wb',
                           max_features: int = 100000,
                           batch_size: int = 2000) -> np.ndarray:
    """
    Trả về mảng group_id (int, 0..k-1) cùng độ dài với texts — mỗi câu
    thuộc đúng 1 group; câu không giống câu nào khác (>= threshold) sẽ
    tự thành 1 group riêng (singleton).

    Dùng NearestNeighbors.radius_neighbors thay vì cosine_similarity()
    đầy đủ (n x n dày đặc) để không nổ RAM khi n lớn — chỉ giữ các cặp
    cạnh có similarity >= threshold làm sparse graph, phần còn lại bỏ qua.
    """
    n = len(texts)
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)

    nn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
    nn.fit(X)

    radius = 1.0 - threshold  # cosine distance <= radius  <=>  cosine similarity >= threshold
    rows, cols = [], []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # radius_neighbors trả về, với mỗi điểm trong batch, index của MỌI
        # điểm khác (trong toàn bộ X) có cosine distance <= radius.
        dist_list, idx_list = nn.radius_neighbors(X[start:end], radius=radius, sort_results=False)
        for local_i, (dists, idxs) in enumerate(zip(dist_list, idx_list)):
            i = start + local_i
            for j in idxs:
                if j != i:
                    rows.append(i)
                    cols.append(int(j))

    data = np.ones(len(rows), dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    # Đảm bảo đối xứng (i~j nhưng có thể j~i không được liệt kê lại do cách
    # duyệt theo batch) trước khi tìm connected components.
    graph = graph.maximum(graph.T)

    n_components, labels = connected_components(graph, directed=False, connection='weak')
    return labels, n_components


def main():
    parser = argparse.ArgumentParser(
        description="Sinh cột near_dup_group_id bằng connected components trên cosine similarity (char n-gram)."
    )
    parser.add_argument('--input_dir', default='input_data')
    parser.add_argument('--output_dir', default='input_data_deduped')
    parser.add_argument('--text_col', default='comment')
    parser.add_argument('--threshold', type=float, default=0.95,
                         help="Ngưỡng cosine similarity để coi 2 câu là 'gần trùng lặp' (mặc định 0.95, khớp audit trong docx).")
    args = parser.parse_args()

    print(f"Đang đọc dữ liệu từ '{args.input_dir}' ...")
    loaded = _load_all_files(args.input_dir, args.text_col)
    lengths = [len(df) for _, df in loaded]
    total_n = sum(lengths)
    print(f"  Tổng cộng {len(loaded)} file, {total_n} dòng.")

    all_texts = []
    for _, df in loaded:
        all_texts.extend(df[args.text_col].astype(str).tolist())

    print(f"Đang tính cosine similarity (char n-gram) + connected components ở ngưỡng {args.threshold} ...")
    labels, n_components = build_near_dup_groups(all_texts, threshold=args.threshold)

    group_sizes = pd.Series(labels).value_counts()
    n_singleton = int((group_sizes == 1).sum())
    n_multi = n_components - n_singleton
    print(f"  → {n_components} group, trong đó {n_multi} group có >=2 câu gần trùng lặp "
          f"(chiếm {total_n - n_singleton}/{total_n} = {(total_n - n_singleton) / total_n:.1%} tổng số câu).")
    print(f"  Group lớn nhất có {int(group_sizes.max())} câu.")

    # Kiểm tra CHÍNH XÁC điều Bước 0 cần: có bao nhiêu group trải dài
    # trên NHIỀU FILE khác nhau — đây là bằng chứng trực tiếp cho việc
    # "2 biến thể cùng khuôn mẫu" bị rải rác vào các file khác nhau,
    # điều mà chỉ concat rồi split ngẫu nhiên (không group-aware) sẽ bỏ sót.
    file_id_per_row = np.concatenate([[i] * n for i, n in enumerate(lengths)])
    df_check = pd.DataFrame({'group': labels, 'file_id': file_id_per_row})
    n_files_per_group = df_check.groupby('group')['file_id'].nunique()
    n_cross_file_groups = int((n_files_per_group > 1).sum())
    print(f"  Trong đó {n_cross_file_groups} group có câu nằm ở >= 2 file khác nhau "
          f"(near-duplicate xuyên-file — dấu hiệu dữ liệu cùng khuôn mẫu bị rải rác).")

    # Ghi group_id NGƯỢC LẠI đúng file/đúng dòng gốc — không ghi đè input_dir.
    os.makedirs(args.output_dir, exist_ok=True)
    offset = 0
    for (filepath, df), n in zip(loaded, lengths):
        out = df.copy()
        out['near_dup_group_id'] = [f"grp_{g:07d}" for g in labels[offset:offset + n]]
        offset += n
        out_path = os.path.join(args.output_dir, os.path.basename(filepath))
        if out_path.endswith('.xlsx'):
            out.to_excel(out_path, index=False)
        else:
            out.to_csv(out_path, index=False)
        print(f"  Đã ghi '{out_path}' ({n} dòng, cột near_dup_group_id đã thêm).")

    print(f"\n✅ Xong. Trỏ train_v41.py vào input_dir='{args.output_dir}' để split_data_v41() "
          f"dùng được GroupShuffleSplit theo near_dup_group_id thay vì fallback stratified split.")


if __name__ == '__main__':
    main()
