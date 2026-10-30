#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
finish_analysis.py  —  HOÀN TẤT PHẦN PHÂN TÍCH CÒN THIẾU CỦA BÀI BÁO
===============================================================================

Script này giải quyết đúng 3 thiếu sót mà phản biện đã chỉ ra:

  (1) Table 5 — Error Analysis.
      Hiện tại hàng PhoBERT dùng error_analysis_by_subgroup() gốc, còn hàng
      AERS-Pro dùng một hàm gán nhãn phân nhóm TÁI DỰNG khác → hai hàng KHÔNG
      so sánh được (bằng chứng: mean n của trai_chieu là 1.912 so với 1.577).
      Script dùng MỘT bộ gán nhãn DUY NHẤT (`tag_subgroups`) cho CẢ HAI mô hình,
      trên CÙNG các dòng test của CÙNG seed → có thể kiểm định bắt cặp.

  (2) Figure 3 & 4 — reliability diagram + confusion matrix.
      Hiện tại chỉ có phía AERS-Pro, và bản đang có TRONG BÀI BỊ SAI
      (xem CHECK-FIDELITY bên dưới). Script vẽ cặp hình side-by-side cho cả
      hai mô hình từ export xác suất theo từng mẫu.

  (3) CHECK-FIDELITY — chốt an toàn quan trọng nhất của script.
      Hình 4 trong bản thảo hiện tại cho accuracy = 0,8561 và F1-macro = 0,8524,
      trong khi log seed 42 của chính pipeline V40 ghi 0,8702 / 0,8672
      (lệch 1,41 và 1,48 điểm phần trăm; recall lớp neutral 75,2% so với 81%).
      Nguyên nhân: hình được vẽ bằng cách nạp lại model bundle rồi dựng lại
      split, và đường nạp lại đó KHÔNG tái tạo đúng dự đoán gốc.
      Script vì vậy BẮT BUỘC so khớp F1/accuracy tính lại với giá trị đã log;
      lệch quá ngưỡng thì DỪNG, không xuất hình. Không bao giờ đưa vào bài một
      hình mà số không khớp bảng.

-------------------------------------------------------------------------------
CÁCH DÙNG
-------------------------------------------------------------------------------
  # Bước 1 — xuất xác suất theo từng mẫu cho V40 (chạy MỘT lần, xem PHẦN A)
  python finish_analysis.py export-v40

  # Bước 2 — sinh Table 5 hợp nhất + toàn bộ hình (xem PHẦN B, C)
  python finish_analysis.py analyze

Yêu cầu thư mục (mặc định, sửa ở CONFIG):
  data/reviews_labeled.csv                     <- corpus 25.289 dòng, có near_dup_group_id
  models/production/v40/seed_{seed}/           <- 10 bundle AERS-Pro đã lưu
  reports/phobert_logits/phobert_test_seed_{seed}.csv   <- ĐÃ CÓ SẴN (log xác nhận đủ 10 seed)
  reports/v40_probs/v40_test_seed_{seed}.csv   <- Bước 1 sẽ tạo ra
===============================================================================
"""

import os
import sys
import json
import glob
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# FIX PICKLE / __main__ — bắt buộc phải đứng TRƯỚC bất kỳ joblib.load() nào.
# Lúc train, bundle được lưu bằng cách chạy TRỰC TIẾP `python train_v40.py`,
# nên các class (BaseLearnersPool, FeaturePipeline, TextPreprocessor,
# MetaFeatureExtractor...) bị pickle với module = "__main__". Ở đây,
# finish_analysis.py mới là "__main__" đang chạy, còn các class đó nằm trong
# module "train_v40" sau khi import — joblib.load() sẽ báo
# AttributeError: Can't get attribute 'BaseLearnersPool' on module '__main__'.
# Nạp toàn bộ tên public của train_v40 vào __main__ trước khi unpickle để
# pickle tìm đúng chỗ.
# -----------------------------------------------------------------------------
import train_v40 as _train_v40
for _name in dir(_train_v40):
    if not _name.startswith("_"):
        setattr(sys.modules["__main__"], _name, getattr(_train_v40, _name))

# =============================================================================
# CONFIG — sửa cho khớp môi trường của bạn
# =============================================================================
SEEDS        = [42, 7, 123, 2024, 99, 555, 2026, 31337, 8080, 12321]
CLASSES      = ["negative", "neutral", "positive"]
GROUP_COL    = "near_dup_group_id"
TEXT_COL     = "comment"       # <- đã đổi: dataset_cleaned.csv dùng "comment", không phải "text"
LABEL_COL    = "sentiment"     # <- đã đổi: dataset_cleaned.csv dùng "sentiment", không phải "label"

DATA_CSV     = r"D:\About_AERS\Old_Version\[LexEval]_V40_Train_Multiseed_Final\input_data\dataset_cleaned.csv"   # <- đường dẫn thật trên máy bạn
V40_BUNDLE   = "models/production/v40/seed_{seed}"
PHO_TEST_CSV = "reports/phobert_logits/phobert_test_seed_{seed}.csv"
V40_TEST_CSV = "reports/v40_probs/v40_test_seed_{seed}.csv"
FIG_DIR      = "reports/figures_final"
OUT_DIR      = "reports/final_analysis"

# F1-macro / accuracy đã ghi trong log gốc — dùng cho CHECK-FIDELITY.
# (đọc từ train_v40_20260720_193018.log và train_phobert_20260727_132805.log)
LOGGED = {
    "v40": {
        42:    (0.8672, 0.8702), 7:     (0.8775, 0.8792), 123:  (0.8613, 0.8640),
        2024:  (0.8576, 0.8624), 99:    (0.8776, 0.8814), 555:  (0.8532, 0.8555),
        2026:  (0.8669, 0.8706), 31337: (0.8727, 0.8752), 8080: (0.8691, 0.8705),
        12321: (0.8637, 0.8683),
    },
    "phobert": {
        42:    (0.8781, 0.8830), 7:     (0.8729, 0.8798), 123:  (0.8457, 0.8531),
        2024:  (0.8489, 0.8616), 99:    (0.8687, 0.8734), 555:  (0.8574, 0.8664),
        2026:  (0.8806, 0.8850), 31337: (0.8673, 0.8731), 8080: (0.8854, 0.8912),
        12321: (0.8788, 0.8854),
    },
}
FIDELITY_TOL = 0.002   # cho phép lệch tối đa 0,2 điểm phần trăm so với log

# Bảng màu dùng chung cho mọi hình
C_V40, C_PHO, C_GAP = "#2f6fb3", "#c0504d", "#d9534f"


# =============================================================================
# BỘ GÁN NHÃN PHÂN NHÓM DÙNG CHUNG  —  MẤU CHỐT SỬA TABLE 5
# =============================================================================
# QUAN TRỌNG: phải là ĐÚNG các danh sách mà error_analysis_by_subgroup() gốc
# (dùng cho hàng PhoBERT) đã dùng. Hãy copy y nguyên từ file gốc vào đây và
# xoá phần placeholder. Nếu không copy đúng, hàng PhoBERT sinh lại sẽ lệch so
# với results_phobert_multiseed.json và CHECK-SUBGROUP bên dưới sẽ báo lỗi.

NEGATION_WORDS = [
    "không", "ko", "k", "chẳng", "chả", "chưa", "đâu", "hem", "hok", "chớ",
]
CONTRAST_MARKERS = [
    "nhưng", "tuy nhiên", "mà", "thế mà", "vậy mà", "ai ngờ",
]
HEDGE_WORDS = [
    "hơi", "khá", "có lẽ", "có thể", "hình như", "dường như", "chắc là",
]
SHORT_MAX_TOKENS = 5     # cau_ngan: số token tối đa


def tag_subgroups(texts):
    """Gán 4 phân nhóm cho một chuỗi văn bản. MỘT hàm duy nhất, dùng cho CẢ
    HAI mô hình → đây chính là điều làm Table 5 trở thành so sánh bắt cặp hợp lệ.

    Trả về: dict {tên_phân_nhóm: mảng boolean}
    """
    t = pd.Series(texts).fillna("").astype(str).str.lower()

    def has_any(words):
        # so khớp theo biên từ để "mà" không khớp vào "màu", "mà cả"...
        pat = r"(?<!\w)(?:" + "|".join(
            w.replace(" ", r"\s+") for w in sorted(words, key=len, reverse=True)
        ) + r")(?!\w)"
        return t.str.contains(pat, regex=True, na=False).to_numpy()

    n_tokens = t.str.split().apply(len).to_numpy()

    return {
        "phu_dinh":   has_any(NEGATION_WORDS),
        "hedge":      has_any(HEDGE_WORDS),
        "trai_chieu": has_any(CONTRAST_MARKERS),
        "cau_ngan":   n_tokens <= SHORT_MAX_TOKENS,
    }


# =============================================================================
# PHẦN A — XUẤT XÁC SUẤT THEO TỪNG MẪU CHO V40
# =============================================================================
# Pipeline V40 hiện KHÔNG xuất xác suất theo từng mẫu (log không có dòng
# EXPORT nào; pipeline PhoBERT thì có). Đây là gốc rễ của việc Figure 3/4 phải
# dựng lại từ bundle và sai số. Bổ sung bước xuất này một lần là xong vĩnh viễn.

def export_v40_per_sample():
    """Nạp từng bundle V40 đã lưu, dự đoán trên ĐÚNG tập test của PhoBERT cho
    seed đó (không tự chia lại split_data_v40()), ghi ra CSV: y_true, y_pred,
    p_negative, p_neutral, p_positive, text.

    TẠI SAO LẤY TẬP TEST TỪ FILE PHOBERT thay vì tự gọi split_data_v40():
    thử tự chia lại bằng split_data_v40(df_all, y_all, seed=seed) cho ra
    F1/acc LỆCH so với log ở TẤT CẢ 10/10 seed, dù đã xác nhận predict_v40()
    tự nó đúng (F1 trên 2000 dòng train = 0.9928). Nghĩa là idx_te sinh lại
    không phải đúng tập test lúc train — nhiều khả năng do scikit-learn đã
    lên version khác giữa lúc train (2026-07-20) và bây giờ, khiến thuật
    toán nội bộ StratifiedGroupKFold chia fold khác đi dù cùng random_state.
    Thay vì cố dò lại đúng version cũ (dễ vỡ, còn phụ thuộc cả numpy/pandas),
    dùng lại chính tập test PhoBERT đã export (cùng seed, cùng GROUP_COL,
    theo đúng thiết kế ở seeds_config.py: "Baseline vs Proposed đánh giá
    trên ĐÚNG 10 seed/tập test giống hệt nhau") làm tập test chuẩn cho V40 —
    vừa né được vấn đề version drift, vừa đảm bảo đúng yêu cầu paired mà
    Table 5 cần (build_table5() sau này assert y_true 2 bên phải giống hệt).
    """
    from sklearn.preprocessing import LabelEncoder
    load_v40_bundle = _train_v40.load_v40_bundle
    predict_v40 = _train_v40.predict_v40
    split_data_v40 = _train_v40.split_data_v40   # vẫn dùng để lấy idx_tr cho debug-check

    os.makedirs(os.path.dirname(V40_TEST_CSV.format(seed=42)), exist_ok=True)
    df_all = pd.read_csv(DATA_CSV)

    le = LabelEncoder()
    y_all = le.fit_transform(df_all[LABEL_COL])

    for seed in SEEDS:
        bundle = load_v40_bundle(V40_BUNDLE.format(seed=seed))
        print(f"  [DEBUG] seed {seed}: best_T={bundle['best_T']}  "
          f"calibrator={'CÓ' if bundle['calibrator'] is not None else 'None'}  "
          f"clusterer={'CÓ' if bundle['clusterer'] is not None else 'None'}")

        if seed == 42:
            idx_tr, _, _ = split_data_v40(df_all, y_all, group_col=GROUP_COL, seed=seed)
            tr_sample = df_all.iloc[idx_tr[:2000]]
            yp_tr, _ = predict_v40(tr_sample[TEXT_COL].tolist(), bundle)
            f1_tr = f1_score(tr_sample[LABEL_COL].to_numpy(), yp_tr, average="macro")
            print(f"  [DEBUG-TRAIN-CHECK] seed 42: F1 trên 2000 dòng TRAIN = {f1_tr:.4f} "
                  f"(kỳ vọng: rất cao, gần 0.95-1.0)")

        # --- LẤY TẬP TEST TỪ FILE PHOBERT (không tự chia lại) --------------
        pho_path = PHO_TEST_CSV.format(seed=seed)
        if not os.path.exists(pho_path):
            sys.exit(f"[DỪNG] Không thấy {pho_path} — cần có export test set "
                      f"của PhoBERT trước (đây là nguồn duy nhất còn lại cho "
                      f"đúng tập test gốc, vì tự chia lại bằng split_data_v40() "
                      f"đã xác nhận LỆCH do version drift).")
        pho_df = pd.read_csv(pho_path)
        c_text = _find_col(pho_df, "text", TEXT_COL, "comment", "review", "content")
        c_true = _find_col(pho_df, "y_true", "label", "true_label", "y")
        if c_text is None or c_true is None:
            sys.exit(f"[DỪNG] {pho_path}: không tìm được cột text/nhãn thật. "
                      f"Các cột hiện có: {list(pho_df.columns)}")

        te_texts = pho_df[c_text].astype(str).tolist()
        y_true = pho_df[c_true].astype(str).to_numpy()
        if set(np.unique(y_true)) <= {"0", "1", "2"}:
            y_true = np.array(CLASSES)[y_true.astype(int)]

        y_pred, proba = predict_v40(te_texts, bundle)   # proba: (n, 3)

        # An toàn: proba trả về theo thứ tự bundle['le'].classes_ (LabelEncoder
        # RIÊNG của từng seed) — về lý thuyết luôn là alphabetical giống CLASSES
        # ("negative","neutral","positive"), nhưng sắp xếp lại tường minh ở đây
        # để không bao giờ phụ thuộc vào giả định đó im lặng.
        order = [list(bundle["le"].classes_).index(c) for c in CLASSES]
        proba = proba[:, order]

        out = pd.DataFrame({
            "y_true": y_true, "y_pred": y_pred,
            "p_negative": proba[:, 0], "p_neutral": proba[:, 1],
            "p_positive": proba[:, 2], "text": te_texts,
        })
        path = V40_TEST_CSV.format(seed=seed)
        out.to_csv(path, index=False)

        f1 = f1_score(y_true, y_pred, average="macro")
        acc = accuracy_score(y_true, y_pred)
        lf1, lacc = LOGGED["v40"][seed]
        flag = "OK " if (abs(f1 - lf1) <= FIDELITY_TOL and
                         abs(acc - lacc) <= FIDELITY_TOL) else "LỆCH"
        print(f"  [{flag}] seed {seed:>5}  n={len(te):>5}  "
              f"F1={f1:.4f} (log {lf1:.4f})  acc={acc:.4f} (log {lacc:.4f})  -> {path}")

    print("\nXong. Nếu có seed nào 'LỆCH', ĐỪNG chạy analyze — hãy sửa đường "
          "dự đoán trước (thường là thiếu bước áp dụng calibration head, hoặc "
          "fit lại MinMaxScaler/k-means thay vì nạp cái đã fit).")


# =============================================================================
# PHẦN B — ĐỌC EXPORT + CHECK-FIDELITY
# =============================================================================
def _find_col(df, *cands):
    low = {c.lower().strip(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def load_per_sample(kind, seed):
    """Đọc export xác suất theo từng mẫu. Chấp nhận vài cách đặt tên cột."""
    path = (V40_TEST_CSV if kind == "v40" else PHO_TEST_CSV).format(seed=seed)
    if not os.path.exists(path):
        sys.exit(f"[DỪNG] Không thấy {path}")
    df = pd.read_csv(path)

    c_true = _find_col(df, "y_true", "label", "true_label", "y")
    if c_true is None:
        sys.exit(f"[DỪNG] {path}: không tìm được cột nhãn thật.")

    pcols = [_find_col(df, f"p_{c}", f"prob_{c}", c) for c in CLASSES]
    if any(c is None for c in pcols):
        pcols = [_find_col(df, f"logit_{c}", f"logit_{i}") for i, c in enumerate(CLASSES)]
        if any(c is None for c in pcols):
            sys.exit(f"[DỪNG] {path}: không tìm được 3 cột xác suất/logit. "
                     f"Các cột hiện có: {list(df.columns)}")
        z = df[pcols].to_numpy(float)
        z = z - z.max(1, keepdims=True)
        proba = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    else:
        proba = df[pcols].to_numpy(float)
        proba = proba / proba.sum(1, keepdims=True)

    y_true = df[c_true].astype(str).to_numpy()
    if set(np.unique(y_true)) <= {"0", "1", "2"}:
        y_true = np.array(CLASSES)[y_true.astype(int)]

    c_pred = _find_col(df, "y_pred", "pred", "prediction")
    y_pred = (df[c_pred].astype(str).to_numpy() if c_pred
              else np.array(CLASSES)[proba.argmax(1)])
    if set(np.unique(y_pred)) <= {"0", "1", "2"}:
        y_pred = np.array(CLASSES)[y_pred.astype(int)]

    c_txt = _find_col(df, "text", TEXT_COL, "review", "content")
    texts = df[c_txt].astype(str).to_numpy() if c_txt else None

    return dict(y_true=y_true, y_pred=y_pred, proba=proba, texts=texts, n=len(df))


def check_fidelity(kind, seed, d):
    """CHỐT AN TOÀN: export phải tái tạo đúng số đã log. Đây là bước lẽ ra
    phải có từ đầu — nó sẽ chặn được đúng cái lỗi của Figure 4 hiện tại."""
    f1 = f1_score(d["y_true"], d["y_pred"], average="macro")
    acc = accuracy_score(d["y_true"], d["y_pred"])
    lf1, lacc = LOGGED[kind][seed]
    ok = abs(f1 - lf1) <= FIDELITY_TOL and abs(acc - lacc) <= FIDELITY_TOL
    if not ok:
        print(f"  [LỆCH] {kind} seed {seed}: F1={f1:.4f} (log {lf1:.4f}), "
              f"acc={acc:.4f} (log {lacc:.4f})")
    return ok, f1, acc


# =============================================================================
# PHẦN C — METRIC, TABLE 5, HÌNH
# =============================================================================
def ece_bins(conf, correct, M=10):
    """ECE theo đúng quy ước Mục 4.2: B1=[0,0.1], Bm=(0.1(m-1), 0.1m]."""
    edges = np.linspace(0, 1, M + 1)
    rows, ece = [], 0.0
    n = len(conf)
    for m in range(M):
        lo, hi = edges[m], edges[m + 1]
        sel = (conf >= lo) & (conf <= hi) if m == 0 else (conf > lo) & (conf <= hi)
        k = int(sel.sum())
        if k == 0:
            rows.append((lo, hi, 0, np.nan, np.nan)); continue
        a, c = correct[sel].mean(), conf[sel].mean()
        ece += k / n * abs(a - c)
        rows.append((lo, hi, k, a, c))
    return ece, rows


def brier(proba, y_true):
    Y = np.zeros_like(proba)
    for i, c in enumerate(CLASSES):
        Y[:, i] = (y_true == c)
    return float(((proba - Y) ** 2).sum(1).mean())


def fit_temperature(logits_cal, y_cal):
    """Temperature scaling (Guo et al. 2017) — fit T bằng NLL trên Cal split."""
    from scipy.optimize import minimize_scalar
    yi = np.array([CLASSES.index(c) for c in y_cal])

    def nll(logT):
        T = np.exp(logT)
        z = logits_cal / T
        z = z - z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -logp[np.arange(len(yi)), yi].mean()

    r = minimize_scalar(nll, bounds=(-2, 2), method="bounded")
    return float(np.exp(r.x))


def build_table5(per_sample):
    """Table 5 hợp nhất: cùng bộ gán nhãn, cùng dòng test, kiểm định bắt cặp."""
    groups = ["phu_dinh", "hedge", "trai_chieu", "cau_ngan"]
    acc = {g: {"v40": [], "phobert": []} for g in groups}
    f1m = {g: {"v40": [], "phobert": []} for g in groups}
    nn = {g: [] for g in groups}

    for seed in SEEDS:
        dv, dp = per_sample["v40"][seed], per_sample["phobert"][seed]
        texts = dv["texts"] if dv["texts"] is not None else dp["texts"]
        if texts is None:
            sys.exit("[DỪNG] Export không có cột text → không gán được phân nhóm. "
                     "Thêm cột 'text' vào file export.")
        if not np.array_equal(dv["y_true"], dp["y_true"]):
            sys.exit(f"[DỪNG] seed {seed}: y_true của hai mô hình khác nhau → "
                     "hai export không cùng thứ tự dòng. Hãy xuất lại kèm cột "
                     "index gốc và sort theo nó.")
        masks = tag_subgroups(texts)
        for g in groups:
            m = masks[g]
            nn[g].append(int(m.sum()))
            for kind, d in (("v40", dv), ("phobert", dp)):
                acc[g][kind].append(accuracy_score(d["y_true"][m], d["y_pred"][m]))
                f1m[g][kind].append(
                    f1_score(d["y_true"][m], d["y_pred"][m], average="macro"))

    rows = []
    for g in groups:
        a_v, a_p = np.array(acc[g]["v40"]), np.array(acc[g]["phobert"])
        f_v, f_p = np.array(f1m[g]["v40"]), np.array(f1m[g]["phobert"])
        d = f_v - f_p
        _, p_t = stats.ttest_rel(f_v, f_p)
        try:
            _, p_w = stats.wilcoxon(f_v, f_p)
        except Exception:
            p_w = np.nan
        se = d.std(ddof=1) / np.sqrt(len(d))
        tc = stats.t.ppf(0.975, len(d) - 1)
        rows.append(dict(
            subgroup=g, mean_n=float(np.mean(nn[g])),
            n_min=int(np.min(nn[g])), n_max=int(np.max(nn[g])),
            v40_acc=a_v.mean(), v40_acc_sd=a_v.std(ddof=1),
            pho_acc=a_p.mean(), pho_acc_sd=a_p.std(ddof=1),
            v40_f1=f_v.mean(), v40_f1_sd=f_v.std(ddof=1),
            pho_f1=f_p.mean(), pho_f1_sd=f_p.std(ddof=1),
            dF1=d.mean(), ci_lo=d.mean() - tc * se, ci_hi=d.mean() + tc * se,
            dz=d.mean() / d.std(ddof=1), p_ttest=p_t, p_wilcoxon=p_w))

    print("\n" + "=" * 110)
    print("TABLE 5 (đã hợp nhất) — cùng bộ gán nhãn cho cả hai mô hình, "
          "trung bình ± SD trên 10 seed bắt cặp")
    print("=" * 110)
    hdr = (f"{'Subgroup':<14}{'mean n':>8}{'  AERS-Pro acc':>18}{'  PhoBERT acc':>18}"
           f"{'  AERS-Pro F1':>18}{'  PhoBERT F1':>18}{'  ΔF1':>9}{'  p':>9}")
    print(hdr)
    for r in rows:
        print(f"{r['subgroup']:<14}{r['mean_n']:>8.0f}"
              f"{r['v40_acc']:>11.4f}±{r['v40_acc_sd']:.4f}"
              f"{r['pho_acc']:>11.4f}±{r['pho_acc_sd']:.4f}"
              f"{r['v40_f1']:>11.4f}±{r['v40_f1_sd']:.4f}"
              f"{r['pho_f1']:>11.4f}±{r['pho_f1_sd']:.4f}"
              f"{r['dF1']:>+9.4f}{r['p_ttest']:>9.4f}")

    # CHECK-SUBGROUP: hàng PhoBERT sinh lại phải khớp results_phobert_multiseed.json
    ref = "results_phobert_multiseed.json"
    if os.path.exists(ref):
        J = json.load(open(ref, encoding="utf-8"))["metrics_by_seed"]
        print("\nCHECK-SUBGROUP — hàng PhoBERT sinh lại vs. results_phobert_multiseed.json:")
        for r in rows:
            g = r["subgroup"]
            rf1 = np.mean([J[str(s)]["subgroup_results"][g]["f1_macro"] for s in SEEDS])
            rn = np.mean([J[str(s)]["subgroup_results"][g]["n"] for s in SEEDS])
            dn, df = r["mean_n"] - rn, r["pho_f1"] - rf1
            ok = abs(df) < 0.005 and abs(dn) / max(rn, 1) < 0.05
            print(f"  {g:<12} n: {r['mean_n']:>6.0f} vs {rn:>6.0f} ({dn:+.0f})   "
                  f"F1: {r['pho_f1']:.4f} vs {rf1:.4f} ({df:+.4f})   "
                  f"{'OK' if ok else '<-- danh sách từ khoá CHƯA khớp bản gốc'}")
        print("  Nếu có dòng chưa khớp: sửa NEGATION_WORDS / CONTRAST_MARKERS /\n"
              "  HEDGE_WORDS / SHORT_MAX_TOKENS ở đầu file cho đúng bản gốc rồi chạy lại.")

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/table5_unified.csv", index=False)
    print(f"\n-> {OUT_DIR}/table5_unified.csv")
    return rows


def fig_reliability(per_sample, seed, path):
    """Figure 3 — reliability diagram SIDE-BY-SIDE cho cả hai mô hình."""
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0),
                             gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    for j, (kind, name, col) in enumerate(
            [("v40", "AERS-Pro", C_V40), ("phobert", "PhoBERT-base-v2", C_PHO)]):
        d = per_sample[kind][seed]
        conf = d["proba"].max(1)
        correct = (d["y_pred"] == d["y_true"]).astype(float)
        e, rows = ece_bins(conf, correct)

        ax, axc = axes[0, j], axes[1, j]
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1.2,
                label="Perfect calibration")
        for lo, hi, k, a, c in rows:
            if k == 0:
                continue
            ax.bar((lo + hi) / 2, a, width=0.093, color=col,
                   edgecolor="black", lw=0.5, zorder=2)
            ax.plot([(lo + hi) / 2] * 2, sorted([a, c]), color=C_GAP, lw=2.4, zorder=3)
            axc.bar((lo + hi) / 2, k, width=0.093, color="#8a8a8a",
                    edgecolor="black", lw=0.5)
        ax.set_title(f"{name}  (seed {seed}, n={d['n']:,})\nECE = {e:.4f}", fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel("Accuracy"); axc.set_ylabel("Count")
        ax.legend(loc="upper left", fontsize=8)
        axc.set_xlabel("Confidence (max predicted probability)")
        axc.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"-> {path}")


def fig_confusion(per_sample, seed, path):
    """Figure 4 — confusion matrix SIDE-BY-SIDE cho cả hai mô hình."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, (kind, name) in zip(axes, [("v40", "AERS-Pro"),
                                       ("phobert", "PhoBERT-base-v2")]):
        d = per_sample[kind][seed]
        cm = confusion_matrix(d["y_true"], d["y_pred"], labels=CLASSES)
        pct = cm / cm.sum(1, keepdims=True) * 100
        im = ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:,}\n({pct[i, j]:.1f}%)",
                        ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > cm.max() * 0.55 else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES)
        ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
        ax.set_title(f"{name}  (seed {seed}, n={cm.sum():,})\n"
                     f"acc = {np.trace(cm)/cm.sum():.4f}", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"-> {path}")


def find_disagreements(per_sample, seed, k=12):
    """Ví dụ định tính BẮT CẶP: đúng những dòng hai mô hình dự đoán khác nhau.
    Đây là thứ Mục 5.4 định làm nhưng chưa làm được vì thiếu export PhoBERT."""
    dv, dp = per_sample["v40"][seed], per_sample["phobert"][seed]
    texts = dv["texts"] if dv["texts"] is not None else dp["texts"]
    if texts is None:
        return None
    m = dv["y_pred"] != dp["y_pred"]
    out = pd.DataFrame({
        "text": texts[m], "y_true": dv["y_true"][m],
        "aers_pred": dv["y_pred"][m], "aers_conf": dv["proba"][m].max(1),
        "pho_pred": dp["y_pred"][m], "pho_conf": dp["proba"][m].max(1),
    })
    out["aers_correct"] = out.y_true == out.aers_pred
    out["pho_correct"] = out.y_true == out.pho_pred
    os.makedirs(OUT_DIR, exist_ok=True)
    p = f"{OUT_DIR}/disagreements_seed{seed}.csv"
    out.to_csv(p, index=False)
    print(f"\n{len(out):,}/{len(dv['y_true']):,} dòng hai mô hình dự đoán khác nhau "
          f"({100*len(out)/len(dv['y_true']):.1f}%)  -> {p}")
    print(f"  AERS-Pro đúng / PhoBERT sai : {int((out.aers_correct & ~out.pho_correct).sum()):,}")
    print(f"  PhoBERT đúng / AERS-Pro sai : {int((out.pho_correct & ~out.aers_correct).sum()):,}")
    print(f"  cả hai đều sai              : {int((~out.aers_correct & ~out.pho_correct).sum()):,}")
    print("\n  Ứng viên cho Example 1-3 của Mục 5.4 (lệch nhau, một bên rất tự tin):")
    cand = out.reindex(
        (out.aers_conf - out.pho_conf).abs().sort_values(ascending=False).index)
    for _, r in cand.head(k).iterrows():
        print(f"   • {r.text[:70]!r} true={r.y_true} | "
              f"AERS={r.aers_pred}({r.aers_conf:.3f}) Pho={r.pho_pred}({r.pho_conf:.3f})")
    return out


# =============================================================================
# MAIN
# =============================================================================
def analyze():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 110)
    print("CHECK-FIDELITY — export theo từng mẫu có tái tạo đúng số đã log không?")
    print("=" * 110)
    per_sample, all_ok = {"v40": {}, "phobert": {}}, True
    for kind in ("v40", "phobert"):
        for seed in SEEDS:
            d = load_per_sample(kind, seed)
            ok, f1, acc = check_fidelity(kind, seed, d)
            all_ok &= ok
            per_sample[kind][seed] = d
        print(f"  {kind:<8}: {'tất cả 10 seed KHỚP log' if all_ok else 'CÓ SEED LỆCH'}")

    if not all_ok:
        sys.exit(
            "\n[DỪNG] Có seed không tái tạo được số đã log. KHÔNG xuất hình.\n"
            "  Đây đúng là lỗi đang có trong bản thảo: Figure 4 hiện tại cho\n"
            "  accuracy 0,8561 / F1 0,8524 trong khi log seed 42 ghi 0,8702 / 0,8672.\n"
            "  Cần kiểm tra theo thứ tự:\n"
            "   1. Có áp dụng calibration head (Eq. 3-4) ở đúng 6 seed đã 'ÁP DỤNG'?\n"
            "   2. MinMaxScaler của nhánh lexicon là cái ĐÃ FIT trên train, hay bị fit lại?\n"
            "   3. Cluster k-means là cái ĐÃ FIT trên train, hay bị fit lại?\n"
            "   4. Ngưỡng neutral T có đúng giá trị đã fit của seed đó\n"
            "      (T=2.0 ở 8 seed, 1.5 ở seed 8080, 3.0 ở seed 12321)?\n"
            "   5. FeaturePipeline dùng best_params của ĐÚNG seed đó?\n")

    rows = build_table5(per_sample)

    ref_seed = 42
    fig_reliability(per_sample, ref_seed, f"{FIG_DIR}/fig3_reliability_both.png")
    fig_confusion(per_sample, ref_seed, f"{FIG_DIR}/fig4_confusion_both.png")
    find_disagreements(per_sample, ref_seed)

    # ECE/Brier tính lại trên cả 10 seed, để đối chiếu với Table 2
    print("\n" + "=" * 110)
    print("ECE / Brier tính lại từ export (đối chiếu Table 2)")
    print("=" * 110)
    summ = {}
    for kind in ("v40", "phobert"):
        E, B = [], []
        for seed in SEEDS:
            d = per_sample[kind][seed]
            conf = d["proba"].max(1)
            e, _ = ece_bins(conf, (d["y_pred"] == d["y_true"]).astype(float))
            E.append(e); B.append(brier(d["proba"], d["y_true"]))
        summ[kind] = (np.array(E), np.array(B))
        print(f"  {kind:<8} ECE {np.mean(E):.4f} ± {np.std(E, ddof=1):.4f}   "
              f"Brier {np.mean(B):.4f} ± {np.std(B, ddof=1):.4f}")
    t, p = stats.ttest_rel(summ["v40"][0], summ["phobert"][0])
    print(f"  paired t-test ECE: t={t:.3f}  p={p:.3g}   "
          f"ratio = {np.mean(summ['phobert'][0])/np.mean(summ['v40'][0]):.2f}x")

    json.dump({"table5": rows}, open(f"{OUT_DIR}/summary.json", "w"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\nHOÀN TẤT. Hình: {FIG_DIR}/  |  Bảng: {OUT_DIR}/")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "export-v40":
        export_v40_per_sample()
    elif cmd == "analyze":
        analyze()
    else:
        sys.exit(__doc__)
