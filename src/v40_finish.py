#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
v40_finish.py — HOÀN TẤT PHÍA V40. KHÔNG CẦN FILE CSV CỦA PHOBERT.
===============================================================================

BỐI CẢNH — hai vấn đề đã được giải, không phải né

  (1) Version drift (chẩn đoán của bạn đúng).
      Đã dò và xác nhận: scikit-learn 1.5.2 / 1.6.1 / 1.7.2 đều tái lập ĐÚNG
      10/10 seed khớp log gốc (seed 42 → 14849/5125/5315). Riêng 1.8.0 lệch,
      vì bản này đổi thuật toán StratifiedGroupKFold sang cân bằng kích thước
      fold tuyệt đối: mọi seed đều ra 5058, mất hẳn dao động 4877–5315 của
      thuật toán cũ.
      → Chỉ cần pin version. Không cần lấy tập test qua file PhoBERT.
      Đã kiểm chứng thêm: với 1.7.2, cả 30 con số support-theo-lớp trên 10 seed
      khớp tuyệt đối với classification report trong CẢ HAI log → các dòng test
      giống hệt nhau, không chỉ giống kích thước.

  (2) Bộ gán nhãn phân nhóm — đã lấy ĐÚNG từ train_phobert.py.
      Các định nghĩa dưới đây copy nguyên từ _tag_phenomena() trong
      train_phobert.py và đã kiểm chứng tái lập CHÍNH XÁC cả 40 con số n
      (4 nhóm × 10 seed) mà log PhoBERT đã ghi. Vì n là hàm tất định của bộ
      gán nhãn và của các dòng test, khớp đồng thời 40/40 nghĩa là hai hàng
      của Table 5 đo trên CÙNG các tập con → so sánh bắt cặp hợp lệ.

  (3) Table 5 KHÔNG cần CSV PhoBERT: results_phobert_multiseed.json đã có
      subgroup_results đủ 10 seed. Thiếu duy nhất cột AERS-Pro — script này
      tính ra, kèm ΔF1, CI 95%, dz, p (t-test bắt cặp) và p (Wilcoxon).

  (4) Vẫn cần CSV PhoBERT cho panel BÊN PHẢI của Figure 3 và 4. Script này lo
      xong panel bên trái + toàn bộ Table 5.

-------------------------------------------------------------------------------
CÁCH DÙNG
-------------------------------------------------------------------------------
    pip install "scikit-learn==1.7.2"
    python v40_finish.py check     # ~20 giây, KHÔNG nạp model
    python v40_finish.py run       # nạp 10 bundle, export + Table 5 + hình

`check` phải ĐẠT trước khi `run`. Nó bắt mọi lỗi đã tốn thời gian mấy ngày qua
(cột group biến mất, sai version sklearn, split lệch, tagger lệch) trong 20
giây thay vì sau 8 phút Optuna hay sau khi export xong 10 file vô dụng.
===============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# FIX PICKLE / __main__ — phải đứng TRƯỚC mọi joblib.load().
# Bundle được pickle khi chạy trực tiếp `python train_v40.py`, nên các class
# (BaseLearnersPool, FeaturePipeline, TextPreprocessor...) mang module
# "__main__". Ở đây "__main__" là file này, nên joblib.load() sẽ báo
# AttributeError: Can't get attribute 'BaseLearnersPool' on module '__main__'.
# (Giữ nguyên cách xử lý bạn đã tự tìm ra — nó đúng.)
# -----------------------------------------------------------------------------
import train_v40 as _v40
for _n in dir(_v40):
    if not _n.startswith("_"):
        setattr(sys.modules["__main__"], _n, getattr(_v40, _n))

# =============================================================================
# CONFIG — sửa cho khớp máy bạn
# =============================================================================
SEEDS     = [42, 7, 123, 2024, 99, 555, 2026, 31337, 8080, 12321]
CLASSES   = ["negative", "neutral", "positive"]
GROUP_COL = "near_dup_group_id"
TEXT_COL  = "comment"
LABEL_COL = "sentiment"

DATA_CSV   = os.path.join("input_data", "dataset_cleaned.csv")
V40_BUNDLE = os.path.join("models", "production", "v40", "seed_{seed}")
PHO_JSON   = "results_phobert_multiseed.json"
OUT_CSV    = os.path.join("reports", "v40_probs", "v40_test_seed_{seed}.csv")
FIG_DIR    = os.path.join("reports", "figures_final")
OUT_DIR    = os.path.join("reports", "final_analysis")

FIDELITY_TOL = 0.002
REF_SEED     = 42          # seed tham chiếu cho Figure 3 & 4

# (Train, Calibration, Test) ghi trong CẢ HAI log — đã kiểm chứng khớp tuyệt
# đối ở 10/10 seed, cả kích thước phân hoạch lẫn support từng lớp.
EXPECTED_SPLIT = {
    42: (14849, 5125, 5315), 7: (15267, 5145, 4877), 123: (14993, 5149, 5147),
    2024: (15140, 5141, 5008), 99: (15224, 5174, 4891), 555: (15251, 4979, 5059),
    2026: (15273, 4939, 5077), 31337: (14940, 5131, 5218), 8080: (15260, 5140, 4889),
    12321: (15087, 5047, 5155),
}
# Support test theo lớp (negative, neutral, positive) — từ classification report
# của cả hai log. 30 con số này là bằng chứng các DÒNG test giống hệt nhau.
EXPECTED_SUPPORT = {
    42: (2046, 1402, 1867), 7: (1834, 1181, 1862), 123: (1960, 1346, 1841),
    2024: (2005, 1223, 1780), 99: (1789, 1288, 1814), 555: (1878, 1324, 1857),
    2026: (1925, 1286, 1866), 31337: (1868, 1454, 1896), 8080: (1844, 1252, 1793),
    12321: (1981, 1240, 1934),
}
# F1-macro / accuracy trên TEST ghi trong train_v40_20260720_193018.log
LOGGED_V40 = {
    42: (0.8672, 0.8702), 7: (0.8775, 0.8792), 123: (0.8613, 0.8640),
    2024: (0.8576, 0.8624), 99: (0.8776, 0.8814), 555: (0.8532, 0.8555),
    2026: (0.8669, 0.8706), 31337: (0.8727, 0.8752), 8080: (0.8691, 0.8705),
    12321: (0.8637, 0.8683),
}

# =============================================================================
# BỘ GÁN NHÃN PHÂN NHÓM — COPY NGUYÊN TỪ train_phobert.py::_tag_phenomena()
# =============================================================================
# ĐỪNG "cải tiến" phần này. Mọi thay đổi (so khớp theo biên từ thay cho chuỗi
# con, đổi tokenizer, đổi `<` thành `<=`) sẽ làm hai hàng Table 5 đo trên hai
# tập con khác nhau — đúng lỗi mà phản biện đã bắt.
#
# Ba điểm dễ sửa sai, ghi rõ ở đây:
#   • trai_chieu so khớp CHUỖI CON, không theo biên từ → "mà" bắt cả "màu",
#     "mà cả". Đổi sang biên từ làm lệch ~290 dòng ở seed 42 (2005 → 1714).
#   • cau_ngan dùng `t.split()` thuần và `< 5` (tức tối đa 4 token), KHÔNG
#     phải `<= 5` và KHÔNG qua TextPreprocessor._tokenize_for_lexicon.
#   • HEDGE_WORDS ở đây KHÔNG phải từ rào đón nhận thức ("có lẽ", "hình như").
#     Nó gom các cách nói ĐÁNH GIÁ DÈ DẶT / TRUNG TÍNH. Lưu ý cho bài báo:
#     đoạn phân tích nhóm hedge ở Mục 5.4 hiện đang lập luận bằng từ rào đón
#     nhận thức — không khớp bản chất dữ liệu của nhóm và cần viết lại. Chính
#     nhóm này giải thích trực tiếp lợi thế của AERS-Pro ở lớp neutral, vì
#     "tạm", "bình thường", "cũng ổn" là ngôn ngữ của lớp neutral. Đáng chú ý:
#     "bình thường" có trong teencode_map của V40 ("bt") nhưng KHÔNG có trong
#     positive_words lẫn negative_words → rơi đúng vào vùng mà đặc trưng chỉ
#     báo trung tính (ngưỡng T) phải xử lý.
NEGATION_WORDS = {'không', 'chẳng', 'chả', 'đâu', 'chưa', 'k', 'ko', 'chớ',
                  'hok', 'hem'}
HEDGE_WORDS = {'tạm', 'tạm được', 'cũng được', 'bình thường', 'cũng ổn',
               'tàm tạm', 'không đến nỗi', 'cũng tạm'}
CONTRAST_MARKERS = ('nhưng', 'mà', 'tuy nhiên', 'ai ngờ', 'vậy mà', 'thế mà')
SHORT_MAX_TOKENS = 5      # điều kiện là len(toks) < SHORT_MAX_TOKENS


def tag_subgroups(raw_texts):
    """Gán 4 phân nhóm cho văn bản THÔ (cột 'comment', chưa tiền xử lý).

    Bản sao chính xác của train_phobert.py::_tag_phenomena, vector hoá theo lô.
    Đã kiểm chứng: tái lập đúng 40/40 con số n trong log PhoBERT.

    Các nhóm KHÔNG loại trừ nhau và KHÔNG phân hoạch tập test.
    """
    a, b, c, d = [], [], [], []
    for t in raw_texts:
        t = str(t).lower()
        toks = t.split()
        a.append(any(w in NEGATION_WORDS for w in toks))
        b.append(any(h in t for h in HEDGE_WORDS))
        c.append(any(m in t for m in CONTRAST_MARKERS))
        d.append(len(toks) < SHORT_MAX_TOKENS)
    return {"phu_dinh": np.array(a, bool), "hedge": np.array(b, bool),
            "trai_chieu": np.array(c, bool), "cau_ngan": np.array(d, bool)}


GROUPS = ["phu_dinh", "hedge", "trai_chieu", "cau_ngan"]

# =============================================================================
# TIỆN ÍCH
# =============================================================================
def load_corpus():
    if not os.path.exists(DATA_CSV):
        sys.exit(f"[DỪNG] Không thấy {DATA_CSV}. Sửa DATA_CSV ở đầu file.")
    df = pd.read_csv(DATA_CSV)
    for need in (TEXT_COL, LABEL_COL, GROUP_COL):
        if need not in df.columns:
            sys.exit(f"[DỪNG] {DATA_CSV} thiếu cột '{need}'. "
                     f"Cột hiện có: {list(df.columns)}")
    if df[GROUP_COL].isna().any():
        sys.exit(f"[DỪNG] Cột '{GROUP_COL}' có "
                 f"{int(df[GROUP_COL].isna().sum())} giá trị NaN — thường do "
                 f"input_data/ có nhiều file bị pd.concat gộp lẫn. Để lại đúng "
                 f"1 file đã chạy dedup_clustering.py.")
    return df


def pho_subgroups():
    if not os.path.exists(PHO_JSON):
        sys.exit(f"[DỪNG] Không thấy {PHO_JSON}.")
    J = json.load(open(PHO_JSON, encoding="utf-8"))["metrics_by_seed"]
    miss = [k for k, v in J.items() if "subgroup_results" not in v]
    if miss:
        sys.exit(f"[DỪNG] {PHO_JSON} thiếu subgroup_results ở seed: {miss}")
    return {int(k): v["subgroup_results"] for k, v in J.items()}


# =============================================================================
# LỆNH 1 — CHECK  (không nạp model)
# =============================================================================
def cmd_check(verbose=True):
    import sklearn
    P = (lambda s: print(s)) if verbose else (lambda s: None)
    ok = True

    P("=" * 76)
    P("CHECK — điều kiện tiên quyết (không nạp model)")
    P("=" * 76)

    P(f"[1/4] scikit-learn = {sklearn.__version__}")
    mm = tuple(int(v) for v in sklearn.__version__.split(".")[:2])
    if mm >= (1, 8):
        P("      ❌ 1.8+ đổi thuật toán StratifiedGroupKFold → split KHÔNG tái lập.")
        P('         Sửa: pip install "scikit-learn==1.7.2"')
        ok = False
    elif mm < (1, 5):
        P("      ⚠️  Thấp hơn dải đã kiểm chứng (1.5–1.7). Xem mục [3/4] để biết kết quả.")
    else:
        P("      ✅ Nằm trong dải đã xác nhận tái lập đúng split (1.5–1.7).")

    P("[2/4] Dữ liệu")
    df = load_corpus()
    y = LabelEncoder().fit_transform(df[LABEL_COL])
    texts = df[TEXT_COL].astype(str).to_numpy()
    ng = df[GROUP_COL].nunique()
    P(f"      {len(df)} dòng | {ng} nhóm '{GROUP_COL}' | 0 NaN")
    if len(df) != 25289:
        P("      ⚠️  Bài báo dùng 25.289 dòng.")
    if ng != 9476:
        P("      ⚠️  Mục 5.1 của bài ghi 9.476 nhóm.")

    P("[3/4] Split — đối chiếu 10 seed (kích thước) và 30 con số support")
    splits, nbad = {}, 0
    for s in SEEDS:
        tr, cal, te = _v40.split_data_v40(df, y, group_col=GROUP_COL, seed=s)
        splits[s] = te
        got = (len(tr), len(cal), len(te))
        sup = tuple(int((y[te] == k).sum()) for k in range(3))
        bad = (got != EXPECTED_SPLIT[s]) or (sup != EXPECTED_SUPPORT[s])
        if bad:
            nbad += 1
            P(f"      ❌ seed {s:>5}: {got} / support {sup}")
            P(f"                cần {EXPECTED_SPLIT[s]} / support {EXPECTED_SUPPORT[s]}")
    if nbad:
        P(f"      ❌ {nbad}/10 seed lệch → mọi kiểm định bắt cặp sẽ vô hiệu.")
        ok = False
    else:
        P("      ✅ 10/10 seed khớp kích thước VÀ support → các dòng test giống")
        P("         hệt tập test của PhoBERT.")

    P("[4/4] Bộ gán nhãn phân nhóm — đối chiếu 40 con số n với log PhoBERT")
    if nbad == 0:
        PH = pho_subgroups()
        tot = 0
        for g in GROUPS:
            hits = sum(int(tag_subgroups(texts[splits[s]])[g].sum()) == PH[s][g]["n"]
                       for s in SEEDS)
            tot += hits
            P(f"      {'✅' if hits == 10 else '❌'} {g:<12} khớp {hits}/10 seed")
        if tot == 40:
            P("      ✅ 40/40 khớp chính xác → định nghĩa trùng hàm gốc của PhoBERT,")
            P("         Table 5 là so sánh bắt cặp hợp lệ.")
        else:
            P(f"      ❌ chỉ {tot}/40 khớp. ĐỪNG chạy `run`: hai hàng Table 5 sẽ đo")
            P("         trên hai tập con khác nhau. Kiểm lại NEGATION_WORDS /")
            P("         HEDGE_WORDS / CONTRAST_MARKERS / SHORT_MAX_TOKENS so với")
            P("         train_phobert.py::_tag_phenomena().")
            ok = False
    else:
        P("      (bỏ qua vì split đang lệch)")

    P("=" * 76)
    P("✅ ĐẠT — chạy được `python v40_finish.py run`." if ok
      else "❌ CHƯA ĐẠT — sửa các dòng ❌ ở trên.")
    P("=" * 76)
    return ok


# =============================================================================
# LỆNH 2 — RUN
# =============================================================================
def ece_bins(conf, correct, M=10):
    """ECE theo đúng quy ước Mục 4.2: B1 = [0, 0.1], Bm = (0.1(m-1), 0.1m]."""
    edges = np.linspace(0, 1, M + 1)
    rows, e, n = [], 0.0, len(conf)
    for m in range(M):
        lo, hi = edges[m], edges[m + 1]
        sel = (conf >= lo) & (conf <= hi) if m == 0 else (conf > lo) & (conf <= hi)
        k = int(sel.sum())
        if k:
            a, c = correct[sel].mean(), conf[sel].mean()
            e += k / n * abs(a - c)
            rows.append((lo, hi, k, a, c))
        else:
            rows.append((lo, hi, 0, np.nan, np.nan))
    return e, rows


def cmd_run():
    if not cmd_check(verbose=True):
        sys.exit("\n[DỪNG] `check` chưa đạt. Không nạp model, không export.")

    for d in (os.path.dirname(OUT_CSV.format(seed=42)), FIG_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)

    df = load_corpus()
    y = LabelEncoder().fit_transform(df[LABEL_COL])
    texts_all = df[TEXT_COL].astype(str).to_numpy()
    labels_all = df[LABEL_COL].astype(str).to_numpy()

    # ---------------- EXPORT + CHECK-FIDELITY ----------------
    print("\n" + "=" * 76)
    print("EXPORT + CHECK-FIDELITY — export có tái tạo đúng số đã log không?")
    print("=" * 76)
    store, nbad = {}, 0
    for seed in SEEDS:
        bundle = _v40.load_v40_bundle(V40_BUNDLE.format(seed=seed))
        _, _, te = _v40.split_data_v40(df, y, group_col=GROUP_COL, seed=seed)
        te_texts = texts_all[te].tolist()
        y_true = labels_all[te]

        y_pred, proba = _v40.predict_v40(te_texts, bundle)
        # Sắp lại cột theo CLASSES thay vì tin ngầm thứ tự LabelEncoder của seed.
        order = [list(bundle["le"].classes_).index(c) for c in CLASSES]
        proba = np.asarray(proba, dtype=float)[:, order]
        y_pred = np.asarray(y_pred).astype(str)

        f1 = f1_score(y_true, y_pred, average="macro")
        acc = accuracy_score(y_true, y_pred)
        lf1, lacc = LOGGED_V40[seed]
        good = abs(f1 - lf1) <= FIDELITY_TOL and abs(acc - lacc) <= FIDELITY_TOL
        nbad += (not good)
        print(f"  [{'OK ' if good else 'LỆCH'}] seed {seed:>5}  n={len(te):>5}  "
              f"F1={f1:.4f} (log {lf1:.4f})  acc={acc:.4f} (log {lacc:.4f})")

        pd.DataFrame({
            "y_true": y_true, "y_pred": y_pred,
            "p_negative": proba[:, 0], "p_neutral": proba[:, 1],
            "p_positive": proba[:, 2], "text": te_texts,
        }).to_csv(OUT_CSV.format(seed=seed), index=False, encoding="utf-8-sig")
        store[seed] = dict(y_true=y_true, y_pred=y_pred, proba=proba,
                           texts=np.array(te_texts, dtype=object))

    if nbad:
        sys.exit(
            f"\n[DỪNG] {nbad}/10 seed không tái tạo được số đã log. KHÔNG vẽ hình.\n"
            "  Đây đúng loại lỗi của Figure 4 hiện tại (0,8561/0,8524 so với\n"
            "  0,8702/0,8672 đã log). Kiểm predict_v40 theo thứ tự:\n"
            "   1. Có áp calibration head (Eq. 3-4) ở đúng 6 seed đã 'ÁP DỤNG'\n"
            "      (42, 7, 99, 31337, 8080, 12321)?\n"
            "   2. MinMaxScaler nhánh lexicon: nạp cái đã fit hay fit lại?\n"
            "   3. Cluster k-means: nạp cái đã fit hay fit lại?\n"
            "   4. bundle['best_T'] đúng của seed đó (2.0 ở 8 seed, 1.5 ở seed\n"
            "      8080, 3.0 ở seed 12321)?\n")
    print("  ✅ 10/10 seed khớp log → export dùng được.")

    # ---------------- TABLE 5 ----------------
    PH = pho_subgroups()
    rows = []
    for g in GROUPS:
        av, fv, nn = [], [], []
        for seed in SEEDS:
            d = store[seed]
            m = tag_subgroups(d["texts"])[g]
            nn.append(int(m.sum()))
            av.append(accuracy_score(d["y_true"][m], d["y_pred"][m]))
            fv.append(f1_score(d["y_true"][m], d["y_pred"][m], average="macro"))
        av, fv = np.array(av), np.array(fv)
        ap = np.array([PH[s][g]["accuracy"] for s in SEEDS])
        fp = np.array([PH[s][g]["f1_macro"] for s in SEEDS])
        dd = fv - fp
        _, p_t = stats.ttest_rel(fv, fp)
        try:
            _, p_w = stats.wilcoxon(fv, fp)
        except Exception:
            p_w = float("nan")
        se = dd.std(ddof=1) / np.sqrt(len(dd))
        tc = stats.t.ppf(0.975, len(dd) - 1)
        rows.append(dict(
            subgroup=g, mean_n=float(np.mean(nn)),
            n_min=int(min(nn)), n_max=int(max(nn)),
            v40_acc=av.mean(), v40_acc_sd=av.std(ddof=1),
            v40_f1=fv.mean(), v40_f1_sd=fv.std(ddof=1),
            pho_acc=ap.mean(), pho_acc_sd=ap.std(ddof=1),
            pho_f1=fp.mean(), pho_f1_sd=fp.std(ddof=1),
            dF1=dd.mean(), ci_lo=dd.mean() - tc * se, ci_hi=dd.mean() + tc * se,
            dz=dd.mean() / dd.std(ddof=1), p_ttest=p_t, p_wilcoxon=p_w))

    print("\n" + "=" * 96)
    print("TABLE 5 — DÁN 8 Ô [fill] TỪ HAI CỘT 'AERS-Pro' DƯỚI ĐÂY")
    print("=" * 96)
    print(f"{'Subgroup':<12}{'n':>7}{'  AERS-Pro Acc':>20}{'  AERS-Pro F1':>20}"
          f"{'  PhoBERT F1':>20}")
    for r in rows:
        print(f"{r['subgroup']:<12}{r['mean_n']:>7.0f}"
              f"{r['v40_acc']:>13.4f}±{r['v40_acc_sd']:.4f}"
              f"{r['v40_f1']:>13.4f}±{r['v40_f1_sd']:.4f}"
              f"{r['pho_f1']:>13.4f}±{r['pho_f1_sd']:.4f}")

    print("\nCâu cho đoạn phân tích 5.4 (thay chỗ [To complete: ...]):")
    for r in rows:
        sig = "p < 0.001" if r["p_ttest"] < 0.001 else f"p = {r['p_ttest']:.4f}"
        print(f"  {r['subgroup']:<12} ΔF1 = {r['dF1']:+.4f}  "
              f"CI 95% [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  {sig}  "
              f"dz = {r['dz']:+.2f}  Wilcoxon p = {r['p_wilcoxon']:.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "table5_unified.csv"),
                              index=False, encoding="utf-8-sig")

    # ---------------- HÌNH — panel AERS-Pro ----------------
    d = store[REF_SEED]
    conf = d["proba"].max(1)
    correct = (d["y_pred"] == d["y_true"]).astype(float)
    e, brows = ece_bins(conf, correct)

    fig, (ax, axc) = plt.subplots(2, 1, figsize=(5.4, 6.2),
                                  gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1.2, label="Perfect calibration")
    first = True
    for lo, hi, k, a, c in brows:
        if not k:
            continue
        x = (lo + hi) / 2
        ax.bar(x, a, width=0.093, color="#2f6fb3", edgecolor="black", lw=0.5, zorder=2)
        ax.plot([x, x], sorted([a, c]), color="#d9534f", lw=2.4, zorder=3,
                label="Gap" if first else None)
        axc.bar(x, k, width=0.093, color="#8a8a8a", edgecolor="black", lw=0.5)
        first = False
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    ax.set_ylabel("Accuracy"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"AERS-Pro (seed {REF_SEED}, n={len(conf):,})   ECE = {e:.4f}")
    axc.set_ylabel("Count")
    axc.set_xlabel("Confidence (max predicted probability)")
    axc.grid(alpha=0.3)
    p3 = os.path.join(FIG_DIR, "fig3_left_aerspro.png")
    plt.tight_layout(); plt.savefig(p3, dpi=200, bbox_inches="tight"); plt.close()

    cm = confusion_matrix(d["y_true"], d["y_pred"], labels=CLASSES)
    pct = cm / cm.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}\n({pct[i, j]:.1f}%)", ha="center",
                    va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() * 0.55 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"AERS-Pro (seed {REF_SEED}, n={cm.sum():,})   "
                 f"acc = {np.trace(cm) / cm.sum():.4f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    p4 = os.path.join(FIG_DIR, "fig4_left_aerspro.png")
    plt.tight_layout(); plt.savefig(p4, dpi=200, bbox_inches="tight"); plt.close()

    # ---------------- ĐỐI CHIẾU VỚI FIGURE 4 ĐANG IN TRONG BÀI ----------------
    f1_ref = f1_score(d["y_true"], d["y_pred"], average="macro")
    print("\n" + "=" * 96)
    print("ĐỐI CHIẾU VỚI FIGURE 4 ĐANG IN TRONG BÀI THẢO")
    print("=" * 96)
    print(f"  Ma trận MỚI      : acc = {np.trace(cm)/cm.sum():.4f}  "
          f"F1-macro = {f1_ref:.4f}  recall neutral = {cm[1,1]/cm[1].sum():.3f}  "
          f"ECE = {e:.4f}")
    print("  Hình CŨ trong bài: acc = 0.8561  F1-macro = 0.8524  "
          "recall neutral = 0.752  ECE = 0.0234")
    print("  Log gốc seed 42  : acc = 0.8702  F1-macro = 0.8672  "
          "recall neutral = 0.81   ECE = 0.0109")
    print("\nHÌNH:", p3)
    print("     ", p4)
    print("CSV :", OUT_CSV.format(seed="*"))
    print("BẢNG:", os.path.join(OUT_DIR, "table5_unified.csv"))
    print("\nGửi 10 file CSV này (hoặc chỉ seed 42) để ghép panel PhoBERT, dựng")
    print("cặp hình 2 panel cuối cùng và cập nhật vào bản Word.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        sys.exit(0 if cmd_check() else 1)
    elif cmd == "run":
        cmd_run()
    else:
        sys.exit(__doc__)
