#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation.py
===============
Trả lời góp ý của GVHD ở Mục 3:

  "Nếu xem các thành phần bổ sung như cluster indicator và neutral-region
   indicator là đóng góp của mô hình, nên bổ sung ablation study để chứng
   minh ảnh hưởng của chúng."
  "Giải thích cơ sở chọn 4 clusters (k = 4) trong K-means."

Script chạy các biến thể của AERS-Pro trên ĐÚNG 10 seed và ĐÚNG cách chia
group-aware như bài báo, rồi sinh bảng ablation kèm kiểm định bắt cặp
(paired t-test + Wilcoxon + CI 95% + dz) so với mô hình đầy đủ.

TRƯỚC KHI CHẠY: PHẢI ÁP PATCH cho train_v40.py
----------------------------------------------
train_v40() hiện ĐÃ CÓ cờ `use_cluster_features` nhưng CHƯA CÓ cờ tắt
neutral-region indicator và chưa cho đổi k. Xem file PATCH_train_v40.md
(3 chỗ sửa, khoảng 10 dòng). Nếu chưa áp patch, script sẽ báo lỗi và dừng
chứ không lặng lẽ chạy sai.

CÁCH CHẠY
---------
    python run_ablation.py                    # đầy đủ: 6 biến thể x 10 seed
    python run_ablation.py --variants full,no_cluster,no_neutral
    python run_ablation.py --seeds 42,7,123   # chạy thử 3 seed trước
    python run_ablation.py --n_trials 30      # giữ 30 để so được với bài báo

THỜI GIAN
---------
~12 phút/seed (đo từ log gốc: 720.8 ± 30.0 s). Đầy đủ 6 biến thể x 10 seed
≈ 12 giờ. Khuyến nghị: chạy `--variants full,no_cluster,no_neutral,no_both`
(4 biến thể ≈ 8 giờ) qua đêm, phần k-sensitivity chạy sau nếu còn thời gian.

LƯU Ý VỀ BIẾN THỂ "full"
------------------------
Vẫn nên chạy lại `full` thay vì lấy số từ results_v40_multiseed.json, để mọi
biến thể dùng cùng một môi trường/phiên bản thư viện. Nếu muốn tiết kiệm,
dùng --reuse_full để đọc lại từ JSON cũ, nhưng phải nói rõ trong bài rằng
baseline ablation lấy từ lần chạy gốc.

ĐẦU RA
------
  ablation_results.json   — kết quả thô từng biến thể x từng seed
  ablation_table.csv      — bảng đã tính mean ± SD + kiểm định, dán vào bài
"""

import argparse
import inspect
import json
import os
import sys
import time

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Định nghĩa các biến thể ablation
# ---------------------------------------------------------------------------
# Mỗi biến thể là một bộ tham số truyền vào train_v40().
VARIANTS = {
    # Mô hình đầy đủ — mốc so sánh
    "full": dict(use_cluster_features=True, use_neutral_indicator=True, n_clusters=4),

    # Bỏ từng thành phần (đây là 2 ablation cô yêu cầu)
    "no_cluster": dict(use_cluster_features=False, use_neutral_indicator=True, n_clusters=4),
    "no_neutral": dict(use_cluster_features=True, use_neutral_indicator=False, n_clusters=4),
    "no_both": dict(use_cluster_features=False, use_neutral_indicator=False, n_clusters=4),

    # Độ nhạy theo k — trả lời "cơ sở chọn k = 4".
    # Nếu F1 gần như không đổi qua k, luận điểm là: cluster one-hot chỉ là
    # đặc trưng điều kiện thô cho meta-learner tuyến tính, kết quả KHÔNG phụ
    # thuộc việc chọn k, nên k = 4 là lựa chọn thiết kế cố định hợp lý chứ
    # không phải một tuyên bố về cấu trúc cụm trong dữ liệu.
    "k2": dict(use_cluster_features=True, use_neutral_indicator=True, n_clusters=2),
    "k6": dict(use_cluster_features=True, use_neutral_indicator=True, n_clusters=6),
}

METRICS = ["f1", "accuracy", "ece", "brier"]


def check_patch_applied():
    """Dừng sớm nếu train_v40.py chưa được vá — tránh chạy 12 giờ rồi mới biết sai."""
    if not os.path.exists("train_v40.py"):
        sys.exit("LỖI: không thấy train_v40.py trong thư mục hiện tại "
                 f"({os.getcwd()}).\n"
                 "run_ablation.py phải nằm CÙNG CẤP với train_v40.py và seeds_config.py, "
                 "ngang hàng với thư mục input_data/.")
    try:
        from train_v40 import train_v40
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", "") or str(e)
        if missing in ("train_v40", "seeds_config"):
            sys.exit(f"LỖI: không import được '{missing}'.\n"
                     "Đặt run_ablation.py cùng thư mục với train_v40.py và seeds_config.py.")
        sys.exit(f"LỖI: train_v40.py cần thư viện '{missing}' nhưng môi trường chưa có.\n"
                 f"Đây KHÔNG phải lỗi đặt sai thư mục.\n"
                 f"  Kaggle : thư viện này thường có sẵn — kiểm tra đã chọn đúng Environment "
                 f"(Pin to original environment) chưa.\n"
                 f"  Máy cá nhân: pip install {missing}\n"
                 f"  (bộ đầy đủ: scikit-learn>=1.5,<1.8 numpy pandas scipy lightgbm optuna "
                 f"joblib underthesea)")
    except ImportError as e:
        sys.exit(f"LỖI khi import train_v40.py: {e}")

    params = inspect.signature(train_v40).parameters
    missing = [p for p in ("use_cluster_features", "use_neutral_indicator", "n_clusters")
               if p not in params]
    if missing:
        sys.exit(
            "LỖI: train_v40() còn thiếu tham số: " + ", ".join(missing) + "\n"
            "→ Áp PATCH_train_v40.md trước khi chạy ablation.\n"
            "   (use_cluster_features đã có sẵn trong bản gốc;\n"
            "    use_neutral_indicator và n_clusters cần thêm.)"
        )
    return train_v40


def paired_stats(a, b):
    """a = biến thể ablation, b = mô hình full. Trả về hiệu, CI, dz, p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    sd = d.std(ddof=1)
    out = {"delta": float(d.mean()), "n": int(n)}
    if sd == 0 or n < 3:
        out.update(ci_lo=None, ci_hi=None, dz=None, p_ttest=None, p_wilcoxon=None)
        return out
    se = sd / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    out["ci_lo"] = float(d.mean() - tc * se)
    out["ci_hi"] = float(d.mean() + tc * se)
    out["dz"] = float(d.mean() / sd)
    out["p_ttest"] = float(stats.ttest_rel(a, b).pvalue)
    try:
        out["p_wilcoxon"] = float(stats.wilcoxon(a, b).pvalue)
    except Exception:
        out["p_wilcoxon"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="input_data")
    ap.add_argument("--group_col", default="near_dup_group_id")
    ap.add_argument("--n_trials", type=int, default=30,
                    help="Giữ 30 để so sánh được với Bảng 2 của bài báo.")
    ap.add_argument("--seeds", default="",
                    help="Danh sách seed cách nhau bởi dấu phẩy. Mặc định: seeds_config.SEEDS")
    ap.add_argument("--variants", default="full,no_cluster,no_neutral,no_both",
                    help="Biến thể cần chạy. Thêm k2,k6 cho phần k-sensitivity.")
    ap.add_argument("--reuse_full", default="",
                    help="Đường dẫn results_v40_multiseed.json để dùng lại kết quả 'full' "
                         "thay vì chạy lại (tiết kiệm ~2 giờ).")
    ap.add_argument("--resume", action="store_true",
                    help="Đọc lại --out_json và BỎ QUA các (biến thể, seed) đã chạy xong. "
                         "Dùng khi phiên Kaggle trước bị hết giờ.")
    ap.add_argument("--merge_from", default="",
                    help="Danh sách file ablation_results.json của các phiên TRƯỚC, cách nhau "
                         "bởi dấu phẩy, để gộp vào kết quả phiên này (dùng trên Kaggle: thêm "
                         "output của commit trước làm dataset input rồi trỏ vào đây).")
    ap.add_argument("--out_json", default="ablation_results.json")
    ap.add_argument("--out_csv", default="ablation_table.csv")
    args = ap.parse_args()

    train_v40 = check_patch_applied()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        try:
            from seeds_config import SEEDS
            seeds = list(SEEDS)
        except ImportError:
            seeds = [42, 7, 123, 2024, 99, 555, 2026, 31337, 8080, 12321]

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        sys.exit(f"LỖI: biến thể không hợp lệ: {unknown}. "
                 f"Chọn trong: {list(VARIANTS)}")

    results = {}

    # Gộp kết quả từ các phiên chạy trước (Kaggle: mỗi commit là 1 phiên)
    for src in [x.strip() for x in args.merge_from.split(",") if x.strip()]:
        try:
            with open(src, encoding="utf-8") as f:
                old = json.load(f).get("results", {})
            for v, per_seed in old.items():
                results.setdefault(v, {}).update(per_seed)
            print(f"↺ Gộp {sum(len(x) for x in old.values())} kết quả từ {src}")
        except Exception as e:
            sys.exit(f"LỖI: không đọc được --merge_from '{src}': {e}")

    if args.resume and os.path.exists(args.out_json):
        with open(args.out_json, encoding="utf-8") as f:
            old = json.load(f).get("results", {})
        for v, per_seed in old.items():
            results.setdefault(v, {}).update(per_seed)
        print(f"↺ Resume: đã có {sum(len(x) for x in results.values())} kết quả, sẽ bỏ qua.")

    # Dùng lại kết quả 'full' từ lần chạy gốc nếu được yêu cầu
    if args.reuse_full and "full" in variants:
        with open(args.reuse_full, encoding="utf-8") as f:
            old = json.load(f)["metrics_by_seed"]
        results["full"] = {str(s): old[str(s)] for s in seeds if str(s) in old}
        variants = [v for v in variants if v != "full"]
        print(f"↺ Dùng lại 'full' từ {args.reuse_full} "
              f"({len(results['full'])} seed)\n")

    # Ghi tiến độ ngay sau mỗi seed — mất điện giữa chừng vẫn giữ được
    def save():
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"seeds": seeds, "n_trials": args.n_trials,
                       "group_col": args.group_col,
                       "variant_config": {v: VARIANTS[v] for v in VARIANTS},
                       "results": results}, f, ensure_ascii=False, indent=2)

    todo = [(v, sd) for v in variants for sd in seeds
            if str(sd) not in results.get(v, {})]
    skipped = len(variants) * len(seeds) - len(todo)
    if skipped:
        print(f"Bỏ qua {skipped} cặp (biến thể, seed) đã có kết quả.")
    total = len(todo)
    done = 0
    t0 = time.time()
    for v in variants:
        results.setdefault(v, {})
        for s in seeds:
            if (v, s) not in todo:
                continue
            done += 1
            eta = ((time.time() - t0) / max(done - 1, 1)) * (total - done + 1) / 3600
            print(f"[{done}/{total}] biến thể={v} seed={s}"
                  + (f"  (ETA ~{eta:.1f} giờ)" if done > 1 else ""))
            results[v][str(s)] = train_v40(
                input_dir=args.input_dir,
                n_trials=args.n_trials,
                seed=s,
                group_col=args.group_col,
                **VARIANTS[v],
            )
            save()

    # -----------------------------------------------------------------------
    # Tổng hợp bảng ablation
    # -----------------------------------------------------------------------
    def series(v, k):
        return [results[v][str(s)][k] for s in seeds if str(s) in results.get(v, {})]

    lines = ["variant,metric,mean,sd,delta_vs_full,ci_lo,ci_hi,dz,p_ttest,p_wilcoxon"]
    print("\n" + "=" * 78)
    print("BẢNG ABLATION — mean ± SD qua %d seed (SD dùng ddof = 1)" % len(seeds))
    print("=" * 78)
    have_full = "full" in results and series("full", "f1")

    for v in results:
        print(f"\n{v}")
        for k in METRICS:
            a = series(v, k)
            if not a:
                continue
            mean, sd = float(np.mean(a)), float(np.std(a, ddof=1))
            if v == "full" or not have_full:
                print(f"   {k:<10} {mean:.4f} ± {sd:.4f}")
                lines.append(f"{v},{k},{mean:.6f},{sd:.6f},,,,,,")
            else:
                b = series("full", k)
                st = paired_stats(a, b[:len(a)])
                pt = st["p_ttest"]
                star = "" if pt is None else (" *" if pt < 0.05 else " n.s.")
                print(f"   {k:<10} {mean:.4f} ± {sd:.4f}   "
                      f"Δ vs full = {st['delta']:+.4f}"
                      + (f"  CI[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]"
                         f"  dz={st['dz']:+.2f}  p={pt:.4g}{star}"
                         if pt is not None else ""))
                lines.append(",".join([
                    v, k, f"{mean:.6f}", f"{sd:.6f}", f"{st['delta']:.6f}",
                    "" if st["ci_lo"] is None else f"{st['ci_lo']:.6f}",
                    "" if st["ci_hi"] is None else f"{st['ci_hi']:.6f}",
                    "" if st["dz"] is None else f"{st['dz']:.4f}",
                    "" if st["p_ttest"] is None else f"{st['p_ttest']:.6g}",
                    "" if st["p_wilcoxon"] is None else f"{st['p_wilcoxon']:.6g}",
                ]))

    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "=" * 78)
    print(f"Đã ghi: {args.out_json}  và  {args.out_csv}")
    print("=" * 78)
    print("""
CÁCH ĐỌC KẾT QUẢ ĐỂ VIẾT VÀO BÀI
--------------------------------
• Nếu no_cluster / no_neutral làm F1 GIẢM có ý nghĩa (p < 0.05):
  → hai thành phần này là đóng góp thật, giữ nguyên tuyên bố ở Mục 3.1 và
    trích số Δ + CI vào bảng ablation mới (đề xuất đặt là Bảng 6).

• Nếu KHÔNG giảm có ý nghĩa:
  → phải sửa lại cách trình bày ở Mục 3.1 và Mục 5.3. Cụ thể, câu ở Mục 5.3
    "the neutral indicator is thus not a generic regulariser but the component
    that carries this subgroup" KHÔNG còn giữ được, vì đó là suy luận từ
    cơ chế chứ chưa phải bằng chứng ablation. Trung thực hơn: mô tả chúng là
    thành phần thiết kế, và nêu trong limitation rằng ablation không cho thấy
    ảnh hưởng đo được ở mức tổng hợp.

• no_neutral nên xem thêm F1 RIÊNG LỚP NEUTRAL, không chỉ F1-macro: theo
  Bảng 3 lợi thế của AERS-Pro tập trung ở lớp neutral (+0.0200) và ở subgroup
  câu ngắn, nên ảnh hưởng có thể bị pha loãng trong F1-macro. Nếu cần, sửa
  train_v40() để trả thêm per-class F1 vào dict kết quả.

• k2 / k6 gần như bằng k4 → viết: "the result is insensitive to k over
  k ∈ {2, 4, 6}", kèm bảng chẩn đoán chọn k (silhouette/CH/DB) ở phụ lục.
""")


if __name__ == "__main__":
    main()
