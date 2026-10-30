# =====================================================================
# KAGGLE NOTEBOOK — CÁC CELL COPY-PASTE ĐỂ CHẠY ABLATION
# ---------------------------------------------------------------------
# Thiết lập notebook TRƯỚC KHI chạy (panel bên phải):
#   Accelerator  : None (CPU)          <- KHÔNG chọn GPU, sẽ mất quota vô ích
#                                          và giới hạn phiên tụt 12h -> 9h
#   Environment  : Pin to original environment
#   Internet     : Off
#   Add Input    : dataset chứa gói này
#
# Chạy bằng Save Version -> Save & Run All (Commit), KHÔNG chạy phiên
# tương tác (phiên tương tác tự ngắt khi đóng tab hoặc để yên ~20-60 phút).
# =====================================================================


# ─────────────────────────────────────────────────────────────────────
# CELL 1 — Kiểm tra môi trường. CHẠY TRƯỚC, đừng bỏ qua.
# ─────────────────────────────────────────────────────────────────────
import sklearn, numpy, scipy, pandas, sys
print("python  ", sys.version.split()[0])
print("sklearn ", sklearn.__version__, "  <- PHẢI là 1.5.x / 1.6.x / 1.7.x")
print("numpy   ", numpy.__version__)
print("scipy   ", scipy.__version__)
print("pandas  ", pandas.__version__)
import multiprocessing; print("vCPU    ", multiprocessing.cpu_count())
get_ipython().system('lscpu | grep -E "^Model name|^CPU\\(s\\)"')
get_ipython().system('free -g | head -2')

# Nếu sklearn là 1.8.x -> DỪNG LẠI. Cách chia dữ liệu sẽ KHÁC lần chạy gốc
# và mọi số trong bài sẽ lệch. Đổi Environment sang image cũ hơn, hoặc bật
# Internet rồi: !pip install -q scikit-learn==1.6.1  (sau đó restart kernel).


# ─────────────────────────────────────────────────────────────────────
# CELL 2 — Copy gói sang chỗ ghi được (/kaggle/input chỉ đọc)
# ─────────────────────────────────────────────────────────────────────
import shutil, os, glob

# Tự tìm thư mục gói trong /kaggle/input
cands = glob.glob("/kaggle/input/*/LexEval_V40_Ablation_Ready") + \
        glob.glob("/kaggle/input/*/*/LexEval_V40_Ablation_Ready")
assert cands, "Không thấy LexEval_V40_Ablation_Ready trong /kaggle/input — kiểm tra Add Input"
SRC, DST = cands[0], "/kaggle/working/aers"
print("nguồn:", SRC)

if os.path.exists(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST)
os.chdir(DST)
print("cwd:", os.getcwd())

for f in ["train_v40.py", "run_ablation.py", "seeds_config.py",
          "results_v40_multiseed.json", "input_data/dataset_cleaned.csv"]:
    assert os.path.exists(f), f"THIẾU {f}"
print("OK — đủ file.")

# Xác nhận patch đã có trong gói
import inspect, importlib, train_v40
importlib.reload(train_v40)
params = list(inspect.signature(train_v40.train_v40).parameters)
for need in ["use_cluster_features", "use_neutral_indicator", "n_clusters"]:
    print(("  OK   " if need in params else "  THIẾU ") + need)


# ─────────────────────────────────────────────────────────────────────
# CELL 3 — Bằng chứng trùng partition (vài giây, cho Mục 4.1 của bài)
# ─────────────────────────────────────────────────────────────────────
get_ipython().system('python verify_identical_partitions.py --data input_data/dataset_cleaned.csv --group_col near_dup_group_id')
get_ipython().system('cp partition_identity_report.csv /kaggle/working/ 2>/dev/null || true')
# Kết quả PHẢI là: 30/30 tổ hợp trùng khớp, và mỗi seed báo 0 group chồng lấn.


# ─────────────────────────────────────────────────────────────────────
# CELL 4 — SMOKE TEST (~30 phút). CỬA KIỂM SOÁT, đừng bỏ qua.
# ─────────────────────────────────────────────────────────────────────
get_ipython().system('python run_ablation.py --variants full --seeds 42 --n_trials 30 --out_json /kaggle/working/smoke.json')

# ĐIỀU KIỆN BẮT BUỘC: F1 của full/seed 42 phải ra ≈ 0.8680
#   khớp   -> patch không làm lệch gì, và môi trường tái lập được lần chạy 13/08
#   lệch   -> DỪNG. Đừng chạy 10 giờ. Báo lại cho tôi con số nhận được.
import json
print("F1 seed 42 =", round(json.load(open("/kaggle/working/smoke.json"))
                            ["results"]["full"]["42"]["f1"], 4), " (mốc: 0.8680)")


# ─────────────────────────────────────────────────────────────────────
# CELL 5 — Kiểm tra cờ ablation có tác dụng thật (~30 phút)
# ─────────────────────────────────────────────────────────────────────
get_ipython().system('python run_ablation.py --variants no_neutral --seeds 42 --n_trials 30 --out_json /kaggle/working/smoke2.json')
# Trong log PHẢI thấy   : [ABLATION] use_neutral_indicator=False
# Và KHÔNG được thấy    : [FEAT-6] Ngưỡng neutral T tối ưu


# =====================================================================
# CHIA THÀNH 3 PHIÊN — Kaggle giới hạn 12 giờ/phiên, mỗi seed ~31 phút
# =====================================================================

# ─── PHIÊN A (~5 giờ) ────────────────────────────────────────────────
# `full` KHÔNG chạy lại: lấy từ results_v40_multiseed.json (chính là lần
# chạy 13/08 mà bài báo đang báo cáo), tiết kiệm đúng 5 giờ.
get_ipython().system('python run_ablation.py --variants full,no_cluster --reuse_full results_v40_multiseed.json --n_trials 30 --out_json /kaggle/working/ablation_A.json --out_csv /kaggle/working/ablation_A.csv')


# ─── PHIÊN B (~10 giờ) ───────────────────────────────────────────────
# Add Input -> Your Work -> output của phiên A, rồi sửa <slug-A> bên dưới.
get_ipython().system('python run_ablation.py --variants no_neutral,no_both --merge_from /kaggle/input/<slug-A>/ablation_A.json --n_trials 30 --out_json /kaggle/working/ablation_B.json --out_csv /kaggle/working/ablation_B.csv')


# ─── PHIÊN C (~10 giờ) ───────────────────────────────────────────────
get_ipython().system('python run_ablation.py --variants k2,k6 --merge_from /kaggle/input/<slug-B>/ablation_B.json --n_trials 30 --out_json /kaggle/working/ablation_FINAL.json --out_csv /kaggle/working/ablation_FINAL.csv')


# ─────────────────────────────────────────────────────────────────────
# NẾU MỘT PHIÊN BỊ HẾT GIỜ GIỮA ĐƯỜNG
# ─────────────────────────────────────────────────────────────────────
# Script ghi file sau MỖI seed nên kết quả dở dang vẫn còn trong output.
# Phiên sau: copy file đó sang /kaggle/working rồi thêm --resume
#
# import shutil
# shutil.copy("/kaggle/input/<slug-bi-ngat>/ablation_B.json",
#             "/kaggle/working/ablation_B.json")
# !python run_ablation.py --variants no_neutral,no_both --n_trials 30 --resume \
#     --out_json /kaggle/working/ablation_B.json \
#     --out_csv  /kaggle/working/ablation_B.csv


# ─────────────────────────────────────────────────────────────────────
# CELL CUỐI — đóng gói kết quả để tải về
# ─────────────────────────────────────────────────────────────────────
import shutil, os
os.makedirs("/kaggle/working/ablation_out", exist_ok=True)
for f in ["ablation_FINAL.json", "ablation_FINAL.csv",
          "ablation_A.json", "ablation_B.json", "partition_identity_report.csv"]:
    src = f"/kaggle/working/{f}"
    if os.path.exists(src):
        shutil.copy(src, "/kaggle/working/ablation_out/")
shutil.make_archive("/kaggle/working/ablation_out", "zip", "/kaggle/working/ablation_out")
print("→ tải /kaggle/working/ablation_out.zip rồi gửi lại")
