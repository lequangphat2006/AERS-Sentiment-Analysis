# =====================================================================
# KAGGLE — TRAIN LẠI V40 (SEED 42) VÀ ĐO THỜI GIAN / KÍCH THƯỚC (T10)
# ---------------------------------------------------------------------
# Thiết lập notebook:
#   Accelerator : None (CPU)   — đo trên CPU, KHÔNG chọn GPU
#   Internet    : On (cần pip install scikit-learn==1.6.1, underthesea...)
#   Add Input   : dataset chứa AERS-Pro-V40_full.zip (đã giải nén)
#   Chạy bằng Save Version -> Save & Run All (Commit)
#
# Mục tiêu: train lại 10 seed (models/production/v40/seed_*/*.pkl) + đo T10 trên seed 42, rồi
# tải 1 file zip duy nhất về (kết quả nằm ở /kaggle/working/v40_t10_outputs.zip).
# Ghi chú cho bài: số đo là trên MỘT seed (42), CPU của Kaggle (cell 1 in tên CPU).
# =====================================================================


# ── CELL 1 — môi trường (CHẠY TRƯỚC) ─────────────────────────────────
import subprocess, sys, os
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-1500:])
sh("pip install -q 'scikit-learn==1.6.1' underthesea lightgbm optuna joblib")
# SAU LỆNH TRÊN: Run -> Restart session (hoặc Restart kernel), rồi chạy lại từ CELL 2 trở đi.
# (Nếu sklearn đã đúng 1.6.1 từ đầu thì không cần restart.)


# ── CELL 2 — kiểm tra phiên bản và ghi lại (bằng chứng môi trường) ───
import sklearn, numpy, scipy, pandas, multiprocessing, platform
print("python  ", sys.version.split()[0])
print("sklearn ", sklearn.__version__, "  <- PHẢI là 1.6.1")
assert sklearn.__version__ == "1.6.1", "Sai phiên bản scikit-learn, restart session rồi chạy lại"
print("numpy", numpy.__version__, "| scipy", scipy.__version__, "| pandas", pandas.__version__)
print("vCPU", multiprocessing.cpu_count(), "|", platform.platform())
sh("lscpu | grep -E '^Model name|^CPU\\(s\\)|^Thread|^Core'")
sh("free -g | head -2")
os.makedirs("/kaggle/working/out", exist_ok=True)
sh("pip freeze > /kaggle/working/out/pip_freeze.txt; "
   "lscpu > /kaggle/working/out/lscpu.txt; free -m > /kaggle/working/out/free.txt; "
   "wc -l /kaggle/working/out/pip_freeze.txt")


# ── CELL 3 — copy gói sang chỗ ghi được ──────────────────────────────
import glob, shutil
cands = glob.glob("/kaggle/input/*/AERS-Pro-V40") + glob.glob("/kaggle/input/*/*/AERS-Pro-V40")
assert cands, "Không thấy thư mục AERS-Pro-V40 trong /kaggle/input — kiểm tra lại Add Input"
SRC = cands[0]; DST = "/kaggle/working/run"
if os.path.exists(DST): shutil.rmtree(DST)
shutil.copytree(SRC, DST)
os.chdir(DST); sys.path.insert(0, DST)
print("cwd:", os.getcwd()); print(sorted(os.listdir(".")))
assert os.path.exists("input_data/dataset_cleaned.csv")
# dọn kết quả cũ trong bản copy để không nhầm với lần train mới
for d in ("models", "logs"):
    shutil.rmtree(d, ignore_errors=True)


# ── CELL 4 — TRAIN LẠI TOÀN BỘ 10 SEED (đúng cấu hình bài: 30 trial Optuna) ──
# Ước tính 3–5 giờ trên CPU Kaggle (máy gốc ~1156 s/seed). Phải chạy ở chế độ
# Commit (Save & Run All). Kết quả từng seed được ghi ngay sau khi xong, nên nếu
# phiên bị ngắt vẫn giữ được các seed đã chạy. Muốn chạy thử nhanh chỉ để đo T10:
# đổi RUN_SEEDS = [42].
import json, time
from train_v40 import train_v40
from seeds_config import SEEDS
RUN_SEEDS = list(SEEDS)            # ví dụ chỉ đo T10: RUN_SEEDS = [42]
res_all = {}
for i, s in enumerate(RUN_SEEDS, 1):
    t0 = time.time()
    m = train_v40(input_dir="input_data", n_trials=30, seed=s, group_col="near_dup_group_id")
    res_all[str(s)] = {"metrics": m, "wall_clock_sec": time.time() - t0}
    print(f"[{i}/{len(RUN_SEEDS)}] seed={s} F1={m.get('f1'):.4f} ECE={m.get('ece'):.4f} "
          f"size_mb={m.get('model_size_mb'):.2f} train_sec={m.get('training_time_sec'):.0f}")
    json.dump(res_all, open("/kaggle/working/out/retrain_metrics_by_seed.json", "w"), indent=1, default=str)
# So với bài: F1 seed 42 gốc = 0.8680, trung bình 10 seed gốc = 0.8665. Có thể lệch
# vài phần nghìn vì phiên bản underthesea khác. KHÔNG sao; chỉ cần báo lại con số.


# ── CELL 5 — ĐO KÍCH THƯỚC và ĐỘ TRỄ SUY LUẬN (CPU) ──────────────────
import numpy as np, pandas as pd, statistics, joblib
from train_v40 import load_v40_bundle, predict_v40

prod = "models/production/v40/seed_42"
files = sorted(os.listdir(prod))
sizes = {f: os.path.getsize(os.path.join(prod, f)) / 1024**2 for f in files}
print("Kích thước từng file (MB):"); [print(f"  {k:32s} {v:8.3f}") for k, v in sizes.items()]
print("TỔNG (MB):", round(sum(sizes.values()), 3))

bundle = load_v40_bundle(prod)
df = pd.read_csv("input_data/dataset_cleaned.csv", encoding="utf-8-sig")
col = "comment" if "comment" in df.columns else df.columns[0]
rng = np.random.RandomState(0)
texts = df[col].astype(str).sample(1500, random_state=0).tolist()

predict_v40(texts[:20], bundle)                      # khởi động (warm-up), không tính
lat1 = []                                            # độ trễ từng câu (batch = 1)
for t in texts[:300]:
    s = time.perf_counter(); predict_v40([t], bundle); lat1.append((time.perf_counter() - s) * 1000)
batch_res = {}
for B in (32, 256, 1000):                            # thông lượng theo lô
    s = time.perf_counter(); predict_v40(texts[:B], bundle); dt = time.perf_counter() - s
    batch_res[B] = {"sec": dt, "ms_per_sample": dt / B * 1000, "samples_per_sec": B / dt}
res = {
    "model_size_total_mb": sum(sizes.values()), "model_size_by_file_mb": sizes,
    "latency_single_ms": {"median": statistics.median(lat1), "p95": float(np.percentile(lat1, 95)),
                          "mean": float(np.mean(lat1)), "n": len(lat1)},
    "batch": batch_res,
}
print(json.dumps(res, indent=1))
json.dump(res, open("/kaggle/working/out/inference_benchmark.json", "w"), indent=1)


# ── CELL 6 (TÙY CHỌN) — đo PhoBERT trên CPU để so sánh công bằng ─────
# Chỉ cần Internet On. Độ trễ không phụ thuộc trọng số đã fine-tune nên dùng
# đầu phân loại khởi tạo ngẫu nhiên là đủ; chỉ để lấy tốc độ CPU và dung lượng.
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from underthesea import word_tokenize
    tok = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    mdl = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base-v2", num_labels=3).eval()
    torch.set_num_threads(multiprocessing.cpu_count())
    seg = [word_tokenize(t, format="text") for t in texts[:300]]
    def run(batch):
        enc = tok(batch, truncation=True, max_length=192, padding=True, return_tensors="pt")
        with torch.no_grad(): return mdl(**enc).logits
    run(seg[:4])
    lat = []
    for t in seg[:100]:
        s = time.perf_counter(); run([t]); lat.append((time.perf_counter() - s) * 1000)
    s = time.perf_counter(); run(seg[:32]); b32 = time.perf_counter() - s
    nparam = sum(p.numel() for p in mdl.parameters())
    pho = {"params": nparam, "fp32_size_mb": nparam * 4 / 1024**2,
           "latency_single_ms": {"median": statistics.median(lat), "p95": float(np.percentile(lat, 95)), "n": len(lat)},
           "batch32_ms_per_sample": b32 / 32 * 1000}
    print(json.dumps(pho, indent=1))
    json.dump(pho, open("/kaggle/working/out/phobert_cpu_benchmark.json", "w"), indent=1)
except Exception as e:
    print("Bỏ qua cell 6:", repr(e))


# ── CELL 7 — đóng gói (TẢI FILE NÀY VỀ) ──────────────────────────────
shutil.copytree("models/production/v40", "/kaggle/working/out/models_v40", dirs_exist_ok=True)   # mọi seed đã train
for f in glob.glob("logs/*.log"): shutil.copy(f, "/kaggle/working/out/")
shutil.make_archive("/kaggle/working/v40_t10_outputs", "zip", "/kaggle/working/out")
print("Xong:", "/kaggle/working/v40_t10_outputs.zip", round(os.path.getsize("/kaggle/working/v40_t10_outputs.zip") / 1024**2, 1), "MB")
