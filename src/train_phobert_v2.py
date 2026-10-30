#!/usr/bin/env python
# ═══════════════════════════════════════════════════════════════════════════
# PHOBERT BASELINE — cho AERS-Pro V40/V41 (so sánh với SOTA transformer)
# ───────────────────────────────────────────────────────────────────────────
# Mục tiêu: fine-tune vinai/phobert-base(-v2) trên CÙNG một cách chia dữ liệu,
# CÙNG bộ hàm phân tích (subgroup, distribution check, entropy/confidence,
# ECE/Brier) như train_v41.py, để kết quả có thể đưa thẳng vào bảng so sánh
# V40 / V41 / PhoBERT mà không lệch protocol đánh giá.
#
# BẢN CẬP NHẬT NÀY: chạy ĐA SEED (10 seed, dùng CHUNG seeds_config.py với
# V40/V41) thay vì chỉ 1 seed như bản trước — để có mean±std thật sự có thể
# đặt cạnh Table so sánh V40/V41 (vốn đã chạy đủ 10 seed). Cấu trúc vòng lặp
# đa seed, lưu kết quả sau MỖI seed, và format results_*.json copy Y HỆT
# pattern trong train_v40.py / train_v41.py (xem phần "CHẠY CHÍNH — ĐA SEED"
# ở cuối 2 file đó) để 3 file JSON có thể gộp thẳng vào cùng một bảng.
#
# ĐÃ ĐIỀU CHỈNH CHO MÁY THẬT SẼ DÙNG ĐỂ CHẠY: Kaggle Notebook, GPU T4 x2
# (2 x NVIDIA Tesla T4, 16GB VRAM/GPU, kiến trúc Turing TU104 — CÓ Tensor
# Core, khác hẳn GTX 16-series). Không còn ràng buộc VRAM 4GB nữa nên "bung
# lụa": tăng batch/max_length, tắt gradient_checkpointing mặc định để tối đa
# tốc độ. Hệ quả cụ thể cho lựa chọn tham số dưới đây:
#   • 2 GPU: HF Trainer tự phát hiện torch.cuda.device_count()==2 và dùng
#     DataParallel (không cần torchrun/DDP trong Kaggle Notebook — chạy
#     `python train_phobert.py` bình thường là đủ). per_device_batch_size
#     áp dụng CHO MỖI GPU, nên effective batch = batch_size * 2 * grad_accum.
#   • fp16 mixed precision: T4 CÓ Tensor Core nên fp16 vừa tiết kiệm VRAM
#     vừa TĂNG TỐC thật (khác hẳn GTX 1650 Ti trước đây) — luôn bật.
#   • gradient_checkpointing=False làm MẶC ĐỊNH: với 16GB/GPU, phobert-base
#     ở max_length=128, batch=32 không cần đánh đổi tốc độ lấy VRAM nữa. Vẫn
#     giữ flag --gradient_checkpointing nếu bạn tăng batch/max_length quá tay
#     và gặp OOM.
#   • batch_size=32 (mỗi GPU) + grad_accum=1 (effective batch = 32*2*1 = 64)
#     làm MẶC ĐỊNH — tận dụng cả 16GB x2, KHÔNG phải cấu hình dè dặt nữa.
#   • max_length=192 (ĐO THỰC TẾ trên dataset_cleaned.csv, 25289 dòng): p50=17
#     từ, p95=65, p97=76, p99=127 từ (word-level, chưa tính BPE) — 128 sẽ cắt
#     cụt ngay ở đuôi p99, trong khi VRAM T4x2 còn dư nhiều nên nâng lên 192
#     để có biên an toàn qua cả bước tách từ (underthesea) + BPE (PhoBERT).
#   • torch.backends.cudnn.benchmark=True: input luôn padding cố định
#     max_length nên bật benchmark giúp cuDNN chọn thuật toán phù hợp.
#   • PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True + TOKENIZERS_PARALLELISM=false
#     (set TRƯỚC khi import torch/transformers): giảm phân mảnh bộ nhớ CUDA
#     khi chạy liên tục 10 seed, và tránh cảnh báo/deadlock tokenizer đa luồng
#     hay gặp trong môi trường notebook (Kaggle) khi kết hợp với dataloader
#     nhiều worker.
#
# HỖ TRỢ CHẠY TRÊN KAGGLE (phiên có giới hạn thời gian, /kaggle/working
# KHÔNG tự động giữ qua các phiên trừ khi bạn lưu lại làm output/dataset):
#   • Resume tự động: nếu results_phobert_multiseed.json đã có sẵn TRONG
#     phiên hiện tại (bị ngắt giữa chừng nhưng notebook chưa restart), script
#     tự bỏ qua seed đã xong. Dùng --no_resume nếu muốn ép chạy lại từ đầu.
#   • --resume_from <path>: nếu bạn bắt đầu một PHIÊN KAGGLE MỚI (session cũ
#     đã hết giờ/bị đóng), tải results_phobert_multiseed.json của phiên trước
#     lên làm Kaggle Input Dataset rồi trỏ --resume_from vào đường dẫn đó
#     (vd. /kaggle/input/phobert-results-seed-p1/results_phobert_multiseed.json)
#     — script sẽ nạp lại làm điểm xuất phát rồi chạy tiếp seed còn thiếu.
#     ⚠️ Nhớ tải results_phobert_multiseed.json + thư mục logs/ về máy (hoặc
#     "Save Version" làm Output) TRƯỚC KHI phiên Kaggle hết giờ, nếu không sẽ
#     mất toàn bộ tiến độ của phiên đó.
#   • Sau mỗi seed, thư mục checkpoint tạm (models/phobert/checkpoints/seed_*)
#     bị xoá để không tích luỹ dung lượng qua 10 lần chạy — dùng
#     --keep_checkpoints nếu muốn giữ lại để debug.
#   • Model production (models/production/phobert/) bị GHI ĐÈ sau mỗi seed —
#     Y HỆT hành vi của train_v40.py/train_v41.py: mục tiêu của vòng lặp đa
#     seed là mean±std các chỉ số trong results_phobert_multiseed.json, không
#     phải để chọn ra "model tốt nhất để deploy".
#
# QUAN TRỌNG — Word segmentation cho PhoBERT:
#   PhoBERT được train trên văn bản tiếng Việt đã TÁCH TỪ (word-segmented,
#   nối bằng "_"), khác với TF-IDF ở V40/V41 vốn hoạt động ở mức token thô.
#   Nếu bỏ qua bước này, hiệu năng PhoBERT sẽ tụt rất nhiều so với số liệu
#   công bố trong paper gốc. Script này dùng underthesea.word_tokenize để
#   tách từ trước khi đưa vào tokenizer của PhoBERT.
#
#   FIX (2026-07, phát hiện trên dataset_cleaned.csv thực tế): ~2.8%
#   (699/25289) comment là review nhiều-trường kiểu Shopee ghép bằng '\n'
#   KHÔNG có khoảng trắng quanh dấu xuống dòng (vd "...tuyệt lắm\nChất
#   liệu:tốt..."). underthesea.word_tokenize không coi '\n' là ranh giới từ
#   như dấu cách nên sẽ dính 2 từ ở "dòng" trước/sau thành 1 token rác
#   ('tuyệt_Đúng') không có trong vocab PhoBERT. Đã thêm _normalize_whitespace()
#   để gộp mọi chuỗi whitespace (\n, \r, \t, nbsp) thành một dấu cách TRƯỚC
#   khi tách từ, tránh dính chữ ở đúng nhóm review nhiều-trường này.
# ═══════════════════════════════════════════════════════════════════════════

import os
# Phải set TRƯỚC khi import torch/transformers để có hiệu lực.
# - expandable_segments: giảm phân mảnh bộ nhớ CUDA khi chạy liên tục 10 seed
#   trong cùng 1 tiến trình dài (mỗi seed nạp/giải phóng model riêng).
# - TOKENIZERS_PARALLELISM=false: tránh cảnh báo/deadlock tokenizer đa luồng
#   khi kết hợp với dataloader_num_workers>0 trong môi trường notebook.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import re
import sys, glob, json, time, random, shutil, argparse, logging, logging.handlers
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           TrainingArguments, Trainer, EarlyStoppingCallback,
                           set_seed)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from underthesea import word_tokenize
    HAS_UNDER = True
except ImportError:
    HAS_UNDER = False

VERSION = 'phobert'

if torch.cuda.is_available():
    # Input luôn padding cố định về max_length -> kích thước tensor không đổi
    # giữa các batch, nên bật benchmark an toàn và có lợi cho tốc độ trên T4.
    torch.backends.cudnn.benchmark = True

# ═══════════════════════════════════════════════════════════════════
# LOGGING (cùng định dạng với train_v40.py / train_v41.py) — MỘT file log
# duy nhất cho toàn bộ phiên chạy đa seed (dễ theo dõi ETA/tổng thời gian).
# ═══════════════════════════════════════════════════════════════════
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
_log_file = os.path.join(LOG_DIR, f"train_{VERSION}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger('SentimentPhoBERT')
logger.setLevel(logging.DEBUG)
_fh = logging.handlers.RotatingFileHandler(_log_file, maxBytes=15_000_000, backupCount=5, encoding='utf-8')
_ch = logging.StreamHandler(sys.stdout)
_fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
for h in [_fh, _ch]:
    h.setFormatter(_fmt)
    logger.addHandler(h)

for d in ['models', 'models/production', 'reports']:
    os.makedirs(d, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# 1. DATA LOADING — GIỮ NGUYÊN LOGIC NHƯ train_v41.py (không đổi để so
#    sánh công bằng: cùng file input, cùng cách nhận diện cột). Có thêm
#    group_col để đọc cột 'near_dup_group_id' — ĐỒNG BỘ với train_v40.py/
#    train_v41.py (Trục 1, mục 2.2), bắt buộc để so sánh 3 hệ thống trên
#    cùng một split sạch, không tái phạm leakage đã audit trước đó.
# ═══════════════════════════════════════════════════════════════════
def load_data(input_dir='input_data', group_col=None):
    files = glob.glob(os.path.join(input_dir, '*.csv')) + glob.glob(os.path.join(input_dir, '*.xlsx'))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy .csv/.xlsx nào trong '{input_dir}'. "
                                 f"Copy đúng dữ liệu đã dùng cho train_v40/v41 vào đây để so sánh công bằng.")
    dfs = []
    for f in files:
        df = pd.read_csv(f) if f.endswith('.csv') else pd.read_excel(f)
        df.rename(columns=lambda c: 'comment' if c.lower() in ('comment', 'text')
                   else ('sentiment' if c.lower() in ('sentiment', 'label') else c), inplace=True)
        cols = ['comment', 'sentiment']
        if group_col and group_col in df.columns and group_col not in cols:
            cols.append(group_col)
        if 'comment' in df.columns and 'sentiment' in df.columns:
            dfs.append(df[cols])
    if not dfs:
        raise ValueError("Không file nào có đủ cột 'comment' và 'sentiment'.")
    return pd.concat(dfs, ignore_index=True)


def split_data_phobert(df, y, group_col=None, seed=42):
    """
    [SPLIT-16 / PATCH] ĐỒNG BỘ với split_data_v40() HIỆN TẠI (train_v40.py).

    LỊCH SỬ: bản trước của hàm này gọi GroupShuffleSplit và docstring ghi
    "ĐỒNG BỘ với split_data_v40()/split_data_v41()" — đúng tại thời điểm đó.
    Nhưng sau đó split_data_v40() được vá theo note [SPLIT-16]: GroupShuffleSplit
    group-aware nhưng KHÔNG hỗ trợ stratify theo nhãn, nên V40 đổi sang lấy 1
    fold từ StratifiedGroupKFold (group-aware VÀ xấp xỉ cân bằng lớp).
    split_data_phobert() KHÔNG được cập nhật theo — hệ quả: với cùng 1 seed,
    V40 và PhoBERT sinh ra 3 tập Train/Calibration/Test có kích thước khác
    nhau (đối chiếu log gốc, seed=42: V40 Train=14849/Cal=5125/Test=5315 vs
    PhoBERT (bản cũ) Train=15111/Cal=5003/Test=5175 — 0/10 seed khớp nhau).
    Điều này làm câu "exact same train/calibration/test partition" ở Mục 3.3
    của bài báo không đúng với thực tế đã chạy.

    BẢN VÁ NÀY: copy nguyên logic StratifiedGroupKFold(n_splits=5 rồi 4) từ
    split_data_v40() hiện tại — cùng thứ tự gọi, cùng random_state=seed, cùng
    2 tầng fallback (GroupShuffleSplit rồi stratified train_test_split) — để
    V40 và PhoBERT lại sinh ra đúng cùng 1 cách chia trên cùng seed, khôi phục
    đúng tuyên bố "exact same partition". Cần chạy lại toàn bộ 10 seed sau khi
    áp bản vá này; số liệu PhoBERT hiện có (results_phobert_multiseed.json)
    được sinh ra từ bản GroupShuffleSplit cũ nên KHÔNG dùng chung được với
    lần chạy mới.
    """
    idx_all = np.arange(len(df))
    if group_col is not None and group_col in df.columns:
        groups = df[group_col].values
        try:
            sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            idx_tr_cal, idx_te = next(iter(sgkf1.split(idx_all, y, groups)))
            sgkf2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
            rel_tr, rel_cal = next(iter(sgkf2.split(idx_tr_cal, y[idx_tr_cal], groups[idx_tr_cal])))
            idx_tr, idx_cal = idx_tr_cal[rel_tr], idx_tr_cal[rel_cal]
            logger.info(f"[SPLIT-16] Chia dữ liệu bằng StratifiedGroupKFold (lấy 1 fold, 60/20/20) theo "
                        f"cột '{group_col}' — group-aware VÀ xấp xỉ cân bằng lớp.")
            return idx_tr, idx_cal, idx_te
        except Exception as e:
            logger.warning(f"[SPLIT-16] StratifiedGroupKFold lỗi ({e}) — fallback GroupShuffleSplit "
                            f"(group-aware nhưng KHÔNG stratify).")
            try:
                gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                idx_tr_cal, idx_te = next(gss1.split(idx_all, y, groups))
                gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
                rel_tr, rel_cal = next(gss2.split(idx_tr_cal, y[idx_tr_cal], groups[idx_tr_cal]))
                idx_tr, idx_cal = idx_tr_cal[rel_tr], idx_tr_cal[rel_cal]
                logger.info(f"Chia dữ liệu bằng GroupShuffleSplit theo cột '{group_col}' (Trục 1, mục 2.2) "
                            f"— tránh 2 biến thể cùng nguồn rơi vào 2 tập khác nhau.")
                return idx_tr, idx_cal, idx_te
            except Exception as e2:
                logger.warning(f"GroupShuffleSplit cũng lỗi ({e2}) — fallback stratified split ngẫu nhiên.")

    logger.warning("KHÔNG có cột group/near_dup_group_id hợp lệ → dùng stratified split ngẫu nhiên như "
                    "trước đây. CẢNH BÁO (Trục 1, mục 2.2): số liệu này sẽ KHÔNG so sánh công bằng được "
                    "với V40/V41 nếu hai bản đó đã chạy group-aware — chạy dedup_clustering.py trước.")
    idx_tr_cal, idx_te = train_test_split(idx_all, test_size=0.2, stratify=y, random_state=seed)
    idx_tr, idx_cal = train_test_split(idx_tr_cal, test_size=0.25, stratify=y[idx_tr_cal], random_state=seed)
    return idx_tr, idx_cal, idx_te


_NEWLINE_RE = re.compile(r'[\r\n]+')
_WHITESPACE_RE = re.compile(r'\s+')


def _normalize_whitespace(t: str) -> str:
    """Chuẩn hoá whitespace TRƯỚC khi tách từ.
    PHÁT HIỆN TỪ dataset_cleaned.csv THỰC TẾ (~2.8% / 699/25289 dòng): nhiều
    comment là review Shopee ghép nhiều field liên tiếp bằng '\\n', vd:
        "Chất lượng sản phẩm:gất tuyệt\\nĐúng với mô tả:đúng\\nMũ siêu đẹp..."
    Hai vấn đề với underthesea.word_tokenize khi gặp '\\n':
      (1) nếu KHÔNG có khoảng trắng quanh '\\n' (chữ dính sát), 2 từ ở 2 "dòng"
          liền kề bị dính thành 1 token rác không có trong vocab PhoBERT
          (vd 'tuyệt\\nĐúng' -> 'tuyệt_Đúng' nếu chỉ thay '\\n' bằng dấu cách).
      (2) ngay cả khi CÓ khoảng trắng, thiếu dấu câu ở ranh giới giữa 2 field
          khiến underthesea coi 2 field là một câu liền mạch và vẫn có thể
          ghép nhầm compound qua ranh giới field (đã kiểm chứng thực nghiệm).
    → Thay '\\n'/'\\r' bằng " . " (chấm + khoảng trắng hai bên) để tạo ranh
    giới câu rõ ràng cho underthesea, sau đó gộp các whitespace còn lại
    (tab, nbsp, nhiều dấu cách liên tiếp) về một dấu cách."""
    t = _NEWLINE_RE.sub(' . ', str(t))
    return _WHITESPACE_RE.sub(' ', t).strip()


SEGMENTER = 'underthesea'   # 'underthesea' (mac dinh, nhu lan chay goc) hoac 'rdr' (VnCoreNLP RDRSegmenter, khuyen nghi cua VinAI)
_RDR = None


def _rdr_segment(texts):
    """RDRSegmenter qua py_vncorenlp (can Java + tai model, Kaggle can bat Internet). CHUA duoc kiem thu o day."""
    global _RDR
    if _RDR is None:
        import py_vncorenlp
        save_dir = os.path.abspath('vncorenlp_model'); os.makedirs(save_dir, exist_ok=True)
        py_vncorenlp.download_model(save_dir=save_dir)
        _RDR = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
    out = []
    for t in texts:
        try:
            out.append(" ".join(_RDR.word_segment(_normalize_whitespace(t))))
        except Exception:
            out.append(_normalize_whitespace(t))
    return out


def word_segment_batch(texts):
    if SEGMENTER == 'rdr':
        return _rdr_segment(texts)
    """Tách từ tiếng Việt bằng underthesea trước khi đưa vào PhoBERT tokenizer.
    Nếu thiếu underthesea, cảnh báo rõ vì hiệu năng PhoBERT sẽ giảm đáng kể."""
    if not HAS_UNDER:
        logger.warning("KHÔNG có underthesea → PhoBERT sẽ nhận văn bản CHƯA tách từ, "
                        "hiệu năng có thể thấp hơn số liệu công bố trong paper gốc PhoBERT. "
                        "Cài đặt: pip install underthesea")
        return [_normalize_whitespace(t) for t in texts]
    out = []
    for t in texts:
        try:
            out.append(word_tokenize(_normalize_whitespace(t), format="text"))
        except Exception:
            out.append(_normalize_whitespace(t))
    return out


# ═══════════════════════════════════════════════════════════════════
# 2. DATASET
# ═══════════════════════════════════════════════════════════════════
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length',
                                    max_length=max_length, return_tensors=None)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'f1_macro': f1_score(labels, preds, average='macro'),
        'accuracy': accuracy_score(labels, preds),
    }


# ═══════════════════════════════════════════════════════════════════
# 3. PHÂN TÍCH MỞ RỘNG — SAO CHÉP Y HỆT train_v41.py ĐỂ SO SÁNH ĐÚNG
#    PROTOCOL (subgroup, phân phối dự đoán, entropy/confidence)
# ═══════════════════════════════════════════════════════════════════
NEGATION_WORDS = {'không', 'chẳng', 'chả', 'đâu', 'chưa', 'k', 'ko', 'chớ', 'hok', 'hem'}
HEDGE_WORDS = {'tạm', 'tạm được', 'cũng được', 'bình thường', 'cũng ổn', 'tàm tạm', 'không đến nỗi', 'cũng tạm'}


def _tag_phenomena(text: str) -> dict:
    t = str(text).lower()
    toks = t.split()
    has_neg = any(w in NEGATION_WORDS for w in toks)
    has_hedge = any(h in t for h in HEDGE_WORDS)
    has_contrast = any(m in t for m in ('nhưng', 'mà', 'tuy nhiên', 'ai ngờ', 'vậy mà', 'thế mà'))
    is_short = len(toks) < 5
    return {'phu_dinh': has_neg, 'hedge': has_hedge, 'trai_chieu': has_contrast, 'cau_ngan': is_short}


def error_analysis_by_subgroup(raw_texts, y_true, y_pred, le):
    tags = [_tag_phenomena(t) for t in raw_texts]
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    logger.info("── Phân tích lỗi theo phân nhóm ngữ nghĩa (đồng bộ Trục 5 của V41) ──")
    results = {}
    for key in ['phu_dinh', 'hedge', 'trai_chieu', 'cau_ngan']:
        mask = np.array([tg[key] for tg in tags])
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        f1 = f1_score(y_true[mask], y_pred[mask], average='macro')
        results[key] = {'n': int(mask.sum()), 'accuracy': float(acc), 'f1_macro': float(f1)}
        logger.info(f"  {key:12s}: n={mask.sum():4d}  acc={acc:.3f}  f1_macro={f1:.3f}")
    return results


def prediction_distribution_check(y_pred, y_train, le, warn_threshold=0.10):
    logger.info("── Kiểm tra lệch phân phối dự đoán so với train ──")
    y_pred, y_train = np.asarray(y_pred), np.asarray(y_train)
    flagged = []
    for ci, cname in enumerate(le.classes_):
        p_pred = np.mean(y_pred == ci)
        p_train = np.mean(y_train == ci)
        diff = p_pred - p_train
        flag = abs(diff) > warn_threshold
        if flag:
            flagged.append(str(cname))
        logger.info(f"  Lớp {cname}: train={p_train:.1%}  predict(test)={p_pred:.1%}  "
                    f"lệch={diff:+.1%}{'  ⚠️' if flag else ''}")
    if flagged:
        logger.warning(f"  Lệch phân phối > {warn_threshold:.0%} ở lớp: {flagged}")
    return flagged


def confidence_entropy_stats(probs):
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    confidence = np.max(probs, axis=1)
    stats = {
        'mean_entropy': float(np.mean(entropy)),
        'mean_confidence': float(np.mean(confidence)),
        'std_confidence': float(np.std(confidence)),
    }
    logger.info("── Entropy/độ tin cậy trung bình ──")
    logger.info(f"  mean_entropy={stats['mean_entropy']:.4f}  mean_confidence={stats['mean_confidence']:.4f} "
                f"(±{stats['std_confidence']:.4f})")
    return stats


def compute_ece_brier(probs, y_true, n_bins=10):
    """Tính ECE (Expected Calibration Error) và Brier score — CÙNG quy ước bin
    với train_v40.py/train_v41.py (thống nhất 2026-07: bin đầu [lo,hi], các
    bin sau (lo,hi]) để so sánh calibration công bằng giữa cả 3 hệ thống."""
    y_true = np.asarray(y_true)
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    n_classes = probs.shape[1]
    onehot = np.eye(n_classes)[y_true]
    brier = np.mean(np.sum((probs - onehot) ** 2, axis=1))

    logger.info(f"── Calibration: ECE={ece:.4f}  Brier={brier:.4f} ──")
    return {'ece': float(ece), 'brier': float(brier)}


def save_diagnostic_plots(y_te, pred_te, le, out_dir='reports'):
    try:
        os.makedirs(out_dir, exist_ok=True)
        cm = confusion_matrix(y_te, pred_te)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel('Dự đoán'); ax.set_ylabel('Thực tế'); ax.set_title(f'Confusion Matrix — Test ({VERSION})')
        fig.tight_layout()
        out_path = os.path.join(out_dir, f'confusion_matrix_{VERSION}.png')
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        logger.info(f"📊 Đã lưu confusion matrix vào {out_path}")
    except Exception as e:
        logger.warning(f"Không lưu được confusion matrix: {e}")


def export_phobert_logits(seed, idx_cal, y_cal, logits_cal, probs_cal,
                           idx_te, y_te, logits_te, probs_te,
                           out_dir='reports/phobert_logits'):
    """
    [EXPORT-LOGITS PATCH — Bước 2 của lộ trình nâng cấp] Xuất logits/xác suất
    per-sample cho tập Calibration và Test, dùng làm input cho
    fit_temperature_scaling.py (Temperature Scaling, Bước 5) và cho reliability
    diagram / confusion matrix đối chiếu với V40 (Bước 8).

    QUAN TRỌNG: dùng logits THẬT lấy từ trainer.predict(...).predictions
    (không suy ngược từ prob qua log()), vì log(softmax(z)) chỉ bằng z sai
    khác một hằng số cộng theo mỗi sample — không ảnh hưởng tới việc fit T
    nhưng để đúng đặc tả cột "logit_*" thì nên ghi logits gốc của model.

    row_index là chỉ số dòng TRONG DataFrame gốc đã nạp bởi load_data()
    (tức phần tử của idx_cal/idx_te trả về từ split_data_phobert()) — dùng để
    tra ngược lại văn bản gốc khi làm "phân tích lỗi đối sánh" ở Bước 8.

    Ghi 2 file mỗi seed, đúng đặc tả:
        reports/phobert_logits/phobert_calib_seed_<seed>.csv
        reports/phobert_logits/phobert_test_seed_<seed>.csv
    với các cột: seed, split, row_index, y_true, logit_0, logit_1, logit_2,
    prob_0, prob_1, prob_2.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_classes = probs_cal.shape[1]

    def _save(split_name, idx_orig, y_true, logits, probs):
        logits = np.asarray(logits)
        probs = np.asarray(probs)
        data = {
            'seed': seed,
            'split': split_name,
            'row_index': np.asarray(idx_orig),
            'y_true': np.asarray(y_true),
        }
        for c in range(n_classes):
            data[f'logit_{c}'] = logits[:, c]
        for c in range(n_classes):
            data[f'prob_{c}'] = probs[:, c]
        out_df = pd.DataFrame(data)
        out_path = os.path.join(out_dir, f'phobert_{split_name}_seed_{seed}.csv')
        out_df.to_csv(out_path, index=False, encoding='utf-8')
        logger.info(f"💾 [EXPORT-LOGITS] Đã lưu logits/probs {split_name} (seed={seed}) "
                    f"→ {out_path}  ({len(out_df)} dòng, {n_classes} lớp)")
        return out_path

    path_cal = _save('calib', idx_cal, y_cal, logits_cal, probs_cal)
    path_test = _save('test', idx_te, y_te, logits_te, probs_te)
    return path_cal, path_test


# ═══════════════════════════════════════════════════════════════════
# 4. MAIN PIPELINE — nay nhận tham số trực tiếp (giống train_v40()/
#    train_v41()) thay vì argparse.Namespace, để vòng lặp đa seed ở
#    __main__ có thể gọi train_phobert(seed=s, ...) cho từng seed.
# ═══════════════════════════════════════════════════════════════════
def train_phobert(input_dir='input_data',
                   model_name='vinai/phobert-base-v2',
                   max_length=192,
                   batch_size=32,
                   grad_accum=1,
                   epochs=4,
                   lr=2e-5,
                   seed=42,
                   fp16=True,
                   gradient_checkpointing=False,
                   freeze_layers=0,
                   cleanup_checkpoints=True,
                   group_col=None,
                   sweep_mode=False):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    t_start = time.time()

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    effective_batch = batch_size * max(n_gpu, 1) * grad_accum
    logger.info("=" * 60)
    logger.info(f"🚀 BẮT ĐẦU FINE-TUNE {model_name} (seed={seed})")
    logger.info(f"ENV torch={torch.__version__} transformers={__import__('transformers').__version__} "
                f"underthesea={'yes' if HAS_UNDER else 'NO'} segmenter={SEGMENTER} sweep_mode={sweep_mode}")
    logger.info(f"   max_length={max_length}  batch_size/GPU={batch_size}  n_gpu={n_gpu}  "
                f"grad_accum={grad_accum} (effective batch={effective_batch})  fp16={fp16}  "
                f"gradient_checkpointing={gradient_checkpointing}  freeze_layers={freeze_layers}")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("⚠️ KHÔNG phát hiện GPU — script sẽ chạy trên CPU, RẤT CHẬM với transformer. "
                        "Kiểm tra lại Kaggle Notebook đã bật Accelerator = GPU T4 x2 chưa.")
    else:
        for i in range(n_gpu):
            vram_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}  VRAM: {vram_gb:.1f} GB")
        if n_gpu > 1:
            logger.info(f"🚀 Phát hiện {n_gpu} GPU — Trainer sẽ tự dùng DataParallel, "
                        f"per_device_batch_size={batch_size} áp dụng CHO MỖI GPU "
                        f"(tổng batch trước grad_accum = {batch_size * n_gpu}).")
        elif n_gpu == 1 and batch_size >= 16:
            logger.warning(f"⚠️ Chỉ phát hiện 1 GPU nhưng batch_size={batch_size} (mặc định canh cho T4 x2). "
                            f"Nếu đây không phải phiên T4 x2, cân nhắc giảm --batch_size hoặc thêm "
                            f"--gradient_checkpointing để tránh OOM.")

    # ── 1. Nạp dữ liệu — CÙNG FILE với V40/V41 để so sánh công bằng ──
    df = load_data(input_dir, group_col=group_col)
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])
    raw_texts = df['comment'].astype(str).tolist()
    logger.info(f"Đã nạp {len(df)} mẫu. Nhãn: {list(le.classes_)}")

    if group_col is not None and group_col not in df.columns:
        logger.warning(f"Cột group_col='{group_col}' không tồn tại trong dữ liệu — bỏ qua group-aware split.")
        group_col = None

    # ── 2. Chia Train/Cal/Test 60/20/20 — group-aware nếu có group_col
    #      (ĐỒNG BỘ với V40/V41 — Trục 1, mục 2.2, bắt buộc để 3 hệ thống
    #      so sánh trên đúng cùng một split sạch) ──────────────────────
    idx_tr, idx_cal, idx_te = split_data_phobert(df, y, group_col=group_col, seed=seed)
    logger.info(f"Train: {len(idx_tr)} | Calibration: {len(idx_cal)} | Test: {len(idx_te)}")

    raw_tr = [raw_texts[i] for i in idx_tr]
    raw_cal = [raw_texts[i] for i in idx_cal]
    raw_te = [raw_texts[i] for i in idx_te]
    y_tr, y_cal, y_te = y[idx_tr], y[idx_cal], y[idx_te]

    # ── 3. Word segmentation cho PhoBERT ─────────────────────────────
    logger.info("Tách từ (word segmentation) cho PhoBERT...")
    seg_tr = word_segment_batch(raw_tr)
    seg_cal = word_segment_batch(raw_cal)
    seg_te = word_segment_batch(raw_te)

    # ── 4. Tokenizer & Model ──────────────────────────────────────────
    logger.info(f"Nạp tokenizer/model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(le.classes_)
    )

    if freeze_layers > 0:
        logger.info(f"Đóng băng embeddings + {freeze_layers} layer encoder đầu (tiết kiệm VRAM)...")
        base = getattr(model, model.base_model_prefix)
        for p in base.embeddings.parameters():
            p.requires_grad = False
        encoder_layers = base.encoder.layer
        for i in range(min(freeze_layers, len(encoder_layers))):
            for p in encoder_layers[i].parameters():
                p.requires_grad = False

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    def _trunc_rate(segs):
        n_over = sum(1 for t in segs if len(tokenizer.encode(t, add_special_tokens=True)) > max_length)
        return n_over / max(len(segs), 1)
    trunc = {'train': _trunc_rate(seg_tr), 'cal': _trunc_rate(seg_cal), 'test': _trunc_rate(seg_te)}
    logger.info(f"[TRUNC] ti le review bi cat o max_length={max_length}: " + ", ".join(f"{k}={v:.4%}" for k, v in trunc.items()))
    train_ds = SentimentDataset(seg_tr, y_tr, tokenizer, max_length)
    cal_ds = SentimentDataset(seg_cal, y_cal, tokenizer, max_length)
    test_ds = SentimentDataset(seg_te, y_te, tokenizer, max_length)

    # ── 5. Training arguments — tối ưu VRAM thấp (GTX 1650 Ti 4GB) ───
    # output_dir riêng theo seed để tránh Trainer nhầm checkpoint của seed
    # trước là điểm resume cho seed sau (mỗi seed phải train from scratch).
    checkpoint_dir = os.path.join('models', VERSION, 'checkpoints', f'seed_{seed}')
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(batch_size * 2, 8),
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=fp16 and device == "cuda",
        gradient_checkpointing=gradient_checkpointing,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        seed=seed,
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=cal_ds,          # model selection dùng Calibration, KHÔNG dùng Test — cùng nguyên tắc với V41
        compute_metrics=compute_metrics,
        callbacks=([] if sweep_mode else [EarlyStoppingCallback(early_stopping_patience=2)]),
    )

    logger.info("Bắt đầu training...")
    trainer.train()
    curve = [(float(e['epoch']), float(e['eval_f1_macro'])) for e in trainer.state.log_history if 'eval_f1_macro' in e]
    best_epoch, best_cal_f1 = (max(curve, key=lambda x: x[1]) if curve else (None, None))
    logger.info(f"[CAL-CURVE] seed={seed} lr={lr}: " + ", ".join(f"ep{e:.0f}={f:.4f}" for e, f in curve))
    if sweep_mode:   # KHONG dong toi tap Test, khong xuat logits, khong luu model
        if cleanup_checkpoints:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        return {'seed': seed, 'lr': lr, 'epochs': epochs, 'cal_curve': curve, 'best_epoch': best_epoch, 'best_cal_f1': best_cal_f1,
                'training_time_sec': float(time.time() - t_start), 'trunc_rate': trunc}

    # ── 6. Đánh giá DUY NHẤT MỘT LẦN trên Test (đúng nguyên tắc V41) ──
    logger.info("Đánh giá trên tập Test (chạy đúng 1 lần)...")
    pred_output = trainer.predict(test_ds)
    logits = pred_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    pred_te = np.argmax(probs, axis=1)

    f1_test = f1_score(y_te, pred_te, average='macro')
    acc_test = accuracy_score(y_te, pred_te)
    logger.info(f"✅ F1-Macro TEST (chạy đúng 1 lần) = {f1_test:.4f}")
    logger.info(f"✅ Accuracy TEST = {acc_test:.4f}")
    logger.info("\n" + classification_report(y_te, pred_te, target_names=[str(c) for c in le.classes_]))

    # ── 7. Phân tích mở rộng — cùng bộ hàm với V41 ────────────────────
    subgroup_results = error_analysis_by_subgroup(raw_te, y_te, pred_te, le)
    prediction_distribution_check(pred_te, y_tr, le)
    conf_stats = confidence_entropy_stats(probs)
    calib_stats = compute_ece_brier(probs, y_te)
    save_diagnostic_plots(y_te, pred_te, le)

    # ── [EXPORT-LOGITS PATCH — Bước 2] Lấy logits trên tập Calibration ────
    # (tập Test đã có `logits`/`probs` ở bước 6 phía trên; ở đây chỉ cần
    # chạy thêm 1 lượt predict trên cal_ds — KHÔNG ảnh hưởng tới nguyên tắc
    # "đánh giá Test đúng 1 lần", vì đây là dự đoán trên tập Calibration,
    # phục vụ fit Temperature Scaling ở Bước 5, không phải Test) ──────────
    logger.info("Trích xuất logits trên tập Calibration (phục vụ Temperature Scaling, Bước 5)...")
    cal_pred_output = trainer.predict(cal_ds)
    cal_logits = cal_pred_output.predictions
    cal_probs = torch.softmax(torch.tensor(cal_logits), dim=1).numpy()

    export_phobert_logits(
        seed=seed,
        idx_cal=idx_cal, y_cal=y_cal, logits_cal=cal_logits, probs_cal=cal_probs,
        idx_te=idx_te, y_te=y_te, logits_te=logits, probs_te=probs,
    )

    # ── 8. Lưu model + baseline giám sát (cùng schema với V41) ────────
    # LƯU Ý: prod_dir cố định -> bị GHI ĐÈ mỗi seed, giống hệt train_v40.py/
    # train_v41.py. Vòng lặp đa seed ở __main__ giữ lại toàn bộ chỉ số qua
    # giá trị trả về của hàm này, không phụ thuộc vào file model còn lại.
    prod_dir = os.path.join('models', 'production', VERSION)
    os.makedirs(prod_dir, exist_ok=True)
    trainer.save_model(prod_dir)
    tokenizer.save_pretrained(prod_dir)
    with open(os.path.join(prod_dir, 'label_encoder_classes.json'), 'w', encoding='utf-8') as f:
        json.dump(list(le.classes_), f, ensure_ascii=False)

    # ── [EXPORT-LOGITS PATCH — Bước 2] Chỉ giữ checkpoint đầy đủ (tokenizer
    #    + model) cho seed=42 để tiết kiệm dung lượng — prod_dir vốn bị GHI
    #    ĐÈ mỗi seed (xem ghi chú ở đầu file), nên nếu không sao lưu riêng,
    #    checkpoint seed=42 sẽ bị mất khi các seed khác chạy sau nó. Các seed
    #    khác chỉ cần CSV logits đã xuất ở trên, không cần giữ model đầy đủ.
    if seed == 42:
        seed42_dir = os.path.join('models', 'production', f'{VERSION}_seed42')
        try:
            shutil.rmtree(seed42_dir, ignore_errors=True)
            shutil.copytree(prod_dir, seed42_dir)
            logger.info(f"🔒 [EXPORT-LOGITS] Đã sao lưu riêng checkpoint đầy đủ seed=42 vào "
                        f"{seed42_dir} (không bị các seed sau ghi đè).")
        except Exception as e:
            logger.warning(f"Không sao lưu được checkpoint riêng cho seed=42: {e}")

    model_size_mb = sum(
        os.path.getsize(os.path.join(prod_dir, fn))
        for fn in os.listdir(prod_dir)
        if os.path.isfile(os.path.join(prod_dir, fn))
    ) / (1024 * 1024)

    total_time = time.time() - t_start
    monitoring_baseline = {
        'version': VERSION,
        'model_name': model_name,
        'seed': seed,
        'f1_test': float(f1_test),
        'accuracy_test': float(acc_test),
        'mean_entropy': conf_stats['mean_entropy'],
        'mean_confidence': conf_stats['mean_confidence'],
        'ece': calib_stats['ece'],
        'brier': calib_stats['brier'],
        'subgroup_results': subgroup_results,
        'train_label_distribution': {str(c): float(np.mean(y_tr == i)) for i, c in enumerate(le.classes_)},
        'hyperparams': {
            'max_length': max_length, 'batch_size': batch_size,
            'grad_accum': grad_accum, 'lr': lr, 'epochs': epochs,
            'freeze_layers': freeze_layers,
        },
        'total_time_seconds': total_time,
        'model_size_mb': float(model_size_mb),
    }
    with open(os.path.join(prod_dir, 'monitoring_baseline.json'), 'w', encoding='utf-8') as f:
        json.dump(monitoring_baseline, f, ensure_ascii=False, indent=2)

    logger.info(f"📁 Đã lưu mô hình + baseline giám sát vào {prod_dir}  ({model_size_mb:.2f} MB)")
    logger.info(f"⏱️  Tổng thời gian: {total_time:.1f}s")

    # ── 9. Dọn checkpoint tạm của seed này để không cộng dồn dung lượng ─
    if cleanup_checkpoints:
        try:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            logger.info(f"🧹 Đã xoá checkpoint tạm của seed={seed} ({checkpoint_dir}) để tiết kiệm đĩa.")
        except Exception as e:
            logger.warning(f"Không xoá được thư mục checkpoint tạm: {e}")

    # Trả về dict đầy đủ — Y HỆT convention của train_v40()/train_v41() — để
    # vòng lặp đa seed ở __main__ tính mean/std cho bảng so sánh 3 hệ thống.
    return {
        'f1': float(f1_test),
        'accuracy': float(acc_test),
        'ece': calib_stats['ece'],
        'brier': calib_stats['brier'],
        'training_time_sec': float(total_time),
        'model_size_mb': float(model_size_mb),
        'mean_confidence': conf_stats['mean_confidence'],
        'mean_entropy': conf_stats['mean_entropy'],
        'subgroup_results': subgroup_results,
        'hyperparams': monitoring_baseline['hyperparams'],
        'best_epoch': best_epoch, 'best_cal_f1': best_cal_f1, 'cal_curve': curve, 'trunc_rate': trunc,
        'segmenter': SEGMENTER, 'class_weight': None, 'weight_decay': 0.01, 'warmup_ratio': 0.1, 'scheduler': 'linear (Trainer default)',
        'early_stopping_patience': 2, 'checkpoint_selection': 'best calibration F1-macro (load_best_model_at_end)',
    }


# ═══════════════════════════════════════════════════════════════════
# 5. CLI
# ═══════════════════════════════════════════════════════════════════
def build_argparser():
    p = argparse.ArgumentParser(description="Fine-tune PhoBERT baseline cho AERS-Pro (so sánh với V40/V41)")
    p.add_argument('--input_dir', type=str, default='input_data',
                   help="Thư mục chứa CÙNG file dữ liệu đã dùng cho train_v40.py/train_v41.py")
    p.add_argument('--model_name', type=str, default='vinai/phobert-base-v2',
                   help="vinai/phobert-base-v2 (mặc định, để so sánh công bằng số layer/kích thước với "
                        "báo cáo hiện tại). Với T4 x2 (16GB/GPU) vinai/phobert-large cũng khả thi nếu muốn thử.")
    p.add_argument('--max_length', type=int, default=192,
                   help="192 mặc định — ĐO THỰC TẾ trên dataset_cleaned.csv (25289 dòng): p50=17 từ, "
                        "p95=65, p97=76, p99=127 từ (word-level, TRƯỚC khi tách từ/BPE làm đổi số token). "
                        "max_length=128 sẽ cắt cụt một phần đáng kể ngay ở p99, nên nâng lên 192 để có biên an "
                        "toàn qua cả bước word-segment + BPE của PhoBERT, mà vẫn rẻ hơn nhiều so với VRAM dư "
                        "của T4 x2 16GB/GPU. Giảm nếu OOM (xem log 'VRAM' in ra khi chạy để canh).")
    p.add_argument('--batch_size', type=int, default=32,
                   help="Batch size TRÊN MỖI GPU (mặc định 32, canh cho T4 x2 16GB). Trainer tự nhân đôi cho "
                        "2 GPU qua DataParallel. Giảm nếu OOM hoặc chỉ có 1 GPU.")
    p.add_argument('--grad_accum', type=int, default=1,
                   help="Mặc định 1 — với batch_size=32 x 2 GPU đã có effective batch=64, không cần cộng dồn thêm.")
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--fp16', action='store_true', default=True,
                   help="Bật mixed precision — T4 CÓ Tensor Core nên vừa tiết kiệm VRAM vừa tăng tốc thật")
    p.add_argument('--no-fp16', dest='fp16', action='store_false')
    p.add_argument('--gradient_checkpointing', action='store_true', default=False,
                   help="Mặc định TẮT (đủ VRAM trên T4 x2 16GB, tắt để tối đa tốc độ). Bật lại nếu OOM.")
    p.add_argument('--no-gradient_checkpointing', dest='gradient_checkpointing', action='store_false')
    p.add_argument('--freeze_layers', type=int, default=0,
                   help="Số layer encoder đầu bị đóng băng. Thường không cần trên T4 x2, chỉ dùng nếu OOM.")
    p.add_argument('--keep_checkpoints', dest='cleanup_checkpoints', action='store_false', default=True,
                   help="Giữ lại checkpoint tạm của từng seed thay vì xoá sau khi xong (để debug)")
    p.add_argument('--no_resume', action='store_true',
                   help="Bỏ qua results_phobert_multiseed.json cũ, chạy lại từ seed đầu tiên")
    p.add_argument('--resume_from', type=str, default=None,
                   help="Đường dẫn tới results_phobert_multiseed.json của MỘT PHIÊN KAGGLE TRƯỚC (vd. upload "
                        "làm Kaggle Input Dataset rồi trỏ /kaggle/input/.../results_phobert_multiseed.json) — "
                        "dùng khi phiên cũ đã hết giờ và /kaggle/working bị reset, để tiếp tục các seed còn thiếu.")
    p.add_argument('--group_col', type=str, default='near_dup_group_id',
                   help="Tên cột group-aware split (Trục 1, mục 2.2) — PHẢI khớp với GROUP_COL dùng trong "
                        "train_v40.py/train_v41.py để 3 hệ thống so sánh trên cùng một split sạch. "
                        "Cột này do dedup_clustering.py sinh ra — chạy script đó trước. "
                        "Truyền --group_col \"\" để tắt (quay về stratified split cũ, KHÔNG khuyến khích).")
    p.add_argument('--result_path', type=str, default='results_phobert_multiseed.json',
                   help="Tep ket qua (doi ten de khong lan voi lan chay cu, vi du results_phobert_v2.json)")
    p.add_argument('--segmenter', choices=['underthesea', 'rdr'], default='underthesea')
    p.add_argument('--sweep', action='store_true', help="Chi tim LR/so epoch tren tap CALIBRATION (khong dung Test), roi thoat")
    p.add_argument('--sweep_lrs', type=str, default='1e-5,2e-5,3e-5,5e-5')
    p.add_argument('--sweep_seeds', type=str, default='42,7,123')
    p.add_argument('--sweep_epochs', type=int, default=5)
    p.add_argument('--quick_test', action='store_true',
                   help="Chạy kiểm tra nhanh (1 epoch, batch nhỏ, seed=42 duy nhất) đúng README mục 5, "
                        "KHÔNG chạy vòng lặp 10 seed — dùng để xác nhận không OOM/lỗi config trước khi chạy full")
    return p


def run_multiseed(args):
    from seeds_config import SEEDS

    result_path = args.result_path
    results = {}
    if not args.no_resume and os.path.exists(result_path):
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                prev = json.load(f)
            results = {int(k): v for k, v in prev.get('metrics_by_seed', {}).items()}
            if results:
                logger.info(f"↩️  Tìm thấy {result_path} (cùng phiên) — resume, đã có {len(results)}/{len(SEEDS)} "
                             f"seed hoàn thành: {sorted(results.keys())}. Dùng --no_resume để chạy lại từ đầu.")
        except Exception as e:
            logger.warning(f"Không đọc được {result_path} cũ ({e}) — chạy lại từ đầu.")
            results = {}

    # Resume từ PHIÊN KAGGLE TRƯỚC (session mới, /kaggle/working đã bị reset) —
    # chỉ dùng khi chưa có kết quả nào trong phiên hiện tại, để không vô tình
    # ghi đè tiến độ đã có của phiên đang chạy.
    if not results and not args.no_resume and args.resume_from:
        try:
            with open(args.resume_from, 'r', encoding='utf-8') as f:
                prev = json.load(f)
            results = {int(k): v for k, v in prev.get('metrics_by_seed', {}).items()}
            logger.info(f"↩️  Nạp kết quả từ phiên Kaggle trước ({args.resume_from}) — đã có "
                        f"{len(results)}/{len(SEEDS)} seed hoàn thành: {sorted(results.keys())}.")
        except Exception as e:
            logger.warning(f"Không đọc được --resume_from='{args.resume_from}' ({e}) — chạy lại từ đầu.")
            results = {}

    remaining = [s for s in SEEDS if s not in results]
    if not remaining:
        logger.info("✅ Tất cả seed đã hoàn thành từ trước — không còn gì để chạy. "
                     "Xoá results_phobert_multiseed.json hoặc dùng --no_resume nếu muốn chạy lại.")
    else:
        logger.info("#" * 60)
        logger.info(f"CHẠY ĐA SEED PHOBERT — {len(SEEDS)} seed: {SEEDS}  (còn lại {len(remaining)}: {remaining})")
        logger.info("#" * 60)

    METRIC_KEYS = ['f1', 'accuracy', 'ece', 'brier', 'training_time_sec', 'model_size_mb',
                   'mean_confidence', 'mean_entropy']
    summary = {}
    seed_durations = []
    for i, s in enumerate(remaining, 1):
        logger.info(f"\n########## [PHOBERT] SEED {i}/{len(remaining)} còn lại (seed={s}) ##########")
        if seed_durations:
            eta_sec = np.mean(seed_durations) * (len(remaining) - i + 1)
            logger.info(f"⏳ ETA còn lại ước tính: {eta_sec / 3600:.1f} giờ "
                        f"(dựa trên trung bình {np.mean(seed_durations):.0f}s/seed đã chạy)")

        metrics = train_phobert(
            input_dir=args.input_dir, model_name=args.model_name, max_length=args.max_length,
            batch_size=args.batch_size, grad_accum=args.grad_accum, epochs=args.epochs, lr=args.lr,
            seed=s, fp16=args.fp16, gradient_checkpointing=args.gradient_checkpointing,
            freeze_layers=args.freeze_layers, cleanup_checkpoints=args.cleanup_checkpoints,
            group_col=(args.group_col or None),
        )
        results[s] = metrics
        seed_durations.append(metrics['training_time_sec'])

        # Lưu ngay sau MỖI seed — chống mất kết quả nếu máy bị tắt/ngủ giữa chừng
        # (quan trọng hơn cả với V40/V41 vì đây là laptop, không phải máy thuê
        # luôn bật — có thể phải chia làm nhiều phiên chạy trong nhiều ngày).
        summary = {}
        for key in METRIC_KEYS:
            vals = [results[s2][key] for s2 in results if key in results[s2]]
            if vals:
                summary[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({
                'version': 'phobert',
                'model_name': args.model_name,
                'seeds': SEEDS,
                'group_col': (args.group_col or None),
                'metrics_by_seed': {str(k): v for k, v in results.items()},
                'summary': summary,
                'completed_seeds': len(results),
                'total_seeds': len(SEEDS),
            }, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f"✅ PHOBERT — qua {len(results)}/{len(SEEDS)} seed:")
    for key in METRIC_KEYS:
        if key in summary:
            logger.info(f"   {key}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")
    logger.info(f"   Chi tiết từng seed: {sorted(results.keys())}")
    logger.info(f"📁 Đã lưu {result_path}")
    logger.info("=" * 60)


def run_lr_sweep(args):
    """Luoi LR x seed, huan luyen args.sweep_epochs epoch, chon theo F1-macro tren CALIBRATION (khong cham Test).
    Ghi lr_sweep_results.json (resume theo tung o) va in bang tong hop."""
    lrs = [float(x) for x in args.sweep_lrs.split(',')]; seeds = [int(x) for x in args.sweep_seeds.split(',')]
    path = 'lr_sweep_results.json'
    done = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}
    for lr in lrs:
        for s in seeds:
            key = f"{lr:g}_{s}"
            if key in done:
                continue
            done[key] = train_phobert(input_dir=args.input_dir, model_name=args.model_name, max_length=args.max_length,
                                      batch_size=args.batch_size, grad_accum=args.grad_accum, epochs=args.sweep_epochs, lr=lr, seed=s,
                                      fp16=args.fp16, gradient_checkpointing=args.gradient_checkpointing, freeze_layers=args.freeze_layers,
                                      cleanup_checkpoints=True, group_col=(args.group_col or None), sweep_mode=True)
            json.dump(done, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    logger.info("=" * 60 + "\nLR SWEEP (F1-macro tren CALIBRATION; cot = epoch)")
    best = None
    for lr in lrs:
        curves = [dict(done[f"{lr:g}_{s}"]['cal_curve']) for s in seeds if f"{lr:g}_{s}" in done]
        eps = sorted({e for c in curves for e in c})
        mean_by_ep = {e: float(np.mean([c[e] for c in curves if e in c])) for e in eps}
        be = max(mean_by_ep, key=mean_by_ep.get)
        logger.info(f"lr={lr:g}: " + ", ".join(f"ep{e:.0f}={v:.4f}" for e, v in mean_by_ep.items()) + f"  -> best ep{be:.0f}={mean_by_ep[be]:.4f}")
        if best is None or mean_by_ep[be] > best[2]:
            best = (lr, be, mean_by_ep[be])
    logger.info(f"CHON: lr={best[0]:g}, epochs={best[1]:.0f} (mean cal F1={best[2]:.4f}). Chay tiep: --lr {best[0]:g} --epochs {best[1]:.0f}")


def zip_logits(out='phobert_logits'):
    if os.path.isdir('reports/phobert_logits'):
        shutil.make_archive(out, 'zip', 'reports/phobert_logits')
        logger.info(f"📦 Da nen logits vao {out}.zip - TAI FILE NAY VE truoc khi het phien Kaggle")


def run_quick_test(args):
    """Đúng README mục 5: 1 epoch, batch nhỏ, seed=42 duy nhất, chỉ để xác
    nhận pipeline chạy hết mà không OOM/lỗi config trước khi chạy full 10 seed."""
    logger.info("🔎 QUICK TEST — 1 epoch, batch_size=2, grad_accum=4, seed=42 (không lưu vào "
                "results_phobert_multiseed.json). Theo dõi log để chắc chắn không OOM trước khi bỏ --quick_test.")
    train_phobert(
        input_dir=args.input_dir, model_name=args.model_name, max_length=args.max_length,
        batch_size=2, grad_accum=4, epochs=1, lr=args.lr,
        seed=42, fp16=args.fp16, gradient_checkpointing=args.gradient_checkpointing,
        freeze_layers=args.freeze_layers, cleanup_checkpoints=args.cleanup_checkpoints,
        group_col=(args.group_col or None),
    )


# ═══════════════════════════════════════════════════════════════════
# 6. CHẠY CHÍNH — ĐA SEED (10 seed, dùng CHUNG với V40/V41 — seeds_config.py)
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    args = build_argparser().parse_args()
    SEGMENTER = args.segmenter
    if args.quick_test:
        run_quick_test(args)
    elif args.sweep:
        run_lr_sweep(args)
    else:
        run_multiseed(args)
        zip_logits()
