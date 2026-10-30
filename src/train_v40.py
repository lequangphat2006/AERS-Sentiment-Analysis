#!/usr/bin/env python
# ═══════════════════════════════════════════════════════════════════════════
# V40: AERS‑Pro Cải tiến  —  Khắc phục nhược điểm V38/V39
# ───────────────────────────────────────────────────────────────────────
# • Fix lỗi giá trị âm với ComplementNB → dùng MinMaxScaler, bỏ CNB.
# • Stacking chuẩn: OOF riêng cho từng learner → meta‑features đầy đủ.
# • Sử dụng tập Calibration để chọn siêu tham số meta‑learner.
# • Tối ưu siêu tham số feature & base learner bằng Optuna.
# • Mở rộng lexicon, cải thiện trích xuất đặc trưng.
# • Đánh giá ổn định với nhiều seed (tuỳ chọn).
# ───────────────────────────────────────────────────────────────────────
# CHANGELOG v40.1 (bản vá tối ưu hoá lexicon/negation + kiến trúc stacking):
# [BUGFIX-1] extract_features() chạy trên RAW text bằng text_low.split():
#            dấu câu dính liền từ ("tốt," "hỏng!") khiến so khớp lexicon/
#            CLAUSE_BOUNDARY/CONTRAST_MARKERS thất bại phần lớn trên dữ
#            liệu thật → thêm bước đệm khoảng trắng quanh dấu câu trước
#            khi split().
# [BUGFIX-2] Các khoá nhiều từ trong POSITIVE_WORDS/NEGATIVE_WORDS/
#            CONTRAST_MARKERS ("hài lòng", "tuy nhiên"...) không bao giờ
#            khớp với token đơn từ split() → gộp thành token liền bằng
#            dấu "_" trước khi so khớp (áp dụng nhất quán cho cả 3 tập).
# [FIX-3]    Phủ định kép ("không phải là không tốt") trước đây chỉ bị
#            decay dần chứ không đảo dấu → dùng bộ đếm neg_count, đảo dấu
#            theo tính chẵn/lẻ.
# [FIX-4]    "không" đứng cuối mệnh đề / trước "?" (dạng hỏi‑đuôi "...
#            không?") bị hiểu nhầm là phủ định → thêm lookahead bỏ qua.
# [FEAT-5]   Luật "vế sau quan trọng hơn" quanh từ nối tương phản (nhưng/
#            mà/tuy nhiên...): vế trước bị giảm trọng số còn 0.5, vế sau
#            giữ nguyên trọng số — cài trực tiếp trong 1 vòng lặp duy nhất
#            (buffer‑and‑flush theo mệnh đề) thay vì tách riêng.
# [FEAT-6]   Thêm đặc trưng: sentiment_ratio, is_neutral (ngưỡng T được dò
#            trên tập Calibration bằng mutual information), vị trí tương
#            đối của từ cảm xúc cuối cùng, cờ is_question.
# [ARCH-7]   Đặc trưng thủ công (manual lexicon features) giờ được nối
#            TRỰC TIẾP vào meta‑train/cal/test (không chỉ đi qua base
#            learners) — meta‑learner tuyến tính tận dụng tín hiệu lexicon
#            rõ ràng hơn, không bị pha loãng trong không gian TF‑IDF.
# [ARCH-8]   Thêm bước hiệu chỉnh xác suất (calibration head kiểu Platt)
#            fit trên tập Calibration, chỉ áp dụng nếu cải thiện ECE mà
#            không làm giảm F1‑macro quá 0.5 điểm % trên Cal.
# ───────────────────────────────────────────────────────────────────────
# CHANGELOG v40.2 (audit leak tiền xử lý + nhất quán group-aware + tinh
# chỉnh lexicon câu hỏi/phủ định + inference an toàn):
# [LEAK-9]    optuna_objective() và OOF stacking (bước 8) trước đây fit
#             FeaturePipeline (TF‑IDF vocab/idf, MinMaxScaler manual+POS)
#             MỘT LẦN trên toàn bộ train RỒI MỚI chia fold — thống kê tiền
#             xử lý đã "nhìn thấy" phần dữ liệu sẽ là validation fold. Fit
#             một FeaturePipeline MỚI cho từng fold, chỉ trên phần train của
#             fold đó. KHÔNG ảnh hưởng trực tiếp F1/accuracy báo cáo trên
#             Test (pipe dùng cho Test luôn fit lại riêng, chỉ trên Train ở
#             bước 6) — nhưng làm tín hiệu OOF nuôi meta‑learner và điểm nội
#             bộ Optuna đáng tin hơn. Lưu ý: HashingVectorizer KHÔNG leak (là
#             hash cố định, không có state để fit) — chỉ TfidfVectorizer và
#             2 MinMaxScaler mới thực sự leak.
# [GROUPCV-10] CalibratedClassifierCV(cv=3) cho SVM dùng cv mặc định (không
#             group-aware) cho bước hiệu chỉnh Platt — không nhất quán với
#             StratifiedGroupKFold dùng ở Optuna/OOF. BaseLearnersPool giờ tự
#             dựng fold group-aware (StratifiedGroupKFold) rồi truyền vào
#             CalibratedClassifierCV(cv=<list fold>) khi có groups.
# [FIX-11]    Mở rộng FIX-4: bỏ qua các tiểu từ cuối câu (SENTENCE_FINAL_
#             PARTICLES: "vậy", "ạ", "nhỉ"...) trước khi kiểm tra ranh giới
#             mệnh đề, để bắt được "...không vậy?", "...không ạ?" (trước đây
#             chỉ bắt khi "không" đứng NGAY trước dấu câu). Thêm đặc trưng
#             is_tag_question + 2 cột tương tác pos_s_x_tagq/neg_s_x_tagq để
#             meta‑learner tuyến tính có thể học GIẢM trọng số pos_s/neg_s
#             khi câu chỉ đang HỎI chứ không khẳng định — không đụng vào
#             is_question chung (câu hỏi tu từ vẫn mang sentiment thật).
# [LEXFIX-13] "đâu" đứng sau giới từ/động từ chỉ hướng (ở, đi, đến...) hoặc
#             trước từ hỏi tình trạng (rồi, nào, nữa) → coi là từ hỏi địa
#             điểm, không tính vào negation ("ở đâu", "ship đâu rồi" không
#             còn bị hiểu nhầm thành phủ định).
# [SCALE-14]  Clip đầu ra MinMaxScaler.transform() (manual + POS) về [0,1] —
#             transform() (khác fit_transform()) có thể ngoại suy ra ngoài
#             [0,1] nếu Cal/Test có outlier vượt range đã thấy ở Train.
# [SPLIT-16]  split_data_v40(): GroupShuffleSplit không hỗ trợ stratify theo
#             nhãn → đổi sang lấy 1 fold từ StratifiedGroupKFold (group-aware
#             VÀ xấp xỉ cân bằng lớp), giữ GroupShuffleSplit làm fallback bậc
#             2. CẦN đồng bộ sang split_data_v41() để so sánh còn sòng phẳng.
# [INFER-15]  Thêm load_v40_bundle()/predict_v40() — nạp đủ artifact (gồm cả
#             neutral_threshold.pkl) và tái tạo đúng pipeline lúc train, để
#             không còn phụ thuộc việc "nhớ" ghi đè is_neutral ở inference.
# [ARCH-19]   BaseLearnersPool/train_v40() thêm include_svm/include_lgbm/
#             use_cluster_features (mặc định True — không đổi hành vi) để có
#             thể thử "V40 không SVM"/"không cluster one-hot" mà không sửa
#             code, nếu ablation của dự án áp dụng lại cho cả baseline.
# [CLEAN-18]  Xoá key 'đc' bị khai báo lặp trong TEENCODE_MAP (vô hại, cùng
#             giá trị — chỉ dọn code). Thêm assert mềm sau LabelEncoder nếu
#             load_data() gộp nhiều file mà nhãn lệch chính tả/viết hoa.
# ═══════════════════════════════════════════════════════════════════════════

import os, sys, re, glob, json, time, random, warnings, logging, logging.handlers, unicodedata
import multiprocessing
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import hstack as sp_hstack, csr_matrix, issparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      GroupShuffleSplit, StratifiedGroupKFold)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.cluster import KMeans
# [CLEAN-18] bỏ `from sklearn.base import clone` — không còn dùng sau khi
# BaseLearnersPool chuyển sang build_fresh_learner() (xem GROUPCV-10).
import joblib
from joblib import Parallel, delayed
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Optional heavy deps
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from underthesea import word_tokenize, pos_tag
    HAS_UNDER = True
    HAS_POS   = True
except ImportError:
    try:
        from underthesea import word_tokenize
        HAS_UNDER = True
        HAS_POS   = False
    except:
        HAS_UNDER = False
        HAS_POS = False

if HAS_UNDER:
    try:
        word_tokenize("khởi tạo")
        if HAS_POS: pos_tag("khởi tạo")
    except Exception:
        pass

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════
# CONFIG & LOGGING
# ═══════════════════════════════════════════════════════════════════
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
_log_file = os.path.join(LOG_DIR, f"train_v40_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger('SentimentV40')
logger.setLevel(logging.DEBUG)
_fh = logging.handlers.RotatingFileHandler(_log_file, maxBytes=15_000_000, backupCount=5, encoding='utf-8')
_ch = logging.StreamHandler(sys.stdout)
_fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
for h in [_fh, _ch]:
    h.setFormatter(_fmt)
    logger.addHandler(h)

N_CORES = multiprocessing.cpu_count()
N_WORKERS = max(1, N_CORES - 1)
N_PARALLEL_PRE = N_WORKERS       # cho tiền xử lý
N_PARALLEL_TRIALS = max(1, min(N_CORES // 4, 2))

VERSION = 'v40'
N_OOF_FOLDS = 5
JOBLIB_BACKEND = 'loky'

for d in ['models', 'models/production', 'models/archive', 'reports']:
    os.makedirs(d, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# TOP‑LEVEL FUNCTIONS (bắt buộc cho pickle trên Windows)
# ═══════════════════════════════════════════════════════════════════
def _global_extract_features_single(text: str) -> list:
    return TextPreprocessor.extract_features(text)

def _global_preprocess_single(text: str) -> str:
    return TextPreprocessor.preprocess(text)

def _global_extract_meta_single(text: str) -> list:
    return MetaFeatureExtractor.extract(text)

def _global_pos_single(text: str):
    if not HAS_POS:
        return {'ratio_adj': 0.0, 'ratio_adv': 0.0, 'ratio_verb': 0.0, 'ratio_noun': 0.0}, []
    try:
        tags = pos_tag(text)
        total = max(len(tags), 1)
        c = Counter([t[1] for t in tags])
        return {'ratio_adj': c.get('A',0)/total, 'ratio_adv': c.get('R',0)/total, 'ratio_verb': c.get('V',0)/total, 'ratio_noun': c.get('N',0)/total}, tags
    except:
        return {'ratio_adj': 0.0, 'ratio_adv': 0.0, 'ratio_verb': 0.0, 'ratio_noun': 0.0}, []

def _slice_pos_cache(pos_cache, idx):
    """[LEAK-9] Cắt pos_cache (tuple (list_dict, list_tags) trả về bởi
    extract_pos_features_batch) theo idx, để dùng riêng cho một fold con khi
    fit FeaturePipeline fold-local. Chỉ vị trí [0] thực sự được
    FeaturePipeline đọc (ratio_adj/ratio_verb) nên vị trí [1] để rỗng, tránh
    tính lại POS tags không cần thiết."""
    if pos_cache is None:
        return None
    dicts = pos_cache[0]
    return ([dicts[i] for i in idx], [])

# ═══════════════════════════════════════════════════════════════════
# 0. LEXICON MỞ RỘNG & NORMALIZER
# ═══════════════════════════════════════════════════════════════════
class TextNormalizerManager:
    def __init__(self, teencode_map: dict):
        sorted_entries = sorted(teencode_map.items(), key=lambda x: -len(x[0]))
        self._replacement_map = {k.lower(): v for k, v in sorted_entries}
        pattern_parts = [re.escape(k) for k, _ in sorted_entries]
        self._pattern = re.compile(r'\b(' + '|'.join(pattern_parts) + r')\b', flags=re.IGNORECASE)

    def normalize_text(self, text: str) -> str:
        return self._pattern.sub(lambda m: self._replacement_map.get(m.group(0).lower(), m.group(0)), text)

_text_normalizer = None
_text_normalizer_lock = __import__('threading').Lock()

def get_text_normalizer() -> TextNormalizerManager:
    global _text_normalizer
    if _text_normalizer is None:
        with _text_normalizer_lock:
            if _text_normalizer is None:
                _text_normalizer = TextNormalizerManager(TextPreprocessor.TEENCODE_MAP)
    return _text_normalizer

# ═══════════════════════════════════════════════════════════════════
# 1. TEXT PREPROCESSOR NÂNG CAO
# ═══════════════════════════════════════════════════════════════════
class TextPreprocessor:
    # Emoji map mở rộng
    EMOJI_MAP = {
        '😊': ' vui ', '😄': ' vui ', '😍': ' yeu ', '👍': ' tot ',
        '😞': ' buon ', '😡': ' gian ', '👎': ' te ', '💯': ' hoan_hao ',
        '✅': ' ok ', '❤️': ' yeu ', '😢': ' buon ', '😭': ' rat_buon ',
        '🥰': ' yeu ', '🤬': ' cuc_gian '
    }
    # Teencode mở rộng
    TEENCODE_MAP = {
        'ko': 'không', 'khong': 'không', 'k': 'không', 'kg': 'không',
        'dc': 'được', 'đc': 'được', 'j': 'gì', 'vs': 'với', 'ok': 'được',
        'vcl': 'quá', 'vl': 'quá', 'sp': 'sản phẩm', 'ship': 'giao hàng',
        'r': 'rồi', 'đỉnh': 'rất tốt', 'xịn': 'chất lượng', 'ib': 'nhắn tin',
        # [CLEAN-18] xoá key 'đc' bị khai báo lặp lại (v40.1 có 2 dòng cùng
        # giá trị 'được' — vô hại vì trùng giá trị, nhưng dọn cho rõ ràng).
        'hok': 'không', 'hem': 'không', 'bt': 'bình thường',
        'tệ': 'kém', 'tồi': 'kém', 'chán': 'không hài lòng'
    }
    # Từ điển tích cực/tiêu cực mở rộng
    POSITIVE_WORDS = {
        'tốt': 2, 'tuyệt': 3, 'hay': 2, 'đẹp': 2, 'thích': 2,
        'hài lòng': 2, 'ngon': 2, 'tuyệt vời': 3, 'chất lượng': 2,
        'nhanh': 2, 'đỉnh': 3, 'xịn': 2, 'ok': 1, 'bền': 2,
        'xuất sắc': 3, 'hoàn hảo': 3, 'rẻ': 1, 'đáng mua': 2,
        'tuyệt hảo': 3, 'yêu': 2, 'thân thiện': 2, 'nhiệt tình': 2
    }
    NEGATIVE_WORDS = {
        'tệ': 3, 'xấu': 2, 'tồi': 3, 'chậm': 2, 'hỏng': 3,
        'thất vọng': 3, 'buồn': 2, 'chán': 2, 'kém': 2,
        'fake': 3, 'rởm': 3, 'lừa': 3, 'rác': 3, 'đắt': 2,
        'lỗi': 2, 'treo': 3, 'tệ hại': 3, 'bực mình': 3,
        'khó chịu': 3, 'không hài lòng': 2
    }
    NEGATION_WORDS = {'không', 'chẳng', 'chả', 'đâu', 'chưa', 'k', 'ko', 'chớ', 'hok', 'hem'}
    INTENSIFIERS = {'rất': 1.5, 'quá': 1.5, 'cực': 2.0, 'siêu': 2.0, 'hơi': 0.7, 'khá': 1.2, 'cực kỳ': 2.0}
    CLAUSE_BOUNDARY = {'.', ',', ';', '!', '?', '...'}
    CONTRAST_MARKERS = {'nhưng', 'mà', 'tuy nhiên', 'ai ngờ', 'vậy mà', 'thế mà'}

    # [FIX-11] Tiểu từ cuối câu thường xuất hiện GIỮA "không"/phủ định và dấu
    # câu trong câu hỏi đuôi kiểu "...không vậy?", "...không ạ?", "...không
    # nhỉ?" — FIX-4 gốc (v40.1) chỉ bắt được khi "không" đứng NGAY trước dấu
    # câu, bỏ sót mọi trường hợp có tiểu từ chen giữa (đúng như review đã chỉ
    # ra). Lưu ý: KHÔNG đưa 'đâu' vào đây — "không vậy đâu" là một cách phủ
    # định nhấn mạnh thật sự ("không phải vậy đâu"), phải được phân biệt với
    # câu hỏi đuôi, xem _is_negation_skippable().
    SENTENCE_FINAL_PARTICLES = {'vậy', 'ạ', 'nhỉ', 'nhé', 'à', 'hả', 'thế', 'đấy', 'chứ', 'ha', 'hử', 'cơ'}

    # [LEXFIX-13] "đâu" nhập nhằng giữa phủ định ("chẳng đâu", "có...đâu") và
    # từ hỏi địa điểm/tình trạng ("ở đâu", "đâu rồi", "ship đâu vậy" — rất
    # phổ biến trong review than phiền về giao hàng). Heuristic: nếu "đâu"
    # đứng ngay sau một giới từ/động từ chỉ hướng, hoặc ngay trước một từ hỏi
    # tình trạng thường đi kèm, coi là từ hỏi địa điểm — KHÔNG phải phủ định.
    LOCATIVE_DAU_BEFORE = {'ở', 'tại', 'đi', 'đến', 'tới', 'về', 'từ', 'sang', 'qua'}
    LOCATIVE_DAU_AFTER = {'rồi', 'nào', 'nữa'}

    # [BUGFIX-2] Khoá nhiều từ ("hài lòng", "tuy nhiên"...) không bao giờ
    # khớp với token đơn từ do split() trả về từng từ đơn. Ta gộp mọi khoá
    # nhiều từ (lexicon + contrast markers) thành token liền bằng "_" và áp
    # dụng gộp này lên văn bản TRƯỚC khi tách token, để cả vòng lặp so khớp
    # đơn giản (tok in dict) hoạt động đúng cho cả từ đơn lẫn từ ghép.
    _MULTIWORD_SOURCE = {}

    @classmethod
    def _build_multiword_assets(cls):
        """Dựng lại (1) map từ_ghép→trọng số/nhãn dùng dấu '_' và (2) regex
        gộp. Gọi 1 lần và cache — vì các dict lexicon là hằng số class."""
        merged_pos = {k.replace(' ', '_'): v for k, v in cls.POSITIVE_WORDS.items()}
        merged_neg = {k.replace(' ', '_'): v for k, v in cls.NEGATIVE_WORDS.items()}
        merged_contrast = {m.replace(' ', '_') for m in cls.CONTRAST_MARKERS}

        multiword_keys = sorted(
            {k for k in cls.POSITIVE_WORDS if ' ' in k}
            | {k for k in cls.NEGATIVE_WORDS if ' ' in k}
            | {m for m in cls.CONTRAST_MARKERS if ' ' in m},
            key=len, reverse=True
        )
        pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in multiword_keys) + r')\b') \
            if multiword_keys else None

        cls._MERGED_POSITIVE = merged_pos
        cls._MERGED_NEGATIVE = merged_neg
        cls._MERGED_CONTRAST = merged_contrast
        cls._MERGE_PATTERN = pattern

    @classmethod
    def _tokenize_for_lexicon(cls, text_low: str) -> list:
        """[BUGFIX-1] + [BUGFIX-2]: đệm khoảng trắng quanh dấu câu (để dấu
        câu dính liền từ như 'tốt,' 'hỏng!' không làm hỏng so khớp từ/ranh
        giới mệnh đề), rồi gộp cụm từ nhiều-từ thành token liền bằng '_'."""
        if not hasattr(cls, '_MERGE_PATTERN'):
            cls._build_multiword_assets()
        # Tách "..." thành 1 token trước, rồi mới tách các dấu câu đơn khác
        padded = re.sub(r'(\.\.\.)', r' \1 ', text_low)
        padded = re.sub(r'([.,;!?])', r' \1 ', padded)
        if cls._MERGE_PATTERN is not None:
            padded = cls._MERGE_PATTERN.sub(lambda m: m.group(0).replace(' ', '_'), padded)
        return padded.split()

    @classmethod
    def extract_features(cls, text: str) -> list:
        text_low = text.lower()
        tokens = cls._tokenize_for_lexicon(text_low)
        n_tokens = max(len(tokens), 1)

        pos_s, neg_s = 0.0, 0.0
        matched_count = 0
        last_match_idx = -1

        # Buffer theo mệnh đề để cài luật "vế sau quan trọng hơn" [FEAT-5]:
        # tích luỹ điểm của vế TRƯỚC contrast marker vào buffer, khi gặp
        # marker thì flush buffer với trọng số giảm nửa rồi chuyển sang chế
        # độ cộng trực tiếp (trọng số đầy đủ) cho phần còn lại của mệnh đề.
        BEFORE_CONTRAST_WEIGHT = 0.5
        clause_buffer = []       # list[(is_pos: bool, amount: float)]
        direct_mode = False      # True = đã qua contrast marker trong mệnh đề này

        neg_count = 0             # [FIX-3] đếm số lần phủ định còn hiệu lực
        dist_since_neg = 0        # khoảng cách (số token) kể từ lần phủ định gần nhất
        detected_tag_question = False  # [FIX-11] có bắt được mẫu "...không + tiểu từ?" không

        def flush_buffer(weight):
            nonlocal pos_s, neg_s
            for is_pos, amount in clause_buffer:
                if is_pos:
                    pos_s += amount * weight
                else:
                    neg_s += amount * weight
            clause_buffer.clear()

        def reset_clause():
            nonlocal neg_count, dist_since_neg, direct_mode
            flush_buffer(1.0)
            neg_count = 0
            dist_since_neg = 0
            direct_mode = False

        for i, tok in enumerate(tokens):
            if tok in cls.CLAUSE_BOUNDARY:
                reset_clause()
                continue
            if tok in cls._MERGED_CONTRAST or tok in cls.CONTRAST_MARKERS:
                # [FEAT-5] Flush vế trước với trọng số giảm, rồi chuyển
                # sang cộng trực tiếp (trọng số đầy đủ) cho vế sau.
                flush_buffer(BEFORE_CONTRAST_WEIGHT)
                direct_mode = True
                neg_count = 0
                dist_since_neg = 0
                continue
            if tok in cls.NEGATION_WORDS:
                # [LEXFIX-13] "đâu" đứng sau giới từ/động từ chỉ hướng (ở,
                # đi, đến...) hoặc đứng trước từ hỏi tình trạng (rồi, nào,
                # nữa) → từ hỏi địa điểm/tình trạng ("ở đâu", "ship đâu rồi"),
                # KHÔNG phải phủ định — bỏ qua, không tính là negation event.
                if tok == 'đâu':
                    prev_tok = tokens[i - 1] if i > 0 else None
                    next_tok = tokens[i + 1] if i + 1 < len(tokens) else None
                    if (prev_tok in cls.LOCATIVE_DAU_BEFORE) or (next_tok in cls.LOCATIVE_DAU_AFTER):
                        continue

                # [FIX-4 mở rộng — FIX-11] "...không?" / "được không" ở cuối
                # mệnh đề thường là câu hỏi lịch sự, KHÔNG phải phủ định thật.
                # Bản v40.1 chỉ bắt được khi "không" đứng NGAY trước dấu câu;
                # ở đây bỏ qua thêm các tiểu từ cuối câu chen giữa (SENTENCE_
                # FINAL_PARTICLES: "vậy", "ạ", "nhỉ"...) trước khi kiểm tra
                # ranh giới mệnh đề, để bắt được "...không vậy?", "...không
                # ạ?". Cố ý KHÔNG bắt buộc câu phải kết thúc bằng "?": review
                # tiếng Việt kiểu chat thường bỏ dấu hỏi ("ổn không ạ." vẫn
                # là câu hỏi đuôi trên thực tế).
                j = i + 1
                while j < len(tokens) and tokens[j] in cls.SENTENCE_FINAL_PARTICLES:
                    j += 1
                is_last_in_clause = (j >= len(tokens)) or (tokens[j] in cls.CLAUSE_BOUNDARY)
                if is_last_in_clause:
                    if j > i + 1:
                        # Có tiểu từ cuối câu chen giữa → tín hiệu khá chắc
                        # đây là câu hỏi đuôi thật sự (khác với "không" đứng
                        # trơ trọi ngay trước dấu câu, vốn cũng vô hại vì
                        # không có gì để đảo dấu ngay sau đó).
                        detected_tag_question = True
                    continue
                neg_count += 1
                dist_since_neg = 0
                continue

            weight = None
            is_pos = None
            if tok in cls._MERGED_POSITIVE:
                is_pos, base = True, cls._MERGED_POSITIVE[tok]
            elif tok in cls._MERGED_NEGATIVE:
                is_pos, base = False, cls._MERGED_NEGATIVE[tok]
            else:
                base = None

            if base is not None:
                matched_count += 1
                last_match_idx = i
                if neg_count == 0:
                    factor = 1.0
                else:
                    decay = max(1.0 - dist_since_neg * 0.15, 0.3)
                    # lẻ lần phủ định → đảo dấu; chẵn lần → phục hồi nhưng
                    # yếu hơn bản gốc (phủ định kép thường là khẳng định dè
                    # dặt, không mạnh bằng câu khẳng định trực tiếp)
                    factor = -decay if (neg_count % 2 == 1) else 0.7 * decay
                amount = base * factor
                if direct_mode:
                    if is_pos:
                        pos_s += amount
                    else:
                        neg_s += amount
                else:
                    clause_buffer.append((is_pos, amount))

            if neg_count > 0:
                dist_since_neg += 1

        # Flush phần còn lại (câu không kết thúc bằng dấu câu chuẩn)
        flush_buffer(1.0)

        intensity = sum(cls.INTENSIFIERS.get(w, 1.0) for w in cls.INTENSIFIERS if w in text_low)
        sentiment_ratio = matched_count / n_tokens
        last_match_pos_ratio = (last_match_idx / n_tokens) if last_match_idx >= 0 else -1.0
        is_question = 1.0 if text.strip().endswith('?') else 0.0
        pos_minus_neg = pos_s - neg_s
        # Ngưỡng neutral mặc định — có thể bị GHI ĐÈ sau khi dò tối ưu T
        # trên tập Calibration bằng mutual information (xem tune_neutral_threshold
        # trong train_v40()); giữ hằng số này làm fallback an toàn.
        is_neutral = 1.0 if abs(pos_minus_neg) < cls.NEUTRAL_THRESHOLD_DEFAULT else 0.0

        # [FIX-11] is_question (cờ chung, chỉ dựa vào dấu "?" cuối câu) KHÔNG
        # đủ để "ngắt" pos_s/neg_s: pos_s/neg_s vẫn cộng y hệt câu khẳng định
        # dù "Có tốt không?" chỉ đang HỎI về sentiment chứ không KHẲNG ĐỊNH
        # nó. Nhưng cũng không thể đơn giản giảm điểm cho MỌI câu kết thúc
        # bằng "?" — câu hỏi tu từ ("Sao mà tệ vậy?") vẫn mang sentiment thật.
        # Do đó dùng is_tag_question (hẹp hơn is_question — chỉ bật khi bắt
        # được đúng mẫu "không/chẳng... + tiểu từ?" ở trên) làm tín hiệu, và
        # thêm 2 đặc trưng TƯƠNG TÁC (pos_s_x_tagq, neg_s_x_tagq) thay vì tự
        # sửa pos_s/neg_s gốc — vì meta‑learner tuyến tính không tự học được
        # tích chéo is_question × pos_s từ 2 cột cộng riêng lẻ, phải cấp sẵn
        # tích số đó làm cột thứ 3.
        is_tag_question = 1.0 if detected_tag_question else 0.0
        pos_s_x_tagq = pos_s * is_tag_question
        neg_s_x_tagq = neg_s * is_tag_question

        return [
            pos_s, neg_s, intensity,
            text.count('!'), text.count('?'),
            float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
            pos_minus_neg,
            is_neutral,
            sentiment_ratio,
            last_match_pos_ratio,
            is_question,
            is_tag_question,
            pos_s_x_tagq,
            neg_s_x_tagq,
        ]

    # Thứ tự phải khớp CHÍNH XÁC với list trả về ở extract_features().
    FEATURE_NAMES = [
        'pos_s', 'neg_s', 'intensity', 'excl_count', 'quest_count',
        'avg_token_len', 'pos_minus_neg', 'is_neutral', 'sentiment_ratio',
        'last_match_pos_ratio', 'is_question',
        # [FIX-11] — mới trong v40.2, xem changelog
        'is_tag_question', 'pos_s_x_tagq', 'neg_s_x_tagq',
    ]
    IDX_POS_MINUS_NEG = FEATURE_NAMES.index('pos_minus_neg')
    IDX_IS_NEUTRAL = FEATURE_NAMES.index('is_neutral')
    NEUTRAL_THRESHOLD_DEFAULT = 2.0

    @classmethod
    def preprocess(cls, text: str) -> str:
        text = unicodedata.normalize('NFC', str(text))
        for emoji, word in cls.EMOJI_MAP.items():
            text = text.replace(emoji, word)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = get_text_normalizer().normalize_text(text).lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'([!?])\1+', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text) if HAS_UNDER else text.split()
        return ' '.join([w for w in tokens if w not in cls.STOPWORDS and len(w) > 1])

    @classmethod
    def preprocess_batch(cls, raw_texts: list) -> list:
        try:
            with joblib.parallel_config(backend=JOBLIB_BACKEND, n_jobs=N_PARALLEL_PRE):
                return Parallel(n_jobs=N_PARALLEL_PRE)(delayed(_global_preprocess_single)(t) for t in raw_texts)
        except Exception:
            return [cls.preprocess(t) for t in raw_texts]

    @classmethod
    def extract_features_batch(cls, raw_texts: list) -> np.ndarray:
        try:
            with joblib.parallel_config(backend=JOBLIB_BACKEND, n_jobs=N_PARALLEL_PRE):
                feats = Parallel(n_jobs=N_PARALLEL_PRE)(delayed(_global_extract_features_single)(t) for t in raw_texts)
        except Exception:
            feats = [cls.extract_features(t) for t in raw_texts]
        return np.array(feats, dtype=np.float32)

    @classmethod
    def extract_pos_features_batch(cls, raw_texts: list):
        try:
            with joblib.parallel_config(backend=JOBLIB_BACKEND, n_jobs=N_PARALLEL_PRE):
                res = Parallel(n_jobs=N_PARALLEL_PRE)(delayed(_global_pos_single)(t) for t in raw_texts)
        except Exception:
            res = [_global_pos_single(t) for t in raw_texts]
        return [r[0] for r in res], [r[1] for r in res]

TextPreprocessor.STOPWORDS = {'và', 'của', 'một', 'những', 'các', 'đã', 'có', 'cho', 'về', 'từ', 'nếu', 'khi', 'thì', 'rằng', 'lại', 'ra', 'này', 'kia', 'đó', 'hay', 'hoặc', 'do', 'vì'}

# ═══════════════════════════════════════════════════════════════════
# 2. META‑FEATURE EXTRACTOR (giữ nguyên)
# ═══════════════════════════════════════════════════════════════════
class MetaFeatureExtractor:
    @classmethod
    def extract(cls, text: str) -> list:
        t = str(text).lower().split()
        n = max(len(t), 1)
        return [float(np.log1p(len(str(text)))), float(np.log1p(n)), len(set(t)) / n]

    @classmethod
    def extract_batch(cls, raw_texts: list) -> np.ndarray:
        try:
            with joblib.parallel_config(backend=JOBLIB_BACKEND, n_jobs=N_PARALLEL_PRE):
                feats = Parallel(n_jobs=N_PARALLEL_PRE)(delayed(_global_extract_meta_single)(t) for t in raw_texts)
        except Exception:
            feats = [cls.extract(t) for t in raw_texts]
        return np.array(feats, dtype=np.float32)

# ═══════════════════════════════════════════════════════════════════
# 3. FEATURE PIPELINE CẢI TIẾN (MinMaxScaler, tham số linh hoạt)
# ═══════════════════════════════════════════════════════════════════
class FeaturePipeline:
    def __init__(self,
                 tfidf_max_features=15000,
                 tfidf_ngram_range=(1,3),
                 hashing_n_features=2**14,
                 use_pos=True,
                 use_hashing=True,
                 sublinear_tf=True):
        self.tfidf = TfidfVectorizer(
            ngram_range=tfidf_ngram_range,
            max_features=tfidf_max_features,
            sublinear_tf=sublinear_tf
        )
        self.hash_vec = HashingVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            n_features=hashing_n_features,
            alternate_sign=True
        ) if use_hashing else None
        self.scaler_manual = MinMaxScaler(feature_range=(0, 1))  # fix lỗi âm
        self.scaler_pos = MinMaxScaler(feature_range=(0, 1))
        self.use_pos = use_pos
        self.use_hashing = use_hashing

    def fit_transform(self, raw_texts, clean_texts, y, pos_cache=None, manual_features=None):
        X_t = self.tfidf.fit_transform(clean_texts)
        # [ARCH-7] cho phép truyền manual_features đã tính sẵn từ bên ngoài
        # (train_v40() cũng cần ma trận này để nối trực tiếp vào meta‑learner,
        # tránh chạy extract_features_batch 2 lần trên cùng dữ liệu).
        if manual_features is None:
            manual_features = TextPreprocessor.extract_features_batch(raw_texts)
        X_m = self.scaler_manual.fit_transform(manual_features)
        parts = [X_t, csr_matrix(X_m)]
        if self.use_hashing:
            parts.append(self.hash_vec.transform(clean_texts))
        if self.use_pos and pos_cache is not None:
            p_mat = np.array([[d['ratio_adj'], d['ratio_verb']] for d in pos_cache[0]], dtype=np.float32)
            parts.append(csr_matrix(self.scaler_pos.fit_transform(p_mat)))
        return sp_hstack(parts, format='csr')

    def transform(self, raw_texts, clean_texts, pos_cache=None, manual_features=None):
        X_t = self.tfidf.transform(clean_texts)
        if manual_features is None:
            manual_features = TextPreprocessor.extract_features_batch(raw_texts)
        # [SCALE-14] MinMaxScaler.transform() KHÔNG tự clip — nếu Cal/Test có
        # review dài bất thường khiến pos_s/neg_s vượt xa range đã thấy ở
        # train, giá trị scale ra sẽ vượt ngoài [0,1] và có thể nhiễu meta‑
        # learner tuyến tính ở các outlier hiếm gặp. fit_transform() không
        # cần clip (đầu ra vốn đã nằm trong [0,1] với chính dữ liệu dùng để
        # fit), chỉ transform() trên dữ liệu KHÁC mới cần.
        X_m = np.clip(self.scaler_manual.transform(manual_features), 0.0, 1.0)
        parts = [X_t, csr_matrix(X_m)]
        if self.use_hashing:
            parts.append(self.hash_vec.transform(clean_texts))
        if self.use_pos and pos_cache is not None:
            p_mat = np.array([[d['ratio_adj'], d['ratio_verb']] for d in pos_cache[0]], dtype=np.float32)
            parts.append(csr_matrix(np.clip(self.scaler_pos.transform(p_mat), 0.0, 1.0)))  # [SCALE-14]
        return sp_hstack(parts, format='csr')

    def scale_manual_features(self, manual_features):
        """[ARCH-7] Scale ma trận manual_features bằng scaler ĐÃ FIT (dùng
        chung với scaler đã dùng cho base learners) — để nối vào meta‑train/
        cal/test mà không lệch thang đo so với những gì base learners thấy.
        [SCALE-14] Clip về [0,1]: xem transform() ở trên."""
        return np.clip(self.scaler_manual.transform(manual_features), 0.0, 1.0)

# ═══════════════════════════════════════════════════════════════════
# 4. BASE LEARNERS POOL CẢI TIẾN (không dùng ComplementNB)
# ═══════════════════════════════════════════════════════════════════
class BaseLearnersPool:
    """
    [GROUPCV-10] Trước đây SVM dùng CalibratedClassifierCV(cv=3) với cv mặc
    định (StratifiedKFold, KHÔNG group-aware) cho bước hiệu chỉnh Platt
    (sigmoid) — không nhất quán với StratifiedGroupKFold dùng ở khắp nơi
    khác (Optuna CV, OOF stacking): nếu 2 bản ghi near-duplicate cùng
    near_dup_group_id rơi vào 2 phía train/hold-out của chính bước calibrate
    nội bộ này, xác suất SVM đầu ra có thể bị lạc quan hơn thực tế cho những
    review gần giống nhau. Từ v40.2, khi có `groups`, ta tự dựng danh sách
    fold bằng StratifiedGroupKFold rồi truyền thẳng vào CalibratedClassifierCV
    (cv=<list các (train_idx, test_idx)>) — đây là cách group-aware-hoá
    CalibratedClassifierCV mà không phụ thuộc API metadata-routing của các
    bản sklearn mới.

    Learner giờ được TẠO MỚI mỗi lần cần fit (qua build_fresh_learner()) thay
    vì giữ 1 instance cố định rồi clone(): fold group-aware của SVM phụ
    thuộc (y, groups) của CHÍNH dữ liệu sắp fit, khác nhau giữa mỗi fold OOF
    và lần fit full-train cuối cùng — không thể "chuẩn bị sẵn rồi clone".
    """
    def __init__(self, lr_C=1.5, svm_C=1.0, seed=42, svm_cv_splits=3,
                 include_svm=True, include_lgbm=True):
        self.lr_C = lr_C
        self.svm_C = svm_C
        self.seed = seed
        self.svm_cv_splits = svm_cv_splits
        names = ['lr']
        if include_svm:
            names.append('svm')
        # [ARCH-19] include_svm/include_lgbm cho phép tắt từng learner mà
        # không sửa code — hữu ích nếu ablation cho thấy SVM/LinearSVC không
        # đóng góp tốt cho ensemble (cân nhắc riêng, xem thảo luận kèm theo).
        if include_lgbm and HAS_LGBM:
            names.append('lgbm')
        self.learner_names = names
        self.learners = {}   # điền sau khi fit() — xem docstring ở trên

    def build_fresh_learner(self, name, y_for_fit, groups_for_fit=None):
        if name == 'lr':
            return LogisticRegression(C=self.lr_C, class_weight='balanced', max_iter=1000, random_state=self.seed)
        if name == 'lgbm':
            return LGBMClassifier(class_weight='balanced', random_state=self.seed, verbose=-1)
        if name == 'svm':
            base_svm = LinearSVC(C=self.svm_C, class_weight='balanced', random_state=self.seed, max_iter=2000)
            if groups_for_fit is not None:
                n_groups = len(np.unique(groups_for_fit))
                n_splits = min(self.svm_cv_splits, max(2, n_groups))
                try:
                    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
                    folds = list(sgkf.split(np.zeros(len(y_for_fit)), y_for_fit, groups_for_fit))
                    return CalibratedClassifierCV(base_svm, method='sigmoid', cv=folds)
                except Exception as e:
                    logger.warning(f"[GROUPCV-10] StratifiedGroupKFold lỗi cho SVM calibration ({e}) "
                                    f"— fallback cv={self.svm_cv_splits} (không group-aware).")
            return CalibratedClassifierCV(base_svm, method='sigmoid', cv=self.svm_cv_splits)
        raise ValueError(f"Learner không xác định: {name}")

    def fit(self, X, y, groups=None):
        for name in self.learner_names:
            model = self.build_fresh_learner(name, y, groups)
            model.fit(X, y)
            self.learners[name] = model

    def predict_proba_dict(self, X):
        return {name: model.predict_proba(X) for name, model in self.learners.items()}

# ═══════════════════════════════════════════════════════════════════
# 5. META CLUSTERER & STACKING META‑LEARNER
# ═══════════════════════════════════════════════════════════════════
class MetaClusterer:
    def __init__(self, n_clusters=4, seed=42):
        self.nc = n_clusters
        self.sc = StandardScaler()
        self.km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)

    def fit_predict(self, X):
        return self.km.fit_predict(self.sc.fit_transform(X))

    def transform(self, X):
        return self.km.predict(self.sc.transform(X))

    def to_onehot(self, cids):
        oh = np.zeros((len(cids), self.nc), dtype=np.float32)
        oh[np.arange(len(cids)), cids] = 1.0
        return oh

class StackingMetaLearner:
    def __init__(self, C=1.0, seed=42):
        self.model = LogisticRegression(C=C, class_weight='balanced', max_iter=2000, random_state=seed)

    def fit(self, meta_train, y):
        self.model.fit(meta_train, y)

    def predict_proba(self, meta_test):
        return self.model.predict_proba(meta_test)


class ProbabilityCalibrator:
    """[ARCH-8] Calibration head kiểu Platt scaling, cài thủ công bằng một
    LogisticRegression nhỏ huấn luyện trên log‑odds của xác suất thô, để
    tránh phụ thuộc vào API CalibratedClassifierCV(cv='prefit') vốn đã bị
    loại bỏ ở các bản sklearn mới. Fit trên tập Calibration (KHÔNG phải
    Train, để không leak), áp dụng lại lên xác suất ở bất kỳ tập nào khác."""

    def __init__(self, seed=42):
        self.model = LogisticRegression(max_iter=2000, random_state=seed)
        self._eps = 1e-6

    def _to_logit_features(self, probs):
        p = np.clip(probs, self._eps, 1 - self._eps)
        return np.log(p / (1 - p))

    def fit(self, probs_cal, y_cal):
        self.model.fit(self._to_logit_features(probs_cal), y_cal)
        return self

    def predict_proba(self, probs):
        return self.model.predict_proba(self._to_logit_features(probs))


def tune_neutral_threshold(pos_minus_neg_cal, y_cal, candidates=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0)):
    """[FEAT-6] Dò ngưỡng T cho is_neutral bằng mutual information giữa cờ
    is_neutral(T) và nhãn thật trên tập Calibration — không cần biết trước
    tên nhãn "neutral" cụ thể là gì, chỉ cần T nào tách nhãn tốt nhất."""
    best_t, best_mi = candidates[0], -1.0
    for t in candidates:
        flag = (np.abs(pos_minus_neg_cal) < t).astype(np.float64).reshape(-1, 1)
        try:
            mi = mutual_info_classif(flag, y_cal, discrete_features=[True], random_state=42)[0]
        except Exception:
            mi = 0.0
        if mi > best_mi:
            best_mi, best_t = mi, t
    return best_t, best_mi

# ═══════════════════════════════════════════════════════════════════
# 6. DATA LOADING
# ═══════════════════════════════════════════════════════════════════
def load_data(input_dir='input_data', group_col=None):
    """
    group_col (MỚI — đồng bộ với train_v41.py): nếu truyền vào và cột tồn tại
    trong file (vd. 'near_dup_group_id' sinh bởi dedup_clustering.py), cột
    này được GIỮ LẠI cùng comment/sentiment để split_data_v40() dùng
    GroupShuffleSplit. Nếu không sửa ở đây, group_col sẽ bị cắt mất trước
    khi kịp dùng — v40 trước đây luôn hardcode về đúng 2 cột.
    """
    files = glob.glob(os.path.join(input_dir, '*.csv')) + glob.glob(os.path.join(input_dir, '*.xlsx'))
    if not files:
        logger.warning("Không có dữ liệu thật. Tạo 3000 mẫu giả để kiểm thử pipeline...")
        return pd.DataFrame({
            'comment': ['sản phẩm tuyệt vời', 'tệ quá hỏng rồi', 'bình thường dùng tạm'] * 1000,
            'sentiment': [2, 0, 1] * 1000
        })
    dfs = []
    for f in files:
        df = pd.read_csv(f) if f.endswith('.csv') else pd.read_excel(f)
        df.rename(columns=lambda c: 'comment' if c.lower() in ('comment', 'text') else ('sentiment' if c.lower() in ('sentiment', 'label') else c), inplace=True)
        cols = ['comment', 'sentiment']
        if group_col and group_col in df.columns and group_col not in cols:
            cols.append(group_col)
        if 'comment' in df.columns and 'sentiment' in df.columns:
            dfs.append(df[cols])
    return pd.concat(dfs, ignore_index=True)

# ═══════════════════════════════════════════════════════════════════
# 6a. CHIA DỮ LIỆU — GROUP-AWARE (đồng bộ với split_data_v41() trong
#     train_v41.py, copy nguyên logic để V40/V41 dùng CHUNG một cách chia,
#     bắt buộc cho so sánh Baseline vs Proposed Method sòng phẳng — Trục 1,
#     mục 2.2)
# ═══════════════════════════════════════════════════════════════════
def split_data_v40(df, y, group_col=None, seed=42):
    """
    [SPLIT-16] GroupShuffleSplit (bản v40.1) đảm bảo group không bị cắt đôi,
    nhưng KHÔNG hỗ trợ stratify theo nhãn — với phân bố lớp lệch tự nhiên
    (~37.7/25.5/36.8%), tỉ lệ lớp giữa Train/Cal/Test có thể lệch nhau đáng
    kể tuỳ seed, đặc biệt nếu vài near_dup_group lớn thiên hẳn về 1 lớp. Vì
    sklearn chưa có "StratifiedGroupShuffleSplit", cách chuẩn để vừa
    group-aware vừa xấp xỉ cân bằng lớp là lấy 1 fold từ StratifiedGroupKFold
    (n_splits=5 rồi 4) thay vì gọi GroupShuffleSplit trực tiếp — vẫn ra đúng
    tỉ lệ 60/20/20 nhưng ổn định hơn qua các seed. GroupShuffleSplit được
    giữ lại làm fallback bậc 2 nếu StratifiedGroupKFold gặp lỗi (vd. quá ít
    group cho một lớp hiếm).

    LƯU Ý: nếu áp dụng fix này, cần đồng bộ sang split_data_v41() trong
    train_v41.py — hai hàm này BẮT BUỘC dùng chung logic để so sánh Baseline
    vs Proposed Method còn sòng phẳng (Trục 1, mục 2.2).
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
                    "trước đây. CẢNH BÁO (Trục 1, mục 2.2): nếu dữ liệu có nhiều biến thể gần giống nhau, "
                    "F1/accuracy đo được có thể lạc quan hơn thực tế do leakage train/test.")
    idx_tr_cal, idx_te = train_test_split(idx_all, test_size=0.2, stratify=y, random_state=seed)
    idx_tr, idx_cal = train_test_split(idx_tr_cal, test_size=0.25, stratify=y[idx_tr_cal], random_state=seed)
    return idx_tr, idx_cal, idx_te

# ═══════════════════════════════════════════════════════════════════
# 6b. CALIBRATION METRICS (ECE, Brier) — bổ sung theo yêu cầu báo cáo đa chỉ số
# ═══════════════════════════════════════════════════════════════════
def compute_calibration_metrics(y_true, probs, n_bins=10):
    """
    ECE (Expected Calibration Error) và Brier score đa lớp trên xác suất dự đoán.
    - Brier (đa lớp, Brier 1950): trung bình của sum_c (p_c - onehot_c)^2 trên mỗi
      mẫu. Không chuẩn hoá theo số lớp — càng thấp càng tốt, 0 là hoàn hảo.
    - ECE: chia mẫu vào n_bins theo max-confidence, đo |accuracy_bin - confidence_bin|,
      lấy trung bình có trọng số theo số mẫu mỗi bin.

    QUY ƯỚC BIÊN BIN (đã thống nhất 2026-07 — trước đó v40 dùng [lo,hi) nghiêng
    trái còn train_v41.py/train_phobert.py dùng (lo,hi] nghiêng phải cho bin
    đầu tiên là [lo,hi]; hai công thức khác nhau khiến bất kỳ confidence nào
    rơi đúng mốc 0.1/0.2/.../0.9 bị xếp lệch 1 bin, dù docstring cũ của v41
    từng ghi nhầm là "copy nguyên vẹn từ v40"): giờ CẢ BA FILE dùng chung quy
    ước bin đầu [lo,hi], các bin sau (lo,hi] — bắt buộc để ECE so sánh được
    đúng nghĩa giữa V40/V41/PhoBERT trong cùng một bảng.
    """
    n_classes = probs.shape[1]
    y_onehot = np.eye(n_classes)[y_true]
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences <= hi) if i == 0 \
            else (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(accuracies[mask].mean() - confidences[mask].mean())

    return {'ece': float(ece), 'brier': float(brier)}

# ═══════════════════════════════════════════════════════════════════
# 7. OPTUNA OBJECTIVE (tối ưu feature + base learner LR)
# ═══════════════════════════════════════════════════════════════════
def optuna_objective(trial, raw_train, clean_train, y_train, groups_train=None,
                      manual_train=None, pos_train=None):
    """
    [LEAK-9] Bản v40.1 fit pipe.fit_transform() (TF‑IDF vocab/idf,
    MinMaxScaler min/max cho manual + POS) MỘT LẦN trên toàn bộ raw_train,
    RỒI MỚI chia fold để cross_val_score() — nghĩa là thống kê tiền xử lý đã
    "nhìn thấy" đúng phần dữ liệu sẽ đóng vai trò validation fold, trước khi
    nó kịp bị coi là hold-out. Đây là leak thống kê feature (không phải leak
    nhãn), khiến F1 nội bộ của Optuna lạc quan hơn thực tế một chút — KHÔNG
    ảnh hưởng trực tiếp đến F1/accuracy báo cáo trên tập Test thật (pipe
    dùng cho Test luôn được fit lại từ đầu, chỉ trên Train, ở train_v40()),
    nhưng có thể khiến Optuna đánh giá sai lệch tương đối giữa các cấu hình
    ứng viên. Từ v40.2: fit một FeaturePipeline MỚI cho từng fold, chỉ trên
    phần train của fold đó, rồi transform() phần validation — đúng nguyên
    tắc "mọi bước fit() phải nằm trong vòng lặp CV".

    manual_train/pos_train (mới): nhận sẵn manual lexicon features / POS
    cache đã tính 1 lần ở train_v40() (không phụ thuộc hyperparameter nào ở
    đây) — tránh tính lại extract_features_batch()/extract_pos_features_batch()
    trên toàn bộ raw_train ở MỖI trial (bản gốc làm vậy dù kết quả giống hệt
    nhau qua các trial, lãng phí đáng kể nếu POS tagging chạy qua underthesea).
    """
    max_feat = trial.suggest_int('tfidf_max_feat', 8000, 20000, step=1000)
    ngram_choice = trial.suggest_categorical('ngram_range', [(1,2), (1,3), (1,4)])
    lr_C = trial.suggest_float('lr_C', 0.1, 10.0, log=True)
    hash_feat = trial.suggest_categorical('hash_features', [2**14, 2**15])
    sublinear = trial.suggest_categorical('sublinear_tf', [True, False])

    if groups_train is not None:
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        splits = list(cv.split(np.zeros(len(y_train)), y_train, groups_train))
    else:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        splits = list(cv.split(np.zeros(len(y_train)), y_train))

    fold_f1 = []
    for tr_idx, va_idx in splits:
        pipe_fold = FeaturePipeline(
            tfidf_max_features=max_feat,
            tfidf_ngram_range=ngram_choice,
            hashing_n_features=hash_feat,
            use_pos=HAS_POS,
            use_hashing=True,
            sublinear_tf=sublinear
        )
        X_fold_tr = pipe_fold.fit_transform(
            [raw_train[i] for i in tr_idx], [clean_train[i] for i in tr_idx], y_train[tr_idx],
            pos_cache=_slice_pos_cache(pos_train, tr_idx),
            manual_features=manual_train[tr_idx] if manual_train is not None else None
        )
        X_fold_va = pipe_fold.transform(
            [raw_train[i] for i in va_idx], [clean_train[i] for i in va_idx],
            pos_cache=_slice_pos_cache(pos_train, va_idx),
            manual_features=manual_train[va_idx] if manual_train is not None else None
        )
        learner = LogisticRegression(C=lr_C, class_weight='balanced', max_iter=1000, random_state=42)
        learner.fit(X_fold_tr, y_train[tr_idx])
        pred = learner.predict(X_fold_va)
        fold_f1.append(f1_score(y_train[va_idx], pred, average='macro'))

    return float(np.mean(fold_f1))

# ═══════════════════════════════════════════════════════════════════
# 8. MAIN PIPELINE V40
# ═══════════════════════════════════════════════════════════════════
def train_v40(input_dir='input_data', n_trials=30, seed=42, group_col=None,
               include_svm=True, include_lgbm=True, use_cluster_features=True,
               use_neutral_indicator=True, n_clusters=4):
    """
    [ARCH-19] include_svm / include_lgbm / use_cluster_features: mặc định
    True (giữ nguyên hành vi V40 gốc). Thêm các cờ này để có thể thử nghiệm
    "V40 không SVM" / "V40 không cluster one-hot" mà không phải sửa code —
    hữu ích nếu quyết định loại LinearSVC / bỏ KMeans cluster encoding cho
    kiến trúc chung của dự án cũng nên áp dụng lại cho baseline V40, chứ
    không chỉ cho phiên bản đề xuất. Xem thảo luận kèm theo về việc liệu V40
    (baseline) có NÊN giữ nguyên các thành phần này để phép so sánh "V40 vs
    V41" phản ánh đúng kiến trúc gốc, hay nên đồng bộ luôn.
    """
    random.seed(seed)
    np.random.seed(seed)
    t_start = time.time()

    # ── 1. Nạp dữ liệu ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"🚀 BẮT ĐẦU HUẤN LUYỆN V40 (seed={seed})")
    logger.info("=" * 60)
    df = load_data(input_dir, group_col=group_col)
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])
    raw_texts = df['comment'].astype(str).tolist()
    logger.info(f"Đã nạp {len(df)} mẫu. Nhãn: {le.classes_}")
    # [DATA-18] Nếu input_dir có NHIỀU file được gộp (load_data() dùng
    # glob + pd.concat), một chênh lệch chính tả/viết hoa nhỏ giữa các file
    # (vd. "Positive" ở file này, "positive" ở file khác) sẽ lặng lẽ tạo
    # thêm lớp thứ 4+ mà không báo lỗi — assert này chỉ là lưới an toàn rẻ
    # tiền, không tự sửa dữ liệu.
    if len(le.classes_) != 3:
        logger.warning(f"⚠️  Kỳ vọng 3 lớp (negative/neutral/positive) nhưng LabelEncoder thấy "
                        f"{len(le.classes_)}: {list(le.classes_)} — kiểm tra chính tả/viết hoa nhãn "
                        f"nếu input_dir có nhiều file.")

    if group_col is not None and group_col not in df.columns:
        logger.warning(f"Cột group_col='{group_col}' không tồn tại trong dữ liệu — bỏ qua group-aware split.")
        group_col = None

    # ── 2. Tiền xử lý văn bản ─────────────────────────────────────
    logger.info("Tiền xử lý văn bản...")
    clean_texts = TextPreprocessor.preprocess_batch(raw_texts)

    # ── 3. Phân chia dữ liệu: Train / Calibration / Test (60/20/20) ─
    #      Group-aware nếu có group_col (đồng bộ V40/V41 — Trục 1, mục 2.2)
    idx_tr, idx_cal, idx_te = split_data_v40(df, y, group_col=group_col, seed=seed)

    raw_tr   = [raw_texts[i] for i in idx_tr]
    clean_tr = [clean_texts[i] for i in idx_tr]
    y_tr     = y[idx_tr]

    raw_cal   = [raw_texts[i] for i in idx_cal]
    clean_cal = [clean_texts[i] for i in idx_cal]
    y_cal     = y[idx_cal]

    raw_te   = [raw_texts[i] for i in idx_te]
    clean_te = [clean_texts[i] for i in idx_te]
    y_te     = y[idx_te]

    logger.info(f"Train: {len(y_tr)} | Calibration: {len(y_cal)} | Test: {len(y_te)}")

    # groups_tr (FIX mục 3.3): dùng cho StratifiedGroupKFold trong Optuna CV +
    # OOF stacking bên dưới, để CV nội bộ trên tập train cũng group-aware,
    # không chỉ split ngoài cùng (train/cal/test).
    groups_tr = df[group_col].values[idx_tr] if group_col is not None else None

    # ── 4. Cache đặc trưng nặng (POS) ────────────────────────────
    logger.info("Trích xuất POS features...")
    pos_tr = TextPreprocessor.extract_pos_features_batch(raw_tr) if HAS_POS else None
    pos_cal = TextPreprocessor.extract_pos_features_batch(raw_cal) if HAS_POS else None
    pos_te = TextPreprocessor.extract_pos_features_batch(raw_te) if HAS_POS else None

    # ── 4b. [ARCH-7][FEAT-6] Manual lexicon features — tính 1 LẦN DUY NHẤT
    # ở đây để (a) tránh chạy extract_features_batch lặp lại bên trong
    # FeaturePipeline lẫn khi nối vào meta‑learner, (b) dò ngưỡng is_neutral
    # tối ưu trên tập Calibration bằng mutual information (thay ngưỡng cứng
    # NEUTRAL_THRESHOLD_DEFAULT=2.0 nếu tìm được T tốt hơn).
    logger.info("Trích xuất manual lexicon features (pos_s/neg_s/sentiment_ratio/...)...")
    manual_tr = TextPreprocessor.extract_features_batch(raw_tr)
    manual_cal = TextPreprocessor.extract_features_batch(raw_cal)
    manual_te = TextPreprocessor.extract_features_batch(raw_te)

    idx_pmn = TextPreprocessor.IDX_POS_MINUS_NEG
    idx_neu = TextPreprocessor.IDX_IS_NEUTRAL
    if use_neutral_indicator:
        best_T, best_mi = tune_neutral_threshold(manual_cal[:, idx_pmn], y_cal)
        logger.info(f"[FEAT-6] Ngưỡng neutral T tối ưu trên Cal: T={best_T} (MI={best_mi:.4f}) "
                    f"— ghi đè cột is_neutral cho train/cal/test.")
        for mat in (manual_tr, manual_cal, manual_te):
            mat[:, idx_neu] = (np.abs(mat[:, idx_pmn]) < best_T).astype(np.float32)
    else:
        # [ABLATION] Tắt neutral-region indicator: đặt cột về 0 cho cả 3 tập thay vì XOÁ
        # cột, để IDX_IS_NEUTRAL và toàn bộ code inference không bị lệch chỉ số.
        # Cột hằng 0 không mang thông tin nên tương đương loại bỏ đặc trưng.
        best_T, best_mi = float('nan'), float('nan')
        logger.info('[ABLATION] use_neutral_indicator=False — cột is_neutral bị vô hiệu '
                    'hoá (đặt = 0), KHÔNG dò ngưỡng T trên tập Cal.')
        for mat in (manual_tr, manual_cal, manual_te):
            mat[:, idx_neu] = 0.0

    # ── 5. Tối ưu siêu tham số bằng Optuna ───────────────────────
    logger.info(f"Tối ưu siêu tham số với Optuna ({n_trials} trials)...")
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(
        lambda trial: optuna_objective(trial, raw_tr, clean_tr, y_tr, groups_train=groups_tr,
                                        manual_train=manual_tr, pos_train=pos_tr),
        n_trials=n_trials,
        n_jobs=1,  # tránh xung đột với joblib bên trong
        show_progress_bar=True
    )
    best = study.best_params
    logger.info(f"Best params: {best}")
    logger.info(f"Best CV F1: {study.best_value:.4f}")

    # ── 6. Fit Feature Pipeline cuối cùng với best params ───────
    logger.info("Fit Feature Pipeline với best params...")
    pipe = FeaturePipeline(
        tfidf_max_features=best['tfidf_max_feat'],
        tfidf_ngram_range=best['ngram_range'],
        hashing_n_features=best['hash_features'],
        use_pos=HAS_POS,
        use_hashing=True,
        sublinear_tf=best['sublinear_tf']
    )
    X_tr = pipe.fit_transform(raw_tr, clean_tr, y_tr, pos_cache=pos_tr, manual_features=manual_tr)
    X_cal = pipe.transform(raw_cal, clean_cal, pos_cache=pos_cal, manual_features=manual_cal)
    X_te = pipe.transform(raw_te, clean_te, pos_cache=pos_te, manual_features=manual_te)

    # ── 7. Meta‑features: Cluster one‑hot ────────────────────────
    # [ARCH-19] use_cluster_features=False → bỏ qua KMeans cluster one-hot
    # (mảng 0 cột, hstack phía dưới không cần đổi) — dùng để thử "V40 không
    # cluster encoding" nếu ablation của dự án cho thấy thành phần này không
    # đóng góp tốt cho ensemble.
    clusterer = MetaClusterer(n_clusters=n_clusters, seed=seed)
    if use_cluster_features:
        logger.info("Tạo cluster one‑hot...")
        meta_feat_tr = MetaFeatureExtractor.extract_batch(raw_tr)
        oh_tr = clusterer.to_onehot(clusterer.fit_predict(meta_feat_tr))
        oh_cal = clusterer.to_onehot(clusterer.transform(MetaFeatureExtractor.extract_batch(raw_cal)))
        oh_te  = clusterer.to_onehot(clusterer.transform(MetaFeatureExtractor.extract_batch(raw_te)))
    else:
        logger.info("[ARCH-19] use_cluster_features=False — bỏ qua cluster one-hot.")
        oh_tr, oh_cal, oh_te = (np.zeros((len(y_tr), 0)), np.zeros((len(y_cal), 0)), np.zeros((len(y_te), 0)))

    # ── 8. Tạo OOF cho từng base learner (stacking) — [LEAK-9] FIX ──────
    # Bản v40.1 cắt fold TRỰC TIẾP trên X_tr (đã fit TF‑IDF/MinMaxScaler một
    # lần trên toàn bộ raw_tr ở bước 6) — nghĩa là thống kê tiền xử lý của
    # MỌI fold (kể cả phần sẽ là validation) đều đã "nhìn thấy" nhau trước
    # khi bị chia. Từ v40.2: với MỖI fold, fit một FeaturePipeline MỚI chỉ
    # trên phần train của fold đó rồi transform() phần validation — giống hệt
    # cách X_te được tạo ra ở bước 6 (pipe fit trên Train, transform Test),
    # chỉ khác là lặp lại ở quy mô fold nội bộ. Vòng lặp ngoài giờ là FOLD
    # (không phải learner) để 3 learner dùng CHUNG một lần fit pipeline/fold
    # thay vì mỗi learner tự fit lại pipeline (lãng phí gấp 3 lần).
    logger.info(f"Tạo OOF với {N_OOF_FOLDS} folds (fold-local feature fitting — [LEAK-9])...")
    pool = BaseLearnersPool(lr_C=best['lr_C'], seed=seed, include_svm=include_svm, include_lgbm=include_lgbm)
    learner_names = pool.learner_names
    n_classes = len(le.classes_)

    if groups_tr is not None:
        skf = StratifiedGroupKFold(n_splits=N_OOF_FOLDS, shuffle=True, random_state=seed)
        fold_indices = list(skf.split(np.zeros(len(y_tr)), y_tr, groups_tr))
    else:
        skf = StratifiedKFold(n_splits=N_OOF_FOLDS, shuffle=True, random_state=seed)
        fold_indices = list(skf.split(np.zeros(len(y_tr)), y_tr))

    oof_by_learner = {name: np.zeros((len(y_tr), n_classes), dtype=np.float64) for name in learner_names}

    for fold_i, (tr_idx, va_idx) in enumerate(fold_indices, 1):
        logger.debug(f"  OOF fold {fold_i}/{N_OOF_FOLDS}: fit fold-local FeaturePipeline...")
        pipe_fold = FeaturePipeline(
            tfidf_max_features=best['tfidf_max_feat'],
            tfidf_ngram_range=best['ngram_range'],
            hashing_n_features=best['hash_features'],
            use_pos=HAS_POS, use_hashing=True, sublinear_tf=best['sublinear_tf']
        )
        y_fold_tr = y_tr[tr_idx]
        groups_fold_tr = groups_tr[tr_idx] if groups_tr is not None else None

        X_fold_tr = pipe_fold.fit_transform(
            [raw_tr[i] for i in tr_idx], [clean_tr[i] for i in tr_idx], y_fold_tr,
            pos_cache=_slice_pos_cache(pos_tr, tr_idx), manual_features=manual_tr[tr_idx]
        )
        X_fold_va = pipe_fold.transform(
            [raw_tr[i] for i in va_idx], [clean_tr[i] for i in va_idx],
            pos_cache=_slice_pos_cache(pos_tr, va_idx), manual_features=manual_tr[va_idx]
        )

        for name in learner_names:
            # [GROUPCV-10] build_fresh_learner tự dựng SVM group-aware nếu có groups_fold_tr
            model = pool.build_fresh_learner(name, y_fold_tr, groups_fold_tr)
            model.fit(X_fold_tr, y_fold_tr)
            oof_by_learner[name][va_idx] = model.predict_proba(X_fold_va)

    # Ghép OOF của tất cả learner thành meta‑train (n_train, n_classes * len(learners))
    oof_meta_train = np.hstack([oof_by_learner[name] for name in learner_names])

    # Huấn luyện lại toàn bộ learner trên full train để dùng cho cal/test.
    # X_tr ở đây được tạo từ `pipe` fit ở bước 6 trên TOÀN BỘ raw_tr — điều
    # này KHÔNG leak gì (đây là lần fit "thật" dùng để deploy, X_cal/X_te
    # chưa từng được pipe/pool nhìn thấy). [GROUPCV-10]: truyền groups_tr để
    # SVM cũng group-aware ở lần fit full-train này.
    pool.fit(X_tr, y_tr, groups=groups_tr)

    # Tạo meta‑test: ghép xác suất từ các learner trên cal/test
    prob_dict_cal = pool.predict_proba_dict(X_cal)
    prob_dict_te  = pool.predict_proba_dict(X_te)

    oof_meta_cal = np.hstack([prob_dict_cal[name] for name in learner_names])
    oof_meta_te  = np.hstack([prob_dict_te[name] for name in learner_names])

    # Gắn thêm cluster one‑hot + [ARCH-7] manual lexicon features (scaled
    # bằng CHÍNH scaler đã fit trong pipe, để cùng thang đo với những gì
    # base learners đã thấy) — để meta‑learner tuyến tính nhìn thấy trực
    # tiếp pos_s/neg_s/is_neutral/sentiment_ratio/... thay vì chỉ suy ra
    # gián tiếp qua xác suất của base learners.
    manual_tr_scaled = pipe.scale_manual_features(manual_tr)
    manual_cal_scaled = pipe.scale_manual_features(manual_cal)
    manual_te_scaled = pipe.scale_manual_features(manual_te)

    meta_train = np.hstack([oof_meta_train, oh_tr, manual_tr_scaled])
    meta_cal   = np.hstack([oof_meta_cal, oh_cal, manual_cal_scaled])
    meta_te    = np.hstack([oof_meta_te, oh_te, manual_te_scaled])

    # ── 9. Huấn luyện Meta‑learner, chọn C tốt nhất trên tập Cal ─
    logger.info("Chọn siêu tham số Meta‑learner dùng tập Calibration...")
    best_c = None
    best_f1_cal = -1
    for C in [0.01, 0.1, 1.0, 5.0, 10.0]:
        meta_tmp = StackingMetaLearner(C=C, seed=seed)
        meta_tmp.fit(meta_train, y_tr)
        pred_cal = np.argmax(meta_tmp.predict_proba(meta_cal), axis=1)
        f1_cal = f1_score(y_cal, pred_cal, average='macro')
        logger.debug(f"  C={C:.3f} → F1-macro trên Cal = {f1_cal:.4f}")
        if f1_cal > best_f1_cal:
            best_f1_cal = f1_cal
            best_c = C
    logger.info(f"Chọn C={best_c} với F1-Cal={best_f1_cal:.4f}")

    # Huấn luyện meta‑learner cuối cùng
    meta = StackingMetaLearner(C=best_c, seed=seed)
    meta.fit(meta_train, y_tr)

    # ── 9b. [ARCH-8] Hiệu chỉnh xác suất (calibration head kiểu Platt) ──
    # Fit trên tập Calibration (KHÔNG phải Train, tránh leak). Chỉ CHẤP
    # NHẬN nếu ECE trên Cal cải thiện mà F1‑macro trên Cal không giảm quá
    # 0.5 điểm % — vì calibration có thể đánh đổi khả năng phân biệt lớp.
    logger.info("[ARCH-8] Fit calibration head trên tập Calibration...")
    prob_cal_raw = meta.predict_proba(meta_cal)
    pred_cal_raw = np.argmax(prob_cal_raw, axis=1)
    f1_cal_raw = f1_score(y_cal, pred_cal_raw, average='macro')
    calib_cal_raw = compute_calibration_metrics(y_cal, prob_cal_raw)

    calibrator = ProbabilityCalibrator(seed=seed).fit(prob_cal_raw, y_cal)
    prob_cal_calibrated = calibrator.predict_proba(prob_cal_raw)
    pred_cal_calibrated = np.argmax(prob_cal_calibrated, axis=1)
    f1_cal_calibrated = f1_score(y_cal, pred_cal_calibrated, average='macro')
    calib_cal_calibrated = compute_calibration_metrics(y_cal, prob_cal_calibrated)

    F1_DROP_TOLERANCE = 0.005  # 0.5 điểm %
    use_calibrator = (
        calib_cal_calibrated['ece'] < calib_cal_raw['ece'] and
        f1_cal_calibrated >= f1_cal_raw - F1_DROP_TOLERANCE
    )
    logger.info(f"  Cal (raw)        : F1={f1_cal_raw:.4f} ECE={calib_cal_raw['ece']:.4f}")
    logger.info(f"  Cal (calibrated) : F1={f1_cal_calibrated:.4f} ECE={calib_cal_calibrated['ece']:.4f}")
    logger.info(f"  → {'ÁP DỤNG' if use_calibrator else 'GIỮ NGUYÊN (bỏ)'} calibration head "
                f"cho probability đầu ra cuối cùng.")

    # ── 10. Đánh giá trên tập Test ────────────────────────────────
    logger.info("Đánh giá trên tập Test...")
    prob_te = meta.predict_proba(meta_te)
    if use_calibrator:
        prob_te = calibrator.predict_proba(prob_te)
    pred_te = np.argmax(prob_te, axis=1)
    f1_test = f1_score(y_te, pred_te, average='macro')
    acc_test = accuracy_score(y_te, pred_te)
    calib = compute_calibration_metrics(y_te, prob_te)
    logger.info(f"✅ F1-Macro trên TEST = {f1_test:.4f}")
    logger.info(f"✅ Accuracy      = {acc_test:.4f}")
    logger.info(f"✅ ECE={calib['ece']:.4f}  Brier={calib['brier']:.4f}")
    logger.info("\n" + classification_report(y_te, pred_te, target_names=[str(c) for c in le.classes_]))

    # ── 11. Lưu mô hình ─────────────────────────────────────────
    # FIX (mục 3.1 báo cáo review): thư mục lưu model được TÁCH THEO VERSION
    # (models/production/v40/ thay vì models/production/ dùng chung với V41).
    # Trước đây V40 và V41 ghi đè lẫn nhau khi chạy trong cùng thư mục — đã
    # xác nhận bằng MD5 các file .pkl thay đổi sau khi chạy V41 sau V40.
    #
    # FIX #2 (mục #4 checklist Cross-Domain Tiki): thư mục lưu model được
    # TÁCH THÊM THEO SEED (models/production/v40/seed_{seed}/). Trước đây
    # prod_dir cố định theo VERSION nhưng KHÔNG theo seed → vòng lặp 10 seed
    # ở __main__ ghi đè cùng 6 file .pkl mỗi lần, chỉ còn model của seed chạy
    # cuối (SEEDS[-1]) trên đĩa, 9 model kia mất — dù results_*_multiseed.json
    # (JSON) vẫn trông "đủ 10 seed" vì metrics được tính trước khi model bị
    # ghi đè. Giai đoạn C (inference Tiki) cần nạp model THẬT của cả 10 seed,
    # nên bắt buộc phải tách thư mục theo seed ở đây.
    prod_dir = os.path.join('models', 'production', VERSION, f'seed_{seed}')
    os.makedirs(prod_dir, exist_ok=True)
    joblib.dump(pool, os.path.join(prod_dir, 'base_learners.pkl'))
    joblib.dump(meta, os.path.join(prod_dir, 'meta_learner.pkl'))
    joblib.dump(pipe, os.path.join(prod_dir, 'feature_pipeline.pkl'))
    # [ARCH-19] Lưu None nếu use_cluster_features=False — clusterer chưa
    # từng được fit() trong trường hợp đó, để predict_v40() biết bỏ qua
    # thay vì gọi .transform() trên một KMeans/StandardScaler chưa fit.
    joblib.dump(clusterer if use_cluster_features else None, os.path.join(prod_dir, 'clusterer.pkl'))
    joblib.dump(le, os.path.join(prod_dir, 'label_encoder.pkl'))
    joblib.dump(learner_names, os.path.join(prod_dir, 'learner_names.pkl'))
    # [ARCH-8] Lưu calibrator CHỈ KHI được chấp nhận (else lưu None để
    # inference biết bỏ qua bước hiệu chỉnh). [FEAT-6] Lưu kèm best_T để
    # inference tái tạo đúng cột is_neutral như lúc train.
    joblib.dump(calibrator if use_calibrator else None, os.path.join(prod_dir, 'probability_calibrator.pkl'))
    joblib.dump(float(best_T), os.path.join(prod_dir, 'neutral_threshold.pkl'))

    # Kích thước mô hình — liệt kê tường minh (không dùng glob *.pkl) để tránh
    # tính nhầm phải file .pkl của V41 nếu hai script cùng chạy trong 1 thư mục
    # models/production dùng chung.
    v40_pkl_files = ['base_learners.pkl', 'meta_learner.pkl', 'feature_pipeline.pkl',
                      'clusterer.pkl', 'label_encoder.pkl', 'learner_names.pkl',
                      'probability_calibrator.pkl', 'neutral_threshold.pkl']
    model_size_mb = sum(
        os.path.getsize(os.path.join(prod_dir, fn))
        for fn in v40_pkl_files if os.path.exists(os.path.join(prod_dir, fn))
    ) / (1024 * 1024)
    training_time_sec = time.time() - t_start

    logger.info(f"📁 Đã lưu mô hình vào {prod_dir}  ({model_size_mb:.2f} MB)")
    logger.info(f"⏱️  Tổng thời gian: {training_time_sec:.1f}s")

    # Trả về dict đầy đủ (thay vì chỉ f1_test) để vòng lặp đa seed ở __main__
    # lưu được accuracy/ECE/Brier/thời gian/kích thước cho từng seed.
    return {
        'f1': float(f1_test),
        'accuracy': float(acc_test),
        'ece': calib['ece'],
        'brier': calib['brier'],
        'training_time_sec': float(training_time_sec),
        'model_size_mb': float(model_size_mb),
        'best_params': {k: (list(v) if isinstance(v, tuple) else v) for k, v in best.items()},
        # [FEAT-6] / [ARCH-8] — thông tin bổ sung để theo dõi/so sánh giữa
        # các seed và giữa các phiên bản (V40 gốc vs V40 đã vá).
        'neutral_threshold_T': float(best_T),
        'neutral_threshold_mi': float(best_mi),
        'calibrator_applied': bool(use_calibrator),
        'f1_cal_raw': float(f1_cal_raw),
        'f1_cal_calibrated': float(f1_cal_calibrated),
        'ece_cal_raw': float(calib_cal_raw['ece']),
        'ece_cal_calibrated': float(calib_cal_calibrated['ece']),
    }

# ═══════════════════════════════════════════════════════════════════
# 8b. [INFER-15] INFERENCE CHUẨN — nạp đủ artifact + tái tạo ĐÚNG pipeline
# ═══════════════════════════════════════════════════════════════════
# Rủi ro version-mismatch (mục 7 trong review): best_T (neutral_threshold.pkl)
# được lưu RIÊNG theo từng seed. Nếu script inference gọi thẳng
# TextPreprocessor.extract_features() mà quên nạp lại best_T để ghi đè cột
# is_neutral, nó sẽ lặng lẽ dùng NEUTRAL_THRESHOLD_DEFAULT=2.0 — lệch với
# ngưỡng thực sự dùng lúc train (training/serving skew). Hai hàm dưới đây là
# cách DUY NHẤT nên dùng để suy luận với model V40 đã lưu — nạp toàn bộ 8
# file .pkl (kể cả neutral_threshold.pkl) và tái tạo pipeline TỪNG BƯỚC một
# giống hệt lúc train (bước 6 → 10), để không ai phải "nhớ" ghi đè gì nữa.

def load_v40_bundle(prod_dir):
    """Nạp toàn bộ artifact cần cho inference từ models/production/v40/seed_{seed}/."""
    return {
        'pool': joblib.load(os.path.join(prod_dir, 'base_learners.pkl')),
        'meta': joblib.load(os.path.join(prod_dir, 'meta_learner.pkl')),
        'pipe': joblib.load(os.path.join(prod_dir, 'feature_pipeline.pkl')),
        'clusterer': joblib.load(os.path.join(prod_dir, 'clusterer.pkl')),
        'le': joblib.load(os.path.join(prod_dir, 'label_encoder.pkl')),
        'learner_names': joblib.load(os.path.join(prod_dir, 'learner_names.pkl')),
        'calibrator': joblib.load(os.path.join(prod_dir, 'probability_calibrator.pkl')),
        'best_T': joblib.load(os.path.join(prod_dir, 'neutral_threshold.pkl')),
    }

def predict_v40(raw_texts, bundle):
    """
    Trả về (nhãn dự đoán, ma trận xác suất). `raw_texts` là list[str] văn
    bản THÔ (chưa preprocess) — hàm tự làm toàn bộ các bước, THEO ĐÚNG THỨ
    TỰ đã dùng lúc train_v40(): preprocess → manual features (+ ghi đè
    is_neutral bằng best_T đã lưu) → pipe.transform() → base learners →
    OOF-style stacking input (nối theo ĐÚNG thứ tự learner_names đã lưu) →
    cluster one-hot → meta-learner → calibrator (nếu có).
    """
    clean_texts = TextPreprocessor.preprocess_batch(raw_texts)
    pos_cache = TextPreprocessor.extract_pos_features_batch(raw_texts) if HAS_POS else None
    manual_feats = TextPreprocessor.extract_features_batch(raw_texts)

    idx_pmn = TextPreprocessor.IDX_POS_MINUS_NEG
    idx_neu = TextPreprocessor.IDX_IS_NEUTRAL
    manual_feats[:, idx_neu] = (np.abs(manual_feats[:, idx_pmn]) < bundle['best_T']).astype(np.float32)

    X = bundle['pipe'].transform(raw_texts, clean_texts, pos_cache=pos_cache, manual_features=manual_feats)
    prob_dict = bundle['pool'].predict_proba_dict(X)
    oof_meta = np.hstack([prob_dict[name] for name in bundle['learner_names']])

    if bundle['clusterer'] is not None:
        meta_feat = MetaFeatureExtractor.extract_batch(raw_texts)
        oh = bundle['clusterer'].to_onehot(bundle['clusterer'].transform(meta_feat))
    else:
        # [ARCH-19] use_cluster_features=False lúc train → clusterer.pkl là None
        oh = np.zeros((len(raw_texts), 0))

    manual_scaled = bundle['pipe'].scale_manual_features(manual_feats)  # đã clip [0,1] — [SCALE-14]
    meta_input = np.hstack([oof_meta, oh, manual_scaled])

    prob = bundle['meta'].predict_proba(meta_input)
    if bundle['calibrator'] is not None:
        prob = bundle['calibrator'].predict_proba(prob)

    pred_idx = np.argmax(prob, axis=1)
    pred_label = bundle['le'].inverse_transform(pred_idx)
    return pred_label, prob

# ═══════════════════════════════════════════════════════════════════
# 9. CHẠY CHÍNH — ĐA SEED (10 seed, dùng CHUNG với V41 — xem seeds_config.py)
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    from seeds_config import SEEDS

    # ── GROUP-AWARE SPLIT (đồng bộ với train_v41.py — Trục 1, mục 2.2) ──
    # PHẢI khớp với GROUP_COL trong train_v41.py để V40 (Baseline) và V41
    # (Proposed Method) được đánh giá trên đúng cùng một cách chia group,
    # nếu không so sánh "Baseline vs Proposed" sẽ không sòng phẳng. Chạy
    # dedup_clustering.py trước để cột này tồn tại trong input_data/*.csv.
    GROUP_COL = 'near_dup_group_id'

    # n_trials cho Optuna khi chạy đa seed: dùng ĐÚNG 30 — GIỐNG lần chạy đơn
    # seed=42 đã báo cáo trong bài báo (Section 4.1, Table 8: Optuna runs 30
    # trials per system). Trước đây có giảm xuống 15 để tiết kiệm thời gian,
    # nhưng timing thật đo được ở 30 trials (~1156s/seed cho V40) cho thấy
    # chạy đủ 10 seed chỉ mất khoảng 3 giờ — không đáng đánh đổi lấy nguy cơ
    # seed=42 trong bảng đa-seed cho ra số KHÁC với Table 8 đã công bố.
    N_TRIALS_MULTISEED = 30

    logger.info("#" * 60)
    logger.info(f"CHẠY ĐA SEED V40 — {len(SEEDS)} seed: {SEEDS} — group_col='{GROUP_COL}'")
    logger.info("#" * 60)

    # Các chỉ số numeric sẽ được tính mean/std qua các seed trong 'summary'.
    METRIC_KEYS = ['f1', 'accuracy', 'ece', 'brier', 'training_time_sec', 'model_size_mb']
    results = {}
    summary = {}
    for i, s in enumerate(SEEDS, 1):
        logger.info(f"\n########## [V40] SEED {i}/{len(SEEDS)} = {s} ##########")
        metrics = train_v40(input_dir='input_data', n_trials=N_TRIALS_MULTISEED, seed=s,
                            group_col=GROUP_COL)
        results[s] = metrics
        # Lưu kết quả ngay sau MỖI seed — nếu máy bị ngắt giữa chừng (rớt kết
        # nối cloud, hết giờ thuê máy...) vẫn giữ được kết quả các seed đã chạy.
        summary = {}
        for key in METRIC_KEYS:
            vals = [results[s2][key] for s2 in results if key in results[s2]]
            if vals:
                summary[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        with open('results_v40_multiseed.json', 'w', encoding='utf-8') as f:
            json.dump({
                'version': 'v40',
                'seeds': SEEDS,
                'group_col': GROUP_COL,
                'n_trials': N_TRIALS_MULTISEED,
                'metrics_by_seed': {str(k): v for k, v in results.items()},
                'summary': summary,
                'completed_seeds': len(results),
                'total_seeds': len(SEEDS),
            }, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f"✅ V40 — qua {len(SEEDS)} seed:")
    for key in METRIC_KEYS:
        if key in summary:
            logger.info(f"   {key}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")
    logger.info(f"   Chi tiết từng seed: {results}")
    logger.info("📁 Đã lưu results_v40_multiseed.json")
    logger.info("=" * 60)