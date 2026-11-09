"""Va analyze_saved.py va finish_analysis.py: lay van ban tu corpus theo row_index
(khong con doc cot `text` trong file xac suat). Chay MOT lan tu thu muc goc repo:
    py scripts/patch_text_from_masked.py
Moi doan thay the phai khop dung 1 lan, neu khong script dung va khong ghi gi."""
import io, sys

def patch(path, pairs):
    s = io.open(path, encoding='utf-8', newline='').read()
    crlf = '\r\n' in s
    s = s.replace('\r\n', '\n')
    for i, (old, new) in enumerate(pairs, 1):
        n = s.count(old)
        if n != 1:
            sys.exit(f'[DUNG] {path}: doan #{i} khop {n} lan (can dung 1). Khong ghi file.')
        s = s.replace(old, new)
    if crlf:
        s = s.replace('\n', '\r\n')
    io.open(path, 'w', encoding='utf-8', newline='').write(s)
    print('Da va:', path)

# ----------------------------------------------------------------- analyze_saved.py
A = [
# 1) duong dan + corpus + row_index helpers
("""sys.path.insert(0, sys.argv[1]); R, O = sys.argv[1], sys.argv[2]
SEEDS = [42, 7, 123, 2024, 99, 555, 2026, 31337, 8080, 12321]
df = pd.read_csv(f'{R}/input_data/dataset_cleaned.csv', encoding='utf-8-sig')
""",
"""sys.path.insert(0, sys.argv[1]); R, O = sys.argv[1], sys.argv[2]
SEEDS = [42, 7, 123, 2024, 99, 555, 2026, 31337, 8080, 12321]
import os
def first_existing(*cands):
    for c in cands:
        if os.path.exists(c): return c
    sys.exit('[DUNG] Khong thay file: ' + ' | '.join(cands))
# Corpus: mac dinh la ban DA CHE PII trong repo. Muon dung ban goc: set AERS_CORPUS=<duong dan>.
CORPUS = os.environ.get('AERS_CORPUS') or first_existing(f'{R}/data/dataset_cleaned_masked.csv', f'{R}/input_data/dataset_cleaned.csv')
print('Corpus dang dung:', CORPUS, '(PII-masked)' if 'masked' in os.path.basename(CORPUS) else '(BAN GOC - khong cong bo)')
df = pd.read_csv(CORPUS, encoding='utf-8-sig')
PROBS = first_existing(f'{R}/results/v40_probs', f'{R}/reports/v40_probs')
def probs_path(s): return f'{PROBS}/v40_test_seed_{s}.csv'
"""),
("""A = json.load(open(f'{R}/results_v40_multiseed.json'))['metrics_by_seed']
P = json.load(open(f'{R}/results_phobert_multiseed.json'))['metrics_by_seed']
""",
"""A = json.load(open(first_existing(f'{R}/results/results_v40_multiseed.json', f'{R}/results_v40_multiseed.json')))['metrics_by_seed']
P = json.load(open(first_existing(f'{R}/results/results_phobert_multiseed.json', f'{R}/results_phobert_multiseed.json')))['metrics_by_seed']
"""),
# 2) doc file xac suat + kiem tra thu tu bang row_index
("""    p = pd.read_csv(f'{R}/reports/v40_probs/v40_test_seed_{s}.csv', encoding='utf-8-sig')
    # khop thu tu hang voi te
    assert (p.text.astype(str).values == df.comment.values[te].astype(str)).all(), 'thu tu khac'
""",
"""    p = pd.read_csv(probs_path(s), encoding='utf-8-sig')
    # khop thu tu hang voi te: bang row_index (file moi, khong co cot text); file cu con cot text thi doi chieu van ban
    if 'row_index' in p.columns:
        assert (p.row_index.values == te).all(), f'seed {s}: row_index khac chi so test tinh lai (sai ban sklearn? can 1.6.1)'
    else:
        assert (p.text.astype(str).values == df.comment.values[te].astype(str)).all(), 'thu tu khac'
    assert (p.y_true.astype(str).values == np.array(cls)[y[te]]).all(), f'seed {s}: y_true khac nhan corpus tai te'
"""),
# 3) dong cuoi doc lai file xac suat de tinh share
("""np.mean([len(pd.read_csv(f'{R}/reports/v40_probs/v40_test_seed_{s}.csv', encoding="utf-8-sig")) for s in SEEDS])""",
 """np.mean([len(pd.read_csv(probs_path(s), encoding="utf-8-sig")) for s in SEEDS])"""),
]

# ----------------------------------------------------------------- finish_analysis.py
F = [
("""DATA_CSV     = r"D:\\About_AERS\\Old_Version\\[LexEval]_V40_Train_Multiseed_Final\\input_data\\dataset_cleaned.csv"   # <- đường dẫn thật trên máy bạn
V40_BUNDLE   = "models/production/v40/seed_{seed}"
PHO_TEST_CSV = "reports/phobert_logits/phobert_test_seed_{seed}.csv"
V40_TEST_CSV = "reports/v40_probs/v40_test_seed_{seed}.csv"
FIG_DIR      = "reports/figures_final"
OUT_DIR      = "reports/final_analysis"
""",
"""# Corpus: mac dinh la ban DA CHE PII di kem repo. Dat bien moi truong AERS_CORPUS de dung ban khac.
# (Ban goc chua che chi can khi chay `export-v40`, vi du doan phu thuoc van ban.)
DATA_CSV     = os.environ.get("AERS_CORPUS", "data/dataset_cleaned_masked.csv")
V40_BUNDLE   = "models/production/v40/seed_{seed}"
PHO_TEST_CSV = "results/phobert_logits/phobert_test_seed_{seed}.csv"
V40_TEST_CSV = "results/v40_probs/v40_test_seed_{seed}.csv"
FIG_DIR      = "results/figures_final"
OUT_DIR      = "results/final_analysis"

_CORPUS_CACHE = {}


def load_corpus(path=None):
    \"\"\"Doc corpus 1 lan (cache). Van ban/nhan lay tu day theo row_index.\"\"\"
    path = path or DATA_CSV
    if path not in _CORPUS_CACHE:
        if not os.path.exists(path):
            sys.exit(f"[DỪNG] Không thấy corpus: {path}")
        c = pd.read_csv(path, encoding="utf-8-sig")
        print(f"  [corpus] {path}  ({len(c):,} dòng)" +
              ("  PII-masked" if "masked" in os.path.basename(path) else "  BẢN GỐC (không công bố)"))
        _CORPUS_CACHE[path] = c
    return _CORPUS_CACHE[path]
"""),
# load_per_sample: row_index + texts tu corpus
("""    c_txt = _find_col(df, "text", TEXT_COL, "review", "content")
    texts = df[c_txt].astype(str).to_numpy() if c_txt else None

    return dict(y_true=y_true, y_pred=y_pred, proba=proba, texts=texts, n=len(df))
""",
"""    c_txt = _find_col(df, "text", TEXT_COL, "review", "content")
    c_ri = _find_col(df, "row_index")
    row_index = df[c_ri].to_numpy(int) if c_ri else None
    if c_txt:                                   # file cũ còn cột văn bản
        texts = df[c_txt].astype(str).to_numpy()
    elif row_index is not None:                 # file mới: lấy văn bản từ corpus theo row_index
        corp = load_corpus()
        if row_index.max() >= len(corp) or row_index.min() < 0:
            sys.exit(f"[DỪNG] {path}: row_index ngoài phạm vi corpus ({len(corp):,} dòng).")
        # Chốt an toàn: nhãn trong corpus tại row_index phải trùng y_true của file.
        lab = corp["sentiment"].to_numpy()[row_index]
        if not np.array_equal(lab.astype(str), y_true.astype(str)):
            sys.exit(f"[DỪNG] {path}: nhãn corpus tại row_index KHÁC y_true "
                     f"({int((lab.astype(str) != y_true.astype(str)).sum())} dòng) → sai corpus hoặc sai thứ tự.")
        texts = corp[TEXT_COL].astype(str).to_numpy()[row_index]
    else:
        texts = None

    return dict(y_true=y_true, y_pred=y_pred, proba=proba, texts=texts, n=len(df), row_index=row_index)
"""),
# build_table5: kiem tra cung row_index
("""        if not np.array_equal(dv["y_true"], dp["y_true"]):
            sys.exit(f"[DỪNG] seed {seed}: y_true của hai mô hình khác nhau → \"""",
"""        if dv.get("row_index") is not None and dp.get("row_index") is not None \\
                and not np.array_equal(dv["row_index"], dp["row_index"]):
            sys.exit(f"[DỪNG] seed {seed}: row_index của hai mô hình khác nhau → "
                     "không cùng tập/thứ tự dòng test, không ghép cặp được.")
        if not np.array_equal(dv["y_true"], dp["y_true"]):
            sys.exit(f"[DỪNG] seed {seed}: y_true của hai mô hình khác nhau → \""""),
("""            sys.exit("[DỪNG] Export không có cột text → không gán được phân nhóm. "
                     "Thêm cột 'text' vào file export.")""",
"""            sys.exit("[DỪNG] Export không có cột text lẫn row_index → không gán được phân nhóm. "
                     "Thêm cột 'row_index' vào file export.")"""),
# export-v40: doc van ban tu corpus theo row_index cua file PhoBERT, ghi row_index, bo text
("""        c_text = _find_col(pho_df, "text", TEXT_COL, "comment", "review", "content")
        c_true = _find_col(pho_df, "y_true", "label", "true_label", "y")
        if c_text is None or c_true is None:
            sys.exit(f"[DỪNG] {pho_path}: không tìm được cột text/nhãn thật. "
                      f"Các cột hiện có: {list(pho_df.columns)}")

        te_texts = pho_df[c_text].astype(str).tolist()
""",
"""        c_text = _find_col(pho_df, "text", TEXT_COL, "comment", "review", "content")
        c_ri = _find_col(pho_df, "row_index")
        c_true = _find_col(pho_df, "y_true", "label", "true_label", "y")
        if c_true is None or (c_text is None and c_ri is None):
            sys.exit(f"[DỪNG] {pho_path}: không tìm được cột text hoặc row_index / nhãn thật. "
                      f"Các cột hiện có: {list(pho_df.columns)}")

        if c_text is not None:
            te_texts = pho_df[c_text].astype(str).tolist()
            ridx = pho_df[c_ri].to_numpy(int) if c_ri else None
        else:
            # Dự đoán phụ thuộc văn bản → cần corpus GỐC (chưa che) để tái tạo đúng số đã log.
            if "masked" in os.path.basename(DATA_CSV):
                print("  [CẢNH BÁO] Đang dự đoán trên corpus ĐÃ CHE PII: 71/25.289 dòng khác bản gốc, "
                      "kết quả có thể lệch rất nhẹ so với log. Dùng AERS_CORPUS=<bản gốc> để tái tạo chính xác.")
            ridx = pho_df[c_ri].to_numpy(int)
            te_texts = df_all[TEXT_COL].astype(str).to_numpy()[ridx].tolist()
"""),
("""        out = pd.DataFrame({
            "y_true": y_true, "y_pred": y_pred,
            "p_negative": proba[:, 0], "p_neutral": proba[:, 1],
            "p_positive": proba[:, 2], "text": te_texts,
        })
""",
"""        out = pd.DataFrame({
            "y_true": y_true, "y_pred": y_pred,
            "p_negative": proba[:, 0], "p_neutral": proba[:, 1],
            "p_positive": proba[:, 2],
        })
        if ridx is not None:
            out.insert(0, "row_index", ridx)      # KHÔNG ghi văn bản review vào file công bố
        else:
            out["text"] = te_texts                # chỉ khi PhoBERT export cũ không có row_index
"""),
("""    df_all = pd.read_csv(DATA_CSV)

    le = LabelEncoder()""",
"""    df_all = load_corpus()

    le = LabelEncoder()"""),
# fidelity: tham chieu = JSON ket qua dung trong bai (LOGGED cu la log 2026-07, khong con dung)
("""def check_fidelity(kind, seed, d):""",
"""REF_JSON = {"v40": "results/results_v40_multiseed.json",
            "phobert": "results/results_phobert_5ep.json"}   # PhoBERT 5 epoch = ban chinh trong so sanh cuoi (8 epoch chi la do nhay)


def _reference_metrics(kind, seed):
    \"\"\"(F1, accuracy) tham chieu cho check-fidelity: lay tu JSON ket qua dang dung trong bai.
    Bang LOGGED o tren la so cua log cu (truoc khi train lai) nen KHONG dung lam chuan nua.\"\"\"
    path = REF_JSON[kind]
    if os.path.exists(path):
        m = json.load(open(path))["metrics_by_seed"][str(seed)]
        return m["f1"], m["accuracy"]
    return LOGGED[kind][seed]


def check_fidelity(kind, seed, d):"""),
("""    lf1, lacc = LOGGED[kind][seed]
    ok = abs(f1 - lf1)""",
"""    lf1, lacc = _reference_metrics(kind, seed)
    ok = abs(f1 - lf1)"""),
("""        lf1, lacc = LOGGED["v40"][seed]""",
"""        lf1, lacc = _reference_metrics("v40", seed)"""),
]

if __name__ == '__main__':
    patch('scripts/analyze_saved.py', A)
    patch('src/finish_analysis.py', F)
