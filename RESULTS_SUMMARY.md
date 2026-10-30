# Tổng hợp kết quả AERS-Pro so với PhoBERT (bản lưu bằng chứng, 30/10/2026)

Bài báo: *A Hybrid Machine Learning Framework for Vietnamese Sentiment Analysis in E-Commerce* (AERS-Pro, QNUJS-B2662). Mọi số liệu dưới đây được tính lại từ file gốc (logits, xác suất, JSON), 10 seed, cùng tập test cho hai hệ thống, so sánh theo cặp bằng kiểm định t Nadeau–Bengio (hệ số phương sai 1/J + n_test/n_train ≈ 0,43).

## 1. Kết luận

1. PhoBERT đã tune (lr 3e-5, tối đa 8 epoch) cao hơn AERS-Pro 0,0169 F1 macro (0,8834 so với 0,8665), CI 95% N–B [−0,0337; −0,0002], p = 0,048, AERS-Pro thắng 1/10 seed. p này chưa hiệu chỉnh đa so sánh: nếu áp dụng Holm trên 4 chỉ số chính (F1, Accuracy, ECE, Brier) thì p của F1 khoảng 0,096 (tính tay từ các p ở mục 3, cần xác nhận lại bằng `scripts/stats.py`). Không thể viết "tương đương PhoBERT" (TOST không đạt ở ±0,01, ±0,015, ±0,02, ±0,03).
2. Lớp neutral ngang nhau (0,8426 so với 0,8415, p = 0,94); chênh lệch nằm ở negative (−0,0265) và positive (−0,0255).
3. AERS-Pro hiệu chỉnh tốt hơn PhoBERT thô (ECE 0,0241 so với 0,1069). Khi PhoBERT được hiệu chỉnh bằng TS hoặc Platt thì chênh ECE và Brier không có ý nghĩa thống kê.
4. Không nhóm con nào (phủ định, hedge, trái chiều, câu ngắn) có khác biệt có ý nghĩa; kết luận không đổi khi loại near-duplicate.
5. Ablation: stacking cao hơn LR đã tune khoảng 0,013 F1 (9/10 seed, p N–B = 0,12); lexicon, POS và class weight không đóng góp đo được (thay đổi dưới 0,002, p > 0,7).
6. Phần "nhẹ" chỉ còn đúng ở dung lượng (khoảng 69 lần nhỏ hơn), không cần GPU, và thông lượng khi xử lý theo lô. AERS-Pro không nhanh hơn khi huấn luyện và chậm hơn khi suy luận từng câu.

## 2. Thiết lập

| Hạng mục | Giá trị |
| --- | --- |
| Dữ liệu | 25.289 dòng, 3 lớp |
| Chia | StratifiedGroupKFold 5-fold rồi 4-fold theo `near_dup_group_id`, 10 seed (42, 7, 123, 2024, 99, 555, 2026, 31337, 8080, 12321) |
| Cỡ tập test | 4.877–5.315 mẫu (seed 42: train 14.849, calibration 5.125, test 5.315) |
| scikit-learn | 1.6.1 (bắt buộc; 1.9.1 cho phân hoạch khác: 15.173/5.058/5.058; xem `evidence/split_audit_sklearn_*.csv`) |
| AERS-Pro | TF-IDF + n-gram ký tự băm + lexicon + POS + KMeans(k=4), stacking LR/LinearSVC/LightGBM, meta-LR, Platt 3×3 có điều kiện; Optuna 30 trial, 5 siêu tham số |
| PhoBERT | vinai/phobert-base-v2, tách từ underthesea, max_len 192, batch 32×2 GPU T4, fp16, lr 3e-5, tối đa 8 epoch, dừng sớm (patience 2), chọn checkpoint theo F1 trên tập calibration |
| Tỉ lệ bị cắt max_len | 0,35% (tối đa 0,67%) |

## 3. So sánh chính, PhoBERT 8 epoch (số chính) và 5 epoch (độ nhạy)

| Chỉ số (AERS − PhoBERT) | 8 epoch: chênh [CI 95% N–B], p | 5 epoch: chênh [CI 95% N–B], p |
| --- | --- | --- |
| F1 macro | −0,0169 [−0,0337; −0,0002], 0,048 | −0,0140 [−0,0302; +0,0021], 0,081 |
| Accuracy | −0,0193 [−0,0343; −0,0042], 0,018 | −0,0162 [−0,0305; −0,0019], 0,031 |
| ECE (PhoBERT thô) | −0,0827 [−0,1073; −0,0582], <0,001 | −0,0847 [−0,1059; −0,0634], <0,001 |
| Brier (PhoBERT thô) | −0,0231 [−0,0561; +0,0099], 0,147 | −0,0270 [−0,0508; −0,0031], 0,031 |
| F1 negative | −0,0265 [−0,0489; −0,0040], 0,026 | −0,0214 [−0,0429; +0,0001], 0,051 |
| F1 neutral | +0,0011 [−0,0338; +0,0361], 0,943 | +0,0013 [−0,0328; +0,0353], 0,935 |
| F1 positive | −0,0255 [−0,0440; −0,0070], 0,012 | −0,0220 [−0,0375; −0,0065], 0,011 |
| ECE so với PhoBERT + TS | −0,0073 [−0,0366; +0,0221], 0,589 | −0,0169 [−0,0457; +0,0119], 0,216 |
| ECE so với PhoBERT + Platt | +0,0017 [−0,0125; +0,0158], 0,797 | −0,0046 [−0,0268; +0,0176], 0,652 |

F1 PhoBERT: 5 epoch 0,8805, 8 epoch 0,8834 (+0,0029; 8 epoch cao hơn ở 6/10 seed). AERS-Pro thắng 0/10 seed (5 epoch) và 1/10 seed (8 epoch); Wilcoxon p = 0,002 và 0,004 (chỉ để tham khảo). TOST (p của hướng khó hơn), 8 epoch: ±0,01 → 0,814; ±0,015 → 0,601; ±0,02 → 0,345; ±0,03 → 0,056. 5 epoch: ±0,005 → 0,881; ±0,01 → 0,707; ±0,015 → 0,448; ±0,02 → 0,213; ±0,03 → 0,026.

F1 từng seed (AERS-Pro lần gốc; PhoBERT 8 epoch; PhoBERT 5 epoch): 42: 0,8680; 0,8922; 0,8866 · 7: 0,8719; 0,8922; 0,8814 · 123: 0,8543; 0,8877; 0,8795 · 2024: 0,8528; 0,8638; 0,8578 · 99: 0,8689; 0,8786; 0,8869 · 555: 0,8596; 0,8650; 0,8612 · 2026: 0,8743; 0,8957; 0,8810 · 31337: 0,8772; 0,8746; 0,8799 · 8080: 0,8704; 0,9005; 0,9056 · 12321: 0,8673; 0,8838; 0,8851.

## 4. Hiệu chỉnh xác suất (trung bình 10 seed)

| Hệ thống | F1 | ECE | Brier |
| --- | --- | --- | --- |
| AERS-Pro lần chạy gốc (công bố) | 0,8665 | 0,0241 | 0,1946 |
| AERS-Pro stacking raw (môi trường ablation) | 0,8622 | 0,0262 | 0,1984 |
| AERS-Pro + TS (môi trường ablation) | 0,8622 | 0,0246 | 0,1986 |
| AERS-Pro + Platt 3×3 luôn bật (môi trường ablation) | 0,8645 | 0,0247 | 0,1955 |
| AERS-Pro quy tắc có điều kiện (môi trường ablation) | 0,8628 | 0,0249 | 0,1980 |
| PhoBERT 8 epoch raw | 0,8834 | 0,1069 | 0,2177 |
| PhoBERT 8 epoch + TS | 0,8834 | 0,0314 | 0,1940 |
| PhoBERT 8 epoch + Platt 3×3 | 0,8834 | 0,0225 | 0,1919 |
| PhoBERT 5 epoch raw / + TS / + Platt | 0,8805 / 0,8805 / 0,8813 | 0,1088 / 0,0411 / 0,0287 | 0,2216 / 0,1958 / 0,1925 |

Nhiệt độ T (PhoBERT 5 epoch) trung bình 3,28 (2,21–4,47); độ tin cậy trung bình 0,994. Bảng bin ECE 15 và 20 nằm trong `tables/per_seed_all_metrics_*.csv`.

## 5. Theo lớp và nhóm con (trung bình theo seed, PhoBERT 8 epoch)

| Lớp | P / R / F1 AERS-Pro | P / R / F1 PhoBERT |
| --- | --- | --- |
| negative | 0,8391 / 0,8680 / 0,8529 | 0,8518 / 0,9096 / 0,8794 |
| neutral | 0,8711 / 0,8170 / 0,8426 | 0,8899 / 0,8012 / 0,8415 |
| positive | 0,9012 / 0,9069 / 0,9038 | 0,9311 / 0,9279 / 0,9293 |

| Nhóm con | n trung bình | F1 AERS-Pro | F1 PhoBERT | Chênh | AERS thắng | p N–B |
| --- | --- | --- | --- | --- | --- | --- |
| Phủ định | 2.417 | 0,8361 | 0,8530 | −0,0169 | 1/10 | 0,095 |
| Hedge | 312 | 0,8434 | 0,8679 | −0,0245 | 2/10 | 0,711 |
| Trái chiều | 1.912 | 0,8132 | 0,8349 | −0,0217 | 2/10 | 0,250 |
| Câu ngắn (<5 token) | 127 | 0,8969 | 0,8929 | +0,0040 | 6/10 | 0,919 |

Định nghĩa nhóm giữ nguyên bản gốc (`train_phobert.py::_tag_phenomena`): phủ định = token chính xác thuộc {không, chẳng, chả, đâu, chưa, k, ko, chớ, hok, hem}; hedge = chứa chuỗi {tạm, tạm được, cũng được, bình thường, cũng ổn, tàm tạm, không đến nỗi, cũng tạm}; trái chiều = chứa chuỗi {nhưng, mà, tuy nhiên, ai ngờ, vậy mà, thế mà}; câu ngắn = dưới 5 token.

Near-duplicate (proxy TF-IDF từ, không phải không gian đặc trưng đầy đủ), 8 epoch: chênh F1 −0,0170 (loại 0,4% ở ngưỡng cosine 0,99), −0,0177 (4,4% ở 0,95), −0,0191 (14,4% ở 0,90); ECE AERS-Pro 0,0241–0,0287, PhoBERT 0,1071–0,1223.

Ma trận nhầm lẫn AERS-Pro seed 42 (hàng = nhãn thật; negative, neutral, positive): [[1803, 96, 147], [144, 1147, 111], [127, 64, 1676]]. Lỗi lớn nhất là negative → positive (147).

## 6. Tune PhoBERT

Sweep lr (F1 macro trên tập calibration, trung bình 3 seed 42, 7, 123, 5 epoch):

| lr | ep1 | ep2 | ep3 | ep4 | ep5 |
| --- | --- | --- | --- | --- | --- |
| 1e-5 | 0,7774 | 0,8253 | 0,8520 | 0,8481 | 0,8561 |
| 2e-5 | 0,8102 | 0,8527 | 0,8551 | 0,8688 | 0,8756 |
| **3e-5** | 0,8339 | 0,8607 | 0,8718 | 0,8749 | **0,8759** |
| 5e-5 | 0,8332 | 0,8596 | 0,8662 | 0,8690 | 0,8749 |

Epoch được chọn: lần 5 epoch: 5 ở 5 seed, 4 ở 3 seed, 3 ở 2 seed. Lần 8 epoch (dừng sớm): 3 (seed 555, 8080), 4 (42, 99), 5 (7, 2026, 12321), 6 (31337), 8 (123, 2024). Đường cong F1 calibration theo epoch từng seed nằm trong log gốc (dòng `CAL-CURVE`).

## 7. Ablation 10 seed (cùng môi trường; đối chiếu với `full`, F1 0,8628)

| So sánh | Chênh F1 | CI 95% N–B | p N–B | Thắng/10 |
| --- | --- | --- | --- | --- |
| Stacking so với LR trên đặc trưng gộp | +0,0127 | [−0,0042; +0,0296] | 0,122 | 9 |
| Stacking so với trung bình 3 mô hình cơ sở | +0,0087 | [−0,0082; +0,0256] | 0,273 | 8 |
| Bỏ lexicon | −0,0008 | [−0,0091; +0,0075] | 0,831 | 3 |
| Bỏ POS | +0,0004 | [−0,0072; +0,0081] | 0,898 | 6 |
| Bỏ cả hai | +0,0002 | [−0,0093; +0,0096] | 0,966 | 6 |
| Bỏ class weight | +0,0017 | [−0,0080; +0,0114] | 0,706 | 6 |
| Platt luôn bật so với raw | +0,0023 | [−0,0046; +0,0092] | 0,467 | 7 |
| Quy tắc có điều kiện so với raw | +0,0006 | [−0,0012; +0,0024] | 0,453 | 3 |

(Holm = 1,00 cho mọi dòng.) Stacking giúp rõ nhất ở lớp neutral: F1 0,8350 so với 0,8128 (LR) và 0,8145 (trung bình 3). F1 neutral của PhoBERT: 0,8414.

## 8. Train lại V40 10 seed (kiểm tra tái lập) và T10

Môi trường: Kaggle CPU, Intel Xeon 2,20 GHz, 4 vCPU, 32 GB RAM, scikit-learn 1.6.1, 30 trial Optuna mỗi seed.

| | Lần gốc | Train lại |
| --- | --- | --- |
| F1 trung bình (10 seed) | 0,8665 | 0,8625 |
| Chênh theo seed | | từ −0,0189 (12321) đến +0,0114 (99); độ lệch chuẩn 0,0087 |
| ECE trung bình | 0,0241 | 0,0242 |
| Số seed giữ bộ hiệu chỉnh Platt | 4 | 6 |
| `best_params` Optuna khác lần gốc | | 10/10 seed |
| Thời gian huấn luyện trung bình / seed | | 2.082 giây (độ lệch chuẩn 75) |
| Kích thước mô hình | | 7,44 MB (5,95–8,27) |

Kích thước PhoBERT 515–517 MB (135 triệu tham số, fp32); tỉ lệ khoảng 69 lần (62–87 lần theo seed). PhoBERT huấn luyện 1.658 giây trung bình trên 2 GPU T4 (5 epoch).

Độ trễ suy luận trên cùng CPU (seed 42; AERS-Pro từ văn bản thô; PhoBERT không gồm bước tách từ, đầu phân loại khởi tạo ngẫu nhiên):

| | AERS-Pro | PhoBERT |
| --- | --- | --- |
| Một câu, trung vị / p95 | 171 / 511 ms | 87 / 194 ms |
| Lô 32, mỗi câu | 24,4 ms | 280 ms |
| Lô 256, mỗi câu | 5,5 ms | chưa đo |
| Lô 1000, mỗi câu | 3,9 ms (259 câu/giây) | chưa đo |

## 9. Kiểm tra tái lập đã làm

- F1 tính lại từ logits PhoBERT và xác suất V40 khớp JSON (sai khác tối đa 0).
- Tập test 10 seed của hai hệ thống trùng nhau; 10/10 seed tái lập đúng với scikit-learn 1.6.1.
- Che thông tin cá nhân bằng `scripts/mask_pii.py` (bản v2) trên 25.289 dòng: 19 số điện thoại, 53 URL, 0 email, 15 mục tài khoản mạng xã hội; quét lại sau khi che còn 0 số điện thoại. File phát hành là `data/dataset_cleaned_masked.csv`; file thô `dataset_cleaned.csv` không được phát hành. Mã che chỉ xử lý số điện thoại, URL, email và tài khoản mạng xã hội; tên riêng và địa chỉ chưa được rà.

## 10. Giới hạn cần nêu trong bài

- Sweep lr chỉ trên 3 seed, 4 mức lr; PhoBERT dùng underthesea để tách từ (chưa kiểm tra RDRSegmenter).
- Near-duplicate được đánh giá trong không gian proxy; chia dữ liệu không phân tầng tuyệt đối.
- Ablation chạy ở môi trường khác lần chạy chính (underthesea 9.5.0), `full` thấp hơn 0,0037.
- Tối ưu Optuna không tái lập giữa hai môi trường (best_params khác ở 10/10 seed).
- Release note scikit-learn 1.8 đã được đọc: mục `model_selection` chỉ ghi một sửa lỗi cho `StratifiedGroupKFold`, rằng phân tầng giữa các fold cũng được giữ khi `shuffle=True` (#32540); không có thay đổi nào về việc cân bằng kích thước fold. Audit chia dữ liệu cho thấy dưới 1.6.1 tỉ lệ lớp lệch tuyệt đối tối đa khoảng 0,037 (calibration) và 0,024 (test) so với tỉ lệ toàn corpus, còn dưới 1.9.1 lệch không quá 0,0002. Chưa huấn luyện lại hai hệ thống trên phân hoạch 1.9.1 (kiểm tra độ nhạy).
- Số đo độ trễ chỉ trên một seed và một loại CPU; PhoBERT đo bằng đầu phân loại ngẫu nhiên và chưa gồm tách từ.

## 11. Việc còn lại

- Gán nhãn độc lập lần hai và tính κ (`scripts/make_label_sample.py`, `scripts/compute_kappa.py`); hạn nhận nhãn 1/11/2026.
- Thông tin LLM rà soát nhãn (tên, phiên bản, prompt, số nhãn bị đổi) và số liệu hai giai đoạn thu thập.
- Tách từ cho PhoBERT bằng RDRSegmenter, hoặc nêu rõ dùng underthesea và thu hẹp kết luận; đoạn biện minh chọn PhoBERT thay cho ViSoBERT.
- Độ nhạy khi tái sử dụng tập calibration (chia đôi tập calibration); từng base learner riêng lẻ; script và file theo từng seed cho B2/B3/B4.
- Funding, điều khoản Shopee, Luật Bảo vệ dữ liệu cá nhân (tự xác minh số hiệu văn bản trước khi trích dẫn).
- Phát hành bất biến trên repo đã làm sạch lịch sử git (không còn `dataset_cleaned.csv`), ghi DOI mới.
- Chốt cách định vị bài, viết lại bản thảo và thư phản hồi reviewer; nộp trước 12/11 (hạn 13/11/2026).

---
## Cập nhật (T06): baseline đơn giản B2, B3, B4

Các baseline dùng cùng phân chia dữ liệu và scikit-learn==1.6.1. Giá trị Δ tính so với AERS-Pro lần chạy gốc (F1 = 0,8665).

| Mô hình | F1 macro (trung bình ± SD, 10 seed) | Δ so với AERS-Pro |
| --- | --- | --- |
| AERS-Pro (stacking) | 0,8665 ± 0,0094 | |
| B2: LinearSVC trên TF-IDF + hash ký tự | 0,8454 ± 0,0128 | −0,0211 |
| B3: chỉ TF-IDF từ | 0,8252 ± 0,0076 | −0,0413 |
| B4: chỉ hash ký tự | 0,7972 ± 0,0153 | −0,0693 |

Kết quả cho thấy việc kết hợp TF-IDF từ và hash ký tự tốt hơn từng khối riêng lẻ. Ablation ở mục 7 cho thấy khối lexicon và POS không đóng góp đo được (dưới 0,002), và lợi thế của stacking so với LR đã tune (+0,013) chưa có ý nghĩa sau hiệu chỉnh Nadeau–Bengio (p = 0,12). Số liệu thô: `tables/baseline_b3_b4_results.csv`. Lưu ý: B2/B3/B4 chưa có script chạy và file theo từng seed trong repo; cần bổ sung trước khi dùng trong bài.
