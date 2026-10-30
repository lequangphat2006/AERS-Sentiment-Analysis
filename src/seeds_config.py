# ═══════════════════════════════════════════════════════════════════════════
# SEED DÙNG CHUNG CHO V40 VÀ V41
# ───────────────────────────────────────────────────────────────────────────
# QUAN TRỌNG: đây là file DUY NHẤT định nghĩa danh sách seed. Cả train_v40.py
# và train_v41.py đều import từ đây (KHÔNG tự khai seed riêng trong mỗi file)
# để đảm bảo hai hệ thống luôn được đánh giá trên ĐÚNG 10 seed giống hệt nhau
# — điều kiện bắt buộc để paired t-test giữa V40 và V41 có ý nghĩa (mỗi cặp
# điểm F1 phải đến từ cùng một lần chia dữ liệu train/cal/test).
#
# 10 giá trị dưới đây được chọn cố định một lần (không phải random mỗi lần
# chạy) — nếu random lại mỗi lần chạy thì không thể tái lập kết quả cho
# reviewer. Gồm 42 (giữ lại từ lần chạy gốc để so sánh ngược) + 9 giá trị
# khác nhau, rải đều, không theo một pattern số học dễ đoán.
# ═══════════════════════════════════════════════════════════════════════════

SEEDS = [42, 7, 123, 2024, 99, 555, 2026, 31337, 8080, 12321]

assert len(SEEDS) == 10 and len(set(SEEDS)) == 10, "SEEDS phải có đúng 10 giá trị, không trùng lặp."
