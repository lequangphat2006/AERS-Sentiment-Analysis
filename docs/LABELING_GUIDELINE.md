# Hướng dẫn gán nhãn cảm xúc bình luận

- **Phiên bản:** 1.0
- **Bộ dữ liệu:** `dataset_cleaned_masked.csv`
- **Cột dữ liệu:** `comment`, `sentiment`, `near_dup_group_id`
- **Nhãn hợp lệ:** `positive`, `neutral`, `negative`

## 1. Mục tiêu

Hướng dẫn này mô tả cách gán nhãn `sentiment` cho từng bình luận trong bộ dữ liệu, đảm bảo:
- Nhãn phản ánh **thái độ tổng thể** của người viết đối với sản phẩm/dịch vụ/người bán.
- Các annotator áp dụng cùng một quy tắc cho các trường hợp mơ hồ, câu trộn cảm xúc, mỉa mai, spam hoặc nội dung không liên quan.
- `near_dup_group_id` chỉ dùng để nhóm các bình luận gần trùng, **không dùng làm nhãn** và không được máy móc sao chép nhãn giữa các thành viên trong nhóm.

## 2. Định nghĩa nhãn

### `positive`
Người viết thể hiện sự hài lòng, khen ngợi, khuyến khích mua, sẽ mua lại/ủng hộ, hoặc đánh giá tốt về sản phẩm, giao hàng, đóng gói, dịch vụ.

Ví dụ:
- “Chất lượng sản phẩm tuyệt vời Đóng gói sản phẩm rất đẹp và chắc chắn...”
- “sản phẩm đẹp và chắc chắn đóng gói rất cẩn thận. mình rất hài lòng.”
- “tệ lắm, xấu lắm, chậm lắm — nhưng sau khi liên hệ shop thì được giải quyết rất chu đáo” → `positive` vì kết thúc bằng sự hài lòng sau khi được giải quyết.

### `neutral`
Bình luận trung tính, mang tính thông tin, hỏi đáp, mô tả, hoặc không thể hiện rõ thái độ tích cực/tiêu cực. Các cụm như “tạm được”, “bình thường”, “cũng được”, “không có gì phàn nàn”, “ok” khi chỉ là xác nhận nhẹ thường thuộc nhóm này.

Ví dụ:
- “tạm được, không có gì phàn nàn ạ”
- “bình thường, không có gì đặc biệt”
- “chưa dùng nên chưa đánh giá được, hàng nhận đủ, đúng mẫu”

### `negative`
Người viết thể hiện sự không hài lòng, phàn nàn, thất vọng, chê chất lượng, hàng lỗi/giả, giao sai, giao chậm, đóng gói kém, dịch vụ tệ, không đáng tiền, sẽ không mua lại.

Ví dụ:
- “hàng dỏm, về ko dùng dc, thực sự tiền mất toi, ko mua nx”
- “Sản phẩm hỏng k sử dụng đc”
- “6 ngày mới giao được hàng. Quá chậm”

## 3. Quy tắc gán nhãn

1. **Đọc toàn bộ bình luận**, gồm cả emoji, teencode, tiếng Anh xen tiếng Việt và các token đã mask như `[URL]`, `[PHONE]`, `[SOCIAL]`, `[REFCODE]`.
2. **Xác định đối tượng bị đánh giá**: sản phẩm, chất lượng, giao hàng, đóng gói, thái độ shop, giá cả, dịch vụ hậu mãi.
3. **Gán nhãn theo thái độ tổng thể**, không chỉ dựa vào một từ khóa đơn lẻ.
4. **Mỉa mai/châm biếm**: gán theo nghĩa thực mà người viết muốn truyền đạt, không theo nghĩa bề mặt.
   - Ví dụ: “wow, chất lượng xuất sắc — vỡ sau đúng 1 ngày sử dụng” → `negative`.
5. **Bình luận trộn cảm xúc**:
   - Nếu lời chê là chính và vẫn còn thất vọng → `negative`.
   - Nếu sau phàn nàn có giải quyết/hoàn tiền/đổi trả và người viết hài lòng → `positive`.
   - Nếu khen/chê cân bằng, không rõ thái độ → `neutral`.
6. **Hàng lỗi, hàng giả, giao sai mẫu/màu/size, không đúng mô tả**: mặc định `negative`, trừ khi người viết nói rõ đã được xử lý ổn và tỏ ra hài lòng.
7. **Giá rẻ + chất lượng thấp nhưng người viết chấp nhận**:
   - “tiền nào của nấy”, không phàn nàn rõ → `neutral`.
   - “phí tiền”, “không đáng tiền”, “mua về bực mình” → `negative`.
8. **Câu hỏi mang tính phàn nàn**:
   - “Sao đồng hồ lại bị hấp hơi nước bên trong vậy?” → `negative` nếu thể hiện sự khó chịu.
   - Câu hỏi thông tin trung tính → `neutral`.
9. **Quảng cáo, tin nhắn hệ thống, nội dung không liên quan đến sản phẩm/dịch vụ**:
   - Gán `neutral`.
   - Nếu schema có cột flag, đánh dấu `out_of_scope` hoặc `spam`.
10. **Bình luận rất ngắn**:
    - “tốt”, “đẹp”, “ưng” → `positive`.
    - “tệ”, “dở”, “hỏng” → `negative`.
    - “ok”, “bình thường”, “tạm được” → `neutral`, trừ khi có sắc thái rõ như “rất ok”, “quá tốt”.
11. **Không suy diễn ngoài nội dung**. Không dùng thông tin ẩn, ID, hay nhãn của bình luận khác để đoán.

## 4. Quy tắc với `near_dup_group_id`

- `near_dup_group_id` dùng để nhóm các bình luận gần trùng nhau.
- **Không dùng group ID làm nhãn** và không mặc định mọi bình luận trong cùng nhóm có cùng nhãn.
- Nếu hai bình luận gần trùng có nội dung cảm xúc giống nhau, nhãn kỳ vọng giống nhau.
- Nếu nội dung khác nhau về sắc thái, gán nhãn theo từng bình luận.
- Khi QA, ưu tiên kiểm tra các nhóm có nhãn không nhất quán để xem có khác biệt thực sự về nội dung hay không.

## 5. Bảng quyết định nhanh

| Tình huống | Nhãn |
|---|---|
| Khen chất lượng, đóng gói, giao hàng, dịch vụ | `positive` |
| Sẽ mua lại, giới thiệu, rất hài lòng | `positive` |
| Phàn nàn nhưng shop xử lý tốt và người viết hài lòng | `positive` |
| Mô tả, hỏi thông tin, “bình thường”, “tạm được” | `neutral` |
| Quảng cáo, spam, nội dung không liên quan | `neutral` + flag nếu có |
| Hàng lỗi, giả, sai mẫu, không đúng mô tả | `negative` |
| Giao chậm, đóng gói kém, thái độ tệ | `negative` |
| Mỉa mai nhưng ý chê | `negative` |
| Khen chê lẫn lộn, không rõ bên nào chiếm ưu thế | `neutral` |

## 6. Quy trình kiểm soát chất lượng (đã thực hiện)

1. Toàn bộ 25.289 bình luận được gán nhãn thủ công bởi người gán nhãn thứ nhất theo hướng dẫn này (không dùng số sao đánh giá làm nhãn).
2. Kiểm tra test–retest: 300 bình luận ngẫu nhiên được cùng người đó gán lại sau một tuần, Cohen's κ = 0,808.
3. Kiểm tra độ đồng thuận giữa hai người: một người gán nhãn thứ hai, nằm ngoài nhóm tác giả và không tham gia gán nhãn gốc, gán độc lập 500 bình luận (khoảng 2% bộ dữ liệu), lấy mẫu phân tầng theo nhãn gốc (200 neutral, 150 negative, 150 positive), văn bản đã che và không thấy nhãn gốc. Kết quả: đồng thuận thô 86,8%; Cohen's κ = 0,803; κ có trọng số tuyến tính 0,836; κ theo từng lớp lần lượt 0,737 (negative), 0,741 (neutral), 0,943 (positive). Chi tiết trong `t01_annotation/`.
4. Các ca bất đồng tập trung ở ranh giới neutral/negative (48 trong 52 ca bất đồng của lớp neutral). Người gán nhãn thứ nhất xem lại các ca này và sửa 24/500 nhãn trong tập tham chiếu. Việc phân xử này không độc lập; κ được tính theo nhãn gốc và nhãn của bộ dữ liệu không bị thay đổi.
5. Mục tiêu đặt ra: Cohen's κ ≥ 0,75; kết quả thực tế đạt 0,803.

## 7. Nguyên tắc cuối cùng

- Ưu tiên **thái độ tổng thể** hơn từ khóa.
- Ưu tiên **ý định thực** hơn nghĩa bề mặt trong câu mỉa mai.
- Không dùng `near_dup_group_id` để suy ra nhãn.
- Khi không chắc chắn giữa `neutral` và một nhãn cực, chọn `neutral` nếu không có tín hiệu cảm xúc rõ ràng.
- Khi có phàn nàn rõ về lỗi sản phẩm/dịch vụ, mặc định `negative` trừ khi người viết thể hiện rõ sự hài lòng sau xử lý.