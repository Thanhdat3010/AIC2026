# 🏆 HƯỚNG DẪN NỘP BÀI & VẬN HÀNH HỆ THỐNG AIC 2026 (SƠ TUYỂN)

> **Tài liệu hướng dẫn nội bộ:** Dành cho tất cả các thành viên trong đội để nắm rõ quy chế thi của BTC và biết cách sử dụng hệ thống (CLI + Streamlit Console) để xuất file nộp bài chuẩn 100%.

---

## 📌 PHẦN 1: QUY CHUẨN NỘP BÀI CHÍNH THỨC CỦA BTC AIC 2026

### 1.1. Ba Dạng Truy Vấn & Định Dạng File CSV
Mỗi câu truy vấn tương ứng với một file `.csv` thuần túy (không có dòng Header, mã hóa UTF-8, tối đa 100 dòng):

| Tác vụ | Hậu tố File | Định dạng từng dòng | Ví dụ mẫu |
| :--- | :--- | :--- | :--- |
| **Textual KIS** | `*kis.csv` | `<Tên_video>, <Frame_id>` | `L01_V028, 25300`<br>`L00_V055, 5555` |
| **Visual Q&A** | `*qa.csv` | `<Tên_video>, <Frame_id>, "<Answer>"` | `L01_V028, 3450, "5"`<br>`L02_V011, 1200, "Năm người"`<br>`L03_V005, 2800, "Màu đỏ, rất đẹp"` |
| **TRAKE** | `*trake.csv` | `<Tên_video>, <Frame_E1>, <Frame_E2>, ..., <Frame_En>` | `L10_V001, 1200, 1850, 2100, 2450` *(4 events)*<br>`L26_V497, 3009, 3100, 3497, 3844, 3934` *(5 events)* |

### 1.2. ⚠️ Các Quy Định "Sống Còn" Tránh Bị 0 Điểm
1. **Cấu trúc File ZIP:** File `.zip` nộp lên hệ thống **PHẢI** chứa thư mục con `submission/` ở trong:
   ```text
   team_AIC2026.zip
   └── submission/
       ├── query-1-kis.csv
       ├── query-2-kis.csv
       ├── query-3-qa.csv
       └── query-4-trake.csv
   ```
   *(Tuyệt đối KHÔNG nén trực tiếp các file `.csv` ở thư mục gốc của file ZIP).*
2. **Tên Video:** **KHÔNG** có đuôi `.mp4` (`L01_V028` ✅ | `L01_V028.mp4` ❌).
3. **TRAKE Event Count:** Số lượng `frame_id` trên mỗi dòng phải **khớp chính xác tuyệt đối** với số lượng sự kiện $N$ được yêu cầu trong đề bài.
4. **Q&A Answer:** Tối đa 100 ký tự. Nếu câu trả lời có dấu phẩy hoặc ngoặc kép thì bắt buộc phải bọc trong `""` (ví dụ `"Có 3 người, bao gồm nam và nữ"`).
5. **Số lượt nộp:** Tối đa **3 lần / gói câu hỏi**. Kết quả của lần nộp cuối cùng sẽ được tính điểm.

---

## ⚡ PHẦN 2: CÁCH CHẠY HỆ THỐNG ĐỂ TỰ ĐỘNG SINH KẾT QUẢ (CLI)

Hệ thống đã được tích hợp sẵn script tự động hóa hoàn toàn từ khâu đọc đề bài $\rightarrow$ chạy mô hình AI $\rightarrow$ xuất CSV $\rightarrow$ đóng gói ZIP chuẩn Codabench.

### 2.1. Cú pháp Chạy Sinh Kết Quả
Mở terminal (trong môi trường conda `AIC2026`):

```bash
# 1. Chạy gói đề thi với Cấu hình Quán quân SOTA Master tốt nhất (Config 25 - WRRF)
python scripts/submission/run_submission.py --input query/THUNGHIEM-bo-de-thi --output_dir output/thunghiem --config 25

# 2. Hoặc chạy trên bất kỳ thư mục đề thi mới nào từ BTC
python scripts/submission/run_submission.py --input path/to/queries_dir --output_dir output/chinhthuc --config 25
```

### 2.2. Ý Nghĩa Các Tham Số (Arguments):
* `--input`: Đường dẫn thư mục chứa các file `.txt` đề bài (hoặc file `.json`).
* `--output_dir`: Thư mục lưu kết quả (VD: `output/thunghiem`). Hệ thống sẽ tự tạo thư mục con `output/thunghiem/submission/` và file `output/thunghiem/submission.zip` tại đây.
* `--config`: Mã cấu hình muốn chạy (Mặc định: `25` - SOTA Master với Weighted Reciprocal Rank Fusion - WRRF, Segmental DP và Dynamic Gating).
* `--top_k`: Số lượng dòng dự đoán tối đa cho mỗi query (Mặc định: `100`).

---

## 🔍 PHẦN 3: KIỂM TRA ĐỊNH DẠNG TRƯỚC KHI NỘP (VALIDATOR)

Để đảm bảo không bị lỗi parse hay mất lượt nộp oan uổng, hãy chạy công cụ kiểm tra tự động:

```bash
python scripts/submission/validate_submission.py
```
* Script sẽ tự động quét qua toàn bộ các file `.csv` trong thư mục output, kiểm tra cấu trúc từng cột, định dạng số nguyên, độ dài chuỗi Q&A và tính hợp lệ của thư mục nộp bài.

---

## 🖥️ PHẦN 4: HƯỚNG DẪN DÙNG STREAMLIT CONSOLE ĐỂ SOI ẢNH & CHỈNH TAY

Trong các kỳ thi thực chiến, sau khi AI chạy tự động lần 1, các thành viên nên dùng giao diện Web để rà soát lại và nâng điểm từ 7 $\rightarrow$ 10-11+ điểm.

### 4.1. Khởi động Giao diện Web
```bash
python -m streamlit run app/streamlit_app.py
```
👉 Truy cập trình duyệt: **[http://localhost:8501](http://localhost:8501)**

### 4.2. Các Bước Thao Tác Trên Giao Diện:
1. Vào mục: **📂 Đề Thi Chính Thức & Thử Nghiệm BTC** (ở Sidebar bên trái).
2. Chọn gói đề thi tương ứng ở dropdown (Ví dụ: `🧪 Gói Thử Nghiệm: query/THUNGHIEM-bo-de-thi`).
3. Chọn từng câu hỏi trong danh sách để xem:
   * **Soi Top 10 Ảnh Ứng Viên:** Xem ảnh trực quan của 10 dự đoán hàng đầu. Nếu thấy Rank #2 hoặc #3 đúng hơn, chỉ cần bấm **`⭐ Đặt làm Rank #1`**.
   * **Kính Lúp Vi Sai (Dense Video Inspector):** Kéo thanh trượt Slider để chỉnh dịch chuyển vài chục frame bắt đúng khoảnh khắc hành động $\rightarrow$ bấm **`💾 Cập nhật Rank #1`**.
   * **Bộ Chỉnh Sửa QA:** Gõ lại đáp án ngắn gọn (1 - 3 từ) $\rightarrow$ bấm **`💾 Lưu Câu Trả Lời QA`**.
   * **Studio Chuỗi Sự Kiện TRAKE:** Xem song song ảnh của tất cả các sự kiện $E_1, E_2..E_n$, chỉnh frame độc lập và hệ thống tự kiểm tra tính đơn điệu tăng dần thời gian ($E_1 < E_2 < \dots < E_n$) $\rightarrow$ bấm **`💾 Lưu Toàn Bộ Chuỗi TRAKE`**.

> [!IMPORTANT]
> **Cơ chế Auto-Sync Thông Minh:**
> Mọi thao tác chỉnh sửa tay trên Streamlit đều **tự động ghi đè file CSV và tự động đóng gói cập nhật lại file `submission.zip` trên ổ cứng ngay lập tức**! Bạn có thể bấm nút **"📦 Tải Gói Nộp Bài Đầy Đủ"** ở cuối trang để nộp ngay.

---

## 🚨 5 LỖI THƯỜNG GẶP & CÁCH KHẮC PHỤC

| Lỗi | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| 🔴 **Bị 0 điểm toàn gói** | Nén trực tiếp các file `.csv` vào zip mà thiếu thư mục `submission/`. | Dùng trực tiếp file `submission.zip` do hệ thống tự sinh ra (đã chuẩn cấu trúc). |
| 🔴 **TRAKE bị chấm 0 điểm** | Số lượng frame ID không khớp với số events yêu cầu trong đề bài. | Kiểm tra đúng số $N$ events (Dùng `--config 22` hệ thống tự động regex chuẩn). |
| 🔴 **Tên video dính `.mp4`** | Định dạng `L01_V028.mp4` thay vì `L01_V028`. | Hệ thống tự động làm sạch qua hàm `clean_video_name()`. |
| 🔴 **Q&A bị mất điểm do text dài** | Câu trả lời dài dòng hoặc chứa giải thích thừa. | Chỉ trả lời ngắn gọn đúng từ khóa (VD: `"5"`, `"Màu đỏ"`, `"Xã Giang Ly"`). |
| 🔴 **Lệch vài chục frame KIS** | AI chọn đúng video nhưng khung hình trễ 1-2 giây. | Mở Streamlit kéo Slider vi sai bắt đúng khoảnh khắc rồi bấm Lưu. |

---

## ✅ CHECKLIST TRƯỚC KHI BẤM NỘP BÀI LÊN HỆ THỐNG BTC:
- [ ] File nộp là **`.zip`** (không phải file `.rar`, `.7z` hay `.csv` rời).
- [ ] Bên trong file zip có chứa thư mục con **`submission/`**.
- [ ] Tất cả các file trong `submission/` đều có đuôi **`.csv`**, encoding **UTF-8**, không có Header.
- [ ] Tên file CSV khớp chính xác với mã câu truy vấn của BTC (ví dụ `query-p1-1-kis.csv`).
- [ ] Các câu TRAKE có đúng số lượng khung hình theo yêu cầu.
- [ ] Đã chạy qua `validate_submission.py` và nhận thông báo **`[SUCCESS] 100% HỢP LỆ`**.
- [ ] Đã kiểm tra số lần nộp còn lại của gói câu hỏi (tối đa 3 lần).
