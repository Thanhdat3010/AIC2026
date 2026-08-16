# 🚀 KẾ HOẠCH PHÁT TRIỂN HỆ THỐNG AIC2026

Dựa trên cấu trúc cuộc thi và luồng dữ liệu, hệ thống được thiết kế tách biệt hoàn toàn thành 2 luồng (Pipeline) chính: **Luồng Xử lý Dữ liệu (Làm Data - Chạy trước khi thi)** và **Luồng Sinh Kết Quả (Chạy Task - Chạy trong lúc thi)**. Việc tách biệt này giúp hệ thống gọn gàng, truy xuất siêu tốc và không bị nhầm lẫn.

---

## 🏗️ LUỒNG 1: XỬ LÝ DỮ LIỆU & ĐÁNH CHỈ MỤC (OFFLINE DATA PIPELINE)
Đây là khâu "nấu chín" toàn bộ dữ liệu thô do BTC cung cấp. Luồng này chạy cật lực trên GPU A100 và chỉ chạy 1 lần duy nhất trước ngày thi.

**1. Trích xuất thông tin đa phương thức:**
- **Văn bản trên hình (OCR) - ĐÃ XONG:** Đọc toàn bộ chữ trên khung hình bằng mô hình CRAFT + VietOCR. Kết quả lưu tại `data/batch_1/processed/ocr_results.parquet`.
- **Lời thoại âm thanh (ASR) - ĐÃ XONG:** Nghe và chép lời toàn bộ video bằng VinAI PhoWhisper. Kết quả lưu tại `data/batch_1/processed/transcripts.parquet`.
- **Đặc trưng hình ảnh (Visual Embeddings) - SẮP LÀM:** Dùng mô hình SigLIP quét qua tất cả keyframes. Kết quả lưu tại `data/batch_1/processed/siglip_features.npy`.
- **Siêu dữ liệu (Metadata):** Đọc các file JSON trong `data/batch_1/processed/videos.parquet`.

> [!NOTE]
> **Cấu trúc Batch:** Toàn bộ dữ liệu hiện tại được gom trong `data/batch_1/`. Khi BTC phát hành đợt 2, hệ thống chỉ việc tạo thêm `data/batch_2/` và chạy tương tự mà không sợ bị ghi đè hay lẫn lộn dữ liệu!

**2. Liên kết dữ liệu (Data Linking & Indexing):**
- Đưa tất cả văn bản (chữ OCR, lời thoại ASR, Metadata) vào một kho dữ liệu chữ siêu tốc (như Elasticsearch).
- Đưa các chuỗi số Vector của hình ảnh vào kho Vector (FAISS).
- **Mục tiêu:** Tạo ra một "cây cầu" liên kết chéo giữa mốc thời gian -> lời nói -> chữ viết -> hình ảnh, sẵn sàng chờ truy vấn.

---

## 🎯 LUỒNG 2: SINH KẾT QUẢ CHO CÁC TASK THI ĐẤU (ONLINE INFERENCE PIPELINE)
Luồng này là "bộ não" thực sự, sẽ chạy trực tiếp trên máy tính lúc thi đấu. Khi có đề bài từ BTC, luồng này sẽ lấy thông tin từ "kho dữ liệu đã nấu chín" ở Luồng 1 để giải 3 bài toán (Task).

### 📝 Task 1: Tìm kiếm Video (Textual KIS - Known-Item Search)
- **Cách hoạt động:** Khi người dùng nhập một câu truy vấn (vd: "một chiếc xe máy ngã trước ngã tư lúc trời mưa").
  1. Hệ thống tìm kiếm sẽ quét trong kho Vector hình ảnh xem hình nào giống miêu tả nhất.
  2. Đồng thời quét trong kho chữ OCR và lời thoại ASR xem có thông tin liên quan không.
  3. Gộp điểm lại để tìm ra cảnh quay chuẩn xác nhất.
- **Đầu ra:** Xuất đúng định dạng `<video_id>,<frame_idx>` để nộp cho BTC.

### 🗣️ Task 2: Hỏi Đáp Video (Video Q&A)
- **Cách hoạt động:** Khi BTC hỏi một câu cần suy luận (vd: "Biển số xe của chiếc ô tô gây tai nạn là bao nhiêu?").
  1. Hệ thống dùng luồng tìm kiếm (Task 1) để lôi ra đoạn video ngắn chứa vụ tai nạn.
  2. Trích xuất toàn bộ Lời thoại + Chữ + Hình ảnh của đoạn đó.
  3. Đưa toàn bộ vào một mô hình Trí tuệ Nhân tạo đa phương thức (VLM - Vision Language Model) để nó đóng vai trò "người xem video" và tự trả lời bằng chữ.
- **Đầu ra:** Một đoạn văn bản (Text) trả lời chính xác câu hỏi.

### 🔗 Task 3: Liên kết Dấu vết Thực thể (TRAKE)
- **Cách hoạt động:** Tracking một đối tượng giao thông (ví dụ: truy tìm một chiếc xe tải cụ thể).
  1. Hệ thống nhận diện đặc điểm phương tiện (màu sắc, loại xe, chữ trên thân xe, biển số).
  2. Truy quét ngược lại toàn bộ kho dữ liệu (nhờ sự liên kết chéo ở Luồng 1) để bốc ra *tất cả* các đoạn video ở các thời điểm khác nhau có dính líu đến phương tiện này.
- **Đầu ra:** Danh sách tập hợp các video và mốc thời gian chứa phương tiện.

---

## 📅 LỘ TRÌNH PHÁT TRIỂN CỤ THỂ

| Giai đoạn | Nhiệm vụ trọng tâm | Thuộc Luồng | Trạng thái |
| :--- | :--- | :--- | :--- |
| **Giai đoạn 1** | Hoàn thiện công cụ trích xuất siêu tốc OCR & ASR | Luồng 1 | ✅ Đã xong |
| **Giai đoạn 2** | Trích xuất đặc trưng hình (CLIP Vector) & Metadata | Luồng 1 | 🚀 Tiếp theo |
| **Giai đoạn 3** | Xây kho liên kết dữ liệu (Indexing chéo các thông tin) | Luồng 1 | ⏳ Chờ |
| **Giai đoạn 4** | Phát triển bộ máy tìm kiếm KIS (Task 1) | Luồng 2 | ⏳ Chờ |
| **Giai đoạn 5** | Tích hợp AI suy luận trả lời câu hỏi (Task 2 & 3) | Luồng 2 | ⏳ Chờ |
| **Giai đoạn 6** | Đóng gói Giao diện tương tác (UI) phục vụ lúc thi đấu | Luồng 2 | ⏳ Chờ |
