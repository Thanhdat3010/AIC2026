# AIC2026 KIS Search Engine

Dự án Hệ thống Tìm kiếm KIS (Textual Known Item Search) cho cuộc thi AIC 2026.

## Các tính năng
- Vector Search với FAISS
- Multilingual Text Embedding với CLIP và Marian MT
- Reranking với TF-IDF trên Metadata
- UI tìm kiếm trực tiếp bằng Streamlit

## Hướng dẫn cài đặt
1. Tạo môi trường: `conda create -n AIC2026 python=3.11`
2. Cài đặt thư viện: `pip install -r requirements.txt`
3. Chạy giao diện UI: `conda run -n AIC2026 streamlit run app/streamlit_app.py`
