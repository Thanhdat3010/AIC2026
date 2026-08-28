import sys
import os
import time
import json
import io
import re
import zipfile
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.unified_search_core import UnifiedSearchCore
from src.query.llm_query_refiner import LLMQueryRefiner
from src.tasks.clean_task_handlers import KISHandler, QAHandler, TRAKEHandler
from src.retrieval.keyframe_loader import KeyframeZipLoader
from src.retrieval.video_player_manager import VideoPlayerManager
from src.submission.submission_validator import SubmissionValidator

st.set_page_config(
    page_title="AIC 2026 SOTA Multimodal Search & Championship Console",
    layout="wide",
    page_icon="🏆",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        color: white;
    }
    .card-r1 {
        background: #1e293b;
        border: 2px solid #eab308 !important;
        box-shadow: 0 0 16px rgba(234, 179, 8, 0.45) !important;
        border-radius: 10px;
        padding: 8px;
        margin-bottom: 12px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card-active {
        background: #1e293b;
        border: 2px solid #a855f7 !important;
        box-shadow: 0 0 16px rgba(168, 85, 247, 0.5) !important;
        border-radius: 10px;
        padding: 8px;
        margin-bottom: 12px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card-top5 {
        background: #1e293b;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px;
        padding: 8px;
        margin-bottom: 12px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card-normal {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px;
        margin-bottom: 12px;
        transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    }
    .card-r1:hover, .card-active:hover, .card-top5:hover, .card-normal:hover {
        transform: translateY(-2px);
    }
    .glass-badge-r1 {
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.95), rgba(202, 138, 4, 0.95));
        color: #0f172a;
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 800;
        font-size: 0.78rem;
    }
    .glass-badge-top5 {
        background: rgba(59, 130, 246, 0.9);
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.75rem;
    }
    .glass-badge-normal {
        background: rgba(71, 85, 105, 0.85);
        color: #f1f5f9;
        padding: 2px 5px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.75rem;
    }
    .glass-time {
        background: rgba(15, 23, 42, 0.85);
        color: #38bdf8;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-family: monospace;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .studio-panel {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        position: sticky;
        top: 1rem;
    }
    .event-card {
        background: #0f172a;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_search_core():
    return UnifiedSearchCore(engine="siglip2", batch="batch_1")

@st.cache_resource
def get_llm_refiner():
    return LLMQueryRefiner()

@st.cache_resource
def get_keyframe_loader():
    return KeyframeZipLoader()

@st.cache_resource
def get_video_manager():
    return VideoPlayerManager()

@st.cache_resource
def get_validator():
    return SubmissionValidator()

search_core = get_search_core()
llm_refiner = get_llm_refiner()
kis_handler = KISHandler(search_core, llm_refiner)
qa_handler = QAHandler(search_core, llm_refiner)
trake_handler = TRAKEHandler(search_core, llm_refiner)

keyframe_loader = get_keyframe_loader()
video_manager = get_video_manager()
validator = get_validator()

def sync_submission_zip(csv_dir: Path, zip_dest: Path):
    """Tự động đóng gói và đồng bộ file submission.zip chuẩn 100% BTC."""
    if not csv_dir.exists():
        return
    zip_dest.parent.mkdir(parents=True, exist_ok=True)
    temp_zip = zip_dest.parent / f"{zip_dest.stem}_temp.zip"
    try:
        with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for csv_f in sorted(list(csv_dir.glob("*.csv"))):
                z.write(csv_f, arcname=f"submission/{csv_f.name}")
        if temp_zip.exists():
            if zip_dest.exists():
                zip_dest.unlink(missing_ok=True)
            temp_zip.rename(zip_dest)
    except Exception as e:
        print(f"⚠️ Lỗi đồng bộ zip: {e}", flush=True)
def save_undo_state(query_name: str, current_rows: list):
    """Lưu snapshot danh sách rows vào session_state để hỗ trợ nút Hoàn Tác (Undo)."""
    if "undo_history" not in st.session_state:
        st.session_state["undo_history"] = {}
    if query_name not in st.session_state["undo_history"]:
        st.session_state["undo_history"][query_name] = []
    st.session_state["undo_history"][query_name].append([r.copy() for r in current_rows])
    # Giữ tối đa 10 bước lịch sử gần nhất
    st.session_state["undo_history"][query_name] = st.session_state["undo_history"][query_name][-10:]

def parse_trake_subevents(query_text: str) -> list[str]:
    """Bóc tách các sự kiện con E1, E2, ... từ văn bản truy vấn tiếng Việt chuẩn xác 100%."""
    lines = [l.strip() for l in query_text.split("\n") if l.strip()]
    events = []
    
    # 1. Tìm các dòng có tiền tố sự kiện: E1, E2, Sự kiện 1, Event 1, 1., 1)
    for l in lines:
        if re.search(r'^(?:sự kiện|event|bước|e)\s*\d+[\s:.-]*', l, re.IGNORECASE) or re.search(r'^\d+[\.\)]\s*', l):
            cleaned = re.sub(r'^(?:sự kiện|event|bước|e)\s*\d+[\s:.-]*\s*', '', l, flags=re.IGNORECASE)
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned)
            if cleaned:
                events.append(cleaned)

    # 2. Nếu không có tiền tố ở đầu dòng, thử bóc tách bằng regex bên trong văn bản
    if not events:
        inline_matches = re.findall(r'(?:[eE]\d+|sự kiện\s*\d+|event\s*\d+)[:\s.-]+([^;\n\.]+(?:[\.\?!](?![eE]\d+|sự kiện|event))*)', query_text, flags=re.IGNORECASE)
        if inline_matches:
            events = [m.strip() for m in inline_matches if len(m.strip()) > 5]

    # 3. Fallback cuối cùng nếu không có bất kỳ ký hiệu E nào
    if not events:
        parts = [p.strip() for p in re.split(r'(?:;\s*|\n|(?<=[\.\?!])\s+)', query_text) if len(p.strip()) > 8]
        if len(parts) >= 2:
            events = parts
        else:
            events = ["Sự kiện 1", "Sự kiện 2", "Sự kiện 3"]
            
    return events

# Sidebar
with st.sidebar:
    st.image("https://img.shields.io/badge/AIC_2026-CHAMPIONSHIP_CONSOLE-gold?style=for-the-badge&logo=google", use_container_width=True)
    st.header("⚙️ Trung tâm Điều khiển")
    st.caption("Engine: **Google SigLIP-2 (1152d)** + **Gemini 3.5 Flash Lite** + **On-Demand Video Player**")
    
    st.divider()
    active_tab = st.radio(
        "Chế độ làm việc:",
        [
            "📊 Báo Cáo Thí Nghiệm & Ablation Leaderboard",
            "📂 Duyệt & Chỉnh Sửa Kết Quả Nộp Bài (Submission Console)",
            "🔍 Tìm kiếm Trực tiếp (Live Search)"
        ]
    )

# =============================================================================
# TAB 1: BENCHMARK & GROUND TRUTH REVIEW CONSOLE (47 CÂU EVALUATION)
# =============================================================================
if active_tab == "📊 Báo Cáo Thí Nghiệm & Ablation Leaderboard":
    st.title("🏆 AIC 2026: Ablation Leaderboard & Diagnostic Matrix")
    st.caption("Bảng tổng sắp hiệu năng 47 test cases chuẩn BTC trên tập dữ liệu Video Retrieval")

    # Load dữ liệu benchmark mới nhất
    ablation_summary_file = PROJECT_ROOT / "data" / "benchmark" / "ablation_study_summary.json"
    if ablation_summary_file.exists():
        with open(ablation_summary_file, "r", encoding="utf-8") as f:
            all_summaries = json.load(f)

        if all_summaries:
            best_s = max(all_summaries, key=lambda x: x.get("macro_score", 0.0))
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("🏆 Top Macro BTC Score", f"{best_s.get('macro_score', 0.0):.4f}", f"Cấu hình {best_s.get('config_id')}")
            with m2:
                st.metric("🎯 Video Recall@100", f"{best_s.get('video_r100', 0.0):.1%}", "Toàn bộ tập test")
            with m3:
                st.metric("🥇 Video Recall@1", f"{best_s.get('video_r1', 0.0):.1%}", "Rank 1 Tuyệt Đối")
            with m4:
                st.metric("⚡ Độ trễ trung bình", f"{best_s.get('avg_latency_ms', 0.0):.0f} ms", "Real-time Ready")

            st.divider()

            # Bảng Leaderboard Động
            st.subheader("🥇 Bảng Tổng Sắp Đối Đầu Ablation Study (Ground Truth 47 Câu)")
            df_lb = pd.DataFrame(all_summaries)
            cols_map = {
                "config_id": "Cấu Hình",
                "macro_score": "Macro Score 🏆",
                "kis_score": "KIS Score",
                "qa_score": "QA Score",
                "trake_score": "TRAKE Score",
                "video_r20": "Video-R@20",
                "video_r100": "Video-R@100",
                "avg_latency_ms": "Độ Trễ (ms)"
            }
            display_cols = [c for c in cols_map.keys() if c in df_lb.columns]
            df_display = df_lb[display_cols].rename(columns=cols_map).sort_values(by="Macro Score 🏆", ascending=False)
            st.dataframe(df_display, use_container_width=True)

# =============================================================================
# TAB 2: DUYỆT & CHỈNH SỬA KẾT QUẢ NỘP BÀI (SUBMISSION CONSOLE - REALTIME)
# =============================================================================
elif active_tab == "📂 Duyệt & Chỉnh Sửa Kết Quả Nộp Bài (Submission Console)":
    st.title("📂 AIC 2026: Championship Submission Console")
    st.caption("Giao diện kiểm duyệt trực quan, phát Video MP4 On-Demand, hiệu chỉnh Rank 1, QA Text và Chuỗi sự kiện TRAKE thời gian thực.")

    # 1. Quét động các thư mục con bên trong output/
    output_base_dir = PROJECT_ROOT / "output"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    available_subdirs = [p.name for p in output_base_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]

    if not available_subdirs:
        available_subdirs = ["thunghiem"]
        (output_base_dir / "thunghiem" / "submission").mkdir(parents=True, exist_ok=True)

    default_sub_idx = 0
    if "sotuyen1" in available_subdirs:
        default_sub_idx = available_subdirs.index("sotuyen1")
    elif "thunghiem" in available_subdirs:
        default_sub_idx = available_subdirs.index("thunghiem")

    col_cfg1, col_cfg2, col_cfg3 = st.columns([1.5, 1.5, 1.5])
    with col_cfg1:
        selected_subdir_name = st.selectbox("📁 Gói kết quả (trong output/):", available_subdirs, index=default_sub_idx)

    output_dir = output_base_dir / selected_subdir_name / "submission"
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_output_path = output_base_dir / selected_subdir_name / "submission.zip"

    # Thư mục đề bài tương ứng (nếu có)
    query_base_dir = PROJECT_ROOT / "query"
    avail_q_dirs = [p.name for p in query_base_dir.iterdir() if p.is_dir() and not p.name.startswith(".")] if query_base_dir.exists() else []
    
    default_q_idx = 0
    if "SOTUYEN1-bo-de-thi" in avail_q_dirs:
        default_q_idx = avail_q_dirs.index("SOTUYEN1-bo-de-thi")
    elif "THUNGHIEM-bo-de-thi" in avail_q_dirs:
        default_q_idx = avail_q_dirs.index("THUNGHIEM-bo-de-thi")

    with col_cfg2:
        selected_qdir_name = st.selectbox("📋 Gói đề bài (trong query/):", avail_q_dirs if avail_q_dirs else ["Không có"], index=default_q_idx)

    selected_query_dir = query_base_dir / selected_qdir_name if selected_qdir_name != "Không có" else None

    # Quét danh sách câu hỏi
    query_files = sorted(list(selected_query_dir.glob("*.txt"))) if selected_query_dir and selected_query_dir.exists() else []
    existing_csv_files = sorted(list(output_dir.glob("*.csv")))

    with col_cfg3:
        st.metric("📊 Tiến độ nộp bài", f"{len(existing_csv_files)} / {len(query_files) if query_files else len(existing_csv_files)} câu", f"Zip: {zip_output_path.name}")

    st.divider()



    # Query Selector
    query_map = {}
    if query_files:
        for q_p in query_files:
            with open(q_p, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            task_tag = "QA" if "qa" in q_p.name.lower() else ("TRAKE" if "trake" in q_p.name.lower() else "KIS")
            short_txt = txt[:60] + "..." if len(txt) > 60 else txt
            label = f"[{task_tag}] {q_p.name} - {short_txt}"
            query_map[label] = (q_p, txt, task_tag)
    else:
        for csv_p in existing_csv_files:
            task_tag = "QA" if "qa" in csv_p.name.lower() else ("TRAKE" if "trake" in csv_p.name.lower() else "KIS")
            label = f"[{task_tag}] {csv_p.name}"
            query_map[label] = (csv_p, f"Query từ kết quả {csv_p.name}", task_tag)

    if not query_map:
        st.warning("⚠️ Chưa tìm thấy câu hỏi hoặc kết quả nào. Hãy kiểm tra lại thư mục query/ hoặc output/.")
    else:
        selected_label = st.selectbox("📂 Chọn câu hỏi để soi ứng viên và hiệu chỉnh:", list(query_map.keys()))
        selected_q_path, q_content, task_tag = query_map[selected_label]
        selected_q_name = selected_q_path.name
        csv_stem = selected_q_path.stem
        target_csv_path = output_dir / f"{csv_stem}.csv"

        # Khung hiển thị đề bài
        tag_color = "#3b82f6" if task_tag == "KIS" else ("#10b981" if task_tag == "QA" else "#f59e0b")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); padding: 16px; border-radius: 10px; border-left: 6px solid {tag_color}; border: 1px solid #334155; margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: bold; font-size: 1.1rem; color: #f8fafc;">📋 ĐỀ BÀI TRUY VẤN: {selected_q_name}</span>
                <span style="background: {tag_color}; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 0.85rem;">{task_tag} TASK</span>
            </div>
            <div style="font-size: 1.05rem; line-height: 1.6; color: #e2e8f0; font-style: italic; background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px;">
                "{q_content}"
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Hàng nút điều khiển chính: SOTA Cân Bằng, Pure Visual, Boost ASR, Boost OCR
        st.markdown("##### ⚡ Điều Khiển Chạy Lại & Tăng Cường Đa Phương Thức (Modality Boost Controls):")
        col_btn_gpu, col_btn_vis, col_btn_asr, col_btn_ocr = st.columns([1.3, 1.4, 1.2, 1.2])
        with col_btn_gpu:
            run_gpu_btn = st.button("⚡ SOTA Engine (Tự Động)", type="primary", use_container_width=True, help="Tự động nhận diện thực thể và cân bằng đa phương thức")
        with col_btn_vis:
            run_vis_btn = st.button("👁️ Pure Visual (SigLIP + Dịch LLM)", use_container_width=True, help="Tắt hoàn toàn ASR & OCR, chỉ dùng bản dịch LLM tiếng Anh và tính toán tương đồng trực quan SigLIP2 + Intra-Video Reranker")
        with col_btn_asr:
            run_asr_btn = st.button("🎙️ Boost ASR (3.5x - Lời Thoại)", use_container_width=True, help="Ưu tiên cực cao cho lời thuyết minh, phỏng vấn, tên riêng, năm sản xuất, đạo diễn...")
        with col_btn_ocr:
            run_ocr_btn = st.button("🔤 Boost OCR (3.5x - Chữ Viết)", use_container_width=True, help="Ưu tiên cực cao cho chữ viết trên màn hình, biển báo, số áo, logo...")

        col_topk, col_filter, col_adv = st.columns([1.1, 1.2, 1.7])
        with col_topk:
            display_top_k = st.selectbox(
                "👀 Số lượng ứng viên hiển thị:",
                [10, 20, 30, 50, 100],
                index=1, # Mặc định Top 20
                key="disp_top_k_select"
            )

        with col_filter:
            filter_kw = st.text_input("🔍 Lọc nhanh theo Video ID:", placeholder="ví dụ: L26, L30...", key="filter_vid_kw")

        with col_adv:
            with st.expander("⚙️ Tùy Chỉnh Trọng Số Chi Tiết (Custom Weights)", expanded=False):
                custom_w_asr = st.slider("🎙️ Trọng số ASR (Lời thoại):", 0.0, 5.0, 1.8, 0.2, key="slider_w_asr")
                custom_w_ocr = st.slider("🔤 Trọng số OCR (Chữ in):", 0.0, 5.0, 1.8, 0.2, key="slider_w_ocr")

        if run_gpu_btn or run_vis_btn or run_asr_btn or run_ocr_btn:
            override_asr = 3.5 if run_asr_btn else (0.0 if run_vis_btn else (custom_w_asr if run_gpu_btn else 0.0))
            override_ocr = 3.5 if run_ocr_btn else (0.0 if run_vis_btn else (custom_w_ocr if run_gpu_btn else 0.0))

            if run_vis_btn:
                status_label = "👁️ Đang chạy Pure Visual (SigLIP2 + Bản Dịch Gemini LLM, Tắt ASR/OCR)..."
            elif run_asr_btn:
                status_label = "🎙️ Đang chạy với ASR Boost 3.5x..."
            elif run_ocr_btn:
                status_label = "🔤 Đang chạy với OCR Boost 3.5x..."
            else:
                status_label = "⚡ Đang chạy SOTA Engine..."

            with st.spinner(status_label):
                if task_tag == "QA":
                    preds, info, lat = engine.search_qa(
                        q_content,
                        top_k=100,
                        use_intra_reranker=True,
                        use_cue=(not run_vis_btn),
                        use_multimodal=(not run_vis_btn),
                        use_rrf=(not run_vis_btn)
                    )
                elif task_tag == "TRAKE":
                    preds, info, lat = engine.search_trake(
                        q_content,
                        top_k=100,
                        use_multi_query=True,
                        use_event_coverage=True,
                        use_row_norm_dp=True,
                        use_segmental_dp=True
                    )
                else:
                    preds, info, lat = engine.search_kis(
                        q_content,
                        top_k=100,
                        use_intra_reranker=True,
                        use_multimodal=(not run_vis_btn),
                        use_rrf=(not run_vis_btn),
                        w_asr_override=override_asr,
                        w_ocr_override=override_ocr
                    )

                with open(target_csv_path, "w", encoding="utf-8") as f:
                    for p in preds:
                        if task_tag == "QA":
                            ans = p.get("answer", info.get("generated_qa_answer", ""))
                            ans_clean = f'"{ans}"' if ans else '""'
                            f.write(f"{p['video_id']},{p['frame_idx']},{ans_clean}\n")
                        elif task_tag == "TRAKE" and "event_frames" in p:
                            ev_str = ",".join([str(x) for x in p["event_frames"]])
                            f.write(f"{p['video_id']},{ev_str}\n")
                        else:
                            f.write(f"{p['video_id']},{p['frame_idx']}\n")

                sync_submission_zip(output_dir, zip_output_path)
                st.success(f"✅ Đã chạy xong và cập nhật kết quả vào `{target_csv_path.name}` ({len(preds)} dòng) & tự động đồng bộ `{zip_output_path.name}`!")
                st.rerun()

        # Đọc dữ liệu CSV hiện tại
        rows = []
        if target_csv_path.exists():
            with open(target_csv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(",")
                        rows.append(parts)

        # Khu vực Thêm Thủ Công Clip & Frame Đúng
        with st.expander("➕ **THÊM / CHÈN THỦ CÔNG CLIP & FRAME ĐÚNG (MANUAL OVERRIDE)**", expanded=False):
            st.markdown("Nhập chính xác mã Video và Frame Index (hoặc Đáp án QA / Chuỗi TRAKE) để ghim thẳng lên đầu bảng kết quả:")
            if task_tag == "QA":
                c_in1, c_in2, c_in3 = st.columns([1.2, 1.2, 2.0])
            elif task_tag == "TRAKE":
                c_in1, c_in2, c_in3 = st.columns([1.2, 1.2, 2.0])
            else:
                c_in1, c_in2 = st.columns([1.5, 1.5])
                c_in3 = None

            with c_in1:
                man_vid = st.text_input("🎬 Mã Video (Video ID):", placeholder="VD: L22_V022", key=f"man_vid_{selected_q_name}").strip()
            with c_in2:
                man_frame = st.number_input("🖼️ Frame Index:", min_value=0, max_value=500000, value=0, step=1, key=f"man_frame_{selected_q_name}")
            
            man_qa_ans = ""
            man_trake_events = ""
            if task_tag == "QA" and c_in3:
                with c_in3:
                    man_qa_ans = st.text_input("💬 Đáp án QA (Answer text):", placeholder="VD: Đèo Ngang", key=f"man_qa_{selected_q_name}").strip()
            elif task_tag == "TRAKE" and c_in3:
                with c_in3:
                    man_trake_events = st.text_input("⏱️ Chuỗi frames (phân cách bằng dấu phẩy):", placeholder="VD: 6080,6536,10184", key=f"man_trake_{selected_q_name}").strip()

            c_act1, c_act2, c_spacer = st.columns([1.5, 1.3, 2.0])
            with c_act1:
                btn_set_r1 = st.button("👑 Đặt Làm Rank #1 Ngay Lập Tức", key=f"btn_man_r1_{selected_q_name}", use_container_width=True)
            with c_act2:
                btn_append = st.button("➕ Thêm Vào Cuối Danh Sách", key=f"btn_man_app_{selected_q_name}", use_container_width=True)

            if btn_set_r1 or btn_append:
                if not man_vid:
                    st.error("⚠️ Vui lòng nhập Mã Video (Video ID)!")
                else:
                    # Chuẩn hóa tên video
                    clean_vid = man_vid.replace(".mp4", "").replace(".MP4", "").strip()
                    if task_tag == "QA":
                        clean_ans = man_qa_ans.replace('"', '""')
                        new_row = [clean_vid, str(int(man_frame)), f'"{clean_ans}"']
                    elif task_tag == "TRAKE":
                        if man_trake_events:
                            ev_list = [str(int(x.strip())) for x in man_trake_events.split(",") if x.strip().isdigit()]
                            new_row = [clean_vid] + (ev_list if ev_list else [str(int(man_frame))])
                        else:
                            new_row = [clean_vid, str(int(man_frame))]
                    else:
                        new_row = [clean_vid, str(int(man_frame))]

                    # Lưu undo state trước khi thay đổi
                    save_undo_state(selected_q_name, rows)

                    # Xóa phần tử cũ nếu trùng chính xác (video_id, frame_idx)
                    rows = [r for r in rows if not (r[0] == clean_vid and len(r) > 1 and r[1] == str(int(man_frame)))]

                    if btn_set_r1:
                        rows.insert(0, new_row)
                    else:
                        rows.append(new_row)

                    # Giữ tối đa 100 dòng
                    rows = rows[:100]

                    with open(target_csv_path, "w", encoding="utf-8") as f:
                        for row_item in rows:
                            f.write(",".join(row_item) + "\n")

                    sync_submission_zip(output_dir, zip_output_path)
                    st.session_state["inspect_target"] = {
                        "query": selected_q_name,
                        "video_id": clean_vid,
                        "frame_idx": int(man_frame),
                        "rank": 1 if btn_set_r1 else len(rows)
                    }
                    st.success(f"✅ Đã thêm `{clean_vid}` (Frame `{man_frame}`) vào vị trí {'Rank #1 (Đã đẩy các frame cũ xuống)' if btn_set_r1 else 'cuối'} và đồng bộ file nộp bài!")
                    st.rerun()

        if not rows:
            st.info("Chưa có kết quả dự đoán nào cho câu hỏi này. Hãy bấm 'Chạy Lại Riêng Câu Này Trên GPU' ở trên hoặc thêm thủ công ở khung phía trên.")
        else:
            # Lọc theo từ khóa nếu có
            filtered_rows_with_idx = []
            for original_idx, r in enumerate(rows):
                if filter_kw.strip():
                    if filter_kw.strip().lower() in r[0].lower():
                        filtered_rows_with_idx.append((original_idx, r))
                else:
                    filtered_rows_with_idx.append((original_idx, r))

            # Khởi tạo an toàn 100% session_state cho video inspector
            if "inspect_target" not in st.session_state or not isinstance(st.session_state["inspect_target"], dict) or st.session_state["inspect_target"].get("query") != selected_q_name:
                st.session_state["inspect_target"] = {
                    "query": selected_q_name,
                    "video_id": rows[0][0],
                    "frame_idx": int(rows[0][1]) if len(rows[0]) > 1 and rows[0][1].isdigit() else 0,
                    "rank": 1
                }

            # =================================================================
            # MASTER-DETAIL SPLIT-SCREEN STUDIO LAYOUT (VBS SOTA DESIGN)
            # =================================================================
            col_left_grid, col_right_studio = st.columns([1.12, 0.88], gap="medium")

            # -----------------------------------------------------------------
            # CỘT TRÁI: LƯỚI ỨNG VIÊN & TOOLBAR HOÁN ĐỔI / HOÀN TÁC
            # -----------------------------------------------------------------
            with col_left_grid:
                st.markdown(f"#### 🖼️ Lưới Ứng Viên ({min(display_top_k, len(filtered_rows_with_idx))} / {len(rows)} dòng)")
                
                # Thanh Quick Swap & Undo
                c_sw_title, c_undo_box = st.columns([3.0, 2.0])
                with c_sw_title:
                    st.markdown("🔄 **Hoán đổi Rank:**")
                with c_undo_box:
                    has_undo = ("undo_history" in st.session_state and selected_q_name in st.session_state["undo_history"] and len(st.session_state["undo_history"][selected_q_name]) > 0)
                    if has_undo:
                        if st.button("↩️ Hoàn Tác (Undo)", key=f"undo_btn_{selected_q_name}", use_container_width=True):
                            prev_rows = st.session_state["undo_history"][selected_q_name].pop()
                            rows = prev_rows
                            with open(target_csv_path, "w", encoding="utf-8") as f:
                                for row_item in rows:
                                    f.write(",".join(row_item) + "\n")
                            sync_submission_zip(output_dir, zip_output_path)
                            st.success("✅ Đã hoàn tác lại!")
                            st.rerun()

                c_sw1, c_sw2, c_sw3 = st.columns([1.1, 1.1, 1.4])
                with c_sw1:
                    rank_a = st.number_input("Rank A:", min_value=1, max_value=max(1, len(rows)), value=1, step=1, key=f"swap_a_{selected_q_name}")
                with c_sw2:
                    rank_b = st.number_input("Rank B:", min_value=1, max_value=max(1, len(rows)), value=min(2, len(rows)), step=1, key=f"swap_b_{selected_q_name}")
                with c_sw3:
                    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                    if st.button(f"🔄 Đổi #{rank_a}↔#{rank_b}", key=f"do_swap_{selected_q_name}", use_container_width=True):
                        if rank_a != rank_b and 1 <= rank_a <= len(rows) and 1 <= rank_b <= len(rows):
                            save_undo_state(selected_q_name, rows)
                            rows[rank_a - 1], rows[rank_b - 1] = rows[rank_b - 1], rows[rank_a - 1]
                            with open(target_csv_path, "w", encoding="utf-8") as f:
                                for row_item in rows:
                                    f.write(",".join(row_item) + "\n")
                            sync_submission_zip(output_dir, zip_output_path)
                            st.success(f"✅ Đã đổi vị trí #{rank_a} ↔ #{rank_b}!")
                            st.rerun()

                # Lưới 3 Cột trong cột trái
                cols_3 = st.columns(3)
                cur_inspect_vid = st.session_state.get("inspect_target", {}).get("video_id", rows[0][0])
                for render_count, (orig_idx, r) in enumerate(filtered_rows_with_idx[:display_top_k]):
                    col = cols_3[render_count % 3]
                    with col:
                        vid = r[0]
                        f_idx = int(r[1]) if len(r) > 1 and r[1].isdigit() else 0
                        img = keyframe_loader.get_keyframe_image(vid, f_idx)
                        pts_sec = keyframe_loader.get_pts_time(vid, f_idx)
                        min_sec_str = f"{int(pts_sec // 60):02d}:{int(pts_sec % 60):02d}"

                        # Phân biệt viền thẻ theo trạng thái
                        is_r1 = (orig_idx == 0)
                        is_active_inspect = (vid == cur_inspect_vid)
                        card_class = "card-r1" if is_r1 else ("card-active" if is_active_inspect else ("card-top5" if orig_idx < 5 else "card-normal"))

                        st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)

                        # Glassmorphism Badges đè trên ảnh
                        c_bd1, c_bd2 = st.columns([1.2, 1.0])
                        with c_bd1:
                            if is_r1:
                                st.markdown(f"<span class='glass-badge-r1'>👑 #1 {vid}</span>", unsafe_allow_html=True)
                            elif orig_idx < 5:
                                st.markdown(f"<span class='glass-badge-top5'>#{orig_idx+1} {vid}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span class='glass-badge-normal'>#{orig_idx+1} {vid}</span>", unsafe_allow_html=True)
                        with c_bd2:
                            st.markdown(f"<span class='glass-time'>⏱️ {min_sec_str}</span>", unsafe_allow_html=True)

                        if img is not None:
                            st.image(img, use_container_width=True)
                        else:
                            st.info(f"Frame {f_idx}")

                        if task_tag == "QA" and len(r) > 2:
                            ans_display = ",".join(r[2:]).strip('"')
                            st.caption(f"💬 `{ans_display[:18]}`")
                        elif task_tag == "TRAKE":
                            st.caption(f"⏱️ `{len(r)-1} sự kiện`")

                        # 4 nút mini điều hướng
                        c_b1, c_b2, c_b3, c_b4 = st.columns([1, 1, 1, 1])
                        with c_b1:
                            if orig_idx > 0:
                                if st.button("⬆️", key=f"up_btn_{orig_idx}_{selected_q_name}", help="Đẩy lên 1 bậc", use_container_width=True):
                                    save_undo_state(selected_q_name, rows)
                                    rows[orig_idx], rows[orig_idx - 1] = rows[orig_idx - 1], rows[orig_idx]
                                    with open(target_csv_path, "w", encoding="utf-8") as f:
                                        for row_item in rows:
                                            f.write(",".join(row_item) + "\n")
                                    sync_submission_zip(output_dir, zip_output_path)
                                    st.rerun()
                        with c_b2:
                            if orig_idx < len(rows) - 1:
                                if st.button("⬇️", key=f"down_btn_{orig_idx}_{selected_q_name}", help="Hạ xuống 1 bậc", use_container_width=True):
                                    save_undo_state(selected_q_name, rows)
                                    rows[orig_idx], rows[orig_idx + 1] = rows[orig_idx + 1], rows[orig_idx]
                                    with open(target_csv_path, "w", encoding="utf-8") as f:
                                        for row_item in rows:
                                            f.write(",".join(row_item) + "\n")
                                    sync_submission_zip(output_dir, zip_output_path)
                                    st.rerun()
                        with c_b3:
                            if orig_idx > 0:
                                if st.button("⭐", key=f"promo_btn_{orig_idx}_{selected_q_name}", help="Chuyển lên Rank #1", use_container_width=True):
                                    save_undo_state(selected_q_name, rows)
                                    item_to_promote = rows.pop(orig_idx)
                                    rows.insert(0, item_to_promote)
                                    with open(target_csv_path, "w", encoding="utf-8") as f:
                                        for row_item in rows:
                                            f.write(",".join(row_item) + "\n")
                                    sync_submission_zip(output_dir, zip_output_path)
                                    st.session_state["inspect_target"] = {
                                        "query": selected_q_name,
                                        "video_id": item_to_promote[0],
                                        "frame_idx": int(item_to_promote[1]) if len(item_to_promote) > 1 and item_to_promote[1].isdigit() else 0,
                                        "rank": 1
                                    }
                                    st.rerun()
                        with c_b4:
                            if st.button("🎬", key=f"inspect_btn_{orig_idx}_{selected_q_name}", help="Mở soi video ngay bên phải", use_container_width=True):
                                st.session_state["inspect_target"] = {
                                    "query": selected_q_name,
                                    "video_id": vid,
                                    "frame_idx": f_idx,
                                    "rank": orig_idx + 1
                                }
                                st.rerun()

                        st.markdown("</div>", unsafe_allow_html=True)

            # -----------------------------------------------------------------
            # CỘT PHẢI: STUDIO SOI VIDEO MP4 & KEYFRAME FILMSTRIP (TRỰC QUAN 100%)
            # -----------------------------------------------------------------
            with col_right_studio:
                cur_insp = st.session_state.get("inspect_target", {"video_id": rows[0][0], "frame_idx": int(rows[0][1]), "rank": 1})
                insp_vid = cur_insp.get("video_id", rows[0][0])
                insp_fidx = cur_insp.get("frame_idx", int(rows[0][1]))
                insp_rank = cur_insp.get("rank", 1)

                # 📌 Khung hiển thị câu hỏi trực tiếp trên video studio
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.9); border-left: 5px solid {tag_color}; padding: 12px 14px; border-radius: 8px; margin-bottom: 12px; border: 1px solid #334155; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                        <span style="font-size: 0.85rem; font-weight: bold; color: {tag_color};">📋 ĐỀ BÀI [{task_tag}]: {selected_q_name}</span>
                        <span style="font-size: 0.8rem; color: #94a3b8;">Đang soi: <b style="color: #f1f5f9;">{insp_vid}</b> (Rank #{insp_rank})</span>
                    </div>
                    <div style="font-size: 1.0rem; color: #f8fafc; line-height: 1.5; font-style: italic; background: rgba(0,0,0,0.25); padding: 8px 10px; border-radius: 6px;">
                        "{q_content}"
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"#### 🎬 Studio Soi Video: `{insp_vid}` (Rank #{insp_rank})")

                # Lưu lại mốc frame ban đầu khi mở video để hỗ trợ nút quay lại
                if "initial_inspect_frame" not in st.session_state or st.session_state.get("initial_inspect_vid") != insp_vid:
                    st.session_state["initial_inspect_vid"] = insp_vid
                    st.session_state["initial_inspect_frame"] = insp_fidx

                init_fidx = st.session_state.get("initial_inspect_frame", insp_fidx)
                init_pts = keyframe_loader.get_pts_time(insp_vid, init_fidx)
                init_min_sec = f"{int(init_pts//60):02d}:{int(init_pts%60):02d}"

                ts_widget_key = f"ts_input_{selected_q_name}_{insp_vid}"
                fidx_widget_key = f"custom_fidx_{selected_q_name}_{insp_vid}"

                active_f_cur = st.session_state.get(fidx_widget_key, insp_fidx)
                pts_time_cur = keyframe_loader.get_pts_time(insp_vid, active_f_cur)
                cur_min_sec = f"{int(pts_time_cur//60):02d}:{int(pts_time_cur%60):02d}"

                play_mode = st.radio(
                    "Chế độ phát video:",
                    ["⚡ Đoạn Ngắn (60s - Siêu Mượt)", "🌐 Toàn Bộ Video Gốc"],
                    horizontal=True,
                    key=f"play_mode_{selected_q_name}_{insp_vid}"
                )

                with st.spinner("⏳ Đang chuẩn bị video..."):
                    is_short_clip = "Đoạn Ngắn" in play_mode and hasattr(video_manager, "get_optimized_clip")
                    if is_short_clip:
                        clip_path, clip_start, clip_dur = video_manager.get_optimized_clip(insp_vid, pts_time_cur, clip_window=60.0)
                        if clip_path and clip_path.exists():
                            offset_in_clip = max(0.0, pts_time_cur - clip_start)
                            st.video(str(clip_path), start_time=int(offset_in_clip))
                            c_start_m = f"{int(clip_start//60):02d}:{int(clip_start%60):02d}"
                            c_end_m = f"{int((clip_start+clip_dur)//60):02d}:{int((clip_start+clip_dur)%60):02d}"
                            st.caption(f"⚡ Clip `{c_start_m}` đến `{c_end_m}` | Mốc chọn: `{cur_min_sec}` -> Frame `{active_f_cur}`")
                        else:
                            kf_big = keyframe_loader.get_keyframe_image(insp_vid, active_f_cur)
                            if kf_big:
                                st.image(kf_big, use_container_width=True)
                    else:
                        v_path = video_manager.get_video_path(insp_vid)
                        if v_path and v_path.exists():
                            st.video(str(v_path), start_time=int(max(0, pts_time_cur - 1.5)))
                            st.caption(f"📁 Video gốc | Mốc tua: `{cur_min_sec}` -> Frame `{active_f_cur}`")
                        else:
                            kf_big = keyframe_loader.get_keyframe_image(insp_vid, active_f_cur)
                            if kf_big:
                                st.image(kf_big, use_container_width=True)

                st.markdown("##### ⏱️ Bắt Khung Hình Khi Xem Video:")
                col_ts1, col_btn_calc, col_ts2 = st.columns([1.2, 0.8, 1.0])
                with col_ts1:
                    user_time_str = st.text_input(
                        "Thời gian (MM:SS):",
                        value="",
                        key=f"ts_raw_input_{selected_q_name}_{insp_vid}",
                        placeholder="VD: 01:40..."
                    )

                with col_btn_calc:
                    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                    btn_calc_time = st.button("⚡ Tính", use_container_width=True, help="Quy đổi thời gian thành Frame")

                if (btn_calc_time or user_time_str.strip()) and user_time_str.strip():
                    try:
                        clean_ts = user_time_str.strip()
                        if ":" in clean_ts:
                            m_s, s_s = clean_ts.split(":")
                            t_sec = float(m_s) * 60.0 + float(s_s)
                        else:
                            t_sec = float(clean_ts)

                        if hasattr(keyframe_loader, "get_exact_frame_from_time"):
                            new_f = keyframe_loader.get_exact_frame_from_time(insp_vid, t_sec)
                        else:
                            df_v = keyframe_loader.df_frames[keyframe_loader.df_frames["video_id"] == insp_vid]
                            fps = float(df_v.iloc[0]["fps"]) if not df_v.empty and "fps" in df_v.columns and float(df_v.iloc[0]["fps"]) > 0 else 25.0
                            new_f = int(round(t_sec * fps))

                        st.session_state[fidx_widget_key] = new_f
                        st.session_state["inspect_target"]["frame_idx"] = new_f
                    except Exception as e:
                        st.error(f"Lỗi: {e}")

                current_target_frame = st.session_state.get(fidx_widget_key, insp_fidx)

                with col_ts2:
                    custom_f_input = st.number_input(
                        "Số Frame:",
                        min_value=0,
                        max_value=300000,
                        value=current_target_frame,
                        step=1,
                        key=f"fnum_input_{selected_q_name}_{insp_vid}_{current_target_frame}"
                    )
                    if custom_f_input != current_target_frame:
                        st.session_state[fidx_widget_key] = custom_f_input
                        st.session_state["inspect_target"]["frame_idx"] = custom_f_input
                        current_target_frame = custom_f_input

                active_chosen_frame = st.session_state.get(fidx_widget_key, current_target_frame)
                exact_pts_chosen = keyframe_loader.get_pts_time(insp_vid, active_chosen_frame)
                exact_pts_str = f"{int(exact_pts_chosen//60):02d}:{int(exact_pts_chosen%60):02d}"

                col_act_r1, col_act_reset = st.columns([1.6, 1.0])
                with col_act_r1:
                    if st.button(f"👑 Chèn Lên Rank #1 (Frame {active_chosen_frame})", type="primary", use_container_width=True):
                        save_undo_state(selected_q_name, rows)
                        target_row = [insp_vid, str(active_chosen_frame)]
                        if len(rows[0]) > 2:
                            target_row.extend(rows[0][2:])
                        
                        rows = [r for r in rows if not (r[0] == insp_vid and len(r) > 1 and r[1] == str(active_chosen_frame))]
                        rows.insert(0, target_row)
                        rows = rows[:100]

                        with open(target_csv_path, "w", encoding="utf-8") as f:
                            for row_item in rows:
                                f.write(",".join(row_item) + "\n")
                        sync_submission_zip(output_dir, zip_output_path)
                        st.session_state["inspect_target"] = {
                            "query": selected_q_name,
                            "video_id": insp_vid,
                            "frame_idx": active_chosen_frame,
                            "rank": 1
                        }
                        st.success(f"🎉 Đã chèn `{insp_vid}` - Frame `{active_chosen_frame}` lên Rank #1 (Đã đẩy các frame cũ xuống an toàn)!")
                        st.rerun()

                with col_act_reset:
                    if st.button(f"🔄 Mốc gốc ({init_min_sec})", use_container_width=True):
                        st.session_state[fidx_widget_key] = init_fidx
                        st.session_state["inspect_target"]["frame_idx"] = init_fidx
                        st.rerun()

                # Dải Keyframe Filmstrip
                st.markdown("##### 🎞️ Dải Keyframe lân cận (Click để gán):")
                surr_kfs = keyframe_loader.get_surrounding_keyframes(insp_vid, active_chosen_frame, count=8)
                if surr_kfs:
                    k_cols = st.columns(4)
                    for k_idx, sk in enumerate(surr_kfs):
                        with k_cols[k_idx % 4]:
                            is_curr = sk["is_current"]
                            f_val = sk["frame_idx"]
                            border_style = "border:2px solid #eab308;" if is_curr else "border:1px solid #334155;"
                            st.markdown(f"<div style='text-align:center; padding:2px; {border_style} border-radius:4px;'>", unsafe_allow_html=True)
                            if sk["image"]:
                                st.image(sk["image"], use_container_width=True)
                            st.caption(f"{'🎯 ' if is_curr else ''}`{f_val}`")
                            if st.button(f"Chọn", key=f"pick_kf_{insp_vid}_{f_val}_{k_idx}"):
                                st.session_state[fidx_widget_key] = f_val
                                st.session_state["inspect_target"]["frame_idx"] = f_val
                                st.rerun()
                            st.markdown("</div>", unsafe_allow_html=True)

            # =================================================================
            # STUDIO TRAKE CHUYÊN DỤNG (REBUILT FOR MULTI-EVENT SEQUENCE)
            # =================================================================
            if task_tag == "TRAKE":
                st.divider()
                st.subheader("⏱️ Studio Căn Chỉnh Chuỗi Sự Kiện TRAKE Đa Tầng (Interactive TRAKE Sequencer)")
                st.caption("Xem từng sự kiện con theo thứ tự thời gian, vi chỉnh độc lập từng khung hình và tự động kiểm tra quy chế BTC ($E_1 < E_2 < E_3$).")

                # Bóc tách các mô tả sự kiện con từ câu hỏi
                parsed_events = parse_trake_subevents(q_content)
                
                # Cho phép chọn bất kỳ video ứng viên nào trong Top 10 để chỉnh sửa chuỗi sự kiện
                trake_cand_options = [f"Rank #{i+1}: {r[0]} ({len(r)-1} events)" for i, r in enumerate(rows[:10])]
                selected_trake_cand = st.selectbox("🎯 Chọn Video ứng viên để chỉnh sửa chuỗi sự kiện:", trake_cand_options, index=0)
                selected_trake_idx = int(selected_trake_cand.split(":")[0].replace("Rank #", "")) - 1
                trake_row = rows[selected_trake_idx]
                trake_vid = trake_row[0]
                trake_frames = [int(x) for x in trake_row[1:] if x.isdigit()]

                # SỐ LƯỢNG SỰ KIỆN PHẢI ĐỒNG BỘ CHUẨN XÁC VỚI KẾT QUẢ ĐỀ BÀI (KHÔNG SINH THỪA SỰ KIỆN)
                if trake_frames:
                    num_events = len(trake_frames)
                else:
                    num_events = len(parsed_events) if parsed_events else 3
                    all_kfs = keyframe_loader.get_all_video_keyframes(trake_vid)
                    trake_frames = all_kfs[:num_events] if len(all_kfs) >= num_events else [100 * (i+1) for i in range(num_events)]

                # Cắt hoặc gán mô tả đúng chuẩn num_events
                if len(parsed_events) > num_events:
                    parsed_events = parsed_events[:num_events]
                while len(parsed_events) < num_events:
                    parsed_events.append(f"Sự kiện {len(parsed_events)+1}")

                trake_frames = trake_frames[:num_events]

                ev_cols = st.columns(num_events)
                updated_trake_frames = []

                all_vid_kfs = keyframe_loader.get_all_video_keyframes(trake_vid)
                min_f = all_vid_kfs[0] if all_vid_kfs else 0
                max_f = all_vid_kfs[-1] if all_vid_kfs else 200000

                for e_idx in range(num_events):
                    ev_desc = parsed_events[e_idx] if e_idx < len(parsed_events) else f"Sự kiện {e_idx+1}"
                    cur_f = trake_frames[e_idx]

                    with ev_cols[e_idx]:
                        st.markdown(f"""
                        <div class="event-card">
                            <span style="font-weight:bold; color:#f59e0b;">SỰ KIỆN E{e_idx+1}</span><br/>
                            <small style="color:#cbd5e1; font-style:italic;">"{ev_desc[:40]}"</small>
                        </div>
                        """, unsafe_allow_html=True)

                        ev_img = keyframe_loader.get_dense_video_frame(trake_vid, cur_f) or keyframe_loader.get_keyframe_image(trake_vid, cur_f)
                        if ev_img:
                            st.image(ev_img, use_container_width=True)
                        else:
                            st.info(f"Frame {cur_f}")

                        # Điều khiển tăng giảm
                        new_f_val = st.number_input(
                            f"Frame E{e_idx+1}:",
                            min_value=0,
                            max_value=300000,
                            value=cur_f,
                            step=5,
                            key=f"trake_num_{selected_q_name}_{selected_trake_idx}_{e_idx}"
                        )
                        updated_trake_frames.append(new_f_val)

                # Kiểm tra tính đơn điệu
                is_valid_monotonic = all(updated_trake_frames[i] < updated_trake_frames[i+1] for i in range(len(updated_trake_frames)-1))

                st.markdown("<br/>", unsafe_allow_html=True)
                col_val_stat, col_val_btn = st.columns([2, 1])

                with col_val_stat:
                    chain_repr = " ➔ ".join([f"E{i+1}({f})" for i, f in enumerate(updated_trake_frames)])
                    if is_valid_monotonic:
                        st.success(f"✅ **Chuỗi thời gian HỢP LỆ chuẩn quy chế BTC ($E_1 < E_2 < \dots < E_n$):**\n`{chain_repr}`")
                    else:
                        st.error(f"⚠️ **LỖI THỨ TỰ THỜI GIAN:** Các khung hình sự kiện bắt buộc phải tăng dần theo thời gian!\n`{chain_repr}`")

                with col_val_btn:
                    if st.button("💾 Lưu Chuỗi TRAKE & Đặt Làm Rank 1", type="primary", use_container_width=True, disabled=not is_valid_monotonic):
                        # Cập nhật chuỗi sự kiện và đưa lên Rank 1
                        new_row = [trake_vid] + [str(x) for x in updated_trake_frames]
                        if selected_trake_idx > 0:
                            rows.pop(selected_trake_idx)
                        else:
                            rows.pop(0)
                        rows.insert(0, new_row)

                        with open(target_csv_path, "w", encoding="utf-8") as f:
                            for row_item in rows:
                                f.write(",".join(row_item) + "\n")
                        sync_submission_zip(output_dir, zip_output_path)
                        st.success(f"🎉 Đã lưu chuỗi sự kiện TRAKE cho `{trake_vid}` lên Rank #1 & tự động cập nhật submission.zip!")
                        st.rerun()

            # =================================================================
            # STUDIO CHỈNH SỬA ĐÁP ÁN QA CHUYÊN DỤNG
            # =================================================================
            elif task_tag == "QA":
                st.divider()
                st.subheader("💬 Studio Chỉnh Sửa Đáp Án Visual Q&A")
                st.caption("Nhập hoặc sửa câu trả lời cho Rank #1, tự động định dạng chuẩn CSV dạng `video_id,frame_idx,\"câu trả lời\"`.")

                curr_qa_ans = ""
                if rows and len(rows[0]) > 2:
                    curr_qa_ans = ",".join(rows[0][2:]).strip().strip('"')

                qa_c1, qa_c2 = st.columns([2, 1])
                with qa_c1:
                    new_qa_text = st.text_input(
                        "📝 Câu trả lời chính thức cho Rank #1:",
                        value=curr_qa_ans,
                        key=f"qa_input_box_{selected_q_name}",
                        placeholder="ví dụ: Giang Ly, bánh xèo, mũ bảo hiểm..."
                    )
                with qa_c2:
                    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                    if st.button("💾 Lưu Đáp Án QA", type="primary", use_container_width=True):
                        clean_qa_ans = new_qa_text.strip()
                        if len(rows[0]) >= 2:
                            rows[0] = [rows[0][0], rows[0][1], f'"{clean_qa_ans}"']
                        with open(target_csv_path, "w", encoding="utf-8") as f:
                            for row_item in rows:
                                if len(row_item) > 2:
                                    f.write(f'{row_item[0]},{row_item[1]},{row_item[2]}\n')
                                else:
                                    f.write(f'{row_item[0]},{row_item[1]},"{clean_qa_ans}"\n')
                        sync_submission_zip(output_dir, zip_output_path)
                        st.success(f'🎉 Đã lưu đáp án QA: `"{clean_qa_ans}"` & tự động cập nhật submission.zip!')
                        st.rerun()

            # =================================================================
            # XUẤT VÀ KIỂM TRA SUBMISSION.ZIP
            # =================================================================
            st.divider()
            st.subheader("📦 Kiểm Tra Chuẩn Quy Chế & Tải Gói Nộp Bài (.ZIP)")
            sync_submission_zip(output_dir, zip_output_path)
            
            val_res = validator.validate_directory(output_dir)
            if val_res.get("all_valid", False):
                st.success(f"🎉 Toàn bộ {val_res['total_files']} file CSV trong thư mục `{output_dir.name}` đều HỢP LỆ 100% chuẩn quy chế BTC!")
            else:
                st.warning(f"⚠️ Kiểm tra file CSV: {val_res.get('error', 'Một số file cần kiểm tra thêm')}")

            if zip_output_path.exists():
                with open(zip_output_path, "rb") as fz:
                    zip_data_bytes = fz.read()
                st.download_button(
                    label="📦 TẢI GÓI NỘP BÀI CHÍNH THỨC (submission.zip)",
                    data=zip_data_bytes,
                    file_name="submission.zip",
                    mime="application/zip",
                    type="primary",
                    use_container_width=True
                )

# =============================================================================
# TAB 3: TÌM KIẾM TRỰC TIẾP (LIVE SEARCH)
# =============================================================================
else:
    st.title("🔍 AIC 2026: Live Multimodal Search Engine")
    st.caption("Truy vấn đa phương thức tiếng Việt với định tuyến thông minh theo Task (KIS / QA / TRAKE)")

    col1, col2 = st.columns([3, 1])
    with col1:
        query_text = st.text_input("🔍 Nhập câu truy vấn tiếng Việt:", placeholder="ví dụ: Người thợ gốm nhào đất trên bàn xoay...")
    with col2:
        task_choice = st.selectbox("Loại bài toán:", ["Auto (LLM Refiner)", "KIS (Khoảnh khắc)", "QA (Hỏi - Đáp)", "TRAKE (Chuỗi hành động)"])

    if st.button("🚀 Bắt đầu Tìm kiếm", type="primary", use_container_width=True):
        if not query_text.strip():
            st.warning("Vui lòng nhập nội dung truy vấn!")
        else:
            with st.spinner("Đang tiền xử lý qua LLM và truy xuất đa phương thức siêu tốc..."):
                t0 = time.time()
                if "KIS" in task_choice or task_choice == "Auto (LLM Refiner)":
                    preds, info, lat = kis_handler.search(query_text, top_k=50)
                elif "QA" in task_choice:
                    preds, info, lat = qa_handler.search(query_text, top_k=50)
                else:
                    preds, info, lat = trake_handler.search(query_text, top_k=50)

                total_ms = (time.time() - t0) * 1000

            st.success(f"✅ Hoàn tất trong {total_ms:.1f} ms | Tìm thấy {len(preds)} kết quả!")
            
            if "vlm_answer" in info and info["vlm_answer"]:
                st.info(f"💡 **Câu trả lời Visual Q&A (VLM):** `{info['vlm_answer']}`")

            if "refined" in info:
                ref = info["refined"]
                with st.expander("🧠 Chi tiết Tiền xử lý & Làm giàu truy vấn từ LLM Refiner"):
                    st.json(ref)

            # Grid View Top 16
            cols = st.columns(4)
            for idx, cand in enumerate(preds[:16]):
                with cols[idx % 4]:
                    vid = cand["video_id"]
                    sc = cand.get("score", 0.0)
                    
                    if "events" in cand and isinstance(cand["events"], list):
                        # TRAKE multi-event sequence
                        ev_str = " → ".join(str(e) for e in cand["events"])
                        st.markdown(f"<span class='rank-badge-normal'>Rank #{idx+1}</span> **{vid}**", unsafe_allow_html=True)
                        st.caption(f"Events: `{ev_str}`")
                        first_f = int(cand["events"][0]) if cand["events"] else 0
                        img = keyframe_loader.get_keyframe_image(vid, first_f)
                        if img is not None:
                            st.image(img, use_container_width=True, caption=f"E1: {first_f}")
                    else:
                        # KIS / QA single keyframe
                        f_idx = int(cand.get("frame_idx", 0))
                        img = keyframe_loader.get_keyframe_image(vid, f_idx)
                        st.markdown(f"<span class='rank-badge-normal'>Rank #{idx+1}</span> **{vid}** : `{f_idx}`", unsafe_allow_html=True)
                        if img is not None:
                            st.image(img, use_container_width=True)
                        else:
                            st.info(f"Frame: {f_idx}")
                        st.caption(f"Score: `{sc:.4f}`")
                        if "answer" in cand and cand["answer"]:
                            st.caption(f"Ans: `{cand['answer']}`")
