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

from src.retrieval.task_specialized_engine import TaskSpecializedEngine
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
    .candidate-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 12px;
        transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    }
    .candidate-card:hover {
        transform: translateY(-3px);
        border-color: #3b82f6;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
    }
    .rank-badge-1 {
        background: linear-gradient(135deg, #eab308, #ca8a04);
        color: #0f172a;
        padding: 3px 8px;
        border-radius: 5px;
        font-weight: 800;
        font-size: 0.85rem;
    }
    .rank-badge-top5 {
        background: #3b82f6;
        color: white;
        padding: 3px 7px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .rank-badge-normal {
        background: #475569;
        color: #f1f5f9;
        padding: 3px 6px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 0.8rem;
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
def get_engine():
    return TaskSpecializedEngine(engine="siglip2", batch="batch_1")

@st.cache_resource
def get_keyframe_loader():
    return KeyframeZipLoader()

@st.cache_resource
def get_video_manager():
    return VideoPlayerManager()

@st.cache_resource
def get_validator():
    return SubmissionValidator()

engine = get_engine()
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
        if temp_zip.exists():
            temp_zip.unlink(missing_ok=True)

def parse_trake_subevents(query_text: str) -> list[str]:
    """Bóc tách các sự kiện con E1, E2, ... từ văn bản truy vấn tiếng Việt."""
    lines = [l.strip() for l in query_text.split("\n") if l.strip()]
    events = []
    for l in lines:
        if re.search(r'^(sự kiện|event|bước|e)\s*\d+[:.-]', l, re.IGNORECASE) or re.search(r'^\d+[\.\)]', l):
            cleaned = re.sub(r'^(sự kiện|event|bước|e)\s*\d+[:.-]\s*', '', l, flags=re.IGNORECASE)
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned)
            if cleaned:
                events.append(cleaned)
    if not events:
        # Fallback tách theo dấu chấm hoặc gạch đầu dòng
        parts = [p.strip() for p in re.split(r'[\n;]', query_text) if len(p.strip()) > 8]
        if len(parts) >= 2:
            events = parts
        else:
            events = ["Sự kiện 1", "Sự kiện 2", "Sự kiện 3"]
    return events

# Sidebar
with st.sidebar:
    st.image("https://img.shields.io/badge/AIC_2026-CHAMPIONSHIP_CONSOLE-gold?style=for-the-badge&logo=google", use_container_width=True)
    st.header("⚙️ Trung tâm Điều khiển")
    st.caption("Engine: **Google SigLIP-2 (1152d)** + **Gemini 2.5 Flash Lite** + **On-Demand Video Player**")
    
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
    latest_res_file = PROJECT_ROOT / "data" / "benchmark" / "latest_ablation_results.json"
    if latest_res_file.exists():
        with open(latest_res_file, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        s = bench_data.get("summary", {})
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("🏆 Macro BTC Score", f"{s.get('final_score', 0.0):.4f}", "Config 25 Quán Quân")
        with m2:
            st.metric("🎯 Video Recall@100", f"{s.get('video_recall_100', 0.0):.1f}%", "46/47 Video")
        with m3:
            st.metric("🥇 Video Recall@1", f"{s.get('video_recall_1', 0.0):.1f}%", "Rank 1 Tuyệt Đối")
        with m4:
            st.metric("⚡ Độ trễ TB", f"{s.get('latency_ms', 0.0):.0f} ms", "Real-time Ready")

        st.divider()

        # Bảng Leaderboard
        st.subheader("🥇 Bảng Tổng Sắp Cấu Hình (Leaderboard)")
        lb_records = [
            {"Cấu hình": "Config 25 (WRRF SOTA)", "Macro": 0.5532, "KIS": 0.6500, "QA": 0.3714, "TRAKE": 0.5200, "Video-R@100": "97.9%", "Video-R@20": "87.2%", "Status": "🥇 Quán Quân"},
            {"Cấu hình": "Config 22 (Baseline SOTA)", "Macro": 0.5415, "KIS": 0.6286, "QA": 0.3714, "TRAKE": 0.5300, "Video-R@100": "93.6%", "Video-R@20": "80.9%", "Status": "🥈 Top 2"},
            {"Cấu hình": "Config 24 (Dual Indexing)", "Macro": 0.5064, "KIS": 0.6214, "QA": 0.2857, "TRAKE": 0.4800, "Video-R@100": "91.5%", "Video-R@20": "80.9%", "Status": "🥉 Top 3"},
            {"Cấu hình": "Config 23 (Fast Linguistic Gate)", "Macro": 0.5021, "KIS": 0.5786, "QA": 0.3286, "TRAKE": 0.5600, "Video-R@100": "93.6%", "Video-R@20": "80.9%", "Status": "#4"},
            {"Cấu hình": "Config 26 (Master Tri-Modal)", "Macro": 0.4989, "KIS": 0.6000, "QA": 0.3000, "TRAKE": 0.4900, "Video-R@100": "95.7%", "Video-R@20": "83.0%", "Status": "#5"}
        ]
        st.dataframe(pd.DataFrame(lb_records), use_container_width=True)

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
    if "thunghiem" in available_subdirs:
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
    
    with col_cfg2:
        selected_qdir_name = st.selectbox("📋 Gói đề bài (trong query/):", avail_q_dirs if avail_q_dirs else ["Không có"])

    selected_query_dir = query_base_dir / selected_qdir_name if selected_qdir_name != "Không có" else None

    # Quét danh sách câu hỏi
    query_files = sorted(list(selected_query_dir.glob("*.txt"))) if selected_query_dir and selected_query_dir.exists() else []
    existing_csv_files = sorted(list(output_dir.glob("*.csv")))

    with col_cfg3:
        st.metric("📊 Tiến độ nộp bài", f"{len(existing_csv_files)} / {len(query_files) if query_files else len(existing_csv_files)} câu", f"Zip: {zip_output_path.name}")

    st.divider()

    # Nút Auto-Run toàn bộ gói đề thi nếu cần
    if query_files:
        with st.expander("⚡ Chạy Tự Động Toàn Bộ Gói Đề Thi (Batch Auto-Run)", expanded=False):
            st.info(f"Phát hiện {len(query_files)} câu hỏi trong `{selected_query_dir.name}`. Bấm nút dưới để chạy tự động toàn bộ bằng Config 25 SOTA.")
            if st.button("🚀 BẮT ĐẦU CHẠY TOÀN BỘ CÂU HỎI TRÊN GPU", type="primary"):
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                for idx, q_path in enumerate(query_files):
                    status_text.text(f"[{idx+1}/{len(query_files)}] Đang xử lý câu: {q_path.name}...")
                    with open(q_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    is_qa = "qa" in q_path.name.lower()
                    is_trake = "trake" in q_path.name.lower()
                    if is_qa:
                        preds, info, _ = engine.search_qa(content, top_k=100, use_intra_reranker=True, use_cue=True, use_multimodal=True, use_rrf=True)
                    elif is_trake:
                        preds, info, _ = engine.search_trake(content, top_k=100, use_multi_query=True, use_event_coverage=True, use_row_norm_dp=True, use_segmental_dp=True)
                    else:
                        preds, info, _ = engine.search_kis(content, top_k=100, use_intra_reranker=True, use_cue=True, use_multimodal=True, use_rrf=True)

                    out_csv = output_dir / f"{q_path.stem}.csv"
                    with open(out_csv, "w", encoding="utf-8") as f:
                        for p in preds:
                            if is_qa:
                                ans = p.get("answer", info.get("generated_qa_answer", ""))
                                ans_clean = f'"{ans}"' if ans else '""'
                                f.write(f"{p['video_id']},{p['frame_idx']},{ans_clean}\n")
                            elif is_trake and "event_frames" in p:
                                ev_str = ",".join([str(x) for x in p["event_frames"]])
                                f.write(f"{p['video_id']},{ev_str}\n")
                            else:
                                f.write(f"{p['video_id']},{p['frame_idx']}\n")
                    progress_bar.progress((idx + 1) / len(query_files))
                sync_submission_zip(output_dir, zip_output_path)
                status_text.success(f"🎉 Đã hoàn tất {len(query_files)} câu và tự động cập nhật submission.zip!")
                st.rerun()

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

        # Hàng nút điều khiển chính & Bộ chọn Top-K
        col_btn_gpu, col_topk, col_filter = st.columns([1.5, 1.2, 1.3])
        with col_btn_gpu:
            run_gpu_btn = st.button("⚡ Chạy Lại Riêng Câu Này Trên GPU (SOTA Engine)", type="primary", use_container_width=True)

        with col_topk:
            display_top_k = st.selectbox(
                "👀 Số lượng ứng viên hiển thị:",
                [10, 20, 30, 50, 100],
                index=1, # Mặc định Top 20
                key="disp_top_k_select"
            )

        with col_filter:
            filter_kw = st.text_input("🔍 Lọc nhanh theo Video ID:", placeholder="ví dụ: L26, L30...", key="filter_vid_kw")

        if run_gpu_btn:
            with st.spinner("Đang chạy mô hình AI trên GPU..."):
                if task_tag == "QA":
                    preds, info, lat = engine.search_qa(q_content, top_k=100, use_intra_reranker=True, use_cue=True, use_multimodal=True, use_rrf=True)
                elif task_tag == "TRAKE":
                    preds, info, lat = engine.search_trake(q_content, top_k=100, use_multi_query=True, use_event_coverage=True, use_row_norm_dp=True, use_segmental_dp=True)
                else:
                    preds, info, lat = engine.search_kis(q_content, top_k=100, use_intra_reranker=True, use_dense_video_refiner=False)

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

        if not rows:
            st.info("Chưa có kết quả dự đoán nào cho câu hỏi này. Hãy bấm 'Chạy Lại Riêng Câu Này Trên GPU' ở trên.")
        else:
            # Lọc theo từ khóa nếu có
            filtered_rows_with_idx = []
            for original_idx, r in enumerate(rows):
                if filter_kw.strip():
                    if filter_kw.strip().lower() in r[0].lower():
                        filtered_rows_with_idx.append((original_idx, r))
                else:
                    filtered_rows_with_idx.append((original_idx, r))

            st.markdown(f"### 🖼️ Lưới Ứng Viên Đa Tầng (Hiển thị {min(display_top_k, len(filtered_rows_with_idx))} / {len(rows)} dòng)")

            # Khởi tạo session_state cho video inspector nếu chưa có
            if "inspect_target" not in st.session_state or st.session_state["inspect_target"].get("query") != selected_q_name:
                st.session_state["inspect_target"] = {
                    "query": selected_q_name,
                    "video_id": rows[0][0],
                    "frame_idx": int(rows[0][1]) if len(rows[0]) > 1 and rows[0][1].isdigit() else 0,
                    "rank": 1
                }

            # Hiển thị Lưới 5 Cột
            cols_5 = st.columns(5)
            for render_count, (orig_idx, r) in enumerate(filtered_rows_with_idx[:display_top_k]):
                col = cols_5[render_count % 5]
                with col:
                    vid = r[0]
                    f_idx = int(r[1]) if len(r) > 1 and r[1].isdigit() else 0
                    img = keyframe_loader.get_keyframe_image(vid, f_idx)
                    pts_sec = keyframe_loader.get_pts_time(vid, f_idx)
                    min_sec_str = f"{int(pts_sec // 60):02d}:{int(pts_sec % 60):02d}"

                    # Header thẻ ứng viên
                    if orig_idx == 0:
                        st.markdown(f"<span class='rank-badge-1'>👑 RANK #1</span> **{vid}**", unsafe_allow_html=True)
                    elif orig_idx < 5:
                        st.markdown(f"<span class='rank-badge-top5'>RANK #{orig_idx+1}</span> **{vid}**", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span class='rank-badge-normal'>#{orig_idx+1}</span> **{vid}**", unsafe_allow_html=True)

                    if img is not None:
                        st.image(img, use_container_width=True)
                    else:
                        st.info(f"Frame {f_idx}")

                    st.caption(f"Frame: `{f_idx}` ({min_sec_str})")
                    if task_tag == "QA" and len(r) > 2:
                        ans_display = ",".join(r[2:]).strip('"')
                        st.caption(f"Ans: `{ans_display[:20]}`")
                    elif task_tag == "TRAKE":
                        st.caption(f"Chuỗi: `{len(r)-1} events`")

                    col_c1, col_c2 = st.columns([1, 1])
                    with col_c1:
                        if orig_idx > 0:
                            if st.button("⭐ Đặt R1", key=f"promo_btn_{orig_idx}_{selected_q_name}", use_container_width=True):
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

                    with col_c2:
                        if st.button("🎬 Soi Video", key=f"inspect_btn_{orig_idx}_{selected_q_name}", use_container_width=True):
                            st.session_state["inspect_target"] = {
                                "query": selected_q_name,
                                "video_id": vid,
                                "frame_idx": f_idx,
                                "rank": orig_idx + 1
                            }
                            st.rerun()

            # =================================================================
            # TRÌNH PHÁT VIDEO MP4 ON-DEMAND & KÍNH HIỂN VI KEYFRAME
            # =================================================================
            st.divider()
            cur_insp = st.session_state.get("inspect_target", {"video_id": rows[0][0], "frame_idx": int(rows[0][1]), "rank": 1})
            insp_vid = cur_insp.get("video_id", rows[0][0])
            insp_fidx = cur_insp.get("frame_idx", int(rows[0][1]))
            insp_rank = cur_insp.get("rank", 1)

            st.subheader(f"🎬 Studio Soi Video MP4 & Keyframe Filmstrip: `{insp_vid}` (Đang chọn từ Rank #{insp_rank})")
            st.caption("Xem video thực tế với đầy đủ âm thanh/chuyển động, tua dòng thời gian và gán ngay frame ưng ý nhất vào file nộp bài.")

            col_vid_player, col_kf_strip = st.columns([1.3, 0.9])

            with col_vid_player:
                st.markdown(f"#### 🎥 Trình Phát Video Trực Tiếp: `{insp_vid}.mp4`")
                v_path = video_manager.get_video_path(insp_vid)
                pts_time_cur = keyframe_loader.get_pts_time(insp_vid, insp_fidx)
                cur_min_sec = f"{int(pts_time_cur//60):02d}:{int(pts_time_cur%60):02d}"

                if v_path and v_path.exists():
                    st.video(str(v_path), start_time=int(max(0, pts_time_cur - 1.5)))
                    st.caption(f"⏱️ Mốc thời gian bắt đầu phát: `{cur_min_sec}` ({pts_time_cur:.1f}s) -> Frame `{insp_fidx}`")
                else:
                    st.warning(f"⚠️ Chưa tìm thấy file video MP4 gốc cho `{insp_vid}` trong `raw/batch_1/Videos/`. Đang hiển thị ảnh Keyframe thay thế.")
                    kf_big = keyframe_loader.get_keyframe_image(insp_vid, insp_fidx)
                    if kf_big:
                        st.image(kf_big, use_container_width=True)

                st.markdown("##### ⏱️ Bắt Khung Hình Chính Xác (Theo Phút:Giây hoặc Số Frame):")
                col_ts1, col_ts2 = st.columns([1, 1])
                with col_ts1:
                    ts_text_input = st.text_input(
                        "Nhập mốc thời gian (MM:SS hoặc Giây):",
                        value=cur_min_sec,
                        key=f"ts_input_{selected_q_name}_{insp_vid}",
                        help="Ví dụ nhập 01:24 hoặc 84.5 để tự động quy đổi ra frame chính xác"
                    )
                
                # Tính frame từ text input
                calc_frame_from_ts = insp_fidx
                try:
                    if ":" in ts_text_input:
                        m_str, s_str = ts_text_input.strip().split(":")
                        total_sec = float(m_str) * 60.0 + float(s_str)
                    else:
                        total_sec = float(ts_text_input.strip())
                    calc_frame_from_ts = int(total_sec * 25.0) # Chuẩn 25 FPS
                except Exception:
                    calc_frame_from_ts = insp_fidx

                with col_ts2:
                    custom_f_input = st.number_input(
                        "Hoặc nhập trực tiếp số Frame:",
                        min_value=0,
                        max_value=300000,
                        value=calc_frame_from_ts,
                        step=5,
                        key=f"custom_fidx_{selected_q_name}_{insp_vid}"
                    )

                # Nút Chốt làm Rank 1
                if st.button(f"📌 Chốt Đúng Mốc Này (Frame {custom_f_input}) Làm Rank #1", type="primary", use_container_width=True):
                    if rows[0][0] == insp_vid:
                        rows[0][1] = str(custom_f_input)
                    else:
                        target_row = [insp_vid, str(custom_f_input)]
                        if len(rows[0]) > 2:
                            target_row.append(rows[0][2])
                        rows.insert(0, target_row)

                    with open(target_csv_path, "w", encoding="utf-8") as f:
                        for row_item in rows:
                            f.write(",".join(row_item) + "\n")
                    sync_submission_zip(output_dir, zip_output_path)
                    st.success(f"🎉 Đã chốt Rank #1: `{insp_vid}` - Frame `{custom_f_input}` ({int((custom_f_input/25)//60):02d}:{int((custom_f_input/25)%60):02d}) & tự động cập nhật submission.zip!")
                    st.rerun()

            with col_kf_strip:
                st.markdown("#### 🎞️ Dải Phim Ngữ Cảnh & Xem Trước Frame")
                # Hiển thị ảnh trích xuất trực tiếp tại frame đang chọn
                live_preview_img = keyframe_loader.get_dense_video_frame(insp_vid, custom_f_input) or keyframe_loader.get_keyframe_image(insp_vid, custom_f_input)
                if live_preview_img:
                    st.image(live_preview_img, caption=f"📸 Khung hình thực tế tại Frame {custom_f_input} ({int((custom_f_input/25)//60):02d}:{int((custom_f_input/25)%60):02d})", use_container_width=True)

                st.markdown("**Dải Keyframe lân cận (Click để gán tức thì):**")
                surr_kfs = keyframe_loader.get_surrounding_keyframes(insp_vid, custom_f_input, count=8)
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

                if not trake_frames:
                    # Fallback lấy frame mặc định từ video đó
                    all_kfs = keyframe_loader.get_all_video_keyframes(trake_vid)
                    trake_frames = all_kfs[:len(parsed_events)] if len(all_kfs) >= len(parsed_events) else [100 * (i+1) for i in range(len(parsed_events))]

                num_events = max(len(trake_frames), len(parsed_events))
                # Đồng bộ độ dài
                while len(trake_frames) < num_events:
                    trake_frames.append(trake_frames[-1] + 100 if trake_frames else 100)

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

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        query_text = st.text_input("🔍 Nhập câu truy vấn tiếng Việt:", placeholder="ví dụ: Người thợ gốm nhào đất trên bàn xoay...")
    with col2:
        task_choice = st.selectbox("Loại bài toán:", ["Auto (Gemini Router)", "KIS (Khoảnh khắc)", "QA (Hỏi - Đáp)", "TRAKE (Chuỗi hành động)"])
    with col3:
        use_layer3 = st.toggle("Kích hoạt GPU Layer 3 (Vi sai)", value=False)

    if st.button("🚀 Bắt đầu Tìm kiếm", type="primary", use_container_width=True):
        if not query_text.strip():
            st.warning("Vui lòng nhập nội dung truy vấn!")
        else:
            with st.spinner("Đang truy xuất và rerank qua mạng nơ-ron trên GPU..."):
                t0 = time.time()
                if "KIS" in task_choice or task_choice == "Auto (Gemini Router)":
                    preds, info, lat = engine.search_kis(query_text, top_k=50, use_intra_reranker=True, use_dense_video_refiner=use_layer3)
                elif "QA" in task_choice:
                    preds, info, lat = engine.search_qa(query_text, top_k=50, use_intra_reranker=True, use_cue=True, use_multimodal=True)
                else:
                    preds, info, lat = engine.search_trake(query_text, top_k=50)

                total_ms = (time.time() - t0) * 1000

            st.success(f"✅ Hoàn tất trong {total_ms:.1f} ms | Tìm thấy {len(preds)} kết quả!")
            
            if "generated_qa_answer" in info and info["generated_qa_answer"]:
                st.info(f"💡 **Câu trả lời Visual Q&A:** `{info['generated_qa_answer']}`")

            # Grid View Top 16
            cols = st.columns(4)
            for idx, cand in enumerate(preds[:16]):
                with cols[idx % 4]:
                    vid = cand["video_id"]
                    f_idx = cand["frame_idx"]
                    sc = cand.get("score", 0.0)
                    img = keyframe_loader.get_keyframe_image(vid, f_idx)
                    
                    st.markdown(f"<span class='rank-badge-normal'>Rank #{idx+1}</span> **{vid}** : `{f_idx}`", unsafe_allow_html=True)
                    if img is not None:
                        st.image(img, use_container_width=True)
                    else:
                        st.info(f"Frame: {f_idx}")
                    st.caption(f"Score: `{sc:.4f}`")
                    if "answer" in cand and cand["answer"]:
                        st.caption(f"Ans: `{cand['answer']}`")
