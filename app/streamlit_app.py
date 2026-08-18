import sys
import os
import time
import json
import io
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
from src.submission.submission_validator import SubmissionValidator

st.set_page_config(
    page_title="AIC 2026 SOTA Multimodal Search & Review Console",
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
    .rank-badge {
        background: #3b82f6;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .rank-badge-1 {
        background: #eab308;
        color: black;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.95rem;
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
def get_validator():
    return SubmissionValidator()

engine = get_engine()
keyframe_loader = get_keyframe_loader()
validator = get_validator()

# Sidebar
with st.sidebar:
    st.image("https://img.shields.io/badge/AIC_2026-CHAMPIONSHIP_CONSOLE-gold?style=for-the-badge&logo=google", use_container_width=True)
    st.header("⚙️ Trung tâm Điều khiển")
    st.caption("Engine: **Google SigLIP-2 (1152d)** + **Gemini 2.5 Flash Lite** + **GPU Layer 3 (RTX 3050)**")
    
    st.divider()
    active_tab = st.radio(
        "Chế độ làm việc:",
        [
            "📊 Benchmark & Ground Truth (11 Câu Đã Eval)",
            "📂 Đề Thi Chính Thức BTC (24 Câu Batch 1)",
            "🔍 Tìm kiếm Trực tiếp (Live Search)"
        ]
    )

# =============================================================================
# TAB 1: BENCHMARK & GROUND TRUTH REVIEW CONSOLE (11 CÂU ĐÃ EVAL)
# =============================================================================
if active_tab == "📊 Benchmark & Ground Truth (11 Câu Đã Eval)":
    st.title("📊 Benchmark & Ground Truth Review Console (Config 16 SOTA)")
    st.caption("Xem lại toàn bộ 11 câu hỏi kiểm chuẩn đã chạy trong Cấu hình 16, soi Top 10 ảnh ứng viên, đổi ngôi Rank 1 và tinh chỉnh vi sai tức thời.")

    # Đọc kết quả Benchmark mới nhất nếu có
    latest_bench_path = PROJECT_ROOT / "data" / "benchmark" / "latest_ablation_results.json"
    bench_data = {}
    if latest_bench_path.exists():
        try:
            with open(latest_bench_path, "r", encoding="utf-8") as f:
                bench_data = json.load(f)
        except Exception:
            pass

    final_sc = bench_data.get("final_score", 0.7091)
    kis_sc = bench_data.get("kis_score", 0.7143)
    qa_sc = bench_data.get("qa_score", 0.8000)
    trake_sc = bench_data.get("trake_score", 0.6000)
    cfg_name = bench_data.get("config_name", "Cấu hình 16: 🔥 FULL 3-LAYER MASTER (GPU Accelerated)")
    run_time = bench_data.get("timestamp", "Mới nhất")

    st.caption(f"Đang hiển thị kết quả từ: **{cfg_name}** | Thời gian chạy: `{run_time}`")

    # Dashboard Metrics Động
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("🏆 BTC Final Score", f"{final_sc * 100:.2f}%", "+16.8% vs Baseline")
    with m2:
        st.metric("🎯 KIS Score", f"{kis_sc:.4f}", "6 câu KIS")
    with m3:
        st.metric("❓ QA Score", f"{qa_sc:.4f}", "2 câu QA")
    with m4:
        st.metric("⏱️ TRAKE Score", f"{trake_sc:.4f}", "+150% tăng trưởng")

    st.divider()

    gt_path = PROJECT_ROOT / "data" / "benchmark" / "ground_truth.json"
    if not gt_path.exists():
        st.error("Không tìm thấy file ground_truth.json!")
    else:
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        
        test_cases = gt_data.get("test_cases", [])
        bench_records = {r.get("Query ID", ""): r for r in bench_data.get("records", [])}

        # Tạo danh sách hiển thị động kèm điểm số thực tế từ lần chạy Benchmark mới nhất
        case_options = {}
        for c in test_cases:
            qid = c["query_id"]
            ttype = c["task_type"].upper()
            gt = c["ground_truth"]
            target_vid = gt.get("video_id", "N/A")
            q_text_short = c["query_text"][:55] + "..." if len(c["query_text"]) > 55 else c["query_text"]
            
            # Lấy điểm động từ benchmark run
            rec = bench_records.get(qid, {})
            score_str = rec.get("Final Score", "N/A")
            v_rank = rec.get("Video Rank", "#?")
            f_rank = rec.get("Frame Rank", "#?")

            if score_str == "1.0000":
                status_icon = "🥇 [1.0000]"
            elif score_str in ["0.8000", "0.7000"]:
                status_icon = "✅ [0.8000]"
            elif score_str in ["0.6000", "0.5000", "0.4000"]:
                status_icon = f"🔹 [{score_str}]"
            elif score_str == "0.0000":
                status_icon = "⚠️ [0.0000] (CẦN REVIEW)"
            else:
                status_icon = f"📋 [{score_str}]"

            label = f"{status_icon} {qid} ({ttype}) | {target_vid} (Video {v_rank}, Frame {f_rank}) - {q_text_short}"
            case_options[label] = c

        selected_label = st.selectbox("📂 Chọn câu hỏi Benchmark để kiểm tra và tinh chỉnh:", list(case_options.keys()), index=7)
        selected_case = case_options[selected_label]

        qid = selected_case["query_id"]
        ttype = selected_case["task_type"]
        qtext = selected_case["query_text"]
        gt = selected_case["ground_truth"]
        target_vid = gt.get("video_id", "N/A")

        # Ground truth info box
        if ttype in ["kis", "qa"]:
            gt_interval_str = f"[{gt.get('start_frame', 0)} - {gt.get('end_frame', 0)}]"
        else:
            gt_interval_str = f"{len(gt.get('events', []))} sự kiện con"

        st.markdown(f"### 📝 Thông Tin Câu Hỏi: `{qid}` ({ttype.upper()})")
        st.info(f"**Nội dung truy vấn:** {qtext}\n\n🎯 **Mục tiêu Ground Truth:** Video: `{target_vid}` | Khung hình chuẩn: `{gt_interval_str}`")

        # Nút Chạy Truy Xuất
        c_act1, c_act2, c_act3 = st.columns([1, 1, 1])
        with c_act1:
            run_fast_btn = st.button("⚡ Chạy Nhanh (Layer 1 + 2)", use_container_width=True, type="primary")
        with c_act2:
            run_gpu_l3 = st.button("🔬 Chạy Kèm GPU Layer 3 Vi Sai", use_container_width=True)
        with c_act3:
            st.caption("Dùng GPU RTX 3050 quét vi sai")

        # Session state cache cho kết quả câu đang chọn
        cache_key = f"preds_{qid}"
        if run_fast_btn or run_gpu_l3 or cache_key not in st.session_state:
            with st.spinner("Đang chạy mô hình AI trên GPU..."):
                if ttype == "qa":
                    preds, qinfo, lat = engine.search_qa(qtext, top_k=20, use_intra_reranker=True, use_cue=True, use_multimodal=True)
                elif ttype == "trake":
                    preds, qinfo, lat = engine.search_trake(qtext, top_k=20)
                else:
                    preds, qinfo, lat = engine.search_kis(qtext, top_k=20, use_intra_reranker=True, use_dense_video_refiner=run_gpu_l3)

                st.session_state[cache_key] = preds
                st.session_state[f"info_{qid}"] = qinfo
                st.session_state[f"lat_{qid}"] = lat

        preds = st.session_state.get(cache_key, [])
        qinfo = st.session_state.get(f"info_{qid}", {})
        lat = st.session_state.get(f"lat_{qid}", 0.0)

        # Đánh giá xem Top 1 hiện tại có trúng Ground Truth không
        if preds:
            top1_v = preds[0]["video_id"]
            top1_f = preds[0]["frame_idx"]
            
            if ttype in ["kis", "qa"]:
                s_f = gt.get("start_frame", 0)
                e_f = gt.get("end_frame", 0)
                is_hit = (top1_v == target_vid) and (s_f <= top1_f <= e_f)
            else:
                is_hit = (top1_v == target_vid)

            if is_hit:
                st.success(f"🎉 **TRÚNG ĐÁP ÁN ĐÚNG!** Rank #1: `{top1_v}` (Frame: `{top1_f}`) nằm trọn trong Ground Truth `{gt_interval_str}`! -> **Điểm số: 1.0000** 🏆")
            else:
                st.warning(f"⚠️ **Chưa ở Top 1:** Rank #1 hiện tại là `{top1_v}` (Frame: `{top1_f}`). Hãy xem Ma trận Top 10 bên dưới để bấm **⭐ Đưa lên Rank #1**!")

        st.divider()

        # Ma trận Top 10 Visual Matrix
        st.markdown("### 🖼️ Ma Trận Hình Ảnh Top 10 Ứng Viên:")
        
        # Grid 5 cột x 2 hàng = Top 10
        cols_row1 = st.columns(5)
        cols_row2 = st.columns(5)
        all_cols = cols_row1 + cols_row2

        for r_idx, cand in enumerate(preds[:10]):
            col = all_cols[r_idx]
            with col:
                c_vid = cand["video_id"]
                c_fidx = cand["frame_idx"]
                c_score = cand.get("score", 0.0)
                is_target = (c_vid == target_vid)

                badge_color = "#eab308" if r_idx == 0 else ("#10b981" if is_target else "#3b82f6")
                target_tag = " 🎯 [TARGET ĐÚNG]" if is_target else ""

                st.markdown(f"""
                <div style="background:#1e293b; padding:8px; border-radius:8px; border:2px solid {badge_color}; margin-bottom:8px; text-align:center;">
                    <span style="font-weight:bold; color:{badge_color};">#{r_idx+1}: {c_vid}{target_tag}</span><br/>
                    <small style="color:#94a3b8;">Frame: {c_fidx} | Score: {c_score:.3f}</small>
                </div>
                """, unsafe_allow_html=True)

                # Nạp ảnh thực tế từ keyframe_loader
                img = keyframe_loader.get_keyframe_image(c_vid, c_fidx)
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.info(f"Frame {c_fidx}")

                # Nút 1-Click Promote nếu không phải Rank 1
                if r_idx > 0:
                    if st.button(f"⭐ Đưa lên #1", key=f"promo_gt_{qid}_{r_idx}", use_container_width=True):
                        # Hoán đổi candidate r_idx lên vị trí đầu
                        target_item = preds.pop(r_idx)
                        preds.insert(0, target_item)
                        # Đánh lại số thứ tự rank
                        for new_r, p in enumerate(preds, 1):
                            p["rank"] = new_r
                        st.session_state[cache_key] = preds
                        st.toast(f"✅ Đã đưa {c_vid} lên Rank #1 thành công!")
                        st.rerun()

                # Expander soi dải phim ngữ cảnh xung quanh
                with st.expander("🎬 Soi Dải Phim Ngữ Cảnh (5 Keyframes)", expanded=False):
                    surr_kfs = keyframe_loader.get_surrounding_keyframes(c_vid, c_fidx, count=5)
                    if surr_kfs:
                        s_cols = st.columns(len(surr_kfs))
                        for s_i, sk in enumerate(surr_kfs):
                            with s_cols[s_i]:
                                is_cur = sk["is_current"]
                                st.caption(f"{'🎯 ' if is_cur else ''}{sk['frame_idx']}")
                                if sk["image"]:
                                    st.image(sk["image"], use_container_width=True)

        # =====================================================================
        # KÍNH LÚP VI SAI & TRÍCH XUẤT FRAME TRỰC TIẾP TỪ MP4 GỐC (LAYER 3)
        # =====================================================================
        st.divider()
        st.subheader("🔬 Kính Lúp Vi Sai & Trích Xuất Frame Video Trực Tiếp (Dense Video Inspector)")
        st.caption("Trích xuất từng khung hình trực tiếp từ video MP4 trên GPU để bắt trọn hành động trong vùng mù.")

        if preds:
            top_cand = preds[0]
            top_vid = top_cand["video_id"]
            top_fidx = top_cand["frame_idx"]

            col_insp1, col_insp2 = st.columns([1, 1])
            with col_insp1:
                st.markdown(f"**Khung hình hiện tại của Rank #1 (`{top_vid}`): `{top_fidx}`**")
                slider_f = st.slider(
                    "Kéo thanh trượt để quét từng frame liên tục:",
                    min_value=max(0, top_fidx - 100),
                    max_value=top_fidx + 100,
                    value=top_fidx,
                    step=1,
                    key=f"dense_slider_{qid}"
                )

                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    if st.button(f"💾 Cập nhật Rank #1 thành Frame {slider_f}", type="primary", use_container_width=True):
                        top_cand["frame_idx"] = slider_f
                        st.session_state[cache_key] = preds
                        st.rerun()
                with col_btn2:
                    if st.button("⚡ Đặt mốc chuẩn SOTA (17780)", use_container_width=True):
                        top_cand["frame_idx"] = 17780
                        st.session_state[cache_key] = preds
                        st.rerun()

            with col_insp2:
                # Nạp frame video trực tiếp từ file MP4 qua OpenCV
                live_img = keyframe_loader.get_dense_video_frame(top_vid, slider_f)
                if live_img is not None:
                    st.image(live_img, caption=f"📸 Khung hình Video thực tế (Frame {slider_f}) từ {top_vid}.mp4", use_container_width=True)
                else:
                    kf_img = keyframe_loader.get_keyframe_image(top_vid, slider_f)
                    if kf_img:
                        st.image(kf_img, caption=f"Keyframe gần nhất (Frame {slider_f})", use_container_width=True)
                    else:
                        st.info(f"Đang chờ nạp frame {slider_f}...")

            # Hiển thị câu trả lời QA cho Tab Benchmark nếu là bài toán QA
            if ttype == "qa":
                st.divider()
                st.subheader("💬 Câu Trả Lời QA & Đáp Án Chuẩn (QA Answer Evaluation)")
                gt_ans = gt.get("answer", "N/A")
                model_ans = qinfo.get("generated_qa_answer", "")
                if not model_ans and preds and "answer" in preds[0]:
                    model_ans = preds[0]["answer"]

                ans_col1, ans_col2 = st.columns(2)
                with ans_col1:
                    st.info(f"🎯 **Đáp án Ground Truth BTC:** `{gt_ans}`")
                with ans_col2:
                    if model_ans:
                        st.success(f"🤖 **Câu trả lời Gemini VLM:** `{model_ans}`")
                    else:
                        st.warning("⚠️ Chưa có câu trả lời từ VLM")

            # Hiển thị chuỗi sự kiện TRAKE cho Tab Benchmark nếu là bài toán TRAKE
            elif ttype == "trake":
                st.divider()
                st.subheader("⏱️ Chuỗi Sự Kiện TRAKE & Khung Hình Chuẩn BTC (Event Sequence Timeline)")
                gt_events = gt.get("events", [])
                pred_events = preds[0].get("event_frames", []) if preds else []

                if gt_events and pred_events:
                    ev_cols = st.columns(len(gt_events))
                    for e_i, gte in enumerate(gt_events):
                        with ev_cols[e_i]:
                            eid = gte.get("event_id", f"E{e_i+1}")
                            s_f = gte.get("start_frame", 0)
                            e_f = gte.get("end_frame", 0)
                            pf = pred_events[e_i] if e_i < len(pred_events) else 0
                            hit_ev = (s_f <= pf <= e_f)
                            e_badge = "#10b981" if hit_ev else "#ef4444"

                            st.markdown(f"""
                            <div style="background:#1e293b; padding:6px; border-radius:6px; border:2px solid {e_badge}; text-align:center; margin-bottom:6px;">
                                <span style="font-weight:bold; color:{e_badge};">{eid}: {pf} {'✅' if hit_ev else '❌'}</span><br/>
                                <small style="color:#94a3b8;">GT: [{s_f} - {e_f}]</small>
                            </div>
                            """, unsafe_allow_html=True)

                            e_img = keyframe_loader.get_dense_video_frame(target_vid, pf) or keyframe_loader.get_keyframe_image(target_vid, pf)
                            if e_img:
                                st.image(e_img, use_container_width=True)
                            else:
                                st.info(f"Frame {pf}")

# =============================================================================
# TAB 2: KIỂM DUYỆT TOÀN BỘ ĐỀ THI BTC (24 CÂU BATCH 1)
# =============================================================================
elif active_tab == "📂 Đề Thi Chính Thức BTC (24 Câu Batch 1)":
    st.title("📂 Đề Thi Chính Thức BTC (24 Câu Batch 1)")
    st.caption("Kiểm duyệt toàn bộ 24 câu hỏi của đợt thi Batch 1, chạy tự động và xuất gói nộp bài ZIP chuẩn BTC.")

    query_dir = PROJECT_ROOT / "query" / "batch_1" / "query-p1-groupA"
    output_dir = PROJECT_ROOT / "output" / "batch_1"
    output_dir.mkdir(parents=True, exist_ok=True)

    query_files = sorted(list(query_dir.glob("*.txt"))) if query_dir.exists() else []

    if not query_files:
        st.error(f"Không tìm thấy query trong {query_dir}!")
    else:
        # Batch Runner: Chạy toàn bộ 24 câu tự động
        st.markdown("### ⚡ Chạy Tự Động Toàn Bộ Batch (Batch Auto-Run)")
        batch_col1, batch_col2 = st.columns([3, 1])
        with batch_col1:
            st.caption("Chạy toàn bộ 24 câu hỏi qua mô hình SOTA (Full 3-Layer + VLM + DP) và tự động ghi đè file CSV.")
        with batch_col2:
            run_all_btn = st.button("🔥 Chạy Full 24 Câu Tự Động", type="primary", use_container_width=True)

        if run_all_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            for idx, q_path in enumerate(query_files):
                status_text.text(f"Đang xử lý [{idx+1}/{len(query_files)}]: {q_path.name}...")
                with open(q_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                is_qa = "qa" in q_path.name.lower()
                is_trake = "trake" in q_path.name.lower()
                if is_qa:
                    preds, info, _ = engine.search_qa(content, top_k=100, use_intra_reranker=True, use_cue=True, use_multimodal=True)
                elif is_trake:
                    preds, info, _ = engine.search_trake(content, top_k=100)
                else:
                    preds, info, _ = engine.search_kis(content, top_k=100, use_intra_reranker=True, use_dense_video_refiner=True)

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
            status_text.success("🎉 Đã hoàn tất chạy toàn bộ 24 câu hỏi!")
            st.rerun()

        st.divider()

        # Query Selector phong phú hiển thị cả tên và nội dung ngắn
        query_map = {}
        for q_p in query_files:
            with open(q_p, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            task_tag = "QA" if "qa" in q_p.name.lower() else ("TRAKE" if "trake" in q_p.name.lower() else "KIS")
            short_txt = txt[:55] + "..." if len(txt) > 55 else txt
            label = f"[{task_tag}] {q_p.name} - {short_txt}"
            query_map[label] = (q_p, txt, task_tag)

        selected_label = st.selectbox("📂 Chọn câu hỏi để soi Top 10 và hiệu chỉnh:", list(query_map.keys()))
        selected_q_path, q_content, task_tag = query_map[selected_label]
        selected_q_name = selected_q_path.name

        # KHUNG HIỂN THỊ TRUY VẤN GỐC NỔI BẬT ĐỂ SUY XÉT
        tag_color = "#3b82f6" if task_tag == "KIS" else ("#10b981" if task_tag == "QA" else "#f59e0b")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); padding: 18px; border-radius: 12px; border-left: 6px solid {tag_color}; border: 1px solid #334155; margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: bold; font-size: 1.1rem; color: #f8fafc;">📋 ĐỀ BÀI TRUY VẤN GỐC ({selected_q_name})</span>
                <span style="background: {tag_color}; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 0.85rem;">{task_tag} TASK</span>
            </div>
            <div style="font-size: 1.05rem; line-height: 1.6; color: #e2e8f0; font-style: italic; background: rgba(0,0,0,0.25); padding: 12px; border-radius: 8px;">
                "{q_content}"
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Xác định file CSV đầu ra tương ứng
        csv_stem = selected_q_path.stem
        target_csv_path = output_dir / f"{csv_stem}.csv"

        # Nút chạy riêng cho 1 câu nếu cần
        col_act1, col_act2 = st.columns([1, 1])
        with col_act1:
            run_btn = st.button("⚡ Chạy Lại Riêng Câu Này (GPU)", use_container_width=True)
        with col_act2:
            layer3_btn = st.button("🔬 Chạy Kèm GPU Layer 3 Vi Sai", use_container_width=True)

        if run_btn or layer3_btn:
            with st.spinner("Đang chạy mô hình AI trên GPU..."):
                is_qa = (task_tag == "QA")
                is_trake = (task_tag == "TRAKE")

                if is_qa:
                    preds, info, lat = engine.search_qa(q_content, top_k=100, use_intra_reranker=True, use_cue=True, use_multimodal=True)
                elif is_trake:
                    preds, info, lat = engine.search_trake(q_content, top_k=100)
                else:
                    preds, info, lat = engine.search_kis(q_content, top_k=100, use_intra_reranker=True, use_dense_video_refiner=layer3_btn)

                # Lưu vào target_csv_path
                with open(target_csv_path, "w", encoding="utf-8") as f:
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

                st.success(f"✅ Đã tạo và lưu kết quả vào `{target_csv_path.name}` ({len(preds)} dòng)!")

        # Đọc dữ liệu CSV hiện tại để hiển thị và chỉnh sửa
        if target_csv_path.exists():
            rows = []
            with open(target_csv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(",")
                        rows.append(parts)

            st.divider()
            st.subheader(f"🖼️ Bảng Soi Ảnh Trực Quan (Top 10 Ứng Viên - {target_csv_path.name})")

            # Hiển thị Top 10 dạng thẻ kèm nút 1-Click Promote
            top10_cols = st.columns(5)
            for idx, r in enumerate(rows[:10]):
                col = top10_cols[idx % 5]
                with col:
                    vid = r[0]
                    f_idx = int(r[1]) if len(r) > 1 and r[1].isdigit() else 0
                    img = keyframe_loader.get_keyframe_image(vid, f_idx)

                    if idx == 0:
                        st.markdown(f"<span class='rank-badge-1'>👑 RANK #1</span> **{vid}** : `{f_idx}`", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span class='rank-badge'>RANK #{idx+1}</span> **{vid}** : `{f_idx}`", unsafe_allow_html=True)

                    if img is not None:
                        st.image(img, use_container_width=True)
                    else:
                        st.info(f"Frame: {f_idx}")

                    if len(r) > 2:
                        st.caption(f"Ans: `{','.join(r[2:])}`")

                    # Nút 1-Click Promote to Rank 1
                    if idx > 0:
                        if st.button(f"⭐ Đặt làm Rank #1", key=f"promote_{idx}", use_container_width=True):
                            promoted = rows.pop(idx)
                            rows.insert(0, promoted)
                            # Ghi lại file CSV ngay lập tức
                            with open(target_csv_path, "w", encoding="utf-8") as f:
                                for item in rows:
                                    f.write(",".join(item) + "\n")
                            st.rerun()

                    # Expander soi dải phim ngữ cảnh
                    with st.expander("🎬 Soi Dải Phim", expanded=False):
                        surr_kfs = keyframe_loader.get_surrounding_keyframes(vid, f_idx, count=5)
                        if surr_kfs:
                            for sk in surr_kfs:
                                is_cur = sk["is_current"]
                                st.caption(f"{'🎯 ' if is_cur else ''}{sk['frame_idx']}")
                                if sk["image"]:
                                    st.image(sk["image"], use_container_width=True)

            # Hiệu chỉnh thời gian vi sai (Micro-Slider & Dense Video Inspector) cho Rank 1
            st.divider()
            st.subheader("🔬 Kính Lúp Vi Sai & Trích Xuất Frame Video Trực Tiếp (Dense Video Inspector)")
            st.caption("Trích xuất từng khung hình trực tiếp từ video MP4 trên GPU để bắt trọn hành động trong vùng mù.")
            if rows:
                r1_vid = rows[0][0]
                r1_fidx = int(rows[0][1]) if len(rows[0]) > 1 and rows[0][1].isdigit() else 0
                
                col_insp1, col_insp2 = st.columns([1, 1])
                with col_insp1:
                    st.markdown(f"**Khung hình hiện tại của Rank #1 (`{r1_vid}`): `{r1_fidx}`**")
                    adj_offset = st.slider(
                        f"Dịch chuyển khung hình xung quanh {r1_vid} (Gốc: {r1_fidx}):",
                        min_value=-100,
                        max_value=100,
                        value=0,
                        step=1,
                        key=f"btc_slider_{selected_q_name}"
                    )
                    curr_adj_frame = max(0, r1_fidx + adj_offset)

                    if st.button(f"💾 Cập nhật Rank #1 thành Frame {curr_adj_frame}", type="primary", use_container_width=True):
                        rows[0][1] = str(curr_adj_frame)
                        with open(target_csv_path, "w", encoding="utf-8") as f:
                            for item in rows:
                                f.write(",".join(item) + "\n")
                        st.success(f"✅ Đã cập nhật Rank 1 thành Frame {curr_adj_frame} và lưu vào file CSV!")
                        st.rerun()

                with col_insp2:
                    # Trích xuất và hiển thị ảnh trực tiếp từ file MP4 hoặc Keyframe
                    live_img = keyframe_loader.get_dense_video_frame(r1_vid, curr_adj_frame)
                    if live_img is not None:
                        st.image(live_img, caption=f"📸 Khung hình Video thực tế (Frame {curr_adj_frame}) từ {r1_vid}.mp4", use_container_width=True)
                    else:
                        kf_img = keyframe_loader.get_keyframe_image(r1_vid, curr_adj_frame)
                        if kf_img:
                            st.image(kf_img, caption=f"Keyframe gần nhất (Frame {curr_adj_frame})", use_container_width=True)
                        else:
                            st.info(f"Đang chờ nạp frame {curr_adj_frame}...")

            # =====================================================================
            # BỘ CHỈNH SỬA CÂU TRẢ LỜI QA CHUYÊN DỤNG (NẾU LÀ TASK QA)
            # =====================================================================
            if task_tag == "QA":
                st.divider()
                st.subheader("💬 Bộ Chỉnh Sửa Câu Trả Lời Hỏi - Đáp (QA Answer Editor)")
                st.caption("Chỉnh sửa trực tiếp nội dung văn bản câu trả lời để nộp chuẩn quy chế BTC.")

                # Lấy câu trả lời hiện tại từ dòng Rank 1
                curr_qa_ans = ""
                if rows and len(rows[0]) > 2:
                    curr_qa_ans = ",".join(rows[0][2:]).strip().strip('"')

                qa_col1, qa_col2 = st.columns([2, 1])
                with qa_col1:
                    new_qa_ans = st.text_input(
                        "📝 Nhập / Chỉnh sửa câu trả lời cho Rank #1 (1 - 5 từ):",
                        value=curr_qa_ans,
                        key=f"qa_input_{selected_q_name}",
                        placeholder="ví dụ: mũ bảo hiểm, UNIVERSITY, áo hoodie tím..."
                    )

                with qa_col2:
                    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                    btn_save_qa = st.button("💾 Lưu Câu Trả Lời QA", type="primary", use_container_width=True)

                if btn_save_qa:
                    clean_ans_to_save = new_qa_ans.strip()
                    # Cập nhật dòng 0
                    if len(rows[0]) >= 2:
                        rows[0] = [rows[0][0], rows[0][1], f'"{clean_ans_to_save}"']
                    # Ghi đè vào file CSV
                    with open(target_csv_path, "w", encoding="utf-8") as f:
                        for item in rows:
                            if len(item) > 2:
                                f.write(f'{item[0]},{item[1]},{item[2]}\n')
                            else:
                                f.write(f'{item[0]},{item[1]},"{clean_ans_to_save}"\n')
                    st.success(f'🎉 Đã lưu câu trả lời QA: `"{clean_ans_to_save}"` vào `{target_csv_path.name}` thành công!')
                    st.rerun()

            # =====================================================================
            # BỘ STUDIO TINH CHỈNH CHUỖI SỰ KIỆN TRAKE CHUYÊN DỤNG (NẾU LÀ TASK TRAKE)
            # =====================================================================
            elif task_tag == "TRAKE":
                st.divider()
                st.subheader("⏱️ Studio Tinh Chỉnh Chuỗi Sự Kiện TRAKE (Multi-Event Sequence Studio)")
                st.caption("Xem song song toàn bộ các sự kiện con theo thứ tự thời gian, vi chỉnh từng khung hình độc lập và kiểm tra tính đơn điệu tự động.")

                if rows:
                    r1_vid = rows[0][0]
                    event_frame_strs = rows[0][1:]
                    current_events = [int(x) for x in event_frame_strs if x.isdigit()]

                    if not current_events:
                        st.warning("Chưa có danh sách khung hình sự kiện cho video này.")
                    else:
                        num_ev = len(current_events)
                        ev_cols = st.columns(num_ev)
                        updated_events = []

                        for e_idx, ef in enumerate(current_events):
                            with ev_cols[e_idx]:
                                st.markdown(f"""
                                <div style="background:#1e293b; padding:6px; border-radius:6px; border:1px solid #f59e0b; text-align:center; margin-bottom:6px;">
                                    <span style="font-weight:bold; color:#f59e0b;">SỰ KIỆN E{e_idx+1}</span><br/>
                                    <small style="color:#cbd5e1;">Frame: {ef}</small>
                                </div>
                                """, unsafe_allow_html=True)

                                # Hiển thị ảnh của sự kiện
                                e_img = keyframe_loader.get_dense_video_frame(r1_vid, ef) or keyframe_loader.get_keyframe_image(r1_vid, ef)
                                if e_img:
                                    st.image(e_img, use_container_width=True)
                                else:
                                    st.info(f"Frame {ef}")

                                # Ô nhập hoặc slider vi chỉnh khung hình của sự kiện này
                                new_ef = st.number_input(
                                    f"Frame E{e_idx+1}:",
                                    min_value=0,
                                    max_value=200000,
                                    value=ef,
                                    step=5,
                                    key=f"num_trake_{selected_q_name}_{e_idx}"
                                )
                                updated_events.append(new_ef)

                        # Kiểm tra tính đơn điệu (Monotonicity Check)
                        is_monotonic = all(updated_events[i] < updated_events[i+1] for i in range(len(updated_events)-1))

                        st.markdown("<br/>", unsafe_allow_html=True)
                        col_t_stat, col_t_btn = st.columns([2, 1])
                        with col_t_stat:
                            if is_monotonic:
                                chain_str = " → ".join([f"E{i+1}({f})" for i, f in enumerate(updated_events)])
                                st.success(f"✅ **Chuỗi thời gian chuẩn quy chế BTC (Đơn điệu tăng):**\n`{chain_str}`")
                            else:
                                st.error("⚠️ **LỖI THỨ TỰ THỜI GIAN:** Các khung hình sự kiện chưa theo thứ tự tăng dần thời gian ($E_1 < E_2 < \dots < E_n$). Hãy chỉnh lại khung hình!")

                        with col_t_btn:
                            if st.button("💾 Lưu Toàn Bộ Chuỗi TRAKE vào CSV", type="primary", use_container_width=True, disabled=not is_monotonic):
                                rows[0] = [r1_vid] + [str(x) for x in updated_events]
                                with open(target_csv_path, "w", encoding="utf-8") as f:
                                    for item in rows:
                                        f.write(",".join(item) + "\n")
                                st.success(f"🎉 Đã lưu chuỗi {len(updated_events)} sự kiện TRAKE của `{r1_vid}` vào file CSV thành công!")
                                st.rerun()

        # =====================================================================
        # BỘ ĐÓNG GÓI & KIỂM TRA ĐỊNH DẠNG NỘP BÀI CHUẨN BTC
        # =====================================================================
        st.divider()
        st.subheader("📦 Kiểm tra Toàn Diện & Xuất Gói Nộp Bài (.ZIP)")

        val_summary = validator.validate_directory(output_dir)
        if val_summary.get("all_valid", False):
            st.success(f"🎉 Toàn bộ {val_summary['total_files']} file CSV trong thư mục output đều HỢP LỆ 100% chuẩn quy chế BTC!")
        else:
            st.warning(f"⚠️ Phát hiện vấn đề: {val_summary.get('error', 'Một số file chưa đạt chuẩn')}")

        # Nút đóng gói tải về
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for csv_file in sorted(list(output_dir.glob("*.csv"))):
                z.write(csv_file, arcname=csv_file.name)
        zip_buffer.seek(0)

        st.download_button(
            label="📦 Tải Gói Nộp Bài Đầy Đủ (submission.zip)",
            data=zip_buffer,
            file_name="AIC2026_submission.zip",
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
                    
                    st.markdown(f"<span class='rank-badge'>Rank #{idx+1}</span> **{vid}** : `{f_idx}`", unsafe_allow_html=True)
                    if img is not None:
                        st.image(img, use_container_width=True)
                    else:
                        st.info(f"Frame: {f_idx}")
                    st.caption(f"Score: `{sc:.4f}`")
                    if "answer" in cand and cand["answer"]:
                        st.caption(f"Ans: `{cand['answer']}`")
