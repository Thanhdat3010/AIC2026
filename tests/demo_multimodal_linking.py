import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import settings

def build_unified_multimodal_view():
    print("=" * 85)
    print("🌐 HỆ THỐNG LIÊN KẾT ĐA PHƯƠNG THỨC 360° (MULTIMODAL 360 LINKAGE)")
    print("=" * 85)
    
    proc_dir = settings.directories.processed
    # 1. Nạp tất cả các nguồn dữ liệu
    frames_df = pd.read_parquet(proc_dir / "frames.parquet")
    videos_df = pd.read_parquet(proc_dir / "videos.parquet") if (proc_dir / "videos.parquet").exists() else None
    objects_df = pd.read_parquet(proc_dir / "object_summary.parquet") if (proc_dir / "object_summary.parquet").exists() else None
    ocr_df = pd.read_parquet(proc_dir / "ocr_results.parquet") if (proc_dir / "ocr_results.parquet").exists() else None
    asr_df = pd.read_parquet(proc_dir / "transcripts.parquet") if (proc_dir / "transcripts.parquet").exists() else None

    # Merge frames với objects
    full_df = frames_df.copy()
    if objects_df is not None:
        full_df = pd.merge(full_df, objects_df, on="global_id", how="left")
    
    # Merge với OCR
    if ocr_df is not None:
        full_df = pd.merge(
            full_df, 
            ocr_df[['video_id', 'keyframe_index', 'ocr_text', 'confidence']], 
            on=['video_id', 'keyframe_index'], 
            how='left'
        )
    
    # Merge với Videos metadata
    if videos_df is not None:
        full_df = pd.merge(
            full_df,
            videos_df[['video_id', 'title', 'author', 'publish_date', 'keywords']],
            on='video_id',
            how='left'
        )

    # Thử lấy một số Keyframes mẫu có đầy đủ tất cả thông tin
    sample_videos = ["L21_V001", "L21_V002", "L21_V005"]
    
    print("\n🔍 ĐỐI CHIẾU HỢP NHẤT TOÀN DIỆN CÁC TẦNG THÔNG TIN TRÊN TỪNG KEYFRAME:\n")

    for vid in sample_videos:
        sub = full_df[(full_df['video_id'] == vid) & (full_df['ocr_text'].notna())]
        if sub.empty:
            continue
            
        # Lấy 2 frame tiêu biểu
        sample_rows = sub.iloc[[len(sub)//4, len(sub)//2]]
        
        for _, row in sample_rows.iterrows():
            gid = row['global_id']
            kf_idx = row['keyframe_index']
            f_idx = row['frame_idx']
            pts = row['pts_time']
            
            # Tìm câu thoại khớp mốc thời gian pts_time này
            matched_speech = "---"
            if asr_df is not None:
                v_asr = asr_df[asr_df['video_id'] == vid]
                speech_match = v_asr[(v_asr['start_time'] <= pts) & (v_asr['end_time'] >= pts)]
                if not speech_match.empty:
                    matched_speech = speech_match.iloc[0]['transcript']
                else:
                    # Tìm câu thoại gần nhất
                    closest = v_asr.iloc[(v_asr['start_time'] - pts).abs().argsort()[:1]]
                    if not closest.empty:
                        c_row = closest.iloc[0]
                        matched_speech = f"(Gần nhất [{c_row['start_time']:.1f}s-{c_row['end_time']:.1f}s]): {c_row['transcript']}"

            print("┌" + "─" * 83 + "┐")
            print(f"│ 🎬 VIDEO: {vid} | KEYFRAME #{kf_idx:03d} (Global ID: {gid}) | Frame #{f_idx} | Thời điểm: {pts:.2f}s │")
            print("├" + "─" * 83 + "┤")
            print(f"│ 📌 [1. METADATA VIDEO]")
            print(f"│    • Tiêu đề     : {row.get('title', 'N/A')}")
            print(f"│    • Kênh / Đài  : {row.get('author', 'N/A')} | Ngày đăng: {row.get('publish_date', 'N/A')}")
            print(f"│    • Từ khóa     : {str(row.get('keywords', 'N/A'))[:65]}...")
            print("│")
            print(f"│ 👁️ [2. THỊ GIÁC / OBJECTS (YOLO + CLIP Embeddings)]")
            top_objs = row.get('top_entities', [])
            high_objs = row.get('high_conf_entities', [])
            persons = row.get('person_count', 0)
            print(f"│    • Người (Persons) : {persons} người")
            print(f"│    • Thực thể nhận diện : {list(top_objs) if isinstance(top_objs, (list, np.ndarray)) else top_objs}")
            print(f"│    • Vật thể độ tin cậy cao: {list(high_objs) if isinstance(high_objs, (list, np.ndarray)) else high_objs}")
            print("│")
            print(f"│ 🖼️ [3. CHỮ TRÊN MÀN HÌNH (CRAFT + VietOCR SOTA)]")
            print(f"│    • OCR Banners : \"{row.get('ocr_text', 'N/A')}\"")
            print("│")
            print(f"│ 🎙️ [4. GIỌNG NÓI / LỜI THOẠI ĐỒNG BỘ (VinAI PhoWhisper-large)]")
            print(f"│    • Lời thoại   : \"{matched_speech[:150]}...\"")
            print("└" + "─" * 83 + "┘\n")

if __name__ == "__main__":
    build_unified_multimodal_view()
