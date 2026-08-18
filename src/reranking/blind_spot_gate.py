import os
import sys
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Danh sách các động từ hành động vi sai ngắn (Micro-actions)
ACTION_VERBS = {
    "rót", "đổ", "chế", "gập", "gấp", "phết", "quết", "bôi", "chạm", "bấm",
    "nhấn", "lướt", "ném", "quăng", "nhảy", "bật", "cắt", "thái", "thả",
    "cầm", "nắm", "mở", "đóng", "rửa", "vẽ", "viết", "lau", "chùi", "nhấc",
    "pour", "fold", "spread", "touch", "press", "click", "swipe", "throw",
    "jump", "cut", "slice", "drop", "hold", "open", "close", "wash", "draw"
}

class MultiSignalBlindSpotGate:
    """
    Bộ Cổng Nhận Diện Vùng Mù Đa Tín Hiệu (Multi-Signal Blind Spot Gate):
    Tự động phát hiện khi nào một hành động mục tiêu bị lọt khe giữa 2 Keyframe của BTC:
    
    1. Tín hiệu Thị giác (Context Plateau Dip): Điểm bối cảnh cao nhưng thiếu đỉnh hành động vi sai.
    2. Tín hiệu Lời thoại (Gemini Semantic ASR Grounding): Lời thoại nhắc đến hành động tại mốc thời gian không có keyframe.
    3. Tín hiệu Ngữ nghĩa NLP (Action-Verb Sensitivity): Nhận diện câu hỏi chứa động từ chuyển động ngắn.
    4. Tín hiệu Khoảng cách Keyframe (Temporal Gap Anomaly): Khoảng cách giữa 2 keyframe liền kề >= 75 frames.
    5. Tín hiệu Chuỗi TRAKE (Monotonic Sequence Crack): Phát hiện sự kiện con E_i bị hụt điểm giữa chuỗi.
    """
    def __init__(self, df_frames: pd.DataFrame, batch: str = "batch_1", gemini_api_keys: List[str] = None):
        self.df_frames = df_frames
        self.batch = batch
        self.processed_dir = BASE_DIR / "data" / batch / "processed"
        self.gemini_keys = gemini_api_keys or []
        self._key_idx = 0

        # Tải sẵn ASR transcripts mapping
        self.video_to_asr = {}
        asr_path = self.processed_dir / "transcripts.parquet"
        if asr_path.exists():
            df_asr = pd.read_parquet(asr_path)
            for v_id, grp in df_asr.groupby("video_id"):
                self.video_to_asr[v_id] = grp.sort_values("start_time")[["start_time", "end_time", "transcript"]].to_dict("records")

        # Tạo mapping video -> sorted keyframe indices & pts_times
        self.video_kfs = {}
        for v_id, grp in self.df_frames.groupby("video_id"):
            s_grp = grp.sort_values("pts_time")
            self.video_kfs[v_id] = {
                "frame_indices": s_grp["frame_idx"].to_numpy(),
                "pts_times": s_grp["pts_time"].to_numpy()
            }

    def _get_gemini_client(self) -> Optional[genai.Client]:
        if not self.gemini_keys:
            return None
        key = self.gemini_keys[self._key_idx % len(self.gemini_keys)]
        self._key_idx += 1
        return genai.Client(api_key=key)

    def has_action_verb(self, query_text: str) -> bool:
        """Kiểm tra xem câu hỏi có chứa động từ hành vi ngắn hay không."""
        words = set(re.findall(r'\b\w+\b', query_text.lower()))
        return bool(words.intersection(ACTION_VERBS))

    def check_keyframe_gap(self, video_id: str, frame_idx: int) -> Tuple[bool, int, float]:
        """
        Kiểm tra khoảng cách trống lớn nhất quanh frame_idx trong video_id.
        Trả về: (is_gap_large, max_gap_frames, max_gap_seconds)
        """
        v_data = self.video_kfs.get(video_id)
        if not v_data or len(v_data["frame_indices"]) < 2:
            return True, 100, 4.0

        kfs = v_data["frame_indices"]
        idx = np.searchsorted(kfs, frame_idx)
        idx = min(idx, len(kfs) - 1)

        f_prev = kfs[idx - 1] if idx > 0 else kfs[0]
        f_next = kfs[idx + 1] if idx + 1 < len(kfs) else kfs[-1]

        gap_left = frame_idx - f_prev
        gap_right = f_next - frame_idx
        max_gap = max(gap_left, gap_right)
        gap_sec = max_gap / 25.0

        is_large = (max_gap >= 75)
        return is_large, int(max_gap), float(gap_sec)

    def ground_asr_semantic(self, video_id: str, query_text: str) -> Optional[Dict[str, Any]]:
        """
        Dùng Gemini 2.5 Flash Lite đọc lời thoại của video và trích xuất mốc thời gian [start_time, end_time].
        Nếu mốc này không có keyframe nào đại diện -> trả về vị trí cần Seek!
        """
        asr_records = self.video_to_asr.get(video_id, [])
        if not asr_records:
            return None

        # Nối tối đa 25 đoạn hội thoại gần nhất
        transcript_text = "\n".join([
            f"[{r['start_time']:.1f}s - {r['end_time']:.1f}s]: {r['transcript']}"
            for r in asr_records[:35]
        ])

        if not transcript_text.strip():
            return None

        client = self._get_gemini_client()
        if client is None:
            return None

        prompt = f"""Bạn là chuyên gia phân tích video. Dưới đây là lời thoại kèm mốc thời gian của một video:
{transcript_text}

Câu hỏi tìm kiếm sự kiện: "{query_text}"

Hãy xác định xem có đoạn hội thoại nào đang nói trực tiếp hoặc liên quan mật thiết đến sự việc/hành động trên không.
Nếu CÓ, hãy trả về mốc thời gian bắt đầu và kết thúc (theo giây).
Nếu KHÔNG liên quan, trả về JSON với "found": false.

Định dạng JSON trả về:
{{"found": true, "start_time": 12.5, "end_time": 15.0, "reason": "nhân vật nhắc tới hành động này"}}
hoặc
{{"found": false}}
"""
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            data = json.loads(resp.text.strip())
            if data.get("found", False) and "start_time" in data:
                target_t = float(data["start_time"])
                target_f = int(target_t * 25.0)

                # Kiểm tra xem tại target_f có keyframe nào của BTC không
                v_data = self.video_kfs.get(video_id)
                if v_data:
                    times = v_data["pts_times"]
                    diffs = np.abs(times - target_t)
                    min_diff = diffs.min() if len(diffs) > 0 else 999.0
                    # Nếu không có keyframe nào trong vòng 0.8s -> Đích thực là Vùng Mù Lời Thoại!
                    if min_diff > 0.8:
                        return {
                            "grounded": True,
                            "target_pts_time": target_t,
                            "target_frame_idx": target_f,
                            "min_kf_distance_sec": float(min_diff),
                            "reason": data.get("reason", "ASR Semantic Anchor")
                        }
        except Exception:
            pass

        return None

    def evaluate_blind_spot(
        self,
        video_id: str,
        frame_idx: int,
        score: float,
        query_text: str,
        task_type: str = "kis"
    ) -> Dict[str, Any]:
        """
        Tổng hợp toàn bộ các tín hiệu để đưa ra quyết định KÍCH HOẠT LAYER 3 hay KHÔNG.
        """
        # 1. Kiểm tra Action Verb
        has_verb = self.has_action_verb(query_text)

        # 2. Kiểm tra khoảng trống Keyframe thô
        is_gap_large, max_gap, gap_sec = self.check_keyframe_gap(video_id, frame_idx)

        # 3. Tín hiệu Điểm số (Plateau Dip):
        # Nếu điểm số đã bão hòa cực cao (>= 0.75) và không có khoảng trống lớn -> An toàn tuyệt đối
        if score >= 0.75 and not is_gap_large:
            return {
                "trigger_layer3": False,
                "reason": f"High confidence score ({score:.3f}) with normal keyframe density",
                "target_frame_idx": frame_idx
            }

        # 4. Nếu có khoảng trống lớn (>= 75 frames) -> Tự động kích hoạt (Cứu ca test-kis-08)
        if is_gap_large:
            return {
                "trigger_layer3": True,
                "reason": f"Temporal Blind Spot detected: Gap = {max_gap} frames ({gap_sec:.1f}s)",
                "target_frame_idx": frame_idx,
                "window_seconds": 2.5
            }

        # 5. Nếu điểm số trung bình (0.50 - 0.70) và có động từ hành động -> Kiểm tra thêm ASR Grounding
        if has_verb and (score < 0.70):
            asr_res = self.ground_asr_semantic(video_id, query_text)
            if asr_res and asr_res.get("grounded", False):
                return {
                    "trigger_layer3": True,
                    "reason": f"ASR Semantic Grounding: {asr_res['reason']} (Target time: {asr_res['target_pts_time']}s)",
                    "target_frame_idx": asr_res["target_frame_idx"],
                    "target_pts_time": asr_res["target_pts_time"],
                    "window_seconds": 2.0
                }

        # Mặc định an toàn: Không cần chạy Layer 3
        return {
            "trigger_layer3": False,
            "reason": "Normal dense keyframe coverage without anomalies",
            "target_frame_idx": frame_idx
        }
