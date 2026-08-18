import os
import re
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BTC_K_THRESHOLDS = [1, 5, 20, 50, 100]
QA_JUDGE_CACHE_FILE = BASE_DIR / "data" / "benchmark" / "qa_judge_cache.json"

def normalize_text(text: Optional[str]) -> str:
    """Chuẩn hóa văn bản tổng quát: chữ thường, loại bỏ khoảng trắng và dấu câu thừa."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

STOPWORDS = {"màu", "người", "là", "và", "ở", "trong", "có", "con", "cái", "chiếc", "những", "các", "khoảng", "từ", "đến", "vào", "đang", "thì", "được"}

def is_qa_match_fast(pred_answer: Optional[str], gt_answer: Optional[str]) -> bool:
    """
    Kiểm tra độ khớp nhanh (Fast Rule-based) theo ngữ nghĩa từ khóa và chuỗi con.
    """
    p = normalize_text(pred_answer)
    g = normalize_text(gt_answer)
    if not p or not g:
        return False
    if p == g or g in p or p in g:
        return True
    
    # Đối sánh tập từ khóa ngữ nghĩa cốt lõi
    p_tokens = set(p.split()) - STOPWORDS
    g_tokens = set(g.split()) - STOPWORDS
    overlap = p_tokens.intersection(g_tokens)
    
    min_len = min(len(p_tokens), len(g_tokens))
    if min_len > 0:
        if len(overlap) >= 2 or (min_len == 1 and len(overlap) == 1):
            return True
    return False

class LLMSemanticJudge:
    """
    Bộ chấm điểm ngữ nghĩa chuẩn BTC sử dụng mô hình Gemini 3.5 Flash Lite với Disk Cache.
    Đánh giá độ tương đương thông tin giữa câu trả lời của thí sinh và Ground Truth.
    """
    def __init__(self, cache_file: Path = QA_JUDGE_CACHE_FILE):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, bool] = self._load_cache()
        self._key_pool = None

    def _load_cache(self) -> Dict[str, bool]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def judge_equivalence(
        self,
        question: str,
        gt_answer: str,
        pred_answer: str
    ) -> bool:
        p = normalize_text(pred_answer)
        g = normalize_text(gt_answer)
        if not p or not g:
            return False

        # 1. Nếu khớp quy tắc nhanh thì trả về True ngay (0.001ms)
        if is_qa_match_fast(pred_answer, gt_answer):
            return True

        # 2. Kiểm tra Cache trên ổ cứng
        cache_key = f"{normalize_text(question)}|||{g}|||{p}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 3. Gọi Gemini 3.5 Flash Lite Judge
        try:
            from src.query.gemini_router import GeminiKeyPool
            from google import genai
            from google.genai import types

            if self._key_pool is None:
                self._key_pool = GeminiKeyPool()

            api_key = self._key_pool.get_next_key()
            if not api_key:
                # Fallback rule-based nếu không có key
                res = is_qa_match_fast(pred_answer, gt_answer)
                self.cache[cache_key] = res
                self._save_cache()
                return res

            client = genai.Client(api_key=api_key)
            prompt = f"""Bạn là Giám khảo AI chính thức của cuộc thi AI Challenge TP.HCM.
Nhiệm vụ: Hãy so sánh Câu trả lời của thí sinh với Đáp án chuẩn (Ground Truth) trong ngữ cảnh câu hỏi và xác định xem câu trả lời của thí sinh có ĐÚNG NGỮ NGHĨA và CÙNG THÔNG TIN CỐT LÕI hay không.

Câu hỏi: "{question}"
Đáp án chuẩn (Ground Truth): "{gt_answer}"
Câu trả lời của thí sinh (Predicted): "{pred_answer}"

Quy tắc chấm điểm:
1. Chấp nhận từ đồng nghĩa, đảo ngữ, câu trả lời đầy đủ hơn hoặc ngắn hơn nhưng đúng trọng tâm.
2. Từ chối nếu câu trả lời nói sai sự thật, sai màu sắc, sai đối tượng, hoặc mâu thuẫn trực tiếp với đáp án chuẩn.

Chỉ trả về duy nhất định dạng JSON:
{{"is_match": true}} hoặc {{"is_match": false}}
"""
            resp = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            res_json = json.loads(resp.text.strip())
            is_match = bool(res_json.get("is_match", False))
            self.cache[cache_key] = is_match
            self._save_cache()
            return is_match

        except Exception as e:
            # Fallback
            res = is_qa_match_fast(pred_answer, gt_answer)
            self.cache[cache_key] = res
            self._save_cache()
            return res

# Global instance for shared caching
_global_qa_judge = LLMSemanticJudge()

def is_qa_match(
    pred_answer: Optional[str],
    gt_answer: Optional[str],
    question: Optional[str] = "",
    use_llm_judge: bool = True
) -> bool:
    """Kiểm tra độ khớp câu trả lời QA hỗ trợ cả Fast Rule và Gemini LLM-as-a-Judge."""
    if not pred_answer or not gt_answer:
        return False
    if is_qa_match_fast(pred_answer, gt_answer):
        return True
    if use_llm_judge:
        return _global_qa_judge.judge_equivalence(question or "", gt_answer, pred_answer)
    return False

def calculate_frame_distance(f_pred: int, s_frame: int, e_frame: int) -> int:
    """Tính hàm khoảng cách d(f, [s, e]) = max(s - f, 0, f - e). d = 0 nếu nằm trong GT."""
    if s_frame <= f_pred <= e_frame:
        return 0
    if f_pred < s_frame:
        return s_frame - f_pred
    return f_pred - e_frame

def calculate_r_score(
    prediction: Dict[str, Any],
    ground_truth: Dict[str, Any],
    task_type: str,
    check_qa_answer: bool = True,
    question_text: str = "",
    use_llm_judge: bool = True
) -> Tuple[float, Dict[str, Any]]:
    """
    Tính Điểm Tương Quan (R-Score) và chi tiết chẩn đoán cho 1 dòng dự đoán.
    Returns: (r_score, diagnostic_meta)
    """
    target_video = ground_truth.get("video_id", "")
    pred_video = prediction.get("video_id", "")
    meta: Dict[str, Any] = {
        "video_match": (bool(pred_video) and pred_video == target_video),
        "frame_match": False,
        "answer_match": False,
        "frame_distance": 999999,
        "trake_event_hits": []
    }

    if not meta["video_match"]:
        return 0.0, meta

    task = task_type.lower()

    if task == "kis":
        s_frame = ground_truth.get("start_frame", 0)
        e_frame = ground_truth.get("end_frame", 0)
        f_pred = int(prediction.get("frame_idx", 0))
        dist = calculate_frame_distance(f_pred, s_frame, e_frame)
        meta["frame_distance"] = dist
        meta["frame_match"] = (dist == 0)
        return (1.0 if dist == 0 else 0.0), meta

    elif task == "qa":
        s_frame = ground_truth.get("start_frame", 0)
        e_frame = ground_truth.get("end_frame", 0)
        f_pred = int(prediction.get("frame_idx", 0))
        dist = calculate_frame_distance(f_pred, s_frame, e_frame)
        meta["frame_distance"] = dist
        meta["frame_match"] = (dist == 0)

        if not check_qa_answer:
            return (1.0 if dist == 0 else 0.0), meta

        gt_answer = ground_truth.get("answer", "")
        pred_answer = prediction.get("answer", "")
        if not gt_answer:
            meta["answer_match"] = True
            return (1.0 if dist == 0 else 0.0), meta

        ans_match = is_qa_match(pred_answer, gt_answer, question=question_text, use_llm_judge=use_llm_judge)
        meta["answer_match"] = ans_match

        # R-Score = 1.0 nếu đúng cả video, đúng frame và đúng answer
        r = 1.0 if (dist == 0 and ans_match) else 0.0
        return r, meta

    elif task == "trake":
        events = ground_truth.get("events", [])
        n_events = len(events)
        if n_events == 0:
            return 0.0, meta

        event_frames = prediction.get("event_frames", [])
        hit_count = 0
        event_status = []

        if event_frames:
            for j, ev in enumerate(events):
                s_j = ev.get("start_frame", 0)
                e_j = ev.get("end_frame", 0)
                if j < len(event_frames):
                    f_j = int(event_frames[j])
                    dist_j = calculate_frame_distance(f_j, s_j, e_j)
                    if dist_j == 0:
                        hit_count += 1
                        event_status.append(f"E{j+1}: HIT (d=0)")
                    else:
                        sign = "+" if f_j > e_j else "-"
                        event_status.append(f"E{j+1}: {sign}{dist_j}f")
                else:
                    event_status.append(f"E{j+1}: MISS")
        else:
            f_pred = int(prediction.get("frame_idx", 0))
            for j, ev in enumerate(events):
                s_j = ev.get("start_frame", 0)
                e_j = ev.get("end_frame", 0)
                dist_j = calculate_frame_distance(f_pred, s_j, e_j)
                if dist_j == 0:
                    hit_count += 1
                    event_status.append(f"E{j+1}: HIT (d=0)")
                    break
                else:
                    event_status.append(f"E{j+1}: MISS")

        meta["trake_event_hits"] = event_status
        meta["frame_match"] = (hit_count > 0)
        r_score = hit_count / n_events
        return r_score, meta

    return 0.0, meta

def evaluate_query_predictions(
    predictions: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    task_type: str,
    check_qa_answer: bool = True,
    question_text: str = "",
    use_llm_judge: bool = True
) -> Dict[str, Any]:
    """
    Tính Điểm Cuối Cùng (Final Score) và Toàn Bộ Chỉ Số Chẩn Đoán Phân Tầng cho 1 Câu Truy Vấn.
    """
    target_video = ground_truth.get("video_id", "")
    task = task_type.lower()

    # 1. Video-level Recall & Health
    unique_videos = []
    seen_v = set()
    for p in predictions:
        v = p.get("video_id", "")
        if v and v not in seen_v:
            unique_videos.append(v)
            seen_v.add(v)

    video_hit_rank = -1
    for idx, v in enumerate(unique_videos, 1):
        if v == target_video:
            video_hit_rank = idx
            break

    video_recall_at_k = {}
    for k in [1, 5, 10, 20, 50, 100]:
        video_recall_at_k[f"V-R@{k}"] = 1.0 if (video_hit_rank != -1 and video_hit_rank <= k) else 0.0

    video_mrr = (1.0 / video_hit_rank) if video_hit_rank != -1 else 0.0

    # 2. Window Size Category (Duration in frames)
    s_gt = ground_truth.get("start_frame", 0)
    e_gt = ground_truth.get("end_frame", 0)
    window_duration = max(1, e_gt - s_gt + 1)
    if window_duration <= 30:
        window_cat = "SHORT (<=30f)"
    elif window_duration <= 120:
        window_cat = "MEDIUM (31-120f)"
    else:
        window_cat = "LONG (>120f)"

    # 3. R-Score Computation across 100 submission rows
    r_scores = []
    diagnostics = []
    for p in predictions[:100]:
        r, d = calculate_r_score(
            prediction=p,
            ground_truth=ground_truth,
            task_type=task_type,
            check_qa_answer=check_qa_answer,
            question_text=question_text,
            use_llm_judge=use_llm_judge
        )
        r_scores.append(r)
        diagnostics.append(d)

    if len(r_scores) < 100:
        pad_len = 100 - len(r_scores)
        r_scores.extend([0.0] * pad_len)
        diagnostics.extend([{}] * pad_len)

    # 4. R@k tại đúng 5 mốc BTC
    r_at_k = {}
    for k in BTC_K_THRESHOLDS:
        r_at_k[f"R@{k}"] = max(r_scores[:k]) if k <= len(r_scores) else 0.0

    # Final Score là trung bình cộng của 5 mốc
    final_score = sum(r_at_k.values()) / len(BTC_K_THRESHOLDS)

    # 5. First Positive & First Perfect Ranks
    first_pos_rank = -1
    first_perfect_rank = -1
    min_frame_distance = 999999
    near_miss_5 = False
    near_miss_25 = False
    trake_event_summary = []

    for idx, (sc, d) in enumerate(zip(r_scores, diagnostics), 1):
        if sc > 0.0 and first_pos_rank == -1:
            first_pos_rank = idx
        if sc >= 1.0 and first_perfect_rank == -1:
            first_perfect_rank = idx
        
        f_dist = d.get("frame_distance", 999999)
        if f_dist < min_frame_distance:
            min_frame_distance = f_dist
        if f_dist <= 5:
            near_miss_5 = True
        if f_dist <= 25:
            near_miss_25 = True

        if task == "trake" and d.get("trake_event_hits") and not trake_event_summary:
            trake_event_summary = d["trake_event_hits"]

    # 6. Error Taxonomy Assignment
    if video_hit_rank == -1:
        error_type = "VIDEO_NOT_IN_TOP100"
    elif video_hit_rank > 20:
        error_type = "VIDEO_LATE (>Top20)"
    elif first_perfect_rank == 1:
        error_type = "PERFECT_RANK_1"
    elif first_perfect_rank != -1 and first_perfect_rank <= 5:
        error_type = "PERFECT_IN_TOP5"
    elif first_pos_rank != -1:
        error_type = "PARTIAL_HIT"
    elif min_frame_distance <= 25:
        error_type = "TEMPORAL_NEAR_MISS (<=25f)"
    elif task == "qa" and any(d.get("frame_match", False) and not d.get("answer_match", False) for d in diagnostics[:20]):
        error_type = "QA_ANSWER_MISMATCH"
    else:
        error_type = "FRAME_MISS"

    return {
        "r_at_k": r_at_k,
        "final_score": final_score,
        "first_pos_rank": first_pos_rank,
        "first_perfect_rank": first_perfect_rank,
        "video_hit_rank": video_hit_rank,
        "video_mrr": video_mrr,
        "video_recall_at_k": video_recall_at_k,
        "min_frame_distance": min_frame_distance if min_frame_distance != 999999 else -1,
        "near_miss_5": near_miss_5,
        "near_miss_25": near_miss_25,
        "window_duration": window_duration,
        "window_cat": window_cat,
        "trake_event_summary": trake_event_summary,
        "error_type": error_type,
        "r_scores": r_scores
    }

def summarize_ablation_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tổng hợp toàn bộ chỉ số vĩ mô (Macro Average), Phân nhóm Task, và Thống kê Phân loại Lỗi.
    """
    if not all_results:
        return {}

    n_total = len(all_results)
    final_scores = [r["final_score"] for r in all_results]
    macro_final_score = sum(final_scores) / n_total

    # Macro R@k
    macro_r_at_k = {}
    for k in BTC_K_THRESHOLDS:
        macro_r_at_k[f"R@{k}"] = sum(r["r_at_k"][f"R@{k}"] for r in all_results) / n_total

    # Task breakdowns
    tasks = {}
    for r in all_results:
        t = r.get("task", "KIS").upper()
        if t not in tasks:
            tasks[t] = []
        tasks[t].append(r)

    task_summary = {}
    for t_name, t_list in tasks.items():
        t_scores = [x["final_score"] for x in t_list]
        task_summary[t_name] = {
            "count": len(t_list),
            "macro_score": sum(t_scores) / len(t_list),
            "r1": sum(x["r_at_k"]["R@1"] for x in t_list) / len(t_list),
            "r5": sum(x["r_at_k"]["R@5"] for x in t_list) / len(t_list),
            "r20": sum(x["r_at_k"]["R@20"] for x in t_list) / len(t_list),
            "r50": sum(x["r_at_k"]["R@50"] for x in t_list) / len(t_list),
            "r100": sum(x["r_at_k"]["R@100"] for x in t_list) / len(t_list),
        }

    # Score Buckets
    buckets = {1.0: 0, 0.8: 0, 0.6: 0, 0.4: 0, 0.2: 0, 0.0: 0}
    for sc in final_scores:
        rounded = round(sc, 1)
        if rounded >= 0.95:
            buckets[1.0] += 1
        elif rounded >= 0.75:
            buckets[0.8] += 1
        elif rounded >= 0.55:
            buckets[0.6] += 1
        elif rounded >= 0.35:
            buckets[0.4] += 1
        elif rounded >= 0.15:
            buckets[0.2] += 1
        else:
            buckets[0.0] += 1

    # Video Recall & MRR
    v_mrrs = [r["video_mrr"] for r in all_results]
    macro_video_mrr = sum(v_mrrs) / n_total

    v_recalls = {}
    for k in [1, 5, 10, 20, 50, 100]:
        v_recalls[f"V-R@{k}"] = (sum(r["video_recall_at_k"][f"V-R@{k}"] for r in all_results) / n_total) * 100.0

    # Error Taxonomy Counts
    error_counts: Dict[str, int] = {}
    for r in all_results:
        err = r.get("error_type", "UNKNOWN")
        error_counts[err] = error_counts.get(err, 0) + 1

    return {
        "macro_final_score": macro_final_score,
        "macro_r_at_k": macro_r_at_k,
        "task_summary": task_summary,
        "score_buckets": buckets,
        "video_mrr": macro_video_mrr,
        "video_recalls": v_recalls,
        "error_counts": error_counts
    }
