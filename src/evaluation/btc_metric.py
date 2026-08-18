import re
from typing import List, Dict, Any, Optional

BTC_K_THRESHOLDS = [1, 5, 20, 50, 100]

def normalize_text(text: Optional[str]) -> str:
    """Chuẩn hóa văn bản tổng quát: chữ thường, loại bỏ khoảng trắng và dấu câu thừa."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

STOPWORDS = {"màu", "người", "là", "và", "ở", "trong", "có", "con", "cái", "chiếc", "những", "các", "khoảng", "từ", "đến", "vào", "đang", "thì", "được"}

def is_qa_match(pred_answer: Optional[str], gt_answer: Optional[str]) -> bool:
    """
    Kiểm tra độ khớp câu trả lời Visual QA theo ngữ nghĩa:
    Hỗ trợ đối sánh trực tiếp, chứa chuỗi con, hoặc đồng quy tập từ khóa cốt lõi (loại trừ hư từ).
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

def calculate_r_score(
    prediction: Dict[str, Any],
    ground_truth: Dict[str, Any],
    task_type: str,
    check_qa_answer: bool = True
) -> float:
    """
    Tính Điểm Tương Quan (R-Score) cho 1 câu trả lời theo đúng Mục 2.1 của BTC.
    
    1. Textual KIS: R-Score = I(v_i == GT_v and id_i in [s, e])
    2. QA:          R-Score = I(v_i == GT_v and id_i in [s, e] and a_i == GT_a)
    3. TRAKE:       R-Score = (1/N) * sum(I(id_{i,j} in [s_j, e_j])) if v_i == GT_v else 0.0
    """
    target_video = ground_truth.get("video_id", "")
    pred_video = prediction.get("video_id", "")

    if not pred_video or pred_video != target_video:
        return 0.0

    task = task_type.lower()

    if task == "kis":
        s_frame = ground_truth.get("start_frame", 0)
        e_frame = ground_truth.get("end_frame", 0)
        f_pred = prediction.get("frame_idx", 0)
        return 1.0 if s_frame <= f_pred <= e_frame else 0.0

    elif task == "qa":
        s_frame = ground_truth.get("start_frame", 0)
        e_frame = ground_truth.get("end_frame", 0)
        f_pred = prediction.get("frame_idx", 0)
        frame_match = (s_frame <= f_pred <= e_frame)

        if not frame_match:
            return 0.0

        if not check_qa_answer:
            return 1.0

        gt_answer = ground_truth.get("answer", "")
        pred_answer = prediction.get("answer", "")
        # Nếu GT không yêu cầu answer hoặc chưa có trong GT, chỉ xét frame
        if not gt_answer:
            return 1.0
        return 1.0 if is_qa_match(pred_answer, gt_answer) else 0.0

    elif task == "trake":
        events = ground_truth.get("events", [])
        n_events = len(events)
        if n_events == 0:
            return 0.0

        event_frames = prediction.get("event_frames", [])
        hit_count = 0

        if event_frames:
            for j, ev in enumerate(events):
                if j < len(event_frames):
                    s_j = ev.get("start_frame", 0)
                    e_j = ev.get("end_frame", 0)
                    if s_j <= event_frames[j] <= e_j:
                        hit_count += 1
        else:
            # Fallback nếu model chỉ output 1 frame_idx duy nhất
            f_pred = prediction.get("frame_idx", 0)
            for ev in events:
                s_j = ev.get("start_frame", 0)
                e_j = ev.get("end_frame", 0)
                if s_j <= f_pred <= e_j:
                    hit_count += 1
                    break

        return hit_count / n_events

    return 0.0

def evaluate_query_predictions(
    predictions: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    task_type: str,
    check_qa_answer: bool = True
) -> Dict[str, Any]:
    """
    Tính Điểm Cuối Cùng (Final Score) cho 1 câu truy vấn theo Mục 2.2 của BTC.
    
    1. Top-k R-Score (R@k): R@k = max_{1 <= i <= k} { R-Score(r_i) } với k in {1, 5, 20, 50, 100}
    2. Final Score:        (1/5) * sum_{k in {1, 5, 20, 50, 100}} R@k
    """
    target_video = ground_truth.get("video_id", "")

    # Đánh giá Video-level recall (chẩn đoán)
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

    # Tính R-Score cho từng dòng nộp (tối đa 100 câu trả lời)
    r_scores = []
    for p in predictions[:100]:
        r_scores.append(calculate_r_score(p, ground_truth, task_type, check_qa_answer=check_qa_answer))

    if len(r_scores) < 100:
        r_scores.extend([0.0] * (100 - len(r_scores)))

    # Tính R@k tại đúng 5 mốc BTC
    r_at_k = {}
    for k in BTC_K_THRESHOLDS:
        r_at_k[f"R@{k}"] = max(r_scores[:k]) if k <= len(r_scores) else 0.0

    # Final Score là trung bình cộng của 5 mốc
    final_score = sum(r_at_k.values()) / len(BTC_K_THRESHOLDS)

    # Tìm thứ hạng đầu tiên trúng (Frame Rank)
    first_hit_rank = -1
    for idx, sc in enumerate(r_scores):
        if sc > 0.0:
            first_hit_rank = idx + 1
            break

    return {
        "r_at_k": r_at_k,
        "final_score": final_score,
        "first_hit_rank": first_hit_rank,
        "first_hit_score": r_scores[first_hit_rank - 1] if first_hit_rank != -1 else 0.0,
        "video_hit_rank": video_hit_rank,
        "video_recall_at_k": video_recall_at_k,
        "r_scores": r_scores
    }
