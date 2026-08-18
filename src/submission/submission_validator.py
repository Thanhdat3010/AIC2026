import os
import re
import csv
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Any

class SubmissionValidator:
    """
    Trình kiểm tra cú pháp và đóng gói file nộp bài chuẩn 100% quy chế BTC AI Challenge 2026.
    """
    def __init__(self):
        pass

    def validate_csv_file(self, file_path: Path, task_type: str = "auto") -> Dict[str, Any]:
        file_path = Path(file_path)
        if not file_path.exists():
            return {"valid": False, "error": f"File không tồn tại: {file_path.name}"}

        # Auto-detect task type from filename if needed
        fname = file_path.stem.lower()
        if task_type == "auto":
            if "qa" in fname:
                task_type = "qa"
            elif "trake" in fname:
                task_type = "trake"
            else:
                task_type = "kis"

        lines = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    lines.append(row)

        if len(lines) == 0:
            return {"valid": False, "error": f"{file_path.name} bị rỗng (0 dòng)!"}

        if len(lines) > 100:
            return {"valid": False, "error": f"{file_path.name} vượt quá giới hạn 100 dòng ({len(lines)} dòng)!"}

        # Kiểm tra dòng đầu tiên xem có bị dính Header chữ không
        first_row = lines[0]
        if first_row[0].strip().lower() in ["video_id", "video", "videoid", "id", "rank"]:
            return {"valid": False, "error": f"{file_path.name} bị dính dòng Header ('{first_row[0]}'), quy chế BTC cấm có header!"}

        # Kiểm tra cấu trúc từng dòng theo từng Task
        for r_idx, row in enumerate(lines, 1):
            if len(row) < 2:
                return {"valid": False, "error": f"Dòng {r_idx} trong {file_path.name} thiếu cột ({row})"}

            video_id = row[0].strip()
            if not video_id:
                return {"valid": False, "error": f"Dòng {r_idx} trong {file_path.name} có video_id bị rỗng"}

            if task_type == "kis":
                if len(row) != 2:
                    return {"valid": False, "error": f"Dòng {r_idx} (KIS) phải có đúng 2 cột <video_id>,<frame_idx>, thực tế có {len(row)} cột"}
                try:
                    f_idx = int(row[1].strip())
                    if f_idx < 0:
                        return {"valid": False, "error": f"Dòng {r_idx} có frame_idx âm ({f_idx})"}
                except ValueError:
                    return {"valid": False, "error": f"Dòng {r_idx} có frame_idx không phải số nguyên ({row[1]})"}

            elif task_type == "qa":
                if len(row) < 3:
                    return {"valid": False, "error": f"Dòng {r_idx} (QA) phải có ít nhất 3 cột <video_id>,<frame_idx>,<answer>"}
                try:
                    f_idx = int(row[1].strip())
                    if f_idx < 0:
                        return {"valid": False, "error": f"Dòng {r_idx} có frame_idx âm ({f_idx})"}
                except ValueError:
                    return {"valid": False, "error": f"Dòng {r_idx} có frame_idx không phải số nguyên ({row[1]})"}

                answer = ",".join(row[2:]).strip()
                if len(answer) > 100:
                    return {"valid": False, "error": f"Dòng {r_idx} câu trả lời QA vượt quá 100 ký tự ({len(answer)} ký tự)"}

            elif task_type == "trake":
                if len(row) < 3:
                    return {"valid": False, "error": f"Dòng {r_idx} (TRAKE) phải có <video_id> và ít nhất 2 event frame_idx"}
                for col_idx, f_str in enumerate(row[1:], 1):
                    try:
                        f_idx = int(f_str.strip())
                        if f_idx < 0:
                            return {"valid": False, "error": f"Dòng {r_idx} có frame_id_{col_idx} âm"}
                    except ValueError:
                        return {"valid": False, "error": f"Dòng {r_idx} có frame_id_{col_idx} không phải số nguyên ({f_str})"}

        return {
            "valid": True,
            "filename": file_path.name,
            "task_type": task_type,
            "total_rows": len(lines),
            "sample_top1": lines[0]
        }

    def validate_directory(self, dir_path: Path) -> Dict[str, Any]:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            return {"all_valid": False, "error": f"Thư mục {dir_path} không tồn tại"}

        csv_files = sorted(list(dir_path.glob("*.csv")))
        if not csv_files:
            return {"all_valid": False, "error": f"Không tìm thấy file .csv nào trong {dir_path}"}

        results = []
        all_valid = True
        for csv_f in csv_files:
            res = self.validate_csv_file(csv_f)
            results.append(res)
            if not res.get("valid", False):
                all_valid = False

        return {
            "all_valid": all_valid,
            "total_files": len(csv_files),
            "details": results
        }

    def package_submission(self, dir_path: Path, output_zip: Path) -> Dict[str, Any]:
        dir_path = Path(dir_path)
        output_zip = Path(output_zip)

        val_res = self.validate_directory(dir_path)
        if not val_res["all_valid"]:
            return {"success": False, "validation": val_res, "error": "Một số file CSV không đạt chuẩn BTC!"}

        output_zip.parent.mkdir(parents=True, exist_ok=True)
        csv_files = sorted(list(dir_path.glob("*.csv")))

        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for f in csv_files:
                z.write(f, arcname=f.name)

        return {
            "success": True,
            "zip_path": str(output_zip),
            "files_included": [f.name for f in csv_files],
            "zip_size_bytes": output_zip.stat().st_size
        }
