import shutil
from pathlib import Path

def migrate():
    print("=== TỰ ĐỘNG CẬP NHẬT CẤU TRÚC THƯ MỤC SANG BATCH 1 TRÊN SERVER ===")
    
    root = Path(__file__).resolve().parents[1]
    
    # 1. Di chuyển data/processed -> data/batch_1/processed
    old_proc = root / "data" / "processed"
    new_proc = root / "data" / "batch_1" / "processed"
    new_proc.mkdir(parents=True, exist_ok=True)
    
    if old_proc.exists() and old_proc != new_proc:
        for item in old_proc.iterdir():
            dest = new_proc / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
                print(f"  🚚 Đã chuyển: data/processed/{item.name} -> data/batch_1/processed/{item.name}")
        # Xóa thư mục processed cũ nếu đã rỗng
        try:
            old_proc.rmdir()
        except Exception:
            pass

    # 2. Di chuyển raw/ -> raw/batch_1/
    old_raw = root / "raw"
    new_raw = root / "raw" / "batch_1"
    new_raw.mkdir(parents=True, exist_ok=True)
    
    if old_raw.exists():
        for item in list(old_raw.iterdir()):
            if item.name in ["batch_1", "batch_2", ".gitkeep"]:
                continue
            dest = new_raw / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
                print(f"  🚚 Đã chuyển: raw/{item.name} -> raw/batch_1/{item.name}")

    # 3. Tạo sẵn các placeholder cho batch_2
    (root / "data" / "batch_2" / "processed").mkdir(parents=True, exist_ok=True)
    (root / "raw" / "batch_2").mkdir(parents=True, exist_ok=True)
    (root / "query" / "batch_1").mkdir(parents=True, exist_ok=True)
    (root / "query" / "batch_2").mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "batch_1" / "submission").mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "batch_2" / "submission").mkdir(parents=True, exist_ok=True)
    
    print("\n✅ HOÀN TẤT! Toàn bộ dữ liệu trên Server đã được sắp xếp chuẩn xác theo cấu trúc Batch mới.")

if __name__ == "__main__":
    migrate()
