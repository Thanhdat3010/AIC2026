import time, torch, cv2, numpy as np
from extract_ocr_from_drive import UltraFastMaxAccuracyOCR

print('=== 1. Khởi tạo OCR Engine ===')
ocr = UltraFastMaxAccuracyOCR(use_vietocr=True, use_gpu=True, batch_size=16)

print('=== 2. Tạo lô 16 mẩu chữ tiếng Việt ===')
sample_texts = [
    'TRUONG DAI HOC KHOA HOC TU NHIEN',
    'THANH PHO HO CHI MINH',
    'BAN TIN THOI SU 60 GIAY',
    'CHAY LON TAI THANH HOA',
    'NGUY HIEM KHONG LAI GAN',
    'DONG BANG SONG CUU LONG',
    'SUT LUN DANG DIEN RA',
    'VIET NAM NHAT BAN HOP TAC'
] * 2

crops = []
for text in sample_texts:
    img = np.zeros((40, 320, 3), dtype=np.uint8) + 255
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    crops.append(img)

print('=== 3. Đo tốc độ giải mã GPU Batch ===')
t0 = time.time()
results = ocr.batch_vietocr.predict_batch(crops, batch_size=16)
elapsed = time.time() - t0

print(f'==> Thời gian: {elapsed:.3f}s')
for i, r in enumerate(results): 
    print(f'{i}: {r}')
