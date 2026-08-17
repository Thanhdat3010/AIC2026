import os
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

class UnifiedTextEncoder:
    """
    Bộ mã hóa văn bản thống nhất hỗ trợ cả:
    1. SigLIP 2 (google/siglip2-so400m-patch14-384 -> 1152d)
       * Lưu ý kiến trúc: SigLIP 2 bắt buộc dùng padding='max_length', max_length=64
    2. CLIP BTC (openai/clip-vit-base-patch32 -> 512d)
    """
    def __init__(self, engine: str = "siglip2", device: str = None):
        self.engine = engine.lower()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.engine == "siglip2":
            self.model_name = "google/siglip2-so400m-patch14-384"
            self.dim = 1152
            self.max_length = 64
        elif self.engine == "clip":
            self.model_name = "openai/clip-vit-base-patch32"
            self.dim = 512
            self.max_length = 77
        else:
            raise ValueError(f"Unsupported engine: {self.engine}. Use 'siglip2' or 'clip'.")

        print(f"[*] Khởi tạo Text Encoder [{self.engine.upper()}]: {self.model_name} (Device: {self.device})...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Đã tải xong Text Encoder [{self.engine.upper()}] ({self.dim} chiều)", flush=True)

    def _extract_tensor(self, outputs) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            feat = outputs
        elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            feat = outputs.pooler_output
        elif hasattr(outputs, 'text_embeds') and outputs.text_embeds is not None:
            feat = outputs.text_embeds
        elif hasattr(outputs, 'last_hidden_state'):
            feat = outputs.last_hidden_state[:, 0, :]
        else:
            feat = outputs[0]
        # Chuẩn hóa L2 norm
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """
        Mã hóa một câu text thành vector numpy float32 chuẩn hóa L2 norm (shape: [1, dim]).
        """
        if self.engine == "siglip2":
            inputs = self.tokenizer([text], padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        else:
            inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.get_text_features(**inputs)
        features = self._extract_tensor(outputs)
        return features.cpu().to(torch.float32).numpy()

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Mã hóa danh sách text theo batch.
        """
        if self.engine == "siglip2":
            inputs = self.tokenizer(texts, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        else:
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.get_text_features(**inputs)
        features = self._extract_tensor(outputs)
        return features.cpu().to(torch.float32).numpy()

if __name__ == "__main__":
    sample_query = "Inside a room, a woman wraps and adjusts an orange-yellow sarong around the waist of a man wearing a blue shirt."
    
    # Test SigLIP 2
    enc_siglip = UnifiedTextEncoder("siglip2")
    vec_siglip = enc_siglip.encode_text(sample_query)
    print(f"SigLIP 2 Vector shape: {vec_siglip.shape}, L2 Norm: {np.linalg.norm(vec_siglip):.4f}")

    # Test CLIP
    enc_clip = UnifiedTextEncoder("clip")
    vec_clip = enc_clip.encode_text(sample_query)
    print(f"CLIP Vector shape: {vec_clip.shape}, L2 Norm: {np.linalg.norm(vec_clip):.4f}")
