import torch
from transformers import CLIPTextModelWithProjection, CLIPProcessor
import numpy as np
from abc import ABC, abstractmethod

class TextEncoderInterface(ABC):
    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        pass

class CLIPTextEncoder(TextEncoderInterface):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP text model {model_name} on {self.device}...")
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def encode(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros((1, 512), dtype=np.float32)
            
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            text_features = outputs.text_embeds
            
        # L2 Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().astype(np.float32)

class MultiLingualQueryEncoder(TextEncoderInterface):
    def __init__(self, encoder: CLIPTextEncoder, translator, use_translation: bool = True, fusion_alpha: float = 0.5):
        self.encoder = encoder
        self.translator = translator
        self.use_translation = use_translation
        self.fusion_alpha = fusion_alpha
        
    def encode(self, text: str) -> np.ndarray:
        # If translation is off, just encode the raw Vietnamese text (CLIP may perform poorly)
        if not self.use_translation:
            return self.encoder.encode(text)
            
        # Translate to English
        en_text = self.translator.translate(text)
        
        vi_vec = self.encoder.encode(text)
        en_vec = self.encoder.encode(en_text)
        
        # Vector Fusion
        fused_vec = self.fusion_alpha * vi_vec + (1.0 - self.fusion_alpha) * en_vec
        
        # Re-normalize
        norm = np.linalg.norm(fused_vec, axis=1, keepdims=True)
        if norm[0][0] > 0:
            fused_vec = fused_vec / norm
            
        return fused_vec.astype(np.float32)
