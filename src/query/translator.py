from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class VietnameseTranslator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-vi-en"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading translation model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
            
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
            
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
