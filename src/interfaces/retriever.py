from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VideoRetriever(ABC):
    """
    Interface for future video-level or alternative modality retrievers (e.g. BLIP/Audio).
    Batch 2 will likely include new modalities which must implement this interface
    for seamless integration into the Fusion Engine.
    """
    
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        pass
        
    @abstractmethod
    def get_modalities(self) -> List[str]:
        """Returns ['text', 'audio', 'image']"""
        pass
