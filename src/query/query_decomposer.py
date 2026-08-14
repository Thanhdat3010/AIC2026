import re
from typing import List

class QueryDecomposer:
    """
    Decomposes a complex textual query into multiple sub-cues (temporal or logical).
    In the baseline (M5), it uses simple rule-based splitting (by commas, dots, or "và").
    """
    def __init__(self):
        # Regex to split on common delimiters: commas, dots, hyphens, and the word "và"
        self.split_pattern = re.compile(r'[,.\-]|(?:\s+và\s+)')
        
    def decompose(self, query: str) -> List[str]:
        # Basic cleanup
        query = query.strip()
        if not query:
            return []
            
        # Split query
        raw_cues = self.split_pattern.split(query)
        
        # Clean up individual cues
        cues = []
        for cue in raw_cues:
            clean_cue = cue.strip()
            if len(clean_cue) > 2:  # Ignore very short meaningless fragments
                cues.append(clean_cue)
                
        # If no valid cues were extracted (e.g. query was just "và"), return the original
        if not cues and len(query) > 0:
            return [query]
            
        return cues
