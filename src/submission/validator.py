import re
from pathlib import Path
from typing import Tuple

class SubmissionValidator:
    """
    Validates the generated CSV submission to ensure it strictly matches BTC requirements.
    """
    def __init__(self):
        # Pattern: Lxx_Vxxx,frame_idx (where x is digit)
        self.line_pattern = re.compile(r'^L\d{2}_V\d{3},\d+$')
        
    def validate(self, filepath: Path) -> Tuple[bool, str]:
        if not filepath.exists():
            return False, f"File {filepath} does not exist."
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            if len(lines) == 0:
                return False, "File is empty."
                
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue # skip empty lines at EOF if any
                    
                if not self.line_pattern.match(line):
                    return False, f"Invalid format at line {i+1}: '{line}'. Expected format: Lxx_Vxxx,frame_idx"
                    
            return True, "Validation passed."
            
        except Exception as e:
            return False, f"Error reading file: {e}"
