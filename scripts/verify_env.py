import sys

def verify_environment():
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    missing_pkgs = []
    
    try:
        import numpy as np
        print(f"numpy: {np.__version__}")
    except ImportError:
        missing_pkgs.append("numpy")
        
    try:
        import pandas as pd
        print(f"pandas: {pd.__version__}")
    except ImportError:
        missing_pkgs.append("pandas")
        
    try:
        import pyarrow
        print(f"pyarrow: {pyarrow.__version__}")
    except ImportError:
        missing_pkgs.append("pyarrow")
        
    try:
        import faiss
        print(f"faiss: {faiss.__version__}")
    except ImportError:
        missing_pkgs.append("faiss-cpu")
        
    try:
        import torch
        print(f"torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        missing_pkgs.append("torch")
        
    try:
        import transformers
        print(f"transformers: {transformers.__version__}")
    except ImportError:
        missing_pkgs.append("transformers")
        
    try:
        import open_clip
        print(f"open_clip: installed")
    except ImportError:
        missing_pkgs.append("open_clip_torch")
        
    try:
        import streamlit as st
        print(f"streamlit: {st.__version__}")
    except ImportError:
        missing_pkgs.append("streamlit")
        
    try:
        import pydantic
        print(f"pydantic: {pydantic.__version__}")
    except ImportError:
        missing_pkgs.append("pydantic")
        
    try:
        import yaml
        print(f"pyyaml: installed")
    except ImportError:
        missing_pkgs.append("pyyaml")
        
    try:
        import tqdm
        print(f"tqdm: {tqdm.__version__}")
    except ImportError:
        missing_pkgs.append("tqdm")
        
    try:
        import sklearn
        print(f"scikit-learn: {sklearn.__version__}")
    except ImportError:
        missing_pkgs.append("scikit-learn")
        
    try:
        from PIL import Image
        print(f"pillow: installed")
    except ImportError:
        missing_pkgs.append("pillow")

    if missing_pkgs:
        print("\n[ERROR] Missing packages:")
        for pkg in missing_pkgs:
            print(f"  - {pkg}")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All core packages imported successfully!")
        sys.exit(0)

if __name__ == "__main__":
    verify_environment()
