import os
import sys
import torch
import cv2
import numpy as np

def check_env():
    print("--- Digital Video Forensics Environment Check ---")
    
    # 1. Python Version
    print(f"Python Version: {sys.version}")
    
    # 2. Key Libraries
    print(f"PyTorch Version: {torch.__version__}")
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"NumPy Version: {np.__version__}")
    
    # 3. GPU Availability
    gpu_available = torch.cuda.is_available()
    print(f"GPU (CUDA) Available: {gpu_available}")
    if gpu_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
    else:
        print("Note: Using CPU for training. This will be SLOW.")
    
    # 4. Folder Structure Check
    required_folders = ['data', 'processed', 'models', 'checkpoints', 'scripts', 'results']
    print("\n--- Folder Structure Check ---")
    for folder in required_folders:
        if os.path.exists(folder):
            print(f"OK: {folder}/")
        else:
            print(f"MISSING: {folder}/")

if __name__ == "__main__":
    check_env()
