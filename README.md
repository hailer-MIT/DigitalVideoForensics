# 📽️ Digital Video Forensics Project

This project implements a **Video Tampering Detection Algorithm** designed to detect both spatial (frame-level) and temporal (sequence-level) manipulations.

## 🧬 Detection Strategy
We use a **Dual Architecture** approach:
1. **CNN (ResNet/EfficientNet)**: Detects spatial anomalies (Copy-move, splicing, deepfake texture).
2. **LSTM/RNN layer**: Detects temporal anomalies (Frame deletion, insertion, speed changes).

## 📁 Project Structure
- `data/`: Raw original and tampered videos (e.g., from Kaggle).
- `processed/`: Extracted and normalized frames ready for training.
- `models/`: Neural network architecture scripts.
- `checkpoints/`: Saved model weights during training.
- `scripts/`: Utilities for preprocessing, training, and evaluation.
- `results/`: Performance metrics (Accuracy, F1-Score) and visualization plots.

## 🚀 Getting Started

### 1. Environment Setup
Install the necessary dependencies using:
```bash
pip install -r requirements.txt
```

### 2. Verify Your Environment
Run the check script to ensure your GPU is detected:
```bash
python check_env.py
```

### 3. Workflow for Cloud (Lightning AI)
1. Commit and push any code changes to your GitHub repository.
2. Pull the repository on Lightning AI.
3. Run the training script using a Cloud GPU for maximum speed!

---
*Created for the Digital Video Forensics Training.*
