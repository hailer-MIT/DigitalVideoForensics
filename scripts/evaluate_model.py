import os
import sys

# 🕵️ SMART PATH FINDER (Ensures brother folders can see each other)
current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import pandas as pd
from torch.utils.data import DataLoader
from scripts.train_model import ForensicDataset, ForensicDualModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms

def evaluate():
    print("🧪 Starting Model Evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #  AUTO-CREATE RESULTS FOLDER
    os.makedirs('results', exist_ok=True)
    
    # 1. Load Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ForensicDataset(csv_file='processed/labels.csv', seq_len=5, transform=transform)
    test_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # 2. Load the BEST model from checkpoints
    model = ForensicDualModel().to(device)
    # Automatically pick the last saved checkpoint
    if not os.path.exists('checkpoints'):
        print("❌ Error: 'checkpoints/' directory not found.")
        return
        
    checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
    if not checkpoints:
        print("❌ No checkpoints found! Please train the model first.")
        return
    
    latest_cp = os.path.join('checkpoints', sorted(checkpoints)[-1])
    print(f"📦 Loading Model Checkpoint: {latest_cp}")
    model.load_state_dict(torch.load(latest_cp, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # 3. Prediction Loop
    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.to(device)
            outputs = model(frames)
            # Probability > 0.5 means Fake (1)
            preds = (outputs > 0.5).float().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    # 4. Calculate Final Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    report = f"""
    --- DIGITAL VIDEO FORENSICS REPORT ---
    --------------------------------------
    Model: CNN (ResNet) + LSTM (Temporal)
    Dataset: FaceForensics++ C23
    
    FINAL METRICS:
    ✅ Accuracy:  {acc:.4f}
    ✅ Precision: {prec:.4f}
    ✅ Recall:    {rec:.4f}
    ✅ F1 Score:  {f1:.4f}
    
    Evidence saved in: results/
    """
    
    print(report)
    
    # 📝 Save the text report
    with open('results/final_report.txt', 'w') as f:
        f.write(report)
    print("📑 Text report saved to: results/final_report.txt")
    
    # 5. Save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Authentic', 'Fake'], 
                yticklabels=['Authentic', 'Fake'])
    plt.title('Video Forensics Confusion Matrix')
    plt.xlabel('Predicted by AI')
    plt.ylabel('Ground Truth')
    plt.savefig('results/confusion_matrix.png')
    print("📈 Confusion Matrix image saved to: results/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
