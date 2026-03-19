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
    checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
    if not checkpoints:
        print("❌ No checkpoints found! Please train the model first.")
        return
    
    latest_cp = os.path.join('checkpoints', sorted(checkpoints)[-1])
    model.load_state_dict(torch.load(latest_cp, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # 3. Prediction Loop
    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.to(device)
            outputs = model(frames)
            preds = (outputs > 0.5).float().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    # 4. Calculate Final Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"\n--- Forensic Performance Report ---")
    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"✅ Recall:    {rec:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")
    
    # 5. Save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Video Forensics Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.savefig('results/confusion_matrix.png')
    print("📈 Confusion Matrix saved to results/confusion_matrix.png")

if __name__ == "__main__":
    import os
    evaluate()
