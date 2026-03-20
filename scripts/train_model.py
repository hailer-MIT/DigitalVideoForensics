import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import sys

# 🕵️ SMART PATH FINDER (Ensures brother folders can see each other)
current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from models.forensic_model import ForensicDualModel
from tqdm import tqdm

class ForensicDataset(Dataset):
    def __init__(self, csv_file, seq_len=5, transform=None):
        self.data = pd.read_csv(csv_file)
        self.seq_len = seq_len
        self.transform = transform
        
    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        frames = []
        for i in range(self.seq_len):
            img_path = self.data.iloc[idx + i]['path']
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        
        frames = torch.stack(frames)
        label = torch.tensor([self.data.iloc[idx + self.seq_len - 1]['label']], dtype=torch.float32)
        return frames, label

def train():
    print("🚀 Starting Training Phase...")
    
    # 📁 Auto-Create Folders
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 1. Hyperparameters
    batch_size = 8
    epochs = 10
    lr = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Data Preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check if labels.csv exists
    if not os.path.exists('processed/labels.csv'):
        print("❌ Error: 'processed/labels.csv' not found. Please run 'extract_frames.py' first.")
        return
        
    dataset = ForensicDataset(csv_file='processed/labels.csv', seq_len=5, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Model Setup
    model = ForensicDualModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 📉 LOGS: Store training history
    history = []
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, leave=True)
        for frames, labels in loop:
            frames, labels = frames.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader)
        history.append({'epoch': epoch+1, 'loss': epoch_loss})
            
        # Save Checkpoint
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth')
        print(f"✅ Epoch {epoch+1} Complete. Avg Loss: {epoch_loss:.4f}")

    # 📊 Save Training History to CSV (Presentation Proof)
    history_df = pd.DataFrame(history)
    history_df.to_csv('results/training_history.csv', index=False)
    print("📉 Training history log saved to: results/training_history.csv")

if __name__ == "__main__":
    train()
