import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from models.forensic_model import ForensicDualModel
from tqdm import tqdm

class ForensicDataset(Dataset):
    def __init__(self, csv_file, seq_len=5, transform=None):
        self.data = pd.read_csv(csv_file)
        self.seq_len = seq_len
        self.transform = transform
        
    def __len__(self):
        # We need to ensure we have enough frames for a sequence
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        frames = []
        # Load a sequence of frames
        for i in range(self.seq_len):
            img_path = self.data.iloc[idx + i]['path']
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        
        # Stack frames: (Seq_Len, C, H, W)
        frames = torch.stack(frames)
        label = torch.tensor([self.data.iloc[idx + self.seq_len - 1]['label']], dtype=torch.float32)
        return frames, label

def train():
    print("🚀 Starting Training Phase...")
    
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
    
    dataset = ForensicDataset(csv_file='processed/labels.csv', seq_len=5, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Model Setup
    model = ForensicDualModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
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
            
        # Save Checkpoint
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth')
        print(f"✅ Epoch {epoch+1} Complete. Loss: {running_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    train()
