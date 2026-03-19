import torch
import torch.nn as nn
import torchvision.models as models

class ForensicDualModel(nn.Module):
    def __init__(self, hidden_dim=256, lstm_layers=1):
        super(ForensicDualModel, self).__init__()
        
        # 1. Spatial Feature Extractor (CNN)
        # Using ResNet-18 (Lightweight and Professional)
        # We use 'DEFAULT' weights to get a strong starting point
        resnet = models.resnet18(weights='DEFAULT')
        
        # We strip the last 'fc' layer to get raw features (512 dimensions)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512
        
        # 2. Temporal Analyzer (LSTM)
        # It takes the sequence of frame features and analyzes the motion
        self.lstm = nn.LSTM(input_size=self.feature_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=lstm_layers, 
                            batch_first=True)
        
        # 3. Decision Head
        # Binary Classification (0: Authentic vs 1: Tampered)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, C, H, W)
        batch_size, seq_len, c, h, w = x.shape
        
        # Flatten sequence into a single batch to process through CNN
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn(x) # Shape: (B*S, 512, 1, 1)
        
        # Reshape back for LSTM: (Batch, Sequence, Feature_Dim)
        features = features.view(batch_size, seq_len, -1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(features)
        
        # Take the output of the LAST frame in the sequence
        # This contains the accumulated 'temporal memory'
        final_feature = lstm_out[:, -1, :]
        
        # Final probability
        prediction = self.classifier(final_feature)
        return prediction

if __name__ == "__main__":
    # Test if the model initializes correctly
    model = ForensicDualModel()
    print("✅ ForensicDualModel (CNN+LSTM) initialized successfully!")
