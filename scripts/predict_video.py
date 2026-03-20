import torch
import cv2
import os
import sys
from torchvision import transforms
from PIL import Image

# 🕵️ SMART PATH FINDER (Ensures brother folders can see each other)
current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from models.forensic_model import ForensicDualModel

def predict_video(video_path, seq_len=5):
    print(f"🕵️ Analyzing Forensic Integrity: {video_path}...")
    
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load the Most Recent Model Checkpoint
    model = ForensicDualModel().to(device)
    if not os.path.exists('checkpoints'):
        print("❌ Error: 'checkpoints/' directory not found.")
        return
        
    checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
    if not checkpoints:
        print("❌ Error: No trained model found in 'checkpoints/'! Please train the model first.")
        return
    
    latest_cp = os.path.join('checkpoints', sorted(checkpoints)[-1])
    print(f"📦 Loading Trained Brain: {latest_cp}")
    model.load_state_dict(torch.load(latest_cp, map_location=device))
    model.eval()
    
    # 3. Preprocessing (ResNet standard)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Extract Sample Sequence (Middle of the video)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, (total_frames // 2) - seq_len)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    count = 0
    while cap.isOpened() and count < seq_len:
        ret, frame = cap.read()
        if not ret: break
        
        # Convert CV2 (BGR) to PIL (RGB)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = transform(image)
        frames.append(image)
        count += 1
    cap.release()
    
    if len(frames) < seq_len:
        print(f"❌ Error: Video too short or corrupted. Found {len(frames)} frames.")
        return
        
    # 5. Forensic Inference
    # Input shape: (Batch=1, Seq_Len=5, C, H, W)
    input_tensor = torch.stack(frames).unsqueeze(0).to(device)
    
    with torch.no_grad():
        probability = model(input_tensor).item()
    
    # 6. Final Verdict
    confidence = probability * 100 if probability > 0.5 else (1 - probability) * 100
    verdict = "🛑 TAMPERED / FORGED" if probability > 0.5 else "✅ AUTHENTIC / ORIGINAL"
    
    print("\n--- FORENSIC SCAN RESULT ---")
    print(f"📽️ Video:  {os.path.basename(video_path)}")
    print(f"🔬 Verdict: {verdict}")
    print(f"📊 Accuracy Confidence: {confidence:.2f}%")
    print("----------------------------\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("💡 Usage: python scripts/predict_video.py <path_to_video>")
    else:
        video_file = sys.argv[1]
        if os.path.exists(video_file):
            predict_video(video_file)
        else:
            print(f"❌ Error: File '{video_file}' not found.")
