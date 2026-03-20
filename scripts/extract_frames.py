import os
import cv2
import pandas as pd
from tqdm import tqdm
import sys

# 🕵️ SMART PATH FINDER (Ensures brother folders can see each other)
current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

def extract_frames(input_root='data/FaceForensics++_C23', output_dir='processed', frames_per_video=30):
    os.makedirs(output_dir, exist_ok=True)
    
    # Store frame paths and labels
    data_log = []
    
    # Define folders and their ground truth labels (0=Authentic, 1=Tampered)
    categories = {
        'original': 0,
        'Deepfakes': 1,
        'DeepFakeDetection': 1,
        'Face2Face': 1,
        'FaceShifter': 1,
        'FaceSwap': 1,
        'NeuralTextures': 1
    }
    
    print(f"🕵️ Starting Forensic Preprocessing on {input_root}...")
    
    for folder_name, label in categories.items():
        folder_path = os.path.join(input_root, folder_name)
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: Folder '{folder_name}' not found. Skipping.")
            continue
            
        # Get list of videos
        videos = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi'))]
        
        # 🔥 Subset Limit: 100 per category to save time/space
        videos = videos[:100] 
        print(f"🎥 Processing {len(videos)} videos in '{folder_name}'...")
        
        for video_name in tqdm(videos):
            video_path = os.path.join(folder_path, video_name)
            cap = cv2.VideoCapture(video_path)
            
            # Count total frames & calculate interval
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0: continue
            
            interval = max(1, total_frames // frames_per_video)
            
            count = 0
            saved_count = 0
            while cap.isOpened() and saved_count < frames_per_video:
                ret, frame = cap.read()
                if not ret: break
                
                if count % interval == 0:
                    base_name = video_name.split('.')[0]
                    frame_name = f"{folder_name}_{base_name}_fr{saved_count}.jpg"
                    frame_path = os.path.join(output_dir, frame_name)
                    
                    # 1. Resize for ResNet consistency
                    frame_resized = cv2.resize(frame, (224, 224))
                    
                    # 2. Save image
                    cv2.imwrite(frame_path, frame_resized)
                    
                    # 3. Log data
                    data_log.append({'path': frame_path, 'label': label})
                    saved_count += 1
                    
                count += 1
            cap.release()
            
    # Save the labels CSV
    if data_log:
        df = pd.DataFrame(data_log)
        csv_path = os.path.join(output_dir, 'labels.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Preprocessing complete! {len(data_log)} frames saved to '{output_dir}/'.")
        print(f"📊 CSV file created: {csv_path}")
    else:
        print("❌ No frames were extracted! Check your data path paths or dataset content.")

if __name__ == "__main__":
    # Pointing to the specific folder structure found in your studio
    extract_frames(input_root='data/FaceForensics++_C23', output_dir='processed')
