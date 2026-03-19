import cv2
import os
import pandas as pd
from tqdm import tqdm

def extract_frames(input_dir, output_dir, frames_per_video=30):
    """
    Extracts frames from videos and saves them into labeled subfolders.
    input_dir: Path to directory with 'original' and 'fake' subfolders.
    output_dir: Path to save processed images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Store frame paths and labels
    data_log = []
    
    # Categories based on FaceForensics structure
    categories = ['original', 'fake'] # Adjust based on actual folder names after download
    
    for category in categories:
        cat_input = os.path.join(input_dir, category)
        cat_output = os.path.join(output_dir, category)
        os.makedirs(cat_output, exist_ok=True)
        
        if not os.path.exists(cat_input):
            print(f"Warning: Category folder {cat_input} not found. Skipping.")
            continue
            
        videos = [f for f in os.listdir(cat_input) if f.endswith(('.mp4', '.avi'))]
        print(f"🎥 Processing {len(videos)} videos in '{category}'...")
        
        for video_name in tqdm(videos):
            video_path = os.path.join(cat_input, video_name)
            cap = cv2.VideoCapture(video_path)
            
            # Count total frames & calculate interval
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(1, total_frames // frames_per_video)
            
            count = 0
            saved_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame at specific intervals
                if count % interval == 0 and saved_count < frames_per_video:
                    # Rename to avoid collisions
                    frame_name = f"{video_name.split('.')[0]}_frame_{saved_count}.jpg"
                    frame_path = os.path.join(cat_output, frame_name)
                    
                    # 1. Resize for CNN
                    frame_resized = cv2.resize(frame, (224, 224))
                    
                    # 2. Save
                    cv2.imwrite(frame_path, frame_resized)
                    
                    # 3. Log data
                    label = 0 if category == 'original' else 1
                    data_log.append({'path': frame_path, 'label': label})
                    
                    saved_count += 1
                
                count += 1
            cap.release()
            
    # Save the labels Cheat Sheet
    df = pd.DataFrame(data_log)
    df.to_csv('processed/labels.csv', index=False)
    print(f"✅ Preprocessing complete. {len(data_log)} frames saved to processed/ folder.")

if __name__ == "__main__":
    # We will run this after download_data.py
    # data/original and data/fake are standard names
    extract_frames(input_dir='data', output_dir='processed')
