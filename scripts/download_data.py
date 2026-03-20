import os
import zipfile
import sys

# 🕵️ SMART PATH FINDER (Standardized and Shielded)
current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 🗝️ DIRECT AUTHENTICATION (Locked in)
os.environ['KAGGLE_USERNAME'] = "hailom"
os.environ['KAGGLE_KEY'] = "976bca26b0d163da284475978ced95ff"

import kaggle 

def download_ff_c23():
    print("🚀 Starting FaceForensics++ (FF-C23) Download...")
    
    # 🎯 TARGET DATASET
    dataset_slug = "xdxd003/ff-c23"
    
    # 2. Setup Data Dir
    os.makedirs('data', exist_ok=True)
    
    # 3. Pull from Kaggle via API
    print(f"Downloading {dataset_slug} from Kaggle...")
    os.system(f"kaggle datasets download -d {dataset_slug} -p data")
    
    # 4. Unzip logic for this specific slug
    zip_path = os.path.join('data', "ff-c23.zip")
    
    if os.path.exists(zip_path):
        print(f"📦 Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data')
        
        # Cleanup zip to save cloud space
        os.remove(zip_path)
        print("✅ Dataset ready in data/ folder.")
    else:
        print("❌ Download failed. Make sure you accepted the terms for 'xdxd003/ff-c23' on the Kaggle site!")

if __name__ == "__main__":
    download_ff_c23()
