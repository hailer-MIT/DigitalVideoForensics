import os
import zipfile

def download_faceforensics_subset():
    print("🚀 Starting FaceForensics++ (Subset) Download...")
    
    # 1. Define the Kaggle Dataset slug
    # This is a manageable 1GB subset
    dataset_slug = "mbaltatu/faceforensics-subset"
    
    # 2. Ensure the data directory exists
    os.makedirs('data', exist_ok=True)
    
    # 3. Run the Kaggle download command
    # MUST have kaggle.json in ~/.kaggle/ or project root
    print(f"Downloading {dataset_slug} from Kaggle...")
    os.system(f"kaggle datasets download -d {dataset_slug} -p data")
    
    # 4. Unzip the dataset
    zip_name = dataset_slug.split('/')[-1] + ".zip"
    zip_path = os.path.join('data', zip_name)
    
    if os.path.exists(zip_path):
        print(f"📦 Unzipping {zip_name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data')
        
        # Cleanup zip file to save space
        os.remove(zip_path)
        print("✅ Dataset ready in data/ folder.")
    else:
        print("❌ Download failed. Did you upload your kaggle.json?")

if __name__ == "__main__":
    download_faceforensics_subset()
