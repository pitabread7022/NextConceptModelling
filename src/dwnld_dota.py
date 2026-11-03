import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from collections import Counter

def download_and_analyze_dota8():
    data_dir = Path("data")    
    zip_path = data_dir / "dota8.zip"
    dota8_root = data_dir / "dota8"    
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/dota8.zip"
    if not dota8_root.exists():
        print(f"Downloading DOTA8 dataset from {url}...")
        try:
            urlretrieve(url, zip_path)
            print("Download complete. Extracting files into 'data/'...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"Successfully extracted to {dota8_root}")
            if zip_path.exists():
                os.remove(zip_path)
        except Exception as e:
            print(f"Error downloading or extracting the dataset: {e}")
            return
    else:
        print(f"DOTA8 dataset already exists at {dota8_root}")

    # 3. Basic Exploratory Dataset Analysis (EDA)
    splits = ["train", "val"]
    total_images = 0
    total_annotations = 0
    class_counter = Counter()
    
    # DOTA 1.0 class mapping subset present in DOTA8
    class_names = {
        0: "plane", 1: "ship", 2: "storage tank", 3: "baseball diamond", 
        4: "tennis court", 5: "basketball court", 6: "ground track field", 
        7: "harbor", 8: "bridge", 9: "large vehicle", 10: "small vehicle", 
        11: "helicopter", 12: "roundabout", 13: "soccer ball field", 14: "swimming pool"
    }

    print("\n" + "="*45)
    print(" DOTA8 DATASET STRUCTURAL ANALYSIS")
    print("="*45)

    for split in splits:
        img_dir = dota8_root / "images" / split
        lbl_dir = dota8_root / "labels" / split
        
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) if img_dir.exists() else []
        num_imgs = len(images)
        total_images += num_imgs
        
        split_annotations = 0
        if lbl_dir.exists():
            for img_file in images:
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    with open(lbl_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                cls_id = int(parts[0])
                                class_counter[cls_id] += 1
                                split_annotations += 1
                                
        total_annotations += split_annotations
        print(f"Split [{split.upper()}]:")
        print(f"  - Directory   : {img_dir}")
        print(f"  - Images      : {num_imgs}")
        print(f"  - Annotations : {split_annotations}")

    print("-" * 45)
    print(f"Total Images Across Splits : {total_images}")
    print(f"Total OBB Labels Found     : {total_annotations}")
    print("-" * 45)
    
    print("\n Class Distribution:")
    if class_counter:
        for cls_id, count in sorted(class_counter.items()):
            name = class_names.get(cls_id, f"Class {cls_id}")
            print(f"  - {name} (ID {cls_id}): {count} instance(s)")
    else:
        print("  No label files found.")

if __name__ == "__main__":
    download_and_analyze_dota8()