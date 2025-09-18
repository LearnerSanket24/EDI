import os
import shutil
from sklearn.model_selection import train_test_split
import glob

# Dataset paths
dataset_root = "../CrowdHuman Cropped/Dataset CrowdHuman"
classes = ['crowd', 'non crowd']

# Create train/val dirs
for split in ['train', 'val']:
    for cls in classes:
        os.makedirs(os.path.join(dataset_root, split, cls), exist_ok=True)

for cls in classes:
    cls_path = os.path.join(dataset_root, cls)
    images = glob.glob(os.path.join(cls_path, "*.jpg"))
    
    if len(images) == 0:
        print(f"No images found in {cls_path}")
        continue
    
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    
    for img in train_imgs:
        shutil.move(img, os.path.join(dataset_root, 'train', cls, os.path.basename(img)))
    for img in val_imgs:
        shutil.move(img, os.path.join(dataset_root, 'val', cls, os.path.basename(img)))
    
    print(f"Split {cls}: {len(train_imgs)} train, {len(val_imgs)} val")

print("Dataset split complete.")