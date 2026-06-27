"""
Damage Claim AI - Computer Vision Training Pipeline
Downloads images from web links in claims.csv and trains a ResNet-50 model
to detect Damaged vs. Clean objects.
"""

import os
import requests
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Setup paths
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGE_DOWNLOAD_DIR = BASE_DIR / "downloaded_images"

print("📸 Initializing Damage Claim Computer Vision Pipeline...")

# 1. Download Images from CSV Web Links dynamically
def download_dataset_images(csv_path):
    df = pd.read_csv(csv_path)
    
    # If status_label doesn't exist yet, default all claims to 'damaged'
    if 'status_label' not in df.columns:
        df['status_label'] = 'damaged'
        
    for idx, row in df.iterrows():
        url = row['pictures']
        label = row['status_label']
        obj_type = row['claim_object']
        
        # Structure directories like: downloaded_images/train/car_damaged/
        category_dir = IMAGE_DOWNLOAD_DIR / f"{obj_type}_{label}"
        category_dir.mkdir(parents=True, exist_ok=True)
        
        image_path = category_dir / f"{row['user_id']}.jpg"
        
        if not image_path.exists():
            try:
                print(f"📥 Downloading image for {row['user_id']} ({obj_type} - {label})...")
                response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                if response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    # Verify it's a valid PIL image
                    with Image.open(image_path) as img:
                        img.verify()
                else:
                    print(f"⚠️ Failed download status {response.status_code} for {url}")
            except Exception as e:
                print(f"❌ Error downloading {url}: {e}")
                if image_path.exists():
                    os.remove(image_path)

# Execute image downloads using your claims.csv
csv_file_path = DATASET_DIR / "claims.csv"
download_dataset_images(csv_file_path)

# 2. Build PyTorch Custom Dataset 
class ClaimVisionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 3. Scan directories and prepare data splits
all_images = []
all_labels = []
# Map classes: clean = 0, damaged = 1
label_map = {'clean': 0, 'damaged': 1}

for folder in IMAGE_DOWNLOAD_DIR.iterdir():
    if folder.is_dir():
        # folder name format is: object_status (e.g. car_damaged)
        status = folder.name.split('_')[-1]
        if status in label_map:
            for img_file in folder.glob("*.jpg"):
                all_images.append(str(img_file))
                all_labels.append(label_map[status])

if len(set(all_labels)) < 2:
    print("\n⚠️ Training requires BOTH 'clean' and 'damaged' images in your dataset directory!")
    print("Please add some clean object rows to your CSV or manually add clean images into the folder splits to run the loop.")
    exit()

# Train/Validation Split
train_paths, val_paths, train_lbls, val_lbls = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# 4. Standard Image Transformations for ResNet-50
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ClaimVisionDataset(train_paths, train_lbls, transform=train_transforms)
val_dataset = ClaimVisionDataset(val_paths, val_lbls, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 5. Load Pre-trained ResNet-50 Architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze lower layers to speed up tuning on standard machines
for param in model.parameters():
    param.requires_grad = False

# Replace the final linear head with our binary classifier (Clean vs Damaged)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# 6. Training Optimization Loop
print(f"🚀 Firing up Computer Vision training loop on hardware device: {device}...")
epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}] - Classification Loss: {epoch_loss:.4f}")

# Save the trained vision brain
model_output_path = BASE_DIR / "saved_damage_vision_model.pt"
torch.save(model.state_dict(), model_output_path)
print(f"🎉 Computer vision model weights successfully compiled and saved to {model_output_path}!")