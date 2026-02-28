#hello this is the EDA file (data cleaning)


# ==========================================
# 1. SETUP & IMPORTS
# ==========================================
# Install required libraries if you are running locally:
# %pip install numpy pandas pillow torch torchvision matplotlib

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms.functional as F2

# Challenge Dataset Paths
train_path = "/kaggle/input/competitions/acm-ai-mini-ml-challenge-2026/train/train"
test_path = "/kaggle/input/competitions/acm-ai-mini-ml-challenge-2026/test/test"
csv_drive_path = "/kaggle/input/competitions/acm-ai-mini-ml-challenge-2026"

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. DATASET DEFINITION
# ==========================================

class ImageCSVDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, label_map=None, is_test=False):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.label_map = label_map
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # We use os.path.basename to clean up the path just in case 
        # the CSV contains "train/img_0.jpg" so we don't end up with "train/train/img_0.jpg"
        clean_filename = os.path.basename(row['path']) 
        img_path = os.path.join(self.root_dir, clean_filename)

        # Convert to RGB to ensure 3 channels for pre-trained models
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            image_id = int(row['ID'])
            return image, image_id  # useful for mapping back to submission
        else:
            label = self.label_map[row['target']]
            character = row['character']
            return image, label, character

# Define standard ImageNet transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

label_map = {"bad": 0, "good": 1}

# Instantiate the training dataset
train_dataset = ImageCSVDataset(
    csv_file=os.path.join(csv_drive_path, "train.csv"), 
    root_dir=train_path,    
    transform=transform,
    label_map=label_map,
)

# ==========================================
# 3. VISUALIZATION PLAYGROUND
# ==========================================

# 1. Analyze Class Distribution
print("--- Class Distribution ---")
class_counts = train_dataset.df['target'].value_counts()
print(class_counts)

# Plot a simple bar chart
plt.figure(figsize=(6, 4))
class_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Good vs. Bad Characters')
plt.ylabel('Count')
plt.xticks(rotation=0) # Keeps the text upright
plt.show()

# ------------------------------------------

# 2. Display a Random Grid of Images
print("\n--- Sample Images ---")
plt.figure(figsize=(10, 10))

# Grab 9 random rows from our dataframe directly
sample_df = train_dataset.df.sample(9)

# Loop through those 9 rows and plot them
for i, (_, row) in enumerate(sample_df.iterrows()):
    # Reconstruct the exact path using our Kaggle root path
    # (Assumes your dataframe paths look like 'train/img_0.jpg')
    img_path = os.path.join("/kaggle/input/competitions/acm-ai-mini-ml-challenge-2026/train", row['path'])
    
    # Open the raw image (no math required!)
    img = Image.open(img_path)
    
    # Plot it in a 3x3 grid
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    
    # Add a nice title
    label_name = "Good" if row['target'] == "good" else "Bad"
    plt.title(f"{row['character']}\n({label_name})", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()
