import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights
from PIL import Image
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import pickle


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels[
            (self.img_labels["bb_skin"] == "Light")
            | (self.img_labels["bb_skin"] == "Dark")
        ]
        self.img_labels["id"] = self.img_labels["id"].astype(
            int
        )  # Ensure IDs are integers
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_id = self.img_labels.iloc[idx, 0]
        img_name = f"masked_COCO_val2014_{str(img_id).zfill(12)}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        # bb_skin
        label = self.img_labels.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        return image, label, img_id


# Load pre-trained ResNet model with the recommended new method
weights = ResNet50_Weights.DEFAULT
resnet = models.resnet50(weights=weights)
resnet.fc = torch.nn.Identity()  # Modify the model to output features directly
resnet.eval()  # Set model to evaluation mode

# Image transformations
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create dataset
dataset = CustomImageDataset(
    annotations_file="C:/Users/ewang/OneDrive/Desktop/Spring_2024/COS_429/Final/imagecaptioning-bias/annotations/images_val2014.csv",
    img_dir="C:/Datasets/COCO/masked",
    transform=transform,
)

# DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

# Initialize empty lists to store features, assuming max ID is known or calculated
max_id = dataset.img_labels["id"].max() + 1
light_features = []
dark_features = []

for images, labels, ids in tqdm(dataloader):
    images = images.to(device)
    with torch.no_grad():
        features = resnet(images)
        # print(features)
    for feature, label, img_id in zip(features, labels, ids):
        # print(label)
        # print(img_id.item())
        if label == "Light":
            light_features.append(feature.cpu().numpy())
            # print(feature.cpu().numpy())
            # print(light_features[img_id.item()])
        elif label == "Dark":
            dark_features.append(feature.cpu().numpy())


np.save(
    "C:/Users/ewang/OneDrive/Desktop/Spring_2024/COS_429/Final/light_features.npy",
    light_features,
)
np.save(
    "C:/Users/ewang/OneDrive/Desktop/Spring_2024/COS_429/Final/dark_features.npy",
    dark_features,
)

# with open(
#     "C:/Users/ewang/OneDrive/Desktop/Spring_2024/COS_429/Final/light_features.pkl", "wb"
# ) as f:
#     pickle.dump(light_features, f)
# with open(
#     "C:/Users/ewang/OneDrive/Desktop/Spring_2024/COS_429/Final/dark_features.pkl", "wb"
# ) as f:
#     pickle.dump(dark_features, f)
