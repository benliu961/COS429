import pandas as pd
import numpy as np
import os
from skimage import io, feature, color
from skimage.transform import resize
from tqdm import tqdm

# Load the annotations file
annotations_file = "C:/Users/ewang/OneDrive/Desktop/Spring_2024/COS_429/Final/imagecaptioning-bias/annotations/images_val2014.csv"
img_dir = "C:/Datasets/COCO/masked"
output_dir = "C:/Users/ewang/OneDrive/Desktop/Spring_2024/COS_429/Final/"

# Read the CSV file into DataFrame
df = pd.read_csv(annotations_file)

# Filter out rows where skin color is "Unsure"
df = df[df["bb_skin"].isin(["Light", "Dark"])]


# Function to extract HoG features
def extract_hog_features(image_path):
    image = io.imread(image_path, as_gray=True)  # Load image in grayscale
    image = resize(
        image, (128, 128), anti_aliasing=True
    )  # Resize to handle different sizes
    fd = feature.hog(
        image,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=False,
        channel_axis=-1,
    )
    return fd


# Initialize lists to store features
light_features = []
dark_features = []

# Process each image in the DataFrame
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    img_id = row["id"]
    bb_skin = row["bb_skin"]
    img_name = f"masked_COCO_val2014_{str(img_id).zfill(12)}.jpg"
    img_path = os.path.join(img_dir, img_name)

    # Extract HoG features
    if os.path.exists(img_path):
        fd = extract_hog_features(img_path)
        if bb_skin == "Light":
            light_features.append(fd)
        elif bb_skin == "Dark":
            dark_features.append(fd)

# Convert lists to numpy arrays
light_features = np.array(light_features)
dark_features = np.array(dark_features)

# Save the feature arrays
np.save(os.path.join(output_dir, "light_features_hog.npy"), light_features)
np.save(os.path.join(output_dir, "dark_features_hog.npy"), dark_features)
