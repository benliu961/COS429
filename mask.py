import numpy as np
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import pandas as pd
import os
from tqdm import tqdm


def mask_and_fill(image_path, annotation, image_id):
    image = Image.open(image_path).convert("RGB")

    mask = Image.new("L", (image.width, image.height), 0)
    draw = ImageDraw.Draw(mask)

    # Masks of humans
    for ann in annotation.loadAnns(
        annotation.getAnnIds(imgIds=image_id, catIds=[1], iscrowd=None)
    ):
        segmentation = ann["segmentation"]
        if isinstance(segmentation, list):
            # Polygonal segmentations
            for seg in segmentation:
                draw.polygon(seg, fill=255)
        else:
            # Mask RLE
            binary_mask = annotation.annToMask(ann)
            for i in range(binary_mask.shape[0]):
                for j in range(binary_mask.shape[1]):
                    if binary_mask[i, j]:
                        mask.putpixel((j, i), 255)

    # Compute the average color of non-masked (background) pixels
    np_image = np.array(image)
    np_mask = np.array(mask)
    background_pixels = np_image[np_mask == 0]
    average_color = tuple(np.mean(background_pixels, axis=0).astype(int))

    # Apply average color to masked areas in the image
    np_image[np_mask == 255] = average_color
    processed_image = Image.fromarray(np_image)

    return processed_image


output_directory = "C:/Datasets/COCO/masked"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Path to COCO annotations and images
annotation_path = "C:/Datasets/COCO/instances_val2014.json"
image_directory = "C:/Datasets/COCO/val2014/"

# Initialize COCO api for instance annotations
coco_annotation = COCO(annotation_path)

# # Example image id
# image_id = 136  # Replace with an actual image ID from your dataset

# # Get the file name and full image path
# img_info = coco_annotation.loadImgs(image_id)[0]
# image_path = image_directory + img_info["file_name"]

# Process the image
# result_image = mask_and_fill(image_path, coco_annotation, image_id)
# result_image.show()

skin_ann = "C:/Users/ewang/OneDrive/Desktop/Spring_2024/COS_429/Final/imagecaptioning-bias/annotations/images_val2014.csv"
data = pd.read_csv(skin_ann)

# Filter for relevant skin types
mask_data = data[data["bb_skin"].isin(["Light", "Dark"])]

for index, row in tqdm(
    mask_data.iterrows(), total=mask_data.shape[0], desc="Processing images"
):
    image_id = row["id"]
    skin_type = row["bb_skin"]

    # Load image details from COCO
    img_info = coco_annotation.loadImgs(image_id)[0]
    image_path = image_directory + img_info["file_name"]

    result_image = mask_and_fill(image_path, coco_annotation, image_id)
    # Save the masked image
    save_path = os.path.join(output_directory, f'masked_{img_info["file_name"]}')
    result_image.save(save_path)
