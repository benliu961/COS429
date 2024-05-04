import os
from PIL import Image


def view_coco_image(image_id):
    """
    Display an image from the COCO 2014 validation dataset given the image ID.

    Args:
    image_id (str): The COCO image ID (e.g., '000000000139')

    Returns:
    None
    """
    # Define the path to the COCO 2014 validation images
    images_path = "C:/Datasets/COCO/val2014"

    # Construct the filename from the image ID
    # COCO filenames are typically in the format 'COCO_val2014_000000000139.jpg'
    filename = f"COCO_val2014_{image_id.zfill(12)}.jpg"
    file_path = os.path.join(images_path, filename)

    # Check if the file exists
    if not os.path.exists(file_path):
        print("Image file not found.")
        return

    # Open and display the image
    with Image.open(file_path) as img:
        img.show()


# Example usage:
# Replace '000000000139' with the ID of the image you want to view
view_coco_image("508370")
view_coco_image("11149")
