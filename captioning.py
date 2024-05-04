import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from pycocotools.coco import COCO
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import LlavaForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def caption(image_path, model_type="GIT"):
    try:
        # Load the image
        image = Image.open(image_path)
        # Perform captioning
        # caption = model(image)
        if model_type == "GIT":
            processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
            model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
            model = model.to(device)
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(
                device
            )
            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            generated_caption = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            return generated_caption
        elif model_type == "LLAVA":
            model = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-7b-hf"
            )
            model = model.to(device)
            processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            prompt = "USER: <image>\nCaption this image ASSISTANT:"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(
                device
            )
            generate_ids = model.generate(**inputs, max_new_tokens=15)
            generated_text = processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return generated_text[generated_text.find("ASSISTANT:") + 1 :]
        elif model_type == "BLIP-2":
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                load_in_8bit=True,
                device_map={"": 0},
                torch_dtype=torch.float16,
            )
            model = model.to(device)
            inputs = processor(images=image, return_tensors="pt").to(
                device, torch.float16
            )
            generated_ids = model.generate(**inputs)
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            return generated_text
        else:
            print(f"Unsupported model type: {model_type}")
            return None
        # caption = "This is a caption."
        # return caption
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


pairs = "C:/Users/Benjamin Liu/Downloads/sim_stable_percentiles.csv"
data = pd.read_csv(pairs)

# Path to COCO images
# annotation_path = ""
image_directory = "C:/Datasets/COCO/val2014/"

# coco_annotation = COCO(annotation_path)

output = []

for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing images"):
    dark_id = str(int(row["dark_id"]))
    light_id = str(int(row["light_id"]))
    dark_filename = f"COCO_val2014_{dark_id.zfill(12)}.jpg"
    light_filename = f"COCO_val2014_{light_id.zfill(12)}.jpg"
    dark_image_path = os.path.join(image_directory, dark_filename)
    light_image_path = os.path.join(image_directory, light_filename)
    # dark_img_info = coco_annotation.loadImgs(dark_id)[0]
    # light_img_info = coco_annotation.loadImgs(light_id)[0]
    # dark_image_path = image_directory + str(dark_img_info)
    # light_image_path = image_directory + str(light_img_info)

    dark_caption = caption(dark_image_path, model_type="LLAVA")
    light_caption = caption(light_image_path, model_type="LLAVA")

    output.append([dark_caption, light_caption, dark_id, light_id, row["similarity"]])

output_df = pd.DataFrame(
    output,
    columns=["dark_caption", "light_caption", "dark_id", "light_id", "similarity"],
)

output_df.to_csv("C:/Users/Benjamin Liu/Downloads/captions_LLAVA.csv", index=False)
