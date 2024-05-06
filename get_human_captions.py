import pandas as pd
import json

# Load the CSV file with the ID pairs
git_captions_file = "C:/Users/ewang/OneDrive/Desktop/Spring_2024/COS_429/Final/captions_GIT_20_percent.csv"
git_captions = pd.read_csv(git_captions_file)

# Load the JSON file with human captions
human_captions_file = "C:/Users/ewang/OneDrive/Desktop/Spring_2024/COS_429/Final/imagecaptioning-bias/annotations/captions_val2014.json"
with open(human_captions_file, "r") as file:
    human_captions_data = json.load(file)

# Create a dictionary to map image IDs to captions
id_to_caption = {}
for annotation in human_captions_data["annotations"]:
    image_id = annotation["image_id"]
    caption = annotation["caption"]
    if image_id not in id_to_caption:
        id_to_caption[image_id] = []
    id_to_caption[image_id].append(caption)

# Create a new DataFrame to replicate the structure
new_captions_data = {
    "light_caption": [],
    "dark_caption": [],
    "light_id": [],
    "dark_id": [],
}

for index, row in git_captions.iterrows():
    light_id = row["light_id"]
    dark_id = row["dark_id"]
    light_captions = id_to_caption.get(light_id, [""])[0]
    dark_captions = id_to_caption.get(dark_id, [""])[0]
    new_captions_data["light_caption"].append(light_captions)
    new_captions_data["dark_caption"].append(dark_captions)
    new_captions_data["light_id"].append(light_id)
    new_captions_data["dark_id"].append(dark_id)

human_captions = pd.DataFrame(new_captions_data)

# Display the new DataFrame
print(human_captions.head())

# Save the new DataFrame to a CSV file if needed
human_captions.to_csv("captions_human_20_percent.csv", index=False)
