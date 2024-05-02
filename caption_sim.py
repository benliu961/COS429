import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_captions(dark, light):
    sentences = [dark, light]
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
    embeddings = model.encode(sentences)
    dist = np.linalg.norm(embeddings[0]-embeddings[1])
    return dist

captions = "C:/Users/Benjamin Liu/Downloads/captions_GIT.csv"
data = pd.read_csv(captions)

output = []

for index, row in tqdm(
    data.iterrows(), total=data.shape[0], desc="Processing captions"
):
    dark_id = str(int(row["dark_id"]))
    light_id = str(int(row["light_id"]))
    dark_caption = row["dark_caption"]
    light_caption = row["light_caption"]
    caption_similarity = evaluate_captions(dark_caption, light_caption)
    output.append([dark_id, light_id, row["similarity"], caption_similarity])

output_df = pd.DataFrame(output, columns=["dark_id", "light_id", "similarity", "caption similarity"])

output_df.to_csv("C:/Users/Benjamin Liu/Downloads/caption_similarity_GIT.csv", index=False)