import torch
import clip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

# Install clip if needed: pip install openai-clip

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)

image_folder = "LoomRAG/data/images/LoomGen/LoomGen"
embeddings_folder = "LoomRAG/embeddings"
os.makedirs(embeddings_folder, exist_ok=True)

image_files = [f for f in os.listdir(image_folder)
               if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

embeddings = {}

print(f"Embedding {len(image_files)} images...\n")

for filename in tqdm(image_files):
    try:
        img_path = os.path.join(image_folder, filename)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.encode_image(image)

        embeddings[filename] = embedding.cpu().numpy()
    except Exception as e:
        print(f"Error: {filename} - {e}")

# Save embeddings
np.save(os.path.join(embeddings_folder, "clip_embeddings.npy"), embeddings)

print(f"\n Saved {len(embeddings)} CLIP embeddings!")
print(f"Embedding shape: {embeddings[image_files[0]].shape}")
