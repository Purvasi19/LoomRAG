from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import os
import json
from tqdm import tqdm
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP2
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto")

image_folder = "LoomRAG/data/images/LoomGen/LoomGen"
metadata = {}

image_files = sorted([f for f in os.listdir(image_folder)
                      if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])

print(f"Generating rich metadata for {len(image_files)} images...\n")

for filename in tqdm(image_files):
    img_path = os.path.join(image_folder, filename)

    try:
        image = Image.open(img_path)
        inputs = processor(image, text="Describe this Assamese textile motif pattern in detail:", return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=150)
        description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        metadata[filename] = {
            "filename": filename,
            "description": description
        }
    except Exception as e:
        print(f"Error {filename}: {e}")
        metadata[filename] = {"filename": filename, "description": "Assamese textile motif pattern"}

# Save metadata
with open("LoomRAG/data/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n Generated detailed metadata for {len(metadata)} images!")

# Show samples
print("\n📸 Sample descriptions:")
for filename in list(metadata.keys())[:5]:
    print(f"\n{filename}:")
    print(f"  {metadata[filename]['description'][:150]}...")
