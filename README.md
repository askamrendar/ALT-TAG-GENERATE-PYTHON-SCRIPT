!pip install transformers

from transformers import AutoProcessor, AutoModelForCausalLM
import requests
import torch
from tqdm import tqdm
from PIL import Image
import pandas as pd

df = pd.read_csv("//content/DG ALT TAG MISSING - ALT TAG MISSING.csv")
alt_text = []
git_processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
git_model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")
device = "cuda" if torch.cuda.is_available() else "cpu"
git_model.to(device)

def generate_caption(processor, model, image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

for i in tqdm(range(len(df))):
    url = str(df.iloc[i][0])
    try:
        image = Image.open(requests.get(url, stream=True).raw)
        caption = generate_caption(git_processor, git_model, image)
        alt_text.append(caption)
    except:
        alt_text.append("NaN")

df['alt_text'] = alt_text
df.to_csv("alt-text.csv", index=False)



