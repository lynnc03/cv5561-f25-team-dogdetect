# Dog Image Generation with Diffusion

## Team Members

| Name | Email |
|------|-------|
| **Joseph Schaak** | schaa140@umn.edu | 
| **Ning-Shan Chang** | chan2497@umn.edu | 
| **Wenhui Cheng** | chen9005@umn.edu | 

## Table of Contents

-   [Overview](#overview)
-   [Dependencies](#dependencies)
-   [Installation](#installation)
-   [How to Run the Code](#how-to-run-the-code)
-   [Results](#results)
-   [References](#references)

# Part A - SD v1.5 and Low Rank Adaptation

## Overview

### Objective:
Generate more realistic Golden Retriever images using LoRA (Low-Rank Adaptation) on a pre-existing diffusion model (Stable Diffusion)

### Motivation
Generating lifelike Golden Retriever images not only advances diffusion model research but also highlights how AI-generated animals can bring emotional comfort and support mental health.

### Input/Output
Text prompt → Image of Golden Retriever.

### Method
Use the LoRA fine-tuning on the diffusion model to adapt the model for golden retriever style


## Dependencies

* transformers
* diffusers
* accelerator
* torch
* numpy
* datasets
* huggingface_hub
* peft
* gradio
* safetensors


## Installation

```bash
# Clone repo
git clone https://github.com/lynnc03/cv5561-f25-team-dogdetect
cd cv5561-f25-team-dogdetect

# (optional) Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

## How to Run the Code

### To initiate LoRA fine-tuning:
```bash
python -m accelerate.commands.launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path="/path/to/stable-diffusion-v1-5" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/path/to/tuning/output" \
  --train_data_dir="/path/to/golden_retriever_imgs"
```
(Note: Different hyperparameters will provide varying results)

### To generate images with default SD v1.5:
```bash
python sd-pretrained.py
```
(Note: Directories, hardware optimizations, and input text prompts are defined within the script rather than as parameters. Change these in sd-pretrained.py to match your machine and desired prompts)

### To generate images with LoRA fine-tuned SD v1.5:
```bash
python sd-lora.py
```
(Note: Directories, hardware optimizations, and input text prompts are defined within the script rather than as parameters. Change these in sd-lora.py to match your machine and desired prompts)

### To generate FID, LPIPS, and CLIP scores:

## Results

### Example Results
| Prompt | SD v1.5 |SD + fine-tuning steps=5 | SD + fine-tuning steps=2000 | SD + fine-tuning steps=15000 |
|--------|---------|-------------------------|-----------------------------|-----------------------------|
| "Golden Retriever" | ![EG1](imgs/eg-pretrained.png) | ![EG2](imgs/eg-lora-5.png) | ![EG3](imgs/eg-lora-2000.png) | ![EG4](imgs/eg-lora-15000.png) |

### Evalutation

```bash
import os
from PIL import Image
import torch
from torchvision import transforms
from pytorch_fid import fid_score
import lpips
import clip
import numpy as np
from tqdm import tqdm

#＃ Paths
real_images_path = "/projects/standard/csci5561/shared/G4/Images/n02099601-golden_retriever"
generated_images_path = "/projects/standard/csci5561/shared/G4/results/tuning-outputs150"

# Temporary folders for resized images
tmp_real = "/tmp/real_resized"
tmp_gen = "/tmp/gen_resized"
os.makedirs(tmp_real, exist_ok=True)
os.makedirs(tmp_gen, exist_ok=True)

# Transform function
resize_transform = transforms.Compose([
    transforms.Resize((299, 299)), 
    transforms.ToTensor()
])

def preprocess_folder(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(input_folder, filename)).convert("RGB")
            img = resize_transform(img)
            out_path = os.path.join(output_folder, filename)
            transforms.ToPILImage()(img).save(out_path)

print("Preprocessing real images...")
preprocess_folder(real_images_path, tmp_real)

print("Preprocessing generated images...")
preprocess_folder(generated_images_path, tmp_gen)

# 1. FID
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Calculating FID...")
fid_value = fid_score.calculate_fid_given_paths(
    [tmp_real, tmp_gen],
    batch_size=8,
    device=device,
    dims=2048
)
print("FID:", fid_value)

# 2. LPIPS
print("Calculating LPIPS...")
lpips_model = lpips.LPIPS(net='alex').to(device)

def load_tensor_image(path):
    img = Image.open(path).convert("RGB")
    img = transforms.Resize((64, 64))(img) 
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return tensor * 2 - 1  # Normalize to [-1, 1]

lpips_scores = []
gen_files = sorted([f for f in os.listdir(tmp_gen) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
real_files = sorted([f for f in os.listdir(tmp_real) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

for g_file, r_file in zip(gen_files, real_files):
    g_img = load_tensor_image(os.path.join(tmp_gen, g_file))
    r_img = load_tensor_image(os.path.join(tmp_real, r_file))
    score = lpips_model(g_img, r_img)
    lpips_scores.append(score.item())

print("LPIPS mean:", np.mean(lpips_scores))

# 3. CLIP similarity
print("Calculating CLIP similarity...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def clip_encode(path):
    img = Image.open(path).convert("RGB")
    image_input = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_input)
        embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding

clip_scores = []

for g_file, r_file in zip(gen_files, real_files):
    g_emb = clip_encode(os.path.join(tmp_gen, g_file))
    r_emb = clip_encode(os.path.join(tmp_real, r_file))
    similarity = (g_emb @ r_emb.T).item()
    clip_scores.append(similarity)

print("CLIP similarity mean:", np.mean(clip_scores))

```

| Method | FID | LPIPS | CLIP |
|--------|-----|-------|------|
| SD v1.5 |  72.45 | 0.4444 | 0.7529 |
| SD + fine-tuning steps=5 | 74.30 | 0.4371 | 0.7578 |
| SD + fine-tuning steps=2000 | 73.87 | 0.4401 | 0.7612 |
| SD + fine-tuning steps=15000 | 73.64 | 0.4307 | 0.7621 |


## References

**deepanway.** (2023). <br>
*text_to_image.*  <br>
Available at: https://huggingface.co/spaces/declare-lab/mustango/blob/293436cffbc0e0e15507bbb031dac78247df681f/diffusers/examples/text_to_image/train_text_to_image_lora.py  <br>
Accessed: 2025-11-18.

**Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B.** (2022).  <br>
*High-Resolution Image Synthesis With Latent Diffusion Models.*  <br>
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10684–10695.  <br>
https://arxiv.org/abs/2112.10752


**Stanford Vision Lab.** (2011). <br>
*ImageNet-Dogs Dataset. Stanford University.* <br>
Available at: http://vision.stanford.edu/aditya86/ImageNetDogs/ <br>
Accessed: 2025-11-06.

# Part B - Built Diffusion Model
