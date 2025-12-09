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
real_images_path = "/path/to/Images/n02099601-golden_retriever"
generated_images_path = "/path/to/results/tuning-outputs150"

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