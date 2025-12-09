from diffusers import StableDiffusionPipeline
import torch

if torch.backends.mps.is_available():
    device = "mps"          # Mac M1/M2
elif torch.cuda.is_available():
    device = "cuda"         # NVIDIA GPU
else:
    device = "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
)

pipe.to(device)

# M1/M2 specific optimization
if device == "mps":
    pipe.enable_model_cpu_offload()
    
def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)

pipe.safety_checker = dummy_safety_checker
    
prompts = []
prompts.append("golden retriever")
prompts.append("friendly golden retriever")
prompts.append("golden retriever running")
prompts.append("golden retriever barking")

autocast_device = "mps" if device == "mps" else "cuda"

with torch.autocast(autocast_device, enabled=(device != "cpu")):
    for p in prompts:
        for i in range(50):
            image = pipe(p,
                         num_inference_steps=20).images[0]
            image.save(f"//path/to/results/pretrained-outputs/SD_V1.5_2/{p}-{i}.png") 
print("Image generation completed!")