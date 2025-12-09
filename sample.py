"""
Stanford Dogs Diffusion - 采样脚本

从训练好的模型生成狗图像
所有配置已硬编码，直接运行即可
"""
import os
import sys
from pathlib import Path

import torch
import torchvision.utils as vutils
import numpy as np
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, DataConfig, ModelConfig, DiffusionConfig, TrainingConfig
from models import UNetModel, count_parameters
from diffusion import DDIMScheduler, DDPMScheduler
from dataset import denormalize


# ============================================================
# 硬编码配置 - 直接修改这里的值即可
# ============================================================

# 检查点路径
CHECKPOINT_PATH = "/users/5/chen9005/FFF/dogs_diff/checkpoints/best_model.pt"

# 输出配置
OUTPUT_IMAGE = "/users/5/chen9005/FFF/dogs_diff/outputs/generated_samples.png"
OUTPUT_DIR = "/users/5/chen9005/FFF/dogs_diff/outputs/individual_samples"  # 设为 None 则不保存单独图像
SAVE_INDIVIDUAL = True  # 是否保存单独的图像文件

# 生成配置
NUM_SAMPLES = 16  # 生成样本数量
CLASS_LABELS = None  # 指定类别标签列表，None 表示随机选择，例如 [0, 1, 2, 3]
GUIDANCE_SCALE = 7.5  # Classifier-Free Guidance 强度 (推荐3.0-10.0，7.5效果最好)
NUM_INFERENCE_STEPS = 50  # DDIM 采样步数 (越多质量越好，但更慢)
USE_DDIM = True  # 是否使用 DDIM (False 则使用 DDPM，需要1000步)
ETA = 0.0  # DDIM eta 参数 (0=确定性, 1=DDPM)
SEED = 42  # 随机种子，设为 None 则不固定
NROW = 4  # 网格图每行图像数

# 重要: 确保采样时使用类别条件！
# 如果生成噪声，请检查:
# 1. GUIDANCE_SCALE > 1.0 (启用CFG)
# 2. 模型训练时使用了类别条件
# 3. CLASS_LABELS 不为空或设为 None 让程序自动随机选择

# 设备
DEVICE = "cuda"  # "cuda" 或 "cpu"

# ============================================================


def load_model(checkpoint_path: str, device: str = "cuda"):
    """加载训练好的模型"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 获取配置
    config = checkpoint.get("config", None)
    if config is None:
        # 使用默认配置
        print("No config found in checkpoint, using default config")
        config = Config()
        config.data = DataConfig()
        config.model = ModelConfig()
        config.diffusion = DiffusionConfig()
        config.training = TrainingConfig()

    # 创建模型
    num_classes = config.model.num_classes if config.model.use_class_condition else None

    model = UNetModel(
        image_size=config.data.image_size,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        model_channels=config.model.model_channels,
        num_res_blocks=config.model.num_res_blocks,
        attention_resolutions=config.model.attention_resolutions,
        dropout=0.0,  # 推理时不使用dropout
        channel_mult=config.model.channel_mult,
        num_classes=num_classes,
        num_heads=8,
        use_scale_shift_norm=True,
        class_dropout_prob=0.0,  # 推理时不需要类别dropout
    ).to(device)

    # 加载权重
    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"]["ema_model"])
        print("Loaded EMA model weights")
    else:
        model.load_state_dict(checkpoint["model"])
        print("Loaded model weights")

    model.eval()
    print(f"Model loaded: {count_parameters(model):,} parameters")

    return model, config


def sample_images(model, config, device):
    """生成图像样本"""
    if SEED is not None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        print(f"Random seed set to {SEED}")

    # 创建调度器
    scheduler = DDIMScheduler(
        num_timesteps=config.diffusion.num_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        beta_schedule=config.diffusion.beta_schedule,
        device=device,
    )

    # 准备类别标签
    if config.model.use_class_condition:
        if CLASS_LABELS is None:
            y = torch.randint(0, config.model.num_classes, (NUM_SAMPLES,), device=device)
            print(f"Randomly selected class labels: {y.tolist()}")
        else:
            if len(CLASS_LABELS) == 1:
                y = torch.full((NUM_SAMPLES,), CLASS_LABELS[0], device=device, dtype=torch.long)
            else:
                labels = CLASS_LABELS * (NUM_SAMPLES // len(CLASS_LABELS) + 1)
                y = torch.tensor(labels[:NUM_SAMPLES], device=device, dtype=torch.long)
            print(f"Using specified class labels: {y.tolist()}")
    else:
        y = None
        print("Unconditional generation (no class labels)")

    # 采样
    print(f"\nGenerating {NUM_SAMPLES} samples...")
    print(f"  Image size: {config.data.image_size}x{config.data.image_size}")
    print(f"  Sampling steps: {NUM_INFERENCE_STEPS}")
    print(f"  Guidance scale: {GUIDANCE_SCALE}")
    print(f"  Method: {'DDIM' if USE_DDIM else 'DDPM'}")

    with torch.no_grad():
        if USE_DDIM:
            samples = scheduler.sample(
                model=model,
                shape=(NUM_SAMPLES, 3, config.data.image_size, config.data.image_size),
                num_inference_steps=NUM_INFERENCE_STEPS,
                eta=ETA,
                y=y,
                guidance_scale=GUIDANCE_SCALE,
                show_progress=True,
            )
        else:
            ddpm_scheduler = DDPMScheduler(
                num_timesteps=config.diffusion.num_timesteps,
                beta_start=config.diffusion.beta_start,
                beta_end=config.diffusion.beta_end,
                beta_schedule=config.diffusion.beta_schedule,
                device=device,
            )
            samples = ddpm_scheduler.sample(
                model=model,
                shape=(NUM_SAMPLES, 3, config.data.image_size, config.data.image_size),
                y=y,
            )

    return samples, y


def save_grid_image(images, output_path, nrow=4):
    """保存网格图像"""
    images = denormalize(images)
    images = images.clamp(0, 1)

    grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(grid_np).save(output_path)
    print(f"\nGrid image saved to: {output_path}")


def save_individual_images(images, output_dir, labels=None):
    """保存单独的图像文件"""
    os.makedirs(output_dir, exist_ok=True)

    images = denormalize(images)
    images = images.clamp(0, 1)

    for i, img in enumerate(images):
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        if labels is not None:
            filename = f"sample_{i:04d}_class{labels[i].item()}.png"
        else:
            filename = f"sample_{i:04d}.png"

        Image.fromarray(img_np).save(os.path.join(output_dir, filename))

    print(f"Saved {len(images)} individual images to: {output_dir}")


def main():
    """主函数"""
    print("=" * 60)
    print("Stanford Dogs Diffusion - Image Generation")
    print("=" * 60)

    # 检查设备
    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    print(f"Using device: {device}")

    # 检查检查点文件
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nError: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please train the model first using: python train.py")
        return

    # 加载模型
    model, config = load_model(CHECKPOINT_PATH, device)

    # 生成样本
    samples, labels = sample_images(model, config, device)

    # 保存网格图
    save_grid_image(samples, OUTPUT_IMAGE, nrow=NROW)

    # 保存单独的图像
    if SAVE_INDIVIDUAL and OUTPUT_DIR is not None:
        save_individual_images(samples, OUTPUT_DIR, labels)

    print("\n" + "=" * 60)
    print("Generation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
