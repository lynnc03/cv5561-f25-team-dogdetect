"""
Stanford Dogs Dataset - 数据预处理和分析工具
"""
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


def analyze_dataset(root_dir: str) -> Dict:
    """
    分析数据集统计信息

    返回:
        包含数据集统计信息的字典
    """
    root_dir = Path(root_dir)
    image_dir = root_dir / "Images"

    stats = {
        "total_images": 0,
        "num_classes": 0,
        "images_per_class": {},
        "image_sizes": [],
        "aspect_ratios": [],
        "file_sizes": [],
        "class_names": [],
    }

    breed_dirs = sorted([d for d in image_dir.iterdir() if d.is_dir()])
    stats["num_classes"] = len(breed_dirs)

    for breed_dir in tqdm(breed_dirs, desc="Analyzing dataset"):
        breed_name = breed_dir.name
        stats["class_names"].append(breed_name)

        image_files = list(breed_dir.glob("*.jpg"))
        stats["images_per_class"][breed_name] = len(image_files)
        stats["total_images"] += len(image_files)

        # 采样分析图像尺寸 (只分析前10张)
        for img_path in image_files[:10]:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    stats["image_sizes"].append((w, h))
                    stats["aspect_ratios"].append(w / h)
                stats["file_sizes"].append(os.path.getsize(img_path))
            except Exception:
                pass

    # 计算汇总统计
    if stats["image_sizes"]:
        widths = [s[0] for s in stats["image_sizes"]]
        heights = [s[1] for s in stats["image_sizes"]]
        stats["size_stats"] = {
            "width": {"min": min(widths), "max": max(widths), "mean": np.mean(widths)},
            "height": {"min": min(heights), "max": max(heights), "mean": np.mean(heights)},
            "aspect_ratio": {
                "min": min(stats["aspect_ratios"]),
                "max": max(stats["aspect_ratios"]),
                "mean": np.mean(stats["aspect_ratios"]),
            },
        }

    if stats["file_sizes"]:
        stats["file_size_stats"] = {
            "min_kb": min(stats["file_sizes"]) / 1024,
            "max_kb": max(stats["file_sizes"]) / 1024,
            "mean_kb": np.mean(stats["file_sizes"]) / 1024,
            "total_mb": sum(stats["file_sizes"]) / (1024 * 1024),
        }

    # 每类样本数统计
    class_counts = list(stats["images_per_class"].values())
    stats["class_distribution"] = {
        "min": min(class_counts),
        "max": max(class_counts),
        "mean": np.mean(class_counts),
        "std": np.std(class_counts),
    }

    return stats


def compute_dataset_statistics(
    root_dir: str,
    image_size: int = 256,
    num_samples: int = 1000,
) -> Dict[str, Tuple[float, float, float]]:
    """
    计算数据集的均值和标准差 (用于归一化)

    参数:
        root_dir: 数据集根目录
        image_size: 调整图像大小
        num_samples: 采样数量

    返回:
        {"mean": (r, g, b), "std": (r, g, b)}
    """
    from torchvision import transforms

    root_dir = Path(root_dir)
    image_dir = root_dir / "Images"

    # 收集图像路径
    image_paths = list(image_dir.rglob("*.jpg"))
    np.random.shuffle(image_paths)
    image_paths = image_paths[:num_samples]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # 计算均值
    mean_sum = np.zeros(3)
    for img_path in tqdm(image_paths, desc="Computing mean"):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).numpy()
            mean_sum += tensor.mean(axis=(1, 2))
        except Exception:
            pass

    mean = mean_sum / len(image_paths)

    # 计算标准差
    std_sum = np.zeros(3)
    for img_path in tqdm(image_paths, desc="Computing std"):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).numpy()
            std_sum += ((tensor - mean.reshape(3, 1, 1)) ** 2).mean(axis=(1, 2))
        except Exception:
            pass

    std = np.sqrt(std_sum / len(image_paths))

    return {
        "mean": tuple(mean),
        "std": tuple(std),
    }


def create_preprocessed_dataset(
    root_dir: str,
    output_dir: str,
    image_size: int = 256,
    format: str = "jpg",
    quality: int = 95,
    num_workers: int = 8,
):
    """
    预处理并保存调整大小后的图像

    参数:
        root_dir: 原始数据集目录
        output_dir: 输出目录
        image_size: 目标图像大小
        format: 输出格式 ('jpg', 'png')
        quality: JPEG质量 (1-100)
        num_workers: 并行工作进程数
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    image_dir = root_dir / "Images"
    output_image_dir = output_dir / "Images"

    # 创建输出目录
    output_image_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有图像
    image_paths = []
    breed_dirs = sorted([d for d in image_dir.iterdir() if d.is_dir()])

    for breed_dir in breed_dirs:
        breed_name = breed_dir.name
        (output_image_dir / breed_name).mkdir(exist_ok=True)

        for img_path in breed_dir.glob("*.jpg"):
            output_path = output_image_dir / breed_name / f"{img_path.stem}.{format}"
            image_paths.append((str(img_path), str(output_path)))

    def process_image(args):
        src_path, dst_path = args
        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                # 保持宽高比的中心裁剪
                w, h = img.size
                if w > h:
                    left = (w - h) // 2
                    img = img.crop((left, 0, left + h, h))
                elif h > w:
                    top = (h - w) // 2
                    img = img.crop((0, top, w, top + w))

                img = img.resize((image_size, image_size), Image.LANCZOS)

                if format == "jpg":
                    img.save(dst_path, "JPEG", quality=quality)
                else:
                    img.save(dst_path, "PNG")
                return True
        except Exception as e:
            return False

    # 并行处理
    success_count = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_image, args): args for args in image_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            if future.result():
                success_count += 1

    print(f"Processed {success_count}/{len(image_paths)} images")
    print(f"Output saved to: {output_dir}")

    # 保存元数据
    metadata = {
        "image_size": image_size,
        "format": format,
        "total_images": success_count,
        "num_classes": len(breed_dirs),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def create_train_val_test_split(
    root_dir: str,
    output_file: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    创建数据集划分文件

    参数:
        root_dir: 数据集根目录
        output_file: 输出JSON文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
    """
    np.random.seed(seed)

    root_dir = Path(root_dir)
    image_dir = root_dir / "Images"

    splits = {"train": [], "val": [], "test": []}
    class_to_idx = {}

    breed_dirs = sorted([d for d in image_dir.iterdir() if d.is_dir()])

    for idx, breed_dir in enumerate(breed_dirs):
        breed_name = breed_dir.name
        class_to_idx[breed_name] = idx

        image_files = list(breed_dir.glob("*.jpg"))
        np.random.shuffle(image_files)

        n_train = int(len(image_files) * train_ratio)
        n_val = int(len(image_files) * val_ratio)

        for i, img_path in enumerate(image_files):
            relative_path = str(img_path.relative_to(root_dir))
            entry = {"path": relative_path, "class_idx": idx, "class_name": breed_name}

            if i < n_train:
                splits["train"].append(entry)
            elif i < n_train + n_val:
                splits["val"].append(entry)
            else:
                splits["test"].append(entry)

    # 打乱每个split
    for split in splits.values():
        np.random.shuffle(split)

    output_data = {
        "splits": splits,
        "class_to_idx": class_to_idx,
        "idx_to_class": {v: k for k, v in class_to_idx.items()},
        "statistics": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "total": sum(len(s) for s in splits.values()),
            "num_classes": len(class_to_idx),
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Split file saved to: {output_file}")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")


def visualize_dataset_statistics(stats: Dict, save_path: Optional[str] = None):
    """
    可视化数据集统计信息

    参数:
        stats: analyze_dataset() 返回的统计字典
        save_path: 保存图像路径 (如果为None则显示)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 每类样本数分布
    ax1 = axes[0, 0]
    class_counts = list(stats["images_per_class"].values())
    ax1.hist(class_counts, bins=20, edgecolor="black", alpha=0.7)
    ax1.axvline(np.mean(class_counts), color="red", linestyle="--", label=f"Mean: {np.mean(class_counts):.1f}")
    ax1.set_xlabel("Number of Images per Class")
    ax1.set_ylabel("Number of Classes")
    ax1.set_title("Class Distribution")
    ax1.legend()

    # 2. 图像尺寸分布
    ax2 = axes[0, 1]
    if stats["image_sizes"]:
        widths = [s[0] for s in stats["image_sizes"]]
        heights = [s[1] for s in stats["image_sizes"]]
        ax2.scatter(widths, heights, alpha=0.5)
        ax2.set_xlabel("Width (pixels)")
        ax2.set_ylabel("Height (pixels)")
        ax2.set_title("Image Size Distribution")

    # 3. 宽高比分布
    ax3 = axes[1, 0]
    if stats["aspect_ratios"]:
        ax3.hist(stats["aspect_ratios"], bins=30, edgecolor="black", alpha=0.7)
        ax3.axvline(1.0, color="red", linestyle="--", label="Square (1:1)")
        ax3.set_xlabel("Aspect Ratio (W/H)")
        ax3.set_ylabel("Count")
        ax3.set_title("Aspect Ratio Distribution")
        ax3.legend()

    # 4. 前20个类别的样本数
    ax4 = axes[1, 1]
    sorted_classes = sorted(stats["images_per_class"].items(), key=lambda x: x[1], reverse=True)[:20]
    names = [c[0].split("-")[-1][:15] for c in sorted_classes]  # 简化名称
    counts = [c[1] for c in sorted_classes]
    ax4.barh(names, counts, color="steelblue")
    ax4.set_xlabel("Number of Images")
    ax4.set_title("Top 20 Classes by Sample Count")
    ax4.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_sample_images(
    root_dir: str,
    num_classes: int = 10,
    images_per_class: int = 5,
    image_size: int = 128,
    save_path: Optional[str] = None,
):
    """
    可视化样本图像

    参数:
        root_dir: 数据集根目录
        num_classes: 显示的类别数
        images_per_class: 每个类别显示的图像数
        image_size: 显示图像大小
        save_path: 保存路径
    """
    root_dir = Path(root_dir)
    image_dir = root_dir / "Images"

    breed_dirs = sorted([d for d in image_dir.iterdir() if d.is_dir()])[:num_classes]

    fig, axes = plt.subplots(num_classes, images_per_class, figsize=(images_per_class * 2, num_classes * 2))

    for i, breed_dir in enumerate(breed_dirs):
        breed_name = breed_dir.name.split("-")[-1]
        image_files = list(breed_dir.glob("*.jpg"))[:images_per_class]

        for j, img_path in enumerate(image_files):
            ax = axes[i, j] if num_classes > 1 else axes[j]
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((image_size, image_size))
                ax.imshow(img)
            except Exception:
                pass

            ax.axis("off")
            if j == 0:
                ax.set_title(breed_name[:20], fontsize=8, loc="left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    root_dir = "/users/5/chen9005/FFF/dogs_diff"
    output_dir = "/users/5/chen9005/FFF/dogs_diff/outputs"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Stanford Dogs Dataset Analysis")
    print("=" * 60)

    # 1. 分析数据集
    print("\n1. Analyzing dataset...")
    stats = analyze_dataset(root_dir)

    print(f"\nDataset Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Number of classes: {stats['num_classes']}")
    print(f"  Class distribution: min={stats['class_distribution']['min']}, "
          f"max={stats['class_distribution']['max']}, mean={stats['class_distribution']['mean']:.1f}")

    if "size_stats" in stats:
        print(f"\nImage Size Statistics:")
        print(f"  Width: {stats['size_stats']['width']}")
        print(f"  Height: {stats['size_stats']['height']}")
        print(f"  Aspect ratio: {stats['size_stats']['aspect_ratio']}")

    # 2. 计算数据集统计量
    print("\n2. Computing dataset normalization statistics...")
    norm_stats = compute_dataset_statistics(root_dir, image_size=256, num_samples=500)
    print(f"  Mean: {norm_stats['mean']}")
    print(f"  Std: {norm_stats['std']}")

    # 3. 创建数据集划分
    print("\n3. Creating train/val/test split...")
    split_file = os.path.join(output_dir, "dataset_split.json")
    create_train_val_test_split(root_dir, split_file)

    # 4. 可视化
    print("\n4. Generating visualizations...")
    visualize_dataset_statistics(stats, save_path=os.path.join(output_dir, "dataset_statistics.png"))
    visualize_sample_images(root_dir, save_path=os.path.join(output_dir, "sample_images.png"))

    print("\nDone!")
