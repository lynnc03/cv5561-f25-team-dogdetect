"""
Stanford Dogs Dataset - 数据加载器和预处理
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np


class StanfordDogsDataset(Dataset):
    """
    Stanford Dogs 数据集

    参数:
        root_dir: 数据集根目录
        image_size: 输出图像大小
        transform: 自定义变换 (如果为None则使用默认变换)
        use_bbox: 是否使用边界框裁剪
        split: 数据集划分 ('train', 'val', 'test', None表示全部)
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
        transform: Optional[Callable] = None,
        use_bbox: bool = True,
        split: Optional[str] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "Images"
        self.annotation_dir = self.root_dir / "Annotation"
        self.image_size = image_size
        self.use_bbox = use_bbox
        self.split = split

        # 加载数据
        self.samples, self.class_to_idx, self.idx_to_class = self._load_dataset()

        # 数据集划分
        if split is not None:
            self.samples = self._split_dataset(train_ratio, val_ratio, seed)

        # 设置变换
        self.transform = transform if transform is not None else self._get_default_transform()

        print(f"Loaded {len(self.samples)} samples, {len(self.class_to_idx)} classes")

    def _load_dataset(self) -> Tuple[List[Dict], Dict[str, int], Dict[int, str]]:
        """加载数据集元信息"""
        samples = []
        class_to_idx = {}
        idx_to_class = {}

        # 遍历所有犬种文件夹
        breed_dirs = sorted([d for d in self.image_dir.iterdir() if d.is_dir()])

        for idx, breed_dir in enumerate(breed_dirs):
            breed_name = breed_dir.name
            class_to_idx[breed_name] = idx
            idx_to_class[idx] = breed_name

            # 获取该犬种的所有图像
            image_files = list(breed_dir.glob("*.jpg"))

            for img_path in image_files:
                # 对应的标注文件
                img_name = img_path.stem
                annotation_path = self.annotation_dir / breed_name / img_name

                sample = {
                    "image_path": str(img_path),
                    "annotation_path": str(annotation_path) if annotation_path.exists() else None,
                    "class_idx": idx,
                    "class_name": breed_name,
                }
                samples.append(sample)

        return samples, class_to_idx, idx_to_class

    def _split_dataset(self, train_ratio: float, val_ratio: float, seed: int) -> List[Dict]:
        """划分数据集"""
        np.random.seed(seed)
        indices = np.random.permutation(len(self.samples))

        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)

        if self.split == "train":
            selected_indices = indices[:n_train]
        elif self.split == "val":
            selected_indices = indices[n_train:n_train + n_val]
        elif self.split == "test":
            selected_indices = indices[n_train + n_val:]
        else:
            selected_indices = indices

        return [self.samples[i] for i in selected_indices]

    def _parse_annotation(self, annotation_path: str) -> Optional[Tuple[int, int, int, int]]:
        """解析XML标注文件获取边界框"""
        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()

            # 获取边界框
            bbox = root.find(".//bndbox")
            if bbox is not None:
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)
                return (xmin, ymin, xmax, ymax)
        except Exception:
            pass
        return None

    def _get_default_transform(self) -> transforms.Compose:
        """获取默认变换 (v4优化: 加强数据增强 + diffusion标准归一化)"""
        if self.split == "train":
            return transforms.Compose([
                # 随机裁剪增强多样性
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                # 加强颜色抖动
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # 随机旋转
                transforms.RandomRotation(15),
                # 随机仿射变换
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                # 使用diffusion标准归一化 [-1, 1]
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                # 使用diffusion标准归一化 [-1, 1]
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # 加载图像
        image = Image.open(sample["image_path"]).convert("RGB")

        # 使用边界框裁剪 (如果启用)
        if self.use_bbox and sample["annotation_path"]:
            bbox = self._parse_annotation(sample["annotation_path"])
            if bbox is not None:
                # 稍微扩大边界框
                xmin, ymin, xmax, ymax = bbox
                w, h = image.size
                pad = 0.1  # 10% padding
                xmin = max(0, int(xmin - (xmax - xmin) * pad))
                ymin = max(0, int(ymin - (ymax - ymin) * pad))
                xmax = min(w, int(xmax + (xmax - xmin) * pad))
                ymax = min(h, int(ymax + (ymax - ymin) * pad))
                image = image.crop((xmin, ymin, xmax, ymax))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "class_idx": torch.tensor(sample["class_idx"], dtype=torch.long),
            "class_name": sample["class_name"],
        }


class StanfordDogsDatasetSimple(Dataset):
    """
    简化版数据集 - 用于无条件生成 (不使用类别标签)
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "Images"
        self.image_size = image_size

        # 收集所有图像路径
        self.image_paths = list(self.image_dir.rglob("*.jpg"))
        print(f"Found {len(self.image_paths)} images")

        # 设置变换
        self.transform = transform if transform is not None else self._get_default_transform()

    def _get_default_transform(self) -> transforms.Compose:
        """默认变换 - 适用于diffusion模型"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # 归一化到 [-1, 1] (diffusion模型常用)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def get_transforms(
    image_size: int,
    split: str = "train",
    normalize_to_minus1_1: bool = True,
) -> transforms.Compose:
    """
    获取数据变换

    参数:
        image_size: 目标图像大小
        split: 数据集划分 ('train' 或 'val'/'test')
        normalize_to_minus1_1: 是否归一化到[-1,1] (diffusion常用)
    """
    if normalize_to_minus1_1:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if split == "train":
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.1)),  # 稍大一些以便随机裁剪
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])


def create_dataloaders(
    root_dir: str,
    image_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    use_class_condition: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    创建训练、验证、测试数据加载器

    返回:
        train_loader, val_loader, test_loader, dataset_info
    """
    if use_class_condition:
        # 使用带类别标签的数据集
        train_transform = get_transforms(image_size, "train")
        val_transform = get_transforms(image_size, "val")

        train_dataset = StanfordDogsDataset(
            root_dir=root_dir,
            image_size=image_size,
            transform=train_transform,
            split="train",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        val_dataset = StanfordDogsDataset(
            root_dir=root_dir,
            image_size=image_size,
            transform=val_transform,
            split="val",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        test_dataset = StanfordDogsDataset(
            root_dir=root_dir,
            image_size=image_size,
            transform=val_transform,
            split="test",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        def collate_fn(batch):
            images = torch.stack([item["image"] for item in batch])
            class_idx = torch.stack([item["class_idx"] for item in batch])
            return {"image": images, "class_idx": class_idx}

        dataset_info = {
            "num_classes": len(train_dataset.class_to_idx),
            "class_to_idx": train_dataset.class_to_idx,
            "idx_to_class": train_dataset.idx_to_class,
            "num_train": len(train_dataset),
            "num_val": len(val_dataset),
            "num_test": len(test_dataset),
        }
    else:
        # 简化版数据集 (无条件生成)
        full_dataset = StanfordDogsDatasetSimple(
            root_dir=root_dir,
            image_size=image_size,
        )

        # 手动划分
        n_total = len(full_dataset)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )

        collate_fn = None
        dataset_info = {
            "num_classes": 0,
            "num_train": len(train_dataset),
            "num_val": len(val_dataset),
            "num_test": len(test_dataset),
        }

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, dataset_info


def denormalize(tensor: torch.Tensor, mean: Tuple = (0.5, 0.5, 0.5), std: Tuple = (0.5, 0.5, 0.5)) -> torch.Tensor:
    """反归一化图像张量"""
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """将张量转换为PIL图像"""
    # 假设输入范围是[-1, 1]
    tensor = denormalize(tensor.unsqueeze(0) if tensor.dim() == 3 else tensor)
    tensor = tensor.squeeze(0).clamp(0, 1)
    tensor = (tensor * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(tensor)


# 测试代码
if __name__ == "__main__":
    root_dir = "/users/5/chen9005/FFF/dogs_diff"

    print("=" * 50)
    print("测试带类别标签的数据集")
    print("=" * 50)

    train_loader, val_loader, test_loader, info = create_dataloaders(
        root_dir=root_dir,
        image_size=256,
        batch_size=16,
        use_class_condition=True,
    )

    print(f"\n数据集信息:")
    print(f"  类别数: {info['num_classes']}")
    print(f"  训练集: {info['num_train']} 样本")
    print(f"  验证集: {info['num_val']} 样本")
    print(f"  测试集: {info['num_test']} 样本")

    # 获取一个batch
    batch = next(iter(train_loader))
    print(f"\nBatch信息:")
    print(f"  图像形状: {batch['image'].shape}")
    print(f"  类别形状: {batch['class_idx'].shape}")
    print(f"  图像范围: [{batch['image'].min():.2f}, {batch['image'].max():.2f}]")

    print("\n" + "=" * 50)
    print("测试简化版数据集 (无条件生成)")
    print("=" * 50)

    train_loader2, _, _, info2 = create_dataloaders(
        root_dir=root_dir,
        image_size=128,
        batch_size=32,
        use_class_condition=False,
    )

    batch2 = next(iter(train_loader2))
    print(f"\nBatch信息:")
    print(f"  图像形状: {batch2.shape}")
    print(f"  图像范围: [{batch2.min():.2f}, {batch2.max():.2f}]")
