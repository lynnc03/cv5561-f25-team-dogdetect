"""
Stanford Dogs Diffusion Model - Configuration
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import torch


@dataclass
class DataConfig:
    """数据配置"""
    data_root: str = "/users/5/chen9005/FFF/dogs_diff"
    image_dir: str = "Images"
    annotation_dir: str = "Annotation"

    # 图像预处理参数
    image_size: int = 256  # 可选: 64, 128, 256, 512
    num_classes: int = 120  # Stanford Dogs 有120个犬种

    # 数据集划分
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # 数据增强
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    random_crop_scale: Tuple[float, float] = (0.8, 1.0)
    color_jitter: bool = True
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    """模型配置 - DDPM/UNet"""
    # UNet 架构参数
    in_channels: int = 3
    out_channels: int = 3
    model_channels: int = 128  # 基础通道数
    channel_mult: Tuple[int, ...] = (1, 2, 2, 4)  # 各层通道倍数
    num_res_blocks: int = 2  # 每个分辨率的残差块数量
    attention_resolutions: Tuple[int, ...] = (16, 8)  # 在哪些分辨率使用注意力
    dropout: float = 0.1

    # 条件生成参数
    num_classes: int = 120  # 犬种类别数
    class_embed_dim: int = 256  # 类别嵌入维度
    use_class_condition: bool = True  # 是否使用类别条件

    # 时间嵌入
    time_embed_dim: int = 512


@dataclass
class DiffusionConfig:
    """扩散过程配置"""
    # 噪声调度
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # "linear", "cosine", "quadratic"

    # 采样参数
    sampling_timesteps: int = 250  # DDIM 采样步数
    ddim_sampling_eta: float = 0.0  # 0 for DDIM, 1 for DDPM

    # 损失函数
    loss_type: str = "l2"  # "l1", "l2", "huber"
    prediction_type: str = "epsilon"  # "epsilon", "v_prediction", "x_start"


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    batch_size: int = 16
    num_epochs: int = 500
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    # 学习率调度
    lr_scheduler: str = "cosine"  # "cosine", "linear", "constant"
    warmup_steps: int = 1000

    # 优化器
    optimizer: str = "adamw"  # "adam", "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # EMA (Exponential Moving Average)
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_update_every: int = 10

    # 梯度相关
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"

    # 检查点和日志
    save_every: int = 5000  # 每N步保存
    sample_every: int = 1000  # 每N步生成样本
    log_every: int = 100  # 每N步记录日志
    num_samples: int = 16  # 每次采样生成的图像数

    # 路径
    output_dir: str = "/users/5/chen9005/FFF/dogs_diff/outputs"
    checkpoint_dir: str = "/users/5/chen9005/FFF/dogs_diff/checkpoints"
    log_dir: str = "/users/5/chen9005/FFF/dogs_diff/logs"

    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True

    # 恢复训练
    resume_from_checkpoint: Optional[str] = None


@dataclass
class Config:
    """总配置"""
    data: DataConfig = None
    model: ModelConfig = None
    diffusion: DiffusionConfig = None
    training: TrainingConfig = None

    # 实验名称
    experiment_name: str = "stanford_dogs_ddpm"
    seed: int = 42

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.diffusion is None:
            self.diffusion = DiffusionConfig()
        if self.training is None:
            self.training = TrainingConfig()


def get_config() -> Config:
    """获取默认配置"""
    return Config()


# 预定义配置模板
def get_small_config() -> Config:
    """小型模型配置 (用于快速实验)"""
    config = Config()
    config.data.image_size = 64
    config.model.model_channels = 64
    config.model.channel_mult = (1, 2, 4)
    config.model.attention_resolutions = (8,)
    config.training.batch_size = 32
    config.diffusion.num_timesteps = 500
    return config


def get_medium_config() -> Config:
    """中型模型配置"""
    config = Config()
    config.data.image_size = 128
    config.model.model_channels = 128
    config.model.channel_mult = (1, 2, 2, 4)
    config.model.attention_resolutions = (16, 8)
    config.training.batch_size = 16
    return config


def get_large_config() -> Config:
    """大型模型配置"""
    config = Config()
    config.data.image_size = 256
    config.model.model_channels = 192
    config.model.channel_mult = (1, 2, 3, 4)
    config.model.attention_resolutions = (32, 16, 8)
    config.model.num_res_blocks = 3
    config.training.batch_size = 8
    config.training.gradient_accumulation_steps = 2
    return config
