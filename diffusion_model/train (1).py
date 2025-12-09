"""
Stanford Dogs Diffusion - 训练脚本

使用 DDPM 训练条件扩散模型生成狗图像
所有配置已硬编码，直接运行即可
支持单GPU和多GPU分布式训练（DDP）
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, DataConfig, ModelConfig, DiffusionConfig, TrainingConfig
from dataset import create_dataloaders, denormalize
from models import UNetModel, count_parameters
from diffusion import GaussianDiffusion, EMA, DDIMScheduler, CombinedDiffusionLoss

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 硬编码配置 - 直接修改这里的值即可
# ============================================================

# 数据配置
DATA_ROOT = "/users/5/chen9005/FFF/dogs_diff"
IMAGE_SIZE = 128  # 可选: 64, 128, 256
NUM_CLASSES = 120  # Stanford Dogs 有120个犬种

# 模型配置 (v4优化: 适中模型规模)
MODEL_CHANNELS = 64  # 基础通道数 (参数量约~20M)
CHANNEL_MULT = (1, 2, 2, 4)  # 各层通道倍数 -> 64, 128, 128, 256
NUM_RES_BLOCKS = 2  # 每个分辨率的残差块数量
# 注意: 对于128x128图像, 下采样因子ds从1->2->4->8, 所以应该用(8,4)而不是(16,8)
ATTENTION_RESOLUTIONS = (8, 4)  # 在哪些分辨率使用注意力 (ds=8对应16x16, ds=4对应32x32)
DROPOUT = 0.1
USE_CLASS_CONDITION = False  # 无条件生成模式 - 更简单，更容易训练

# 扩散配置
NUM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
BETA_SCHEDULE = "cosine"  # "linear", "cosine", "quadratic"
LOSS_TYPE = "l2"  # "l1", "l2", "huber"
PREDICTION_TYPE = "epsilon"  # "epsilon", "x_start", "v_prediction"

# 训练配置 (v5优化: 从epoch214继续微调)
NUM_EPOCHS = 350  # 训练轮数 (不需要500个epoch，避免过拟合)
BATCH_SIZE = 32  # 每GPU批次大小 (A100-40GB显存，batch=32约用15GB)
LEARNING_RATE = 5e-5  # 学习率 (降低一半，更精细的微调)
WEIGHT_DECAY = 0.0
OPTIMIZER = "adamw"  # "adam", "adamw"
LR_SCHEDULER = "cosine"  # "cosine", "linear", "constant"
WARMUP_STEPS = 1000
MAX_GRAD_NORM = 1.0
MIXED_PRECISION = "fp16"  # "no", "fp16", "bf16"

# EMA 配置
USE_EMA = True
EMA_DECAY = 0.995  # 降低衰减率，让EMA更快跟上主模型 (原0.9999太慢)
EMA_UPDATE_EVERY = 5  # 每5步更新一次EMA (原10步)
EMA_START_EPOCH = 100  # 从第100个epoch开始使用EMA模型进行采样 (原20太早)

# 视觉损失配置 (提升生成质量)
# v5优化: 启用所有损失来解决低时间步损失过高的问题
USE_VISUAL_LOSS = True   # 是否使用视觉损失 (启用以支持动态损失)
USE_SSIM_LOSS = True     # SSIM结构相似性损失 (直接启用)
USE_PERCEPTUAL_LOSS = True   # VGG感知损失 (启用以学习更好的纹理)
USE_FREQUENCY_LOSS = True    # 频率域损失 (启用以学习高频细节)
USE_EDGE_LOSS = True         # 边缘损失 (直接启用)
SSIM_WEIGHT = 0.2            # SSIM损失权重
PERCEPTUAL_WEIGHT = 0.1      # 感知损失权重 (从0.05提高到0.1，增强细节学习)
FREQUENCY_WEIGHT = 0.05      # 频率损失权重
EDGE_WEIGHT = 0.15           # 边缘损失权重 (从0.1提高到0.15，增强边缘清晰度)
# 动态启用视觉损失的epoch阈值 (已不需要，直接启用)
SSIM_START_EPOCH = 0         # 从头启用SSIM损失
EDGE_START_EPOCH = 0         # 从头启用Edge损失
USE_SNR_WEIGHTING = True     # Min-SNR加权 (启用以解决低时间步损失高的问题)
SNR_GAMMA = 3.0              # SNR加权参数 (从5.0降到3.0，减少对高时间步的过度加权)

# 日志和保存配置
LOG_EVERY = 100  # 每N步记录日志
SAMPLE_EVERY = 2000  # 每N步生成样本
SAVE_EVERY = 5000  # 每N步保存检查点
NUM_SAMPLES = 16  # 每次采样生成的图像数
VAL_SAMPLE_EVERY_EPOCH = 10  # 每N个epoch在验证后生成样本检查质量

# 输出目录 (v5: 方案A优化 - 调整损失权重)
OUTPUT_DIR = "/users/5/chen9005/FFF/dogs_diff/outputs_v5"
CHECKPOINT_DIR = "/users/5/chen9005/FFF/dogs_diff/checkpoints_v5"
LOG_DIR = "/users/5/chen9005/FFF/dogs_diff/logs_v5"

# 高级训练优化
GRADIENT_ACCUMULATION_STEPS = 1  # 梯度累积步数 (增大等效batch size)
LABEL_SMOOTHING = 0.0  # 标签平滑 (0.0-0.1)
GRADIENT_CHECKPOINTING = False  # 梯度检查点 (省显存但慢一点)

# 采样参数 (用于训练时评估)
DEFAULT_GUIDANCE_SCALE = 1.0  # 默认CFG强度 (1.0=关闭CFG，先测试基础效果)
DEFAULT_DDIM_STEPS = 50  # 默认DDIM采样步数

# 其他
SEED = 42
NUM_WORKERS = 4
EXPERIMENT_NAME = "stanford_dogs_ddpm_v5_finetune"  # v5: 方案A优化 - 从epoch214微调
RESUME_CHECKPOINT = "/users/5/chen9005/FFF/dogs_diff/checkpoints_v4/checkpoint_step_110000.pt"  # 从step 110000 (~epoch 214) 继续训练，图像质量最佳时期
# v5变更说明 (方案A):
# 1. LEARNING_RATE = 5e-5 (从1e-4降低一半，更精细的微调)
# 2. SNR_GAMMA = 3.0 (从5.0降低，减少对高时间步的过度加权)
# 3. PERCEPTUAL_WEIGHT = 0.1 (从0.05提高，增强细节学习)
# 4. EDGE_WEIGHT = 0.15 (从0.1提高，增强边缘清晰度)
# 5. NUM_EPOCHS = 350 (减少epoch数避免过拟合)
# 6. 从 checkpoint_step_110000.pt 恢复 (图像质量最佳时期)

# ============================================================


def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        # 单GPU训练
        return 0, 1, 0


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """判断是否为主进程"""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def create_config():
    """根据硬编码值创建配置对象"""
    config = Config()

    # 数据配置
    config.data = DataConfig()
    config.data.data_root = DATA_ROOT
    config.data.image_size = IMAGE_SIZE
    config.data.num_classes = NUM_CLASSES

    # 模型配置
    config.model = ModelConfig()
    config.model.model_channels = MODEL_CHANNELS
    config.model.channel_mult = CHANNEL_MULT
    config.model.num_res_blocks = NUM_RES_BLOCKS
    config.model.attention_resolutions = ATTENTION_RESOLUTIONS
    config.model.dropout = DROPOUT
    config.model.num_classes = NUM_CLASSES
    config.model.use_class_condition = USE_CLASS_CONDITION

    # 扩散配置
    config.diffusion = DiffusionConfig()
    config.diffusion.num_timesteps = NUM_TIMESTEPS
    config.diffusion.beta_start = BETA_START
    config.diffusion.beta_end = BETA_END
    config.diffusion.beta_schedule = BETA_SCHEDULE
    config.diffusion.loss_type = LOSS_TYPE
    config.diffusion.prediction_type = PREDICTION_TYPE

    # 训练配置
    config.training = TrainingConfig()
    config.training.num_epochs = NUM_EPOCHS
    config.training.batch_size = BATCH_SIZE
    config.training.learning_rate = LEARNING_RATE
    config.training.weight_decay = WEIGHT_DECAY
    config.training.optimizer = OPTIMIZER
    config.training.lr_scheduler = LR_SCHEDULER
    config.training.warmup_steps = WARMUP_STEPS
    config.training.max_grad_norm = MAX_GRAD_NORM
    config.training.mixed_precision = MIXED_PRECISION
    config.training.use_ema = USE_EMA
    config.training.ema_decay = EMA_DECAY
    config.training.ema_update_every = EMA_UPDATE_EVERY
    config.training.log_every = LOG_EVERY
    config.training.sample_every = SAMPLE_EVERY
    config.training.save_every = SAVE_EVERY
    config.training.num_samples = NUM_SAMPLES
    config.training.output_dir = OUTPUT_DIR
    config.training.checkpoint_dir = CHECKPOINT_DIR
    config.training.log_dir = LOG_DIR
    config.training.num_workers = NUM_WORKERS

    config.seed = SEED
    config.experiment_name = EXPERIMENT_NAME

    return config


class Trainer:
    """扩散模型训练器 - 支持单GPU和多GPU分布式训练"""

    def __init__(self, config: Config, rank: int = 0, world_size: int = 1, local_rank: int = 0):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_distributed = world_size > 1
        self.is_main = rank == 0

        # 设置设备
        if self.is_distributed:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建目录 (只在主进程创建)
        self.output_dir = Path(config.training.output_dir)
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.log_dir = Path(config.training.log_dir)

        if self.is_main:
            for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

        # 同步所有进程，确保目录已创建
        if self.is_distributed:
            dist.barrier()

        # 设置随机种子
        self._set_seed(config.seed)

        # 创建数据加载器
        self._create_dataloaders()

        # 创建模型
        self._create_model()

        # 创建扩散过程
        self._create_diffusion()

        # 创建优化器和调度器
        self._create_optimizer()

        # 创建 EMA (使用原始模型，非 DDP)
        if config.training.use_ema:
            model_for_ema = self.model.module if self.is_distributed else self.model
            self.ema = EMA(
                model_for_ema,
                decay=config.training.ema_decay,
                update_every=config.training.ema_update_every,
            )
        else:
            self.ema = None

        # 混合精度训练
        self.use_amp = config.training.mixed_precision != "no"
        if self.use_amp:
            self.scaler = GradScaler()
            self.amp_dtype = torch.float16 if config.training.mixed_precision == "fp16" else torch.bfloat16
        else:
            self.scaler = None
            self.amp_dtype = torch.float32

        # 创建视觉损失函数
        self.visual_loss_fn = None
        if USE_VISUAL_LOSS:
            self.visual_loss_fn = CombinedDiffusionLoss(
                base_loss_type=self.config.diffusion.loss_type,
                use_ssim=USE_SSIM_LOSS,
                use_perceptual=USE_PERCEPTUAL_LOSS,
                use_frequency=USE_FREQUENCY_LOSS,
                use_edge=USE_EDGE_LOSS,
                ssim_weight=SSIM_WEIGHT,
                perceptual_weight=PERCEPTUAL_WEIGHT,
                frequency_weight=FREQUENCY_WEIGHT,
                edge_weight=EDGE_WEIGHT,
                apply_on_x0=True,
            ).to(self.device)
            if self.is_main:
                logger.info("Visual loss enabled: SSIM=%s, Perceptual=%s, Frequency=%s, Edge=%s",
                           USE_SSIM_LOSS, USE_PERCEPTUAL_LOSS, USE_FREQUENCY_LOSS, USE_EDGE_LOSS)

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # 日志记录器 (只在主进程设置)
        self.writer = None
        if self.is_main:
            self._setup_logging()

        if self.is_main:
            logger.info(f"Trainer initialized on {self.device}")
            logger.info(f"World size: {self.world_size}")
            logger.info(f"Model parameters: {count_parameters(self.model):,}")

    def _set_seed(self, seed: int):
        """设置随机种子"""
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_dataloaders(self):
        """创建数据加载器 - 支持分布式训练"""
        from dataset import StanfordDogsDataset, get_transforms

        # 创建数据集
        train_transform = get_transforms(self.config.data.image_size, "train")
        val_transform = get_transforms(self.config.data.image_size, "val")

        train_dataset = StanfordDogsDataset(
            root_dir=self.config.data.data_root,
            image_size=self.config.data.image_size,
            transform=train_transform,
            split="train",
            train_ratio=self.config.data.train_ratio,
            val_ratio=self.config.data.val_ratio,
        )
        val_dataset = StanfordDogsDataset(
            root_dir=self.config.data.data_root,
            image_size=self.config.data.image_size,
            transform=val_transform,
            split="val",
            train_ratio=self.config.data.train_ratio,
            val_ratio=self.config.data.val_ratio,
        )
        test_dataset = StanfordDogsDataset(
            root_dir=self.config.data.data_root,
            image_size=self.config.data.image_size,
            transform=val_transform,
            split="test",
            train_ratio=self.config.data.train_ratio,
            val_ratio=self.config.data.val_ratio,
        )

        # collate_fn
        def collate_fn(batch):
            images = torch.stack([item["image"] for item in batch])
            class_idx = torch.stack([item["class_idx"] for item in batch])
            return {"image": images, "class_idx": class_idx}

        # 创建分布式采样器
        if self.is_distributed:
            self.train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            shuffle = False  # 使用sampler时不能shuffle
        else:
            self.train_sampler = None
            shuffle = True

        # 创建DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            sampler=self.train_sampler,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        self.dataset_info = {
            "num_classes": len(train_dataset.class_to_idx),
            "class_to_idx": train_dataset.class_to_idx,
            "idx_to_class": train_dataset.idx_to_class,
            "num_train": len(train_dataset),
            "num_val": len(val_dataset),
            "num_test": len(test_dataset),
        }

        if self.is_main:
            logger.info(f"Dataset loaded: {self.dataset_info['num_train']} train, {self.dataset_info['num_val']} val")

    def _create_model(self):
        """创建 UNet 模型 - 支持 DDP"""
        image_size = self.config.data.image_size
        num_classes = self.config.model.num_classes if self.config.model.use_class_condition else None

        self.model = UNetModel(
            image_size=image_size,
            in_channels=self.config.model.in_channels,
            out_channels=self.config.model.out_channels,
            model_channels=self.config.model.model_channels,
            num_res_blocks=self.config.model.num_res_blocks,
            attention_resolutions=self.config.model.attention_resolutions,
            dropout=self.config.model.dropout,
            channel_mult=self.config.model.channel_mult,
            num_classes=num_classes,
            num_heads=8,
            use_scale_shift_norm=True,
            class_dropout_prob=0.1,
        ).to(self.device)

        # DDP 包装
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )

        if self.is_main:
            # 获取原始模型的参数量
            model_for_count = self.model.module if self.is_distributed else self.model
            logger.info(f"Model created with {count_parameters(model_for_count):,} parameters")

    def _create_diffusion(self):
        """创建扩散过程"""
        # 使用原始模型 (非 DDP 包装)
        model_for_diffusion = self.model.module if self.is_distributed else self.model

        self.diffusion = GaussianDiffusion(
            model=model_for_diffusion,
            num_timesteps=self.config.diffusion.num_timesteps,
            beta_start=self.config.diffusion.beta_start,
            beta_end=self.config.diffusion.beta_end,
            beta_schedule=self.config.diffusion.beta_schedule,
            loss_type=self.config.diffusion.loss_type,
            prediction_type=self.config.diffusion.prediction_type,
            device=self.device,
        )

        # 创建 DDIM 采样器 (用于快速采样)
        # 注意: 必须传入 prediction_type 以确保与训练一致
        self.ddim_scheduler = DDIMScheduler(
            num_timesteps=self.config.diffusion.num_timesteps,
            beta_start=self.config.diffusion.beta_start,
            beta_end=self.config.diffusion.beta_end,
            beta_schedule=self.config.diffusion.beta_schedule,
            prediction_type=self.config.diffusion.prediction_type,  # 确保与训练一致
            device=self.device,
        )

    def _create_optimizer(self):
        """创建优化器和学习率调度器"""
        if self.config.training.optimizer == "adam":
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
                weight_decay=self.config.training.weight_decay,
            )
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
                weight_decay=self.config.training.weight_decay,
            )

        total_steps = len(self.train_loader) * self.config.training.num_epochs
        warmup_steps = self.config.training.warmup_steps

        if self.config.training.lr_scheduler == "cosine":
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps],
            )
        elif self.config.training.lr_scheduler == "linear":
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=total_steps,
            )
        else:
            self.scheduler = None

    def _setup_logging(self):
        """设置日志记录"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir / self.config.experiment_name)
            logger.info(f"TensorBoard logging enabled at {self.log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available, skipping logging setup")
            self.writer = None

    def train_step(self, batch, log_timestep_loss: bool = False) -> dict:
        """单步训练

        参数:
            batch: 数据批次
            log_timestep_loss: 是否记录不同时间步的损失分布
        """
        self.model.train()

        if isinstance(batch, dict):
            images = batch["image"].to(self.device)
            labels = batch.get("class_idx", None)
            if labels is not None:
                labels = labels.to(self.device)
        else:
            images = batch.to(self.device)
            labels = None

        t = torch.randint(
            0, self.config.diffusion.num_timesteps,
            (images.shape[0],), device=self.device
        )

        self.optimizer.zero_grad()

        if self.use_amp:
            with autocast(dtype=self.amp_dtype):
                # 使用视觉损失训练
                if self.visual_loss_fn is not None:
                    loss_dict = self.diffusion.training_losses_with_visual(
                        images, t, labels,
                        visual_loss_fn=self.visual_loss_fn,
                        snr_gamma=SNR_GAMMA,
                        use_snr_weighting=USE_SNR_WEIGHTING,
                    )
                elif USE_SNR_WEIGHTING:
                    loss_dict = self.diffusion.training_losses_with_snr_weighting(
                        images, t, labels, snr_gamma=SNR_GAMMA
                    )
                else:
                    loss_dict = self.diffusion.training_losses(images, t, labels)
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 使用视觉损失训练
            if self.visual_loss_fn is not None:
                loss_dict = self.diffusion.training_losses_with_visual(
                    images, t, labels,
                    visual_loss_fn=self.visual_loss_fn,
                    snr_gamma=SNR_GAMMA,
                    use_snr_weighting=USE_SNR_WEIGHTING,
                )
            elif USE_SNR_WEIGHTING:
                loss_dict = self.diffusion.training_losses_with_snr_weighting(
                    images, t, labels, snr_gamma=SNR_GAMMA
                )
            else:
                loss_dict = self.diffusion.training_losses(images, t, labels)
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm,
            )
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        if self.ema is not None:
            self.ema.update()

        self.global_step += 1

        # 返回更详细的损失信息
        result = {"loss": loss.item()}
        if "loss_dict" in loss_dict:
            result.update(loss_dict["loss_dict"])
        return result

    @torch.no_grad()
    def validate(self) -> dict:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating", leave=False, mininterval=10, ncols=80):
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
                labels = batch.get("class_idx", None)
                if labels is not None:
                    labels = labels.to(self.device)
            else:
                images = batch.to(self.device)
                labels = None

            t = torch.randint(
                0, self.config.diffusion.num_timesteps,
                (images.shape[0],), device=self.device
            )

            loss_dict = self.diffusion.training_losses(images, t, labels)
            total_loss += loss_dict["loss"].item()
            num_batches += 1

        return {"val_loss": total_loss / num_batches}

    @torch.no_grad()
    def analyze_timestep_loss(self, num_batches: int = 10) -> dict:
        """
        分析不同时间步的损失分布

        这有助于诊断模型在哪些时间步学习困难
        """
        self.model.eval()

        # 将时间步分成几个区间
        num_bins = 5
        timestep_bins = [
            (0, 200),      # 低噪声
            (200, 400),    # 中低噪声
            (400, 600),    # 中噪声
            (600, 800),    # 中高噪声
            (800, 1000),   # 高噪声
        ]
        bin_losses = {f"t_{low}-{high}": [] for low, high in timestep_bins}

        batch_count = 0
        for batch in self.val_loader:
            if batch_count >= num_batches:
                break

            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
                labels = batch.get("class_idx", None)
                if labels is not None:
                    labels = labels.to(self.device)
            else:
                images = batch.to(self.device)
                labels = None

            # 对每个时间步区间计算损失
            for low, high in timestep_bins:
                t = torch.randint(low, high, (images.shape[0],), device=self.device)
                noise = torch.randn_like(images)
                x_t = self.diffusion.q_sample(images, t, noise)

                model_output = self.model(x_t, t, labels) if not self.is_distributed else self.model.module(x_t, t, labels)
                loss = F.mse_loss(model_output, noise)
                bin_losses[f"t_{low}-{high}"].append(loss.item())

            batch_count += 1

        # 计算每个区间的平均损失
        result = {}
        for key, losses in bin_losses.items():
            if losses:
                result[key] = sum(losses) / len(losses)

        if self.is_main:
            logger.info("=" * 50)
            logger.info("Timestep Loss Analysis:")
            for key, loss in result.items():
                logger.info(f"  {key}: {loss:.4f}")
            logger.info("=" * 50)

        return result

    @torch.no_grad()
    def sample(self, num_samples: int = 16, use_ddim: bool = True, ddim_steps: int = 50,
               guidance_scale: float = DEFAULT_GUIDANCE_SCALE, class_labels: torch.Tensor = None,
               debug: bool = False, use_main_model: bool = None) -> torch.Tensor:
        """生成样本

        参数:
            num_samples: 生成样本数量
            use_ddim: 是否使用DDIM采样
            ddim_steps: DDIM采样步数
            guidance_scale: CFG引导强度 (推荐3.0-10.0)
            class_labels: 指定类别标签，None则随机选择
            debug: 是否输出调试信息
            use_main_model: 是否使用主模型（而不是EMA），None则自动根据epoch决定
        """
        # 自动决定使用哪个模型：epoch >= EMA_START_EPOCH 时使用EMA
        if use_main_model is None:
            use_main_model = self.epoch < EMA_START_EPOCH

        # 选择模型
        if use_main_model or self.ema is None:
            model = self.model.module if self.is_distributed else self.model
            if debug:
                logger.info(f"[DEBUG] Using MAIN model for sampling (epoch {self.epoch})")
        else:
            model = self.ema.get_ema_model()
            if debug:
                logger.info(f"[DEBUG] Using EMA model for sampling (epoch {self.epoch}, EMA step {self.ema.step})")

        model.eval()
        # 转换为FP32以避免数值问题
        model = model.float()
        if debug:
            logger.info(f"[DEBUG] Model converted to FP32 for stable sampling")

        # 生成类别标签（条件生成的关键！）
        if self.config.model.use_class_condition:
            if class_labels is not None:
                y = class_labels.to(self.device)
            else:
                y = torch.randint(0, self.config.model.num_classes, (num_samples,), device=self.device)
            logger.info(f"Sampling with class labels: {y.tolist()[:8]}...")
        else:
            y = None
            guidance_scale = 1.0  # 无条件生成时不使用CFG

        if use_ddim:
            # 使用自定义采样循环以便调试 - FP32精度
            self.ddim_scheduler.set_timesteps(ddim_steps, self.device)
            shape = (num_samples, 3, self.config.data.image_size, self.config.data.image_size)
            samples = torch.randn(shape, device=self.device, dtype=torch.float32)  # FP32

            if debug:
                logger.info(f"[DEBUG] Initial noise (FP32): min={samples.min():.4f}, max={samples.max():.4f}")

            for i, t in enumerate(self.ddim_scheduler.timesteps):
                t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)

                # CFG
                if guidance_scale > 1.0 and y is not None:
                    noise_pred_cond = model(samples, t_batch, y)
                    noise_pred_uncond = model(samples, t_batch, None)
                    model_output = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    model_output = model(samples, t_batch, y)

                # 检查NaN/Inf
                if torch.isnan(model_output).any() or torch.isinf(model_output).any():
                    if debug:
                        logger.warning(f"[DEBUG] Step {i}: model_output has NaN/Inf, fixing...")
                    model_output = torch.nan_to_num(model_output, nan=0.0, posinf=0.0, neginf=0.0)

                samples, _ = self.ddim_scheduler.step(
                    model_output=model_output,
                    timestep=t.item(),
                    sample=samples,
                    eta=0.0,
                )

                # 检查采样结果
                if torch.isnan(samples).any() or torch.isinf(samples).any():
                    if debug:
                        logger.warning(f"[DEBUG] Step {i}: samples has NaN/Inf, fixing...")
                    samples = torch.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)

                if debug and i % 10 == 0:
                    logger.info(f"[DEBUG] Step {i}: samples min={samples.min():.4f}, max={samples.max():.4f}")

            if debug:
                logger.info(f"[DEBUG] Final samples: min={samples.min():.4f}, max={samples.max():.4f}")
        else:
            if guidance_scale > 1.0 and y is not None:
                # 使用CFG采样
                samples = self.diffusion.sample_with_cfg(
                    batch_size=num_samples,
                    image_size=self.config.data.image_size,
                    y=y,
                    guidance_scale=guidance_scale,
                    show_progress=False,
                )
            else:
                samples = self.diffusion.sample(
                    batch_size=num_samples,
                    image_size=self.config.data.image_size,
                    y=y,
                    show_progress=False,
                )

        return samples

    def save_samples(self, samples: torch.Tensor, filename: str):
        """保存生成的样本"""
        import torchvision.utils as vutils
        from PIL import Image
        import numpy as np

        # 调试：检查样本统计信息
        if self.is_main:
            has_nan = torch.isnan(samples).any().item()
            has_inf = torch.isinf(samples).any().item()
            logger.info(f"[DEBUG] samples stats before denorm: min={samples.min().item():.4f}, "
                       f"max={samples.max().item():.4f}, mean={samples.mean().item():.4f}, "
                       f"has_nan={has_nan}, has_inf={has_inf}")

        # 处理NaN和Inf
        if torch.isnan(samples).any() or torch.isinf(samples).any():
            logger.warning(f"[WARNING] Found NaN/Inf in samples, replacing with zeros")
            samples = torch.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)

        samples = denormalize(samples)
        samples = samples.clamp(0, 1)

        if self.is_main:
            logger.info(f"[DEBUG] samples stats after denorm+clamp: min={samples.min().item():.4f}, "
                       f"max={samples.max().item():.4f}")

        grid = vutils.make_grid(samples, nrow=int(samples.shape[0] ** 0.5), padding=2)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        grid_np = np.nan_to_num(grid_np, nan=0.0, posinf=1.0, neginf=0.0)  # 安全处理
        grid_np = (grid_np * 255).astype(np.uint8)
        Image.fromarray(grid_np).save(self.output_dir / filename)

    @torch.no_grad()
    def sample_with_process_visualization(self, epoch: int, num_samples: int = 4, guidance_scale: float = DEFAULT_GUIDANCE_SCALE):
        """
        生成样本并可视化扩散过程
        展示从噪声到图像的去噪过程，包含完整的调试信息
        同时对比主模型和EMA模型的效果

        参数:
            epoch: 当前epoch
            num_samples: 生成样本数量
            guidance_scale: CFG引导强度
        """
        import torchvision.utils as vutils
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        logger.info(f"=" * 60)
        logger.info(f"[DIAGNOSTIC] Diffusion Process Analysis - Epoch {epoch}")
        logger.info(f"CFG scale: {guidance_scale}, num_samples: {num_samples}")
        logger.info(f"=" * 60)

        # 根据epoch自动选择模型：epoch >= EMA_START_EPOCH 时使用EMA
        use_ema = epoch >= EMA_START_EPOCH and self.ema is not None

        if use_ema:
            model = self.ema.get_ema_model()
            logger.info(f"[DEBUG] Using EMA model (epoch {epoch} >= {EMA_START_EPOCH}, EMA step {self.ema.step})")
        else:
            model = self.model.module if self.is_distributed else self.model
            logger.info(f"[DEBUG] Using MAIN model (epoch {epoch} < {EMA_START_EPOCH})")
            if self.ema is not None:
                logger.info(f"[DEBUG] EMA step: {self.ema.step}, decay: {self.ema.decay}")

        model.eval()

        # 重要：在FP32精度下进行采样，避免FP16的数值问题
        model = model.float()
        logger.info(f"[DEBUG] Model converted to FP32 for stable sampling")

        # 设置采样步数和要保存的中间步骤
        num_inference_steps = 50
        # 保存更多步骤的中间结果以便调试
        save_steps = [0, 5, 10, 20, 30, 40, 49]

        # 设置时间步
        self.ddim_scheduler.set_timesteps(num_inference_steps, self.device)
        logger.info(f"[DEBUG] Timesteps set: {self.ddim_scheduler.timesteps[:5].tolist()}...{self.ddim_scheduler.timesteps[-5:].tolist()}")

        # 随机选择类别标签
        if self.config.model.use_class_condition:
            y = torch.randint(0, self.config.model.num_classes, (num_samples,), device=self.device)
            use_cfg = guidance_scale > 1.0
            logger.info(f"[DEBUG] Class labels: {y.tolist()}, use_cfg: {use_cfg}")
        else:
            y = None
            use_cfg = False

        # 从纯噪声开始 - 使用FP32
        shape = (num_samples, 3, self.config.data.image_size, self.config.data.image_size)
        sample = torch.randn(shape, device=self.device, dtype=torch.float32)
        logger.info(f"[DEBUG] Initial noise (FP32): min={sample.min().item():.4f}, max={sample.max().item():.4f}, mean={sample.mean().item():.4f}")

        # 存储中间结果和统计信息
        intermediates = [sample.clone()]
        debug_stats = []

        # DDIM 采样过程 (带CFG)
        for i, t in enumerate(self.ddim_scheduler.timesteps):
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)

            # 使用 Classifier-Free Guidance
            if use_cfg and y is not None:
                noise_pred_cond = model(sample, t_batch, y)
                noise_pred_uncond = model(sample, t_batch, None)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = model(sample, t_batch, y)

            # 检查模型输出是否有问题
            has_nan = torch.isnan(noise_pred).any().item()
            has_inf = torch.isinf(noise_pred).any().item()

            if has_nan or has_inf:
                logger.warning(f"[WARNING] Step {i}, t={t.item()}: noise_pred has NaN={has_nan}, Inf={has_inf}")
                noise_pred = torch.nan_to_num(noise_pred, nan=0.0, posinf=0.0, neginf=0.0)

            # DDIM 步进
            sample, pred_x0 = self.ddim_scheduler.step(
                model_output=noise_pred,
                timestep=t.item(),
                sample=sample,
                eta=0.0,
            )

            # 检查采样结果
            sample_has_nan = torch.isnan(sample).any().item()
            sample_has_inf = torch.isinf(sample).any().item()

            if sample_has_nan or sample_has_inf:
                logger.warning(f"[WARNING] Step {i}, t={t.item()}: sample has NaN={sample_has_nan}, Inf={sample_has_inf}")
                sample = torch.nan_to_num(sample, nan=0.0, posinf=1.0, neginf=-1.0)

            # 保存指定步骤的中间结果
            if i in save_steps:
                intermediates.append(sample.clone())
                stats = {
                    'step': i,
                    't': t.item(),
                    'sample_min': sample.min().item(),
                    'sample_max': sample.max().item(),
                    'sample_mean': sample.mean().item(),
                    'noise_pred_min': noise_pred.min().item(),
                    'noise_pred_max': noise_pred.max().item(),
                    'pred_x0_min': pred_x0.min().item() if pred_x0 is not None else 0,
                    'pred_x0_max': pred_x0.max().item() if pred_x0 is not None else 0,
                }
                debug_stats.append(stats)
                logger.info(f"[DEBUG] Step {i:2d} (t={t.item():4d}): sample [{stats['sample_min']:.3f}, {stats['sample_max']:.3f}], "
                           f"noise_pred [{stats['noise_pred_min']:.3f}, {stats['noise_pred_max']:.3f}]")

        # 输出完整的调试统计
        logger.info(f"\n[DEBUG] Complete diffusion chain statistics:")
        logger.info(f"{'Step':>6} {'t':>6} {'sample_min':>12} {'sample_max':>12} {'sample_mean':>12}")
        logger.info("-" * 60)
        for s in debug_stats:
            logger.info(f"{s['step']:>6} {s['t']:>6} {s['sample_min']:>12.4f} {s['sample_max']:>12.4f} {s['sample_mean']:>12.4f}")

        # 创建可视化图像
        num_steps_to_show = len(intermediates)
        logger.info(f"[DEBUG] Total intermediate steps to visualize: {num_steps_to_show}")

        # 反归一化所有中间结果
        processed_intermediates = []
        for idx, inter in enumerate(intermediates):
            # 检查并处理NaN/Inf
            if torch.isnan(inter).any() or torch.isinf(inter).any():
                logger.warning(f"[WARNING] Intermediate {idx} has NaN/Inf, replacing...")
                inter = torch.nan_to_num(inter, nan=0.0, posinf=1.0, neginf=-1.0)

            inter = denormalize(inter)
            inter = inter.clamp(0, 1)
            processed_intermediates.append(inter)

            if idx == len(intermediates) - 1:
                logger.info(f"[DEBUG] Final output after denorm: min={inter.min().item():.4f}, max={inter.max().item():.4f}")

        # 步骤标签
        step_labels = ["Noise"] + [f"Step {s}" for s in save_steps[1:]] + ["Final"]
        if len(step_labels) < num_steps_to_show:
            step_labels = step_labels[:num_steps_to_show]

        # 创建大图: 每个样本一行，每列是一个去噪步骤
        fig_width = num_steps_to_show * self.config.data.image_size + (num_steps_to_show + 1) * 10
        fig_height = num_samples * self.config.data.image_size + (num_samples + 1) * 50

        # 创建白色背景
        result_image = Image.new('RGB', (fig_width, fig_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(result_image)

        for row in range(num_samples):
            for col, inter in enumerate(processed_intermediates):
                # 获取单张图像
                img_tensor = inter[row]
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                img_np = np.nan_to_num(img_np, nan=0.5, posinf=1.0, neginf=0.0)
                img_np = np.clip(img_np, 0, 1)
                img_np = (img_np * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)

                # 计算位置
                x = 10 + col * (self.config.data.image_size + 10)
                y_pos = 50 + row * (self.config.data.image_size + 50)

                # 粘贴图像
                result_image.paste(img_pil, (x, y_pos))

                # 添加步骤标签 (只在第一行)
                if row == 0 and col < len(step_labels):
                    draw.text((x + 5, 5), step_labels[col], fill=(0, 0, 0))
                    # 添加统计信息
                    if col > 0 and col - 1 < len(debug_stats):
                        stats = debug_stats[col - 1]
                        draw.text((x + 5, 20), f"t={stats['t']}", fill=(100, 100, 100))

            # 添加样本编号
            if y is not None:
                sample_label = f"Sample {row+1} (Class {y[row].item()})"
            else:
                sample_label = f"Sample {row+1}"
            draw.text((5, 50 + row * (self.config.data.image_size + 50) + self.config.data.image_size // 2),
                      sample_label, fill=(100, 100, 100))

        # 添加epoch信息
        draw.text((fig_width // 2 - 50, fig_height - 20), f"Epoch {epoch}", fill=(0, 0, 0))

        # 保存可视化图像
        vis_filename = f"diffusion_process_epoch_{epoch}.png"
        result_image.save(self.output_dir / vis_filename)
        logger.info(f"Diffusion process visualization saved: {vis_filename}")

        # 同时保存最终生成的样本网格
        final_samples = processed_intermediates[-1]
        grid = vutils.make_grid(final_samples, nrow=2, padding=2)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        grid_np = np.nan_to_num(grid_np, nan=0.5, posinf=1.0, neginf=0.0)
        grid_np = np.clip(grid_np, 0, 1)
        grid_np = (grid_np * 255).astype(np.uint8)
        Image.fromarray(grid_np).save(self.output_dir / f"samples_epoch_{epoch}.png")
        logger.info(f"Final samples saved: samples_epoch_{epoch}.png")
        logger.info(f"=" * 60)

    def save_checkpoint(self, filename: str = None):
        """保存检查点 - 只在主进程保存"""
        if not self.is_main:
            return

        if filename is None:
            filename = f"checkpoint_step_{self.global_step}.pt"

        # 使用原始模型的 state_dict (非 DDP)
        model_to_save = self.model.module if self.is_distributed else self.model

        checkpoint = {
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": self.config,
        }

        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()
        if self.scaler is not None:
            checkpoint["scaler"] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)
        logger.info(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点 - 支持 DDP"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 加载到原始模型 (非 DDP)
        model_to_load = self.model.module if self.is_distributed else self.model
        model_to_load.load_state_dict(checkpoint["model"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        if self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.ema is not None and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
        if self.scaler is not None and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        if self.is_main:
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            logger.info(f"Resuming from step {self.global_step}, epoch {self.epoch}")

    def train(self):
        """完整训练循环 - 支持分布式训练"""
        model_for_count = self.model.module if self.is_distributed else self.model

        if self.is_main:
            logger.info("=" * 60)
            logger.info("Stanford Dogs Diffusion Model Training")
            logger.info("=" * 60)
            logger.info(f"Device: {self.device}")
            logger.info(f"World size: {self.world_size}")
            logger.info(f"Image size: {self.config.data.image_size}")
            logger.info(f"Batch size per GPU: {self.config.training.batch_size}")
            logger.info(f"Effective batch size: {self.config.training.batch_size * self.world_size}")
            logger.info(f"Total epochs: {self.config.training.num_epochs}")
            logger.info(f"Learning rate: {self.config.training.learning_rate}")
            logger.info(f"Model parameters: {count_parameters(model_for_count):,}")
            logger.info("=" * 60)

        for epoch in range(self.epoch, self.config.training.num_epochs):
            self.epoch = epoch

            # 动态启用视觉损失
            if self.visual_loss_fn is not None:
                # 60 epoch后启用SSIM损失
                if epoch >= SSIM_START_EPOCH and not self.visual_loss_fn.use_ssim:
                    self.visual_loss_fn.use_ssim = True
                    if self.is_main:
                        logger.info(f"Epoch {epoch}: 启用SSIM损失 (weight={SSIM_WEIGHT})")
                # 120 epoch后启用Edge损失
                if epoch >= EDGE_START_EPOCH and not self.visual_loss_fn.use_edge:
                    self.visual_loss_fn.use_edge = True
                    if self.is_main:
                        logger.info(f"Epoch {epoch}: 启用Edge损失 (weight={EDGE_WEIGHT})")

            # 设置分布式采样器的 epoch (确保每个 epoch 数据打乱方式不同)
            if self.is_distributed and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            epoch_loss = 0.0
            num_batches = 0

            # 只在主进程显示进度条 (减少输出频率)
            if self.is_main:
                pbar = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch + 1}/{self.config.training.num_epochs}",
                    mininterval=30,  # 最少30秒更新一次，减少输出
                    ncols=100,       # 固定宽度
                    leave=True,
                )
            else:
                pbar = self.train_loader

            for batch in pbar:
                loss_dict = self.train_step(batch)
                epoch_loss += loss_dict["loss"]
                num_batches += 1

                # 每100步更新一次进度条信息，减少输出
                if self.is_main and num_batches % 100 == 0:
                    if hasattr(pbar, 'set_postfix'):
                        pbar.set_postfix({
                            "loss": f"{loss_dict['loss']:.4f}",
                            "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        })

                if self.global_step % self.config.training.log_every == 0:
                    if self.is_main and self.writer is not None:
                        self.writer.add_scalar("train/loss", loss_dict["loss"], self.global_step)
                        self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)

                if self.global_step % self.config.training.sample_every == 0 and self.global_step > 0:
                    if self.is_main:
                        logger.info(f"Generating samples at step {self.global_step}...")
                        samples = self.sample(
                            num_samples=self.config.training.num_samples,
                            guidance_scale=DEFAULT_GUIDANCE_SCALE,
                            debug=True,  # 启用调试输出
                        )
                        self.save_samples(samples, f"samples_step_{self.global_step}.png")

                if self.global_step % self.config.training.save_every == 0 and self.global_step > 0:
                    self.save_checkpoint()

            avg_epoch_loss = epoch_loss / num_batches

            if self.is_main:
                logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

            val_dict = self.validate()

            if self.is_main:
                logger.info(f"Validation loss: {val_dict['val_loss']:.4f}")

            # 每 VAL_SAMPLE_EVERY_EPOCH 个 epoch 生成扩散过程可视化 (只在主进程)
            if (epoch + 1) % VAL_SAMPLE_EVERY_EPOCH == 0 and self.is_main:
                self.sample_with_process_visualization(epoch + 1)
                # 同时分析不同时间步的损失分布
                timestep_losses = self.analyze_timestep_loss(num_batches=5)
                if self.writer is not None:
                    for key, loss in timestep_losses.items():
                        self.writer.add_scalar(f"timestep_loss/{key}", loss, epoch)

            if self.is_main and self.writer is not None:
                self.writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
                self.writer.add_scalar("val/loss", val_dict["val_loss"], epoch)

            if val_dict["val_loss"] < self.best_loss:
                self.best_loss = val_dict["val_loss"]
                self.save_checkpoint("best_model.pt")
                if self.is_main:
                    logger.info("Best model saved!")

            # 在epoch 99保存checkpoint，方便从此处开始训练（EMA_START_EPOCH=100之前）
            if epoch == 99:
                self.save_checkpoint("checkpoint_epoch_99.pt")
                if self.is_main:
                    logger.info("Checkpoint saved at epoch 99 (before EMA sampling starts)")

            # 同步所有进程
            if self.is_distributed:
                dist.barrier()

        if self.is_main:
            logger.info("=" * 60)
            logger.info("Training completed!")
            logger.info("=" * 60)

        self.save_checkpoint("final_model.pt")

        if self.is_main and self.writer is not None:
            self.writer.close()


def main():
    """主函数 - 支持分布式训练"""
    # 初始化分布式训练
    rank, world_size, local_rank = setup_distributed()

    # 创建配置
    config = create_config()

    # 只在主进程打印配置信息
    if rank == 0:
        print("=" * 60)
        print("Training Configuration (v3)")
        print("=" * 60)
        print(f"Data root: {DATA_ROOT}")
        print(f"Image size: {IMAGE_SIZE}")
        print(f"Batch size per GPU: {BATCH_SIZE}")
        print(f"Effective batch size: {BATCH_SIZE * world_size}")
        print(f"Epochs: {NUM_EPOCHS}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Model channels: {MODEL_CHANNELS}")
        print(f"Use class condition: {USE_CLASS_CONDITION} (无条件生成)")
        print(f"EMA start epoch: {EMA_START_EPOCH}")
        print(f"Mixed precision: {MIXED_PRECISION}")
        print(f"Output dir: {OUTPUT_DIR}")
        print(f"World size (GPUs): {world_size}")
        print("=" * 60)

    # 创建训练器
    trainer = Trainer(config, rank=rank, world_size=world_size, local_rank=local_rank)

    # 恢复训练 (如果设置了检查点路径)
    if RESUME_CHECKPOINT is not None:
        trainer.load_checkpoint(RESUME_CHECKPOINT)

    # 开始训练
    try:
        trainer.train()
    finally:
        # 清理分布式环境
        cleanup_distributed()


if __name__ == "__main__":
    main()
