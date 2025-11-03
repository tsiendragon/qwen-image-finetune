"""
Abstract Base Trainer for all trainer implementations.
Defines the core interface that all trainers must implement.
"""

import gc
import glob
import importlib
import json
import logging
import os
import shutil
import signal
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import PIL
import safetensors.torch
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import FluxKontextPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from peft.utils import get_peft_model_state_dict
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,  # BackwardPrefetch
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy  # transformer_auto_wrap_policy
from tqdm import tqdm

from qflux.data.cache_manager import EmbeddingCacheManager
from qflux.data.config import Config
from qflux.models.quantize import quantize_model_to_fp8
from qflux.scheduler.custom_flowmatch_scheduler import FlowMatchEulerDiscreteScheduler
from qflux.trainer.constants import LORA_FILE_BASE_NAME
from qflux.trainer.validation import ValidationMixin
from qflux.utils.huggingface import download_lora
from qflux.utils.logger import LoggerManager
from qflux.utils.lora_utils import FpsLogger, classify_lora_weight, get_lora_layers, get_lora_state_dict_oom_safe
from qflux.utils.model_summary import print_model_summary_table
from qflux.utils.sampling import calculate_shift, retrieve_timesteps
from qflux.utils.tools import calculate_sha256_file, get_git_info, instantiate_class


logger = logging.getLogger(__name__)


class BaseTrainer(ValidationMixin, ABC):
    """
    Abstract base class for all trainer implementations.
    Defines the core interface that all trainers must implement.
    """

    def __init__(self, config: Config):
        """Initialize trainer with configuration."""
        self.config = config
        self.accelerator: Accelerator
        self.optimizer: torch.optim.Optimizer
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler
        self.global_step = 0
        self.scheduler: FlowMatchEulerDiscreteScheduler  # For training
        self.sampling_scheduler: FlowMatchEulerDiscreteScheduler  # For validation/sampling

        # Common attributes that all trainers should have
        self.weight_dtype = torch.bfloat16
        self.batch_size = self.config.data.batch_size
        self.use_cache = self.config.cache.use_cache
        self.cache_dir = self.config.cache.cache_dir
        self.fps_logger = FpsLogger()
        self.cache_manager = EmbeddingCacheManager(self.cache_dir)
        self.cache_exist = self.cache_manager.exist(self.cache_dir)
        self.quantize = self.config.model.quantize
        self.adapter_name = self.config.model.lora.adapter_name
        self.predict_setted = False

        # 为save_last_checkpoint功能添加属性
        self.training_interrupted = False
        self.log_model_info()
        self.load_preprocessor()
        self.pipeline_class = self.get_pipeline_class()

    @abstractmethod
    def get_pipeline_class(self):
        """return the pipeline class to use the classmethod"""
        return FluxKontextPipeline

    def load_preprocessor(self):
        class_path = self.config.data.init_args.processor.class_path
        init_args = self.config.data.init_args.processor.init_args
        self.preprocessor = instantiate_class(class_path, init_args)

    def __repr__(self) -> str:
        msg = f"{self.__class__.__name__}(config={self.config})"
        return msg

    def setup_signal_handlers(self):
        """设置信号处理器来捕获Ctrl+C中断"""

        def signal_handler(signum, frame):
            logging.info("收到中断信号，准备保存最后一个检查点...")
            self.training_interrupted = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def log_model_info(self):
        """Log model information."""
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Use Cache: {self.use_cache}")

    def setup_versioned_logging_dir(self):
        """设置版本化的日志目录"""
        base_output_dir = self.config.logging.output_dir
        project_name = self.config.logging.tracker_project_name

        # 创建项目目录结构: output_dir/project_name/v0
        project_dir = os.path.join(base_output_dir, project_name)

        # 如果项目目录不存在，直接使用 v0
        if not os.path.exists(project_dir):
            versioned_dir = os.path.join(project_dir, "v0")
            self.config.logging.output_dir = versioned_dir
            os.makedirs(versioned_dir, exist_ok=True)
            logging.info(f"创建新的训练版本目录: {versioned_dir}")
            self.versioned_dir = versioned_dir
            self.experiment_name = "v0"
            return

        # 查找现有版本
        existing_versions = []
        for item in os.listdir(project_dir):
            item_path = os.path.join(project_dir, item)
            if os.path.isdir(item_path) and item.startswith("v") and item[1:].isdigit():
                version_num = int(item[1:])
                existing_versions.append((version_num, item_path))

        # 清理无效版本（训练步数 < 5）
        valid_versions = []
        for version_num, version_path in existing_versions:
            if self._is_valid_training_version(version_path):
                valid_versions.append(version_num)
            else:
                logging.info(f"移除无效训练版本: {version_path}")
                try:
                    # Remove contents of the directory instead of deleting the directory itself
                    for item in os.listdir(version_path):
                        item_path = os.path.join(version_path, item)
                        print("remove item_path", item_path)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                except Exception as e:
                    logging.info(f"移除无效训练版本失败: {version_path}, {e}")

        # 确定新版本号
        if valid_versions:
            next_version = max(valid_versions) + 1
        else:
            next_version = 0

        # 创建新版本目录
        versioned_dir = os.path.join(project_dir, f"v{next_version}")
        self.versioned_dir = versioned_dir
        self.experiment_name = f"v{next_version}"

        self.config.logging.output_dir = versioned_dir
        logging.info(f"使用训练版本目录: {versioned_dir}")

    def _is_valid_training_version(self, version_path):
        """if the folder consist checkpoint, return True"""
        # 检查 checkpoint 目录

        checkpoints = glob.glob(f"{version_path}/*/*.safetensors")
        return len(checkpoints) > 0

    @classmethod
    def convert_img_shapes_to_latent(
        cls, img_shapes_original: list, vae_scale_factor: int = 8, packing_factor: int = 2
    ) -> list:
        """Convert original image shapes to latent space shapes for single sample

        This method transforms image shapes from pixel space to latent space,
        accounting for VAE downsampling and packing operations.\n
        `img_shapes single example`:
        ```python
        [(3, 512, 512), (3, 512, 512), (3, 640, 640)]
        ```
        `img_shapes batch example`:
        ```python
        [
            [(3, 512, 512), (3, 512, 512), (3, 640, 640)],
            [(3, 128, 512), (3, 128, 512), (3, 128, 640)]
        ]
        ```
        Args:
            img_shapes_original: List of tuples [(C, H, W), ...] in original pixel space
                                where C is usually 3 (RGB) or 1 (grayscale)
            vae_scale_factor: VAE downsampling factor (default: 8)
            packing_factor: Additional packing factor (default: 2)

        Returns:
            List of tuples [(1, H_latent, W_latent), ...] in latent space where:
                - C is always 1 in latent space
                - H_latent = H_original // vae_scale_factor // packing_factor
                - W_latent = W_original // vae_scale_factor // packing_factor

        Example:
            >>> original = [(3, 512, 512), (3, 640, 640)]
            >>> latent = cls.convert_img_shapes_to_latent(original, vae_scale_factor=8, packing_factor=2)
            >>> latent  # [(1, 32, 32), (1, 40, 40)]

        Note:
            - For Flux/Qwen models: vae_scale_factor=8, packing_factor=2
            - Total downsampling: original_dim // 16 = original_dim // (8 * 2)
        """
        if not img_shapes_original:
            return []
        total_scale = vae_scale_factor * packing_factor
        img_shapes_latent = []
        for shape in img_shapes_original:
            if len(shape) != 3:
                raise ValueError(f"Expected shape tuple (C, H, W), got {shape}")
            c_orig, h_orig, w_orig = shape
            # Convert to latent space dimensions
            # Channel becomes 1 in latent space
            # Height and width are downsampled by total_scale
            h_latent = h_orig // total_scale
            w_latent = w_orig // total_scale
            if h_latent == 0 or w_latent == 0:
                logging.warning(
                    f"Latent dimension became 0: original shape {shape}, "
                    f"total_scale {total_scale}, latent shape ({h_latent}, {w_latent}). "
                    f"Original image may be too small."
                )
            img_shapes_latent.append((1, h_latent, w_latent))
        return img_shapes_latent

    @classmethod
    def validate_img_shapes(cls, img_shapes: list) -> bool:
        """Validate img_shapes format"""
        assert isinstance(img_shapes, list), "img_shapes must be a list"
        if not img_shapes:
            return False
        assert isinstance(img_shapes[0], list), "img_shapes must be a list of lists"
        assert isinstance(img_shapes[0][0], tuple), "img_shapes must be a list of lists of tuples"
        assert len(img_shapes[0][0]) == 3, "img_shapes must be a list of lists of tuples with 3 elements"
        assert img_shapes[0][0][0] in [1, 3], "img_shapes must be a list of lists of tuples with 3 elements"
        assert img_shapes[0][0][1] > 0, "img_shapes must be a list of lists of tuples with 3 elements"
        assert img_shapes[0][0][2] > 0, "img_shapes must be a list of lists of tuples with 3 elements"
        return True

    @classmethod
    def should_use_multi_resolution_mode(cls, batch: dict) -> bool:
        """Determine if multi-resolution mode should be used for this batch

        This is a generic method that can be used by all trainers supporting
        multi-resolution training. It checks both the configuration and batch
        structure to determine if padding and masking should be applied.

        Returns False when:
        1. multi_resolutions not configured in processor config (MOST IMPORTANT)
        2. batch_size == 1 (single sample uses original logic)
        3. All samples have identical dimensions (no padding needed)

        Returns True when:
        1. multi_resolutions IS configured in processor config AND
        2. batch_size > 1 AND
        3. Batch contains samples with different dimensions

        Args:
            batch: Training batch dictionary containing embeddings and metadata

        Returns:
            bool: True if multi-resolution mode should be used, False otherwise
        """
        cls.validate_img_shapes(batch["img_shapes"])
        batch_size = len(batch["img_shapes"])
        if batch_size == 1:
            logging.debug("Single sample, using shared mode")
            return False

        shapes = batch["img_shapes"]  # [[(C,H,W), ...], [(C,H,W), ...], ...]
        if isinstance(shapes, torch.Tensor):
            shapes = shapes.tolist()
        # Extract spatial dimensions (H, W) for all images in each sample
        # Compare the complete resolution profile of each sample
        sample_resolution_profiles = []
        for sample_shapes in shapes:
            if isinstance(sample_shapes, (list, tuple)):
                # Extract (H, W) from each image in this sample, ignoring channel (C)
                resolution_profile = []
                for img_shape in sample_shapes:
                    if isinstance(img_shape, (list, tuple)) and len(img_shape) >= 3:
                        # Extract (H, W), ignoring C
                        resolution_profile.append((img_shape[1], img_shape[2]))
                    else:
                        # Fallback: use the whole shape
                        resolution_profile.append(tuple(img_shape))
                sample_resolution_profiles.append(tuple(resolution_profile))
            else:
                # Fallback for unexpected format
                sample_resolution_profiles.append(tuple(sample_shapes))

        # Check if all samples have identical resolution profiles
        # Using set to find unique resolution profiles
        unique_profiles = len(set(sample_resolution_profiles))
        if unique_profiles == 1:
            return False
        return True

    def accelerator_prepare(self, train_dataloader):
        """Prepare accelerator"""
        # from diffusers.loaders import AttnProcsLayers
        # from qflux.utils.lora_utils import get_lora_layers
        # lora_layers_model = AttnProcsLayers(get_lora_layers(self.dit))
        # 根据配置决定是否启用梯度检查点
        if self.config.train.gradient_checkpointing:
            self.dit.enable_gradient_checkpointing()
        if self.config.resume is not None:
            # self.accelerator.load_state(self.config.train.resume_from_checkpoint)
            self.optimizer.load_state_dict(torch.load(os.path.join(self.config.resume, "optimizer.bin")))
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(self.config.resume, "scheduler.bin")))
            logging.info(f"Loaded optimizer and scheduler from {self.config.resume}")

        # sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)
        if self.is_fsdp_enabled():
            plug = self.accelerator.state.fsdp_plugin
            # torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)

            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
            lora_mods = list(get_lora_layers(self.dit).values())
            assert all(isinstance(m, torch.nn.Module) for m in lora_mods), "get_lora_layers 必须返回 Module 实例"
            plug.ignored_modules = lora_mods
            plug.use_orig_params = True
            plug.limit_all_gathers = True
            # plug.forward_prefetch = False
            # plug.backward_prefetch = BackwardPrefetch.BACKWARD_POST
            plug.forward_prefetch = True
            plug.backward_prefetch = BackwardPrefetch.BACKWARD_PRE  # 边回传边预取
            plug.sync_module_states = True  # 主卡广播初始化，避免不一致
            plug.min_num_params = 20_000_000  # 5_000_000
            plug.mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
                cast_forward_inputs=False,
            )
            # from qflux.models.transformer_qwenimage import QwenImageTransformerBlock  # 你的类路径
            # plug.auto_wrap_policy = partial(
            #     transformer_auto_wrap_policy,
            #     transformer_layer_cls={QwenImageTransformerBlock},  # 注意：是关键字参数
            # )
            plug.auto_wrap_policy = partial[bool](size_based_auto_wrap_policy, min_num_params=plug.min_num_params)
            # from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            plug.sharding_strategy = ShardingStrategy.FULL_SHARD  # FULL_SHARD   # SHARD_GRAD_OP

            self.dit = self.dit.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

            self.dit, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
                self.dit, self.optimizer, train_dataloader, self.lr_scheduler
            )
            u = self.accelerator.unwrap_model(self.dit)

            self.lora_params2device(u, self.accelerator.device)

            bad = [
                (n, str(p.device))
                for n, p in u.named_parameters()
                if "lora_" in n and p.device != self.accelerator.device
            ]
            assert not bad, f"LoRA still on wrong device: {bad}"
        else:
            lora_layers_model = AttnProcsLayers(get_lora_layers(self.dit))
            lora_layers_model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
                lora_layers_model, self.optimizer, train_dataloader, self.lr_scheduler
            )
            self.dit = self.dit.to(self.accelerator.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Trackers already initialized in setup_accelerator
        return train_dataloader

    def lora_params2device(self, u, dev):
        # 1) 模块级搬运：凡是 LoRA 相关子模块，统统 to(device)
        for m in u.modules():
            for attr in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", self.adapter_name):
                if hasattr(m, attr):
                    getattr(m, attr).to(dev)

        # 2) 参数/缓冲兜底（包含 lora_recover.*）
        for n, p in u.named_parameters():
            if ("lora_" in n or self.adapter_name in n) and p.device != dev:
                p.data = p.data.to(dev)
                if p.grad is not None:
                    p.grad.data = p.grad.data.to(dev)

        for n, b in u.named_buffers():
            if ("lora_" in n or self.adapter_name in n) and getattr(b, "device", dev) != dev:
                b.data = b.data.to(dev)

    def merge_lora(self):
        """Merge LoRA weights into base model"""
        self.dit.merge_adapter()
        logging.info("Merged LoRA weights into base model")

    def cache(self, train_dataloader):
        """Pre-compute and cache embeddings/latents for training efficiency."""
        logging.info("Starting embedding caching process...")
        self.load_model()
        self.setup_model_device_train_mode(stage="cache", cache=True)
        # Cache for each item (same loop structure as QwenImageEditTrainer)

        for batch in tqdm(train_dataloader, total=len(train_dataloader), desc="cache_embeddings"):
            batch = self.prepare_embeddings(batch, stage="cache")
            self.cache_step(batch)
        self.destroy_models()
        logging.info("Cache completed")

    def destroy_models(self):
        import gc

        if hasattr(self, "text_encoder"):
            self.text_encoder.cpu()
            del self.text_encoder
        if hasattr(self, "text_encoder_2"):
            self.text_encoder_2.cpu()
            del self.text_encoder_2
        if hasattr(self, "vae"):
            self.vae.cpu()
            del self.vae
        if hasattr(self, "vit"):
            self.vit.cpu()
            del self.vit
        torch.cuda.empty_cache()
        gc.collect()

    def clip_gradients(self):
        """Clip gradients"""
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.dit.parameters(),
                self.config.train.max_grad_norm,
            )

    def training_step(self, batch: dict) -> torch.Tensor:
        """Training step, batch from dataloader"""
        if all(batch["cached"]):
            return self._training_step_cached(batch)
        return self._training_step_compute(batch)

    def _training_step_cached(self, batch: dict) -> torch.Tensor:
        """Training step with cached data"""
        embeddings = self.prepare_cached_embeddings(batch)
        return self._compute_loss(embeddings)

    def _training_step_compute(self, batch: dict) -> torch.Tensor:
        """Training step with real-time data"""
        embeddings = self.prepare_embeddings(batch, stage="fit")
        return self._compute_loss(embeddings)

    @abstractmethod
    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        """Compute loss. returned the flow matching loss tensor"""
        pass

    def forward_loss(self, model_pred, target, weighting=None, edit_mask=None, attention_mask=None):
        """
        Forward loss computation with automatic parameter detection.

        Automatically detects which parameters the loss function needs and passes them accordingly.
        All loss functions should accept **kwargs for compatibility.

        Supports multiple loss function signatures:
        - MseLoss: (model_pred, target, weighting, **kwargs)
        - MaskEditLoss: (edit_mask, model_pred, target, weighting, **kwargs)
        - AttentionMaskMseLoss: (model_pred, target, attention_mask, edit_mask, weighting, **kwargs)

        Args:
            model_pred: Model predictions [B, T, C]
            target: Target values [B, T, C]
            weighting: Optional element-wise weights
            edit_mask: Optional edit mask [B, T] (for edit-based losses)
            attention_mask: Optional attention mask [B, T] (for multi-resolution losses)

        Returns:
            Loss tensor (scalar)
        """
        return self.criterion(
            model_pred=model_pred,
            target=target,
            attention_mask=attention_mask,
            edit_mask=edit_mask,
            weighting=weighting,
        )

    def train_epoch(self, epoch, train_dataloader):
        for _, batch in enumerate(train_dataloader):
            # 检查是否收到中断信号
            # print('got batche', batch.keys())
            if self.training_interrupted:
                logger.info("检测到训练中断信号，保存最后检查点后退出本epoch...")
                # 立刻做一次"last"保存（即使不是 checkpointing_steps 整除）
                self.save_checkpoint(epoch, self.global_step, is_last=True)
                return

            with self.accelerator.accumulate(self.dit):
                # print('self.vae device', self.accelerator.process_index, self.vae.device)
                # print('self.text_encoder device', self.accelerator.process_index, self.text_encoder.device)
                # print('self.text_encoder_2 device', self.accelerator.process_index, self.text_encoder_2.device)
                loss = self.training_step(batch)
                # print('B fwd done', self.accelerator.process_index, loss)
                self.fps_logger.update(
                    batch_size=self.batch_size * self.accelerator.num_processes,
                    num_tokens=None,
                )
                self.accelerator.backward(loss)
                # print('E bwd done', self.accelerator.process_index)
                self.clip_gradients()
                # print('clip_gradients', self.accelerator.process_index)
                self.optimizer.step()
                # print('optimizer step', self.accelerator.process_index)
                self.lr_scheduler.step()
                # print('optimizer step', self.accelerator.process_index)
                self.optimizer.zero_grad()
                # print('sync_gradients', self.accelerator.process_index)
            if self.accelerator.sync_gradients:
                avg_loss = self.accelerator.gather(loss.detach()).mean()
                self.train_loss = avg_loss.item() / self.config.train.gradient_accumulation_steps
                self.running_loss = 0.9 * self.running_loss + 0.1 * self.train_loss
                self.update_progressbar(
                    logs={
                        "loss": self.train_loss,
                        "smooth_loss": self.running_loss,
                        "lr": self.lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "fps": self.fps_logger.total_fps(),
                    }
                )
                self.save_checkpoint(epoch, self.global_step)

                # Run validation if needed
                if self.should_run_validation(self.global_step):
                    # logging.info("Starting validation at step %d", self.global_step)
                    self.fps_logger.pause()
                    # logging.info("FPS logger paused, calling run_validation()")
                    self.run_validation()
                    # logging.info("run_validation() returned, resuming FPS logger")
                    self.fps_logger.resume()
                    # logging.info("Validation complete, FPS logger resumed")

    def setup_progressbar(self):
        self.train_loss = 0.0
        self.running_loss = 0.0
        if self.config.resume is not None:
            with open(os.path.join(self.config.resume, "state.json")) as f:
                st = json.load(f)
            self.global_step = st["global_step"]
            self.start_epoch = st["epoch"]
        else:
            self.global_step = 0
            self.start_epoch = 0
        self.progress_bar = tqdm(
            range(self.global_step, self.config.train.max_train_steps),
            desc="fit",
            disable=(not self.accelerator.is_local_main_process),
        )
        # if max_train_steps exist, use is None, use num_epochs
        self.num_epochs = int(self.config.train.max_train_steps / self.batch_size / self.accelerator.num_processes)

    def update_progressbar(self, logs: dict):
        assert self.accelerator is not None
        # 使用LoggerManager记录指标
        self.logger_manager.log_metrics(logs, step=self.global_step)
        self.logger_manager.flush()
        logs = {
            "loss": f"{logs['loss']:.3f}",
            "smooth_loss": f"{logs['smooth_loss']:.3f}",
            "lr": f"{logs['lr']:.1e}",
            "epoch": logs["epoch"],
            "fps": f"{logs['fps']:.2f}",
        }
        self.progress_bar.update(1)
        self.global_step += 1
        self.progress_bar.set_postfix(logs)

    def fit(self, train_dataloader):
        """Main training loop implementation."""
        import torch.multiprocessing as mp

        mp.set_start_method("spawn", force=True)

        self.setup_signal_handlers()
        self.setup_accelerator()
        self.load_model()
        if self.config.resume is not None:
            import glob

            # add the checkpoint in lora.pretrained_weight config
            model_files = glob.glob(os.path.join(self.config.resume, "*.safetensors"))
            if len(model_files) > 0:
                self.config.model.lora.pretrained_weight = model_files[0]
            else:
                self.config.model.lora.pretrained_weight = os.path.join(self.config.resume, LORA_FILE_BASE_NAME)
            logging.info(f"Loaded checkpoint from {self.config.model.lora.pretrained_weight}")
        if self.config.model.quantize:
            self.dit = self.quantize_model(
                self.dit,
                self.config.predict.devices.dit,
            )
        self.dit.to("cpu")
        self.setup_validation(train_dataloader.dataset)
        self.accelerator.wait_for_everyone()
        self.__class__.load_pretrain_lora_model(self.dit, self.config, self.adapter_name)

        self.setup_model_device_train_mode(stage="fit", cache=self.use_cache)
        self.configure_optimizers()
        self.setup_criterion()
        # if validation dataset not passed, use train dataset instead

        train_dataloader = self.accelerator_prepare(train_dataloader)
        logging.info("***** Running training *****")
        model_summary_info_dict = print_model_summary_table(self.dit)
        self.logger_manager.log_table(
            "model_summary",
            rows=model_summary_info_dict["rows"],
            columns=model_summary_info_dict["columns"],
            step=self.global_step,
        )

        self.fps_logger.start()
        self.save_train_config()
        self.setup_progressbar()
        current_epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.num_epochs):
            current_epoch = epoch
            self.train_epoch(epoch, train_dataloader)
            if self.training_interrupted:
                break

        # 保存最后一个检查点
        self.save_checkpoint(current_epoch, self.global_step, is_last=True)

        logging.info(f"FPS: {self.fps_logger.last_fps()}")
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def setup_criterion(self):
        """
        Setup loss criterion from config.

        Supports two modes:
        1. New flexible mode: Use config.loss.class_path and config.loss.init_args
        2. Legacy mode: Use config.loss.mask_loss flag (for backward compatibility)
        """
        if self.config.loss.class_path is not None:
            # New flexible mode: instantiate from class_path
            logger.info(f"Initializing loss from class_path: {self.config.loss.class_path}")
            init_args = self.config.loss.init_args or {}
            self.criterion = instantiate_class(self.config.loss.class_path, init_args)
        else:
            # Legacy mode: backward compatibility
            if self.config.loss.mask_loss:
                from qflux.losses.edit_mask_loss import MaskEditLoss

                logger.info("Using legacy MaskEditLoss configuration")
                self.criterion = MaskEditLoss(
                    forground_weight=self.config.loss.forground_weight,
                    background_weight=self.config.loss.background_weight,
                )
            else:
                from qflux.losses import MseLoss

                logger.info("Using legacy MseLoss configuration")
                self.criterion = MseLoss(reduction="mean")

        self.criterion.to(self.accelerator.device)
        logger.info(f"Loss criterion initialized: {self.criterion.__class__.__name__}")

    def setup_predict(
        self,
    ):
        if self.predict_setted:
            return
        if not hasattr(self, "dit") or self.vae is None:
            logging.info("Loading model...")
            self.load_model()

        if self.config.model.quantize:
            self.dit = self.quantize_model(
                self.dit,
                self.config.predict.devices.dit,
            )

        if self.config.model.lora.pretrained_weight is not None:
            logging.info("load lora from pretrained weight")
            self.load_pretrain_lora_model(self.dit, self.config, self.config.lora_adapter_name, stage="predict")

        self.setup_model_device_train_mode(stage="predict")
        logging.info("setup_model_device_train_mode done")
        print_model_summary_table(self.dit)
        self.predict_setted = True
        logging.info("setup_predict done")

    @abstractmethod
    def prepare_predict_batch_data(self, *args, **kwargs) -> dict:
        """Prepare predict batch data.
        prepare the data to batch dict that can be used to prepare embeddings similar in the training step.
        We want to reuse the same data preparation code in the training step.
        """
        pass

    def predict(
        self,
        image: PIL.Image.Image | list[PIL.Image.Image],
        prompt: str | list[str] | None = None,
        num_inference_steps: int = 20,
        **kwargs,
    ):
        """Inference/prediction method.
        Prepare the data, prepare the embeddings, sample the latents, decode the latents to images.
        """
        self.setup_predict()
        batch = self.prepare_predict_batch_data(
            image=image, prompt=prompt, num_inference_steps=num_inference_steps, **kwargs
        )
        embeddings: dict = self.prepare_embeddings(batch, stage="predict")
        target_height = embeddings["height"]
        target_width = embeddings["width"]
        latents = self.sampling_from_embeddings(embeddings)
        image_ = self.decode_vae_latent(latents, target_height, target_width)
        output_type = kwargs.get("output_type", "pil")
        if output_type == "pil":
            image_np = image_.detach().permute(0, 2, 3, 1).float().cpu().numpy()
            image_np = (image_np * 255).round().astype("uint8")
            if image_.shape[-1] == 1:
                # special case for grayscale (single channel) images
                pil_images = [PIL.Image.fromarray(image_i.squeeze(), mode="L") for image_i in image_np]
            else:
                pil_images = [PIL.Image.fromarray(image_i) for image_i in image_np]
            return pil_images
        return image_

    def setup_accelerator(self):
        """Initialize accelerator and logging configuration"""
        # Setup versioned logging directory
        self.setup_versioned_logging_dir()

        # Set logging_dir to the versioned output directory directly
        # Use project_dir as logging_dir to avoid extra subdirectory creation
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.config.logging.output_dir,
            logging_dir=self.config.logging.output_dir,
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.train.gradient_accumulation_steps,
            # mixed_precision=self.config.train.mixed_precision,
            mixed_precision="no",  # ← 关键
            # log_with=log_with,  # 传入日志工具类型
            project_config=accelerator_project_config,
        )

        # Prepare validation embeddings now that accelerator is initialized
        if hasattr(self, "validation_samples") and self.validation_samples:
            self.prepare_validation_embeddings()
        # 使用LoggerManager
        self.logger_manager = LoggerManager(self.accelerator, self.config, self.versioned_dir, self.experiment_name)
        logging.info(f"Number of devices used in DDP training: {self.accelerator.num_processes}")

        # Set weight data type
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Create output directory
        if self.accelerator.is_main_process and self.config.logging.output_dir is not None:
            os.makedirs(self.config.logging.output_dir, exist_ok=True)

        logging.info(f"Mixed precision: {self.accelerator.mixed_precision}")

    def is_fsdp_enabled(self):
        plug = self.accelerator.state.fsdp_plugin if hasattr(self.accelerator.state, "fsdp_plugin") else None
        return plug is not None

    def save_lora_fsdp(self, save_folder, adapter_name=None):
        """Save LoRA weights under FSDP with CPU offload & rank0-only."""
        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(self.dit) if self.accelerator is not None else self.dit
        if is_compiled_module(unwrapped):
            unwrapped = unwrapped._orig_mod
        adapter_name = self.adapter_name if adapter_name is None else adapter_name
        if save_folder.endswith(".safetensors"):
            print(f"Warning: save_folder {save_folder} should be a folder")
        dev = self.accelerator.device
        for m in unwrapped.modules():
            for attr in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", adapter_name):
                if hasattr(m, attr):
                    getattr(m, attr).to(dev)

        for n, p in unwrapped.named_parameters():
            if "lora_" in n:
                assert not getattr(p, "_is_sharded", False), f"LoRA param {n} is sharded! Fix ignored_modules."
        with torch.inference_mode():
            lora_sd = get_lora_state_dict_oom_safe(unwrapped, adapter_name=adapter_name)
            lora_sd = convert_state_dict_to_diffusers(lora_sd)

        # with torch.no_grad(), ctx:
        if self.accelerator.is_main_process:
            os.makedirs(save_folder, exist_ok=True)
            self.pipeline_class.save_lora_weights(save_folder, lora_sd, safe_serialization=True)
            logging.info(f"[FSDP] Saved LoRA weights to {save_folder} (local-only, no trunk gather)")
        self.accelerator.wait_for_everyone()

    def save_checkpoint(self, epoch, global_step, is_last=False):
        """Save checkpoint"""
        self.fps_logger.pause()
        if not is_last and (self.global_step % self.config.train.checkpointing_steps != 0):
            self.fps_logger.resume()
            return
        logging.info(f"Saving checkpoint to {self.config.logging.output_dir}")
        save_path = os.path.join(self.config.logging.output_dir, f"checkpoint-{epoch}-{global_step}")
        if is_last:
            save_path = os.path.join(self.config.logging.output_dir, f"checkpoint-last-{epoch}-{global_step}-last")
        if self.is_fsdp_enabled():
            os.makedirs(save_path, exist_ok=True)
            self.save_lora_fsdp(save_path, adapter_name=self.adapter_name)
        elif self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)
            self.save_lora(save_path, adapter_name=self.adapter_name)

        state_info = {"global_step": global_step, "epoch": epoch, "is_last": is_last}
        if is_last:
            if not self.is_fsdp_enabled():
                self.accelerator.save_state(save_path)  # when save in fsdp, will cause problem
            git_info = get_git_info()
            state_info.update(git_info)

        if self.accelerator.is_main_process:
            with open(os.path.join(save_path, "state.json"), "w") as f:
                json.dump(state_info, f)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        self.fps_logger.resume()

    def save_lora(self, save_folder, adapter_name=None):
        """Save LoRA weights"""
        if save_folder.endswith(".safetensors"):
            print(f"Warning: save_folder {save_folder} should a folder")
        if self.accelerator is not None:
            unwrapped_transformer = self.accelerator.unwrap_model(self.dit)
        else:
            unwrapped_transformer = self.dit
        if is_compiled_module(unwrapped_transformer):
            unwrapped_transformer = unwrapped_transformer._orig_mod
        adapter_name = self.adapter_name if adapter_name is None else adapter_name

        lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_transformer, adapter_name=adapter_name)
        )
        # Use FluxKontextPipeline's save method if available, otherwise use generic method
        self.pipeline_class.save_lora_weights(save_folder, lora_state_dict, safe_serialization=True)
        logging.info(f"Saved LoRA weights to {save_folder}")

    def save_train_config(self):
        d_json = self.config.model_dump(mode="json", exclude_none=True)
        train_yaml_file = os.path.join(self.config.logging.output_dir, "train_config.yaml")
        os.makedirs(self.config.logging.output_dir, exist_ok=True)
        with open(train_yaml_file, "w") as f:
            yaml.dump(d_json, f, default_flow_style=False, sort_keys=False)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        from diffusers.optimization import get_scheduler

        trainable_named_params = [(name, param) for name, param in self.dit.named_parameters() if param.requires_grad]
        lora_layers = [param for _, param in trainable_named_params]

        # Log how many parameters are in training and show one example
        if (getattr(self, "accelerator", None) is None) or self.accelerator.is_main_process:
            total_elements = sum(p.numel() for p in lora_layers)
            logging.info(f"Trainable parameters: {len(lora_layers)} tensors, total elements: {total_elements}")
            if len(trainable_named_params) > 0:
                example_name, example_param = trainable_named_params[0]
                logging.info(f"Example trainable param: {example_name}, shape={tuple(example_param.shape)}")
                logging.info(f"Example dtype: {example_param.dtype}")

        # Use optimizer parameters from configuration
        optimizer_config = self.config.optimizer.init_args
        class_path = self.config.optimizer.class_path
        module_name, class_name = class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        logging.info(f"Using optimizer: {cls}, {class_path}")
        self.optimizer = cls(
            lora_layers,
            **optimizer_config,
        )

        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_scheduler.warmup_steps,
            num_training_steps=self.config.train.max_train_steps,
        )

    @classmethod
    def quantize_model(cls, model, device):
        model = quantize_model_to_fp8(
            model,
            engine="bnb",
            verbose=True,
            device=device,
        )
        model = model.to(device)
        return model

    @classmethod
    def add_lora_adapter(cls, transformer: torch.nn.Module, config, adapter_name: str):
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=config.model.lora.r,
            lora_alpha=config.model.lora.lora_alpha,
            init_lora_weights=config.model.lora.init_lora_weights,
            target_modules=config.model.lora.target_modules,
        )
        logging.info(f"add_lora_adapter: {lora_config}, {adapter_name}")
        transformer.add_adapter(lora_config, adapter_name=adapter_name)
        transformer.set_adapter(adapter_name)

    @classmethod
    def load_pretrain_lora_model(
        cls,
        transformer: torch.nn.Module,
        config,
        adapter_name: str,
        stage="fit",
    ):
        """
        load the pretrained lora model. Support both local filepath and huggingface repo-id
        Examples of pretrained_weight:
            - TsienDragon/qwen-image-edit-character-composition
            - TsienDragon/qwen-image-edit-character-composition/model.safetensors
            - <local_path>/pytorch_lora_weights.safetensors
            - <local_path>/<filename>.safetensors
        >>>
        """
        pretrained_weight = getattr(config.model.lora, "pretrained_weight", None)
        if pretrained_weight:
            if not os.path.exists(pretrained_weight):
                # if the pretrained_weight is a repo-id, add
                # try as the huggingface repos LORA_FILE_BASE_NAME
                if pretrained_weight.endswith(".safetensors"):
                    repo_id = "/".join(pretrained_weight.split("/")[:2])
                    filename = "/".join(pretrained_weight.split("/")[2:])
                else:
                    repo_id = pretrained_weight
                    filename = LORA_FILE_BASE_NAME
                try:
                    pretrained_weight = download_lora(repo_id, filename)
                except Exception as e:
                    logging.warning(f"Failed to download lora from {pretrained_weight}: {e}")
                    pass

            sha256 = calculate_sha256_file(pretrained_weight)
            lora_type = classify_lora_weight(pretrained_weight)
            logging.info(f"sha256 for pretrained_weight: {sha256} lora_type {lora_type}")

            # DIFFUSERS can be loaded directly, otherwise, need to add lora first
            if lora_type != "PEFT":
                transformer.load_lora_adapter(pretrained_weight, adapter_name=adapter_name)
                logging.info(f"set_lora: {lora_type} Loaded lora from {pretrained_weight} for {adapter_name}")
            else:
                # add lora first
                # Configure model
                cls.add_lora_adapter(transformer, config, adapter_name)

                missing, unexpected = transformer.load_state_dict(
                    safetensors.torch.load_file(pretrained_weight),
                    strict=False,
                )
                if len(unexpected) > 0:
                    raise ValueError(f"Unexpected keys: {unexpected}")
                logging.info(f"set_lora: {lora_type} Loaded lora from {pretrained_weight} for {adapter_name}")
                logging.info(f"missing keys: {len(missing)}, {missing[0]}")
                # self.load_lora(self.config.model.lora.pretrained_weight)
            logging.info(f"set_lora: Loaded lora from {pretrained_weight}")

        elif stage == "fit":
            cls.add_lora_adapter(transformer, config, adapter_name)

    def normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image from [0,1] to [-1,1]"""
        image = image.to(self.weight_dtype)
        return image * 2.0 - 1.0

    def prepare_predict_timesteps(
        self,
        num_inference_steps: int,
        image_seq_len: int,
        scheduler: FlowMatchEulerDiscreteScheduler | None = None,
    ) -> tuple[torch.Tensor, int]:
        """prepare timesteps for prediction

        Args:
            num_inference_steps: Number of inference steps
            image_seq_len: Image sequence length
            scheduler: Scheduler to use. If None, uses self.sampling_scheduler
                (for validation/sampling) or self.scheduler (for training)
        """
        # Use provided scheduler, or sampling_scheduler for validation, or scheduler for training
        if scheduler is None:
            scheduler = getattr(self, "sampling_scheduler", None) or self.scheduler
        assert scheduler is not None, "scheduler must be initialized"
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        device = next(self.dit.parameters()).device
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

    @abstractmethod
    def load_model(self, **kwargs):
        """Load and initialize model components."""
        pass

    @abstractmethod
    def encode_prompt(self, *args, **kwargs):
        """Encode text prompts to embeddings. Qwen-Edit pass image and prompt, Flux-Kontext pass prompt"""
        pass

    @abstractmethod
    def prepare_latents(self, *args, **kwargs):
        """Prepare latents for fit & predict. Input usually be control images"""
        pass

    @abstractmethod
    def prepare_embeddings(self, batch: dict, stage: str = "fit") -> dict[str, torch.Tensor]:
        """Prepare embeddings for prediction. Call vae encoder and prompt encoder
        to get the embeddings. Used in fit & predict
        Update the embeddings keys in batch dict
        """
        return batch

    @abstractmethod
    def prepare_cached_embeddings(self, batch: dict) -> dict[str, torch.Tensor]:
        """Prepare cached embeddings for prediction. Loaded the cached embeddings from cache.
        Used in fit
        Update the embeddings keys in batch dict
        """
        return batch

    @abstractmethod
    def decode_vae_latent(self, latents: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """Decode VAE latent vectors to RGB images. In range [0,1]"""
        pass

    @abstractmethod
    def sampling_from_embeddings(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sampling from embeddings. Only handle the latent diffusion steps. Output the final latents. Need
        to decode the latents to images.
        """
        pass

    @abstractmethod
    def cache_step(self, data: dict):
        """Cache step"""
        pass

    @abstractmethod
    def setup_model_device_train_mode(self, stage="fit", cache=False):
        pass
