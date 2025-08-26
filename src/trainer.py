import copy
import os
import shutil
import torch
import random
import PIL
from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.loaders import AttnProcsLayers

from diffusers import QwenImagePipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from peft.utils import get_peft_model_state_dict
import math
import numpy as np
from src.models.load_model import load_transformer, load_vae, load_qwenvl
from src.utils.logger import get_logger
from src.data.cache_manager import check_cache_exists


logger = get_logger(__name__, log_level="INFO")


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    """
        计算FlowMatchEulerDiscreteScheduler的mu参数值

    Args:
        image_seq_len: 当前图像的序列长度（token数量）
        base_seq_len: 基准序列长度，默认256
        max_seq_len: 最大序列长度，默认4096
        base_shift: 基准偏移值，默认0.5
                max_shift: 最大偏移值，默认1.16

    Returns:
        计算得到的mu值
    """
    # 线性插值计算mu值
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b

    # 确保mu在合理范围内
    mu = max(base_shift, min(max_shift, mu))

    return mu


def get_lora_layers(model):
    """这个 get_lora_layers 函数的作用是遍历整个神经网络模型,找出并收集所有与LoRA相关的模块"""
    lora_layers = {}

    def fn_recursive_find_lora_layer(name: str, module: torch.nn.Module, processors):
        if "lora" in name:
            lora_layers[name] = module
            print(name)
        for sub_name, child in module.named_children():
            fn_recursive_find_lora_layer(f"{name}.{sub_name}", child, lora_layers)

        return lora_layers

    for name, module in model.named_children():
        fn_recursive_find_lora_layer(name, module, lora_layers)

    return lora_layers


class Trainer:
    def __init__(self, config):
        self.config = config
        self.accelerator = None
        self.models = {}
        self.optimizer = None
        self.lr_scheduler = None
        self.noise_scheduler = None
        self.global_step = 0
        self.cache_exist = check_cache_exists(self.config.cache.cache_dir)
        self.use_cache = self.config.cache.use_cache
        self.qunantize = self.config.model.quantize
        self.weight_dtype = torch.bfloat16  # default use bfloat16 dtype
        self.batch_size = self.config.data.batch_size
        self.prompt_image_dropout_rate = self.config.data.init_args['prompt_image_dropout_rate']

    def setup_models(self):
        """加载和配置模型"""
        # 加载模型
        self.models["text_encoder"] = load_qwenvl(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype,
        )
        self.models["vae"] = load_vae(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype,
        )
        self.models["transformer"] = load_transformer(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype,
        )

        # 冻结非 LoRA 参数
        self.models["vae"].requires_grad_(False)
        self.models["transformer"].requires_grad_(False)
        self.vae_scale_factor = 2 ** len(self.models["vae"].temperal_downsample)
        self.pipline_resize_fn = self.models["text_encoder"].image_processor.resize
        self.vae_latent_mean = self.models["vae"].config.latents_mean
        self.vae_latent_std = self.models["vae"].config.latents_std
        self.vae_z_dim = self.models["vae"].config.z_dim

    def setup_accelerator(self):
        """初始化加速器和日志配置"""
        logging_dir = os.path.join(
            self.config.logging.output_dir, self.config.logging.logging_dir
        )
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.config.logging.output_dir, logging_dir=logging_dir
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.train.gradient_accumulation_steps,
            mixed_precision=self.config.train.mixed_precision,
            log_with=self.config.logging.report_to,
            project_config=accelerator_project_config,
        )

        # 设置权重数据类型

        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # 创建输出目录
        if (
            self.accelerator.is_main_process
            and self.config.logging.output_dir is not None
        ):
            os.makedirs(self.config.logging.output_dir, exist_ok=True)

        logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")

    def unwrap_model(self, model):
        """- Purpose: Removes wrappers added by Hugging Face Accelerate library
        - What it removes:
            - DistributedDataParallel (DDP) wrappers for multi-GPU training
            - DataParallel wrappers
            - Other distributed training wrappers
        - Why needed: During distributed training, Accelerate wraps your model to handle parallel execution
        """
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def quantize_model(self, diffusion_transformer, device):
        from optimum.quanto import quantize, qfloat8, freeze

        torch_dtype = self.weight_dtype
        all_blocks = list(diffusion_transformer.transformer_blocks)
        for block in tqdm(all_blocks):
            block.to(device, dtype=torch_dtype)
            quantize(block, weights=qfloat8)
            freeze(block)
            block.to("cpu")
        diffusion_transformer.to(device, dtype=torch_dtype)
        quantize(diffusion_transformer, weights=qfloat8)
        freeze(diffusion_transformer)
        return diffusion_transformer

    def setup_noise_scheduler(self):
        # 设置调度器
        from diffusers import FlowMatchEulerDiscreteScheduler

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            subfolder="scheduler",
        )

    def setup_lora(self):
        # 配置 LoRA
        from peft import LoraConfig

        if self.qunantize:
            self.models["transformer"] = self.quantize_model(self.models["transformer"], self.accelerator.device)
        else:
            self.models["transformer"].to(self.accelerator.device)

        lora_config = LoraConfig(
            r=self.config.model.lora.r,
            lora_alpha=self.config.model.lora.lora_alpha,
            init_lora_weights=self.config.model.lora.init_lora_weights,
            target_modules=self.config.model.lora.target_modules,
        )

        # 配置模型
        if self.qunantize:
            self.models["transformer"].to(self.accelerator.device)
        else:
            self.models["transformer"].to(
                self.accelerator.device, dtype=self.weight_dtype
            )
        self.models["transformer"].add_adapter(lora_config)
        self.models["transformer"].requires_grad_(False)
        self.models["transformer"].train()
        self.models["transformer"].enable_gradient_checkpointing()

        # 只训练 LoRA 参数
        trainable_params = 0
        for name, param in self.models["transformer"].named_parameters():
            if "lora" in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False

        logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    def load_pretrain_lora(self, pretrained_weight):
        if pretrained_weight is not None:
            self.models['transformer'].load_lora_adapter(
                pretrained_weight,
            )

    def optimize_for_inference(self):
        """优化模型以提升推理性能"""
        # 设置 VAE 为评估模式并启用推理优化
        # 为所有模型启用 bf16 推理优化
        # 确保所有非训练模型使用 bf16
        for model_name, model in self.models.items():
            if model_name != "transformer":  # transformer 在训练时需要特殊处理
                if self.weight_dtype == torch.bfloat16:
                    model.to(dtype=self.weight_dtype)
                model.eval()

        # 启用推理优化标志
        torch.backends.cudnn.benchmark = (
            True  # 对于固定输入尺寸的情况，可以优化卷积操作
        )
        logger.info(f"模型推理优化完成，使用精度: {self.weight_dtype}")

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        lora_layers = filter(
            lambda p: p.requires_grad, self.models["transformer"].parameters()
        )
        self.optimizer = torch.optim.AdamW(
            lora_layers,
            lr=self.config.optimizer.init_args["lr"],
            betas=self.config.optimizer.init_args["betas"],
            weight_decay=self.config.optimizer.init_args["weight_decay"],
            eps=self.config.optimizer.init_args["eps"],
        )

        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_scheduler.warmup_steps
            * self.accelerator.num_processes,
            num_training_steps=self.config.train.max_train_steps
            * self.accelerator.num_processes,
        )

    def accelerator_prepare(self, train_dataloader):
        lora_layers_model = AttnProcsLayers(get_lora_layers(self.models["transformer"]))
        self.models["transformer"].enable_gradient_checkpointing()

        lora_layers_model, optimizer, _, lr_scheduler = self.accelerator.prepare(
            lora_layers_model, self.optimizer, train_dataloader, self.lr_scheduler
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # 初始化追踪器
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                self.config.logging.tracker_project_name, {"test": None}
            )
        return train_dataloader

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        """计算噪声调度器的 sigma 值"""
        noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        sigmas = noise_scheduler_copy.sigmas.to(
            device=self.accelerator.device, dtype=dtype
        )
        schedule_timesteps = noise_scheduler_copy.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def set_model_devices(self, cache_exist=False, use_cache=False, mode="train"):
        """unload embedding models to save memory according to different mode"""
        import gc

        if mode == "train":
            assert hasattr(
                self, "accelerator"
            ), "accelerator must be set before setting model devices"

        # train
        if cache_exist and use_cache and mode == "train":
            # 1. use cache embedding:
            #  need transformer
            # all on devices
            self.models["text_encoder"].cpu()
            torch.cuda.empty_cache()
            self.models["vae"].cpu()
            torch.cuda.empty_cache()
            del self.models["text_encoder"]
            del self.models["vae"]
            gc.collect()

            # setup devices
            self.models["transformer"].to(self.accelerator.device)

        elif use_cache is False and mode == "train":
            # 2. not use cache embedding:
            # need vae.encoder
            # need text_encoder
            # need transformer
            self.models["vae"].decoder.cpu()
            torch.cuda.empty_cache()
            gc.collect()

            self.models["vae"].encoder.to(self.accelerator.device)
            self.models["text_encoder"].to(self.accelerator.device)
            self.models["transformer"].to(self.accelerator.device)

        elif mode == "cache":
            # cache
            # 1.cache embedding
            # need vae.encoder
            # need text_encoder
            self.models["transformer"].cpu()
            torch.cuda.empty_cache()
            del self.models["transformer"]
            gc.collect()
            self.models["vae"].decoder.cpu()
            torch.cuda.empty_cache()
            gc.collect()

    def encode_image(self, image: torch.Tensor, device, device_type="cuda"):
        # 使用 inference_mode() 获得更好的推理性能
        image = self.image_preprocess_for_cache(image, adaptive_resolutioin=False)
        pixel_values = self.vae_image_standarization(image)
        with torch.inference_mode():
            # 使用 autocast 确保 bf16 推理
            with torch.autocast(
                device_type=device_type, dtype=self.weight_dtype, enabled=True
            ):
                pixel_values = pixel_values.to(
                    dtype=self.weight_dtype, device=device, non_blocking=True
                )
                pixel_latents = (
                    self.models["vae"].encode(pixel_values).latent_dist.sample()
                )
        return pixel_latents[0]

    def encode_prompt(
        self, prompt: str, prompt_image: torch.Tensor, device: str, device_type="cuda"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """only support single image"""
        text_encoding_pipeline = self.models["text_encoder"]
        prompt_image = self.image_preprocess_for_cache(
            prompt_image, adaptive_resolutioin=True
        )
        # 使用 inference_mode() 获得更好的推理性能
        with torch.inference_mode():
            # 使用 autocast 确保 bf16 推理
            with torch.autocast(
                device_type=device_type, dtype=self.weight_dtype, enabled=True
            ):
                prompt_embeds, prompt_embeds_mask = (
                    text_encoding_pipeline.encode_prompt(
                        image=prompt_image,
                        prompt=[prompt],
                        device=device,
                        num_images_per_prompt=1,
                        max_sequence_length=1024,
                    )
                )
        return prompt_embeds[0], prompt_embeds_mask[0]

    def encode_empty_prompt(
        self, prompt_image: torch.Tensor, device: str, device_type="cuda"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # random dropout the prompt, will use empty prompt in the dataset by `caption_dropout_rate`
        text_encoding_pipeline = self.models["text_encoder"]
        prompt_image = self.image_preprocess_for_cache(
            prompt_image, adaptive_resolutioin=True
        )
        with torch.inference_mode():
            with torch.autocast(
                device_type=device_type, dtype=self.weight_dtype, enabled=True
            ):
                prompt_embeds, prompt_embeds_mask = (
                    text_encoding_pipeline.encode_prompt(
                        image=prompt_image,
                        prompt=[""],
                        device=device,
                        num_images_per_prompt=1,
                        max_sequence_length=1024,
                    )
                )
        return prompt_embeds[0], prompt_embeds_mask[0]

    def training_step(self, batch):
        """执行单个训练步骤"""
        # 检查是否有缓存数据
        if batch.get("cached", False):
            return self._training_step_cached(batch)
        else:
            return self._training_step_compute(batch)

    def _training_step_cached(self, batch):
        """使用缓存嵌入的训练步骤"""
        # 从缓存数据中获取嵌入
        pixel_latents = batch["pixel_latent"].to(
            self.accelerator.device, dtype=self.weight_dtype
        )
        control_latents = batch["control_latent"].to(
            self.accelerator.device, dtype=self.weight_dtype
        )
        prompt_embeds = batch["prompt_embed"].to(self.accelerator.device)
        prompt_embeds_mask = batch["prompt_embeds_mask"].to(self.accelerator.device)
        # TODO: 暂时不使用prompt_image_dropout_rate
        # if random.random() < self.prompt_image_dropout_rate:
        #     empty_prompt_images_mask = torch.rand(prompt_embeds_mask.shape[0]) < self.prompt_image_dropout_rate
        #     prompt_embeds_mask = torch.where(
        #         empty_prompt_images_mask,
        #         torch.zeros_like(prompt_embeds_mask),
        #         prompt_embeds_mask)
        #     prompt_embeds = torch.where(empty_prompt_images_mask, torch.zeros_like(prompt_embeds), prompt_embeds)

        return self._compute_loss(
            pixel_latents, control_latents, prompt_embeds, prompt_embeds_mask
        )

    def _training_step_compute(self, batch):
        """计算嵌入的训练步骤（无缓存）"""
        image, control, prompt = batch["image"], batch["control"], batch["prompt"]
        pixel_latents = self.encode_image(image)
        control_latents = self.encode_image(control)
        if random.random() < self.config.train.caption_dropout_rate:
            prompt = ""
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt, control)
        # 使用 inference_mode() 进行编码推理，性能更好
        with torch.inference_mode():
            # 编码图像
            # 标准化潜在向量
            vae_config = self.models["vae"].config
            latents_mean = (
                torch.tensor(vae_config.latents_mean, dtype=self.weight_dtype)
                .view(1, 1, vae_config.z_dim, 1, 1)
                .to(pixel_latents.device, non_blocking=True)
            )
            latents_std = (
                (1.0 / torch.tensor(vae_config.latents_std, dtype=self.weight_dtype))
                .view(1, 1, vae_config.z_dim, 1, 1)
                .to(pixel_latents.device, non_blocking=True)
            )
            pixel_latents = (pixel_latents - latents_mean) * latents_std
            control_latents = (control_latents - latents_mean) * latents_std

        return self._compute_loss(
            pixel_latents, control_latents, prompt_embeds, prompt_embeds_mask
        )

    def _compute_loss(
        self, pixel_latents, control_latents, prompt_embeds, prompt_embeds_mask
    ):
        """计算损失的通用方法"""
        pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)
        control_latents = control_latents.permute(0, 2, 1, 3, 4)
        latents_mean = (
            torch.tensor(self.vae_latent_mean)
            .view(1, 1, self.vae_z_dim, 1, 1)
            .to(pixel_latents.device, pixel_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae_latent_std).view(1, 1, self.vae_z_dim, 1, 1).to(
            pixel_latents.device, pixel_latents.dtype
        )
        pixel_latents = (pixel_latents - latents_mean) * latents_std
        control_latents = (control_latents - latents_mean) * latents_std

        with torch.no_grad():
            bsz = pixel_latents.shape[0]
            noise = torch.randn_like(
                pixel_latents, device=self.accelerator.device, dtype=self.weight_dtype
            )
            # sampling time step
            u = compute_density_for_timestep_sampling(
                weighting_scheme="none",
                batch_size=bsz,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler.timesteps[indices].to(
                device=pixel_latents.device
            )

            sigmas = self.get_sigmas(
                timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype
            )
            noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

            # 打包潜在向量
            packed_noisy_model_input = QwenImagePipeline._pack_latents(
                noisy_model_input,
                bsz,
                noisy_model_input.shape[2],
                noisy_model_input.shape[3],
                noisy_model_input.shape[4],
            )

            packed_control_latents = QwenImagePipeline._pack_latents(
                control_latents,
                bsz,
                control_latents.shape[2],
                control_latents.shape[3],
                control_latents.shape[4],
            )

            # 编码文本
            img_shapes = [
                [
                    (
                        1,
                        noisy_model_input.shape[3] // 2,
                        noisy_model_input.shape[4] // 2,
                    ),
                    (1, control_latents.shape[3] // 2, control_latents.shape[4] // 2),
                ]
            ] * bsz
            packed_noisy_model_input_concated = torch.cat(
                [packed_noisy_model_input, packed_control_latents], dim=1
            )
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

            prompt_embeds = prompt_embeds.repeat(bsz, 1, 1)

        # 前向传播
        model_pred = self.models["transformer"](
            hidden_states=packed_noisy_model_input_concated,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]
        model_pred = model_pred[:, : packed_noisy_model_input.size(1)]

        model_pred = QwenImagePipeline._unpack_latents(
            model_pred,
            height=noisy_model_input.shape[3] * self.vae_scale_factor,
            width=noisy_model_input.shape[4] * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )

        # 计算损失
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme="none", sigmas=sigmas
        )
        target = noise - pixel_latents
        target = target.permute(0, 2, 1, 3, 4)

        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                target.shape[0], -1
            ),
            1,
        )
        loss = loss.mean()

        return loss

    def save_checkpoint(self, epoch, global_step):
        """保存检查点"""
        if not self.accelerator.is_main_process:
            return

        # 管理检查点数量
        if self.config.train.checkpoints_total_limit is not None:
            checkpoints = os.listdir(self.config.logging.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            if len(checkpoints) >= self.config.train.checkpoints_total_limit:
                num_to_remove = (
                    len(checkpoints) - self.config.train.checkpoints_total_limit + 1
                )
                removing_checkpoints = checkpoints[0:num_to_remove]

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(
                        self.config.logging.output_dir, removing_checkpoint
                    )
                    shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(
            self.config.logging.output_dir, f"checkpoint-{epoch}-{global_step}"
        )
        os.makedirs(save_path, exist_ok=True)

        # 保存 LoRA 权重
        unwrapped_transformer = self.accelerator.unwrap_model(
            self.models["transformer"]
        )
        if is_compiled_module(unwrapped_transformer):
            unwrapped_transformer = unwrapped_transformer._orig_mod

        lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_transformer)
        )

        QwenImagePipeline.save_lora_weights(
            save_path, lora_state_dict, safe_serialization=True
        )
        logger.info(f"Saved checkpoint to {save_path}")

    def fit(self, train_dataloader):
        """主训练循环"""
        self.setup_accelerator()
        self.setup_models()
        self.setup_lora()
        self.setup_noise_scheduler()
        self.configure_optimizers()
        self.set_model_devices(cache_exist=self.cache_exist, use_cache=self.use_cache, mode="train")
        train_dataloader = self.accelerator_prepare(train_dataloader)

        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {self.batch_size}")
        logger.info(
            f"  Gradient Accumulation steps = {self.config.train.gradient_accumulation_steps}"
        )

        # 进度条
        progress_bar = tqdm(
            range(0, self.config.train.max_train_steps),
            desc="train",
            disable=not self.accelerator.is_local_main_process,
        )

        # 训练循环
        train_loss = 0.0
        running_loss = 0.0
        for epoch in range(self.config.train.num_epochs):
            for _, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.models["transformer"]):
                    loss = self.training_step(batch)

                    # 反向传播
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.models["transformer"].parameters(),
                            self.config.train.max_grad_norm,
                        )

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # 同步梯度时更新
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1

                    # 计算平均损失
                    avg_loss = self.accelerator.gather(
                        loss.repeat(self.config.train.train_batch_size)
                    ).mean()
                    train_loss += (
                        avg_loss.item() / self.config.train.gradient_accumulation_steps
                    )
                    running_loss = train_loss

                    # 记录日志
                    self.accelerator.log(
                        {"train_loss": train_loss}, step=self.global_step
                    )
                    train_loss = 0.0

                    # 保存检查点
                    if self.global_step % self.config.train.checkpointing_steps == 0:
                        self.save_checkpoint(epoch, self.global_step)

                # 更新进度条
                logs = {
                    "loss": f"{running_loss:.3f}",
                    "lr": f"{self.lr_scheduler.get_last_lr()[0]:.1e}",
                }
                progress_bar.set_postfix(**logs)

                # 检查是否达到最大步数
                if self.global_step >= self.config.train.max_train_steps:
                    break

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def image_preprocess_for_cache(
        self, image: torch.Tensor, adaptive_resolutioin=True
    ):
        """image: torch tensor of RGB image with shape [3, img_h, img_w]"""
        # convert to PIL.image
        image = Image.fromarray(image.permute(1, 2, 0).cpu().numpy().astype("uint8"))
        if adaptive_resolutioin:
            calculated_width, calculated_height, _ = calculate_dimensions(
                1024 * 1024, image.size[0] / image.size[1]
            )
            image = self.pipline_resize_fn(image, calculated_height, calculated_width)
        return image

    def vae_image_standarization(self, image: PIL.Image):
        image = np.array(image).astype("float32")
        image = (image / 127.5) - 1
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        pixel_values = image.unsqueeze(2)
        return pixel_values

    def cache_step(self, batch, vae_encoder_device, text_encoder_device):
        """缓存步骤"""
        image, control, prompt = batch["image"], batch["control"], batch["prompt"]
        # image: torch tensor of RGB image with shape [3, img_h, img_w]
        # control: torch tensor of RGB image with shape [3, img_h, img_w]
        # prompt: str
        file_hashes = batch["file_hashes"]
        image_hash = file_hashes["image_hash"]
        control_hash = file_hashes["control_hash"]
        prompt_hash = file_hashes["prompt_hash"]
        empty_prompt_hash = file_hashes["empty_prompt_hash"]
        n_samples = image.shape[0]

        for i in range(n_samples):
            prompt_i = prompt[i]
            image_i = image[i]
            control_i = control[i]

            pixel_latent = self.encode_image(
                image_i, vae_encoder_device
            )  # torch: [B，C，H,W]
            control_latent = self.encode_image(
                control_i, vae_encoder_device
            )  # torch: [B，C，H,W]
            print("size", image_i.size(), control_i.size(), type(image_i), type(control_i))

            prompt_embed, prompt_embeds_mask = self.encode_prompt(
                prompt_i, control_i, text_encoder_device
            )
            empty_prompt_embed, empty_prompt_embeds_mask = self.encode_empty_prompt(
                control_i, text_encoder_device
            )
            print("shape of pixel_latent", pixel_latent.shape)
            print("shape of control_latent", control_latent.shape)
            print("shape of prompt_embed", prompt_embed.shape)
            print("shape of prompt_embeds_mask", prompt_embeds_mask.shape)
            print("shape of empty_prompt_embed", empty_prompt_embed.shape)
            print("shape of empty_prompt_embeds_mask", empty_prompt_embeds_mask.shape)
            # shape of pixel_latent torch.Size([16, 1, 104, 72])
            # shape of control_latent torch.Size([16, 1, 104, 72])
            # shape of prompt_embed torch.Size([1639, 3584])
            # shape of prompt_embeds_mask torch.Size([1639])
            # shape of empty_prompt_embed torch.Size([1340, 3584])
            # shape of empty_prompt_embeds_mask torch.Size([1340])
            self.cache_manager.save_cache("pixel_latent", image_hash[i], pixel_latent)
            self.cache_manager.save_cache(
                "control_latent", control_hash[i], control_latent
            )
            self.cache_manager.save_cache("prompt_embed", prompt_hash[i], prompt_embed)
            self.cache_manager.save_cache(
                "prompt_embeds_mask", prompt_hash[i], prompt_embeds_mask
            )
            self.cache_manager.save_cache(
                "empty_prompt_embed", empty_prompt_hash[i], empty_prompt_embed
            )
            self.cache_manager.save_cache(
                "empty_prompt_embeds_mask",
                empty_prompt_hash[i],
                empty_prompt_embeds_mask,
            )

    def cache(self, train_dataloader):
        from tqdm import tqdm

        self.cache_manager = train_dataloader.cache_manager
        vae_encoder_device = self.config.cache.vae_encoder_device
        text_encoder_device = self.config.cache.text_encoder_device
        self.weight_dtype = torch.bfloat16
        self.setup_models()  # load the model
        self.set_model_devices(cache_exist=False, use_cache=True, mode="cache")

        self.models["text_encoder"].to(text_encoder_device)
        self.models["vae"].to(vae_encoder_device, dtype=self.weight_dtype)
        self.models["vae"].decoder.to("cpu")

        for _, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            self.cache_step(batch, vae_encoder_device, text_encoder_device)
        print("Cache completed")
        self.models["text_encoder"].cpu()
        self.models["vae"].cpu()
        # unload models and exit
        del self.models["text_encoder"]
        del self.models["vae"]
        exit()

    def setup_predict(self):
        # 确保模型处于评估模式
        self.setup_models()
        self.load_pretrain_lora(self.config.model.lora.pretrained_weight)
        self.setup_noise_scheduler()
        self.set_model_devices(cache_exist=False, use_cache=False, mode="predict")
        self.models['transformer'].eval()
        self.models['vae'].eval()
        # allocate to corresponding device and quantize
        if self.qunantize:
            self.models['transformer'] = self.quantize_model(
                self.models['transformer'],
                self.config.predict.devices['transformer']
            )
            # has problem
        else:
            self.models['transformer'].to(self.config.predict.devices['transformer'])
        self.models['vae'].to(self.config.predict.devices['vae'])
        self.models['text_encoder'].to(self.config.predict.devices['text_encoder'])

    def predict(
        self,
        prompt_image: np.ndarray,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        true_cfg_scale: float = 4.0
    ) -> np.ndarray:
        """
        对单张图片进行推理预测
        Args:
            prompt_image: numpy.ndarray RGB [H,W,C], 输入的提示图片
            prompt: str, 文本提示
            negative_prompt: str, 负面文本提示，默认为空
            num_inference_steps: int, 推理步数，默认20
            true_cfg_scale: float, 真实CFG引导强度，默认4.0
        Returns:
            numpy.ndarray: 生成的图片，RGB格式 [H,W,C]
        """

        # 图像预处理：numpy -> torch tensor
        prompt_image_tensor = self._preprocess_image(prompt_image)
        intermediate_images = []
        # 文本和图像编码
        with torch.inference_mode():
            # auto cast 会自动转化某些层为 float32
            if True:
                # with torch.autocast(device_type="cuda", dtype=self.weight_dtype, enabled=True):
                # 编码提示图像
                prompt_embeds, prompt_embeds_mask = self.encode_prompt(
                    prompt,
                    prompt_image_tensor,
                    self.config.predict.devices['text_encoder']
                )
                # [2204, 3584] prompt_embeds
                # [2204] prompt_embeds_mask

                # 编码控制图像 (使用相同的提示图像)
                control_latents = self.encode_image(
                    prompt_image_tensor,
                    self.config.predict.devices['vae']
                )

                # control_latents: [16,1,88,128]
                control_latents = control_latents.unsqueeze(0)
                # [1,16,1,88,128]
                # 重要：对编码后的latents进行维度调整和标准化
                # 从 [B, C, T, H, W] 转换为 [B, T, C, H, W]

                control_latents = control_latents.permute(0, 2, 1, 3, 4)
                # [1,1,16,88,128] [B, T, C, H, W]

                # VAE latents标准化

                latents_mean = (
                    torch.tensor(self.vae_latent_mean, dtype=self.weight_dtype)
                    .view(1, 1, self.vae_z_dim, 1, 1)
                    .to(control_latents.device)
                )
                # [1, 1, 16, 1, 1]: [B, T, C, H, W]
                latents_std = (
                    1.0 / torch.tensor(self.vae_latent_std, dtype=self.weight_dtype)
                    .view(1, 1, self.vae_z_dim, 1, 1)
                    .to(control_latents.device)
                )
                # [1, 1, 16, 1, 1]: [B,T,C,H,W]

                control_latents = (control_latents - latents_mean) * latents_std
                # [1, 1, 16, 88, 128]: [B,T,C,H,W]

                # 生成随机噪声作为初始状态
                latent_height = control_latents.shape[3]
                latent_width = control_latents.shape[4]

                # 计算图像序列长度用于动态mu计算
                # VAE编码后的latent尺寸需要除以2来得到实际的patch数量
                image_seq_len = (latent_height // 2) * (latent_width // 2)
                mu = calculate_shift(image_seq_len)

                # 设置推理调度器
                self.noise_scheduler.set_timesteps(
                    num_inference_steps,
                    device=self.config.predict.devices['transformer'],
                    mu=mu
                )
                timesteps = self.noise_scheduler.timesteps
                # timesteps from 1000 to 0, length = num_inference_steps

                # 生成初始噪声作为起始状态 (pure noise at t=1)
                latents = torch.randn(
                    (1, control_latents.shape[1], control_latents.shape[2], latent_height, latent_width),
                    device=self.config.predict.devices['transformer'],
                    dtype=self.weight_dtype
                )
                # [1, 1, 16, 88, 128]: [B,T,C,H,W]

                print(f"Initial latents shape: {latents.shape}")
                print(f"Initial latents mean: {latents.mean():.4f}, std: {latents.std():.4f}")
                print(f"Timesteps: {timesteps[:5]} ... {timesteps[-5:]}")
                print(f"Scheduler sigmas: {self.noise_scheduler.sigmas[:5]} ... {self.noise_scheduler.sigmas[-5:]}")
                intermediate_images.append(self.decode_image(control_latents))

                # 准备negative prompt embeds (如果需要CFG)
                use_cfg = true_cfg_scale > 1.0 and negative_prompt.strip() != ""
                if use_cfg:
                    # 编码negative prompt
                    negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                        negative_prompt, prompt_image_tensor, self.config.predict.devices['text_encoder']
                    )
                    print(f"Using CFG with scale: {true_cfg_scale}")
                    print(f"Negative prompt: '{negative_prompt}'")
                else:
                    print("No CFG - negative prompt empty or scale <= 1.0")

                # 移动prompt embeds到transformer device
                prompt_embeds = prompt_embeds.to(
                    device=self.config.predict.devices['transformer'], dtype=self.weight_dtype
                )
                prompt_embeds_mask = prompt_embeds_mask.to(
                    device=self.config.predict.devices['transformer']
                )

                if use_cfg:
                    negative_prompt_embeds = negative_prompt_embeds.to(
                        device=self.config.predict.devices['transformer'], dtype=self.weight_dtype
                    )
                    negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(
                        device=self.config.predict.devices['transformer']
                    )

                # 降噪循环
                from tqdm import tqdm
                progress_bar = tqdm(enumerate(timesteps), total=len(timesteps), desc="Generating images")
                for _, t in progress_bar:
                    # 显示当前时间步
                    progress_bar.set_postfix({'timestep': f'{t:.1f}'})

                    intermediate_images.append(self.decode_image(latents))

                    # 打包单个latents（不进行批量扩展）
                    packed_latents_single = QwenImagePipeline._pack_latents(
                        latents,  # [1, 1, 16, 88, 128]
                        1,  # batch_size = 1
                        latents.shape[2],
                        latents.shape[3],
                        latents.shape[4],
                    )
                    # [1, 2816, 64] shape of packed_latents_single

                    packed_control_latents_single = QwenImagePipeline._pack_latents(
                        control_latents,
                        1,  # batch_size = 1
                        control_latents.shape[2],
                        control_latents.shape[3],
                        control_latents.shape[4],
                    )
                    # [1, 2816, 64] shape of packed control latents

                    # 准备单个输入
                    packed_latents_single = packed_latents_single.to(
                            self.config.predict.devices['transformer'],
                            dtype=self.weight_dtype
                        )
                    packed_control_latents_single = packed_control_latents_single.to(
                            self.config.predict.devices['transformer'],
                            dtype=self.weight_dtype
                        )
                    packed_input_single = torch.cat([packed_latents_single, packed_control_latents_single], dim=1)
                    # [1, 5632, 64] shape of packed_input_single
                    packed_input_single = packed_input_single.to(
                        device=self.config.predict.devices['transformer'], dtype=self.weight_dtype
                    )

                    # 准备图像形状信息
                    img_shapes_single = [[
                        (1, latents.shape[3] // 2, latents.shape[4] // 2),
                        (1, control_latents.shape[3] // 2, control_latents.shape[4] // 2),
                    ]]

                    # 准备文本序列长度
                    txt_seq_lens_single = [prompt_embeds_mask.sum().item()]

                    # 准备timestep tensor
                    timestep_tensor = torch.tensor([t / 1000], dtype=self.weight_dtype).to(
                        self.config.predict.devices['transformer']
                    )

                    velocity_pred_final = None

                    # 如果使用CFG，先进行negative forward
                    if use_cfg:
                        # Negative forward pass
                        velocity_pred_uncond = self.models["transformer"](
                            hidden_states=packed_input_single,
                            timestep=timestep_tensor,
                            guidance=None,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask.unsqueeze(0),
                            encoder_hidden_states=negative_prompt_embeds.unsqueeze(0),
                            img_shapes=img_shapes_single,
                            txt_seq_lens=txt_seq_lens_single,
                            return_dict=False,
                        )[0]
                        # [1, 5632, 64] : [B,T,C]

                        # 提取并unpack negative prediction
                        velocity_pred_uncond = velocity_pred_uncond[:, :packed_latents_single.size(1)]
                        velocity_pred_uncond = QwenImagePipeline._unpack_latents(
                            velocity_pred_uncond,
                            height=latents.shape[3] * self.vae_scale_factor,
                            width=latents.shape[4] * self.vae_scale_factor,
                            vae_scale_factor=self.vae_scale_factor,
                        )
                        # [1, 16, 1, 88, 128]: [B,C,T,H,W]

                    # Positive forward pass
                    velocity_pred_text = self.models["transformer"](
                        hidden_states=packed_input_single,
                        timestep=timestep_tensor,
                        guidance=None,
                        encoder_hidden_states_mask=prompt_embeds_mask.unsqueeze(0),
                        encoder_hidden_states=prompt_embeds.unsqueeze(0),
                        img_shapes=img_shapes_single,
                        txt_seq_lens=txt_seq_lens_single,
                        return_dict=False,
                    )[0]
                    # [1, 5632, 64] : [B,T,C]

                    # 提取并unpack positive prediction
                    velocity_pred_text = velocity_pred_text[:, :packed_latents_single.size(1)]
                    velocity_pred_text = QwenImagePipeline._unpack_latents(
                        velocity_pred_text,
                        height=latents.shape[3] * self.vae_scale_factor,
                        width=latents.shape[4] * self.vae_scale_factor,
                        vae_scale_factor=self.vae_scale_factor,
                    )
                    # [1, 16, 1, 88, 128]: [B,C,T,H,W]

                    # 应用CFG公式
                    if use_cfg:
                        velocity_pred_final = (velocity_pred_uncond +
                                               true_cfg_scale * (velocity_pred_text - velocity_pred_uncond))
                    else:
                        velocity_pred_final = velocity_pred_text

                    # 转换维度 [B,C,T,H,W] -> [B,T,C,H,W]
                    velocity_pred_final = velocity_pred_final.permute(0, 2, 1, 3, 4)
                    # [1, 1, 16, 88, 128] : [B,T,C,H,W]
                    # 更新潜在向量
                    print(f"Before step - latents shape: {latents.shape}")
                    print(f"velocity_pred_final shape: {velocity_pred_final.shape}")
                    print(f"Current timestep t: {t}")

                    # 更新潜在向量
                    latents = self.noise_scheduler.step(velocity_pred_final, t, latents, return_dict=False)[0]
                    print(f"After step - latents shape: {latents.shape}")

                image = self.decode_image(latents)

        return image, intermediate_images

    def decode_image(self, latents):
        # correct now
        latents = latents.to(self.config.predict.devices['vae'], dtype=self.weight_dtype)
        latents_mean = (
            torch.tensor(self.vae_latent_mean, dtype=self.weight_dtype)
            .view(1, 1, self.vae_z_dim, 1, 1)
            .to(latents.device)
        )
        latents_std = (
            torch.tensor(self.vae_latent_std, dtype=self.weight_dtype)
            .view(1, 1, self.vae_z_dim, 1, 1)
            .to(latents.device)
        )
        # 逆转标准化：(latents / latents_std) + latents_mean
        latents = latents * latents_std + latents_mean
        # [1, 1, 16, 88, 128]
        # 转换回VAE期望的维度格式 [B, T, C, H, W] -> [B, C, T, H, W]
        latents = latents.permute(0, 2, 1, 3, 4)
        # [1, 16, 1, 88, 128]

        # 解码图像
        image = self.models["vae"].decode(latents).sample
        # [1, 3, 1, 704, 1024] after decode

        # 后处理：tensor -> numpy
        image = self._postprocess_image(image)
        return image

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理输入图像

        Args:
            image: numpy.ndarray RGB [H,W,C]

        Returns:
            torch.Tensor: 预处理后的图像张量 [3,H,W]
        """
        # 确保输入是正确的格式和范围
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # 转换为torch tensor [3,H,W]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        return image_tensor

    def _postprocess_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        后处理输出图像

        Args:
            image_tensor: torch.Tensor 解码后的图像张量

        Returns:
            numpy.ndarray: RGB格式图像 [H,W,C]
        """
        # 确保张量在CPU上
        image = image_tensor.cpu()
        image = image.float()

        # 移除批次维度并调整范围
        image = image.squeeze(2).squeeze(0)  # [C,H,W]
        image = (image / 2 + 0.5).clamp(0, 1)  # 从[-1,1]转换到[0,1]

        # 转换为numpy并调整维度顺序
        image = image.permute(1, 2, 0).numpy()  # [H,W,C]

        # 转换为uint8
        image = (image * 255).astype(np.uint8)

        return image


if __name__ == "__main__":
    config_file = "configs/qwen_image_edit_config.yaml"
    from src.data.config import load_config_from_yaml

    config = load_config_from_yaml(config_file)
    trainer = Trainer(config)
