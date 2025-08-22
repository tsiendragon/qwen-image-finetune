import copy
import os
import shutil
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import math
import numpy as np
from src.models.load_model import load_transformer, load_vae, load_qwenvl
from src.utils.logger import get_logger
from src.data.dataset import loader
from src.data.cache_manager import EmbeddingCacheManager


logger = get_logger(__name__, log_level="INFO")

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None

class Trainer:
    def __init__(self, config):
        self.config = config
        self.accelerator = None
        self.models = {}
        self.optimizer = None
        self.lr_scheduler = None
        self.noise_scheduler = None
        self.global_step = 0
        self.setup()
        self.setup_models()
        self.configure_optimizers()

    def setup(self):
        """初始化加速器和日志配置"""
        logging_dir = os.path.join(self.config.logging.output_dir, self.config.logging.logging_dir)
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.config.logging.output_dir,
            logging_dir=logging_dir
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.train.gradient_accumulation_steps,
            mixed_precision=self.config.train.mixed_precision,
            log_with=self.config.logging.report_to,
            project_config=accelerator_project_config,
        )

        # 设置权重数据类型
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # 创建输出目录
        if self.accelerator.is_main_process and self.config.logging.output_dir is not None:
            os.makedirs(self.config.logging.output_dir, exist_ok=True)

        logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")

    def setup_models(self):
        """加载和配置模型"""
        # 加载模型
        self.models['text_encoder'] = load_qwenvl(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype
        )
        self.models['vae'] = load_vae(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype
        )
        self.models['transformer'] = load_transformer(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype
        )

        # 配置 LoRA
        lora_config = LoraConfig(
            r=self.config.model.lora.r,
            lora_alpha=self.config.model.lora.lora_alpha,
            init_lora_weights=self.config.model.lora.init_lora_weights,
            target_modules=self.config.model.lora.target_modules,
        )

        # 设置调度器
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            subfolder="scheduler",
        )

        # 配置模型
        transformer = self.models['transformer']
        transformer.to(self.accelerator.device, dtype=self.weight_dtype)
        transformer.add_adapter(lora_config)
        transformer.train()
        transformer.enable_gradient_checkpointing()

        self.models['text_encoder'].to(self.accelerator.device)
        self.models['vae'].to(self.accelerator.device, dtype=self.weight_dtype)

        # 冻结非 LoRA 参数
        self.models['vae'].requires_grad_(False)
        transformer.requires_grad_(False)

        # 只训练 LoRA 参数
        trainable_params = 0
        for name, param in transformer.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
        self.vae_scale_factor = 2 ** len(self.models['vae'].temperal_downsample)

        logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

        # 为推理优化模型
        self._optimize_for_inference()

    def _optimize_for_inference(self):
        """优化模型以提升推理性能"""
        # 设置 VAE 为评估模式并启用推理优化

        # 为所有模型启用 bf16 推理优化
        if self.weight_dtype == torch.bfloat16:
            # 确保所有非训练模型使用 bf16
            for model_name, model in self.models.items():
                if model_name != 'transformer':  # transformer 在训练时需要特殊处理
                    model.eval()
                    model.to(dtype=self.weight_dtype)

        # 启用推理优化标志
        torch.backends.cudnn.benchmark = True  # 对于固定输入尺寸的情况，可以优化卷积操作

        logger.info(f"模型推理优化完成，使用精度: {self.weight_dtype}")

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        lora_layers = filter(lambda p: p.requires_grad, self.models['transformer'].parameters())

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
            num_warmup_steps=self.config.lr_scheduler.warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.config.train.max_train_steps * self.accelerator.num_processes,
        )

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        """计算噪声调度器的 sigma 值"""
        noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        sigmas = noise_scheduler_copy.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def encode_image(self, image):
        # 使用 inference_mode() 获得更好的推理性能
        with torch.inference_mode():
            # 使用 autocast 确保 bf16 推理
            with torch.autocast(device_type=self.accelerator.device.type, dtype=self.weight_dtype, enabled=True):
                pixel_values = image.to(dtype=self.weight_dtype, device=self.accelerator.device, non_blocking=True)
                pixel_values = pixel_values.unsqueeze(2)
                pixel_latents = self.models['vae'].encode(pixel_values).latent_dist.sample()
                pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)
        return pixel_latents

    def encode_prompt(self, prompt:str, prompt_image: np.ndarray)->tuple[torch.Tensor, torch.Tensor]:
        text_encoding_pipeline = self.models['text_encoder']
        # 使用 inference_mode() 获得更好的推理性能
        with torch.inference_mode():
            # 使用 autocast 确保 bf16 推理
            with torch.autocast(device_type=self.accelerator.device.type, dtype=self.weight_dtype, enabled=True):
                calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, prompt_image.shape[0] / prompt_image.shape[1])
                prompt_image = text_encoding_pipeline.image_processor.resize(prompt_image, calculated_height, calculated_width)
                prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                        image=prompt_image,
                        prompt=[prompt],
                        device=text_encoding_pipeline.device,
                        num_images_per_prompt=1,
                        max_sequence_length=1024,
                    )
        return prompt_embeds, prompt_embeds_mask

    def training_step(self, batch):
        """执行单个训练步骤"""
        # 检查是否有缓存数据
        if batch.get('cached', False):
            return self._training_step_cached(batch)
        else:
            return self._training_step_compute(batch)

    def _training_step_cached(self, batch):
        """使用缓存嵌入的训练步骤"""
        # 从缓存数据中获取嵌入
        pixel_latents = batch['pixel_latent'].to(self.accelerator.device, dtype=self.weight_dtype)
        control_latents = batch['control_latent'].to(self.accelerator.device, dtype=self.weight_dtype)
        prompt_embeds = batch['prompt_embed'].to(self.accelerator.device)
        prompt_embeds_mask = batch['prompt_embeds_mask'].to(self.accelerator.device)

        return self._compute_loss(pixel_latents, control_latents, prompt_embeds, prompt_embeds_mask)

    def _training_step_compute(self, batch):
        """计算嵌入的训练步骤（无缓存）"""
        image, control, prompt = batch['image'], batch['control'], batch['prompt']
        pixel_latents = self.encode_image(image)
        control_latents = self.encode_image(control)
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt, control)
        # 使用 inference_mode() 进行编码推理，性能更好
        with torch.inference_mode():
            # 编码图像
            # 标准化潜在向量
            vae_config = self.models['vae'].config
            latents_mean = torch.tensor(vae_config.latents_mean, dtype=self.weight_dtype).view(1, 1, vae_config.z_dim, 1, 1).to(
                pixel_latents.device, non_blocking=True
            )
            latents_std = (1.0 / torch.tensor(vae_config.latents_std, dtype=self.weight_dtype)).view(1, 1, vae_config.z_dim, 1, 1).to(
                pixel_latents.device, non_blocking=True
            )
            pixel_latents = (pixel_latents - latents_mean) * latents_std
            control_latents = (control_latents - latents_mean) * latents_std

        return self._compute_loss(pixel_latents, control_latents, prompt_embeds, prompt_embeds_mask)

    def _compute_loss(self, pixel_latents, control_latents, prompt_embeds, prompt_embeds_mask):
        """计算损失的通用方法"""
        with torch.no_grad():
            # 添加噪声
            bsz = pixel_latents.shape[0]
            noise = torch.randn_like(pixel_latents, device=self.accelerator.device, dtype=self.weight_dtype)

            # 采样时间步
            u = compute_density_for_timestep_sampling(
                weighting_scheme="none",
                batch_size=bsz,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler.timesteps[indices].to(device=pixel_latents.device)

            sigmas = self.get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
            noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

            # 打包潜在向量
            packed_noisy_model_input = QwenImagePipeline._pack_latents(
                noisy_model_input, bsz,
                noisy_model_input.shape[2],
                noisy_model_input.shape[3],
                noisy_model_input.shape[4]
            )

            packed_control_latents = QwenImagePipeline._pack_latents(
                control_latents, bsz,
                control_latents.shape[2],
                control_latents.shape[3],
                control_latents.shape[4]
            )

            # 编码文本
            img_shapes = [[(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2),
                              (1, control_latents.shape[3] // 2, control_latents.shape[4] // 2)]] * bsz
            packed_noisy_model_input_concated = torch.cat([packed_noisy_model_input, packed_control_latents], dim=1)
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

            prompt_embeds = prompt_embeds.repeat(bsz, 1, 1)

        # 前向传播
        model_pred = self.models['transformer'](
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
        weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
        target = noise - pixel_latents
        target = target.permute(0, 2, 1, 3, 4)

        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()

        return loss

    def save_checkpoint(self, global_step):
        """保存检查点"""
        if not self.accelerator.is_main_process:
            return

        # 管理检查点数量
        if self.config.train.checkpoints_total_limit is not None:
            checkpoints = os.listdir(self.config.logging.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            if len(checkpoints) >= self.config.train.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - self.config.train.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(self.config.logging.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(self.config.logging.output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)

        # 保存 LoRA 权重
        unwrapped_transformer = self.accelerator.unwrap_model(self.models['transformer'])
        if is_compiled_module(unwrapped_transformer):
            unwrapped_transformer = unwrapped_transformer._orig_mod

        lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_transformer)
        )

        QwenImagePipeline.save_lora_weights(save_path, lora_state_dict, safe_serialization=True)
        logger.info(f"Saved checkpoint to {save_path}")

    def fit(self, train_dataloader):
        """主训练循环"""
        # 准备模型和优化器
        transformer, optimizer, _, lr_scheduler = self.accelerator.prepare(
            self.models['transformer'], self.optimizer, train_dataloader, self.lr_scheduler
        )
        self.models['transformer'] = transformer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # 初始化追踪器
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(self.config.logging.tracker_project_name, {"test": None})

        # 训练信息
        total_batch_size = (
            self.config.train.train_batch_size *
            self.accelerator.num_processes *
            self.config.train.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {self.config.train.train_batch_size}")
        logger.info(f"  Total train batch size = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.train.gradient_accumulation_steps}")

        # 进度条
        progress_bar = tqdm(
            range(0, self.config.train.max_train_steps),
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )

        # 训练循环
        train_loss = 0.0
        for epoch in range(1):
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.models['transformer']):
                    loss = self.training_step(batch)

                    # 反向传播
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.models['transformer'].parameters(),
                            self.config.train.max_grad_norm
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
                    train_loss += avg_loss.item() / self.config.train.gradient_accumulation_steps

                    # 记录日志
                    self.accelerator.log({"train_loss": train_loss}, step=self.global_step)
                    train_loss = 0.0

                    # 保存检查点
                    if self.global_step % self.config.train.checkpointing_steps == 0:
                        self.save_checkpoint(self.global_step)

                # 更新进度条
                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": self.lr_scheduler.get_last_lr()[0]
                }
                progress_bar.set_postfix(**logs)

                # 检查是否达到最大步数
                if self.global_step >= self.config.train.max_train_steps:
                    break

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()


if __name__ == "__main__":
    config_file = 'configs/qwen_image_edit_config.yaml'
    from src.data.config import load_config_from_yaml
    config = load_config_from_yaml(config_file)
    trainer = Trainer(config)
    trainer.setup()
    trainer.setup_models()
