"""
Validation mixin for trainers.
Provides functionality for validation sampling during training.
"""

import logging
import os
from typing import Any

import numpy as np
import PIL
import torch
from accelerate import Accelerator
from diffusers.utils import load_image
from torch.utils.data import Dataset
from tqdm import tqdm

from qflux.data.config import Config, ValidationSample
from qflux.utils.tools import instantiate_class, pad_to_max_shape


logger = logging.getLogger(__name__)


class ValidationMixin:
    """
    Mixin class that adds validation sampling functionality to trainers.
    """

    config: Config  # 由子类在 __init__ 里赋值
    accelerator: Accelerator  # 由子类在 __init__ 里赋值
    global_step: int  # 由子类中定义

    def setup_validation(self, train_dataset: Dataset):
        """
        Setup validation sampling if enabled in config.
        Called during trainer initialization.
        """

        # Skip if no validation config
        if not self.config.validation.enabled:
            # if not hasattr(self.config, "validation") or not self.config.validation.get("enabled", False):
            logger.info("Validation sampling is disabled")
            return
        logger.info("Setting up validation sampling")
        # Store validation config
        self.train_dataset = train_dataset
        self.validation_config = self.config.validation
        self.validation_steps = self.validation_config.steps
        self.validation_seed = self.validation_config.seed
        # Load validation samples
        validation_samples = self._load_validation_samples()
        if not validation_samples:
            logger.warning("No validation samples loaded")
            return
        # Validate that all samples have the same shape for DDP compatibility
        self._validate_samples_shape_consistency(validation_samples)
        # Store raw samples first - we'll prepare them after accelerator is initialized
        self.validation_samples = validation_samples
        logger.info(f"Loaded {len(self.validation_samples)} validation samples for periodic sampling")

        self.validation_embeddings: list[dict[str, Any]] = []
        self.prepare_validation_embeddings()
        self.accelerator.wait_for_everyone()
        self.reload_embeddings()
        logger.info(f"Finished setup validation with number of embeddings {len(self.validation_embeddings)}")

    def _load_validation_samples(self) -> list[dict[str, Any]]:
        """
        Load validation samples based on configuration.

        Returns:
            List of validation samples
        """
        # Load from direct samples list
        if hasattr(self.validation_config, "samples") and self.validation_config.samples is not None:
            logger.info(f"Loading validation datasets from {len(self.validation_config.samples)} configured samples")
            return self._load_from_config_samples(self.validation_config.samples)

        # Load from dataset configuration
        if hasattr(self.validation_config, "dataset") and self.validation_config.dataset is not None:
            logger.info(f"Loading validation datasets from dataset configuration: {self.validation_config.dataset}")
            dataset_config = self.validation_config.dataset
            class_path = dataset_config.class_path
            init_args = dataset_config.init_args
            init_args.use_cache = False
            init_args.cache_dir = ""
            dataset = instantiate_class(class_path, init_args)
            validation_samples: list[dict[str, Any]] = []
            for i in range(self.validation_config.max_samples):
                data = dataset[i]
                prompt = data["prompt"]
                control = data["control"]  # [B,C,H,W]
                control = self.tensor2pil(control)
                images = [control]
                controls_size = [[control.height, control.width]]
                n_controls = data["n_controls"]

                for j in range(n_controls):
                    control = data[f"control_{j + 1}"]
                    control = self.tensor2pil(control)
                    images.append(control)
                    controls_size.append([control.height, control.width])
                processor_init_args = dataset_config.init_args.processor.init_args
                target_size = processor_init_args.target_size
                height = target_size[0] if target_size is not None else None
                width = target_size[1] if target_size is not None else None
                validation_samples.append(
                    {
                        "prompt": prompt,
                        "images": images,
                        "controls_size": controls_size,
                        "height": height,
                        "width": width,
                    }
                )
            return validation_samples
        logger.warning("No valid validation dataset configuration found")
        return []

    def _load_from_config_samples(
        self,
        samples: list[ValidationSample],
    ) -> list[dict[str, Any]]:
        """
        Load validation samples from direct configuration.
        """
        logger.info(f"Loading {len(samples)} validation samples from config")
        validation_samples = []

        for sample in samples:
            images_ = sample.images
            images = [load_image(image).convert("RGB") for image in images_]
            if hasattr(sample, "controls_size"):
                controls_size = sample.controls_size
            else:
                controls_size = [[img.height, img.width] for img in images]
            height = sample.height if hasattr(sample, "height") else images[0].height
            if hasattr(sample, "width"):
                width = sample.width
            else:
                width = images[0].width
            validation_sample = {
                "prompt": sample.prompt,
                "images": images,
                "controls_size": controls_size,
                "height": height,
                "width": width,
            }
            validation_samples.append(validation_sample)

        return validation_samples

    def _validate_samples_shape_consistency(self, validation_samples: list[dict[str, Any]]) -> None:
        """
        Validate that all validation samples have consistent shapes for DDP compatibility.

        In DDP mode, accelerator.gather() requires all processes to have tensors with
        identical shapes. This means all validation samples must have the same height
        and width to avoid deadlock during the gather operation.

        Args:
            validation_samples: List of validation sample dictionaries

        Raises:
            ValueError: If samples have inconsistent shapes
        """
        if len(validation_samples) <= 1:
            return

        # Extract shapes from all samples
        shapes = []
        for idx, sample in enumerate(validation_samples):
            height = sample.get("height")
            width = sample.get("width")
            shapes.append((height, width, idx))

        # Check if all shapes are identical
        first_shape = shapes[0][:2]
        inconsistent_samples = []

        for height, width, idx in shapes[1:]:
            if (height, width) != first_shape:
                inconsistent_samples.append((idx, height, width))

        # If inconsistent shapes found, raise detailed error
        if inconsistent_samples:
            error_msg = (
                f"Validation samples have inconsistent shapes which will cause DDP deadlock!\n"
                f"Sample 0 has shape: height={first_shape[0]}, width={first_shape[1]}\n"
                f"But the following samples have different shapes:\n"
            )
            for idx, height, width in inconsistent_samples:
                error_msg += f"  - Sample {idx}: height={height}, width={width}\n"
            error_msg += (
                "\nFor DDP training, all validation samples MUST have the same height and width.\n"
                "Please update your validation config to use consistent dimensions across all samples."
            )
            raise ValueError(error_msg)

        logger.info(
            f"Validation shape consistency check passed: "
            f"all {len(validation_samples)} samples have shape (height={first_shape[0]}, width={first_shape[1]})"
        )

    def tensor2pil(self, tensor: torch.Tensor) -> PIL.Image.Image:
        """
        Convert a tensor to a PIL image.
        Input: C,H,W, range [0,1]
        """
        img = tensor.float().cpu().numpy()  # C,H,W
        img = img.transpose(1, 2, 0)
        img = (img * 255).round().astype("uint8")
        img_ = PIL.Image.fromarray(img)
        return img_

    def prepare_validation_embeddings(self):
        """
        Prepare validation embeddings after accelerator is initialized.
        This should be called after accelerator.prepare() to distribute samples across GPUs.
        """
        if not self.accelerator.is_main_process:
            return
        # only do this on main process
        device = self.accelerator.device
        if hasattr(self, "vae"):
            self.vae.to(device)
            self.vae.requires_grad_(False).eval()
        if hasattr(self, "text_encoder"):
            self.text_encoder.to(device)
        if hasattr(self, "text_encoder_2"):
            self.text_encoder_2.to(device)

        logging.info(f"### Preparing validation embeddings with {len(self.validation_samples)} samples")
        if not self.validation_samples:
            return

        if len(self.validation_embeddings) > 0:
            return  # Already prepared

        # Prepare validation embeddings for this process's samples
        has_prompt_enhancer = False
        if hasattr(self, "use_vlm_prompt_enhancer") and hasattr(self, "vlm_model"):
            if self.use_vlm_prompt_enhancer and self.vlm_model is not None:
                has_prompt_enhancer = True
        if has_prompt_enhancer:
            origin_device = self.vlm_model.device
            self.vlm_model.to(device)
        print("has_prompt_enhancer", has_prompt_enhancer)
        for i, sample in tqdm(enumerate(self.validation_samples), desc="Preparing validation samples"):
            # Process sample to match predict_batch format
            batch = self._prepare_validation_sample(sample)
            # Get embeddings

            embeddings = self.prepare_embeddings(batch, stage="predict")

            # Move embeddings to CPU to save GPU memory
            cpu_embeddings = {}
            for k, v in embeddings.items():
                if isinstance(v, torch.Tensor):
                    cpu_embeddings[k] = v.cpu()
                else:
                    cpu_embeddings[k] = v
            # save to shared folder

            save_dir = f"/tmp/validation_embeddings/{self.config.logging.tracker_project_name}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(cpu_embeddings, os.path.join(save_dir, f"{i}.pth"))
        if has_prompt_enhancer:
            self.vlm_model.to(origin_device)
            print("vlm_model device after enhance ", self.vlm_model.device)
        # unload vae and text encoders
        if hasattr(self, "vae"):
            self.vae.to("cpu")
        if hasattr(self, "text_encoder"):
            self.text_encoder.to("cpu")
        if hasattr(self, "text_encoder_2"):
            self.text_encoder_2.to("cpu")
        # reload embeddings for this rank

    def reload_embeddings(self):
        self.validation_embeddings = []
        num_process = self.accelerator.num_processes
        local_rank = self.accelerator.process_index
        num_per_rank = max(1, len(self.validation_samples) // num_process)
        save_dir = f"/tmp/validation_embeddings/{self.config.logging.tracker_project_name}"
        for i in range(num_per_rank):
            idx = (i * num_process + local_rank) % len(self.validation_samples)
            embedding_path = os.path.join(save_dir, f"{idx}.pth")
            cpu_embeddings = torch.load(embedding_path, map_location="cpu", weights_only=False)
            cpu_embeddings["idx"] = torch.Tensor([idx])  # use to get the original idx
            assert isinstance(cpu_embeddings, dict), "Embedding should be a dictionary"
            self.validation_embeddings.append(cpu_embeddings)
        logger.info(f"Process {local_rank} prepared {len(self.validation_embeddings)} validation samples")

    def _prepare_validation_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a validation dataset sample to the format expected by prepare_embeddings.
        """
        # Load images
        images = sample["images"]
        controls_size = sample["controls_size"]
        # Determine output dimensions if not provided
        height = sample.get("height", images[0].height)
        width = sample.get("width", images[0].width)

        # Prepare batch using trainer's prepare_predict_batch_data method
        batch = self.prepare_predict_batch_data(  # type: ignore
            image=images,
            prompt=sample["prompt"],
            controls_size=controls_size,
            height=height,
            width=width,
            negative_prompt=sample.get("negative_prompt"),
            num_inference_steps=sample.get("num_inference_steps", 20),
            true_cfg_scale=sample.get("true_cfg_scale", 1.0),
        )

        return batch

    def should_run_validation(self, global_step: int) -> bool:
        """
        Determine if validation should run at the current step.
        """
        if not hasattr(self, "validation_samples") or not self.validation_samples:
            return False
        # Ensure embeddings are prepared after accelerator is initialized
        if not self.validation_embeddings:
            return False
        return global_step % self.validation_steps == 0

    def run_validation(self):
        """
        Run validation sampling and log results to TensorBoard.
        Called periodically during training.
        """

        if not hasattr(self, "validation_embeddings") or not self.validation_embeddings:
            return
        self.accelerator.wait_for_everyone()
        logger.info(f"Running validation sampling at step {self.global_step}")

        # run sampling for each validation sample parallely
        lat_chunks = [] if self.accelerator.is_main_process else None
        idx_chunks = [] if self.accelerator.is_main_process else None

        for _, cpu_embeddings in enumerate(self.validation_embeddings):
            # Move embeddings to current device
            embeddings = {}
            for k, v in cpu_embeddings.items():
                if isinstance(v, torch.Tensor):
                    embeddings[k] = v.to(self.accelerator.device)
                else:
                    embeddings[k] = v
            with torch.inference_mode():
                latents = self.sampling_from_embeddings(embeddings)
            idx = torch.as_tensor(embeddings["idx"], device=self.accelerator.device)
            # --- gather（按 rank 顺序拼接到 dim=0）---
            g_lat = self.accelerator.gather(latents.contiguous())  # [W*B, s, d]
            g_idx = self.accelerator.gather(idx.contiguous())  # [W*B, 2]
            logging.info(f"[{self.accelerator.process_index}] gather latents and idx done")
            if self.accelerator.is_main_process:
                logging.info(f"[{self.accelerator.process_index}] main process append latents and idx")
                lat_chunks.append(g_lat.detach().cpu())
                idx_chunks.append(g_idx.cpu())
                logging.info(f"[{self.accelerator.process_index}] main process append latents and idx done")

        # torch.cuda.synchronize()
        logging.info(f"[{self.accelerator.process_index}] next step decode by vae, wait for everyone")
        self.accelerator.wait_for_everyone()
        logging.info(f"[{self.accelerator.process_index}] next step decode by vae")

        # rank0 合并
        if self.accelerator.is_main_process:
            all_latents = torch.cat(lat_chunks, dim=0)  # [T*W*B, s, d]
            idxes = torch.cat(idx_chunks, dim=0)  # [T*W*B, 1]
            # device vae in rank-0
            self.vae.to(self.accelerator.device)
            batch_size = all_latents.shape[0]
            log_images = []
            log_validation_samples = []
            for jj in range(batch_size):
                latents = all_latents[jj : jj + 1]
                idx = int(idxes[jj].item())
                validation_sample = self.validation_samples[idx]
                images = self.decode_vae_latent(latents, validation_sample["height"], validation_sample["width"])
                images = images.float().detach().cpu()
                log_images.append(images)
                log_validation_samples.append(validation_sample)
            self._log_validation_images(log_images, log_validation_samples)
            logging.info(f"[{self.accelerator.process_index}] validation images logged successfully")
            logging.info(f"[{self.accelerator.process_index}] unload vae to cpu")
        if self.config.cache.use_cache:
            # in cache model, can unload vae to cpu
            self.vae.to("cpu")
            torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        logging.info(f"[{self.accelerator.process_index}] rank {self.accelerator.process_index} finished validation")
        # self.accelerator.wait_for_everyone()

    def _log_validation_images(self, log_images: list[torch.Tensor], validation_samples: list[dict[str, Any]]):
        """
        Log validation images and metadata using LoggerManager.
        """
        # Check if logger_manager is available
        if not hasattr(self, "logger_manager") or self.logger_manager is None:
            logger.warning("LoggerManager not available, skipping validation logging")
            return

        def process_image(img):
            if img.ndim == 4:
                img = img[0]  # If batched, take first image
            if img.ndim == 3 and img.shape[0] == 3:  # Already in [C, H, W] format
                pass
            elif img.ndim == 3:  # Possibly in [H, W, C] format
                img = img.permute(2, 0, 1)

            # Ensure on CPU and in correct value range
            img = img.cpu()
            if img.max() > 1.0:
                img = img / 255.0
            return img

        # Get prompt for display
        log_images = [process_image(img) for img in log_images]
        log_generate_tensor = pad_to_max_shape(log_images)
        # [B, C, H, W]

        prompt = validation_samples[0]["prompt"]

        # Log control images if available
        n_controls = len(validation_samples[0]["images"])

        def process_control_image(img):
            if isinstance(img, PIL.Image.Image):
                img_array = np.array(img)
            else:
                img_array = img
            if len(img_array.shape) == 3:
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            else:
                img_tensor = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0
            return img_tensor

        for i in range(n_controls):
            control_images = [validation_sample["images"][i] for validation_sample in validation_samples]
            control_images = [process_control_image(img) for img in control_images]
            log_image_tensor = pad_to_max_shape(control_images)
            logger.debug(f"Control {i} image tensor shape: {log_image_tensor.shape}")

            # 使用LoggerManager记录控制图像
            self.logger_manager.log_images(
                f"validation/control_{i}",
                log_image_tensor,
                step=self.global_step,
                caption=f"Control {i}",
                commit=False,
            )

        self.logger_manager.log_images(
            "validation/generated_images",
            log_generate_tensor,
            step=self.global_step,
            caption="Generated images",
            commit=True,
        )

        # Log prompt text using LoggerManager
        self.logger_manager.log_text(
            "validation/prompt",
            prompt,
            step=self.global_step,
        )

        # Explicitly flush logger to ensure all operations complete
        logging.info(f"[{self.accelerator.process_index}] flushing logger after validation")
        self.logger_manager.flush()
        logging.info(f"[{self.accelerator.process_index}] logger flushed successfully")
