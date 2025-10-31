import copy
import logging
from typing import Any

import PIL
import torch
import torch.nn.functional as F  # NOQA
from diffusers import QwenImageEditPlusPipeline
from torch._tensor import Tensor

from qflux.models.load_model import load_qwenvl, load_transformer, load_vae
from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer
from qflux.utils.images import calculate_best_resolution, make_image_shape_devisible


class QwenImageEditPlusTrainer(QwenImageEditTrainer):
    """Trainer class based on QwenImageEditPlusTrainer
    Inherits from QwenImageEditTrainer
    Adjust the multi images support
    The image process logics are in the QwenImageEditPipeline
    1. each image is process with a copy of vae images and condition images
    """

    def __init__(self, config):
        super().__init__(config)

    def get_pipeline_class(self):
        return QwenImageEditPlusPipeline

    # Static methods: directly reference QwenImageEditPipeline methods
    _pack_latents = staticmethod(QwenImageEditPlusPipeline._pack_latents)
    _unpack_latents = staticmethod(QwenImageEditPlusPipeline._unpack_latents)

    def load_model(self, text_encoder_device=None):
        """Load and separate components from QwenImageEditPipeline"""
        logging.info("Loading QwenImageEditPipeline and separating components...")

        # Load complete model using pipeline
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            torch_dtype=self.weight_dtype,
            transformer=None,
            vae=None,
        )
        pipe.to("cpu")
        logging.info(f"excution device: {pipe._execution_device}")

        # Separate individual components

        self.vae = load_vae("Qwen/Qwen-Image-Edit-2509", weight_dtype=self.weight_dtype)  # use original one
        # same to model constructed from vae self.vae = pipe.vae
        self.text_encoder = load_qwenvl("Qwen/Qwen-Image-Edit-2509", weight_dtype=self.weight_dtype)  # use original one
        logging.info(f"text_encoder device: {self.text_encoder.device}")
        # self.dit = pipe.transformer this is same as the following, verified

        self.dit = load_transformer(
            self.config.model.pretrained_model_name_or_path,  # could use quantized version
            weight_dtype=self.weight_dtype,
        )
        # load_transformer is same as pipe.transformer

        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
        from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

        self.processor: Qwen2VLProcessor = pipe.processor
        self.tokenizer: Qwen2Tokenizer = pipe.tokenizer
        self.scheduler: FlowMatchEulerDiscreteScheduler = pipe.scheduler
        self.sampling_scheduler: FlowMatchEulerDiscreteScheduler = copy.deepcopy(
            self.scheduler
        )  # Independent scheduler for validation/sampling
        # Initialize image processor (for predict method)
        # Initialize image processor (for predict method)
        from diffusers.image_processor import VaeImageProcessor

        # Set VAE-related parameters
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        self.vae_latent_mean = self.vae.config.latents_mean
        self.vae_latent_std = self.vae.config.latents_std
        self.vae_z_dim = self.vae.config.z_dim

        # Attributes copied from original pipeline
        self.latent_channels = self.vae.config.z_dim
        self._guidance_scale = 1.0
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False
        self.prompt_template_encode = pipe.prompt_template_encode
        self.prompt_template_encode_start_idx = pipe.prompt_template_encode_start_idx

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.num_channels_latents = self.dit.config.in_channels // 4

        # Set models to training/evaluation mode
        self.text_encoder.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.dit.requires_grad_(False).eval()
        torch.cuda.empty_cache()

        logging.info(f"Components loaded successfully. VAE scale factor: {self.vae_scale_factor}")

    def process_condition_image(self, condition_image: torch.Tensor) -> PIL.Image.Image:
        """Process condition image  [1,C,H,W] to PIL.Image.Image"""
        condition_image = self.preprocess_image_for_text_encoder(condition_image)  # to [0,255] range
        condition_image = condition_image[0].permute(1, 2, 0).float().cpu().numpy().astype("uint8")
        condition_image = PIL.Image.fromarray(condition_image)
        best_w, best_h = calculate_best_resolution(condition_image.size[0], condition_image.size[1], 384 * 384)
        condition_image = condition_image.resize((best_w, best_h), resample=PIL.Image.LANCZOS)
        return condition_image

    def prepare_embeddings(self, batch, stage="fit", debug=False):
        """
        used in: _training_step_compute, cache, and predict
        for cache: use prepare_cached_embeddings
        Qwen-Edit
            - image_latent : target used for training
            - control_latent: concatenated version
            - height_image, width_image
            - height_control, width_control: no batch dim
            - height_control_1, width_control_1: no batch dim
            ...
        for control image, need to create two copies:
            1. processed for vae encoder
            2. processed for text encoder
        for additional control images, only pass to vae encoder
        predict mode:

            - no `image` key
            - extra: negative_prompt_embeds, negative_prompt_embeds_mask

        cache mode:
            - extra: empty_prompt_embeds, empty_prompt_embeds_mask
        """

        if "image" in batch:
            batch["image"] = self.preprocess_image_for_vae_encoder(batch["image"])  # B,3,1,H,W
            batch["width"] = batch["image"].shape[4]
            batch["height"] = batch["image"].shape[3]

        condition_images = []
        if "control" in batch:
            control_image = batch["control"]
            # prompt_control = self.preprocess_image_for_text_encoder(
            condition_images.append(self.process_condition_image(control_image))  # remove batch dim
            batch["control"] = self.preprocess_image_for_vae_encoder(
                batch["control"]
            )  # [B,C,1,H,W], float, range [-1,1]
            batch["width_control"] = batch["control"].shape[4]
            batch["height_control"] = batch["control"].shape[3]

        if isinstance(batch["n_controls"], int):
            num_additional_controls = batch["n_controls"]
        else:
            num_additional_controls = batch["n_controls"][0]

        for i in range(num_additional_controls):
            additional_control_key = f"control_{i + 1}"
            if additional_control_key in batch:
                # condition_images =
                additional_control_image = batch[additional_control_key]
                # [B,C,H,W], uint8, range [0,255]
                condition_images.append(self.process_condition_image(additional_control_image))
                batch[additional_control_key] = self.preprocess_image_for_vae_encoder(batch[additional_control_key])
                batch[f"width_control_{i + 1}"] = batch[additional_control_key].shape[4]
                batch[f"height_control_{i + 1}"] = batch[additional_control_key].shape[3]
        if "condition_images" in batch:
            """more precise processing compared with original Qwen-Image-Edit-Plus"""
            processed = []
            for condition_image in batch["condition_images"]:
                best_w, best_h = calculate_best_resolution(condition_image.size[0], condition_image.size[1], 384 * 384)
                condition_image = condition_image.resize((best_w, best_h), resample=PIL.Image.LANCZOS)
                processed.append(condition_image)
            batch["condition_images"] = processed
        else:
            batch["condition_images"] = condition_images
        # condition images resize is not identical to the original qwen-edit-plus. Here it resized two times
        # since we are using the shared images in dataset processor
        if debug:
            prompt_embeds, prompt_embeds_mask, model_inputs, hidden_states = self.encode_prompt(
                prompt=batch["prompt"], image=condition_images, debug=debug
            )
            batch["prompt_embeds_model_inputs"] = model_inputs
            batch["prompt_hidden_states"] = hidden_states

        else:
            prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt=batch["prompt"], image=condition_images)
        batch["prompt_embeds_mask"] = prompt_embeds_mask
        batch["prompt_embeds"] = prompt_embeds

        if stage == "cache":
            empty_prompt_embeds, empty_prompt_embeds_mask = self.encode_prompt(
                prompt=[""],
                image=condition_images,
            )
            batch["empty_prompt_embeds_mask"] = empty_prompt_embeds_mask
            batch["empty_prompt_embeds"] = empty_prompt_embeds

        if "negative_prompt" in batch and batch["true_cfg_scale"] > 1:
            # only for predict stage
            negative_prompt_embeds, negative_prompt_embeds_mask, model_inputs, hidden_states = self.encode_prompt(
                prompt=batch["negative_prompt"], image=condition_images, debug=debug
            )
            batch["negative_prompt_embeds_model_inputs"] = model_inputs
            batch["negative_prompt_embeds_mask"] = negative_prompt_embeds_mask
            batch["negative_prompt_embeds"] = negative_prompt_embeds
            batch["negative_prompt_hidden_states"] = hidden_states

        # get latents
        if "image" in batch:
            image = batch["image"]
            batch_size = image.shape[0]
            height_image, width_image = batch["height"], batch["width"]
            _, image_latents = self.prepare_latents(
                image,
                batch_size,
                self.num_channels_latents,
                height_image,
                width_image,
                self.weight_dtype,
            )
            batch["image_latents"] = image_latents

        if "control" in batch:
            control = batch["control"]
            batch_size = control.shape[0]
            height_control, width_control = batch["height_control"], batch["width_control"]
            _, control_latents = self.prepare_latents(
                control,
                batch_size,
                self.num_channels_latents,
                height_control,
                width_control,
                self.weight_dtype,
            )
            batch["control_latents"] = [control_latents]

        for i in range(1, num_additional_controls + 1):
            control_key = f"control_{i}"
            control = batch[control_key]
            batch_size = control.shape[0]
            height_control, width_control = batch[f"height_control_{i}"], batch[f"width_control_{i}"]
            _, control_latents = self.prepare_latents(
                control,
                batch_size,
                self.num_channels_latents,
                height_control,
                width_control,
                self.weight_dtype,
            )
            batch["control_latents"].append(control_latents)

        if "control_latents" in batch:
            batch["control_latents"] = torch.cat(batch["control_latents"], dim=1)

        # if self.config.loss.mask_loss and "mask" in batch:
        #     mask = batch["mask"]
        #     height_image, width_image = batch["height"], batch["width"]
        #     batch["mask"] = resize_bhw(mask, height_image, width_image)
        #     batch["mask"] = map_mask_to_latent(batch["mask"])
        img_shapes = batch["img_shapes"]
        img_shapes = self.convert_img_shapes_to_latent_space(img_shapes)
        batch["img_shapes"] = img_shapes

        return batch

    def encode_prompt(
        self,
        prompt: str | list[str],
        image: torch.Tensor | None = None,
        debug=False,
    ):
        r"""
        get the embedding of prompt and image via qwen_vl. Support batch inference. For batch inference,
        will pad to largest length of prompt in batch.
        It got grad by default, lets add the inference mode first

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            image (`torch.Tensor`, *optional*):
                image to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """
        device = self.text_encoder.device
        num_images_per_prompt = 1  # 固定为1，支持单图像生成
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        # if image and isinstance(image[0], torch.Tensor):
        #     # make sure it is identical with diffuser pipelines
        #     # when remove this, got relative error 0.49% in prompt embeds
        #     new_images = []
        #     for i in range(len(image)):
        #         img = image[i].permute(1, 2, 0).cpu().numpy().astype("uint8")
        #         img = PIL.Image.fromarray(img)
        #         new_images.append(img)
        #     image = new_images

        # with torch.inference_mode(): # check if inference mode affects the accuracy
        if debug:
            prompt_embeds, prompt_embeds_mask, model_inputs, hidden_states = self._get_qwen_prompt_embeds(
                prompt, image, device, debug=True
            )
        else:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, image, device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)
        if debug:
            return prompt_embeds, prompt_embeds_mask, model_inputs, hidden_states
        return prompt_embeds, prompt_embeds_mask

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str] | None = None,
        image: list[torch.Tensor] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        debug=False,
    ):
        assert prompt is not None, "prompt is required"
        prompt = [prompt] if isinstance(prompt, str) else prompt
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        if isinstance(image, list):
            base_img_prompt = ""
            for i, _ in enumerate[Tensor](image):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif image is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""

        template = self.prompt_template_encode

        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(base_img_prompt + e) for e in prompt]
        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)

        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        if debug:
            return prompt_embeds, encoder_attention_mask, model_inputs, hidden_states
        return prompt_embeds, encoder_attention_mask

    def prepare_predict_batch_data(
        self,
        image: PIL.Image.Image | list[PIL.Image.Image],
        prompt: str | list[str],
        controls_size: list[int] | None = None,  # first one is for main control
        height: int | None = None,
        width: int | None = None,
        negative_prompt: None | str | list[str] = None,
        guidance_scale: float = None,
        num_inference_steps: int = 20,
        true_cfg_scale: float = 4.0,
        image_latents: torch.Tensor = None,
        prompt_embeds: torch.Tensor = None,
        prompt_embeds_mask: torch.Tensor = None,
        use_native_size: bool = False,
        best_resolution_size: bool = False,
        weight_dtype=torch.bfloat16,
        **kwargs,
    ) -> dict:
        """Prepare predict batch data.
        prepare the data to batch dict that can be used to prepare embeddings similar in the training step.
        We want to reuse the same data preparation code in the training step.
        Processing logic:
            if height, width is None: choose the width from prompt_image
        if controls_size is None, use same processing as that in the processor in the config
        if use_native_size = True:
            the prompt image and additional controls will use their own size. That is they will not be resized.
        support for batch size 1
        image: List[PIL.Image.Image] if is a list, represents different control images
        """
        assert image is not None, "prompt_image is required"
        assert prompt is not None, "prompt is required"

        if not isinstance(image, list):
            image = [image]
        # image = [make_image_devisible(image, self.vae_scale_factor) for image in image]

        prompt_image = [image[0]]
        additional_controls = [image[1:]] if len(image) > 1 else None

        if isinstance(prompt, str):
            prompt = [prompt]

        self.weight_dtype = weight_dtype

        data: dict[str, Any] = {}
        control = []
        img_shapes = []
        data["condition_images"] = []
        for img in image:
            print("origin shape", img.size)
            data["condition_images"].append(img.copy())

        if use_native_size:
            controls_size_list: list[list[int]] = [[prompt_image[0].size[1], prompt_image[0].size[0]]]
            if additional_controls:
                controls_size_list.extend([[ctrl.size[1], ctrl.size[0]] for ctrl in additional_controls[0]])
            controls_size = controls_size_list  # type: ignore[assignment]

        if best_resolution_size and controls_size:
            controls_size = [
                list(calculate_best_resolution(c_size[0], c_size[1], 1024 * 1024))  # type: ignore
                for c_size in controls_size
            ]
            logging.info(f"controls_size after best resolution  {controls_size}")

        logging.info(f"controls_size for processing {controls_size}")

        for img in prompt_image:
            # for each image, need to make one copy for text_encoder, another for image_encoder
            # convert to [C,H,W] in range [0,1]
            img = self.preprocessor.preprocess({"control": img}, controls_size=controls_size)["control"]
            control.append(img)
        control = torch.stack(control, dim=0)
        data["control"] = control

        data["prompt"] = prompt
        data["height"] = height if height is not None else control.shape[2]
        data["width"] = width if width is not None else control.shape[3]
        img_shapes.append((3, data["height"], data["width"]))
        img_shapes.append((3, control.shape[2], control.shape[3]))

        if height is None or width is None:
            width, height = control.shape[2], control.shape[1]
        else:
            width, height = make_image_shape_devisible(width, height, self.vae_scale_factor)

        logging.info(f"target shape for generation {width}, {height}")

        if additional_controls:
            n_controls = len(additional_controls[0])
            new_controls: dict[str, list] = {f"control_{i + 1}": [] for i in range(n_controls)}
            # [control_1_batch1, control1_batch2, ..], [control2_batch1, control2_batch2, ..]

            for controls in additional_controls:
                controls = self.preprocessor.preprocess({"controls": controls}, controls_size=controls_size)["controls"]
                for i, control in enumerate(controls):
                    new_controls[f"control_{i + 1}"].append(control)
                    img_shapes.append((3, control.shape[1], control.shape[2]))

            for i in range(n_controls):
                control_stack = torch.stack(new_controls[f"control_{i + 1}"], dim=0)
                data[f"control_{i + 1}"] = control_stack
            data["n_controls"] = n_controls
        else:
            data["n_controls"] = 0

        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            assert len(negative_prompt) == len(data["prompt"]), (
                "the number of negative_prompt should be same of control"
            )  # NOQA
            data["negative_prompt"] = negative_prompt
        data["num_inference_steps"] = num_inference_steps
        data["true_cfg_scale"] = true_cfg_scale
        data["guidance"] = guidance_scale
        data["img_shapes"] = [img_shapes] * len(prompt_image)
        return data
