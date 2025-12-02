# FLUX.2 LoRA Trainer Implementation Plan

## Document Information
- **Created**: 2025-11-28
- **Author**: AI Assistant
- **Purpose**: Implementation plan for FLUX.2 LoRA finetune trainer supporting both text-to-image and image-to-image modes
- **Status**: Planning Phase

---

## 1. Executive Summary

### 1.1 Objective

Implement a `Flux2LoraTrainer` that supports FLUX.2 model fine-tuning with LoRA, handling both:
- **Text-to-Image (T2I)**: Pure text-conditioned generation without reference images
- **Image-to-Image (I2I)**: Generation with optional condition/reference image

The trainer must support all three operational modes: `fit`, `cache`, and `predict`.

### 1.2 Key Requirements

| Requirement | Description |
|------------|-------------|
| **Dual Mode Support** | Both T2I (no condition image) and I2I (with condition image) |
| **BaseTrainer Compatibility** | Inherit from BaseTrainer, implement all abstract methods |
| **Stage Support** | `fit`, `cache`, `predict` stages for both modes |
| **FP16/FP4 Quantization** | Support both FP16 (`black-forest-labs/FLUX.2-dev`) and FP4 (`diffusers/FLUX.2-dev-bnb-4bit`) |

### 1.3 Model Sources

| Precision | Model Path | VRAM Estimate |
|-----------|-----------|---------------|
| FP16/BF16 | `black-forest-labs/FLUX.2-dev` | ~50GB |
| FP4/INT4 | `diffusers/FLUX.2-dev-bnb-4bit` | ~24GB |

---

## 2. Architecture Analysis

### 2.1 FLUX.2 vs FluxKontext Comparison

| Component | FluxKontext | FLUX.2 |
|-----------|-------------|--------|
| **Text Encoder** | CLIP + T5 (dual encoder) | Mistral3 (single VLM) |
| **Text Embeds** | `pooled_prompt_embeds` + `prompt_embeds` | `prompt_embeds` only |
| **Text IDs** | `txt_ids` [seq, 3] | `txt_ids` [seq, 4] |
| **VAE** | `AutoencoderKL` | `AutoencoderKLFlux2` (with batch_norm) |
| **Transformer** | `FluxTransformer2DModel` | `Flux2Transformer2DModel` |
| **Control Images** | Multiple (`control`, `control_1`, ...) | Single optional `image` |
| **Timestep Shift** | `calculate_shift()` | `compute_empirical_mu()` |
| **Latent Processing** | `scaling_factor` + `shift_factor` | `batch_norm` + `patchify` |

### 2.2 FLUX.2 Pipeline Components

From the [Flux2Pipeline source](https://raw.githubusercontent.com/huggingface/diffusers/refs/heads/main/src/diffusers/pipelines/flux2/pipeline_flux2.py):

```python
class Flux2Pipeline(DiffusionPipeline, Flux2LoraLoaderMixin):
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLFlux2,
        text_encoder: Mistral3ForConditionalGeneration,  # Single VLM
        tokenizer: AutoProcessor,  # PixtralProcessor
        transformer: Flux2Transformer2DModel,
    ):
```

### 2.3 Key Algorithm Differences

#### 2.3.1 Text Encoding (FLUX.2)

```python
def format_text_input(prompts: List[str], system_message: str = None):
    """Format prompts into conversation format for Mistral3"""
    return [
        [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        for prompt in prompts
    ]
```

#### 2.3.2 Timestep Calculation (FLUX.2)

```python
def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """FLUX.2 specific mu calculation for timestep scheduling"""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)
```

#### 2.3.3 VAE Decoding (FLUX.2)

```python
# FLUX.2 specific: batch_norm denormalization
latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1)
latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + eps)
latents = latents * latents_bn_std + latents_bn_mean
latents = self._unpatchify_latents(latents)
image = self.vae.decode(latents, return_dict=False)[0]
```

---

## 3. Dual Mode Design

### 3.1 Mode Detection Strategy

The trainer will automatically detect the mode based on batch data:

```python
def _detect_mode(self, batch: dict) -> str:
    """Detect T2I or I2I mode from batch data"""
    has_condition = "condition_image" in batch and batch["condition_image"] is not None
    return "i2i" if has_condition else "t2i"
```

### 3.2 Data Flow Comparison

#### Text-to-Image (T2I) Mode

```
Input:
  - prompt: str
  - image: target image (for training)

Embeddings:
  - prompt_embeds: [B, seq_txt, D]
  - text_ids: [seq_txt, 4]
  - image_latents: [B, seq_img, C] (target, for loss computation)

Forward:
  - hidden_states = noisy_latents  # Only target latents
  - No condition latents concatenation
```

#### Image-to-Image (I2I) Mode

```
Input:
  - prompt: str
  - image: target image (for training)
  - condition_image: reference image

Embeddings:
  - prompt_embeds: [B, seq_txt, D]
  - text_ids: [seq_txt, 4]
  - image_latents: [B, seq_img, C] (target)
  - condition_latents: [B, seq_cond, C] (reference)
  - condition_ids: [seq_cond, 4]

Forward:
  - hidden_states = concat([noisy_latents, condition_latents], dim=1)
  - img_ids = concat([latent_ids, condition_ids], dim=0)
```

### 3.3 Batch Structure Design

```python
# T2I Mode Batch
batch_t2i = {
    "image": torch.Tensor,           # [B, C, H, W] - target image
    "prompt": List[str],             # Text prompts
    # condition_image is None or not present
}

# I2I Mode Batch
batch_i2i = {
    "image": torch.Tensor,           # [B, C, H, W] - target image
    "prompt": List[str],             # Text prompts
    "condition_image": torch.Tensor, # [B, C, H, W] - reference image
}

# After prepare_embeddings()
embeddings = {
    # Common fields
    "prompt_embeds": torch.Tensor,      # [B, seq_txt, D]
    "text_ids": torch.Tensor,           # [seq_txt, 4]
    "image_latents": torch.Tensor,      # [B, seq_img, C]

    # I2I only (None for T2I)
    "condition_latents": torch.Tensor | None,  # [B, seq_cond, C]
    "condition_ids": torch.Tensor | None,      # [seq_cond, 4]

    # Metadata
    "mode": str,  # "t2i" or "i2i"
    "height": int,
    "width": int,
}
```

---

## 4. Implementation Plan

### 4.1 File Structure

```
src/qflux/
├── models/
│   └── flux2_loader.py              # NEW: Model loading utilities
├── trainer/
│   └── flux2_trainer.py             # NEW: Main trainer class
└── data/
    └── config.py                     # MODIFY: Add TrainerKind.Flux2

configs/
├── flux2_t2i_fp16_config.yaml       # NEW: T2I FP16 configuration
├── flux2_t2i_fp4_config.yaml        # NEW: T2I FP4 configuration
├── flux2_i2i_fp16_config.yaml       # NEW: I2I FP16 configuration
└── flux2_i2i_fp4_config.yaml        # NEW: I2I FP4 configuration

tests/
├── test_configs/
│   ├── test_flux2_t2i_fp4.yaml      # NEW: T2I test config
│   └── test_flux2_i2i_fp4.yaml      # NEW: I2I test config
└── src/trainer/
    └── test_flux2_predict.ipynb     # NEW: Test notebook for predict
```

### 4.2 Implementation Phases

#### Phase 1: Model Loader (`flux2_loader.py`)

**Functions to implement:**

```python
def load_flux2_vae(
    model_path: str,
    weight_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cpu",
) -> AutoencoderKLFlux2:
    """Load FLUX.2 VAE with batch_norm architecture"""

def load_flux2_text_encoder(
    model_path: str,
    weight_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cpu",
) -> Mistral3ForConditionalGeneration:
    """Load Mistral3 text encoder"""

def load_flux2_tokenizer(model_path: str) -> AutoProcessor:
    """Load PixtralProcessor for text/image processing"""

def load_flux2_transformer(
    model_path: str,
    weight_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cpu",
) -> Flux2Transformer2DModel:
    """Load FLUX.2 Transformer"""

def load_flux2_scheduler(model_path: str) -> FlowMatchEulerDiscreteScheduler:
    """Load scheduler"""
```

#### Phase 2: Core Trainer (`flux2_trainer.py`)

**Class structure:**

```python
class Flux2LoraTrainer(BaseTrainer):
    """
    FLUX.2 LoRA Trainer supporting both T2I and I2I modes.

    Inherits from BaseTrainer to ensure consistent interface.
    """

    def __init__(self, config: Config):
        super().__init__(config)

        # Components (different from FluxKontext)
        self.vae: AutoencoderKLFlux2
        self.text_encoder: Mistral3ForConditionalGeneration  # Single encoder
        self.tokenizer: AutoProcessor  # PixtralProcessor
        self.dit: Flux2Transformer2DModel
        self.scheduler: FlowMatchEulerDiscreteScheduler

        # No text_encoder_2, tokenizer_2 (unlike FluxKontext)

    # === Abstract method implementations ===

    def get_pipeline_class(self):
        """Return Flux2Pipeline for LoRA saving"""
        from diffusers import Flux2Pipeline
        return Flux2Pipeline

    def load_model(self):
        """Load all model components"""

    def encode_prompt(self, prompt, image=None):
        """Encode prompt using Mistral3"""

    def prepare_latents(self, image, batch_size, ...):
        """Prepare latents for training/inference"""

    def prepare_embeddings(self, batch, stage="fit"):
        """Prepare all embeddings (supports T2I and I2I)"""

    def prepare_cached_embeddings(self, batch):
        """Load cached embeddings"""

    def cache_step(self, data):
        """Save embeddings to cache"""

    def setup_model_device_train_mode(self, stage, cache=False):
        """Configure device placement and train mode"""

    def _compute_loss(self, embeddings):
        """Compute flow matching loss"""

    def sampling_from_embeddings(self, embeddings):
        """Run denoising loop for inference"""

    def decode_vae_latent(self, latents, height, width):
        """Decode latents to images using FLUX.2 VAE"""

    def prepare_predict_batch_data(self, ...):
        """Prepare data for prediction"""
```

#### Phase 3: Methods Implementation Detail

##### 3.1 `load_model()`

```python
def load_model(self):
    """Load and separate components"""
    logging.info("Loading Flux2Pipeline components...")

    # Load individual components using flux2_loader
    pretrains = self.config.model.pretrained_embeddings
    model_path = self.config.model.pretrained_model_name_or_path

    # VAE
    if pretrains and "vae" in pretrains:
        self.vae = load_flux2_vae(pretrains["vae"], self.weight_dtype)
    else:
        self.vae = load_flux2_vae(model_path, self.weight_dtype)

    # Text encoder (single Mistral3)
    if pretrains and "text_encoder" in pretrains:
        self.text_encoder = load_flux2_text_encoder(pretrains["text_encoder"], self.weight_dtype)
    else:
        self.text_encoder = load_flux2_text_encoder(model_path, self.weight_dtype)

    # Tokenizer
    self.tokenizer = load_flux2_tokenizer(model_path)

    # Transformer
    self.dit = load_flux2_transformer(model_path, self.weight_dtype)

    # Scheduler (with copy for sampling)
    self.scheduler = load_flux2_scheduler(model_path)
    self.sampling_scheduler = copy.deepcopy(self.scheduler)

    # Set VAE parameters (FLUX.2 specific)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.latent_channels = self.vae.config.latent_channels
    self.num_channels_latents = self.dit.config.in_channels // 4

    # Set models to eval mode
    self.text_encoder.requires_grad_(False).eval()
    self.vae.requires_grad_(False).eval()
    self.dit.requires_grad_(False).eval()

    # Image processor
    from diffusers.image_processor import VaeImageProcessor
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

    logging.info(f"Loaded FLUX.2 components. VAE scale factor: {self.vae_scale_factor}")
```

##### 3.2 `encode_prompt()`

```python
def encode_prompt(
    self,
    prompt: str | list[str],
    device: torch.device | None = None,
    max_sequence_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode prompts using Mistral3.

    Unlike FluxKontext which uses CLIP+T5, FLUX.2 uses single Mistral3 VLM.

    Returns:
        prompt_embeds: [B, seq, D]
        text_ids: [seq, 4]  # Note: 4 dims, not 3 like FluxKontext
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    device = device or self.text_encoder.device

    # Format prompts for Mistral3
    system_message = "You are a helpful assistant."
    formatted = self._format_text_input(prompt, system_message)

    with torch.inference_mode():
        # Tokenize with PixtralProcessor
        model_inputs = self.tokenizer(
            text=formatted,
            padding=True,
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="pt",
        ).to(device)

        # Get embeddings from Mistral3
        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            output_hidden_states=True,
        )

        # Extract hidden states
        prompt_embeds = outputs.hidden_states[-1]

        # Create text IDs [seq, 4] for FLUX.2
        seq_len = prompt_embeds.shape[1]
        text_ids = torch.zeros(seq_len, 4, device=device, dtype=self.weight_dtype)

    return prompt_embeds, text_ids

def _format_text_input(self, prompts: list[str], system_message: str = None):
    """Format prompts into conversation format for Mistral3"""
    # Remove [IMG] tokens to avoid validation issues
    cleaned_txt = [p.replace("[IMG]", "") for p in prompts]

    return [
        [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        for prompt in cleaned_txt
    ]
```

##### 3.3 `prepare_embeddings()` - Dual Mode Support

```python
def prepare_embeddings(self, batch: dict, stage: str = "fit") -> dict:
    """
    Prepare embeddings supporting both T2I and I2I modes.

    Args:
        batch: Input batch with image, prompt, and optional condition_image
        stage: "fit", "cache", or "predict"

    Returns:
        Updated batch with all embeddings
    """
    # Detect mode
    mode = self._detect_mode(batch)
    batch["mode"] = mode

    # === Common: Text encoding ===
    prompt_embeds, text_ids = self.encode_prompt(
        prompt=batch["prompt"],
        max_sequence_length=512,
    )
    batch["prompt_embeds"] = prompt_embeds
    batch["text_ids"] = text_ids

    # Empty prompt for cache mode
    if stage == "cache":
        empty_embeds, _ = self.encode_prompt(prompt=[""])
        batch["empty_prompt_embeds"] = empty_embeds

    # === Target image encoding (always needed for training) ===
    if "image" in batch:
        image = self._preprocess_image(batch["image"])
        batch["image"] = image
        batch["height"] = image.shape[2]
        batch["width"] = image.shape[3]

        _, image_latents, latent_ids, _ = self.prepare_latents(
            image=image,
            batch_size=image.shape[0],
            height=batch["height"],
            width=batch["width"],
        )
        batch["image_latents"] = image_latents
        batch["latent_ids"] = latent_ids

    # === Condition image encoding (I2I mode only) ===
    if mode == "i2i":
        condition = self._preprocess_image(batch["condition_image"])
        batch["condition_image"] = condition
        batch["condition_height"] = condition.shape[2]
        batch["condition_width"] = condition.shape[3]

        _, condition_latents, _, condition_ids = self.prepare_latents(
            image=condition,
            batch_size=condition.shape[0],
            height=batch["condition_height"],
            width=batch["condition_width"],
        )
        # Mark condition latents with domain ID = 1
        condition_ids[..., 0] = 1
        batch["condition_latents"] = condition_latents
        batch["condition_ids"] = condition_ids
    else:
        batch["condition_latents"] = None
        batch["condition_ids"] = None

    # === Negative prompt (predict mode) ===
    if stage == "predict" and "negative_prompt" in batch:
        neg_embeds, neg_ids = self.encode_prompt(batch["negative_prompt"])
        batch["negative_prompt_embeds"] = neg_embeds
        batch["negative_text_ids"] = neg_ids

    return batch

def _detect_mode(self, batch: dict) -> str:
    """Detect T2I or I2I mode from batch"""
    has_condition = (
        "condition_image" in batch
        and batch["condition_image"] is not None
    )
    return "i2i" if has_condition else "t2i"

def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
    """Preprocess image to [-1, 1] range"""
    if image.max() > 1.0:
        image = image / 255.0
    return image * 2.0 - 1.0
```

##### 3.4 `_compute_loss()` - Dual Mode Support

```python
def _compute_loss(self, embeddings: dict) -> torch.Tensor:
    """
    Compute flow matching loss for both T2I and I2I modes.

    The key difference is whether condition_latents are concatenated.
    """
    device = self.accelerator.device
    mode = embeddings.get("mode", "t2i")

    # Extract common embeddings
    image_latents = embeddings["image_latents"].to(device)
    prompt_embeds = embeddings["prompt_embeds"].to(device)
    text_ids = embeddings["text_ids"].to(device)
    latent_ids = embeddings["latent_ids"].to(device)

    batch_size = image_latents.shape[0]

    with torch.no_grad():
        # Sample noise
        noise = torch.randn_like(image_latents, device=device, dtype=self.weight_dtype)

        # Sample timestep (uniform)
        t = torch.rand((batch_size,), device=device, dtype=self.weight_dtype)
        t_ = t.view(-1, 1, 1)

        # Create noisy input
        noisy_model_input = (1.0 - t_) * image_latents + t_ * noise

        # === Mode-specific input preparation ===
        if mode == "i2i" and embeddings["condition_latents"] is not None:
            # I2I: Concatenate with condition latents
            condition_latents = embeddings["condition_latents"].to(device)
            condition_ids = embeddings["condition_ids"].to(device)

            hidden_states = torch.cat([noisy_model_input, condition_latents], dim=1)
            img_ids = torch.cat([latent_ids, condition_ids], dim=0)
        else:
            # T2I: Only noisy latents
            hidden_states = noisy_model_input
            img_ids = latent_ids

    # Prepare guidance (FLUX.2 specific)
    guidance = None
    if self.dit.config.guidance_embeds:
        guidance = torch.ones((batch_size,), device=device, dtype=self.weight_dtype)

    # Forward pass
    model_pred = self.dit(
        hidden_states=hidden_states.to(self.weight_dtype),
        timestep=t,  # FLUX.2 uses raw timestep, not t/1000
        guidance=guidance,
        encoder_hidden_states=prompt_embeds.to(self.weight_dtype),
        txt_ids=text_ids,
        img_ids=img_ids,
        joint_attention_kwargs={},
        return_dict=False,
    )[0]

    # Extract prediction for target latents only
    model_pred = model_pred[:, :image_latents.size(1)]

    # Compute loss
    target = noise - image_latents
    loss = self.forward_loss(model_pred, target, weighting=None, edit_mask=None)

    return loss
```

##### 3.5 `cache_step()` - Dual Mode Support

```python
def cache_step(self, data: dict):
    """
    Cache embeddings for both T2I and I2I modes.

    Cache structure varies by mode:
    - T2I: image_latents, prompt_embeds, text_ids
    - I2I: + condition_latents, condition_ids
    """
    mode = data.get("mode", "t2i")

    # Common cache items
    cache_embeddings = {
        "image_latents": data["image_latents"].detach().cpu()[0],
        "prompt_embeds": data["prompt_embeds"].detach().cpu()[0],
        "text_ids": data["text_ids"].detach().cpu(),
        "empty_prompt_embeds": data["empty_prompt_embeds"].detach().cpu()[0],
        "mode": mode,
    }

    map_keys = {
        "image_latents": "image_hash",
        "prompt_embeds": "prompt_hash",
        "text_ids": "prompt_hash",
        "empty_prompt_embeds": "prompt_hash",
    }

    # I2I mode: add condition cache
    if mode == "i2i" and data["condition_latents"] is not None:
        cache_embeddings["condition_latents"] = data["condition_latents"].detach().cpu()[0]
        cache_embeddings["condition_ids"] = data["condition_ids"].detach().cpu()
        map_keys["condition_latents"] = "condition_hash"
        map_keys["condition_ids"] = "condition_hash"

    self.cache_manager.save_cache_embedding(cache_embeddings, map_keys, data["file_hashes"])
```

##### 3.6 `prepare_cached_embeddings()` - Dual Mode Support

```python
def prepare_cached_embeddings(self, batch: dict) -> dict:
    """
    Load cached embeddings for both T2I and I2I modes.
    """
    mode = batch.get("mode", "t2i")
    batch["mode"] = mode

    # Common processing
    batch["text_ids"] = batch["text_ids"][0]  # Remove batch dim if needed

    # I2I mode: load condition cache
    if mode == "i2i" and "condition_latents" in batch:
        batch["condition_ids"] = batch["condition_ids"][0]
    else:
        batch["condition_latents"] = None
        batch["condition_ids"] = None

    return batch
```

##### 3.7 `sampling_from_embeddings()` - Dual Mode Support

```python
def sampling_from_embeddings(self, embeddings: dict) -> torch.Tensor:
    """
    Run denoising loop for both T2I and I2I modes.
    """
    mode = embeddings.get("mode", "t2i")
    device = self.dit.device

    num_inference_steps = embeddings["num_inference_steps"]
    batch_size = embeddings["prompt_embeds"].shape[0]
    height = embeddings["height"]
    width = embeddings["width"]

    # Create initial noise
    latents, latent_ids = self._create_noise_latents(
        height=height,
        width=width,
        batch_size=batch_size,
        device=device,
    )

    # Prepare condition (I2I mode)
    if mode == "i2i" and embeddings["condition_latents"] is not None:
        condition_latents = embeddings["condition_latents"].to(device)
        condition_ids = embeddings["condition_ids"].to(device)
    else:
        condition_latents = None
        condition_ids = None

    # Prepare text embeddings
    prompt_embeds = embeddings["prompt_embeds"].to(device)
    text_ids = embeddings["text_ids"].to(device)

    # Calculate timesteps using FLUX.2 specific mu
    image_seq_len = latents.shape[1]
    mu = compute_empirical_mu(image_seq_len, num_inference_steps)
    timesteps, _ = retrieve_timesteps(
        self.sampling_scheduler,
        num_inference_steps,
        device,
        mu=mu,
    )

    # Guidance
    guidance_scale = embeddings.get("guidance", 2.5)
    guidance = torch.full([batch_size], guidance_scale, device=device, dtype=torch.float32)

    # Denoising loop
    self.sampling_scheduler.set_begin_index(0)

    with torch.inference_mode():
        for t in tqdm(timesteps, desc="FLUX.2 Sampling"):
            # Prepare model input
            if condition_latents is not None:
                hidden_states = torch.cat([latents, condition_latents], dim=1)
                img_ids = torch.cat([latent_ids, condition_ids], dim=0)
            else:
                hidden_states = latents
                img_ids = latent_ids

            timestep = t.expand(batch_size).to(latents.dtype)

            # Forward
            noise_pred = self.dit(
                hidden_states=hidden_states,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]

            # Extract target prediction
            noise_pred = noise_pred[:, :latents.size(1)]

            # Scheduler step
            latents = self.sampling_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents
```

##### 3.8 `decode_vae_latent()` - FLUX.2 Specific

```python
def decode_vae_latent(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Decode latents using FLUX.2 VAE with batch_norm denormalization.

    FLUX.2 specific processing:
    1. Unpack latents with IDs
    2. Apply batch_norm inverse transform
    3. Unpatchify
    4. VAE decode
    """
    latents = latents.to(self.vae.device, dtype=self.weight_dtype)

    # Unpack latents
    latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)

    # FLUX.2 specific: batch_norm inverse transform
    bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(
        self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
    ).to(latents.device, latents.dtype)
    latents = latents * bn_std + bn_mean

    # Unpatchify
    latents = self._unpatchify_latents(latents)

    # VAE decode
    with torch.inference_mode():
        image = self.vae.decode(latents, return_dict=False)[0]

    image = self.image_processor.postprocess(image, output_type="pt")
    return image
```

#### Phase 4: Configuration Updates

##### 4.1 Update `config.py`

```python
# In TrainerKind enum
class TrainerKind(str, Enum):
    QwenImageEdit = "QwenImageEdit"
    FluxKontext = "FluxKontext"
    DreamOmni2 = "DreamOmni2"
    Flux2 = "Flux2"  # NEW
```

##### 4.2 Update `main.py`

```python
def import_trainer(trainer_kind: str):
    if trainer_kind == "Flux2":
        from qflux.trainer.flux2_trainer import Flux2LoraTrainer
        return Flux2LoraTrainer
    elif trainer_kind == "FluxKontext":
        from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer
        return FluxKontextLoraTrainer
    # ... other trainers
```

#### Phase 5: Configuration Files

配置文件按 **模式 (T2I/I2I)** 和 **精度 (FP16/FP4)** 分为 4 个：

| 配置文件 | 模式 | 精度 | VRAM | Batch Size |
|---------|------|------|------|------------|
| `flux2_t2i_fp16_config.yaml` | T2I | FP16 | ~50GB | 1 |
| `flux2_t2i_fp4_config.yaml` | T2I | FP4 | ~24GB | 2 |
| `flux2_i2i_fp16_config.yaml` | I2I | FP16 | ~50GB | 1 |
| `flux2_i2i_fp4_config.yaml` | I2I | FP4 | ~24GB | 2 |

##### 5.1 `configs/flux2_t2i_fp16_config.yaml` (Text-to-Image, FP16)

```yaml
trainer: Flux2

model:
  pretrained_model_name_or_path: "black-forest-labs/FLUX.2-dev"
  quantize: False
  lora:
    r: 16
    lora_alpha: 16
    init_lora_weights: "gaussian"
    target_modules: ["to_k", "to_q", "to_v", "to_out.0"]
    pretrained_weight: null

data:
  class_path: "qflux.data.dataset.ImageDataset"
  init_args:
    dataset_path:
      - split: train
        repo_id: TsienDragon/face_segmentation_20
    caption_dropout_rate: 0.05
    prompt_image_dropout_rate: 0.0  # T2I mode: no prompt image dropout
    cache_dir: ${cache.cache_dir}
    use_cache: ${cache.use_cache}
    cache_drop_rate: 0.1
    # T2I mode: no condition_image field in dataset
    processor:
      class_path: "qflux.data.preprocess.ImageProcessor"
      init_args:
        process_type: center_crop
        target_size: [1024, 1024]
  batch_size: 1
  num_workers: 2
  shuffle: true

logging:
  output_dir: "/raid/lilong/data/experiment/flux2-t2i_fp16/"
  report_to: "tensorboard"
  tracker_project_name: "flux2_t2i_face_seg_fp16"

optimizer:
  class_path: bitsandbytes.optim.Adam8bit
  init_args:
    lr: 0.0001
    betas: [0.9, 0.999]

lr_scheduler:
  scheduler_type: "cosine"
  warmup_steps: 50

train:
  gradient_accumulation_steps: 4
  max_train_steps: 6000
  num_epochs: 100
  checkpointing_steps: 100
  checkpoints_total_limit: 20
  max_grad_norm: 1.0
  mixed_precision: "bf16"
  gradient_checkpointing: True
  low_memory: True

cache:
  devices:
    vae: cuda:0
    text_encoder: cuda:0
  cache_dir: "/raid/lilong/data/experiment/flux2-t2i_fp16/cache"
  use_cache: true

predict:
  devices:
    vae: cuda:0
    text_encoder: cuda:0
    dit: cuda:0

resume: null

validation:
  enabled: false
  validation_steps: 200
  num_validation_samples: 4

# VRAM estimate: ~50GB for training with batch_size 1 on A100
# Mode: T2I (Text-to-Image) - no condition image
```

##### 5.2 `configs/flux2_t2i_fp4_config.yaml` (Text-to-Image, FP4)

```yaml
trainer: Flux2

model:
  pretrained_model_name_or_path: "diffusers/FLUX.2-dev-bnb-4bit"
  quantize: False  # Already pre-quantized
  lora:
    r: 16
    lora_alpha: 16
    init_lora_weights: "gaussian"
    target_modules: ["to_k", "to_q", "to_v", "to_out.0"]
    pretrained_weight: null

data:
  class_path: "qflux.data.dataset.ImageDataset"
  init_args:
    dataset_path:
      - split: train
        repo_id: TsienDragon/face_segmentation_20
    caption_dropout_rate: 0.05
    prompt_image_dropout_rate: 0.0  # T2I mode: no prompt image dropout
    cache_dir: ${cache.cache_dir}
    use_cache: ${cache.use_cache}
    cache_drop_rate: 0.1
    processor:
      class_path: "qflux.data.preprocess.ImageProcessor"
      init_args:
        process_type: center_crop
        target_size: [832, 576]
  batch_size: 2
  num_workers: 2
  shuffle: true

logging:
  output_dir: "/raid/lilong/data/experiment/flux2-t2i_fp4/"
  report_to: "tensorboard"
  tracker_project_name: "flux2_t2i_face_seg_fp4"

optimizer:
  class_path: bitsandbytes.optim.Adam8bit
  init_args:
    lr: 0.0001
    betas: [0.9, 0.999]

lr_scheduler:
  scheduler_type: "cosine"
  warmup_steps: 50

train:
  gradient_accumulation_steps: 1
  max_train_steps: 6000
  num_epochs: 100
  checkpointing_steps: 100
  checkpoints_total_limit: 20
  max_grad_norm: 1.0
  mixed_precision: "bf16"
  gradient_checkpointing: True
  low_memory: True

cache:
  devices:
    vae: cuda:0
    text_encoder: cuda:0
  cache_dir: "/raid/lilong/data/experiment/flux2-t2i_fp4/cache"
  use_cache: true

predict:
  devices:
    vae: cuda:0
    text_encoder: cuda:0
    dit: cuda:0

resume: null

validation:
  enabled: false
  validation_steps: 200
  num_validation_samples: 4

# VRAM estimate: ~24GB for training with batch_size 2 on RTX 4090
# Mode: T2I (Text-to-Image) - no condition image
```

##### 5.3 `configs/flux2_i2i_fp16_config.yaml` (Image-to-Image, FP16)

```yaml
trainer: Flux2

model:
  pretrained_model_name_or_path: "black-forest-labs/FLUX.2-dev"
  quantize: False
  lora:
    r: 16
    lora_alpha: 16
    init_lora_weights: "gaussian"
    target_modules: ["to_k", "to_q", "to_v", "to_out.0"]
    pretrained_weight: null

data:
  class_path: "qflux.data.dataset.ImageDataset"
  init_args:
    dataset_path:
      - split: train
        repo_id: TsienDragon/face_segmentation_20
    caption_dropout_rate: 0.05
    prompt_image_dropout_rate: 0.05  # I2I mode: supports dropout
    cache_dir: ${cache.cache_dir}
    use_cache: ${cache.use_cache}
    cache_drop_rate: 0.1
    # I2I mode: dataset should have condition_image field
    # condition_image maps to 'control' in the dataset
    processor:
      class_path: "qflux.data.preprocess.ImageProcessor"
      init_args:
        process_type: center_crop
        target_size: [1024, 1024]
        controls_size: [[1024, 1024]]  # I2I: condition image size
  batch_size: 1
  num_workers: 2
  shuffle: true

logging:
  output_dir: "/raid/lilong/data/experiment/flux2-i2i_fp16/"
  report_to: "tensorboard"
  tracker_project_name: "flux2_i2i_face_seg_fp16"

optimizer:
  class_path: bitsandbytes.optim.Adam8bit
  init_args:
    lr: 0.0001
    betas: [0.9, 0.999]

lr_scheduler:
  scheduler_type: "cosine"
  warmup_steps: 50

train:
  gradient_accumulation_steps: 4
  max_train_steps: 6000
  num_epochs: 100
  checkpointing_steps: 100
  checkpoints_total_limit: 20
  max_grad_norm: 1.0
  mixed_precision: "bf16"
  gradient_checkpointing: True
  low_memory: True

cache:
  devices:
    vae: cuda:0
    text_encoder: cuda:0
  cache_dir: "/raid/lilong/data/experiment/flux2-i2i_fp16/cache"
  use_cache: true

predict:
  devices:
    vae: cuda:0
    text_encoder: cuda:0
    dit: cuda:0

resume: null

validation:
  enabled: false
  validation_steps: 200
  num_validation_samples: 4

# VRAM estimate: ~50GB for training with batch_size 1 on A100
# Mode: I2I (Image-to-Image) - with condition image
```

##### 5.4 `configs/flux2_i2i_fp4_config.yaml` (Image-to-Image, FP4)

```yaml
trainer: Flux2

model:
  pretrained_model_name_or_path: "diffusers/FLUX.2-dev-bnb-4bit"
  quantize: False  # Already pre-quantized
  lora:
    r: 16
    lora_alpha: 16
    init_lora_weights: "gaussian"
    target_modules: ["to_k", "to_q", "to_v", "to_out.0"]
    pretrained_weight: null

data:
  class_path: "qflux.data.dataset.ImageDataset"
  init_args:
    dataset_path:
      - split: train
        repo_id: TsienDragon/face_segmentation_20
    caption_dropout_rate: 0.05
    prompt_image_dropout_rate: 0.05  # I2I mode: supports dropout
    cache_dir: ${cache.cache_dir}
    use_cache: ${cache.use_cache}
    cache_drop_rate: 0.1
    processor:
      class_path: "qflux.data.preprocess.ImageProcessor"
      init_args:
        process_type: center_crop
        target_size: [832, 576]
        controls_size: [[832, 576]]  # I2I: condition image size
  batch_size: 2
  num_workers: 2
  shuffle: true

logging:
  output_dir: "/raid/lilong/data/experiment/flux2-i2i_fp4/"
  report_to: "tensorboard"
  tracker_project_name: "flux2_i2i_face_seg_fp4"

optimizer:
  class_path: bitsandbytes.optim.Adam8bit
  init_args:
    lr: 0.0001
    betas: [0.9, 0.999]

lr_scheduler:
  scheduler_type: "cosine"
  warmup_steps: 50

train:
  gradient_accumulation_steps: 1
  max_train_steps: 6000
  num_epochs: 100
  checkpointing_steps: 100
  checkpoints_total_limit: 20
  max_grad_norm: 1.0
  mixed_precision: "bf16"
  gradient_checkpointing: True
  low_memory: True

cache:
  devices:
    vae: cuda:0
    text_encoder: cuda:0
  cache_dir: "/raid/lilong/data/experiment/flux2-i2i_fp4/cache"
  use_cache: true

predict:
  devices:
    vae: cuda:0
    text_encoder: cuda:0
    dit: cuda:0

resume: null

validation:
  enabled: false
  validation_steps: 200
  num_validation_samples: 4

# VRAM estimate: ~24GB for training with batch_size 2 on RTX 4090
# Mode: I2I (Image-to-Image) - with condition image
```

#### Phase 6: Test Notebook

##### 6.1 `tests/src/trainer/test_flux2_predict.ipynb`

测试 notebook 用于验证原始模型和加载 LoRA 后的模型推理效果。

**Notebook 结构设计：**

```
Cell 0: Setup path
────────────────────────────────────────
package = "../../"
import sys
import os
package = os.path.abspath(package)
sys.path.append(package)

Cell 1: Imports
────────────────────────────────────────
import numpy as np
import torch
import logging
import sys
from qflux.trainer.flux2_trainer import Flux2LoraTrainer
from qflux.data.config import load_config_from_yaml
from diffusers.utils import load_image

Cell 2: Logging setup
────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

Cell 3: Section header
────────────────────────────────────────
# 1. FLUX.2 Text-to-Image (T2I) Mode

Cell 4: Section header
────────────────────────────────────────
## 1.1 T2I Without LoRA (Base Model)

Cell 5: Load test image and prompt
────────────────────────────────────────
# T2I mode: only prompt, no condition image
prompt = 'A beautiful portrait photo of a woman with face segmentation mask overlay'

Cell 6: Load T2I config
────────────────────────────────────────
config = "../../tests/test_configs/test_flux2_t2i_fp4.yaml"
config = load_config_from_yaml(config)

Cell 7: Create trainer (T2I, no LoRA)
────────────────────────────────────────
trainer = Flux2LoraTrainer(config)

Cell 8: T2I inference without LoRA
────────────────────────────────────────
out = trainer.predict(
    image=None,  # T2I mode: no condition image
    prompt=prompt,
    num_inference_steps=20,
    guidance_scale=2.5,
    weight_dtype=torch.bfloat16,
    height=576,
    width=832,
    output_type='pil'
)

Cell 9: Display result
────────────────────────────────────────
out[0]

Cell 10: Section header
────────────────────────────────────────
## 1.2 T2I With LoRA

Cell 11: Set LoRA weight path
────────────────────────────────────────
LORA_WEIGHT = '/path/to/your/flux2_t2i_lora/pytorch_lora_weights.safetensors'
# Or use HuggingFace repo:
# LORA_WEIGHT = 'YourUsername/flux2-t2i-face-segmentation'

Cell 12: Load config with LoRA
────────────────────────────────────────
config = "../../tests/test_configs/test_flux2_t2i_fp4.yaml"
config = load_config_from_yaml(config)
config.model.lora.pretrained_weight = LORA_WEIGHT
trainer = Flux2LoraTrainer(config)

Cell 13: T2I inference with LoRA
────────────────────────────────────────
out_lora = trainer.predict(
    image=None,
    prompt=prompt,
    num_inference_steps=20,
    guidance_scale=2.5,
    weight_dtype=torch.bfloat16,
    height=576,
    width=832,
    output_type='pil'
)

Cell 14: Display result
────────────────────────────────────────
out_lora[0]

Cell 15: Section header
────────────────────────────────────────
# 2. FLUX.2 Image-to-Image (I2I) Mode

Cell 16: Section header
────────────────────────────────────────
## 2.1 I2I Without LoRA (Base Model)

Cell 17: Load condition image
────────────────────────────────────────
IMAGE_PATH = 'https://n.sinaimg.cn/ent/transform/775/w630h945/20201127/cee0-kentcvx8062290.jpg'
condition_image = load_image(IMAGE_PATH)
prompt = 'change the image from the face to the face segmentation mask'

Cell 18: Display condition image
────────────────────────────────────────
condition_image

Cell 19: Load I2I config
────────────────────────────────────────
config = "../../tests/test_configs/test_flux2_i2i_fp4.yaml"
config = load_config_from_yaml(config)

Cell 20: Create trainer (I2I, no LoRA)
────────────────────────────────────────
trainer = Flux2LoraTrainer(config)

Cell 21: I2I inference without LoRA
────────────────────────────────────────
out = trainer.predict(
    image=condition_image,  # I2I mode: with condition image
    prompt=prompt,
    num_inference_steps=20,
    guidance_scale=2.5,
    weight_dtype=torch.bfloat16,
    height=945,
    width=630,
    output_type='pil'
)

Cell 22: Display result
────────────────────────────────────────
out[0]

Cell 23: Section header
────────────────────────────────────────
## 2.2 I2I With LoRA

Cell 24: Set LoRA weight path
────────────────────────────────────────
LORA_WEIGHT = '/path/to/your/flux2_i2i_lora/pytorch_lora_weights.safetensors'
# Or use HuggingFace repo:
# LORA_WEIGHT = 'YourUsername/flux2-i2i-face-segmentation'

Cell 25: Load config with LoRA
────────────────────────────────────────
config = "../../tests/test_configs/test_flux2_i2i_fp4.yaml"
config = load_config_from_yaml(config)
config.model.lora.pretrained_weight = LORA_WEIGHT
trainer = Flux2LoraTrainer(config)

Cell 26: I2I inference with LoRA
────────────────────────────────────────
out_lora = trainer.predict(
    image=condition_image,
    prompt=prompt,
    num_inference_steps=20,
    guidance_scale=2.5,
    weight_dtype=torch.bfloat16,
    height=945,
    width=630,
    output_type='pil'
)

Cell 27: Display result
────────────────────────────────────────
out_lora[0]

Cell 28: Section header
────────────────────────────────────────
# 3. Comparison: Base Model vs LoRA

Cell 29: Side-by-side comparison
────────────────────────────────────────
from PIL import Image
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(condition_image)
axes[0].set_title('Input Image')
axes[0].axis('off')

axes[1].imshow(out[0])
axes[1].set_title('Base Model Output')
axes[1].axis('off')

axes[2].imshow(out_lora[0])
axes[2].set_title('LoRA Model Output')
axes[2].axis('off')

plt.tight_layout()
plt.show()

Cell 30: Save results (optional)
────────────────────────────────────────
# out[0].save('flux2_base_output.png')
# out_lora[0].save('flux2_lora_output.png')
```

##### 6.2 Test Config Files for Notebook

需要创建对应的测试配置文件：

**`tests/test_configs/test_flux2_t2i_fp4.yaml`:**
```yaml
trainer: Flux2

model:
  pretrained_model_name_or_path: "diffusers/FLUX.2-dev-bnb-4bit"
  quantize: False
  lora:
    r: 16
    lora_alpha: 16
    init_lora_weights: "gaussian"
    target_modules: ["to_k", "to_q", "to_v", "to_out.0"]
    pretrained_weight: null

data:
  class_path: "qflux.data.dataset.ImageDataset"
  init_args:
    dataset_path:
      - split: train
        repo_id: TsienDragon/face_segmentation_20
    processor:
      class_path: "qflux.data.preprocess.ImageProcessor"
      init_args:
        process_type: center_crop
        target_size: [832, 576]
  batch_size: 1

cache:
  devices:
    vae: cuda:0
    text_encoder: cuda:0
  cache_dir: "/tmp/flux2_test_cache"
  use_cache: false

predict:
  devices:
    vae: cuda:0
    text_encoder: cuda:0
    dit: cuda:0
```

**`tests/test_configs/test_flux2_i2i_fp4.yaml`:**
```yaml
trainer: Flux2

model:
  pretrained_model_name_or_path: "diffusers/FLUX.2-dev-bnb-4bit"
  quantize: False
  lora:
    r: 16
    lora_alpha: 16
    init_lora_weights: "gaussian"
    target_modules: ["to_k", "to_q", "to_v", "to_out.0"]
    pretrained_weight: null

data:
  class_path: "qflux.data.dataset.ImageDataset"
  init_args:
    dataset_path:
      - split: train
        repo_id: TsienDragon/face_segmentation_20
    processor:
      class_path: "qflux.data.preprocess.ImageProcessor"
      init_args:
        process_type: center_crop
        target_size: [832, 576]
        controls_size: [[832, 576]]
  batch_size: 1

cache:
  devices:
    vae: cuda:0
    text_encoder: cuda:0
  cache_dir: "/tmp/flux2_test_cache"
  use_cache: false

predict:
  devices:
    vae: cuda:0
    text_encoder: cuda:0
    dit: cuda:0
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

| Test | Description | Mode |
|------|-------------|------|
| `test_load_model` | Verify all components load correctly | Both |
| `test_encode_prompt` | Test Mistral3 prompt encoding | Both |
| `test_prepare_latents_t2i` | Test latent preparation without condition | T2I |
| `test_prepare_latents_i2i` | Test latent preparation with condition | I2I |
| `test_prepare_embeddings_t2i` | Full embedding preparation T2I | T2I |
| `test_prepare_embeddings_i2i` | Full embedding preparation I2I | I2I |
| `test_compute_loss_t2i` | Loss computation without condition | T2I |
| `test_compute_loss_i2i` | Loss computation with condition | I2I |
| `test_decode_vae` | Test FLUX.2 VAE decoding | Both |

### 5.2 Integration Tests

| Test | Description |
|------|-------------|
| `test_fit_t2i` | Full training loop T2I mode |
| `test_fit_i2i` | Full training loop I2I mode |
| `test_cache_t2i` | Cache building T2I mode |
| `test_cache_i2i` | Cache building I2I mode |
| `test_predict_t2i` | Inference T2I mode |
| `test_predict_i2i` | Inference I2I mode |

### 5.3 Validation Tests

| Test | Description |
|------|-------------|
| `test_lora_save_load` | LoRA weights save/load correctly |
| `test_output_quality` | Generated images are reasonable |
| `test_mode_detection` | Auto mode detection works |

---

## 6. Timeline Estimate

| Phase | Description | Duration |
|-------|-------------|----------|
| Phase 1 | Model loader | 1 day |
| Phase 2 | Core trainer (load_model, encode_prompt) | 1-2 days |
| Phase 3 | Dual mode methods (prepare_embeddings, _compute_loss) | 2-3 days |
| Phase 4 | Cache and predict methods | 1-2 days |
| Phase 5 | Configuration files | 0.5 day |
| Phase 6 | Testing and validation | 2-3 days |

**Total**: ~8-12 days

---

## 7. Risks and Mitigations

### 7.1 Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Mistral3 memory usage | High VRAM | Use 4-bit quantization, gradient checkpointing |
| VAE batch_norm compatibility | Incorrect decoding | Careful implementation following pipeline |
| LoRA save/load format | Incompatible weights | Use Flux2LoraLoaderMixin |
| Timestep calculation | Poor quality | Match compute_empirical_mu exactly |

### 7.2 Open Questions

1. **Dataset format**: Should we support existing FluxKontext datasets with multiple controls?
2. **Multi-resolution**: Do we need multi-resolution support for FLUX.2?
3. **VLM prompt enhancement**: Should we integrate VLM prompt optimization?

---

## 8. Success Criteria

- [ ] T2I mode: Training runs without errors
- [ ] I2I mode: Training runs without errors
- [ ] Cache mode works for both T2I and I2I
- [ ] Predict mode generates reasonable images
- [ ] FP16 model loads and trains
- [ ] FP4 model loads and trains
- [ ] LoRA weights can be saved and loaded
- [ ] All unit tests pass
- [ ] Integration tests pass

---

## 9. Dependencies

### 9.1 Required Packages

```
diffusers >= 0.30.0  # For FLUX.2 support
transformers >= 4.40.0  # For Mistral3
peft >= 0.10.0
torch >= 2.0.0
```

### 9.2 Model Dependencies

- `black-forest-labs/FLUX.2-dev` - FP16 model
- `diffusers/FLUX.2-dev-bnb-4bit` - FP4 quantized model
- HuggingFace account with access to FLUX.2

---

## Appendix A: Code Snippets from FLUX.2 Pipeline

### A.1 compute_empirical_mu

```python
def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)
```

### A.2 VAE Decode with Batch Norm

```python
if output_type == "latent":
    image = latents
else:
    latents = self._unpack_latents_with_ids(latents, latent_ids)

    latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
        latents.device, latents.dtype
    )
    latents = latents * latents_bn_std + latents_bn_mean
    latents = self._unpatchify_latents(latents)

    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)
```

---

**End of Document**

