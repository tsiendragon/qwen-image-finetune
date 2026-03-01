# Training & Test Status

Track which training runs and automated tests have been executed and verified.
Update this file after each test run. Add notes with GPU model, date, and any issues.

---

## Qwen-Image-Edit-2511 (`QwenImageEditPlusTrainer`)

新增模型支持，以下内容尚未验证。

### 训练流程


| 场景                                | 配置文件                                     | 状态    | 备注                                       |
| --------------------------------- | ---------------------------------------- | ----- | ---------------------------------------- |
| LoRA + embedding cache (cache 阶段) | `qwen2511_lora_with_cache.yaml --cache`  | ⬜ 未测试 |                                          |
| LoRA + embedding cache (train 阶段) | `qwen2511_lora_with_cache.yaml`          | ⬜ 未测试 |                                          |
| LoRA + 无 cache                    | `qwen2511_lora_no_cache.yaml`            | ⬜ 未测试 |                                          |
| LoRA + 本地模型离线训练 (cache 阶段)        | `qwen2511_lora_local_model.yaml --cache` | ⬜ 未测试 | 需提前下载模型                                  |
| LoRA + 本地模型离线训练 (train 阶段)        | `qwen2511_lora_local_model.yaml`         | ⬜ 未测试 |                                          |
| 全层 LoRA (`all-linear`, r=64)      | `qwen2511_lora_full_coverage.yaml`       | ⬜ 未测试 | 显存占用待测                                   |
| 全参数微调（无 LoRA）                     | `qwen2511_full_finetune.yaml`            | ⬜ 未测试 | 需 ~80GB+ 显存；输出 `transformer.safetensors` |
| 多 GPU 训练 (DDP)                    | 任意 2511 config + multi-GPU accelerate    | ⬜ 未测试 |                                          |
| 从已有 LoRA checkpoint 恢复训练          | `lora.pretrained_weight` 字段              | ⬜ 未测试 |                                          |


### 推理 / Sampling


| 场景                                         | 状态    | 备注  |
| ------------------------------------------ | ----- | --- |
| 2511 base model sampling (无 LoRA)          | ⬜ 未测试 |     |
| 2511 + LoRA sampling                       | ⬜ 未测试 |     |
| `true_cfg_scale > 1` (negative prompt CFG) | ⬜ 未测试 |     |
| 多控制图像输入 (`condition_images` list)          | ⬜ 未测试 |     |


### Bug 修复验证

以下 bug 已在代码中修复，但尚未有自动化测试覆盖：


| Bug                                          | 修复位置                                         | 自动化测试 | 备注                                   |
| -------------------------------------------- | -------------------------------------------- | ----- | ------------------------------------ |
| `enumerate[Tensor](image)` 语法错误              | `qwen_image_edit_plus_trainer.py:339`        | ⬜ 无测试 | 修复前会在运行时崩溃                           |
| `prepare_embeddings` negative_prompt 解包错误    | `qwen_image_edit_plus_trainer.py:199`        | ⬜ 无测试 | `debug=False` 时解包 4 个返回值但函数只返回 2 个   |
| `load_qwenvl` 忽略 path 参数（硬编码路径）              | `load_model.py:load_qwenvl`                  | ⬜ 无测试 | 现在使用传入的路径                            |
| VAE/text encoder 硬编码为 `Qwen-Image-Edit-2509` | `qwen_image_edit_plus_trainer.py:load_model` | ⬜ 无测试 | 现在使用 `pretrained_model_name_or_path` |


### 自动化测试（需新增）


| 测试文件（待创建）                                                | 测试内容                                                                | 优先级 |
| -------------------------------------------------------- | ------------------------------------------------------------------- | --- |
| `tests/e2e/test_qwen_2511_sampling.py`                   | 2511 模型 sampling 端到端测试                                              | 高   |
| `tests/src/trainer/test_qwen_image_edit_plus_trainer.py` | `encode_prompt`、`_get_qwen_prompt_embeds`、`prepare_embeddings` 单元测试 | 高   |
| `tests/src/models/test_load_model.py`                    | `load_qwenvl` 路径参数实际生效测试                                            | 中   |
| `tests/src/trainer/test_qwen_2511_negative_prompt.py`    | `debug=False` 时 negative_prompt 流程测试                                | 中   |


### 测试资源（待上传 HuggingFace）

参见 `tests/resources_config.yaml`，以下资源 size 标注为 TBD，尚未上传：


| Resource Group       | 用途                                      | 状态         |
| -------------------- | --------------------------------------- | ---------- |
| `qwen_sampling`      | `test_qwen_image_edit_sampling.py`      | ⬜ TBD（未上传） |
| `qwen_plus_sampling` | `test_qwen_image_edit_plus_sampling.py` | ⬜ TBD（未上传） |
| `qwen_2511_sampling` | 2511 e2e 采样测试（待创建）                      | ⬜ 未创建      |
| `qwen_t2i_sampling`  | T2I e2e 采样测试（待创建）                       | ⬜ 未创建      |

---

## Qwen-Image（文生图 `QwenImageT2ITrainer`）

新增模型支持，以下内容尚未验证。

### 训练流程

| 场景 | 配置文件 | 状态 | 备注 |
|------|---------|------|------|
| LoRA + embedding cache (cache 阶段) | `qwen_t2i_lora_with_cache.yaml --cache` | ⬜ 未测试 | |
| LoRA + embedding cache (train 阶段) | `qwen_t2i_lora_with_cache.yaml` | ⬜ 未测试 | |
| 全参数微调（无 LoRA） | `full_finetune: true` + yaml | ⬜ 未测试 | 需 ~80GB+ 显存 |

### 推理 / Sampling

| 场景 | 状态 | 备注 |
|------|------|------|
| base model sampling (无 LoRA) | ⬜ 未测试 | |
| T2I + LoRA sampling | ⬜ 未测试 | |
| `true_cfg_scale > 1` (CFG) | ⬜ 未测试 | |

### 自动化测试

| 测试文件 | 测试内容 | 状态 |
|---------|---------|------|
| `tests/e2e/test_qwen_image_t2i_vs_diffusers.py` | 与官方 `QwenImagePipeline` 对比：权重、embeddings、端到端输出 | ⬜ 已创建，待 GPU 验证 |
| `tests/e2e/test_qwen_image_t2i_sampling.py` | T2I 端到端采样测试（需上传参考数据） | ⬜ 未创建 |
| `tests/test_configs/test_example_qwen_image_t2i_fp16.yaml` | T2I 测试配置文件 | ✅ 已创建 |

---

## Qwen-Image-Edit（原始版本 `QwenImageEditTrainer`）


| 场景                                             | 状态       | 备注                                        |
| ---------------------------------------------- | -------- | ----------------------------------------- |
| LoRA + cache 训练                                | ✅ 已验证    | `face_seg_config.yaml`                    |
| LoRA + FP4 量化训练                                | ✅ 已验证    | `face_seg_fp4_config.yaml`                |
| E2E sampling 测试                                | ✅ 有自动化测试 | `test_qwen_image_edit_sampling.py`（资源待上传） |
| 本地模型 + `pretrained_embeddings.text_encoder` 路径 | ⬜ 未测试    | 本次新增功能，旧版 trainer 同样支持                    |


---

## Z-Image（`ZImageT2ITrainer`）

新增模型支持，以下内容尚未验证。

架构特点：S3-DiT（单流 6B DiT），Qwen3 文本编码器，可变长度 prompt embeddings，
Timestep 反转约定（0=噪声，1=干净），输出取反（`noise_pred = -model_output`）。

### 训练流程

| 场景 | 配置文件 | 状态 | 备注 |
|------|---------|------|------|
| LoRA + embedding cache (cache 阶段) | `zimage_t2i_lora_with_cache.yaml --cache` | ⬜ 未测试 | |
| LoRA + embedding cache (train 阶段) | `zimage_t2i_lora_with_cache.yaml` | ⬜ 未测试 | |

### 推理 / Sampling

| 场景 | 状态 | 备注 |
|------|------|------|
| base model sampling (无 LoRA) | ⬜ 未测试 | `guidance_scale=0.0` for Turbo |
| Z-Image + LoRA sampling | ⬜ 未测试 | |
| CFG sampling (`guidance_scale > 0`) | ⬜ 未测试 | 非 Turbo 用法 |

### 自动化测试

| 测试文件 | 测试内容 | 状态 |
|---------|---------|------|
| `tests/e2e/test_zimage_t2i_vs_diffusers.py` | 与官方 `ZImagePipeline` 对比：权重、embeddings、端到端输出 | ⬜ 已创建，待 GPU 验证 |
| `tests/test_configs/test_example_zimage_t2i_fp16.yaml` | Z-Image 测试配置文件 | ✅ 已创建 |

---

## HunyuanImage T2I（`HunyuanImageT2ITrainer`）

新增模型支持，以下内容尚未验证。

架构特点：MMDiT（20 双流 + 40 单流块），双文本编码器（Qwen2.5-VL-7B + ByT5），
`AutoencoderKLHunyuanImage`（`spatial_compression_ratio=32`），蒸馏变体（8 步推理），
`distilled_guidance_scale * 1000` 作为 guidance embedding，sigma ∈ [0, 1] 直接作为 timestep。

### 训练流程

| 场景 | 配置文件 | 状态 | 备注 |
|------|---------|------|------|
| LoRA + embedding cache (cache 阶段) | `hunyuan_image_t2i_lora_with_cache.yaml --cache` | ⬜ 未测试 | |
| LoRA + embedding cache (train 阶段) | `hunyuan_image_t2i_lora_with_cache.yaml` | ⬜ 未测试 | |

### 推理 / Sampling

| 场景 | 状态 | 备注 |
|------|------|------|
| base model sampling (无 LoRA) | ⬜ 未测试 | `distilled_guidance_scale=3.25`，~8 步 |
| HunyuanImage + LoRA sampling | ⬜ 未测试 | |

### 自动化测试

| 测试文件 | 测试内容 | 状态 |
|---------|---------|------|
| `tests/e2e/test_hunyuan_image_t2i_vs_diffusers.py` | 与官方 `HunyuanImagePipeline` 对比：权重（4模块）、双编码器 embeddings（Qwen+ByT5）、端到端输出 | ⬜ 已创建，待 GPU 验证 |
| `tests/test_configs/test_example_hunyuan_image_t2i_fp16.yaml` | HunyuanImage 测试配置文件 | ✅ 已创建 |

---

## HunyuanImage IT2I（`HunyuanImageIT2ITrainer`）

新增模型支持，以下内容尚未验证。

架构特点：与 T2I 共用 `HunyuanImageTransformer2DModel`，但使用 `AutoencoderKLHunyuanImageRefiner`
（3D 因果 VAE，`spatial_compression_ratio=16`），单 Qwen 文本编码器（Llama-style 模板，`drop_idx=36`，
`max_len=256`，无 ByT5），5D latents `(B, C, 1, H//16, W//16)` + token interleaving，
model input = `cat([noisy_target, cond_source], dim=1)`（双通道拼接），蒸馏变体（4 步推理）。

数据集格式：`image`（目标图像） + `control`（可选源图像；缺失时退化为自精炼）。

### 训练流程

| 场景 | 配置文件 | 状态 | 备注 |
|------|---------|------|------|
| LoRA + embedding cache (cache 阶段) | `hunyuan_image_it2i_lora_with_cache.yaml --cache` | ⬜ 未测试 | |
| LoRA + embedding cache (train 阶段) | `hunyuan_image_it2i_lora_with_cache.yaml` | ⬜ 未测试 | |

### 推理 / Sampling

| 场景 | 状态 | 备注 |
|------|------|------|
| base model sampling (无 LoRA) | ⬜ 未测试 | `distilled_guidance_scale=3.25`，~4 步 |
| IT2I + LoRA sampling | ⬜ 未测试 | |
| 自精炼模式（无 `control` 图像） | ⬜ 未测试 | source == target |

### 自动化测试

| 测试文件 | 测试内容 | 状态 |
|---------|---------|------|
| `tests/e2e/test_hunyuan_image_it2i_vs_diffusers.py` | 与官方 `HunyuanImageRefinerPipeline` 对比：权重（3模块）、Qwen embeddings、端到端输出（共享 source image） | ⬜ 已创建，待 GPU 验证 |
| `tests/test_configs/test_example_hunyuan_image_it2i_fp16.yaml` | HunyuanImage IT2I 测试配置文件 | ✅ 已创建 |

---

## Ovis-Image T2I（`OvisImageT2ITrainer`）

新增模型支持，以下内容尚未验证。

架构特点：FLUX-like MMDiT，packed 2×2 patch latents，Qwen3 文本编码器，
`apply_chat_template(enable_thinking=False)` + 丢弃前 28 个系统前缀 token，
`last_hidden_state`，`num_channels_latents = in_channels // 4`，
text_ids `[0, i, i]`，img_ids `[0, row, col]`，`t / 1000` 作为 timestep。

### 训练流程

| 场景 | 配置文件 | 状态 | 备注 |
|------|---------|------|------|
| LoRA + embedding cache (cache 阶段) | `ovis_image_t2i_lora_with_cache.yaml --cache` | ⬜ 未测试 | |
| LoRA + embedding cache (train 阶段) | `ovis_image_t2i_lora_with_cache.yaml` | ⬜ 未测试 | |

### 推理 / Sampling

| 场景 | 状态 | 备注 |
|------|------|------|
| base model sampling (无 LoRA) | ⬜ 未测试 | `guidance_scale=0.0` for distilled |
| Ovis-Image + LoRA sampling | ⬜ 未测试 | |
| CFG sampling (`guidance_scale > 1`) | ⬜ 未测试 | |

### 自动化测试

| 测试文件 | 测试内容 | 状态 |
|---------|---------|------|
| `tests/e2e/test_ovis_image_t2i_vs_diffusers.py` | 与官方 `OvisImagePipeline` 对比：权重、Qwen3 embeddings、端到端输出 | ⬜ 已创建，待 GPU 验证 |
| `tests/test_configs/test_example_ovis_image_t2i_fp16.yaml` | Ovis-Image 测试配置文件 | ✅ 已创建 |

---

## LongCat-Image T2I（`LongCatImageT2ITrainer`）

新增模型支持，以下内容尚未验证。

架构特点：FLUX-like MMDiT，packed 2×2 patch latents，Qwen2.5-VL 文本编码器（纯文本模式），
prefix/suffix 模板包裹，`hidden_states[-1]` 提取内容切片，`split_quotation` 对引号文本逐字符分词，
3D RoPE 位置编码（modality_id：text=0, image=1；img_ids offset=512），
`guidance=None` 传给 transformer，`t / 1000` 作为 timestep。

### 训练流程

| 场景 | 配置文件 | 状态 | 备注 |
|------|---------|------|------|
| LoRA + embedding cache (cache 阶段) | `longcat_image_t2i_lora_with_cache.yaml --cache` | ⬜ 未测试 | |
| LoRA + embedding cache (train 阶段) | `longcat_image_t2i_lora_with_cache.yaml` | ⬜ 未测试 | |

### 推理 / Sampling

| 场景 | 状态 | 备注 |
|------|------|------|
| base model sampling (无 LoRA) | ⬜ 未测试 | |
| LongCat-Image + LoRA sampling | ⬜ 未测试 | |
| CFG sampling (`guidance_scale > 1`) | ⬜ 未测试 | |

### 自动化测试

| 测试文件 | 测试内容 | 状态 |
|---------|---------|------|
| `tests/e2e/test_longcat_image_t2i_vs_diffusers.py` | 与官方 `LongCatImagePipeline` 对比：权重、prefix/suffix embeddings、端到端输出 | ⬜ 已创建，待 GPU 验证 |
| `tests/test_configs/test_example_longcat_image_t2i_fp16.yaml` | LongCat-Image 测试配置文件 | ✅ 已创建 |

---

## LongCat-Image-Edit（`LongCatImageEditTrainer`）

新增模型支持，以下内容尚未验证。

架构特点：FLUX-like MMDiT，packed 2×2 patch latents，Qwen2.5-VL 多模态文本编码器，
源图像以半分辨率通过 Qwen2VLProcessor 注入 prompt（扩展图像 token），
`hidden_states[-1][:, prefix_len:-suffix_len]`（prefix_len = `<|vision_start|>` 的位置），
源图像以全分辨率通过 VAE 编码（使用 `mode()`，`(lat - shift) * scale`），
`latent_model_input = cat([noisy_target, source_latents], dim=1)`（token 维度拼接），
img_ids：目标 modality_id=1，源 modality_id=2，偏移量 = seq_len，
model output sliced to `[:, :image_seq_len]`，`guidance=None`。

数据集格式：`image`（目标/编辑后） + `control`（源/编辑前） + `prompt`（编辑指令）。

### 训练流程

| 场景 | 配置文件 | 状态 | 备注 |
|------|---------|------|------|
| LoRA + embedding cache (cache 阶段) | `longcat_image_edit_lora_with_cache.yaml --cache` | ⬜ 未测试 | |
| LoRA + embedding cache (train 阶段) | `longcat_image_edit_lora_with_cache.yaml` | ⬜ 未测试 | |

### 推理 / Sampling

| 场景 | 状态 | 备注 |
|------|------|------|
| base model sampling (无 LoRA) | ⬜ 未测试 | |
| LongCat-Image-Edit + LoRA sampling | ⬜ 未测试 | |
| CFG sampling (`guidance_scale > 1`) | ⬜ 未测试 | |

### 自动化测试

| 测试文件 | 测试内容 | 状态 |
|---------|---------|------|
| `tests/e2e/test_longcat_image_edit_vs_diffusers.py` | 与官方 `LongCatImageEditPipeline` 对比：权重、VAE源编码、多模态 embeddings、端到端输出 | ⬜ 已创建，待 GPU 验证 |
| `tests/test_configs/test_example_longcat_image_edit_fp16.yaml` | LongCat-Image-Edit 测试配置文件 | ✅ 已创建 |

---

## Chroma T2I（`ChromaT2ITrainer`）

新增模型支持，以下内容尚未验证。

架构特点：FLUX-like ChromaTransformer2DModel，packed 2×2 patch latents，T5EncoderModel 文本编码器
（google/t5-v1_1-xxl，max_length=512），`attention_mask`（bool）传给 transformer，
text_ids 全零（FLUX 约定），img_ids `[0, row, col]`，
`num_channels_latents = in_channels // 4`，`t / 1000` 作为 timestep，
`guidance_scale=0.0` 默认（蒸馏/无分类器引导）。

### 训练流程

| 场景 | 配置文件 | 状态 | 备注 |
|------|---------|------|------|
| LoRA + embedding cache (cache 阶段) | `chroma_t2i_lora_with_cache.yaml --cache` | ⬜ 未测试 | |
| LoRA + embedding cache (train 阶段) | `chroma_t2i_lora_with_cache.yaml` | ⬜ 未测试 | |

### 推理 / Sampling

| 场景 | 状态 | 备注 |
|------|------|------|
| base model sampling (无 LoRA) | ⬜ 未测试 | `guidance_scale=0.0` for distilled |
| Chroma + LoRA sampling | ⬜ 未测试 | |
| CFG sampling (`guidance_scale > 1`) | ⬜ 未测试 | |

### 自动化测试

| 测试文件 | 测试内容 | 状态 |
|---------|---------|------|
| `tests/e2e/test_chroma_t2i_vs_diffusers.py` | 与官方 `ChromaPipeline` 对比：权重、T5 embeddings + attention_mask、端到端输出 | ⬜ 已创建，待 GPU 验证 |
| `tests/test_configs/test_example_chroma_t2i_fp16.yaml` | Chroma 测试配置文件 | ✅ 已创建 |

---

## Bria FIBO T2I（`BriaFiboT2ITrainer`）

新增模型支持，以下内容尚未验证。

架构特点：BriaFiboTransformer2DModel，AutoencoderKLWan（5D 因果 VAE，`vae_scale_factor=16`，
per-channel mean/std normalization），SmolLM3 文本编码器（last 2 hidden layers 拼接 → 4096-dim，
同时传入所有层 per-layer conditioning），`FlowMatchEulerDiscreteScheduler` + dynamic shift，
latents packed 为序列 `(B, H*W, C)`（无 2×2 packing），
attention mask 转为 `(B, 1, seq, seq)` 矩阵（einsum + 0/-inf）。

注意：`text_encoder_layers`（所有层隐状态）过大无法缓存，训练时每步重新运行文本编码器。

### 训练流程

| 场景 | 配置文件 | 状态 | 备注 |
|------|---------|------|------|
| LoRA + embedding cache (cache 阶段) | `bria_fibo_t2i_lora_with_cache.yaml --cache` | ⬜ 未测试 | |
| LoRA + embedding cache (train 阶段) | `bria_fibo_t2i_lora_with_cache.yaml` | ⬜ 未测试 | 文本编码器保留在 GPU |

### 自动化测试

| 测试文件 | 测试内容 | 状态 |
|---------|---------|------|
| `tests/e2e/test_bria_fibo_t2i_vs_diffusers.py` | 与官方 `BriaFiboPipeline` 对比：权重、SmolLM3 embeddings、端到端输出 | ⬜ 已创建，待 GPU 验证 |
| `tests/test_configs/test_example_bria_fibo_t2i_fp16.yaml` | Bria FIBO T2I 测试配置文件 | ✅ 已创建 |

---

## Bria FIBO Edit（`BriaFiboEditTrainer`）

新增模型支持，以下内容尚未验证。

架构特点：继承 `BriaFiboT2ITrainer`，新增源图像条件：源图像 VAE 编码（`.latent_dist.mean` + 归一化）
→ packed 序列，`img_ids[..., 0] = 1`（源图像标记），
model input = `cat([noisy_target, source_latents], dim=1)`，
output sliced to `[:, :target_seq_len, :]`。

### 训练流程

| 场景 | 配置文件 | 状态 | 备注 |
|------|---------|------|------|
| LoRA + embedding cache (cache 阶段) | `bria_fibo_edit_lora_with_cache.yaml --cache` | ⬜ 未测试 | |
| LoRA + embedding cache (train 阶段) | `bria_fibo_edit_lora_with_cache.yaml` | ⬜ 未测试 | |

### 自动化测试

| 测试文件 | 测试内容 | 状态 |
|---------|---------|------|
| `tests/e2e/test_bria_fibo_edit_vs_diffusers.py` | 与官方 `BriaFiboEditPipeline` 对比：权重、VAE源编码、embeddings、端到端输出 | ⬜ 已创建，待 GPU 验证 |
| `tests/test_configs/test_example_bria_fibo_edit_fp16.yaml` | Bria FIBO Edit 测试配置文件 | ✅ 已创建 |

---

## Sana Sprint T2I（`SanaSprintT2ITrainer`）

新增模型支持，以下内容尚未验证。

架构特点：SanaTransformer2DModel，4D latents（非 packed），AutoencoderDC（`vae_scale_factor=32`），
Gemma2 文本编码器（max_length=300，select_index=[BOS]+最后299个 token），
`DPMSolverMultistepScheduler`，SCM 时间步变换（`scm_t = sin(t)/(cos(t)+sin(t))`），
`lmi = (noisy/sigma_data) * sqrt(scm_t²+(1-scm_t)²)`，SCM 一致训练目标，
`guidance = distilled_guidance_scale * ones(B)`（蒸馏引导嵌入）。

### 训练流程

| 场景 | 配置文件 | 状态 | 备注 |
|------|---------|------|------|
| LoRA + embedding cache (cache 阶段) | `sana_sprint_t2i_lora_with_cache.yaml --cache` | ⬜ 未测试 | |
| LoRA + embedding cache (train 阶段) | `sana_sprint_t2i_lora_with_cache.yaml` | ⬜ 未测试 | |

### 推理 / Sampling

| 场景 | 状态 | 备注 |
|------|------|------|
| base model sampling (无 LoRA) | ⬜ 未测试 | `guidance_scale=5.0`，~20 步 DPM |
| Sana Sprint + LoRA sampling | ⬜ 未测试 | |
| CFG sampling (`guidance_scale > 1`) | ⬜ 未测试 | |

### 自动化测试

| 测试文件 | 测试内容 | 状态 |
|---------|---------|------|
| `tests/e2e/test_sana_sprint_t2i_vs_diffusers.py` | 与官方 `SanaSprintPipeline` 对比：权重、Gemma2 embeddings + attention_mask、端到端输出（SCM 推理） | ⬜ 已创建，待 GPU 验证 |
| `tests/test_configs/test_example_sana_sprint_t2i_fp16.yaml` | Sana Sprint 测试配置文件 | ✅ 已创建 |

---

## FLUX Kontext（`FluxKontextLoraTrainer`）


| 场景              | 状态       | 备注                      |
| --------------- | -------- | ----------------------- |
| BF16 DDP 训练     | ✅ 已验证    |                         |
| FP4 DDP 训练      | ✅ 已验证    | 25GB 显存，0.4 FPS         |
| BF16 FSDP 训练    | 🔄 进行中   | 见 `docs/TODO.md`        |
| FSDP + LoRA 兼容性 | 🔄 进行中   | 见 `docs/TODO.md`        |
| E2E loss 测试     | ✅ 有自动化测试 | `test_flux_loss.py`     |
| E2E sampling 测试 | ✅ 有自动化测试 | `test_flux_sampling.py` |


---

## 通用基础设施


| 项目                                       | 状态    | 备注                                         |
| ---------------------------------------- | ----- | ------------------------------------------ |
| 多分辨率混合训练                                 | ✅ 已验证 | v3.0.0                                     |
| DreamOmni2 trainer                       | ✅ 有实现 | ⬜ 无 E2E 测试                                 |
| `pretrained_embeddings.text_encoder` 配置项 | ⬜ 未测试 | 本次新增，两个 trainer 均已支持                       |
| 离线下载 + 本地路径训练（无网络）                       | ⬜ 未测试 | 见 `configs/qwen2511_lora_local_model.yaml` |


---

## 操作记录




| 日期  | 测试内容 | GPU | 结果  | 记录人 |
| --- | ---- | --- | --- | --- |
| —   | —    | —   | —   | —   |


---

## 如何更新本文档

1. 完成某个场景的测试后，将 `⬜ 未测试` 改为 `✅ 已验证` 或 `❌ 失败`。
2. 失败时在"备注"列补充错误信息或 issue 链接。
3. 新增训练配置 / 功能时，在对应表格添加新行。
4. 在"操作记录"表追加一行，注明日期、GPU 型号和测试结论。

