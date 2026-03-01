# Training & Test Status

Track which training runs and automated tests have been executed and verified.
Update this file after each test run. Add notes with GPU model, date, and any issues.

---

## Qwen-Image-Edit-2511 (`QwenImageEditPlusTrainer`)

新增模型支持，以下内容尚未验证。

### 训练流程

| 场景 | 配置文件 | 状态 | 备注 |
|------|---------|------|------|
| LoRA + embedding cache (cache 阶段) | `qwen2511_lora_with_cache.yaml --cache` | ⬜ 未测试 | |
| LoRA + embedding cache (train 阶段) | `qwen2511_lora_with_cache.yaml` | ⬜ 未测试 | |
| LoRA + 无 cache | `qwen2511_lora_no_cache.yaml` | ⬜ 未测试 | |
| LoRA + 本地模型离线训练 (cache 阶段) | `qwen2511_lora_local_model.yaml --cache` | ⬜ 未测试 | 需提前下载模型 |
| LoRA + 本地模型离线训练 (train 阶段) | `qwen2511_lora_local_model.yaml` | ⬜ 未测试 | |
| 全层 LoRA (`all-linear`, r=64) | `qwen2511_lora_full_coverage.yaml` | ⬜ 未测试 | 显存占用待测 |
| 全参数微调（无 LoRA） | `qwen2511_full_finetune.yaml` | ⬜ 未测试 | 需 ~80GB+ 显存；输出 `transformer.safetensors` |
| 多 GPU 训练 (DDP) | 任意 2511 config + multi-GPU accelerate | ⬜ 未测试 | |
| 从已有 LoRA checkpoint 恢复训练 | `lora.pretrained_weight` 字段 | ⬜ 未测试 | |

### 推理 / Sampling

| 场景 | 状态 | 备注 |
|------|------|------|
| 2511 base model sampling (无 LoRA) | ⬜ 未测试 | |
| 2511 + LoRA sampling | ⬜ 未测试 | |
| `true_cfg_scale > 1` (negative prompt CFG) | ⬜ 未测试 | |
| 多控制图像输入 (`condition_images` list) | ⬜ 未测试 | |

### Bug 修复验证

以下 bug 已在代码中修复，但尚未有自动化测试覆盖：

| Bug | 修复位置 | 自动化测试 | 备注 |
|-----|---------|-----------|------|
| `enumerate[Tensor](image)` 语法错误 | `qwen_image_edit_plus_trainer.py:339` | ⬜ 无测试 | 修复前会在运行时崩溃 |
| `prepare_embeddings` negative_prompt 解包错误 | `qwen_image_edit_plus_trainer.py:199` | ⬜ 无测试 | `debug=False` 时解包 4 个返回值但函数只返回 2 个 |
| `load_qwenvl` 忽略 path 参数（硬编码路径） | `load_model.py:load_qwenvl` | ⬜ 无测试 | 现在使用传入的路径 |
| VAE/text encoder 硬编码为 `Qwen-Image-Edit-2509` | `qwen_image_edit_plus_trainer.py:load_model` | ⬜ 无测试 | 现在使用 `pretrained_model_name_or_path` |

### 自动化测试（需新增）

| 测试文件（待创建） | 测试内容 | 优先级 |
|-------------------|---------|--------|
| `tests/e2e/test_qwen_2511_sampling.py` | 2511 模型 sampling 端到端测试 | 高 |
| `tests/src/trainer/test_qwen_image_edit_plus_trainer.py` | `encode_prompt`、`_get_qwen_prompt_embeds`、`prepare_embeddings` 单元测试 | 高 |
| `tests/src/models/test_load_model.py` | `load_qwenvl` 路径参数实际生效测试 | 中 |
| `tests/src/trainer/test_qwen_2511_negative_prompt.py` | `debug=False` 时 negative_prompt 流程测试 | 中 |

### 测试资源（待上传 HuggingFace）

参见 `tests/resources_config.yaml`，以下资源 size 标注为 TBD，尚未上传：

| Resource Group | 用途 | 状态 |
|----------------|------|------|
| `qwen_sampling` | `test_qwen_image_edit_sampling.py` | ⬜ TBD（未上传） |
| `qwen_plus_sampling` | `test_qwen_image_edit_plus_sampling.py` | ⬜ TBD（未上传） |
| `qwen_2511_sampling` | 2511 e2e 采样测试（待创建） | ⬜ 未创建 |

---

## Qwen-Image-Edit（原始版本 `QwenImageEditTrainer`）

| 场景 | 状态 | 备注 |
|------|------|------|
| LoRA + cache 训练 | ✅ 已验证 | `face_seg_config.yaml` |
| LoRA + FP4 量化训练 | ✅ 已验证 | `face_seg_fp4_config.yaml` |
| E2E sampling 测试 | ✅ 有自动化测试 | `test_qwen_image_edit_sampling.py`（资源待上传） |
| 本地模型 + `pretrained_embeddings.text_encoder` 路径 | ⬜ 未测试 | 本次新增功能，旧版 trainer 同样支持 |

---

## FLUX Kontext（`FluxKontextLoraTrainer`）

| 场景 | 状态 | 备注 |
|------|------|------|
| BF16 DDP 训练 | ✅ 已验证 | |
| FP4 DDP 训练 | ✅ 已验证 | 25GB 显存，0.4 FPS |
| BF16 FSDP 训练 | 🔄 进行中 | 见 `docs/TODO.md` |
| FSDP + LoRA 兼容性 | 🔄 进行中 | 见 `docs/TODO.md` |
| E2E loss 测试 | ✅ 有自动化测试 | `test_flux_loss.py` |
| E2E sampling 测试 | ✅ 有自动化测试 | `test_flux_sampling.py` |

---

## 通用基础设施

| 项目 | 状态 | 备注 |
|------|------|------|
| 多分辨率混合训练 | ✅ 已验证 | v3.0.0 |
| DreamOmni2 trainer | ✅ 有实现 | ⬜ 无 E2E 测试 |
| `pretrained_embeddings.text_encoder` 配置项 | ⬜ 未测试 | 本次新增，两个 trainer 均已支持 |
| 离线下载 + 本地路径训练（无网络） | ⬜ 未测试 | 见 `configs/qwen2511_lora_local_model.yaml` |

---

## 操作记录

<!-- 每次跑完训练 / 测试后，在此追加一行记录 -->

| 日期 | 测试内容 | GPU | 结果 | 记录人 |
|------|---------|-----|------|--------|
| — | — | — | — | — |

---

## 如何更新本文档

1. 完成某个场景的测试后，将 `⬜ 未测试` 改为 `✅ 已验证` 或 `❌ 失败`。
2. 失败时在"备注"列补充错误信息或 issue 链接。
3. 新增训练配置 / 功能时，在对应表格添加新行。
4. 在"操作记录"表追加一行，注明日期、GPU 型号和测试结论。
