# Multi Control 功能设计文档

## 概述
多控制（Multi Control）功能允许模型同时使用多个控制图像来引导图像生成或编辑过程。控制图像是RGB图片，用于让模型参考特定信息，如布局信息、字体信息等。在Qwen-Image-Edit和Flux Kontext这样的图片到图片编辑模型中，多控制功能可以提供更精细和多样化的控制。

## 数据集设计
### 数据集输出格式
- 数据集应输出控制图像列表（List[str]）
- 不同控制图像可以有不同形状
- 第一个是主控制图像，命名为'control'
- 后续控制图像依次命名为'control_1'、'control_2'等
- 主控制图像保持现有的图像处理逻辑
- 额外的控制图像会根据配置调整大小

### 文件命名约定
数据集中的多控制图像应遵循以下命名约定：
- 主控制图像：`xyz.jpg/png/jpeg`
- 额外控制图像：`xyz_control_1.jpg/png/jpeg`、`xyz_control_2.jpg/png/jpeg`等（使用 _control_N 格式）
- 系统会自动收集并排序这些额外控制图像

## 配置选项
### 控制图像选择
配置文件中可以指定以下参数：
1. **控制图像索引选择**：通过索引数组选择需要的控制图像
   - 例如，如果只指定`[2]`，则只输出第二个控制图像
   - 这允许灵活选择所需的控制图像子集

2. **目标尺寸设置**：为每个选定的控制图像指定目标大小
   - 如果尺寸不匹配，数据集会调整图像大小并保持比例
   - 使用零填充到指定的精确尺寸
   - 这确保了所有控制图像可以被正确处理

## 缓存机制
### 多控制图像的缓存
1. 缓存所有选定的控制图像（如果存在）
   - 主控制图像缓存为`control_latent`
   - 额外控制图像缓存为`control_latent_1`、`control_latent_2`等
   - 使用与当前控制潜在缓存相似的方法

### 缓存实现
- 使用`EmbeddingCacheManager`管理所有缓存
- 为每个控制图像生成唯一哈希值
- 缓存存储在指定目录结构中，便于检索

## 控制嵌入处理
### 编码流程
1. 在Qwen-Image-Edit中，只有主控制图像('control')通过提示编码器处理
2. 多控制图像将通过VAE编码器编码为额外的潜在表示
3. 噪声潜在表示将与所有这些控制潜在表示连接在一起，共同输入到扩散变换器
4. 使用不同的image-id来识别不同的控制信息

### 技术实现
- 对每个控制图像依次调用`prepare_latents`函数进行处理
- 每个控制图像都会通过VAE编码器生成对应的潜在表示
- 使用`_pack_latents`和`_unpack_latents`函数处理每个图像的潜在表示
- 通过`_prepare_latent_image_ids`为每个控制图像生成唯一ID

## 应用场景
多控制功能适用于需要同时考虑多种参考信息的复杂图像编辑任务，例如：
- 同时控制图像的布局和风格
- 结合多种视觉元素进行精确编辑
- 在保持某些区域不变的同时修改其他区域

## 与现有系统集成
本功能设计与现有的Qwen-Image-Edit和Flux Kontext模型架构兼容，只需要对数据处理和模型输入部分进行扩展，不需要对核心模型架构进行重大修改。

## 技术设计方案

### 需要修改的组件

#### 1. 数据集模块修改
**文件路径**: `src/data/dataset.py`

**修改内容**:
- 扩展`ImageDataset`类的`__getitem__`方法，支持返回多个控制图像
- 修改`_scan_image_files`方法，增强对额外控制图像的收集逻辑

```python
# 现有的收集控制图像的函数已经支持多控制图像
# 在_collect_extra_controls函数中，已经实现了对xyz_control_1.png、xyz_control_2.png等的收集
# 只需确保在__getitem__中正确处理这些额外的控制图像
```

#### 2. 缓存管理器扩展
**文件路径**: `src/data/cache_manager.py`

使用什么hash_key

使用control 图片文件的 md5 作为 hash_key
计算 embedding 的时候，如果这个 hash_key 已经存在了，就不需要了

#### 3. 训练器模块修改
**文件路径**:
- `src/trainer/qwen_image_edit_trainer.py`
- `src/trainer/flux_kontext_trainer.py`

**共同修改内容**:
- 扩展`cache_step`方法，处理多个控制图像的缓存
- 扩展`_training_step_cached`和`_training_step_compute`方法，处理多个控制图像

```python
# 在cache_step方法中添加多控制图像处理
def cache_step(self, data: dict, vae_encoder_device: str, text_encoder_device: str, ...):
    # 现有代码处理主控制图像
    control = torch.from_numpy(data["control"][0]).unsqueeze(0)
    control = self.preprocess_image(control)
    _, control_latents, _, _ = self.prepare_latents(...)

    # 添加额外控制图像处理
    for i in range(1, len(data["control"])):
        control_i = torch.from_numpy(data["control"][i]).unsqueeze(0)
        control_i = self.preprocess_image(control_i)
        _, control_latents_i, _, _ = self.prepare_latents(...)
        # 缓存额外控制图像
        self.cache_manager.save_cache(f"control_latent_{i}", prompt_hash, control_latents_i[0].detach().cpu())
```

#### 4. 配置文件扩展
**文件路径**: 配置YAML文件

**修改内容**:
- 添加多控制图像的配置选项

```yaml
...
    processor:
      class_path: qflux.data.preprocess.ImageProcessor
      init_args:
        process_type: center_crop
        resize_mode: bilinear
        target_size: [832, 576]
        controls_size: [[832, 576]]
```

#### 5. 模型输入处理修改
**文件路径**:
- `src/trainer/qwen_image_edit_trainer.py`
- `src/trainer/flux_kontext_trainer.py`

**修改内容**:
- 修改`_compute_loss`方法，处理多个控制图像的输入

```python
def _compute_loss(self, pixel_latents, control_latents, ...):
    # 现有代码处理主控制图像

    # 处理额外控制图像
    extra_control_latents = []
    for i in range(1, len(self.config.data.multi_control.selected_indices) + 1):
        if f"control_latent_{i}" in batch:
            extra_control = batch[f"control_latent_{i}"].to(
                self.accelerator.device, dtype=self.weight_dtype
            )
            extra_control_latents.append(extra_control)

    # 连接所有控制图像的潜在表示
    all_control_latents = [control_latents] + extra_control_latents

    # 为每个控制图像准备image_ids
    all_image_ids = []
    for i, control_latent in enumerate(all_control_latents):
        image_ids = self._prepare_latent_image_ids(...)
        image_ids[..., 0] = i + 1  # 为每个控制图像设置唯一ID
        all_image_ids.append(image_ids)

    # 连接所有控制图像的潜在表示和image_ids
    combined_control_latents = torch.cat(all_control_latents, dim=1)
    combined_image_ids = torch.cat(all_image_ids, dim=0)

    # 准备模型输入
    latent_model_input = torch.cat([noisy_model_input, combined_control_latents], dim=1)
    latent_ids = torch.cat([latent_ids, combined_image_ids], dim=0)
```

#### 6. 预测函数修改
**文件路径**:
- `src/trainer/qwen_image_edit_trainer.py`
- `src/trainer/flux_kontext_trainer.py`

**修改内容**:
- 扩展`predict`方法，支持多个控制图像的输入

```python
def predict(self, prompt_image, prompt, ...):
    # 处理多个控制图像输入
    if isinstance(prompt_image, list) and len(prompt_image) > 1:
        control_images = prompt_image
    else:
        control_images = [prompt_image]

    # 处理主控制图像
    image = self.preprocess_image(control_images[0])
    _, image_latents, _, _ = self.prepare_latents(...)

    # 处理额外控制图像
    extra_latents = []
    extra_ids = []
    for i in range(1, len(control_images)):
        control_i = self.preprocess_image(control_images[i])
        _, control_latents_i, _, image_ids_i = self.prepare_latents(...)
        image_ids_i[..., 0] = i + 1  # 设置唯一ID
        extra_latents.append(control_latents_i)
        extra_ids.append(image_ids_i)

    # 连接所有控制图像的潜在表示和image_ids
    all_control_latents = [image_latents] + extra_latents
    all_image_ids = [image_ids] + extra_ids

    combined_control_latents = torch.cat(all_control_latents, dim=1)
    combined_image_ids = torch.cat(all_image_ids, dim=0)

    # 修改模型输入
    latent_model_input = torch.cat([latents, combined_control_latents], dim=1)
    latent_ids = torch.cat([latent_ids, combined_image_ids], dim=0)
```

### 实现策略

1. **渐进式实现**:
   - 先实现数据集和缓存管理器的扩展
   - 然后实现训练器的修改
   - 最后实现预测函数的修改

2. **兼容性保证**:
   - 所有修改都应向后兼容，不影响现有单控制图像的功能
   - 通过配置选项控制是否启用多控制功能

3. **代码复用**:
   - 尽可能复用现有的`prepare_latents`、`_pack_latents`等函数
   - 对于Qwen-Image-Edit和Flux Kontext的共同逻辑，提取到辅助函数中

### 风险评估

1. **性能影响**:
   - 多控制图像会增加内存使用和计算量
   - 缓解措施: 添加配置选项控制使用的控制图像数量

2. **兼容性问题**:
   - 现有的缓存可能与新格式不兼容
   - 缓解措施: 添加版本检查和兼容性处理

3. **模型行为变化**:
   - 多控制图像可能改变模型的生成行为
   - 缓解措施: 进行充分的对比测试，确保质量不下降
