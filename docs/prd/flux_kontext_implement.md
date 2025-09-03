# Implement the Flux Kontext Lora Training

## 模型
类似  @src/models/load_model.py
增加一个 function 去 load 各个 function
- load clip of flux kontext
- load clip tokenizer
- load vae of flux kontext
- load t5 of flux kontext
- load t5 tokenizer
- load transformer of flux kontext
did not change original  @src/models/load_model.py

write a test function in tests/, to test whehter the loaded model is same when loaded in the following methods
```
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
clip = pipe.text_encoder
t5 = pipe.text_encoder_2
tokenizer2 = pipe.tokenizer_2
tokenizer1 = pipe.tokenizer_1
transformer = pipe.transformer
vae = pipe.vae
```

要比较参数形状是否一致，参数大小是否一致。

## 新增一个 FluxKontextLoraTrainer
新增一个FluxKontextLoraTrainer 类似
@src/qwen_image_edit_trainer.py

需要实现 cache, fit, predict 等 function
### cache
- image_latents
- control_latents
- pooled_prompt_embeds
- prompt_embeds
- empty_prompt_embeds
- empty_pooled_prompt_embeds

### predict
predict 按照这个code
https://raw.githubusercontent.com/huggingface/diffusers/refs/heads/main/src/diffusers/pipelines/flux/pipeline_flux_kontext.py
但是tokenizer, clip, t5, vae, transformer 等使用已经加载的
同时这些模块按照 config 分别加载到不同的 device-id 上

### fit
训练模块和src/qwen_image_edit_trainer.py 类似
分成 caching 模式训练和非 cache 模式训练

### 其他
需要有一个encode_prompt function
和https://raw.githubusercontent.com/huggingface/diffusers/refs/heads/main/src/diffusers/pipelines/flux/pipeline_flux_kontext.py
相同
还有
get_clip_prompt_embeds
get_t5_prompt_embeds
都借鉴https://raw.githubusercontent.com/huggingface/diffusers/refs/heads/main/src/diffusers/pipelines/flux/pipeline_flux_kontext.py

使用prepare_latents得到source 图片和 target 图片的 latent

训练的方法和 loss optimizer ，scheduler 都一样

创建一个 test 测试 trainer 里的 predict 生成的结果 和 直接使用这个 pipeline `pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)` 的差异，包括使用 cfg 和不使用 cfg 的情况，并且记录生成需要的各个步骤的时间。

创建三个 config分别用来测试训练
bf6 的模型使用这个 black-forest-labs/FLUX.1-Krea-dev
fp8 的模型使用这个 AlekseyCalvin/Flux_Kontext_Dev_fp8_scaled_diffusers
fp4 的模型使用这个 eramth/flux-kontext-4bit-fp4

## remark
- 要保持原有的 qwen-image-edit 训练不受影响
- 不要改动原来的代码
- 使用增量迭代，新的 function 可以兼容旧的 但是要保留旧的实现
- 使用 pytest 规范。tests 下的目录最好和 src 下的目录一致。每个 class 和 function 都需要 unittest.

损失函数：
- Flux Kontext使用的损失函数类型 和 Qwen 一样

- 数据预处理：
    - 图像尺寸要求可能与Qwen不同。不需要进行 resize 直接使用 dataset 里出来的尺寸
    - 需要确认latent维度和缩放因子： 使用 FluxKontext pipeline 得到这个缩放因子 和这里的代码src/qwen_image_edit_trainer.py 一致
- LoRA目标模块：
    - Flux transformer的注意力层结构可能不同，会根据 config中的    target_modules: ["to_k", "to_q", "to_v", "to_out.0"] 确定 lora 块

依赖冲突： 下面这些都没有问题
    - diffusers版本要求可能不同
    - transformers版本兼容性
    - 新的第三方依赖
