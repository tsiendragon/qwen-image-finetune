1. import peft error: `ImportError: /home/lilong/.conda/envs/py312/lib/python3.12/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12`

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --force-reinstall --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision torchaudio
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```
reinstall flash atten
```bash
export CUDA_HOME=/usr/local/cuda-12.1   # 按实际改
export TORCH_CUDA_ARCH_LIST=8.0
pip install --no-binary=flash-attn --no-build-isolation --no-cache-dir -v --force-reinstall flash-attn==<match you torch version>
```

2. opencv import error: `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0    # 最小集
# 如果还报 GUI/Qt 相关，再装：
sudo apt-get install -y libxext6 libxrender1 libsm6
```

3. Diffuser import: `ImportError: cannot import name 'Qwen2_5_VLForConditionalGeneration' from 'transformers' (/home/lilong/.conda/envs/py312/lib/python3.12/site-packages/transformers/__init__.py)`

update to latest version
```bash
pip install diffusers==0.35.1
pip install transformers==4.55.4
```

4. CUDA DEVICE: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0! (when checking argument for argument index in method wrapper_CUDA__index_select)`

When multi GPU visible `CUDA_VISIBLE_DEVICES=0,1`, got this problem.

If truns out to be the accelrate hooks problem.
```python
print("hook?", getattr(self.text_encoder, "_hf_hook", None))
>>hook? AlignDevicesHook(execution_device=0, offload=False, io_same_device=True, offload_buffers=False, place_submodules=True, skip_keys='past_key_values')
```

One possible solution
```python
self.text_encoder = load_qwenvl(
    self.config.model.pretrained_model_name_or_path,
    weight_dtype=self.weight_dtype
)
```
Load the qwen-vl individually because the pipeline will set up the accelerator. Did not know how to fix it yet.

5. gcc: `subprocess.CalledProcessError: Command '['/usr/bin/gcc', '/tmp/tmpxa7uy_5f/cuda_utils.c', '-O3', '-shared', '-fPIC', '-Wno-psabi', '-o', '/tmp/tmpxa7uy_5f/cuda_utils.cpython-312-x86_64-linux-gnu.so', '-lcuda', '-L/data/lilong/envs/envs/myenv/lib/python3.12/site-packages/triton/backends/nvidia/lib', '-L/lib/x86_64-linux-gnu', '-I/data/lilong/envs/envs/myenv/lib/python3.12/site-packages/triton/backends/nvidia/include', '-I/tmp/tmpxa7uy_5f', '-I/data/lilong/envs/envs/myenv/include/python3.12']' returned non-zero exit status 1.`
Export cuda path. Refer the [`docs/speed_optimization.md`](docs/speed_optimization.md)

6. vae embedding is different from the diffuser repo

Use `.cache/control.png` this image to get the vae embedding in this repo and compare with diffuser repo. Make sure parameter are same.

Parameter in diffuser
```
image torch.Size([1, 3, 512, 512])
batch_size 1
num_channels_latents 16
height 512
width 512
prompt_embeds.dtype torch.bfloat16
device cuda:0
generator None
latents None
```

Verified the parameter are same. But the latent difference is
```
torch.norm(d_image_latent - image_latents)/torch.norm(image_latents)
>>
1.1641
```

```
torch.norm(d_image_after_process - image)/torch.norm(image)
>>>
1.38
```

Seems find the reason, the normalization is different. Current implementation, normalize to [0,1] form [0,255], while diffuser normalize to [-1,1]. Let's fix it.

Now fixed the vae encoding inconsistency problem.
```
torch.norm(d_image_latent - image_latents)/torch.norm(image_latents)
>>>
tensor(0.0525, dtype=torch.bfloat16)
```

To summarize, the image process is
1. convert to RGB mode
2. resize to make width and height divisible by 16
3. channel to [B,C,H,W]
4. normalzie from [0, 255] to [-1,1]

Now test the difference of prompt embedding error.
```
torch.norm(d_pooled_prompt_embeds - pooled_prompt_embeds.cpu())
>>>tensor(0., dtype=torch.bfloat16)

torch.norm(d_prompt_embeds - prompt_embeds.cpu())
>>>tensor(0., dtype=torch.bfloat16)

torch.norm(dimage_ids - text_ids.cpu())
>>>tensor(0., dtype=torch.bfloat16)
```
Now the embedding parts should be consistant to the diffuser repo. Let train it again.

7. Lora finetune on Flux has no effect
We are train on the face segmentation mask, expected output to be
![](images/result_lora_model.jpg)
But we got.
![alt text](images/image-5.png)

We found a error as indicate in 6. Lets train it again, to see if this resolves the problem.

- Found that the face is not upright while the face mask is all upright. Fixed it. Change resolution to what used in Qwen-Image-Edit Training

8. After refactor of the code with BaseTrainer. The lora finetuning output is noise
![alt text](images/image-13.png)
![alt text](images/image-12.png)
The loss is increasing after several steps.
There must be some problem in training steps. We need to check why this happened. Predict step is good. Also indicate decode step is correct.

The problem maybe occured in cache or fit step.
In cache step, we need to find out whether the cache embedding is correct compared with pipeline processed embeddings.
In the training step, need to make sure every step is correct.

- First Debug the cache step, check if the cached control latents, and prompt embedding is same as that in diffuser pipeline
  - run cache:
    in cache function, choose one example, save the processed image, save to png.
    Load this png in diffuser pipeline. Save different embeddings
    Compare with the cached embedings with that save in diffusers
-

It turns out to be the fp4 training problem. When switch to fp16 training, it seems correct
![alt text](images/image-14.png)
400 steps Lora finetune get
![alt text](images/image-15.png)

fp4 training, got this problem may caused by the larger learning rate. Let me chage to
`0.00002` instead.
![alt text](images/image-16.png)
The loss becomes stable.



We also find that the `T5` encode in fp4 `eramth/flux-kontext-4bit-fp4` has big differences then that of the fp16 version `black-forest-labs/FLUX.1-Kontext-dev`. We may only load the `dit` module in fp4, and load other module with `fp16`


9. Mix use of fp4 and fp16 models. Load the lora trained in fp16 dit,  and inference used in fp4 dit, fp16 image encoder, fp16 text encoder. Found that got Checkerboard pattern
![alt text](images/image-17.png)


10. Multi-Resolution branch training is not good enough
Trained with 3683 steps
![alt text](images/060005_4_023834_FEMALE_33_generate.png)
![alt text](images/060028_4_022580_NONE_25_generate.png)

**Testings**:
- [x] Use old code to train the faceseg dataset, performance good
- [x] Add unittest for sampling steps
- [ ] Add unittest for loss calculation
  - have little differences in loss calculated 0.66 vs 0.64. While using old flux_transformer, the loss is almost zero

Use old code, use two different shape to construct two set of
- image latent, control latents, prompt embedding, loss
Then we
- test the loss with prediction of dit model for each sample without multi-resolution mode
- test the loss with prediction of dit model for each sample with multi-resolution mode.
The loss should be same for each samples in each mode

**Conclusion**:
- Verified sampling step is 100\% correct
- Fixed resolution Traninig is correct in the new code

**test step by step to check which step got the differences**
- hidden_states input is same
- norm_hidden_states got difference after the norm layer
- found that temb has difference, one possible reason maybe the model weight differences
Next to check
```python

class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, guidance, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        return timesteps_proj
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)

        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))  # (N, D)

        time_guidance_emb = timesteps_emb + guidance_emb

        pooled_projections = self.text_embedder(pooled_projection)
        conditioning = time_guidance_emb + pooled_projections

        return conditioning
```
which introduced differences.
First check if the inputs got differences including timestep, guidance, pooled_projections

Input got no Difference
- Next check timesteps_proj, has not difference
- Next check timesteps_emb, got error, check sample
- Check TimestepEmbedding.
  In TimestepEmbedding, the input is same, condition is all None. Linaer_1 leads to difference, this is caused by the bf16. change to float32, no difference

11. pre-commit error:

```bash
 Traceback (most recent call last):
  File "/mnt/dgx3-raid/lilong/data/.cache/pre-commit/repoap5dihi6/py_env-python3.12/bin/mypy", line 7, in <module>
    sys.exit(console_entry())
             ^^^^^^^^^^^^^^^
  File "/mnt/dgx3-raid/lilong/data/.cache/pre-commit/repoap5dihi6/py_env-python3.12/lib/python3.12/site-packages/mypy/__main__.py", line 15, in console_entry
    main()
  File "mypy/main.py", line 127, in main
  File "mypy/main.py", line 211, in run_build
  File "mypy/build.py", line 194, in build
  File "mypy/build.py", line 270, in _build
  File "mypy/build.py", line 2976, in dispatch
  File "mypy/build.py", line 3381, in process_graph
  File "mypy/build.py", line 3468, in process_fresh_modules
  File "mypy/build.py", line 2129, in load_tree
  File "mypy/nodes.py", line 414, in deserialize
  File "mypy/nodes.py", line 4650, in deserialize
  File "mypy/nodes.py", line 4547, in deserialize
  File "mypy/nodes.py", line 268, in deserialize
  File "mypy/nodes.py", line 1315, in deserialize
KeyError: 'setter_type'
```
这是 mypy 自身版本/缓存不一致 导致的内部崩溃。KeyError: 'setter_type' 出现在反序列化 AST 时，通常是：用 不同版本的 mypy（或不同 Python 版本）读取了 旧的 .mypy_cache。解决就是把缓存清掉并固定版本，必要时关掉增量模式。
```
rm -rf .mypy_cache .cache/mypy
pre-commit clean
pkill -f dmypy || true
```
