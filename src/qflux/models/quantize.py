import logging
from typing import Any

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


def quantize_model_to_fp8(
    model: nn.Module,
    engine: str = "te",  # "te" or "bnb"
    device: str = "cuda",
    quantize_config: dict[str, Any] | None = None,
    skip_modules: list[str] | None = None,
    verbose: bool = True,
) -> nn.Module:
    """
    将模型量化为FP8（TE）或INT8（BnB）

    Args:
        model: 输入的PyTorch模型
        engine: 量化引擎，"te"（Transformer Engine）或"bnb"（BitsAndBytes）
        device: 设备类型
        quantize_config: 量化配置字典
        skip_modules: 需要跳过量化的模块名称列表
        verbose: 是否打印详细信息

    Returns:
        量化后的模型

    Example:
        >>> model = YourDiffusionModel()
        >>> # 使用Transformer Engine
        >>> quantized_model = quantize_model_to_fp8(model, engine="te")
        >>> # 使用BitsAndBytes
        >>> quantized_model = quantize_model_to_fp8(model, engine="bnb")
    """

    # 默认配置
    default_configs = {
        "te": {
            "fp8_format": "HYBRID",  # E4M3 for forward, E5M2 for backward
            "margin": 0,
            "interval": 1,
            "amax_history_len": 1024,
            "amax_compute_algo": "max",
            "override_linear_precision": (torch.float8_e4m3fn, torch.float8_e5m2),
        },
        "bnb": {
            "load_in_8bit": True,
            "llm_int8_threshold": 6.0,
            "llm_int8_enable_fp32_cpu_offload": False,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_quant_type": "nf4",  # 如果要用4bit
            "use_4bit": False,  # 默认使用8bit
        },
    }

    # 合并用户配置
    config = default_configs.get(engine, {})
    if quantize_config:
        config.update(quantize_config)

    # 需要跳过的模块
    skip_modules = skip_modules or []

    if engine.lower() == "te":
        quantized_model = _quantize_with_transformer_engine(model, config, skip_modules, device, verbose)
    elif engine.lower() == "bnb":
        quantized_model = _quantize_with_bitsandbytes(model, config, skip_modules, device, verbose)
    else:
        raise ValueError(f"Unsupported engine: {engine}. Choose 'te' or 'bnb'")

    if verbose:
        print(f"✅ Model quantized successfully using {engine.upper()}")
        _print_model_size(quantized_model)

    return quantized_model


def _quantize_with_transformer_engine(
    model: nn.Module, config: dict[str, Any], skip_modules: list[str], device: str, verbose: bool
) -> nn.Module:
    """使用Transformer Engine进行FP8量化"""
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe
    except ImportError:
        raise ImportError("Transformer Engine not installed. Install with: pip install transformer-engine")

    # 检查GPU支持
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Transformer Engine")

    gpu_capability = torch.cuda.get_device_capability()
    if gpu_capability[0] < 8:  # 需要计算能力8.0+（A100及以上）
        logger.warning(
            f"GPU compute capability {gpu_capability} may not fully support FP8. Recommended: 8.0+ (A100/H100)"
        )

    # 创建FP8配方
    fp8_recipe = recipe.DelayedScaling(
        margin=config.get("margin", 0),
        interval=config.get("interval", 1),
        fp8_format=getattr(recipe.Format, config.get("fp8_format", "HYBRID")),
        amax_history_len=config.get("amax_history_len", 1024),
        amax_compute_algo=config.get("amax_compute_algo", "max"),
        override_linear_precision=config.get("override_linear_precision", None),
    )

    # 移动模型到设备
    model = model.to(device)

    # 替换Linear层为FP8 Linear
    def replace_linear_with_fp8(module, name_prefix=""):
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name

            # 检查是否需要跳过
            if any(skip_name in full_name for skip_name in skip_modules):
                if verbose:
                    logger.info(f"Skipping quantization for: {full_name}")
                continue

            if isinstance(child, nn.Linear):
                # 创建FP8 Linear层
                fp8_linear = te.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    device=device,
                )

                # 复制权重（TE会自动处理FP8转换）
                with torch.no_grad():
                    fp8_linear.weight.copy_(child.weight)
                    if child.bias is not None:
                        fp8_linear.bias.copy_(child.bias)

                # 设置FP8配方
                fp8_linear.fp8_recipe = fp8_recipe

                # 替换模块
                setattr(module, name, fp8_linear)

                if verbose:
                    logger.debug(f"Quantized {full_name} to FP8")
            else:
                # 递归处理子模块
                replace_linear_with_fp8(child, full_name)

    # 执行替换
    replace_linear_with_fp8(model)

    # 添加FP8 autocast上下文管理器
    original_forward = model.forward

    def fp8_forward(*args, **kwargs):
        with te.fp8_autocast(enabled=True):
            return original_forward(*args, **kwargs)

    model.forward = fp8_forward

    return model


def _quantize_with_bitsandbytes(
    model: nn.Module, config: dict[str, Any], skip_modules: list[str], device: str, verbose: bool
) -> nn.Module:
    """使用BitsAndBytes进行INT8/NF4量化"""
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear4bit, Linear8bitLt
    except ImportError:
        raise ImportError("BitsAndBytes not installed. Install with: pip install bitsandbytes")

    # 移动模型到设备
    model = model.to(device)

    use_4bit = config.get("use_4bit", False)

    if use_4bit:
        # 4-bit量化（NF4）
        if verbose:
            logger.info("Using 4-bit NF4 quantization")

        def replace_linear_with_nf4(module, name_prefix=""):
            for name, child in module.named_children():
                full_name = f"{name_prefix}.{name}" if name_prefix else name

                # 检查是否需要跳过
                if any(skip_name in full_name for skip_name in skip_modules):
                    if verbose:
                        logger.info(f"Skipping quantization for: {full_name}")
                    continue

                if isinstance(child, nn.Linear):
                    # 创建4bit Linear层
                    nf4_linear = Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=config.get("bnb_4bit_compute_dtype", torch.float16),
                        compress_statistics=True,
                        quant_type=config.get("bnb_4bit_quant_type", "nf4"),
                    )

                    # 复制权重
                    with torch.no_grad():
                        nf4_linear.weight = bnb.nn.Params4bit(
                            child.weight.data.clone(),
                            requires_grad=False,
                            quant_type=config.get("bnb_4bit_quant_type", "nf4"),
                        )
                        if child.bias is not None:
                            nf4_linear.bias.copy_(child.bias)

                    # 替换模块
                    setattr(module, name, nf4_linear)

                    if verbose:
                        logger.debug(f"Quantized {full_name} to 4-bit NF4")
                else:
                    # 递归处理子模块
                    replace_linear_with_nf4(child, full_name)

        replace_linear_with_nf4(model)

    else:
        # 8-bit量化（INT8）
        if verbose:
            logger.info("Using 8-bit INT8 quantization")

        def replace_linear_with_int8(module, name_prefix=""):
            for name, child in module.named_children():
                full_name = f"{name_prefix}.{name}" if name_prefix else name

                # 检查是否需要跳过
                if any(skip_name in full_name for skip_name in skip_modules):
                    if verbose:
                        logger.info(f"Skipping quantization for: {full_name}")
                    continue

                if isinstance(child, nn.Linear):
                    # 创建INT8 Linear层
                    int8_linear = Linear8bitLt(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        has_fp16_weights=False,
                        threshold=config.get("llm_int8_threshold", 6.0),
                    )

                    # 复制权重
                    with torch.no_grad():
                        int8_linear.weight = bnb.nn.Int8Params(child.weight.data.clone(), requires_grad=False)
                        if child.bias is not None:
                            int8_linear.bias.copy_(child.bias)

                    # 替换模块
                    setattr(module, name, int8_linear)

                    if verbose:
                        logger.debug(f"Quantized {full_name} to INT8")
                else:
                    # 递归处理子模块
                    replace_linear_with_int8(child, full_name)

        replace_linear_with_int8(model)

    return model


def _print_model_size(model: nn.Module):
    """打印模型大小信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 估算模型大小（字节）
    model_size = 0
    for param in model.parameters():
        if param.dtype == torch.float32:
            model_size += param.numel() * 4
        elif param.dtype == torch.float16 or param.dtype == torch.bfloat16:
            model_size += param.numel() * 2
        elif param.dtype == torch.int8 or str(param.dtype).startswith("torch.float8"):
            model_size += param.numel()
        else:
            model_size += param.numel() * 4  # 默认假设4字节

    print("Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {model_size / 1024**3:.2f} GB")


if __name__ == "__main__":
    from qflux.models.load_model import load_transformer

    model = load_transformer("Qwen/Qwen-Image-Edit", weight_dtype=torch.bfloat16)
    quantized_model = quantize_model_to_fp8(
        model,
        engine="bnb",
        verbose=True,
        device="cuda:0",
    )
    quantized_model = quantized_model.eval()
    quantized_model._requires_grad = False
    quantized_model = quantized_model.cpu()
    torch.cuda.empty_cache()
    # latent_model_input torch.Size([1, 8208, 64])

    # timestep tensor([1000.], device='cuda:1', dtype=torch.bfloat16)
    # guidance None
    # prompt_embeds_mask torch.Size([1, 646])
    # prompt_embeds torch.Size([1, 646, 3584])
    # img_shapes [[(1, 54, 76), (1, 54, 76)]]
    # txt_seq_lens [646]
    # attention_kwargs {}

    device = "cuda:1"

    weight_dtype = torch.bfloat16

    latent_model_input = torch.randn(1, 8208, 64).to(device).to(weight_dtype)
    timestep = torch.tensor([1000.0], device=device, dtype=weight_dtype)
    guidance = None
    prompt_embeds_mask = torch.randint(0, 2, (1, 646)).to(device).to(torch.int64)
    prompt_embeds = torch.randn(1, 646, 3584).to(device).to(weight_dtype)
    img_shapes = [[(1, 54, 76), (1, 54, 76)]]
    txt_seq_lens = [646]
    attention_kwargs: dict[str, Any] = {}
    model = model.to(device).to(weight_dtype)
    with torch.inference_mode():
        out = model(
            latent_model_input,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            timestep=timestep,
            guidance=guidance,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            attention_kwargs=attention_kwargs,
        )
    model = model.cpu()
    torch.cuda.empty_cache()

    device = "cuda:0"
    quantized_model = quantized_model.to(device)
    latent_model_input = latent_model_input.to(device)
    prompt_embeds = prompt_embeds.to(device)
    prompt_embeds_mask = prompt_embeds_mask.to(device)
    timestep = timestep.to(device)
    img_shapes = [[(1, 54, 76), (1, 54, 76)]]
    txt_seq_lens = [646]
    attention_kwargs = {}
    with torch.inference_mode():
        out2 = quantized_model(
            latent_model_input,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            timestep=timestep,
            guidance=guidance,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            attention_kwargs=attention_kwargs,
        )

    out = out[0].detach().cpu()
    out2 = out2[0].detach().cpu()
    # calcualte the difference between out and out2
    print(out[0].shape)
    print(out2[0].shape)
    print(out[0] - out2[0])
    print(out[0] - out2[0].to(out[0].dtype))
    print(out[0] - out2[0].to(out[0].dtype).abs())
    print(out[0] - out2[0].to(out[0].dtype).abs().mean())
    print(out[0] - out2[0].to(out[0].dtype).abs().mean())
