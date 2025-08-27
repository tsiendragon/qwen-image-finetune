from diffusers import AutoencoderKLQwenImage, QwenImageEditPipeline
from src.models.transformer_qwenimage import QwenImageTransformer2DModel


def load_vae(pretrained_model_name_or_path, weight_dtype):
    vae = AutoencoderKLQwenImage.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
        use_safetensors=True  # 使用 safetensors 格式，加载更快
    )
    return vae


def load_qwenvl(pretrained_model_name_or_path, weight_dtype):
    text_encoding_pipeline = QwenImageEditPipeline.from_pretrained(
        pretrained_model_name_or_path,
        transformer=None,
        vae=None,
        torch_dtype=weight_dtype,
        use_safetensors=True  # 使用 safetensors 格式，加载更快
    )
    return text_encoding_pipeline


def load_transformer(pretrained_model_name_or_path, weight_dtype):
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        use_safetensors=True  # 使用 safetensors 格式，加载更快
    )
    return flux_transformer


if __name__ == "__main__":
    import torch
    transformer = load_transformer("Qwen/Qwen-Image-Edit", torch.bfloat16)

    vae = load_vae("Qwen/Qwen-Image-Edit", torch.bfloat16)

    qwen_vl = load_qwenvl("Qwen/Qwen-Image-Edit", torch.bfloat16)

    transformer2 = load_transformer("ovedrive/qwen-image-edit-4bit", torch.bfloat16)

    from peft import LoraConfig

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights=True,
        target_modules= ["to_k", "to_q", "to_v", "to_out.0"],
    )

    transformer2.add_adapter(lora_config)
    transformer2.requires_grad_(False)
    for p in transformer2.parameters():
        p.requires_grad_(False)
    for n, p in transformer2.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
    print("number of parameters:", sum(p.numel() for p in transformer2.parameters()))
    print("number of trainable parameters:", sum(p.numel() for p in transformer2.parameters() if p.requires_grad))

    # from diffusers.models.attention_processor import LoRAAttnAddedKVProcessor
    # import torch

    # rank = 16
    # alpha = 32

    # # 给 transformer 里所有带 set_attn_processor 的注意力模块装 LoRA
    # for name, module in transformer2.named_modules():
    #     if hasattr(module, "set_attn_processor"):
    #         module.set_attn_processor(
    #             LoRAAttnAddedKVProcessor(
    #                 hidden_size=module.to_q.in_features,  # 关键：直接读到投影维度
    #                 rank=rank, lora_alpha=alpha
    #             )
    #         )

    # # 冻结基座，仅训 LoRA 参数
    # for p in transformer2.parameters():
    #     p.requires_grad_(False)

    # print number of trainable parameters
    print("number of parameters:", sum(p.numel() for p in transformer.parameters()))

    print("number of parameters:", sum(p.numel() for p in transformer2.parameters()))

    print("number of trainable parameters:", sum(p.numel() for p in transformer2.parameters() if p.requires_grad))

    from peft import LoraConfig, get_peft_model
    suffixes = [
        "attn.to_q", "attn.to_k", "attn.to_v",
        "attn.to_out.0",              # 只打到 Linear，不是整个 to_out
        # "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
        # "attn.to_add_out",          # 仅当它是 Linear 时再放开（见下方自检）
    ]

    peft_cfg = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0,
        target_modules=suffixes, bias="none",
        task_type="FEATURE_EXTRACTION",   # 或 CAUSAL_LM，均可
    )
    # transformer2.add_adapter(lora_config)
    transformer2 = get_peft_model(transformer2, peft_cfg)
    transformer2.requires_grad_(False)
    for p in transformer2.parameters():
        p.requires_grad_(False)
    for n, p in transformer2.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
    print("number of trainable parameters:", sum(p.numel() for p in transformer2.parameters() if p.requires_grad))

    # # compare the keys of transformer and transformer2
    # print("transformer keys not in transformer2:", set(transformer.state_dict().keys()) - set(transformer2.state_dict().keys()))
    # print("transformer2 keys not in transformer:", set(transformer2.state_dict().keys()) - set(transformer.state_dict().keys()))



