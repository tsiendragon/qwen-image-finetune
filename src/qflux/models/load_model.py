import logging

import torch
from diffusers import AutoencoderKLQwenImage, QwenImageEditPipeline

from qflux.models.transformer_qwenimage import QwenImageTransformer2DModel


def load_vae(pretrained_model_name_or_path, weight_dtype):
    vae = AutoencoderKLQwenImage.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
        use_safetensors=True,  # 使用 safetensors 格式，加载更快
        device_map="cpu",  # load to cpu instead of gpu
    )
    logging.info(f"loaded vae from {pretrained_model_name_or_path} with weight_dtype {weight_dtype}")
    return vae


def load_qwenvl(pretrained_model_name_or_path, weight_dtype):
    from transformers import Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=weight_dtype,  # CPU 用 fp32
        use_safetensors=True,
        attn_implementation="flash_attention_2",
    )  # 默认就在 CPU；稳妥可再
    logging.info(f"loaded qwen_vl from {pretrained_model_name_or_path} with weight_dtype {weight_dtype}")
    return model


def load_transformer(pretrained_model_name_or_path, weight_dtype, device_map="cuda:1"):
    import logging

    logging.info(f"load model {pretrained_model_name_or_path}")
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        use_safetensors=True,  # 使用 safetensors 格式，加载更快,
        attn_implementation="flash_attention_2",
        device_map="cpu",  # load to cpu instead of gpu
    )
    logging.info(f"loaded transformer from {pretrained_model_name_or_path} with weight_dtype {weight_dtype}")
    return flux_transformer


if __name__ == "__main__":
    # pipe = QwenImageEditPipeline.from_pretrained(
    #         self.config.model.pretrained_model_name_or_path,
    #         torch_dtype=self.weight_dtype
    #     )
    pipe = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        torch_dtype=torch.bfloat16,
    )

    # Separate individual components

    # same to model constructed from vae self.vae = pipe.vae
    text_encoder = pipe.text_encoder  # text_encoder is actually qwen_vl

    qwen_vl = load_qwenvl("Qwen/Qwen-Image-Edit", torch.bfloat16)

    # Compare the two models
    print("=" * 80)
    print("COMPARING text_encoder (from pipe) vs qwen_vl (loaded separately)")
    print("=" * 80)

    def compare_models(model1, model2, model1_name="Model1", model2_name="Model2", tolerance=1e-6):
        """
        Compare two models including parameter shapes and values
        Args:
            model1, model2: Models to compare
            model1_name, model2_name: Names for logging
            tolerance: Tolerance for value differences
        """
        print(f"\n🔍 Comparing {model1_name} vs {model2_name}")
        print("-" * 60)

        # Get state dicts
        state_dict1 = model1.state_dict()
        state_dict2 = model2.state_dict()

        # Compare keys
        keys1 = set(state_dict1.keys())
        keys2 = set(state_dict2.keys())

        print("📊 Parameter Statistics:")
        print(f"  {model1_name}: {len(keys1)} parameters")
        print(f"  {model2_name}: {len(keys2)} parameters")

        # Check for missing/extra keys
        missing_in_model2 = keys1 - keys2
        missing_in_model1 = keys2 - keys1
        common_keys = keys1 & keys2

        if missing_in_model2:
            print(f"\n❌ Parameters in {model1_name} but not in {model2_name}:")
            for key in sorted(missing_in_model2):
                print(f"  - {key}")

        if missing_in_model1:
            print(f"\n❌ Parameters in {model2_name} but not in {model1_name}:")
            for key in sorted(missing_in_model1):
                print(f"  - {key}")

        print(f"\n✅ Common parameters: {len(common_keys)}")

        if not common_keys:
            print("❌ No common parameters found!")
            return False

        # Compare shapes and values for common parameters
        shape_mismatches = []
        value_differences = []
        identical_params = 0

        print(f"\n🔍 Detailed Parameter Comparison (tolerance={tolerance}):")
        print("-" * 60)

        for key in sorted(common_keys):
            param1 = state_dict1[key]
            param2 = state_dict2[key]

            # Compare shapes
            if param1.shape != param2.shape:
                shape_mismatches.append((key, param1.shape, param2.shape))
                print(f"❌ {key}: Shape mismatch - {param1.shape} vs {param2.shape}")
                continue

            # Compare values
            try:
                # Convert to same device and dtype for comparison
                param1_cpu = param1.detach().cpu().float()
                param2_cpu = param2.detach().cpu().float()

                # Calculate differences
                abs_diff = torch.abs(param1_cpu - param2_cpu)
                max_diff = torch.max(abs_diff).item()
                mean_diff = torch.mean(abs_diff).item()

                # Check if parameters are identical within tolerance
                if max_diff <= tolerance:
                    identical_params += 1
                    print(f"✅ {key}: IDENTICAL (max_diff={max_diff:.2e})")
                else:
                    value_differences.append((key, max_diff, mean_diff))
                    print(f"⚠️  {key}: DIFFERENT - max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

            except Exception as e:
                print(f"❌ {key}: Error comparing values - {e}")

        # Summary
        print("\n📋 COMPARISON SUMMARY:")
        print(f"  Total common parameters: {len(common_keys)}")
        print(f"  Identical parameters: {identical_params}")
        print(f"  Shape mismatches: {len(shape_mismatches)}")
        print(f"  Value differences: {len(value_differences)}")

        if shape_mismatches:
            print("\n❌ SHAPE MISMATCHES:")
            for key, shape1, shape2 in shape_mismatches:
                print(f"  {key}: {shape1} vs {shape2}")

        if value_differences:
            print("\n⚠️  LARGEST VALUE DIFFERENCES:")
            # Sort by max difference and show top 10
            value_differences.sort(key=lambda x: x[1], reverse=True)
            for i, (key, max_diff, mean_diff) in enumerate(value_differences[:10]):
                print(f"  {i + 1}. {key}: max={max_diff:.2e}, mean={mean_diff:.2e}")
            if len(value_differences) > 10:
                print(f"  ... and {len(value_differences) - 10} more")

        # Final verdict
        models_identical = len(shape_mismatches) == 0 and len(value_differences) == 0
        print(f"\n🎯 FINAL VERDICT: {'IDENTICAL' if models_identical else 'DIFFERENT'}")

        return models_identical

    # Compare text_encoder and qwen_vl
    are_identical = compare_models(
        text_encoder, qwen_vl, "text_encoder (from pipe)", "qwen_vl (loaded separately)", tolerance=1e-6
    )

    # Also compare their text_encoder components specifically
    if hasattr(qwen_vl, "text_encoder"):
        print("\n" + "=" * 80)
        print("ADDITIONAL COMPARISON: text_encoder vs qwen_vl.text_encoder")
        print("=" * 80)

        are_text_encoders_identical = compare_models(
            text_encoder, qwen_vl.text_encoder, "text_encoder (from pipe)", "qwen_vl.text_encoder", tolerance=1e-6
        )

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
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
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
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",  # 只打到 Linear，不是整个 to_out
        # "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
        # "attn.to_add_out",          # 仅当它是 Linear 时再放开（见下方自检）
    ]

    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=suffixes,
        bias="none",
        task_type="FEATURE_EXTRACTION",  # 或 CAUSAL_LM，均可
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
