import re
import safetensors.torch
import torch

def classify_lora_weight(lora_weight):
    sd = safetensors.torch.load_file(lora_weight)
    keys = list(sd.keys())
    peft = any(re.search(r"\.lora_[AB](\.|$)", k) for k in keys)
    diff = any(".lora.down.weight" in k or ".lora.up.weight" in k for k in keys)
    proc = any(".processor" in k for k in keys)
    if peft and not diff:
        return "PEFT"
    if diff:
        return "DIFFUSERS(attn-processor)" if proc else "DIFFUSERS"
    return "UNKNOWN"


def get_lora_layers(model):
    """Traverse the model to find all LoRA-related modules"""
    lora_layers = {}

    def fn_recursive_find_lora_layer(name: str, module: torch.nn.Module, processors):
        if "lora" in name:
            lora_layers[name] = module
        for sub_name, child in module.named_children():
            fn_recursive_find_lora_layer(f"{name}.{sub_name}", child, lora_layers)
        return lora_layers

    for name, module in model.named_children():
        fn_recursive_find_lora_layer(name, module, lora_layers)

    return lora_layers