from .transformer_qwenimage import QwenImageTransformer2DModel
from diffusers import AutoencoderKLQwenImage, QwenImagePipeline

def load_vae(pretrained_model_name_or_path, weight_dtype):
    vae = AutoencoderKLQwenImage.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype
    )
    return vae

def load_qwenvl(pretrained_model_name_or_path, weight_dtype):
    text_encoding_pipeline = QwenImagePipeline.from_pretrained(
        pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )
    return text_encoding_pipeline

def load_transformer(pretrained_model_name_or_path, weight_dtype):
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer", torch_dtype=weight_dtype)
    return flux_transformer

if __name__ == "__main__":
    import torch
    transformer = load_transformer("Qwen/Qwen-Image-Edit", torch.bfloat16)
    print(transformer.state_dict().keys())

    vae = load_vae("Qwen/Qwen-Image-Edit")
    print(vae.state_dict().keys())

    qwen_vl = load_qwenvl("Qwen/Qwen-Image-Edit", torch.bfloat16)
    print(qwen_vl.state_dict().keys())