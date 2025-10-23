"""
通用测试数据加载工具函数。

提供统一接口从 HuggingFace Hub 下载的测试资源中加载各种格式的测试数据。
"""
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


def load_torch_file(
    test_resources: Path,
    relative_path: str,
    map_location: str = "cpu",
    weights_only: bool = True,
) -> Any:
    """
    从测试资源目录加载单个 PyTorch 文件。

    Args:
        test_resources: 测试资源根目录路径（通常来自 test_resources fixture）
        relative_path: 相对于资源目录的文件路径，例如 "flux_models/transformer/input/flux_input.pth"
        map_location: torch.load 的 map_location 参数
        weights_only: 是否只加载权重（安全模式）

    Returns:
        加载的数据对象

    Examples:
        >>> def test_example(test_resources):
        ...     data = load_torch_file(
        ...         test_resources,
        ...         "flux_models/transformer/input/flux_input.pth"
        ...     )
    """
    file_path = test_resources / relative_path
    if not file_path.exists():
        raise FileNotFoundError(f"测试数据文件不存在: {file_path}")

    logger.debug(f"加载测试数据: {file_path}")
    return torch.load(file_path, map_location=map_location, weights_only=weights_only)


def load_torch_directory(
    test_resources: Path,
    relative_dir: str,
    pattern: str = "*.pt",
    map_location: str = "cpu",
    weights_only: bool = True,
) -> Dict[str, Any]:
    """
    从测试资源目录批量加载 PyTorch 文件。

    Args:
        test_resources: 测试资源根目录路径
        relative_dir: 相对于资源目录的目录路径
        pattern: 文件名匹配模式（glob pattern）
        map_location: torch.load 的 map_location 参数
        weights_only: 是否只加载权重（安全模式）

    Returns:
        字典，key 为文件名（不含扩展名），value 为加载的数据

    Examples:
        >>> def test_example(test_resources):
        ...     data = load_torch_directory(
        ...         test_resources,
        ...         "flux_training/face_segmentation/sample1"
        ...     )
        ...     # 访问: data['sample_control_ids'], data['sample_noise'], etc.
    """
    dir_path = test_resources / relative_dir
    if not dir_path.exists():
        raise FileNotFoundError(f"测试数据目录不存在: {dir_path}")

    data = {}
    for file_path in sorted(dir_path.glob(pattern)):
        key = file_path.stem  # 文件名（不含扩展名）
        logger.debug(f"加载测试数据: {file_path}")
        data[key] = torch.load(file_path, map_location=map_location, weights_only=weights_only)

    if not data:
        logger.warning(f"目录中未找到匹配文件: {dir_path} (pattern: {pattern})")

    return data


def load_flux_training_sample(
    test_resources: Path,
    sample_name: str = "sample1",
    map_location: str = "cpu",
    weights_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    加载 Flux 训练样本数据（预设路径）。

    这是 load_torch_directory 的便捷封装，专门用于加载 flux_training 数据。

    Args:
        test_resources: 测试资源根目录路径
        sample_name: 样本名称，如 "sample1", "sample2"
        map_location: torch.load 的 map_location 参数
        weights_only: 是否只加载权重（安全模式）

    Returns:
        包含所有样本数据的字典

    Examples:
        >>> def test_example(test_resources):
        ...     sample1 = load_flux_training_sample(test_resources, "sample1")
        ...     sample2 = load_flux_training_sample(test_resources, "sample2")
    """
    return load_torch_directory(
        test_resources,
        f"flux_training/face_segmentation/{sample_name}",
        pattern="*.pt",
        map_location=map_location,
        weights_only=weights_only,
    )


def load_flux_transformer_input(
    test_resources: Path,
    map_location: str = "cpu",
    weights_only: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    加载 Flux Transformer 输入测试数据（预设路径）。

    这是 load_torch_file 的便捷封装，专门用于加载 flux transformer 输入数据。

    Args:
        test_resources: 测试资源根目录路径
        map_location: torch.load 的 map_location 参数
        weights_only: 是否只加载权重（安全模式）

    Returns:
        包含输入数据的字典

    Examples:
        >>> def test_example(test_resources):
        ...     input_data = load_flux_transformer_input(test_resources)
        ...     latent_ids = input_data["latent_ids"]
    """
    return load_torch_file(
        test_resources,
        "flux_models/transformer/input/flux_input.pth",
        map_location=map_location,
        weights_only=weights_only,
    )


def load_flux_sampling_embeddings(
    test_resources: Path,
    map_location: str = "cpu",
    weights_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    加载 Flux 采样嵌入数据（预设路径）。

    Args:
        test_resources: 测试资源根目录路径
        map_location: torch.load 的 map_location 参数
        weights_only: 是否只加载权重（安全模式）

    Returns:
        包含所有嵌入数据的字典

    Examples:
        >>> def test_example(test_resources):
        ...     embeddings = load_flux_sampling_embeddings(test_resources)
        ...     control_latents = embeddings["sample_control_latents"]
    """
    return load_torch_directory(
        test_resources,
        "flux_sampling/embeddings",
        pattern="*.pt",
        map_location=map_location,
        weights_only=weights_only,
    )


def move_dict_to_device(
    data: Dict[str, torch.Tensor],
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
    keys: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    将字典中的所有张量移动到指定设备和数据类型。

    Args:
        data: 包含张量的字典
        device: 目标设备
        dtype: 目标数据类型（可选，仅应用于浮点张量）
        keys: 要移动的键列表（如果为 None，则移动所有张量）

    Returns:
        更新后的字典（原地修改）

    Examples:
        >>> data = {"tensor1": torch.randn(10), "tensor2": torch.randn(20)}
        >>> data = move_dict_to_device(data, "cuda", torch.float16)
    """
    keys_to_move = keys or data.keys()

    for key in keys_to_move:
        if key not in data:
            logger.warning(f"Key '{key}' not found in data dict")
            continue

        value = data[key]
        if isinstance(value, torch.Tensor):
            if dtype is not None and value.is_floating_point():
                data[key] = value.to(device=device, dtype=dtype)
            else:
                data[key] = value.to(device=device)

    return data


def prepare_test_data_for_device(
    data: Dict[str, Any],
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
    exclude_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    准备测试数据：将所有张量移动到指定设备，并可选地转换数据类型。

    这是 move_dict_to_device 的增强版本，支持排除某些键。

    Args:
        data: 包含测试数据的字典
        device: 目标设备
        dtype: 目标数据类型（可选）
        exclude_keys: 不移动的键列表

    Returns:
        新的字典（不修改原字典）

    Examples:
        >>> data = load_flux_training_sample(test_resources, "sample1")
        >>> data = prepare_test_data_for_device(
        ...     data, device="cuda", dtype=torch.bfloat16
        ... )
    """
    exclude_keys = exclude_keys or []
    result = {}

    for key, value in data.items():
        if key in exclude_keys:
            result[key] = value
        elif isinstance(value, torch.Tensor):
            if dtype is not None and value.is_floating_point():
                result[key] = value.to(device=device, dtype=dtype)
            else:
                result[key] = value.to(device=device)
        else:
            result[key] = value

    return result
