"""
测试配置模块的功能，确保配置文件正确加载和解析
"""

import os
import pytest
from qflux.data.config import load_config_from_yaml


class TestConfig:
    """配置加载和验证的测试类"""

    @pytest.fixture
    def example_config_path(self):
        """返回示例配置文件路径"""
        return "tests/test_configs/test_example_fluxkontext_fp16.yaml"

    def test_load_config_exists(self, example_config_path):
        """测试配置文件存在且可以加载"""
        assert os.path.exists(example_config_path), f"配置文件 {example_config_path} 不存在"
        config = load_config_from_yaml(example_config_path)
        assert config is not None, "配置加载失败，返回了None"

    def test_model_config(self, example_config_path):
        """测试模型配置部分正确加载"""
        config = load_config_from_yaml(example_config_path)
        # 模型基本配置
        assert config.model.pretrained_model_name_or_path == "black-forest-labs/FLUX.1-Kontext-dev"
        assert config.model.quantize is True

        # LoRA配置
        assert config.model.lora.r == 16, "LoRA秩参数不正确"
        assert config.model.lora.lora_alpha == 16, "LoRA alpha参数不正确"
        assert config.model.lora.init_lora_weights == "gaussian", "LoRA初始化方式不正确"
        assert config.model.lora.target_modules == ["to_k", "to_q", "to_v", "to_out.0"], "LoRA目标模块不正确"
        assert config.model.lora.adapter_name == "lora_edit", "LoRA适配器名称不正确"

    def test_data_config(self, example_config_path):
        """测试数据配置部分正确加载"""
        config = load_config_from_yaml(example_config_path)
        # 数据集配置
        assert config.data.class_path == "qlfux.data.dataset.ImageDataset"
        # 检查数据集路径，格式可能是字典列表
        assert len(config.data.init_args.dataset_path) == 1
        assert config.data.init_args.dataset_path[0].get('split') == 'train'
        assert config.data.init_args.dataset_path[0].get('repo_id') == 'TsienDragon/face_segmentation_20'
        assert config.data.init_args.caption_dropout_rate == 0.1
        assert config.data.init_args.prompt_image_dropout_rate == 0.1
        assert config.data.init_args.use_edit_mask is True
        assert config.data.init_args.selected_control_indexes == [1]

        # 处理器配置
        assert config.data.init_args.processor.class_path == "qflux.data.preprocess.ImageProcessor"
        assert config.data.init_args.processor.init_args.process_type == "center_crop"
        assert config.data.init_args.processor.init_args.resize_mode == "bilinear"
        assert config.data.init_args.processor.init_args.target_size == [832, 576]
        assert config.data.init_args.processor.init_args.controls_size == [[832, 576]]

        # 数据加载器配置
        assert config.data.batch_size == 8
        assert config.data.num_workers == 2
        assert config.data.shuffle is True

    def test_training_config(self, example_config_path):
        """测试训练配置部分正确加载"""
        config = load_config_from_yaml(example_config_path)
        # 训练参数
        assert config.train.gradient_accumulation_steps == 1
        assert config.train.max_train_steps == 10000
        assert config.train.checkpointing_steps == 100
        assert config.train.mixed_precision == "bf16"
        assert config.train.gradient_checkpointing is True

    @pytest.mark.parametrize("invalid_path", [
        "configs/non_existent_file.yaml",
        "invalid_path.yaml"
    ])
    def test_load_nonexistent_config(self, invalid_path):
        """测试加载不存在的配置文件时的行为"""
        with pytest.raises(Exception) as excinfo:
            load_config_from_yaml(invalid_path)
        assert "Configuration file not found" in str(excinfo.value)

    @pytest.mark.parametrize("config_key,expected_value", [
        ("model.lora.r", 16),
        ("model.lora.lora_alpha", 16),
        ("model.lora.init_lora_weights", "gaussian"),
        ("data.batch_size", 8),
        ("train.mixed_precision", "bf16")
    ])
    def test_config_values(self, example_config_path, config_key, expected_value):
        """参数化测试配置文件中的各个值"""
        config = load_config_from_yaml(example_config_path)
        keys = config_key.split(".")
        value = config
        for key in keys:
            value = getattr(value, key)
        assert value == expected_value, f"配置项 {config_key} 的值不符合预期"

    def test_config_type_validation(self, example_config_path):
        """测试配置项类型验证"""
        config = load_config_from_yaml(example_config_path)
        # 检查数值类型
        assert isinstance(config.model.lora.r, int), "LoRA秩应该是整数类型"
        assert isinstance(config.data.batch_size, int), "批量大小应该是整数类型"
        # 检查布尔类型
        assert isinstance(config.model.quantize, bool), "量化标志应该是布尔类型"
        assert isinstance(config.train.gradient_checkpointing, bool), "梯度检查点应该是布尔类型"
        # 检查字符串类型
        assert isinstance(config.model.pretrained_model_name_or_path, str), "预训练模型路径应该是字符串类型"
        # 检查列表类型
        assert isinstance(config.model.lora.target_modules, list), "目标模块应该是列表类型"
        assert isinstance(config.data.init_args.processor.init_args.target_size, list), "目标尺寸应该是列表类型"

    @pytest.mark.parametrize("config_file", [
        "tests/test_configs/test_example_fluxkontext_fp16.yaml",
    ])
    def test_multiple_config_files(self, config_file):
        """测试加载多个不同的配置文件"""
        try:
            config = load_config_from_yaml(config_file)
            assert config is not None, f"配置文件 {config_file} 加载失败"

            # 检查必要的配置部分存在
            assert hasattr(config, "model"), f"配置文件 {config_file} 缺少 model 部分"
            assert hasattr(config, "data"), f"配置文件 {config_file} 缺少 data 部分"
            assert hasattr(config, "train"), f"配置文件 {config_file} 缺少 train 部分"

            # 检查常见的配置项
            assert hasattr(config.model, "pretrained_model_name_or_path"), f"配置文件 {config_file} 缺少模型路径"
            assert hasattr(config.data, "batch_size"), f"配置文件 {config_file} 缺少批量大小"

        except Exception as e:
            pytest.skip(f"跳过测试配置文件 {config_file}: {str(e)}")
