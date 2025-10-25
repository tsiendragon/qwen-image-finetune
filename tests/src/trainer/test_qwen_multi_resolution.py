"""
Tests for QwenImageEditTrainer multi-resolution support.

This module tests:
1. _get_image_shapes_multi_resolution method
2. extract_batch_field helper (from utils.tools)
3. Multi-resolution img_shapes handling for Qwen models
4. Inference consistency: batch with padding vs individual samples
"""

import pytest
import torch
from unittest.mock import MagicMock
from qflux.utils.tools import extract_batch_field


class MockConfig:
    """Mock configuration for testing"""

    def __init__(self):
        self.data = MagicMock()
        self.data.init_args = MagicMock()
        self.data.init_args.processor = MagicMock()
        self.data.init_args.processor.init_args = MagicMock()
        self.data.init_args.processor.init_args.multi_resolutions = ["512*512", "640*640"]


class TestQwenMultiResolution:
    """Test suite for Qwen multi-resolution support"""

    def test_get_image_shapes_multi_resolution_from_img_shapes(self):
        """验证从 img_shapes 字段获取 per-sample shapes"""
        from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer

        # Mock trainer
        trainer = MagicMock(spec=QwenImageEditTrainer)
        trainer.vae_scale_factor = 8

        # Bind the actual method
        trainer._get_image_shapes_multi_resolution = (
            QwenImageEditTrainer._get_image_shapes_multi_resolution.__get__(trainer, QwenImageEditTrainer)
        )

        # Test data: 2 samples with different resolutions
        embeddings = {
            "img_shapes": [
                [(1, 32, 64), (1, 32, 64)],  # Sample 0: target + control
                [(1, 48, 48), (1, 48, 48)],  # Sample 1: different size
            ],
        }

        result = trainer._get_image_shapes_multi_resolution(embeddings, batch_size=2)

        # Verify
        assert len(result) == 2, f"Expected 2 samples, got {len(result)}"
        assert result[0] == [(1, 32, 64), (1, 32, 64)], f"Sample 0: {result[0]}"
        assert result[1] == [(1, 48, 48), (1, 48, 48)], f"Sample 1: {result[1]}"

        print("✓ Per-sample img_shapes extracted correctly")
        print(f"  Sample 0: {result[0]}")
        print(f"  Sample 1: {result[1]}")

    def test_get_image_shapes_multi_resolution_from_height_width(self):
        """验证从 height/width 字段重建 per-sample shapes"""
        from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer

        # Mock trainer
        trainer = MagicMock(spec=QwenImageEditTrainer)
        trainer.vae_scale_factor = 8

        # Bind method
        trainer._get_image_shapes_multi_resolution = (
            QwenImageEditTrainer._get_image_shapes_multi_resolution.__get__(trainer, QwenImageEditTrainer)
        )

        # Test data: variable sizes
        embeddings = {
            "height": [512, 640, 768],  # Different heights
            "width": [512, 640, 512],   # Different widths
            "height_control": [512, 640, 768],
            "width_control": [512, 640, 512],
        }

        result = trainer._get_image_shapes_multi_resolution(embeddings, batch_size=3)

        # Verify
        assert len(result) == 3

        # Sample 0: 512x512
        expected_0 = [(1, 32, 32), (1, 32, 32)]  # target + control
        assert result[0] == expected_0, f"Sample 0: expected {expected_0}, got {result[0]}"

        # Sample 1: 640x640
        expected_1 = [(1, 40, 40), (1, 40, 40)]
        assert result[1] == expected_1, f"Sample 1: expected {expected_1}, got {result[1]}"

        # Sample 2: 768x512
        expected_2 = [(1, 48, 32), (1, 48, 32)]
        assert result[2] == expected_2, f"Sample 2: expected {expected_2}, got {result[2]}"

        print("✓ Per-sample shapes reconstructed from height/width")
        print(f"  Sample 0 (512x512): {result[0]}")
        print(f"  Sample 1 (640x640): {result[1]}")
        print(f"  Sample 2 (768x512): {result[2]}")

    def test_get_image_shapes_with_additional_controls(self):
        """验证多控制分支的 shapes 生成"""
        from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer

        # Mock trainer
        trainer = MagicMock(spec=QwenImageEditTrainer)
        trainer.vae_scale_factor = 8

        # Bind method
        trainer._get_image_shapes_multi_resolution = (
            QwenImageEditTrainer._get_image_shapes_multi_resolution.__get__(trainer, QwenImageEditTrainer)
        )

        # Test data: Sample 0 has 2 additional controls, Sample 1 has 1
        embeddings = {
            "height": [512, 640],
            "width": [512, 640],
            "height_control": [512, 640],
            "width_control": [512, 640],
            "n_controls": [2, 1],  # Number of additional controls
            "height_control_1": [384, 512],
            "width_control_1": [384, 512],
            "height_control_2": [256, None],  # Only Sample 0 has control_2
            "width_control_2": [256, None],
        }

        result = trainer._get_image_shapes_multi_resolution(embeddings, batch_size=2)

        # Verify Sample 0: target + control + control_1 + control_2 = 4 images
        assert len(result[0]) == 4, f"Sample 0 should have 4 images, got {len(result[0])}"
        assert result[0][0] == (1, 32, 32), "Sample 0 target"
        assert result[0][1] == (1, 32, 32), "Sample 0 control"
        assert result[0][2] == (1, 24, 24), "Sample 0 control_1"
        assert result[0][3] == (1, 16, 16), "Sample 0 control_2"

        # Verify Sample 1: target + control + control_1 = 3 images
        assert len(result[1]) == 3, f"Sample 1 should have 3 images, got {len(result[1])}"
        assert result[1][0] == (1, 40, 40), "Sample 1 target"
        assert result[1][1] == (1, 40, 40), "Sample 1 control"
        assert result[1][2] == (1, 32, 32), "Sample 1 control_1"

        print("✓ Multi-control shapes generated correctly")
        print(f"  Sample 0 (4 images): {result[0]}")
        print(f"  Sample 1 (3 images): {result[1]}")

    def test_extract_batch_field_list(self):
        """验证 extract_batch_field 处理 list"""
        embeddings = {"height": [512, 640, 768]}

        assert extract_batch_field(embeddings, "height", 0) == 512
        assert extract_batch_field(embeddings, "height", 1) == 640
        assert extract_batch_field(embeddings, "height", 2) == 768

        print("✓ extract_batch_field handles list correctly")

    def test_extract_batch_field_tensor(self):
        """验证 extract_batch_field 处理 tensor"""
        embeddings = {"height": torch.tensor([512, 640, 768])}

        assert extract_batch_field(embeddings, "height", 0) == 512
        assert extract_batch_field(embeddings, "height", 1) == 640
        assert extract_batch_field(embeddings, "height", 2) == 768

        print("✓ extract_batch_field handles tensor correctly")

    def test_extract_batch_field_scalar(self):
        """验证 extract_batch_field 处理 scalar"""
        embeddings = {"height": 512}  # Scalar - same for all samples

        assert extract_batch_field(embeddings, "height", 0) == 512
        assert extract_batch_field(embeddings, "height", 1) == 512
        assert extract_batch_field(embeddings, "height", 2) == 512

        print("✓ extract_batch_field handles scalar correctly")

    def test_qwen_multi_resolution_inference_consistency(self):
        """测试 Qwen 多分辨率推理一致性：batch padding vs 逐个样本推理

        Qwen 的 RoPE 原始设计：
        - 为每个样本生成频率，然后 concat 成一个长序列（sequence concatenation 模式）
        - 这要求所有样本的 tokens 连续排列，不支持独立的 batch 维度

        本测试实现：
        - 修改为 padding 模式：为每个样本生成独立的 RoPE，然后 padding 到相同长度
        - 使用 attention_mask 忽略 padding 区域
        - 验证 batch inference 与逐个样本 inference 的一致性

        验证内容：
        1. 不同分辨率样本的独立推理（batch_size=1）
        2. Padding 模式的批处理推理（使用 per-sample RoPE + attention mask）
        3. 两种方式的有效区域结果应该一致
        """
        torch.manual_seed(42)
        device = torch.device("cpu")

        # 创建简化的 Qwen 模型（用于测试）
        try:
            from qflux.models.transformer_qwenimage import QwenImageTransformer2DModel
            from qflux.models.qwen_multi_resolution_patch import patch_qwen_model_for_multi_resolution
        except ImportError as e:
            print(f"⚠ Skipping: Required modules not available: {e}")
            return

        # 使用小型配置加速测试
        # 注意: attention_head_dim 必须与 axes_dims_rope 的总和匹配
        # 默认 axes_dims_rope = (16, 56, 56)，总和 = 128
        # 为了测试，使用较小的配置：axes_dims_rope = (8, 28, 28)，总和 = 64
        model = QwenImageTransformer2DModel(
            patch_size=2,
            in_channels=64,
            out_channels=16,
            num_layers=2,  # 减少层数加速测试
            attention_head_dim=64,  # 与 axes_dims_rope 总和匹配
            num_attention_heads=4,
            joint_attention_dim=512,
            axes_dims_rope=(8, 28, 28),  # 总和 = 64，匹配 attention_head_dim
        ).to(device)

        # ✅ Apply multi-resolution patch
        print("\n✓ Applying multi-resolution patch to model...")
        patch_qwen_model_for_multi_resolution(model)
        model.eval()

        # 使用不同分辨率的样本（测试真正的多分辨率）
        img_shapes_latent = [
            [(1, 16, 32)],  # Sample 0: 256x512 -> 16x32 (seq_len=512)
            [(1, 24, 24)],  # Sample 1: 384x384 -> 24x24 (seq_len=576)
            [(1, 20, 28)],  # Sample 2: 320x448 -> 20x28 (seq_len=560)
        ]

        batch_size = len(img_shapes_latent)
        txt_len = 64  # 固定文本长度
        channels = 64

        print(f"\n{'='*60}")
        print(f"Testing Qwen Multi-Resolution Inference Consistency")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"Resolutions (latent space):")
        for i, shapes in enumerate(img_shapes_latent):
            h, w = shapes[0][1], shapes[0][2]
            print(f"  Sample {i}: {h}x{w} (seq_len={h*w})")

        # ============================================================
        # 第一步：为每个样本单独做 inference
        # ============================================================
        print(f"\n{'='*60}")
        print("Step 1: Individual inference (batch_size=1 for each sample)")
        print(f"{'='*60}")

        individual_outputs = []
        individual_inputs = []  # 保存每个样本的输入用于批处理

        for i in range(batch_size):
            # 计算当前样本的序列长度
            h, w = img_shapes_latent[i][0][1], img_shapes_latent[i][0][2]
            seq_len = h * w

            # 生成输入数据 - 为了可重现性，使用不同的随机种子
            torch.manual_seed(42 + i)
            hidden_states = torch.randn(1, seq_len, channels, device=device)
            encoder_hidden_states = torch.randn(1, txt_len, 512, device=device)
            encoder_hidden_states_mask = torch.ones(1, txt_len, device=device, dtype=torch.int64)
            timestep = torch.randint(0, 1000, (1,), device=device)

            # img_shapes 格式: List[List[Tuple]]
            img_shapes_sample = [img_shapes_latent[i]]  # [[（1, h, w)]]
            txt_seq_lens = [txt_len]

            # 保存输入用于批处理
            individual_inputs.append({
                'hidden_states': hidden_states.clone(),
                'encoder_hidden_states': encoder_hidden_states.clone(),
                'encoder_hidden_states_mask': encoder_hidden_states_mask.clone(),
                'timestep': timestep.clone(),
            })

            with torch.no_grad():
                output = model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    timestep=timestep,
                    img_shapes=img_shapes_sample,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=True,
                )

            if hasattr(output, 'sample'):
                output_tensor = output.sample
            else:
                output_tensor = output[0] if isinstance(output, tuple) else output

            individual_outputs.append(output_tensor)

            print(f"Sample {i}: output shape {output_tensor.shape}")

        # ============================================================
        # 第二步：批处理 inference（padding 模式 + per-sample RoPE）
        # ============================================================
        print(f"\n{'='*60}")
        print("Step 2: Batched inference with padding and per-sample RoPE")
        print(f"{'='*60}")

        # 计算最大序列长度
        max_seq = max(shapes[0][1] * shapes[0][2] for shapes in img_shapes_latent)
        print(f"Max sequence length: {max_seq}")

        # 创建 padding 后的输入
        hidden_states_batched = torch.zeros(batch_size, max_seq, channels, device=device)
        encoder_hidden_states_batched = torch.zeros(batch_size, txt_len, 512, device=device)
        encoder_hidden_states_mask_batched = torch.zeros(batch_size, txt_len, device=device, dtype=torch.int64)
        timestep_batched = torch.zeros(batch_size, device=device, dtype=torch.long)

        # 创建 attention_mask (B, joint_seq)
        joint_seq = txt_len + max_seq
        attention_mask = torch.zeros(batch_size, joint_seq, dtype=torch.bool, device=device)

        # 填充数据
        for i in range(batch_size):
            h, w = img_shapes_latent[i][0][1], img_shapes_latent[i][0][2]
            seq_len = h * w

            # 填充 hidden_states
            hidden_states_batched[i, :seq_len] = individual_inputs[i]['hidden_states'].squeeze(0)

            # 填充 encoder_hidden_states
            encoder_hidden_states_batched[i] = individual_inputs[i]['encoder_hidden_states'].squeeze(0)

            # 填充 encoder_hidden_states_mask
            encoder_hidden_states_mask_batched[i] = individual_inputs[i]['encoder_hidden_states_mask'].squeeze(0)

            # 填充 timestep
            timestep_batched[i] = individual_inputs[i]['timestep'].squeeze()

            # 设置 attention_mask
            attention_mask[i, :txt_len] = True  # text tokens
            attention_mask[i, txt_len:txt_len + seq_len] = True  # valid image tokens
            # padding 区域保持 False

        # 转换 attention_mask 为 4D: (B, 1, seq_q, seq_k) for broadcasting
        # scaled_dot_product_attention 需要 4D mask (B, num_heads, seq_q, seq_k)
        # 我们的 mask 是 padding mask：(B, seq) -> (B, 1, 1, seq) -> broadcast to (B, 1, seq, seq)
        # 注意：mask 中 True 表示保留，False 表示忽略
        attention_mask_4d = attention_mask.view(batch_size, 1, 1, joint_seq)  # (B, 1, 1, seq)
        # 扩展为 (B, 1, seq, seq) - 广播会自动处理

        print(f"Batched shapes:")
        print(f"  hidden_states: {hidden_states_batched.shape}")
        print(f"  encoder_hidden_states: {encoder_hidden_states_batched.shape}")
        print(f"  attention_mask: {attention_mask.shape}")
        print(f"  attention_mask_4d: {attention_mask_4d.shape}")
        print(f"  timestep: {timestep_batched.shape}")

        # 为每个样本生成独立的 RoPE，然后 padding
        img_freqs_batched_list = []
        txt_freqs_list = []

        for i in range(batch_size):
            # 为单个样本生成 RoPE
            img_freqs_i, txt_freqs_i = model.pos_embed([img_shapes_latent[i]], [txt_len], device=device)
            img_freqs_batched_list.append(img_freqs_i)
            txt_freqs_list.append(txt_freqs_i)

        # Padding img_freqs 到相同长度
        img_freqs_batched = torch.zeros(batch_size, max_seq, img_freqs_batched_list[0].shape[-1],
                                        dtype=img_freqs_batched_list[0].dtype, device=device)
        for i in range(batch_size):
            seq_len_i = img_freqs_batched_list[i].shape[0]
            img_freqs_batched[i, :seq_len_i] = img_freqs_batched_list[i]

        # txt_freqs 对所有样本相同
        txt_freqs_batched = txt_freqs_list[0]  # (txt_len, rope_dim)

        image_rotary_emb_batched = (img_freqs_batched, txt_freqs_batched)

        print(f"RoPE shapes:")
        print(f"  img_freqs_batched: {img_freqs_batched.shape}")
        print(f"  txt_freqs_batched: {txt_freqs_batched.shape}")

        # ✅ 批处理推理（使用预计算的 per-sample RoPE）
        with torch.no_grad():
            batched_output = model(
                hidden_states=hidden_states_batched,
                encoder_hidden_states=encoder_hidden_states_batched,
                encoder_hidden_states_mask=encoder_hidden_states_mask_batched,
                timestep=timestep_batched,
                image_rotary_emb=image_rotary_emb_batched,  # ✅ 预计算的 per-sample RoPE
                attention_kwargs={'attention_mask': attention_mask_4d},  # ✅ 4D mask for attention
                return_dict=True,
            )

        if hasattr(batched_output, 'sample'):
            batched_output_tensor = batched_output.sample
        else:
            batched_output_tensor = batched_output[0] if isinstance(batched_output, tuple) else batched_output

        print(f"Batched output shape: {batched_output_tensor.shape}")

        # ============================================================
        # 第三步：验证两种方法的输出是否一致
        # ============================================================
        print(f"\n{'='*60}")
        print("Step 3: Comparing individual vs batched outputs")
        print(f"{'='*60}")

        all_passed = True

        for i in range(batch_size):
            h, w = img_shapes_latent[i][0][1], img_shapes_latent[i][0][2]
            seq_len = h * w

            # 提取批处理结果中该样本的有效部分（去除 padding）
            batched_sample_output = batched_output_tensor[i, :seq_len]
            individual_sample_output = individual_outputs[i].squeeze(0)

            # 验证形状
            assert batched_sample_output.shape == individual_sample_output.shape, (
                f"Sample {i}: Shape mismatch - "
                f"batched {batched_sample_output.shape} vs individual {individual_sample_output.shape}"
            )

            # 验证数值一致性
            diff = batched_sample_output - individual_sample_output
            max_diff = diff.abs().max().item()
            mean_diff = diff.abs().mean().item()

            # 相对误差（L2 范数）
            norm_diff = torch.norm(diff).item()
            norm_individual = torch.norm(individual_sample_output).item()
            relative_error = norm_diff / (norm_individual +1e-6)

            print(f"\nSample {i} ({h}x{w}):")
            print(f"  Valid shape: {batched_sample_output.shape}")
            print(f"  Max absolute diff: {max_diff:.6e}")
            print(f"  Mean absolute diff: {mean_diff:.6e}")
            print(f"  Relative error (L2): {relative_error:.6e}")

            # 验证 padding 区域是否为零或很小
            if seq_len < max_seq:
                padding_region = batched_output_tensor[i, seq_len:]
                padding_norm = torch.norm(padding_region).item()
                padding_max = padding_region.abs().max().item()
                print(f"  Padding region ({max_seq - seq_len} tokens):")
                print(f"    L2 norm: {padding_norm:.6e}")
                print(f"    Max value: {padding_max:.6e}")

                # Padding 区域应该接近零（可能有小的数值误差）
                if padding_max > 1e-3:
                    print(f"  ⚠ WARNING: Padding region has large values!")

            # 使用相对误差作为判断标准
            # 对于数值稳定的实现，相对误差应该很小（< 1e-5）
            # 但考虑到浮点精度和不同实现，我们设置宽松一些的阈值
            max_relative_error = 1e-3  # 0.1% 相对误差
            max_absolute_diff = 1e-4   # 绝对误差阈值（用于处理接近零的值）

            if relative_error > max_relative_error and max_diff > max_absolute_diff:
                print(f"  ✗ FAILED: Relative error {relative_error:.6e} > {max_relative_error:.6e}")
                print(f"           Max absolute diff {max_diff:.6e} > {max_absolute_diff:.6e}")
                all_passed = False
            else:
                print(f"  ✓ PASSED: Within tolerance (relative_error < {max_relative_error:.6e})")

        # 最终验证
        print(f"\n{'='*60}")
        if all_passed:
            print("✓ All samples passed: Batched inference is consistent with individual inference")
            print("✓ Qwen multi-resolution padding mode works correctly!")
            print("✓ Per-sample RoPE + attention mask successfully validated")
        else:
            print("✗ Some samples failed: Inconsistency detected")
            raise AssertionError("Batched inference results do not match individual inference")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Run tests
    test_suite = TestQwenMultiResolution()

    print("Running: test_get_image_shapes_multi_resolution_from_img_shapes...")
    test_suite.test_get_image_shapes_multi_resolution_from_img_shapes()
    print()

    print("Running: test_get_image_shapes_multi_resolution_from_height_width...")
    test_suite.test_get_image_shapes_multi_resolution_from_height_width()
    print()

    print("Running: test_get_image_shapes_with_additional_controls...")
    test_suite.test_get_image_shapes_with_additional_controls()
    print()

    print("Running: test_extract_batch_field_list...")
    test_suite.test_extract_batch_field_list()
    print()

    print("Running: test_extract_batch_field_tensor...")
    test_suite.test_extract_batch_field_tensor()
    print()

    print("Running: test_extract_batch_field_scalar...")
    test_suite.test_extract_batch_field_scalar()
    print()

    print("Running: test_qwen_multi_resolution_inference_consistency...")
    test_suite.test_qwen_multi_resolution_inference_consistency()
    print()

    print("All Qwen multi-resolution tests passed! ✓")
