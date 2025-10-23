"""Test per-sample RoPE implementation in FluxTransformer2DModel.

passed: 2025-10-22 10:00:00
"""
import torch
import pytest
from qflux.models.transformer_flux_custom import (
    FluxPosEmbedBatched,
    FluxAttnProcessorPerSample,
    FluxTransformer2DModel,
)


class TestFluxPosEmbedBatched:
    """测试 FluxPosEmbedBatched 的功能"""

    def test_shared_mode(self):
        """测试 shared mode（2D input）"""
        pos_embed = FluxPosEmbedBatched(theta=10000, axes_dim=[16, 56, 56])

        # 2D input → shared mode
        ids = torch.zeros(64, 3)  # (seq, 3)
        for i in range(64):
            ids[i] = torch.tensor([0, i // 8, i % 8])

        rope = pos_embed(ids)
        assert isinstance(rope, tuple)
        freqs_cos, freqs_sin = rope
        assert freqs_cos.shape[0] == 64
        assert freqs_sin.shape[0] == 64

    def test_per_sample_mode(self):
        """测试 per-sample mode（3D input）"""
        pos_embed = FluxPosEmbedBatched(theta=10000, axes_dim=[16, 56, 56])

        batch_size = 4
        max_seq = 64
        # 3D input → per-sample mode
        ids = torch.zeros(batch_size, max_seq, 3)

        # 填充不同尺寸的 img_ids
        # Sample 0: 8x8=64
        for i in range(64):
            ids[0, i] = torch.tensor([0, i // 8, i % 8])
        # Sample 1: 6x6=36
        for i in range(36):
            ids[1, i] = torch.tensor([0, i // 6, i % 6])
        # Sample 2: 4x4=16
        for i in range(16):
            ids[2, i] = torch.tensor([0, i // 4, i % 4])
        # Sample 3: 8x8=64
        for i in range(64):
            ids[3, i] = torch.tensor([0, i // 8, i % 8])

        valid_lengths = torch.tensor([64, 36, 16, 64])
        rope_list = pos_embed(ids, valid_lengths=valid_lengths)

        assert isinstance(rope_list, list)
        assert len(rope_list) == batch_size

        # 检查每个样本的 RoPE
        for b in range(batch_size):
            freqs_cos, freqs_sin = rope_list[b]
            assert freqs_cos.shape[0] == max_seq  # padded to max_seq
            assert freqs_sin.shape[0] == max_seq

    def test_rope_cache(self):
        """测试 RoPE 缓存功能"""
        pos_embed = FluxPosEmbedBatched(
            theta=10000, axes_dim=[16, 56, 56], enable_cache=True
        )

        # 生成 8x8 的 ids
        ids = torch.zeros(64, 3)
        for i in range(64):
            ids[i] = torch.tensor([0, i // 8, i % 8])

        # 第一次计算
        rope1 = pos_embed(ids)
        assert len(pos_embed.rope_cache) == 1
        assert (8, 8) in pos_embed.rope_cache

        # 第二次计算（应该从缓存读取）
        rope2 = pos_embed(ids)
        assert rope1[0] is rope2[0]  # 同一个对象
        assert rope1[1] is rope2[1]

        # 清空缓存
        pos_embed.clear_cache()
        assert len(pos_embed.rope_cache) == 0


class TestFluxAttnProcessorPerSample:
    """测试 FluxAttnProcessorPerSample 的功能"""

    def test_apply_rope_per_sample(self):
        """测试 per-sample RoPE 应用"""
        processor = FluxAttnProcessorPerSample()

        batch_size = 2
        seq_len = 16
        heads = 8
        head_dim = 64

        query = torch.randn(batch_size, seq_len, heads, head_dim)
        key = torch.randn(batch_size, seq_len, heads, head_dim)

        # 生成两个不同的 RoPE
        freqs_cos_1 = torch.randn(seq_len, head_dim)
        freqs_sin_1 = torch.randn(seq_len, head_dim)
        freqs_cos_2 = torch.randn(seq_len, head_dim)
        freqs_sin_2 = torch.randn(seq_len, head_dim)

        rope_list = [
            (freqs_cos_1, freqs_sin_1),
            (freqs_cos_2, freqs_sin_2),
        ]

        query_out, key_out = processor._apply_rope_per_sample(
            query, key, rope_list
        )

        assert query_out.shape == query.shape
        assert key_out.shape == key.shape
        # 应该与输入不同（因为应用了 RoPE）
        assert not torch.allclose(query_out, query)
        assert not torch.allclose(key_out, key)


class TestFluxTransformer2DModelPerSample:
    """测试 FluxTransformer2DModel 的 per-sample 功能"""

    @pytest.fixture
    def model_config(self):
        """创建测试用的模型配置"""
        from diffusers.models.transformers.transformer_flux import (
            FluxTransformer2DModel as _FluxTransformer2DModel,
        )

        config = _FluxTransformer2DModel.load_config(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
        )
        return config

    def test_mode_detection(self, model_config):
        """测试自动模式检测"""
        # 创建小型模型用于测试
        model_config["num_layers"] = 2
        model_config["num_single_layers"] = 2
        model = FluxTransformer2DModel(**model_config)
        model.eval()

        batch_size = 2
        height, width = 32, 32
        seq_len = height * width // 64  # 假设 patch_size=8

        hidden_states = torch.randn(batch_size, seq_len, model_config["in_channels"])
        encoder_hidden_states = torch.randn(batch_size, 77, model_config["joint_attention_dim"])
        pooled_projections = torch.randn(batch_size, model_config["pooled_projection_dim"])
        timestep = torch.tensor([500] * batch_size)

        # Test 1: Shared mode (2D img_ids)
        img_ids_shared = torch.zeros(seq_len, 3)
        for i in range(seq_len):
            img_ids_shared[i] = torch.tensor([0, i // (width // 8), i % (width // 8)])

        txt_ids = torch.zeros(77, 3)

        with torch.no_grad():
            output_shared = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                img_ids=img_ids_shared,
                txt_ids=txt_ids,
            )

        assert output_shared is not None

        # Test 2: Per-sample mode (3D img_ids)
        img_ids_batched = img_ids_shared.unsqueeze(0).repeat(batch_size, 1, 1)
        txt_ids_batched = txt_ids.unsqueeze(0).repeat(batch_size, 1, 1)

        with torch.no_grad():
            output_per_sample = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                img_ids=img_ids_batched,  # 3D → per-sample mode
                txt_ids=txt_ids_batched,
            )

        assert output_per_sample is not None

    def test_per_sample_with_attention_mask(self, model_config):
        """测试 per-sample RoPE + attention mask 的组合"""
        model_config["num_layers"] = 2
        model_config["num_single_layers"] = 2
        model = FluxTransformer2DModel(**model_config)
        model.eval()

        batch_size = 2
        max_seq = 64
        txt_len = 77

        # 不同的实际序列长度
        valid_lengths = [48, 32]

        hidden_states = torch.randn(batch_size, max_seq, model_config["in_channels"])
        encoder_hidden_states = torch.randn(batch_size, txt_len, model_config["joint_attention_dim"])
        pooled_projections = torch.randn(batch_size, model_config["pooled_projection_dim"])
        timestep = torch.tensor([500] * batch_size)

        # 生成 per-sample img_ids
        img_ids = torch.zeros(batch_size, max_seq, 3)
        for b in range(batch_size):
            h = w = int(valid_lengths[b] ** 0.5)
            for i in range(valid_lengths[b]):
                img_ids[b, i] = torch.tensor([b, i // w, i % w])

        txt_ids = torch.zeros(batch_size, txt_len, 3)

        # 生成 attention mask
        attention_mask = torch.zeros(batch_size, txt_len + max_seq, dtype=torch.bool)
        for b in range(batch_size):
            attention_mask[b, :txt_len] = True  # text tokens
            attention_mask[b, txt_len : txt_len + valid_lengths[b]] = True  # valid image tokens

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                attention_mask=attention_mask,
            )

        assert output is not None

    def test_per_sample_equivalence_with_individual_inference(self):
        """测试 per-sample RoPE 批处理结果与逐个推理结果的等价性

        关键验证点：
        1. 使用 per-sample RoPE + attention mask 的批处理推理
        2. 使用 batch_size=1 逐个推理每个样本
        3. 两种方式的输出应该在数值上一致

        误差控制策略：
        - 使用 float32 精度
        - 启用 PyTorch 确定性算法
        - 禁用所有随机性（dropout等）
        - 优化 RoPE 应用方式减少数值误差
        - 目标相对误差：< 1e-4
        """
        # 启用确定性算法以减少误差
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 使用自定义小模型配置，避免依赖 HuggingFace
        model = FluxTransformer2DModel(
            patch_size=1,
            in_channels=64,
            out_channels=64,
            num_layers=2,
            num_single_layers=1,
            attention_head_dim=64,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=16,
            guidance_embeds=False,  # 简化测试
            axes_dims_rope=(8, 28, 28),
        )
        model.eval()

        model_config = {
            "in_channels": 64,
            "joint_attention_dim": 32,
            "pooled_projection_dim": 16,
            "guidance_embeds": False,
        }

        # 设置随机种子以保证可重复性
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        batch_size = 3
        txt_len = 77

        # 定义不同尺寸的样本（模拟多分辨率训练）
        img_shapes = [
            (8, 8),   # 64 tokens
            (6, 8),   # 48 tokens
            (7, 7),   # 49 tokens
        ]

        max_seq = max(h * w for h, w in img_shapes)

        # 准备输入数据
        # 为了确保等价性，我们需要为每个样本准备独立的输入
        hidden_states_list = []
        encoder_hidden_states_list = []
        pooled_projections_list = []
        timestep_list = []
        img_ids_list = []
        txt_ids_list = []

        for i, (h, w) in enumerate(img_shapes):
            seq_len = h * w

            # 生成随机输入（使用 float32 精度）
            hidden_states_list.append(
                torch.randn(1, seq_len, model_config["in_channels"], dtype=torch.float32)
            )
            encoder_hidden_states_list.append(
                torch.randn(1, txt_len, model_config["joint_attention_dim"], dtype=torch.float32)
            )
            pooled_projections_list.append(
                torch.randn(1, model_config["pooled_projection_dim"], dtype=torch.float32)
            )
            # timestep 应该是形状为 (batch_size,) 的张量
            timestep_list.append(torch.tensor([500.0 + i * 10.0], dtype=torch.float32))

            # 生成 img_ids (seq, 3)
            img_ids = torch.zeros(seq_len, 3)
            for idx in range(seq_len):
                img_ids[idx] = torch.tensor([0, idx // w, idx % w])
            img_ids_list.append(img_ids)

            # 生成 txt_ids (txt_len, 3)
            txt_ids = torch.zeros(txt_len, 3)
            for i_txt in range(txt_len):
                txt_ids[i_txt] = torch.tensor([0, float(i_txt), 0.0])
            txt_ids_list.append(txt_ids)

        # ============================================================
        # 方法 1: 使用 batch_size=1 逐个推理
        # ============================================================
        individual_outputs = []

        # 检查模型是否需要 guidance 参数
        needs_guidance = model_config.get("guidance_embeds", True)

        with torch.no_grad():
            for i in range(batch_size):
                kwargs = {
                    "hidden_states": hidden_states_list[i],
                    "encoder_hidden_states": encoder_hidden_states_list[i],
                    "pooled_projections": pooled_projections_list[i],
                    "timestep": timestep_list[i],
                    "img_ids": img_ids_list[i],  # (seq, 3) - shared mode
                    "txt_ids": txt_ids_list[i],
                }

                # 如果模型需要 guidance，添加该参数
                if needs_guidance:
                    kwargs["guidance"] = torch.tensor([3.5])

                output = model(**kwargs)

                if hasattr(output, 'sample'):
                    individual_outputs.append(output.sample)
                else:
                    individual_outputs.append(output[0] if isinstance(output, tuple) else output)

        # ============================================================
        # 方法 2: 使用 per-sample RoPE 批处理推理
        # ============================================================

        # 准备批处理输入（padding 到最大序列长度，使用 float32 精度）
        hidden_states_batched = torch.zeros(
            batch_size, max_seq, model_config["in_channels"], dtype=torch.float32
        )
        encoder_hidden_states_batched = torch.zeros(
            batch_size, txt_len, model_config["joint_attention_dim"], dtype=torch.float32
        )
        pooled_projections_batched = torch.zeros(
            batch_size, model_config["pooled_projection_dim"], dtype=torch.float32
        )
        timestep_batched = torch.zeros(batch_size, dtype=torch.float32)

        # 生成 batched img_ids (B, max_seq, 3) - per-sample mode
        img_ids_batched = torch.zeros(batch_size, max_seq, 3)

        # txt_ids 保持 2D (txt_len, 3) - 所有样本共享相同的文本位置编码
        txt_ids_shared = torch.zeros(txt_len, 3)
        for i_txt in range(txt_len):
            txt_ids_shared[i_txt] = torch.tensor([0, float(i_txt), 0.0])

        # 生成 attention_mask (B, txt_len + max_seq)
        attention_mask = torch.zeros(batch_size, txt_len + max_seq, dtype=torch.bool)

        # 填充数据
        for i in range(batch_size):
            h, w = img_shapes[i]
            seq_len = h * w

            # 填充 hidden_states
            hidden_states_batched[i, :seq_len] = hidden_states_list[i].squeeze(0)

            # 填充 encoder_hidden_states
            encoder_hidden_states_batched[i] = encoder_hidden_states_list[i].squeeze(0)

            # 填充 pooled_projections
            pooled_projections_batched[i] = pooled_projections_list[i].squeeze(0)

            # 填充 timestep
            timestep_batched[i] = timestep_list[i].squeeze()

            # 填充 img_ids
            img_ids_batched[i, :seq_len] = img_ids_list[i]

            # 设置 attention_mask
            attention_mask[i, :txt_len] = True  # text tokens
            attention_mask[i, txt_len:txt_len + seq_len] = True  # valid image tokens

        with torch.no_grad():
            kwargs_batched = {
                "hidden_states": hidden_states_batched,
                "encoder_hidden_states": encoder_hidden_states_batched,
                "pooled_projections": pooled_projections_batched,
                "timestep": timestep_batched,
                "img_ids": img_ids_batched,  # (B, seq, 3) - per-sample mode
                "txt_ids": txt_ids_shared,  # (txt_len, 3) - shared for all samples
                "attention_mask": attention_mask,
            }

            # 如果模型需要 guidance，添加该参数
            if needs_guidance:
                kwargs_batched["guidance"] = torch.tensor([3.5] * batch_size)

            batched_output = model(**kwargs_batched)

            if hasattr(batched_output, 'sample'):
                batched_output_tensor = batched_output.sample
            else:
                batched_output_tensor = batched_output[0] if isinstance(batched_output, tuple) else batched_output

        # ============================================================
        # 验证两种方法的输出是否一致
        # ============================================================

        for i in range(batch_size):
            h, w = img_shapes[i]
            seq_len = h * w

            # 提取批处理结果中该样本的有效部分
            batched_sample_output = batched_output_tensor[i, :seq_len]
            individual_sample_output = individual_outputs[i].squeeze(0)

            # 验证形状
            assert batched_sample_output.shape == individual_sample_output.shape, (
                f"Sample {i}: Shape mismatch - "
                f"batched {batched_sample_output.shape} vs individual {individual_sample_output.shape}"
            )

            # 验证数值一致性
            # 使用 L2 范数计算相对误差: ||x - y|| / ||x||
            diff = batched_sample_output - individual_sample_output
            max_diff = diff.abs().max().item()
            mean_diff = diff.abs().mean().item()

            # 相对误差（L2 范数）
            norm_diff = torch.norm(diff).item()
            norm_individual = torch.norm(individual_sample_output).item()
            relative_error = norm_diff / norm_individual if norm_individual > 0 else 0

            print(f"\nSample {i} ({h}x{w}):")
            print(f"  Max absolute diff: {max_diff:.6e}")
            print(f"  Mean absolute diff: {mean_diff:.6e}")
            print(f"  Relative error (L2): {relative_error:.6e} ({relative_error * 100:.4f}%)")
            print(f"  ||diff||: {norm_diff:.6e}")

            # 验证相对误差小于阈值
            # 目标：相对误差 < 1e-4
            assert relative_error < 1e-4, (
                f"Sample {i}: Relative error too large - "
                f"relative_error={relative_error:.6e} ({relative_error * 100:.4f}%), "
                f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
            )

        print("\n✅ Per-sample RoPE 批处理结果与逐个推理结果完全一致！")

    def test_backward_compatibility(self, model_config):
        """测试向后兼容性：不提供 img_ids 应该也能工作"""
        model_config["num_layers"] = 1
        model_config["num_single_layers"] = 1
        model = FluxTransformer2DModel(**model_config)
        model.eval()

        batch_size = 1
        seq_len = 16

        hidden_states = torch.randn(batch_size, seq_len, model_config["in_channels"])
        encoder_hidden_states = torch.randn(batch_size, 77, model_config["joint_attention_dim"])
        pooled_projections = torch.randn(batch_size, model_config["pooled_projection_dim"])
        timestep = torch.tensor([500])

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                # 不提供 img_ids 和 txt_ids
            )

        assert output is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
