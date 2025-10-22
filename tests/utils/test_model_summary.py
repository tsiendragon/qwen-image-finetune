import torch
import torch.nn as nn
from qflux.utils.model_summary import (
    gather_model_stats,
    print_model_summary_table,
    _dtype_tag,
    _is_attention_module,
    _is_norm,
    _is_mlp_block,
    _human_int,
    _human_bytes,
)


class TestDtypeTag:
    def test_fp32(self):
        """Test FP32 dtype tag"""
        tag = _dtype_tag(torch.float32)
        assert tag == "fp32"

    def test_fp16(self):
        """Test FP16 dtype tag"""
        tag = _dtype_tag(torch.float16)
        assert tag == "fp16"

    def test_bf16(self):
        """Test BF16 dtype tag"""
        if hasattr(torch, "bfloat16"):
            tag = _dtype_tag(torch.bfloat16)
            assert tag == "bf16"

    def test_int8(self):
        """Test INT8 dtype tag"""
        tag = _dtype_tag(torch.int8)
        assert tag == "int8"


class TestModuleDetection:
    def test_is_attention_module(self):
        """Test attention module detection"""
        attn = nn.MultiheadAttention(64, 8)
        assert _is_attention_module(attn) is True

        linear = nn.Linear(64, 64)
        assert _is_attention_module(linear) is False

    def test_is_norm(self):
        """Test normalization module detection"""
        norm = nn.LayerNorm(64)
        assert _is_norm(norm) is True

        linear = nn.Linear(64, 64)
        assert _is_norm(linear) is False

    def test_is_mlp_block(self):
        """Test MLP block detection"""

        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 128)
                self.fc2 = nn.Linear(128, 64)

        mlp = SimpleMLP()
        assert _is_mlp_block(mlp) is True

        linear = nn.Linear(64, 64)
        assert _is_mlp_block(linear) is False


class TestFormattingHelpers:
    def test_human_int(self):
        """Test human-readable integer formatting"""
        assert _human_int(500) == "500"
        assert "K" in _human_int(5000)
        assert "M" in _human_int(5_000_000)
        assert "B" in _human_int(5_000_000_000)

    def test_human_bytes(self):
        """Test human-readable bytes formatting"""
        assert "B" in _human_bytes(500)
        assert "KB" in _human_bytes(5000)
        assert "MB" in _human_bytes(5_000_000)
        assert "GB" in _human_bytes(5_000_000_000)


class TestGatherModelStats:
    def test_gather_stats_simple_model(self):
        """Test gathering stats from simple model"""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))

        stats = gather_model_stats(model)

        assert "parameters" in stats
        assert "modules" in stats
        assert "transformer" in stats
        assert "lora" in stats
        assert "dtypes" in stats

        assert stats["parameters"]["total"] > 0
        assert stats["modules"]["total"] > 0

    def test_gather_stats_with_different_dtypes(self):
        """Test gathering stats with different dtypes"""
        model = nn.Linear(10, 10)
        model = model.to(torch.float16)

        stats = gather_model_stats(model)

        assert len(stats["dtypes"]) > 0
        # Should detect fp16
        dtype_names = [d["dtype"] for d in stats["dtypes"]]
        assert "fp16" in dtype_names

    def test_gather_stats_trainable_params(self):
        """Test gathering stats for trainable parameters"""
        model = nn.Linear(10, 10)

        # Freeze half the parameters
        for i, param in enumerate(model.parameters()):
            if i == 0:
                param.requires_grad = False

        stats = gather_model_stats(model)

        assert stats["parameters"]["trainable"] < stats["parameters"]["total"]
        assert stats["parameters"]["pct_trainable"] < 100.0

    def test_gather_stats_with_attention(self):
        """Test gathering stats for model with attention"""
        model = nn.MultiheadAttention(64, 8)

        stats = gather_model_stats(model)

        assert stats["transformer"]["attention"]["total"] > 0


class TestPrintModelSummaryTable:
    def test_print_summary_simple_model(self, capsys):
        """Test printing model summary"""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))

        stats = print_model_summary_table(model, name="TestModel")

        assert isinstance(stats, dict)
        assert "parameters" in stats
        assert "modules" in stats

    def test_print_summary_with_name(self, capsys):
        """Test printing summary with custom name"""
        model = nn.Linear(10, 10)
        stats = print_model_summary_table(model, name="CustomModel")

        assert isinstance(stats, dict)

    def test_summary_stats_consistency(self):
        """Test that summary stats are consistent"""
        model = nn.Linear(100, 50)

        stats = print_model_summary_table(model, name="TestModel")

        # Total params should equal sum of trainable and non-trainable
        total = stats["parameters"]["total"]
        trainable = stats["parameters"]["trainable"]

        assert trainable <= total
        assert stats["parameters"]["pct_trainable"] <= 100.0
