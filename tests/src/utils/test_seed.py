import pytest
import torch
import numpy as np
import random
import os
from qflux.utils.seed import seed_everything


class TestSeedEverything:
    def test_torch_random_state(self):
        """Test that torch random state is reproducible"""
        seed_everything(42)
        a = torch.rand(10)
        seed_everything(42)
        b = torch.rand(10)
        assert torch.allclose(a, b)

    def test_numpy_random_state(self):
        """Test that numpy random state is reproducible"""
        seed_everything(42)
        a = np.random.rand(10)
        seed_everything(42)
        b = np.random.rand(10)
        assert np.allclose(a, b)

    def test_python_random_state(self):
        """Test that Python random state is reproducible"""
        seed_everything(42)
        a = [random.random() for _ in range(10)]
        seed_everything(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_environment_variables(self):
        """Test that environment variables are set correctly"""
        seed = 123
        seed_everything(seed)
        assert os.environ["PYTHONHASHSEED"] == str(seed)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_random_state(self):
        """Test that CUDA random state is reproducible"""
        seed_everything(42)
        a = torch.cuda.FloatTensor(10).normal_()
        seed_everything(42)
        b = torch.cuda.FloatTensor(10).normal_()
        assert torch.allclose(a, b)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different random numbers"""
        seed_everything(42)
        a = torch.rand(10)
        seed_everything(123)
        b = torch.rand(10)
        assert not torch.allclose(a, b)
