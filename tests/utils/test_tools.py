import pytest
import torch
from PIL import Image
import tempfile
import os
from src.utils.tools import (
    sample_indices_per_rank,
    content_hash_blake3,
    calculate_md5,
    phash_hex_from_image,
    hash_string_md5,
    extract_file_hash,
    get_git_info,
    infer_image_tensor,
    extract_batch_field,
    calculate_sha256_file,
)


class TestSampleIndicesPerRank:
    @pytest.fixture
    def mock_accelerator(self):
        """Create a mock accelerator"""
        class MockAccelerator:
            def __init__(self, process_index=0, num_processes=1):
                self.process_index = process_index
                self.num_processes = num_processes
        return MockAccelerator

    def test_single_rank(self, mock_accelerator):
        """Test sampling with single rank"""
        accelerator = mock_accelerator(0, 1)
        indices = sample_indices_per_rank(accelerator, 100, 10, seed=42)
        assert len(indices) == 10
        assert all(0 <= idx < 100 for idx in indices)

    def test_multiple_ranks_no_overlap(self, mock_accelerator):
        """Test that different ranks get different indices"""
        accelerator0 = mock_accelerator(0, 2)
        accelerator1 = mock_accelerator(1, 2)

        indices0 = sample_indices_per_rank(accelerator0, 100, 10, seed=42, global_shuffle=True)
        indices1 = sample_indices_per_rank(accelerator1, 100, 10, seed=42, global_shuffle=True)

        # No overlap between ranks
        assert len(set(indices0) & set(indices1)) == 0

    def test_with_replacement(self, mock_accelerator):
        """Test sampling with replacement"""
        accelerator = mock_accelerator(0, 1)
        indices = sample_indices_per_rank(accelerator, 10, 50, seed=42, replacement=True)
        assert len(indices) == 50

    def test_insufficient_samples_error(self, mock_accelerator):
        """Test error when requesting too many samples without replacement"""
        accelerator = mock_accelerator(0, 2)
        with pytest.raises(ValueError):
            sample_indices_per_rank(accelerator, 10, 20, seed=42, replacement=False)


class TestHashFunctions:
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_content_hash_blake3(self, temp_file):
        """Test BLAKE3 hashing"""
        hash1 = content_hash_blake3(temp_file)
        hash2 = content_hash_blake3(temp_file)
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # BLAKE3 produces 256-bit (64 hex chars) hash

    def test_calculate_md5(self, temp_file):
        """Test MD5 hashing"""
        hash1 = calculate_md5(temp_file)
        hash2 = calculate_md5(temp_file)
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 produces 128-bit (32 hex chars) hash

    def test_calculate_sha256(self, temp_file):
        """Test SHA256 hashing"""
        hash1 = calculate_sha256_file(temp_file)
        hash2 = calculate_sha256_file(temp_file)
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 produces 256-bit (64 hex chars) hash

    def test_hash_string_md5(self):
        """Test MD5 hashing of strings"""
        hash1 = hash_string_md5("test string")
        hash2 = hash_string_md5("test string")
        assert hash1 == hash2
        assert len(hash1) == 32

    def test_phash_hex_from_image(self):
        """Test perceptual hash from PIL image"""
        img = Image.new("RGB", (100, 100), color="red")
        hash1 = phash_hex_from_image(img)
        hash2 = phash_hex_from_image(img)
        assert hash1 == hash2
        assert isinstance(hash1, str)

    def test_extract_file_hash_with_path(self, temp_file):
        """Test extracting hash from file path"""
        file_hash = extract_file_hash(temp_file)
        assert isinstance(file_hash, str)
        assert len(file_hash) == 32  # MD5

    def test_extract_file_hash_with_image(self):
        """Test extracting hash from PIL image"""
        img = Image.new("RGB", (100, 100))
        file_hash = extract_file_hash(img)
        assert isinstance(file_hash, str)

    def test_extract_file_hash_invalid(self):
        """Test error with invalid input"""
        with pytest.raises(ValueError):
            extract_file_hash("/nonexistent/path")


class TestGitInfo:
    def test_get_git_info(self):
        """Test getting git information"""
        info = get_git_info()
        assert isinstance(info, dict)
        assert "commit" in info
        assert "short_commit" in info
        assert "branch" in info
        assert "remote" in info
        assert "root" in info


class TestInferImageTensor:
    def test_infer_hw_layout(self):
        """Test inferring HW layout"""
        t = torch.randn(64, 64)
        info = infer_image_tensor(t)
        assert info["layout"] == "HW"
        assert info["height"] == 64
        assert info["width"] == 64

    def test_infer_chw_layout(self):
        """Test inferring CHW layout"""
        t = torch.randn(3, 64, 64)
        info = infer_image_tensor(t)
        assert info["layout"] == "CHW"
        assert info["channels"] == 3
        assert info["height"] == 64
        assert info["width"] == 64

    def test_infer_hwc_layout(self):
        """Test inferring HWC layout"""
        t = torch.randn(64, 64, 3)
        info = infer_image_tensor(t)
        assert info["layout"] == "HWC"
        assert info["channels"] == 3
        assert info["height"] == 64
        assert info["width"] == 64

    def test_infer_bchw_layout(self):
        """Test inferring BCHW layout"""
        t = torch.randn(4, 3, 64, 64)
        info = infer_image_tensor(t)
        assert info["layout"] == "BCHW"
        assert info["batch"] == 4
        assert info["channels"] == 3
        assert info["height"] == 64
        assert info["width"] == 64

    def test_infer_bhwc_layout(self):
        """Test inferring BHWC layout"""
        t = torch.randn(4, 64, 64, 3)
        info = infer_image_tensor(t)
        assert info["layout"] == "BHWC"
        assert info["batch"] == 4
        assert info["channels"] == 3
        assert info["height"] == 64
        assert info["width"] == 64

    def test_infer_range_0_1(self):
        """Test inferring 0-1 range"""
        t = torch.rand(3, 64, 64)
        info = infer_image_tensor(t)
        assert info["range"] == "0-1"

    def test_infer_range_minus1_1(self):
        """Test inferring -1-1 range"""
        t = torch.rand(3, 64, 64) * 2 - 1
        info = infer_image_tensor(t)
        assert info["range"] == "-1-1"

    def test_infer_range_0_255(self):
        """Test inferring 0-255 range"""
        t = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        info = infer_image_tensor(t)
        assert info["range"] == "0-255"

    def test_non_tensor_error(self):
        """Test error with non-tensor input"""
        with pytest.raises(TypeError):
            infer_image_tensor([1, 2, 3])


class TestExtractBatchField:
    def test_extract_from_list(self):
        """Test extracting from list"""
        embeddings = {"height": [512, 640, 768]}
        assert extract_batch_field(embeddings, "height", 0) == 512
        assert extract_batch_field(embeddings, "height", 1) == 640
        assert extract_batch_field(embeddings, "height", 2) == 768

    def test_extract_from_tuple(self):
        """Test extracting from tuple"""
        embeddings = {"width": (512, 640, 768)}
        assert extract_batch_field(embeddings, "width", 0) == 512
        assert extract_batch_field(embeddings, "width", 1) == 640

    def test_extract_from_tensor(self):
        """Test extracting from tensor"""
        embeddings = {"height": torch.tensor([512, 640, 768])}
        assert extract_batch_field(embeddings, "height", 0) == 512
        assert extract_batch_field(embeddings, "height", 1) == 640

    def test_extract_scalar(self):
        """Test extracting scalar value"""
        embeddings = {"width": 512}
        assert extract_batch_field(embeddings, "width", 0) == 512
        assert extract_batch_field(embeddings, "width", 1) == 512

    def test_extract_single_element_tensor(self):
        """Test extracting single element tensor"""
        embeddings = {"scale": torch.tensor(2.0)}
        assert extract_batch_field(embeddings, "scale", 0) == 2.0
