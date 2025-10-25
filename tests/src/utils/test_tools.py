import pytest
import torch
from PIL import Image
import tempfile
import os
from qflux.utils.tools import (
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
    pad_latents_for_multi_res,
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


class TestPadLatentsForMultiRes:
    """Test suite for pad_latents_for_multi_res function"""

    def test_basic_padding(self):
        """Test basic padding with varying sequence lengths"""
        latents = [
            torch.randn(100, 64),
            torch.randn(150, 64),
            torch.randn(120, 64)
        ]
        max_seq_len = 150

        padded, mask = pad_latents_for_multi_res(latents, max_seq_len)

        # Check output shapes
        assert padded.shape == (3, 150, 64), f"Expected shape (3, 150, 64), got {padded.shape}"
        assert mask.shape == (3, 150), f"Expected mask shape (3, 150), got {mask.shape}"

        # Check mask correctness
        assert mask[0].sum() == 100, "First mask should have 100 True values"
        assert mask[1].sum() == 150, "Second mask should have 150 True values"
        assert mask[2].sum() == 120, "Third mask should have 120 True values"

        # Check that valid data is preserved
        assert torch.allclose(padded[0, :100], latents[0], atol=1e-6)
        assert torch.allclose(padded[1, :150], latents[1], atol=1e-6)
        assert torch.allclose(padded[2, :120], latents[2], atol=1e-6)

        # Check that padded regions are zero
        assert torch.all(padded[0, 100:] == 0)
        assert torch.all(padded[2, 120:] == 0)

    def test_single_latent(self):
        """Test with single latent tensor"""
        latents = [torch.randn(50, 32)]
        max_seq_len = 100

        padded, mask = pad_latents_for_multi_res(latents, max_seq_len)

        assert padded.shape == (1, 100, 32)
        assert mask.shape == (1, 100)
        assert mask[0].sum() == 50
        assert torch.all(padded[0, 50:] == 0)

    def test_all_same_length(self):
        """Test when all latents have the same length"""
        latents = [
            torch.randn(100, 64),
            torch.randn(100, 64),
            torch.randn(100, 64)
        ]
        max_seq_len = 100

        padded, mask = pad_latents_for_multi_res(latents, max_seq_len)

        assert padded.shape == (3, 100, 64)
        assert torch.all(mask)  # All should be True

    def test_device_dtype_preservation(self):
        """Test that device and dtype are preserved from first latent"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        latents = [
            torch.randn(50, 32, device=device, dtype=torch.float32),
            torch.randn(60, 32, device=device, dtype=torch.float32),
        ]
        max_seq_len = 80

        padded, mask = pad_latents_for_multi_res(latents, max_seq_len)

        assert padded.device.type == device.type
        assert padded.dtype == torch.float32
        assert mask.device.type == device.type
        assert mask.dtype == torch.bool

    def test_mixed_device_handling(self):
        """Test that latents on different devices are moved to first latent's device"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        latents = [
            torch.randn(50, 32, device="cuda"),
            torch.randn(60, 32, device="cpu"),
        ]
        max_seq_len = 80

        padded, mask = pad_latents_for_multi_res(latents, max_seq_len)

        assert padded.device.type == "cuda"
        assert torch.allclose(padded[1, :60].cpu(), latents[1], atol=1e-6)

    def test_empty_list_error(self):
        """Test error when passing empty list"""
        with pytest.raises(ValueError, match="Cannot pad empty latent list"):
            pad_latents_for_multi_res([], 100)

    def test_wrong_dimension_error(self):
        """Test error when latent is not 2D"""
        latents = [
            torch.randn(50, 32),
            torch.randn(60, 32, 16),  # 3D tensor
        ]
        with pytest.raises(ValueError, match="Expected 2D latent tensor"):
            pad_latents_for_multi_res(latents, 100)

    def test_channel_mismatch_error(self):
        """Test error when channel dimensions don't match"""
        latents = [
            torch.randn(50, 32),
            torch.randn(60, 64),  # Different channel size
        ]
        with pytest.raises(ValueError, match="Channel mismatch"):
            pad_latents_for_multi_res(latents, 100)

    def test_exceeds_max_seq_len_error(self):
        """Test error when latent sequence exceeds max_seq_len"""
        latents = [
            torch.randn(50, 32),
            torch.randn(150, 32),  # Exceeds max_seq_len
        ]
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            pad_latents_for_multi_res(latents, 100)

    def test_mask_can_be_used_for_loss(self):
        """Test that mask can be used for masked loss computation"""
        latents = [
            torch.randn(100, 64),
            torch.randn(150, 64),
            torch.randn(120, 64)
        ]
        max_seq_len = 150

        padded, mask = pad_latents_for_multi_res(latents, max_seq_len)

        # Simulate model prediction and target
        pred = torch.randn_like(padded)
        target = torch.randn_like(padded)

        # Compute masked loss
        valid_pred = pred[mask]
        valid_target = target[mask]

        assert valid_pred.shape[0] == (100 + 150 + 120), "Should have correct number of valid elements"
        assert valid_pred.shape[1] == 64, "Should preserve channel dimension"
        assert valid_target.shape == valid_pred.shape, "Target and pred should have same shape"

    def test_various_dtypes(self):
        """Test with different data types"""
        for dtype in [torch.float16, torch.float32, torch.float64]:
            latents = [
                torch.randn(50, 32, dtype=dtype),
                torch.randn(60, 32, dtype=dtype),
            ]
            max_seq_len = 80

            padded, mask = pad_latents_for_multi_res(latents, max_seq_len)

            assert padded.dtype == dtype
            assert mask.dtype == torch.bool

    def test_large_batch(self):
        """Test with large batch size"""
        batch_size = 32
        latents = [torch.randn(50 + i * 5, 64) for i in range(batch_size)]
        # Max length will be 50 + 31*5 = 205, so set max_seq_len to 210
        max_seq_len = 210

        padded, mask = pad_latents_for_multi_res(latents, max_seq_len)

        assert padded.shape == (batch_size, 210, 64)
        assert mask.shape == (batch_size, 210)

        # Verify each sample's mask
        for i in range(batch_size):
            expected_len = 50 + i * 5
            assert mask[i].sum() == expected_len

    def test_edge_case_max_seq_equals_longest(self):
        """Test when max_seq_len equals the longest sequence"""
        latents = [
            torch.randn(100, 64),
            torch.randn(150, 64),
            torch.randn(120, 64)
        ]
        max_seq_len = 150  # Equals longest

        padded, mask = pad_latents_for_multi_res(latents, max_seq_len)

        assert padded.shape == (3, 150, 64)
        assert mask[1].all()  # Second item should have all True
        assert mask[1].sum() == 150
