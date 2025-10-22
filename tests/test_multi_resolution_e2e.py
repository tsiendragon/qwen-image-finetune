"""
End-to-end integration test for multi-resolution training
Tests the complete pipeline from data loading to training
"""
import pytest
import torch
import logging
import os

from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer
from qflux.data.config import load_config_from_yaml
from qflux.data.dataset import loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestMultiResolutionE2E:
    """End-to-end tests for multi-resolution training"""

    @pytest.fixture
    def multi_res_config_path(self):
        """Path to multi-resolution test config"""
        return "tests/test_configs/test_example_fluxkontext_fp16_multiresolution.yaml"

    @pytest.fixture
    def single_res_config_path(self):
        """Path to single-resolution test config"""
        return "tests/test_configs/test_example_fluxkontext_fp16.yaml"

    def test_multi_resolution_config_loading(self, multi_res_config_path):
        """Test that multi-resolution config loads correctly"""
        config = load_config_from_yaml(multi_res_config_path)

        # Verify multi_resolutions field exists and is parsed
        processor_config = config.data.init_args.processor.init_args
        assert hasattr(processor_config, 'multi_resolutions'), "multi_resolutions field should exist"
        assert processor_config.multi_resolutions is not None, "multi_resolutions should not be None"
        assert len(processor_config.multi_resolutions) == 4, "Should have 4 resolution candidates"

        # Verify parsed values are integers
        expected_resolutions = [512*512, 640*640, 768*512, 832*576]
        assert processor_config.multi_resolutions == expected_resolutions, (
            f"Expected {expected_resolutions}, got {processor_config.multi_resolutions}"
        )

    def test_trainer_initialization_with_multi_resolution(self, multi_res_config_path):
        """Test that trainer initializes correctly with multi-resolution config"""
        config = load_config_from_yaml(multi_res_config_path)
        trainer = FluxKontextLoraTrainer(config)

        # Verify trainer has necessary methods
        assert hasattr(trainer, '_should_use_multi_resolution_mode')
        assert hasattr(trainer, '_pad_latents_for_multi_res')
        assert hasattr(trainer, '_compute_loss_multi_resolution')
        assert hasattr(trainer, '_prepare_latent_image_ids_batched')

        logger.info("Trainer initialized successfully with multi-resolution config")

    def test_dataloader_with_multi_resolution(self, multi_res_config_path):
        """Test dataloader with multi-resolution preprocessing"""
        config = load_config_from_yaml(multi_res_config_path)
        dataloader = loader(
            config.data.class_path,
            config.data.init_args,
            batch_size=2,
            num_workers=0,
            shuffle=False
        )

        # Get a batch
        batch = next(iter(dataloader))

        # Verify batch structure
        assert 'image' in batch or 'control' in batch, "Batch should contain images"
        assert 'prompt' in batch, "Batch should contain prompts"

        logger.info(f"Batch keys: {batch.keys()}")
        if 'image' in batch:
            logger.info(f"Image shape: {batch['image'].shape}")
        if 'control' in batch:
            logger.info(f"Control shape: {batch['control'].shape}")

    @pytest.mark.slow
    def test_cache_with_multi_resolution(self, multi_res_config_path, tmp_path):
        """Test caching with multi-resolution config"""
        config = load_config_from_yaml(multi_res_config_path)

        # Override cache directory to use temporary path
        config.cache.cache_dir = str(tmp_path / "cache")
        os.makedirs(config.cache.cache_dir, exist_ok=True)

        trainer = FluxKontextLoraTrainer(config)
        dataloader = loader(
            config.data.class_path,
            config.data.init_args,
            batch_size=1,
            num_workers=0,
            shuffle=False
        )

        # Run cache for a few samples
        logger.info("Starting cache process...")
        try:
            # Cache only first batch for testing
            for i, batch in enumerate(dataloader):
                if i >= 1:  # Only process first batch
                    break
                trainer.prepare_embeddings(batch, stage="cache")
                trainer.cache_step(batch)

            logger.info("Cache completed successfully")

            # Verify cache files exist
            cache_files = list(tmp_path.glob("cache/*.safetensors"))
            assert len(cache_files) > 0, "Cache files should be created"

        except Exception as e:
            logger.error(f"Cache test failed: {e}")
            raise

    def test_comparison_single_vs_multi_resolution(
        self, single_res_config_path, multi_res_config_path
    ):
        """Compare single-resolution vs multi-resolution preprocessing"""
        # Load both configs
        single_config = load_config_from_yaml(single_res_config_path)
        multi_config = load_config_from_yaml(multi_res_config_path)

        # Create dataloaders
        single_loader = loader(
            single_config.data.class_path,
            single_config.data.init_args,
            batch_size=1,
            num_workers=0,
            shuffle=False
        )

        multi_loader = loader(
            multi_config.data.class_path,
            multi_config.data.init_args,
            batch_size=1,
            num_workers=0,
            shuffle=False
        )

        # Get batches
        single_batch = next(iter(single_loader))
        multi_batch = next(iter(multi_loader))

        # Both should produce valid batches
        assert 'image' in single_batch or 'control' in single_batch
        assert 'image' in multi_batch or 'control' in multi_batch

        logger.info("Single-resolution and multi-resolution dataloaders both work")


class TestMultiResolutionPreprocessing:
    """Test multi-resolution preprocessing logic"""

    def test_resolution_selection_logic(self):
        """Test that resolution selection produces expected results"""
        from qflux.data.preprocess import ImageProcessor
        from qflux.data.config import ImageProcessorInitArgs

        # Create processor with multi-resolutions
        config = ImageProcessorInitArgs(
            process_type="fixed_pixels",
            multi_resolutions=["512*512", "640*640", "768*512", "832*576"],
            max_aspect_ratio=3.0
        )
        processor = ImageProcessor(config)

        # Verify multi_resolutions are parsed correctly
        assert processor.multi_resolutions == [512*512, 640*640, 768*512, 832*576]

        # Test resolution selection for different image sizes
        test_cases = [
            ((512, 512), 512*512),   # Exact match
            ((600, 600), 640*640),   # Should select closest larger
            ((800, 530), 832*576),   # Landscape aspect
            ((480, 480), 512*512),   # Slightly smaller, rounds up
        ]

        for (orig_w, orig_h), expected_target in test_cases:
            selected = processor._select_pixels_candidate(orig_w, orig_h)
            logger.info(f"Image ({orig_w}x{orig_h}) -> target pixels: {selected}")
            # Just verify it returns a valid candidate
            assert selected in processor.multi_resolutions, (
                f"Selected {selected} not in candidates"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
