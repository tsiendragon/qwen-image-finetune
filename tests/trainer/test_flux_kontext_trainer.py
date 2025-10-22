import numpy as np
import torch
import logging
import sys
import os
import pytest
import glob
from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer
from qflux.data.config import load_config_from_yaml
from qflux.data.dataset import loader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # 关键：覆盖已有配置
)
logger = logging.getLogger(__name__)
logger.info("it works")


class TestFluxKontextTrainer:
    @pytest.fixture
    def example_config_path(self):
        """返回示例配置文件路径"""
        return "tests/test_configs/test_example_fluxkontext_fp4.yaml"

    def test_cache(self, example_config_path):
        # test cache
        config = load_config_from_yaml(example_config_path)
        trainer = FluxKontextLoraTrainer(config)
        dataloader = loader(config.data.class_path, config.data.init_args, 2, 1, False)
        trainer.cache(dataloader)
        logging.info('cache done')
        dataset_len = len(dataloader.dataset)
        cache_dir = config.cache.cache_dir
        metadatas = glob.glob(os.path.join(cache_dir, '*_metadata.json'))
        assert len(metadatas) == dataset_len, (f'metadata length {len(metadatas)} not'
                                               f' equal to dataset length {dataset_len}')
        logging.info('metadata done')

    def test_predict(self, example_config_path):
        config = load_config_from_yaml(example_config_path)
        trainer = FluxKontextLoraTrainer(config)

        dataloader = loader(config.data.class_path, config.data.init_args, 1, 1, False)

        for batch in dataloader:
            break

        prompt_image = batch['control'][0]
        prompt_image = prompt_image*255
        prompt_image = prompt_image.numpy().transpose(1, 2, 0).astype(np.uint8)

        out = trainer.predict(
            prompt_image=prompt_image,
            prompt="change the hair to yellow",
            num_inference_steps=20,
            true_cfg_scale=1.0,
            negative_prompt="",
            weight_dtype=torch.bfloat16,
            height=prompt_image.shape[0],
            width=prompt_image.shape[1],
            output_type='pil'
        )
        save_path = 'tests/test_flux_kontext_trainer.png'
        save_path = os.path.abspath(save_path)
        out[0].save(save_path)
        logging.info(f'save image to {save_path}')


if __name__ == '__main__':
    config_file = "tests/test_configs/test_example_fluxkontext_fp4.yaml"
    config = load_config_from_yaml(config_file)
    trainer = FluxKontextLoraTrainer(config)
    dataloader = loader(config.data.class_path, config.data.init_args, 2, 1, False)
    trainer.fit(dataloader)
