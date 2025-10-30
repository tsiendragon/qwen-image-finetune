from unittest.mock import Mock, patch, MagicMock, mock_open
from qflux.utils.get_model_config import get_pretrained_model_config, compare_with_local_config


class TestGetPretrainedModelConfig:
    @patch("qflux.utils.get_model_config.QwenImageTransformer2DModel")
    def test_get_pretrained_model_config_success(self, mock_model_class):
        """Test successfully getting pretrained model config"""
        # Setup mock model using configure_mock to avoid attribute setting issues
        mock_model = MagicMock()

        # Create config as a simple object with __dict__
        class MockConfig:
            def __init__(self):
                self.num_layers = 60
                self.num_attention_heads = 24
                self.attention_head_dim = 128
                self.patch_size = 2

        mock_config = MockConfig()

        # Configure mock model
        mock_model.configure_mock(config=mock_config)

        # Mock parameters method
        mock_param = MagicMock()
        mock_param.numel.return_value = 1000000
        mock_model.parameters.return_value = [mock_param]

        mock_model_class.from_pretrained.return_value = mock_model

        # Call function
        with patch("builtins.open", mock_open()):
            with patch("json.dump"):
                config_dict = get_pretrained_model_config()

        # Verify
        assert config_dict is not None
        assert "num_layers" in config_dict
        mock_model_class.from_pretrained.assert_called_once()

    @patch("qflux.utils.get_model_config.QwenImageTransformer2DModel")
    def test_get_pretrained_model_config_failure(self, mock_model_class):
        """Test handling of model loading failure"""
        mock_model_class.from_pretrained = Mock(side_effect=Exception("Network error"))

        config_dict = get_pretrained_model_config()

        assert config_dict is None

    @patch("qflux.utils.get_model_config.QwenImageTransformer2DModel")
    def test_get_pretrained_model_config_saves_json(self, mock_model_class):
        """Test that config is saved to JSON file"""
        # Setup mock model
        mock_model = MagicMock()

        # Create config as a simple object
        class MockConfig:
            def __init__(self):
                self.test_param = 123

        mock_config = MockConfig()
        mock_model.configure_mock(config=mock_config)
        mock_model.parameters.return_value = []

        mock_model_class.from_pretrained.return_value = mock_model

        # Call function with mocked file operations
        with patch("builtins.open", mock_open()):
            with patch("json.dump") as mock_json_dump:
                get_pretrained_model_config()

                # Verify JSON dump was called
                mock_json_dump.assert_called_once()


class TestCompareWithLocalConfig:
    @patch("qflux.utils.get_model_config.QwenImageTransformer2DModel")
    def test_compare_with_local_config_success(self, mock_model_class):
        """Test comparing local config with pretrained"""
        # Setup mock local model
        mock_local_model = Mock()
        mock_local_model.parameters = Mock(return_value=[Mock(numel=Mock(return_value=500000))])

        mock_model_class.return_value = mock_local_model

        # Call function - should not raise
        compare_with_local_config()

        # Verify model was created with expected config
        mock_model_class.assert_called_once()
        call_kwargs = mock_model_class.call_args[1]
        assert "patch_size" in call_kwargs
        assert "num_layers" in call_kwargs

    @patch("qflux.utils.get_model_config.QwenImageTransformer2DModel")
    def test_compare_with_local_config_failure(self, mock_model_class):
        """Test handling of local model creation failure"""
        mock_model_class.side_effect = Exception("Invalid config")

        # Should not raise, just print error
        compare_with_local_config()
