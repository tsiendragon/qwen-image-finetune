import pytest
import sys
from unittest.mock import Mock, patch
from qflux.utils.options import parse_args
from qflux.data.config import TrMode


class TestParseArgs:
    @pytest.fixture
    def mock_config(self):
        """Create a mock config object"""
        config = Mock()
        config.mode = TrMode.fit
        config.resume = ""
        return config

    def test_parse_args_with_config(self, monkeypatch, mock_config):
        """Test parsing args with config file"""
        test_args = ["script.py", "--config", "test_config.yaml"]
        monkeypatch.setattr(sys, "argv", test_args)

        with patch('qflux.utils.options.load_config_from_yaml', return_value=mock_config):
            config = parse_args()
            assert config is not None
            assert config.mode == TrMode.fit

    def test_parse_args_with_cache_flag(self, monkeypatch, mock_config):
        """Test parsing args with cache flag"""
        test_args = ["script.py", "--config", "test_config.yaml", "--cache"]
        monkeypatch.setattr(sys, "argv", test_args)

        with patch('qflux.utils.options.load_config_from_yaml', return_value=mock_config):
            config = parse_args()
            assert config.mode == TrMode.cache

    def test_parse_args_with_resume(self, monkeypatch, mock_config):
        """Test parsing args with resume checkpoint"""
        test_args = ["script.py", "--config", "test_config.yaml", "--resume", "/path/to/checkpoint"]
        monkeypatch.setattr(sys, "argv", test_args)

        with patch('qflux.utils.options.load_config_from_yaml', return_value=mock_config):
            config = parse_args()
            assert config.resume == "/path/to/checkpoint"

    def test_parse_args_missing_config(self, monkeypatch):
        """Test that missing config raises error"""
        test_args = ["script.py"]
        monkeypatch.setattr(sys, "argv", test_args)

        with pytest.raises(SystemExit):
            parse_args()

    def test_parse_args_empty_config(self, monkeypatch):
        """Test that empty config string raises error"""
        test_args = ["script.py", "--config", ""]
        monkeypatch.setattr(sys, "argv", test_args)

        with pytest.raises((ValueError, SystemExit)):
            parse_args()
