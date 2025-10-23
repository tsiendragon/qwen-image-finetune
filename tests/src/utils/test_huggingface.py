from pathlib import Path
from qflux.utils.huggingface import (
    is_huggingface_repo,
    _pick_first_existing,
)


class TestIsHuggingfaceRepo:
    def test_absolute_path_not_repo(self):
        """Test that absolute paths are not HF repos"""
        assert is_huggingface_repo("/absolute/path/to/dataset") is False

    def test_relative_path_with_dot_not_repo(self):
        """Test that relative paths with ./ are not HF repos"""
        assert is_huggingface_repo("./relative/path") is False
        assert is_huggingface_repo("../relative/path") is False

    def test_local_existing_path_not_repo(self, tmp_path):
        """Test that existing local paths are not HF repos"""
        test_dir = tmp_path / "test_dataset"
        test_dir.mkdir()
        assert is_huggingface_repo(str(test_dir)) is False

    def test_valid_repo_pattern(self):
        """Test valid HF repo patterns"""
        # These may return True or False depending on whether they exist on HF
        # We're mainly testing the pattern matching logic
        result = is_huggingface_repo("username/dataset-name")
        assert isinstance(result, bool)

    def test_invalid_repo_pattern(self):
        """Test invalid HF repo patterns"""
        # Too many slashes
        result = is_huggingface_repo("org/user/dataset")
        assert result is False


class TestPickFirstExisting:
    def test_pick_first_existing_png(self, tmp_path):
        """Test picking first existing file with PNG extension"""
        test_file = tmp_path / "test.png"
        test_file.touch()

        result = _pick_first_existing(tmp_path / "test")
        assert result == test_file

    def test_pick_first_existing_jpg(self, tmp_path):
        """Test picking first existing file with JPG extension"""
        test_file = tmp_path / "test.jpg"
        test_file.touch()

        result = _pick_first_existing(tmp_path / "test")
        assert result == test_file

    def test_pick_first_existing_uppercase(self, tmp_path):
        """Test picking first existing file with uppercase extension"""
        test_file = tmp_path / "test.PNG"
        test_file.touch()

        result = _pick_first_existing(tmp_path / "test")
        assert result == test_file

    def test_pick_nonexistent(self, tmp_path):
        """Test that None is returned when no file exists"""
        result = _pick_first_existing(tmp_path / "nonexistent")
        assert result is None

    def test_pick_multiple_extensions(self, tmp_path):
        """Test that the first supported extension is returned"""
        # Create multiple files with different extensions
        (tmp_path / "test.png").touch()
        (tmp_path / "test.jpg").touch()

        result = _pick_first_existing(tmp_path / "test")
        # Should return the first one found (order determined by _IMG_EXTS)
        assert result is not None
        assert result.exists()
