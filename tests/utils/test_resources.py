"""
Test resources downloader from HuggingFace Hub.

This module provides utilities to download test resources from HuggingFace
instead of storing them in git repository.
"""
from pathlib import Path
from typing import Optional, List
import logging
import yaml

logger = logging.getLogger(__name__)

# Default configuration file path
RESOURCES_CONFIG_PATH = Path(__file__).parent.parent / "resources_config.yaml"


def load_resources_config(config_path: Optional[Path] = None) -> dict:
    """
    Load test resources configuration from YAML file.

    Args:
        config_path: Path to the configuration file (default: resources_config.yaml)

    Returns:
        Configuration dictionary
    """
    config_path = config_path or RESOURCES_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Resources config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def get_test_resources_dir() -> Path:
    """Get the local test resources directory."""
    return Path(__file__).parent.parent / "resources"


def get_resource_groups_for_test(test_file: str, config: Optional[dict] = None) -> List[str]:
    """
    Get resource groups needed for a specific test file.

    Args:
        test_file: Test file path (relative to project root)
        config: Configuration dictionary (loaded if not provided)

    Returns:
        List of resource group names
    """
    if config is None:
        config = load_resources_config()

    test_dependencies = config.get('test_dependencies', {})

    # Normalize test file path
    test_file = str(test_file).replace('\\', '/')

    # Find matching test file
    for test_pattern, groups in test_dependencies.items():
        if test_file.endswith(test_pattern) or test_pattern in test_file:
            return groups

    return []


def get_files_for_groups(group_names: List[str], config: Optional[dict] = None) -> List[str]:
    """
    Get all files for specified resource groups.

    Args:
        group_names: List of resource group names
        config: Configuration dictionary (loaded if not provided)

    Returns:
        List of file paths relative to resources directory
    """
    if config is None:
        config = load_resources_config()

    resource_groups = config.get('resource_groups', {})
    files = []

    for group_name in group_names:
        if group_name in resource_groups:
            group_files = resource_groups[group_name].get('files', [])
            files.extend(group_files)

    return files


def download_specific_files(
    files: List[str],
    force_download: bool = False,
    config: Optional[dict] = None,
) -> Path:
    """
    Download specific files from HuggingFace Hub.

    Args:
        files: List of file paths to download (relative to repo root)
        force_download: If True, re-download even if files exist locally
        config: Configuration dictionary (loaded if not provided)

    Returns:
        Path to the local resources directory

    Raises:
        ImportError: If huggingface_hub is not installed
        Exception: If download fails
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download test resources. "
            "Install it with: pip install huggingface_hub"
        )

    if config is None:
        config = load_resources_config()

    repo_config = config.get('repository', {})
    repo_id = repo_config.get('repo_id')
    repo_type = repo_config.get('repo_type', 'dataset')
    revision = repo_config.get('revision', 'main')

    if not repo_id:
        raise ValueError("Repository ID not found in configuration")

    resources_dir = get_test_resources_dir()
    resources_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {len(files)} files from {repo_id}@{revision}...")

    downloaded_count = 0
    for file_path in files:
        local_file = resources_dir / file_path

        # Skip if file exists and not forcing download
        if not force_download and local_file.exists():
            logger.debug(f"File already exists: {file_path}")
            continue

        # Create parent directory if needed
        local_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Download file
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=file_path,
                revision=revision,
                local_dir=resources_dir,
                local_dir_use_symlinks=False,
                force_download=force_download,
            )
            logger.debug(f"Downloaded: {file_path}")
            downloaded_count += 1

        except Exception as e:
            logger.error(f"Failed to download {file_path}: {e}")
            raise

    if downloaded_count > 0:
        logger.info(f"Downloaded {downloaded_count} new files")
    else:
        logger.info("All files already exist locally")

    return resources_dir


def download_resource_groups(
    group_names: List[str],
    force_download: bool = False,
    config: Optional[dict] = None,
) -> Path:
    """
    Download specific resource groups from HuggingFace Hub.

    Args:
        group_names: List of resource group names to download
        force_download: If True, re-download even if files exist locally
        config: Configuration dictionary (loaded if not provided)

    Returns:
        Path to the local resources directory
    """
    if config is None:
        config = load_resources_config()

    files = get_files_for_groups(group_names, config)

    if not files:
        logger.warning(f"No files found for groups: {group_names}")
        return get_test_resources_dir()

    logger.info(f"Downloading resource groups: {', '.join(group_names)}")
    return download_specific_files(files, force_download, config)


def download_all_resources(
    force_download: bool = False,
    config: Optional[dict] = None,
) -> Path:
    """
    Download all test resources from HuggingFace Hub.

    Args:
        force_download: If True, re-download even if files exist locally
        config: Configuration dictionary (loaded if not provided)

    Returns:
        Path to the local resources directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download test resources. "
            "Install it with: pip install huggingface_hub"
        )

    if config is None:
        config = load_resources_config()

    repo_config = config.get('repository', {})
    repo_id = repo_config.get('repo_id')
    repo_type = repo_config.get('repo_type', 'dataset')
    revision = repo_config.get('revision', 'main')

    if not repo_id:
        raise ValueError("Repository ID not found in configuration")

    resources_dir = get_test_resources_dir()

    # Check if resources already exist
    if not force_download and resources_dir.exists() and any(resources_dir.iterdir()):
        logger.info(f"Test resources already exist at {resources_dir}")
        return resources_dir

    logger.info(f"Downloading all test resources from {repo_id}@{revision}...")

    try:
        # Download the entire repository to resources directory
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=resources_dir,
            local_dir_use_symlinks=False,
            force_download=force_download,
        )
        logger.info(f"All test resources downloaded successfully to {resources_dir}")
        return resources_dir

    except Exception as e:
        logger.error(f"Failed to download test resources: {e}")
        raise


def ensure_test_resources(
    test_file: Optional[str] = None,
    download_all: bool = False,
) -> Path:
    """
    Ensure test resources are available locally from HuggingFace Hub.
    Download them if they don't exist.

    Data Source:
        - HuggingFace Repository: TsienDragon/qwen-image-finetune-test-resources
        - Repository Type: dataset
        - Configuration: tests/resources_config.yaml

    Args:
        test_file: Specific test file path (downloads only needed resources)
        download_all: If True, download all resources regardless of test_file

    Returns:
        Path to the local resources directory
    """
    resources_dir = get_test_resources_dir()

    # If download_all is True, download everything
    if download_all:
        logger.info("📥 Downloading all test resources from HuggingFace Hub...")
        return download_all_resources()

    # If test_file is specified, download only needed resources
    if test_file:
        try:
            config = load_resources_config()
            groups = get_resource_groups_for_test(test_file, config)

            if groups:
                logger.info(f"📦 Test {test_file} requires groups: {', '.join(groups)}")
                logger.info("🔄 Checking/downloading from HuggingFace Hub...")
                return download_resource_groups(groups, force_download=False, config=config)
            else:
                logger.info(f"ℹ️  No specific resources required for {test_file}")
                return resources_dir

        except Exception as e:
            logger.warning(f"⚠️  Failed to download specific resources: {e}")
            logger.info("🔄 Falling back to check existing resources...")

    # If resources directory doesn't exist or is empty, download all
    if not resources_dir.exists() or not any(resources_dir.iterdir()):
        logger.info("📥 No local resources found, downloading all from HuggingFace Hub...")
        return download_all_resources()

    logger.info(f"✅ Using cached test resources from: {resources_dir}")
    return resources_dir
