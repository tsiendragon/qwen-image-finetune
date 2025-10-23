#!/usr/bin/env python3
"""
Upload test resources to HuggingFace Hub.

Usage:
    python scripts/upload_test_resources.py --token YOUR_HF_TOKEN

    # Or set HF_TOKEN environment variable
    export HF_TOKEN=your_token
    python scripts/upload_test_resources.py
"""
import argparse
import os
import sys
from pathlib import Path
import yaml


def load_config():
    """Load resources configuration."""
    config_path = Path(__file__).parent.parent / "tests" / "resources_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def upload_resources(
    resources_dir: Path,
    repo_id: str,
    token: str,
    repo_type: str = "dataset",
    private: bool = False,
):
    """
    Upload test resources to HuggingFace Hub.

    Args:
        resources_dir: Path to the organized resources directory
        repo_id: HuggingFace repository ID
        token: HuggingFace API token
        repo_type: Repository type (default: "dataset")
        private: Whether to create a private repository
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Error: huggingface_hub is required")
        print("Install it with: pip install huggingface_hub")
        sys.exit(1)

    if not resources_dir.exists():
        print(f"Error: Resources directory not found: {resources_dir}")
        sys.exit(1)

    api = HfApi(token=token)

    # Create repository if it doesn't exist
    print(f"Creating/checking repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
            token=token,
        )
        print(f"✓ Repository ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        sys.exit(1)

    # Upload entire directory
    print(f"\nUploading files from {resources_dir}...")
    try:
        api.upload_folder(
            folder_path=str(resources_dir),
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message="Upload test resources with organized structure",
        )
        print(f"\n✓ Successfully uploaded all files to {repo_id}")
        print(f"  View at: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"\nError uploading files: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Upload test resources to HuggingFace Hub"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace API token (default: use HF_TOKEN env var)",
    )
    parser.add_argument(
        "--resources-dir",
        type=Path,
        default=Path(__file__).parent.parent / "test_resources_organized",
        help="Path to organized resources directory",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace repository ID (overrides config)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository",
    )

    args = parser.parse_args()

    # Use token from args or environment variable
    token = args.token or os.environ.get("HF_TOKEN")

    if not token:
        print("Error: HuggingFace token is required")
        print("Provide it via:")
        print("  1. Environment variable: export HF_TOKEN=your_token")
        print("  2. Command argument: --token your_token")
        print("\nGet your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)

    # Load config
    config = load_config()
    repo_config = config.get('repository', {})

    repo_id = args.repo_id or repo_config.get('repo_id')
    repo_type = repo_config.get('repo_type', 'dataset')

    if not repo_id:
        print("Error: Repository ID not found in config or arguments")
        sys.exit(1)

    print("=" * 60)
    print("Upload Test Resources to HuggingFace Hub")
    print("=" * 60)
    print(f"Resources directory: {args.resources_dir}")
    print(f"Repository ID: {repo_id}")
    print(f"Repository type: {repo_type}")
    print(f"Private: {args.private}")
    print("=" * 60)

    # Confirm
    response = input("\nProceed with upload? [y/N]: ")
    if response.lower() != 'y':
        print("Upload cancelled")
        sys.exit(0)

    # Upload
    upload_resources(
        resources_dir=args.resources_dir,
        repo_id=repo_id,
        token=token,
        repo_type=repo_type,
        private=args.private,
    )

    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Verify files at: https://huggingface.co/datasets/" + repo_id)
    print("2. Run tests to verify download: pytest tests/")
    print("3. Remove local resources: rm -rf tests/resources/")
    print("=" * 60)


if __name__ == "__main__":
    main()
