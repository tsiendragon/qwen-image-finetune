"""
Model Comparison Utilities
Provides functions to compare models loaded with different methods (pipeline vs direct loading).
"""

import logging
from typing import Any

import numpy as np
import torch


logger = logging.getLogger(__name__)


def compare_model_parameters(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    relative_threshold: float = 1e-6,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Compare two PyTorch models by analyzing parameter keys, shapes, and values.

    Args:
        model1: First model to compare
        model2: Second model to compare
        model1_name: Name for first model (for logging)
        model2_name: Name for second model (for logging)
        relative_threshold: Threshold for considering parameters as different
        verbose: Whether to print detailed comparison results

    Returns:
        Dictionary containing comparison results
    """
    logger.info(f"Comparing {model1_name} vs {model2_name}")

    # Get state dictionaries
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # Initialize results
    results: dict[str, Any] = {
        "summary": {},
        "key_differences": [],
        "shape_differences": [],
        "value_differences": [],
        "missing_keys": {"model1_missing": [], "model2_missing": []},
        "statistics": {},
    }

    # Get all unique keys
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    common_keys = keys1.intersection(keys2)

    # Find missing keys
    results["missing_keys"]["model1_missing"] = list(keys2 - keys1)
    results["missing_keys"]["model2_missing"] = list(keys1 - keys2)

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"MODEL COMPARISON: {model1_name} vs {model2_name}")
        print(f"{'=' * 80}")
        print(f"Total parameters in {model1_name}: {len(keys1)}")
        print(f"Total parameters in {model2_name}: {len(keys2)}")
        print(f"Common parameters: {len(common_keys)}")
        print(f"Missing in {model1_name}: {len(results['missing_keys']['model1_missing'])}")
        print(f"Missing in {model2_name}: {len(results['missing_keys']['model2_missing'])}")

    # Analyze common parameters
    shape_matches = 0
    value_matches = 0
    total_relative_differences = []

    for key in sorted(common_keys):
        param1 = state_dict1[key]
        param2 = state_dict2[key]

        # Check shapes
        shape_match = param1.shape == param2.shape
        if shape_match:
            shape_matches += 1
        else:
            results["shape_differences"].append({"key": key, "shape1": param1.shape, "shape2": param2.shape})
            if verbose:
                print(f"Shape mismatch for {key}: {param1.shape} vs {param2.shape}")
            continue

        # Calculate relative difference: norm(x-y) / (1e-4 + norm(x))
        param1_flat = param1.flatten().float()
        param2_flat = param2.flatten().float()

        diff_norm = torch.norm(param1_flat - param2_flat).item()
        param1_norm = torch.norm(param1_flat).item()

        # Avoid division by zero
        relative_diff = diff_norm / (1e-4 + param1_norm)
        total_relative_differences.append(relative_diff)

        # Check if values are significantly different
        if relative_diff > relative_threshold:
            results["value_differences"].append(
                {
                    "key": key,
                    "relative_difference": relative_diff,
                    "absolute_difference": diff_norm,
                    "param1_norm": param1_norm,
                    "param2_norm": torch.norm(param2_flat).item(),
                    "shape": param1.shape,
                }
            )
            if verbose and len(results["value_differences"]) <= 10:  # Limit verbose output
                print(f"Value difference for {key}: relative_diff={relative_diff:.2e}")
        else:
            value_matches += 1

    # Calculate statistics
    if total_relative_differences:
        results["statistics"] = {
            "mean_relative_diff": np.mean(total_relative_differences),
            "max_relative_diff": np.max(total_relative_differences),
            "min_relative_diff": np.min(total_relative_differences),
            "std_relative_diff": np.std(total_relative_differences),
            "median_relative_diff": np.median(total_relative_differences),
        }

    # Summary
    results["summary"] = {
        "total_common_params": len(common_keys),
        "shape_matches": shape_matches,
        "value_matches": value_matches,
        "shape_mismatches": len(results["shape_differences"]),
        "value_mismatches": len(results["value_differences"]),
        "models_identical": (
            len(results["value_differences"]) == 0
            and len(results["shape_differences"]) == 0
            and len(results["missing_keys"]["model1_missing"]) == 0
            and len(results["missing_keys"]["model2_missing"]) == 0
        ),
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print("SUMMARY")
        print(f"{'=' * 50}")
        print(f"Shape matches: {shape_matches}/{len(common_keys)}")
        print(f"Value matches: {value_matches}/{len(common_keys)}")
        print(f"Models are identical: {results['summary']['models_identical']}")

        if results["statistics"]:
            print("Statistics of relative differences:")
            print(f"Mean: {results['statistics']['mean_relative_diff']:.2e}")
            print(f"Max:  {results['statistics']['max_relative_diff']:.2e}")
            print(f"Min:  {results['statistics']['min_relative_diff']:.2e}")
            print(f"Std:  {results['statistics']['std_relative_diff']:.2e}")

    return results


def compare_flux_kontext_models(
    model_path: str,
    component: str = "transformer",
    weight_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cuda",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Compare Flux Kontext models loaded with and without pipeline.

    Args:
        model_path: Path to the pretrained model
        component: Component to compare ('transformer', 'vae', 'clip', 't5', 'tokenizers')
        weight_dtype: Weight data type for the model
        device_map: Device to load the model on
        verbose: Whether to print detailed comparison results

    Returns:
        Dictionary containing comparison results
    """
    logger.info(f"Comparing Flux Kontext {component} models: pipeline vs direct loading")

    # Import the loader functions
    from qflux.models.flux_kontext_loader import (
        load_flux_kontext_clip,
        load_flux_kontext_t5,
        load_flux_kontext_tokenizers,
        load_flux_kontext_transformer,
        load_flux_kontext_vae,
    )

    try:
        if component == "transformer":
            model_direct = load_flux_kontext_transformer(
                repo=model_path, weight_dtype=weight_dtype, device_map=device_map, use_pipeline=False
            )
            model_pipeline = load_flux_kontext_transformer(
                repo=model_path, weight_dtype=weight_dtype, device_map=device_map, use_pipeline=True
            )

        elif component == "vae":
            model_direct = load_flux_kontext_vae(
                model_path=model_path, weight_dtype=weight_dtype, device_map=device_map, use_pipeline=False
            )
            model_pipeline = load_flux_kontext_vae(
                model_path=model_path, weight_dtype=weight_dtype, device_map=device_map, use_pipeline=True
            )

        elif component == "clip":
            model_direct = load_flux_kontext_clip(
                model_path=model_path, weight_dtype=weight_dtype, device_map=device_map, use_pipeline=False
            )
            model_pipeline = load_flux_kontext_clip(
                model_path=model_path, weight_dtype=weight_dtype, device_map=device_map, use_pipeline=True
            )

        elif component == "t5":
            model_direct = load_flux_kontext_t5(
                model_path=model_path, weight_dtype=weight_dtype, device_map=device_map, use_pipeline=False
            )
            model_pipeline = load_flux_kontext_t5(
                model_path=model_path, weight_dtype=weight_dtype, device_map=device_map, use_pipeline=True
            )

        elif component == "tokenizers":
            # For tokenizers, we compare their vocab and special tokens
            tokenizers_direct = load_flux_kontext_tokenizers(model_path=model_path, use_pipeline=False)
            tokenizers_pipeline = load_flux_kontext_tokenizers(model_path=model_path, use_pipeline=True)

            # Compare tokenizers
            results = compare_tokenizers(
                tokenizers_direct, tokenizers_pipeline, "Direct Loading", "Pipeline Loading", verbose
            )
            return results

        else:
            raise ValueError(f"Unsupported component: {component}")

        # Compare the models
        results = compare_model_parameters(
            model_direct, model_pipeline, "Direct Loading", "Pipeline Loading", verbose=verbose
        )

        # Clean up
        del model_direct, model_pipeline
        torch.cuda.empty_cache()

        return results

    except Exception as e:
        logger.error(f"Error comparing {component} models: {e}")
        raise


def compare_tokenizers(
    tokenizers1: tuple,
    tokenizers2: tuple,
    name1: str = "Tokenizers 1",
    name2: str = "Tokenizers 2",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Compare two sets of tokenizers (CLIP and T5).

    Args:
        tokenizers1: Tuple of (clip_tokenizer, t5_tokenizer) from first method
        tokenizers2: Tuple of (clip_tokenizer, t5_tokenizer) from second method
        name1: Name for first tokenizer set
        name2: Name for second tokenizer set
        verbose: Whether to print detailed comparison results

    Returns:
        Dictionary containing comparison results
    """
    clip_tok1, t5_tok1 = tokenizers1
    clip_tok2, t5_tok2 = tokenizers2

    results: dict[str, Any] = {"clip_tokenizer": {}, "t5_tokenizer": {}, "summary": {}}

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"TOKENIZER COMPARISON: {name1} vs {name2}")
        print(f"{'=' * 80}")

    # Compare CLIP tokenizers
    clip_identical = True
    if hasattr(clip_tok1, "vocab_size") and hasattr(clip_tok2, "vocab_size"):
        vocab_size_match = clip_tok1.vocab_size == clip_tok2.vocab_size
        clip_identical = vocab_size_match

        results["clip_tokenizer"] = {
            "vocab_size_match": vocab_size_match,
            "vocab_size_1": clip_tok1.vocab_size,
            "vocab_size_2": clip_tok2.vocab_size,
        }

        if verbose:
            print(f"CLIP Tokenizer vocab sizes: {clip_tok1.vocab_size} vs {clip_tok2.vocab_size}")
            print(f"CLIP Tokenizer vocab size match: {vocab_size_match}")

    # Compare T5 tokenizers
    t5_identical = True
    if hasattr(t5_tok1, "vocab_size") and hasattr(t5_tok2, "vocab_size"):
        vocab_size_match = t5_tok1.vocab_size == t5_tok2.vocab_size
        t5_identical = vocab_size_match

        results["t5_tokenizer"] = {
            "vocab_size_match": vocab_size_match,
            "vocab_size_1": t5_tok1.vocab_size,
            "vocab_size_2": t5_tok2.vocab_size,
        }

        if verbose:
            print(f"T5 Tokenizer vocab sizes: {t5_tok1.vocab_size} vs {t5_tok2.vocab_size}")
            print(f"T5 Tokenizer vocab size match: {vocab_size_match}")

    results["summary"] = {
        "clip_identical": clip_identical,
        "t5_identical": t5_identical,
        "tokenizers_identical": clip_identical and t5_identical,
    }

    if verbose:
        print(f"Tokenizers are identical: {results['summary']['tokenizers_identical']}")

    return results


def run_full_flux_comparison(
    model_path: str,
    components: list[str] | None = None,
    weight_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cuda",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run full comparison of all Flux Kontext components.

    Args:
        model_path: Path to the pretrained model
        components: List of components to compare. If None, compares all.
        weight_dtype: Weight data type for the model
        device_map: Device to load the model on
        verbose: Whether to print detailed comparison results

    Returns:
        Dictionary containing comparison results for all components
    """
    if components is None:
        components = ["transformer", "vae", "clip", "t5", "tokenizers"]

    all_results = {}

    if verbose:
        print(f"\n{'=' * 100}")
        print("FULL FLUX KONTEXT MODEL COMPARISON")
        print(f"Model Path: {model_path}")
        print(f"Components: {components}")
        print(f"{'=' * 100}")

    for component in components:
        if verbose:
            print(f"\n{'-' * 60}")
            print(f"Comparing {component.upper()} component...")
            print(f"{'-' * 60}")

        try:
            results = compare_flux_kontext_models(
                model_path=model_path,
                component=component,
                weight_dtype=weight_dtype,
                device_map=device_map,
                verbose=verbose,
            )
            all_results[component] = results

        except Exception as e:
            logger.error(f"Failed to compare {component}: {e}")
            all_results[component] = {"error": str(e)}

    # Overall summary
    if verbose:
        print(f"\n{'=' * 100}")
        print("OVERALL SUMMARY")
        print(f"{'=' * 100}")

        for component, results in all_results.items():
            if "error" in results:
                print(f"{component.upper()}: ERROR - {results['error']}")
            elif "summary" in results:
                if "models_identical" in results["summary"]:
                    identical = results["summary"]["models_identical"]
                    print(f"{component.upper()}: {'IDENTICAL' if identical else 'DIFFERENT'}")
                elif "tokenizers_identical" in results["summary"]:
                    identical = results["summary"]["tokenizers_identical"]
                    print(f"{component.upper()}: {'IDENTICAL' if identical else 'DIFFERENT'}")

    return all_results


if __name__ == "__main__":
    # Example usage
    model_path = "black-forest-labs/FLUX.1-Kontext-dev"

    # Compare specific component
    results = compare_flux_kontext_models(model_path=model_path, component="transformer", verbose=True)

    # Or run full comparison
    full_results = run_full_flux_comparison(model_path, verbose=True)
