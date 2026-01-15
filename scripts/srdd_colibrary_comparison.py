#!/usr/bin/env python3
"""
SRDD Co-Library Comparison: llm-compressor vs verl implementations

This script compares quantization error between:
1. llm-compressor's official MXFP4/NVFP4 implementations
2. verl's mxfp4_quant.py and nvfp4_quant.py implementations

Purpose: Validate our SRDD quantization simulation is accurate.

Usage:
    python scripts/srdd_colibrary_comparison.py \
        --model_path /data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/... \
        --output_dir /tmp/srdd_colibrary_results
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_quantization_error(original: torch.Tensor, quantized: torch.Tensor) -> Dict[str, float]:
    """Compute various error metrics between original and quantized tensors."""
    diff = (quantized - original).float()
    orig_float = original.float()

    # Absolute errors
    abs_error = diff.abs()
    mse = (diff ** 2).mean().item()
    mae = abs_error.mean().item()
    max_error = abs_error.max().item()

    # Relative error (avoid division by zero)
    rel_error = (abs_error / (orig_float.abs() + 1e-10)).mean().item()

    # Signal-to-Quantization-Noise Ratio (SQNR) in dB
    signal_power = (orig_float ** 2).mean().item()
    noise_power = mse
    sqnr_db = 10 * torch.log10(torch.tensor(signal_power / (noise_power + 1e-10))).item()

    # Deadzone (values that become zero)
    orig_nonzero = (original.abs() > 1e-10).float().mean().item()
    quant_nonzero = (quantized.abs() > 1e-10).float().mean().item()
    deadzone_ratio = (orig_nonzero - quant_nonzero) / (orig_nonzero + 1e-10)

    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'relative_error': rel_error * 100,  # percentage
        'sqnr_db': sqnr_db,
        'deadzone_ratio': deadzone_ratio * 100,  # percentage
    }


def test_verl_mxfp4(model, device='cpu'):
    """Test verl's MXFP4 implementation."""
    try:
        from verl.utils.mxfp4_quant import mxfp4_quantize
    except ImportError:
        return None, "verl.utils.mxfp4_quant not found"

    results = {}
    layer_errors = []

    for name, param in model.named_parameters():
        if 'weight' not in name or param.dim() < 2:
            continue
        if 'embed' in name or 'lm_head' in name:
            continue

        weight = param.data.to(device).float()
        weight_quant = mxfp4_quantize(weight)

        errors = measure_quantization_error(weight, weight_quant)
        layer_errors.append({
            'layer': name,
            **errors
        })

    # Aggregate statistics
    if layer_errors:
        results['mean_relative_error'] = sum(e['relative_error'] for e in layer_errors) / len(layer_errors)
        results['mean_sqnr_db'] = sum(e['sqnr_db'] for e in layer_errors) / len(layer_errors)
        results['mean_deadzone'] = sum(e['deadzone_ratio'] for e in layer_errors) / len(layer_errors)
        results['num_layers'] = len(layer_errors)
        results['layer_details'] = layer_errors

    return results, None


def test_verl_nvfp4(model, device='cpu'):
    """Test verl's NVFP4 implementation."""
    try:
        from verl.utils.nvfp4_quant import nvfp4_quantize
    except ImportError:
        return None, "verl.utils.nvfp4_quant not found"

    results = {}
    layer_errors = []

    for name, param in model.named_parameters():
        if 'weight' not in name or param.dim() < 2:
            continue
        if 'embed' in name or 'lm_head' in name:
            continue

        weight = param.data.to(device).float()
        weight_quant = nvfp4_quantize(weight)

        errors = measure_quantization_error(weight, weight_quant)
        layer_errors.append({
            'layer': name,
            **errors
        })

    # Aggregate statistics
    if layer_errors:
        results['mean_relative_error'] = sum(e['relative_error'] for e in layer_errors) / len(layer_errors)
        results['mean_sqnr_db'] = sum(e['sqnr_db'] for e in layer_errors) / len(layer_errors)
        results['mean_deadzone'] = sum(e['deadzone_ratio'] for e in layer_errors) / len(layer_errors)
        results['num_layers'] = len(layer_errors)
        results['layer_details'] = layer_errors

    return results, None


def test_llmcompressor_mxfp4(model, tokenizer, device='cuda'):
    """Test llm-compressor's MXFP4 implementation."""
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        import copy
    except ImportError:
        return None, "llmcompressor not found"

    # Store original weights
    original_weights = {}
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            if 'embed' not in name and 'lm_head' not in name:
                original_weights[name] = param.data.clone().cpu()

    # Apply MXFP4 quantization
    model_copy = copy.deepcopy(model)
    recipe = QuantizationModifier(targets="Linear", scheme="MXFP4", ignore=["lm_head"])

    try:
        oneshot(model=model_copy, recipe=recipe)
    except Exception as e:
        return None, f"MXFP4 quantization failed: {e}"

    # Measure errors
    results = {}
    layer_errors = []

    for name, param in model_copy.named_parameters():
        if name not in original_weights:
            continue

        original = original_weights[name].float()
        quantized = param.data.cpu().float()

        # Handle potential shape mismatch from compressed format
        if original.shape != quantized.shape:
            continue

        errors = measure_quantization_error(original, quantized)
        layer_errors.append({
            'layer': name,
            **errors
        })

    # Aggregate statistics
    if layer_errors:
        results['mean_relative_error'] = sum(e['relative_error'] for e in layer_errors) / len(layer_errors)
        results['mean_sqnr_db'] = sum(e['sqnr_db'] for e in layer_errors) / len(layer_errors)
        results['mean_deadzone'] = sum(e['deadzone_ratio'] for e in layer_errors) / len(layer_errors)
        results['num_layers'] = len(layer_errors)
        results['layer_details'] = layer_errors

    del model_copy
    return results, None


def test_llmcompressor_nvfp4(model, tokenizer, device='cuda'):
    """Test llm-compressor's NVFP4 implementation."""
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        import copy
    except ImportError:
        return None, "llmcompressor not found"

    # Store original weights
    original_weights = {}
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            if 'embed' not in name and 'lm_head' not in name:
                original_weights[name] = param.data.clone().cpu()

    # Apply NVFP4A16 quantization
    model_copy = copy.deepcopy(model)
    recipe = QuantizationModifier(targets="Linear", scheme="NVFP4A16", ignore=["lm_head"])

    try:
        oneshot(model=model_copy, recipe=recipe)
    except Exception as e:
        return None, f"NVFP4 quantization failed: {e}"

    # Measure errors
    results = {}
    layer_errors = []

    for name, param in model_copy.named_parameters():
        if name not in original_weights:
            continue

        original = original_weights[name].float()
        quantized = param.data.cpu().float()

        # Handle potential shape mismatch from compressed format
        if original.shape != quantized.shape:
            continue

        errors = measure_quantization_error(original, quantized)
        layer_errors.append({
            'layer': name,
            **errors
        })

    # Aggregate statistics
    if layer_errors:
        results['mean_relative_error'] = sum(e['relative_error'] for e in layer_errors) / len(layer_errors)
        results['mean_sqnr_db'] = sum(e['sqnr_db'] for e in layer_errors) / len(layer_errors)
        results['mean_deadzone'] = sum(e['deadzone_ratio'] for e in layer_errors) / len(layer_errors)
        results['num_layers'] = len(layer_errors)
        results['layer_details'] = layer_errors

    del model_copy
    return results, None


def main():
    parser = argparse.ArgumentParser(description='SRDD Co-Library Comparison')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model')
    parser.add_argument('--output_dir', type=str, default='/tmp/srdd_colibrary_results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--skip_llmcompressor', action='store_true', help='Skip llm-compressor tests')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("SRDD Co-Library Comparison")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map=args.device if args.device == 'cuda' else None
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': args.model_path,
        'device': args.device,
    }

    # Test 1: verl MXFP4
    print("\n[1/4] Testing verl MXFP4...")
    verl_mxfp4, err = test_verl_mxfp4(model, device='cpu')
    if err:
        print(f"  ERROR: {err}")
    else:
        print(f"  Mean Relative Error: {verl_mxfp4['mean_relative_error']:.2f}%")
        print(f"  Mean SQNR: {verl_mxfp4['mean_sqnr_db']:.2f} dB")
        print(f"  Layers: {verl_mxfp4['num_layers']}")
    results['verl_mxfp4'] = verl_mxfp4 if verl_mxfp4 else {'error': err}

    # Test 2: verl NVFP4
    print("\n[2/4] Testing verl NVFP4...")
    verl_nvfp4, err = test_verl_nvfp4(model, device='cpu')
    if err:
        print(f"  ERROR: {err}")
    else:
        print(f"  Mean Relative Error: {verl_nvfp4['mean_relative_error']:.2f}%")
        print(f"  Mean SQNR: {verl_nvfp4['mean_sqnr_db']:.2f} dB")
        print(f"  Layers: {verl_nvfp4['num_layers']}")
    results['verl_nvfp4'] = verl_nvfp4 if verl_nvfp4 else {'error': err}

    if not args.skip_llmcompressor:
        # Test 3: llm-compressor MXFP4
        print("\n[3/4] Testing llm-compressor MXFP4...")
        llmc_mxfp4, err = test_llmcompressor_mxfp4(model, tokenizer, device=args.device)
        if err:
            print(f"  ERROR: {err}")
        else:
            print(f"  Mean Relative Error: {llmc_mxfp4['mean_relative_error']:.2f}%")
            print(f"  Mean SQNR: {llmc_mxfp4['mean_sqnr_db']:.2f} dB")
            print(f"  Layers: {llmc_mxfp4['num_layers']}")
        results['llmcompressor_mxfp4'] = llmc_mxfp4 if llmc_mxfp4 else {'error': err}

        # Test 4: llm-compressor NVFP4
        print("\n[4/4] Testing llm-compressor NVFP4...")
        llmc_nvfp4, err = test_llmcompressor_nvfp4(model, tokenizer, device=args.device)
        if err:
            print(f"  ERROR: {err}")
        else:
            print(f"  Mean Relative Error: {llmc_nvfp4['mean_relative_error']:.2f}%")
            print(f"  Mean SQNR: {llmc_nvfp4['mean_sqnr_db']:.2f} dB")
            print(f"  Layers: {llmc_nvfp4['num_layers']}")
        results['llmcompressor_nvfp4'] = llmc_nvfp4 if llmc_nvfp4 else {'error': err}

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Implementation':<25} {'Rel Error %':<15} {'SQNR (dB)':<15} {'Status'}")
    print("-" * 70)

    for name, data in [
        ('verl MXFP4', results.get('verl_mxfp4')),
        ('verl NVFP4', results.get('verl_nvfp4')),
        ('llm-compressor MXFP4', results.get('llmcompressor_mxfp4')),
        ('llm-compressor NVFP4', results.get('llmcompressor_nvfp4')),
    ]:
        if data is None:
            continue
        if 'error' in data:
            print(f"{name:<25} {'N/A':<15} {'N/A':<15} ERROR: {data['error'][:30]}")
        else:
            print(f"{name:<25} {data['mean_relative_error']:<15.2f} {data['mean_sqnr_db']:<15.2f} OK")

    # Save results
    output_file = os.path.join(args.output_dir, 'srdd_colibrary_comparison.json')

    # Remove layer details for summary file (too large)
    summary = {k: v for k, v in results.items()}
    for key in ['verl_mxfp4', 'verl_nvfp4', 'llmcompressor_mxfp4', 'llmcompressor_nvfp4']:
        if key in summary and isinstance(summary[key], dict) and 'layer_details' in summary[key]:
            del summary[key]['layer_details']

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Compare verl vs llm-compressor
    if results.get('verl_mxfp4') and results.get('llmcompressor_mxfp4'):
        if 'error' not in results['verl_mxfp4'] and 'error' not in results['llmcompressor_mxfp4']:
            verl_err = results['verl_mxfp4']['mean_relative_error']
            llmc_err = results['llmcompressor_mxfp4']['mean_relative_error']
            diff = abs(verl_err - llmc_err)
            if diff < 1.0:
                print(f"MXFP4: verl and llm-compressor MATCH (diff={diff:.2f}%)")
            else:
                print(f"MXFP4: MISMATCH - verl={verl_err:.2f}%, llmc={llmc_err:.2f}%, diff={diff:.2f}%")

    if results.get('verl_nvfp4') and results.get('llmcompressor_nvfp4'):
        if 'error' not in results['verl_nvfp4'] and 'error' not in results['llmcompressor_nvfp4']:
            verl_err = results['verl_nvfp4']['mean_relative_error']
            llmc_err = results['llmcompressor_nvfp4']['mean_relative_error']
            diff = abs(verl_err - llmc_err)
            if diff < 1.0:
                print(f"NVFP4: verl and llm-compressor MATCH (diff={diff:.2f}%)")
            else:
                print(f"NVFP4: MISMATCH - verl={verl_err:.2f}%, llmc={llmc_err:.2f}%, diff={diff:.2f}%")


if __name__ == '__main__':
    main()
