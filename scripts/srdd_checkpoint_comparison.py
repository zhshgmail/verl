#!/usr/bin/env python3
"""
SRDD Checkpoint Comparison Script

Compares quantization robustness across checkpoints from different training runs.
Loads FSDP checkpoints and runs SRDD quantization scan.

Usage:
    python scripts/srdd_checkpoint_comparison.py \
        --baseline_ckpt /tmp/mxfp4_exp_baseline/checkpoints/global_step_58 \
        --mxfp4_only_ckpt /tmp/mxfp4_exp_mxfp4only/checkpoints/global_step_58 \
        --mxfp4_aqn_ckpt /tmp/mxfp4_w4a16_experiment/checkpoints/global_step_58 \
        --output results_srdd_comparison.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from verl.utils.mxfp4_quant import mxfp4_quantize
    HAS_MXFP4 = True
except ImportError:
    HAS_MXFP4 = False
    print("Warning: mxfp4_quant not available")


def load_fsdp_checkpoint(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    """Load and merge FSDP sharded checkpoint."""
    ckpt_path = Path(ckpt_dir)
    actor_path = ckpt_path / "actor"

    # Find all rank files
    rank_files = sorted(actor_path.glob("model_world_size_*_rank_*.pt"))

    if not rank_files:
        raise ValueError(f"No model files found in {actor_path}")

    print(f"Found {len(rank_files)} FSDP shards")

    # Load all shards
    all_states = []
    for rank_file in rank_files:
        state = torch.load(rank_file, map_location='cpu')
        all_states.append(state)

    # Merge shards - FSDP shards are typically identical for full state dict
    # For sharded state dict, we need to concatenate
    merged_state = {}

    # Check if this is full state dict or sharded
    first_state = all_states[0]

    if isinstance(first_state, dict):
        # Full state dict replicated across ranks
        merged_state = first_state
        print(f"Loaded full state dict with {len(merged_state)} keys")
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(first_state)}")

    return merged_state


def compute_layer_sqnr(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """Compute Signal-to-Quantization-Noise Ratio in dB."""
    signal_power = (original ** 2).mean()
    noise = original - quantized
    noise_power = (noise ** 2).mean()

    if noise_power < 1e-10:
        return 100.0  # Very high SQNR (almost no noise)

    sqnr = 10 * torch.log10(signal_power / noise_power)
    return sqnr.item()


def compute_deadzone_ratio(original: torch.Tensor, quantized: torch.Tensor, threshold: float = 1e-6) -> float:
    """Compute ratio of values that became zero after quantization."""
    was_nonzero = original.abs() > threshold
    is_zero = quantized.abs() <= threshold
    deadzone = (was_nonzero & is_zero).float().mean()
    return deadzone.item()


def compute_relative_error(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """Compute mean relative error."""
    abs_error = (original - quantized).abs()
    rel_error = abs_error / (original.abs() + 1e-8)
    return rel_error.mean().item()


def scan_checkpoint(state_dict: Dict[str, torch.Tensor], quant_type: str = 'mxfp4') -> Dict[str, Any]:
    """Run SRDD quantization scan on checkpoint."""
    results = {
        'num_layers': 0,
        'layers': [],
        'summary': {}
    }

    sqnr_values = []
    deadzone_values = []
    rel_error_values = []

    for name, param in state_dict.items():
        # Only scan weight tensors
        if not name.endswith('.weight'):
            continue
        if param.dim() < 2:  # Skip 1D tensors (biases, norms)
            continue

        # Skip embedding and lm_head for now (they're special)
        if 'embed' in name.lower() or 'lm_head' in name.lower():
            continue

        original = param.float()

        # Apply MXFP4 quantization
        if HAS_MXFP4 and quant_type == 'mxfp4':
            quantized, _ = mxfp4_quantize(original)
        else:
            # Fallback: simple FP4 simulation
            scale = original.abs().max() / 7.0  # FP4 range
            quantized = (original / scale).round().clamp(-8, 7) * scale

        # Compute metrics
        sqnr = compute_layer_sqnr(original, quantized)
        deadzone = compute_deadzone_ratio(original, quantized)
        rel_error = compute_relative_error(original, quantized)

        layer_result = {
            'name': name,
            'shape': list(param.shape),
            'sqnr_db': sqnr,
            'deadzone_ratio': deadzone,
            'relative_error': rel_error
        }
        results['layers'].append(layer_result)
        results['num_layers'] += 1

        sqnr_values.append(sqnr)
        deadzone_values.append(deadzone)
        rel_error_values.append(rel_error)

    # Compute summary statistics
    if sqnr_values:
        results['summary'] = {
            'sqnr_db': {
                'mean': sum(sqnr_values) / len(sqnr_values),
                'min': min(sqnr_values),
                'max': max(sqnr_values)
            },
            'deadzone_ratio': {
                'mean': sum(deadzone_values) / len(deadzone_values),
                'min': min(deadzone_values),
                'max': max(deadzone_values)
            },
            'relative_error': {
                'mean': sum(rel_error_values) / len(rel_error_values),
                'min': min(rel_error_values),
                'max': max(rel_error_values)
            }
        }

    return results


def compare_checkpoints(
    baseline_ckpt: Optional[str],
    mxfp4_only_ckpt: Optional[str],
    mxfp4_aqn_ckpt: Optional[str],
    original_model: Optional[str] = None
) -> Dict[str, Any]:
    """Compare SRDD metrics across checkpoints."""

    comparison = {
        'checkpoints': {},
        'comparison': {}
    }

    checkpoints = [
        ('baseline', baseline_ckpt),
        ('mxfp4_only', mxfp4_only_ckpt),
        ('mxfp4_aqn', mxfp4_aqn_ckpt)
    ]

    for name, ckpt_path in checkpoints:
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"Skipping {name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n=== Scanning {name} checkpoint ===")
        print(f"Path: {ckpt_path}")

        try:
            state_dict = load_fsdp_checkpoint(ckpt_path)
            results = scan_checkpoint(state_dict)
            comparison['checkpoints'][name] = results

            print(f"Scanned {results['num_layers']} layers")
            if results['summary']:
                print(f"  SQNR: {results['summary']['sqnr_db']['mean']:.2f} dB")
                print(f"  Deadzone: {results['summary']['deadzone_ratio']['mean']*100:.2f}%")
                print(f"  Rel Error: {results['summary']['relative_error']['mean']*100:.2f}%")
        except Exception as e:
            print(f"Error scanning {name}: {e}")
            import traceback
            traceback.print_exc()

    # Compute comparison
    if len(comparison['checkpoints']) >= 2:
        names = list(comparison['checkpoints'].keys())
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                ckpt1 = comparison['checkpoints'][name1]
                ckpt2 = comparison['checkpoints'][name2]

                if 'summary' in ckpt1 and 'summary' in ckpt2:
                    diff_key = f"{name1}_vs_{name2}"
                    comparison['comparison'][diff_key] = {
                        'sqnr_diff_db': ckpt2['summary']['sqnr_db']['mean'] - ckpt1['summary']['sqnr_db']['mean'],
                        'deadzone_diff': ckpt2['summary']['deadzone_ratio']['mean'] - ckpt1['summary']['deadzone_ratio']['mean'],
                        'rel_error_diff': ckpt2['summary']['relative_error']['mean'] - ckpt1['summary']['relative_error']['mean']
                    }

    return comparison


def main():
    parser = argparse.ArgumentParser(description='SRDD Checkpoint Comparison')
    parser.add_argument('--baseline_ckpt', type=str, help='Baseline checkpoint path')
    parser.add_argument('--mxfp4_only_ckpt', type=str, help='MXFP4-only checkpoint path')
    parser.add_argument('--mxfp4_aqn_ckpt', type=str, help='MXFP4+AQN checkpoint path')
    parser.add_argument('--original_model', type=str, help='Original model path (for reference)')
    parser.add_argument('--output', type=str, default='srdd_comparison.json', help='Output file')

    args = parser.parse_args()

    print("=" * 60)
    print("SRDD Checkpoint Comparison")
    print("=" * 60)

    comparison = compare_checkpoints(
        baseline_ckpt=args.baseline_ckpt,
        mxfp4_only_ckpt=args.mxfp4_only_ckpt,
        mxfp4_aqn_ckpt=args.mxfp4_aqn_ckpt,
        original_model=args.original_model
    )

    # Print comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    for name, results in comparison['checkpoints'].items():
        if 'summary' in results:
            print(f"\n{name}:")
            print(f"  SQNR: {results['summary']['sqnr_db']['mean']:.2f} dB")
            print(f"  Deadzone: {results['summary']['deadzone_ratio']['mean']*100:.2f}%")
            print(f"  Rel Error: {results['summary']['relative_error']['mean']*100:.2f}%")

    if comparison['comparison']:
        print("\n--- Differences ---")
        for diff_key, diff_values in comparison['comparison'].items():
            print(f"\n{diff_key}:")
            print(f"  SQNR diff: {diff_values['sqnr_diff_db']:+.2f} dB")
            print(f"  Deadzone diff: {diff_values['deadzone_diff']*100:+.2f}%")
            print(f"  Rel Error diff: {diff_values['rel_error_diff']*100:+.2f}%")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
