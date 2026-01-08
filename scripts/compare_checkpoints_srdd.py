#!/usr/bin/env python3
"""
SRDD Checkpoint Comparison Script

Compares quantization metrics between multiple checkpoints to measure
the effectiveness of AQN training for quantization robustness.

Usage:
    python scripts/compare_checkpoints_srdd.py \
        --checkpoints /path/to/ckpt1 /path/to/ckpt2 \
        --labels "MXFP4-only" "MXFP4+AQN" \
        --output comparison_results.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

import torch

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import SRDD scanner
try:
    from scripts.srdd_quant_scanner import SRDDQuantScanner, QuantScanConfig
    SCANNER_AVAILABLE = True
except ImportError as e:
    SCANNER_AVAILABLE = False
    print(f"[WARN] SRDD scanner not available: {e}")


def load_checkpoint_model(checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint (supports HF and FSDP formats)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    checkpoint_path = Path(checkpoint_path)

    # Check for different checkpoint formats
    if (checkpoint_path / "config.json").exists():
        # HuggingFace format
        print(f"Loading HF checkpoint from {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    elif (checkpoint_path / "actor").exists():
        # veRL FSDP checkpoint format
        print(f"Loading veRL FSDP checkpoint from {checkpoint_path}")
        actor_path = checkpoint_path / "actor"

        # Find the base model path from checkpoint
        # This requires knowing the original model path
        raise NotImplementedError(
            "FSDP checkpoint loading requires base model path. "
            "Please convert to HF format first using veRL's conversion script."
        )
    else:
        raise ValueError(f"Unknown checkpoint format at {checkpoint_path}")

    model.eval()
    return model, tokenizer


def run_srdd_on_checkpoint(
    checkpoint_path: str,
    quant_type: str = 'mxfp4',
    device: str = 'cuda',
) -> Dict[str, Any]:
    """Run SRDD scan on a single checkpoint."""

    if not SCANNER_AVAILABLE:
        raise RuntimeError("SRDD scanner not available")

    config = QuantScanConfig(quant_type=quant_type)
    scanner = SRDDQuantScanner(
        model_path=checkpoint_path,
        config=config,
        device=device,
    )

    return scanner.run_full_scan()


def compare_checkpoints(
    checkpoint_paths: List[str],
    labels: List[str],
    quant_type: str = 'mxfp4',
    device: str = 'cuda',
) -> Dict[str, Any]:
    """Compare SRDD metrics between multiple checkpoints."""

    results = []

    for i, (path, label) in enumerate(zip(checkpoint_paths, labels)):
        print(f"\n{'='*60}")
        print(f"Scanning checkpoint {i+1}/{len(checkpoint_paths)}: {label}")
        print(f"Path: {path}")
        print(f"{'='*60}")

        try:
            scan_result = run_srdd_on_checkpoint(path, quant_type, device)
            scan_result['label'] = label
            scan_result['checkpoint_path'] = path
            results.append(scan_result)
        except Exception as e:
            print(f"[ERROR] Failed to scan {label}: {e}")
            results.append({
                'label': label,
                'checkpoint_path': path,
                'error': str(e),
            })

    # Generate comparison report
    comparison = generate_comparison_report(results)

    return {
        'checkpoints': results,
        'comparison': comparison,
        'timestamp': datetime.now().isoformat(),
    }


def generate_comparison_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comparison report between checkpoints."""

    # Filter out failed scans
    valid_results = [r for r in results if 'error' not in r]

    if len(valid_results) < 2:
        return {'error': 'Need at least 2 valid scans for comparison'}

    # Extract metrics for comparison
    comparison = {
        'summary': [],
        'improvements': {},
        'per_layer_diff': [],
    }

    # Use first checkpoint as baseline
    baseline = valid_results[0]
    baseline_stats = baseline['report']['statistics']
    baseline_label = baseline['label']

    for result in valid_results[1:]:
        label = result['label']
        stats = result['report']['statistics']

        # Calculate improvements
        sqnr_improvement = stats['sqnr_db']['mean'] - baseline_stats['sqnr_db']['mean']
        deadzone_improvement = baseline_stats['deadzone_ratio']['mean'] - stats['deadzone_ratio']['mean']
        rel_error_improvement = baseline_stats['relative_error']['mean'] - stats['relative_error']['mean']

        comparison['summary'].append({
            'label': label,
            'vs_baseline': baseline_label,
            'sqnr_db': {
                'baseline': baseline_stats['sqnr_db']['mean'],
                'current': stats['sqnr_db']['mean'],
                'improvement': sqnr_improvement,
                'improved': sqnr_improvement > 0,
            },
            'deadzone_ratio': {
                'baseline': baseline_stats['deadzone_ratio']['mean'],
                'current': stats['deadzone_ratio']['mean'],
                'improvement': deadzone_improvement,
                'improved': deadzone_improvement > 0,
            },
            'relative_error': {
                'baseline': baseline_stats['relative_error']['mean'],
                'current': stats['relative_error']['mean'],
                'improvement': rel_error_improvement,
                'improved': rel_error_improvement > 0,
            },
        })

        # Per-layer comparison
        baseline_layers = {l['layer_id']: l for l in baseline['report']['per_layer']}
        current_layers = {l['layer_id']: l for l in result['report']['per_layer']}

        layer_diffs = []
        for layer_id in baseline_layers:
            if layer_id in current_layers:
                bl = baseline_layers[layer_id]
                cl = current_layers[layer_id]
                layer_diffs.append({
                    'layer_id': layer_id,
                    'sqnr_diff': cl['sqnr_db'] - bl['sqnr_db'],
                    'deadzone_diff': bl['deadzone_ratio'] - cl['deadzone_ratio'],
                    'rel_error_diff': bl['relative_error'] - cl['relative_error'],
                })

        comparison['per_layer_diff'].append({
            'label': label,
            'vs_baseline': baseline_label,
            'layers': layer_diffs,
        })

    return comparison


def print_comparison_report(comparison: Dict[str, Any]):
    """Print formatted comparison report."""

    print(f"\n{'='*60}")
    print("COMPARISON REPORT")
    print(f"{'='*60}")

    if 'error' in comparison:
        print(f"Error: {comparison['error']}")
        return

    for summary in comparison['summary']:
        print(f"\n{summary['label']} vs {summary['vs_baseline']}:")
        print("-" * 40)

        sqnr = summary['sqnr_db']
        status = "IMPROVED" if sqnr['improved'] else "DEGRADED"
        print(f"  SQNR (dB):        {sqnr['baseline']:.2f} → {sqnr['current']:.2f} ({sqnr['improvement']:+.2f}) [{status}]")

        dz = summary['deadzone_ratio']
        status = "IMPROVED" if dz['improved'] else "DEGRADED"
        print(f"  Deadzone (%):     {dz['baseline']*100:.2f} → {dz['current']*100:.2f} ({dz['improvement']*100:+.2f}) [{status}]")

        re = summary['relative_error']
        status = "IMPROVED" if re['improved'] else "DEGRADED"
        print(f"  Relative Error (%): {re['baseline']*100:.2f} → {re['current']*100:.2f} ({re['improvement']*100:+.2f}) [{status}]")

    # Show top improved/degraded layers
    for layer_diff in comparison.get('per_layer_diff', []):
        layers = layer_diff['layers']
        if not layers:
            continue

        print(f"\n{layer_diff['label']} - Top 5 Most Improved Layers (SQNR):")
        sorted_layers = sorted(layers, key=lambda x: -x['sqnr_diff'])
        for i, l in enumerate(sorted_layers[:5]):
            print(f"  {i+1}. Layer {l['layer_id']}: {l['sqnr_diff']:+.2f} dB")

        print(f"\n{layer_diff['label']} - Top 5 Most Degraded Layers (SQNR):")
        for i, l in enumerate(sorted_layers[-5:]):
            print(f"  {i+1}. Layer {l['layer_id']}: {l['sqnr_diff']:+.2f} dB")


def main():
    parser = argparse.ArgumentParser(description="Compare SRDD metrics between checkpoints")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels for each checkpoint")
    parser.add_argument("--quant_type", type=str, default="mxfp4", help="Quantization type")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    if len(args.checkpoints) != len(args.labels):
        parser.error("Number of checkpoints must match number of labels")

    if len(args.checkpoints) < 2:
        parser.error("Need at least 2 checkpoints for comparison")

    results = compare_checkpoints(
        checkpoint_paths=args.checkpoints,
        labels=args.labels,
        quant_type=args.quant_type,
        device=args.device,
    )

    print_comparison_report(results['comparison'])

    if args.output:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        results = convert_numpy(results)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
