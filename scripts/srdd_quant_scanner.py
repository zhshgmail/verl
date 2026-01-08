#!/usr/bin/env python3
"""
SRDD Quantization Scanner - Improved metrics for detecting quantization issues

This scanner uses quantization-specific metrics (SQNR, deadzone ratio, range loss)
instead of generic gain/kurtosis scans that were designed for hardware faults.

Metrics:
1. SQNR (Signal-to-Quantization-Noise Ratio) - higher is better, measures fidelity
2. Deadzone Ratio - % of values falling to zero (below min representable)
3. Saturation Ratio - % of values clipped to max representable
4. Relative Error - mean(|error|/|original|), measures proportional error

Usage:
    python scripts/srdd_quant_scanner.py \
        --model_path /data/z00637938/hub/Qwen2.5-1.5B-Instruct \
        --quant_type mxfp4 \
        --output results_quant_scan.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

import torch
import torch.nn.functional as F

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import standalone MXFP4 implementation
try:
    from verl.utils.mxfp4_quant import (
        mxfp4_quantize,
        MXFP4Config,
        compute_mxfp4_error,
    )
    MXFP4_AVAILABLE = True
    print("[OK] verl.utils.mxfp4_quant loaded")
except ImportError as e:
    MXFP4_AVAILABLE = False
    print(f"[WARN] mxfp4_quant not available: {e}")


@dataclass
class LayerQuantMetrics:
    """Quantization metrics for a single layer"""
    layer_id: int
    sqnr_db: float  # Signal-to-Quantization-Noise Ratio in dB
    deadzone_ratio: float  # % of values that become zero
    saturation_ratio: float  # % of values that get clipped
    relative_error: float  # mean(|error|/|original|)
    mean_abs_error: float  # mean(|error|)
    max_abs_error: float  # max(|error|)
    output_norm: float  # ||output|| for reference
    is_problematic: bool = False
    issues: List[str] = field(default_factory=list)


@dataclass
class QuantScanConfig:
    """Configuration for quantization scan"""
    # MXFP4 parameters
    quant_type: str = 'mxfp4'

    # Thresholds for identifying problematic layers
    sqnr_threshold_db: float = 20.0  # Below this = problematic
    deadzone_threshold: float = 0.05  # Above 5% = problematic
    saturation_threshold: float = 0.01  # Above 1% = problematic
    relative_error_threshold: float = 0.1  # Above 10% = problematic


class QuantizationAnalyzer:
    """Analyzes quantization error for a single activation tensor"""

    def __init__(self, quant_type: str = 'mxfp4'):
        self.quant_type = quant_type
        self.config = self._get_mxfp4_config()

    def _get_mxfp4_config(self) -> Optional[MXFP4Config]:
        if not MXFP4_AVAILABLE:
            return None
        return MXFP4Config(
            stochastic_rounding='sr' in self.quant_type,
            truncation_free='tf' in self.quant_type,
            block_h=32 if '2d' in self.quant_type else 1,
            block_w=32,
        )

    def analyze(self, x: torch.Tensor) -> Dict[str, float]:
        """Analyze quantization error for tensor x"""
        if self.config is None:
            return self._fallback_analyze(x)

        # Apply quantization
        x_quant = mxfp4_quantize(x, config=self.config)

        # Compute error
        error = x_quant - x
        abs_error = error.abs()

        # 1. SQNR (Signal-to-Quantization-Noise Ratio)
        signal_power = (x ** 2).mean().item()
        noise_power = (error ** 2).mean().item()
        if noise_power > 0:
            sqnr = signal_power / noise_power
            sqnr_db = 10 * np.log10(sqnr)
        else:
            sqnr_db = 100.0  # Perfect quantization (unlikely)

        # 2. Deadzone ratio (values that became zero but weren't)
        was_nonzero = x.abs() > 1e-10
        is_zero = x_quant.abs() < 1e-10
        deadzone_mask = was_nonzero & is_zero
        deadzone_ratio = deadzone_mask.float().mean().item()

        # 3. Saturation ratio (values that got clipped to max)
        # MXFP4 max value is approximately 6.0 (with proper scaling)
        # We detect saturation by checking if quantized value equals max representable
        x_quant_abs = x_quant.abs()
        max_quant = x_quant_abs.max().item()
        is_saturated = x_quant_abs >= max_quant * 0.99  # Within 1% of max
        saturation_ratio = is_saturated.float().mean().item()

        # 4. Relative error (per-element normalized error)
        relative_error = (abs_error / (x.abs() + 1e-10)).mean().item()

        # 5. Basic error stats
        mean_abs_error = abs_error.mean().item()
        max_abs_error = abs_error.max().item()
        output_norm = x.norm().item()

        return {
            'sqnr_db': sqnr_db,
            'deadzone_ratio': deadzone_ratio,
            'saturation_ratio': saturation_ratio,
            'relative_error': relative_error,
            'mean_abs_error': mean_abs_error,
            'max_abs_error': max_abs_error,
            'output_norm': output_norm,
        }

    def _fallback_analyze(self, x: torch.Tensor) -> Dict[str, float]:
        """Fallback analysis when MXFP4 not available"""
        # Use simple quantization simulation
        shape = x.shape
        x_flat = x.view(-1, 32) if x.numel() >= 32 else x.view(1, -1)

        max_abs = x_flat.abs().max(dim=-1, keepdim=True)[0]
        shared_exp = torch.floor(torch.log2(max_abs + 1e-10))

        x_scaled = x_flat / (2 ** shared_exp)
        x_sign = torch.sign(x_scaled)
        x_abs = x_scaled.abs()

        x_quant = torch.round(x_abs * 2) / 2
        x_quant = x_quant.clamp(0, 6)

        x_dequant = x_sign * x_quant * (2 ** shared_exp)
        x_dequant = x_dequant.view(shape)

        error = x_dequant - x
        abs_error = error.abs()

        signal_power = (x ** 2).mean().item()
        noise_power = (error ** 2).mean().item()
        sqnr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100.0

        was_nonzero = x.abs() > 1e-10
        is_zero = x_dequant.abs() < 1e-10
        deadzone_ratio = (was_nonzero & is_zero).float().mean().item()

        x_quant_abs = x_dequant.abs()
        max_quant = x_quant_abs.max().item()
        saturation_ratio = (x_quant_abs >= max_quant * 0.99).float().mean().item()

        relative_error = (abs_error / (x.abs() + 1e-10)).mean().item()

        return {
            'sqnr_db': sqnr_db,
            'deadzone_ratio': deadzone_ratio,
            'saturation_ratio': saturation_ratio,
            'relative_error': relative_error,
            'mean_abs_error': abs_error.mean().item(),
            'max_abs_error': abs_error.max().item(),
            'output_norm': x.norm().item(),
        }


class SRDDQuantScanner:
    """SRDD scanner with quantization-specific metrics"""

    def __init__(
        self,
        model_path: str,
        config: Optional[QuantScanConfig] = None,
        device: str = 'cuda',
    ):
        self.model_path = model_path
        self.config = config or QuantScanConfig()
        self.device = device

        self.tokenizer = None
        self.model = None
        self.num_layers = None
        self.analyzer = QuantizationAnalyzer(self.config.quant_type)

        # Test prompts covering different domains
        self.test_prompts = [
            "The capital of France is",
            "Machine learning is a subset of",
            "The answer to 2 + 2 is",
            "Python is a programming language that",
            "Water freezes at a temperature of",
            "The sun rises in the",
            "Neural networks are inspired by",
            "In mathematics, pi is approximately",
        ]

    def load_model(self):
        """Load model and tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.num_layers = len(self.model.model.layers)
        print(f"Model loaded: {self.num_layers} layers")

    def scan_layer(self, layer_id: int) -> LayerQuantMetrics:
        """Scan a single layer for quantization issues"""
        layer = self.model.model.layers[layer_id]
        all_metrics = []

        for prompt in self.test_prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # Capture layer output
            captured_output = []

            def capture_hook(module, inp, out):
                if isinstance(out, tuple):
                    captured_output.append(out[0].clone())
                else:
                    captured_output.append(out.clone())
                return out

            handle = layer.register_forward_hook(capture_hook)

            with torch.no_grad():
                self.model(**inputs)

            handle.remove()

            if captured_output:
                metrics = self.analyzer.analyze(captured_output[0])
                all_metrics.append(metrics)

        # Aggregate metrics across prompts
        if all_metrics:
            avg_metrics = {
                k: np.mean([m[k] for m in all_metrics])
                for k in all_metrics[0].keys()
            }
        else:
            avg_metrics = {
                'sqnr_db': 0.0, 'deadzone_ratio': 0.0, 'saturation_ratio': 0.0,
                'relative_error': 0.0, 'mean_abs_error': 0.0, 'max_abs_error': 0.0,
                'output_norm': 0.0,
            }

        # Determine if layer is problematic
        issues = []
        if avg_metrics['sqnr_db'] < self.config.sqnr_threshold_db:
            issues.append(f"low_sqnr={avg_metrics['sqnr_db']:.1f}dB")
        if avg_metrics['deadzone_ratio'] > self.config.deadzone_threshold:
            issues.append(f"high_deadzone={avg_metrics['deadzone_ratio']*100:.1f}%")
        if avg_metrics['saturation_ratio'] > self.config.saturation_threshold:
            issues.append(f"high_saturation={avg_metrics['saturation_ratio']*100:.1f}%")
        if avg_metrics['relative_error'] > self.config.relative_error_threshold:
            issues.append(f"high_rel_error={avg_metrics['relative_error']*100:.1f}%")

        return LayerQuantMetrics(
            layer_id=layer_id,
            sqnr_db=avg_metrics['sqnr_db'],
            deadzone_ratio=avg_metrics['deadzone_ratio'],
            saturation_ratio=avg_metrics['saturation_ratio'],
            relative_error=avg_metrics['relative_error'],
            mean_abs_error=avg_metrics['mean_abs_error'],
            max_abs_error=avg_metrics['max_abs_error'],
            output_norm=avg_metrics['output_norm'],
            is_problematic=len(issues) > 0,
            issues=issues,
        )

    def scan_all_layers(self) -> List[LayerQuantMetrics]:
        """Scan all layers for quantization issues"""
        print(f"\n{'='*60}")
        print("SRDD Quantization Scan")
        print(f"{'='*60}")
        print(f"Quant type: {self.config.quant_type}")
        print(f"SQNR threshold: {self.config.sqnr_threshold_db} dB")
        print(f"Deadzone threshold: {self.config.deadzone_threshold*100}%")
        print(f"Saturation threshold: {self.config.saturation_threshold*100}%")
        print(f"Relative error threshold: {self.config.relative_error_threshold*100}%")

        results = []

        for layer_id in range(self.num_layers):
            print(f"\rScanning layer {layer_id+1}/{self.num_layers}...", end="", flush=True)
            metrics = self.scan_layer(layer_id)
            results.append(metrics)

        print("\n")
        return results

    def analyze_results(self, results: List[LayerQuantMetrics]) -> Dict[str, Any]:
        """Analyze scan results and produce report"""
        problematic_layers = [r for r in results if r.is_problematic]
        healthy_layers = [r for r in results if not r.is_problematic]

        # Statistics
        sqnr_values = [r.sqnr_db for r in results]
        deadzone_values = [r.deadzone_ratio for r in results]
        saturation_values = [r.saturation_ratio for r in results]
        rel_error_values = [r.relative_error for r in results]

        report = {
            'summary': {
                'total_layers': len(results),
                'problematic_layers': len(problematic_layers),
                'healthy_layers': len(healthy_layers),
                'problematic_layer_ids': [r.layer_id for r in problematic_layers],
            },
            'statistics': {
                'sqnr_db': {
                    'min': min(sqnr_values),
                    'max': max(sqnr_values),
                    'mean': np.mean(sqnr_values),
                    'std': np.std(sqnr_values),
                },
                'deadzone_ratio': {
                    'min': min(deadzone_values),
                    'max': max(deadzone_values),
                    'mean': np.mean(deadzone_values),
                    'std': np.std(deadzone_values),
                },
                'saturation_ratio': {
                    'min': min(saturation_values),
                    'max': max(saturation_values),
                    'mean': np.mean(saturation_values),
                    'std': np.std(saturation_values),
                },
                'relative_error': {
                    'min': min(rel_error_values),
                    'max': max(rel_error_values),
                    'mean': np.mean(rel_error_values),
                    'std': np.std(rel_error_values),
                },
            },
            'per_layer': [
                {
                    'layer_id': r.layer_id,
                    'sqnr_db': r.sqnr_db,
                    'deadzone_ratio': r.deadzone_ratio,
                    'saturation_ratio': r.saturation_ratio,
                    'relative_error': r.relative_error,
                    'mean_abs_error': r.mean_abs_error,
                    'max_abs_error': r.max_abs_error,
                    'output_norm': r.output_norm,
                    'is_problematic': r.is_problematic,
                    'issues': r.issues,
                }
                for r in results
            ],
        }

        return report

    def print_report(self, report: Dict[str, Any]):
        """Print formatted report"""
        print(f"\n{'='*60}")
        print("SCAN RESULTS")
        print(f"{'='*60}")

        summary = report['summary']
        print(f"\nSummary:")
        print(f"  Total layers: {summary['total_layers']}")
        print(f"  Problematic: {summary['problematic_layers']}")
        print(f"  Healthy: {summary['healthy_layers']}")

        if summary['problematic_layer_ids']:
            print(f"  Problematic layer IDs: {summary['problematic_layer_ids']}")

        stats = report['statistics']
        print(f"\nStatistics:")
        print(f"  SQNR (dB):         min={stats['sqnr_db']['min']:.1f}, max={stats['sqnr_db']['max']:.1f}, mean={stats['sqnr_db']['mean']:.1f}±{stats['sqnr_db']['std']:.1f}")
        print(f"  Deadzone (%):      min={stats['deadzone_ratio']['min']*100:.2f}, max={stats['deadzone_ratio']['max']*100:.2f}, mean={stats['deadzone_ratio']['mean']*100:.2f}")
        print(f"  Saturation (%):    min={stats['saturation_ratio']['min']*100:.2f}, max={stats['saturation_ratio']['max']*100:.2f}, mean={stats['saturation_ratio']['mean']*100:.2f}")
        print(f"  Relative Error (%): min={stats['relative_error']['min']*100:.2f}, max={stats['relative_error']['max']*100:.2f}, mean={stats['relative_error']['mean']*100:.2f}")

        # Show problematic layers detail
        problematic = [l for l in report['per_layer'] if l['is_problematic']]
        if problematic:
            print(f"\nProblematic Layers Detail:")
            for layer in problematic:
                print(f"  Layer {layer['layer_id']}:")
                print(f"    SQNR: {layer['sqnr_db']:.1f} dB")
                print(f"    Deadzone: {layer['deadzone_ratio']*100:.2f}%")
                print(f"    Saturation: {layer['saturation_ratio']*100:.2f}%")
                print(f"    Relative Error: {layer['relative_error']*100:.2f}%")
                print(f"    Issues: {', '.join(layer['issues'])}")

        # Top 5 worst layers by SQNR
        print(f"\nTop 5 Layers with Lowest SQNR:")
        sorted_by_sqnr = sorted(report['per_layer'], key=lambda x: x['sqnr_db'])
        for i, layer in enumerate(sorted_by_sqnr[:5]):
            status = "⚠️" if layer['is_problematic'] else "✓"
            print(f"  {i+1}. Layer {layer['layer_id']}: {layer['sqnr_db']:.1f} dB {status}")

        # Top 5 worst layers by relative error
        print(f"\nTop 5 Layers with Highest Relative Error:")
        sorted_by_error = sorted(report['per_layer'], key=lambda x: -x['relative_error'])
        for i, layer in enumerate(sorted_by_error[:5]):
            status = "⚠️" if layer['is_problematic'] else "✓"
            print(f"  {i+1}. Layer {layer['layer_id']}: {layer['relative_error']*100:.2f}% {status}")

    def recommend_strategy(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend mitigation strategy based on scan results"""
        problematic_ids = report['summary']['problematic_layer_ids']
        total_layers = report['summary']['total_layers']

        recommendations = {
            'strategy': None,
            'details': [],
            'aqn_layers': [],
            'mixed_precision_layers': [],
        }

        if not problematic_ids:
            recommendations['strategy'] = 'no_action'
            recommendations['details'].append(
                "No problematic layers detected. MXFP4 quantization appears safe for this model."
            )
            return recommendations

        problematic_ratio = len(problematic_ids) / total_layers

        if problematic_ratio < 0.1:  # < 10% of layers
            recommendations['strategy'] = 'targeted_mitigation'
            recommendations['details'].append(
                f"Only {len(problematic_ids)} layers ({problematic_ratio*100:.1f}%) are problematic."
            )
            recommendations['details'].append(
                "Recommend: Keep problematic layers in higher precision (FP16/FP32)."
            )
            recommendations['mixed_precision_layers'] = problematic_ids

        elif problematic_ratio < 0.3:  # 10-30% of layers
            recommendations['strategy'] = 'mixed_approach'
            recommendations['details'].append(
                f"{len(problematic_ids)} layers ({problematic_ratio*100:.1f}%) are problematic."
            )
            recommendations['details'].append(
                "Recommend: Mixed precision for critical layers + AQN for moderate issues."
            )

            # Split by severity
            for layer in report['per_layer']:
                if layer['layer_id'] in problematic_ids:
                    if layer['sqnr_db'] < 15:  # Very low SQNR
                        recommendations['mixed_precision_layers'].append(layer['layer_id'])
                    else:
                        recommendations['aqn_layers'].append(layer['layer_id'])

        else:  # > 30% of layers
            recommendations['strategy'] = 'reconsider_quantization'
            recommendations['details'].append(
                f"{len(problematic_ids)} layers ({problematic_ratio*100:.1f}%) are problematic."
            )
            recommendations['details'].append(
                "Warning: MXFP4 may not be suitable for this model."
            )
            recommendations['details'].append(
                "Recommend: Consider MXFP8 or higher precision format."
            )
            recommendations['mixed_precision_layers'] = problematic_ids

        return recommendations

    def run_full_scan(self) -> Dict[str, Any]:
        """Run full quantization scan and analysis"""
        self.load_model()

        results = self.scan_all_layers()
        report = self.analyze_results(results)
        self.print_report(report)

        recommendations = self.recommend_strategy(report)

        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        print(f"Strategy: {recommendations['strategy']}")
        for detail in recommendations['details']:
            print(f"  • {detail}")

        if recommendations['mixed_precision_layers']:
            print(f"\nLayers to keep in higher precision: {recommendations['mixed_precision_layers']}")
        if recommendations['aqn_layers']:
            print(f"Layers for AQN training: {recommendations['aqn_layers']}")

        return {
            'config': {
                'model_path': self.model_path,
                'quant_type': self.config.quant_type,
                'num_layers': self.num_layers,
                'thresholds': {
                    'sqnr_db': self.config.sqnr_threshold_db,
                    'deadzone': self.config.deadzone_threshold,
                    'saturation': self.config.saturation_threshold,
                    'relative_error': self.config.relative_error_threshold,
                },
            },
            'report': report,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
        }


def main():
    parser = argparse.ArgumentParser(description="SRDD Quantization Scanner")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--quant_type", type=str, default="mxfp4",
                        choices=['mxfp4', 'mxfp4_sr', 'mxfp4_2d'],
                        help="Quantization type")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--sqnr_threshold", type=float, default=20.0,
                        help="SQNR threshold in dB (below = problematic)")
    parser.add_argument("--deadzone_threshold", type=float, default=0.05,
                        help="Deadzone threshold (above = problematic)")
    parser.add_argument("--saturation_threshold", type=float, default=0.01,
                        help="Saturation threshold (above = problematic)")
    parser.add_argument("--relative_error_threshold", type=float, default=0.1,
                        help="Relative error threshold (above = problematic)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    config = QuantScanConfig(
        quant_type=args.quant_type,
        sqnr_threshold_db=args.sqnr_threshold,
        deadzone_threshold=args.deadzone_threshold,
        saturation_threshold=args.saturation_threshold,
        relative_error_threshold=args.relative_error_threshold,
    )

    scanner = SRDDQuantScanner(
        model_path=args.model_path,
        config=config,
        device=args.device,
    )

    results = scanner.run_full_scan()

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
    print("SCAN COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
