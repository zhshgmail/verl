#!/usr/bin/env python3
"""
SRDD + MXFP4 Fake Quantization Experiment

This experiment uses SRDD diagnostic tools to identify quantization-sensitive layers
in MXFP4 quantized models, then validates SRDD-guided AQN vs baseline AQN.

Workflow:
1. Load Qwen2.5-1.5B in BF16
2. Apply MXFP4 fake quantization (via quant_compute)
3. Run SRDD scans on both BF16 and MXFP4 models
4. Compare metrics to identify problematic layers
5. If issues found, test Global AQN vs SRDD-guided AQN

Usage:
    python scripts/srdd_mxfp4_experiment.py \
        --model_path /data/z00637938/hub/Qwen2.5-1.5B-Instruct \
        --quant_type mxfp4 \
        --output results_mxfp4_srdd.json
"""

import os
import sys
import json
import argparse
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import numpy as np

import torch
import torch.nn.functional as F

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/zheng/workspace/quant_compute')

# Try to import quant_compute
try:
    from quant_cy_npu import QType, quant_dequant_float
    QUANT_AVAILABLE = True
    print("[OK] quant_compute library loaded")
except ImportError as e:
    QUANT_AVAILABLE = False
    print(f"[WARN] quant_compute not available: {e}")
    print("[WARN] Will use simulated quantization instead")


@dataclass
class SRDDResult:
    """SRDD scan results for a single layer"""
    layer_id: int
    gain: float
    kurtosis: float
    instability: float


class FakeQuantHook:
    """Hook that applies MXFP4 fake quantization to layer output"""

    def __init__(self, qtype: str = 'mxfp4', force_py: bool = True):
        self.qtype = qtype
        self.force_py = force_py
        self.call_count = 0
        self.total_error = 0.0
        self.total_elements = 0

        if QUANT_AVAILABLE:
            self.Q = QType(qtype)
        else:
            self.Q = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        if self.Q is not None:
            # Use quant_compute library
            hidden_quant = quant_dequant_float(hidden, self.Q, force_py=self.force_py)
        else:
            # Simulated MXFP4 quantization (fallback)
            hidden_quant = self._simulated_mxfp4(hidden)

        # Track quantization error
        error = (hidden_quant - hidden).abs().mean().item()
        self.total_error += error
        self.total_elements += 1
        self.call_count += 1

        if isinstance(output, tuple):
            return (hidden_quant,) + output[1:]
        return hidden_quant

    def _simulated_mxfp4(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback: simulate MXFP4 quantization (E2M1, block size 32)"""
        # MXFP4 has limited precision: values map to {0, 0.5, 1, 1.5, 2, 3, 4, 6}
        # with shared exponent per block of 32

        shape = x.shape
        x_flat = x.view(-1, 32) if x.numel() >= 32 else x.view(1, -1)

        # Compute shared exponent per block
        max_abs = x_flat.abs().max(dim=-1, keepdim=True)[0]
        shared_exp = torch.floor(torch.log2(max_abs + 1e-10))

        # Scale to MXFP4 range
        x_scaled = x_flat / (2 ** shared_exp)

        # Quantize mantissa (E2M1 = 4 values per sign)
        # Values: 0, 0.5, 1, 1.5 (subnormal) and 1, 1.5, 2, 3 (normal)
        x_sign = torch.sign(x_scaled)
        x_abs = x_scaled.abs()

        # Simplified MXFP4 rounding
        x_quant = torch.round(x_abs * 2) / 2
        x_quant = x_quant.clamp(0, 6)  # Max representable value

        # Dequantize
        x_dequant = x_sign * x_quant * (2 ** shared_exp)

        return x_dequant.view(shape)

    def get_avg_error(self) -> float:
        return self.total_error / max(1, self.total_elements)


class GainScanHook:
    """Hook for SRDD gain scan (measures layer transfer function)"""

    def __init__(self, noise_scale: float = 0.1):
        self.noise_scale = noise_scale
        self.baseline_output = None
        self.noisy_output = None
        self.input_noise_std = None

    def set_noise(self, noise_std: float):
        self.input_noise_std = noise_std

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        if self.baseline_output is None:
            self.baseline_output = hidden.clone()
        else:
            self.noisy_output = hidden.clone()

        return output

    def compute_gain(self) -> float:
        if self.baseline_output is None or self.noisy_output is None:
            return 1.0

        output_diff = (self.noisy_output - self.baseline_output).std().item()
        if self.input_noise_std is None or self.input_noise_std == 0:
            return 1.0

        return output_diff / self.input_noise_std

    def reset(self):
        self.baseline_output = None
        self.noisy_output = None
        self.input_noise_std = None


class MXFP4SRDDExperiment:
    """Main experiment class for SRDD + MXFP4 quantization analysis"""

    def __init__(
        self,
        model_path: str,
        quant_type: str = 'mxfp4',
        device: str = 'cuda',
        force_py: bool = True,
    ):
        self.model_path = model_path
        self.quant_type = quant_type
        self.device = device
        self.force_py = force_py

        self.tokenizer = None
        self.model_bf16 = None
        self.num_layers = None

        # Test prompts for SRDD scans
        self.test_prompts = [
            "The capital of France is",
            "Machine learning is a subset of",
            "The answer to 2 + 2 is",
            "Python is a programming language that",
            "Water freezes at a temperature of",
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

        self.model_bf16 = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model_bf16.eval()
        self.num_layers = len(self.model_bf16.model.layers)
        print(f"Model loaded: {self.num_layers} layers")

    def run_gain_scan(
        self,
        model,
        with_quant: bool = False,
        noise_scale: float = 0.1,
    ) -> Dict[int, float]:
        """Run gain scan to detect deadzone issues"""
        print(f"\n--- Gain Scan (with_quant={with_quant}) ---")
        gains = {}

        for layer_id in range(self.num_layers):
            # Register hooks
            hooks = []
            layer = model.model.layers[layer_id]

            # Add quantization hook if requested
            if with_quant:
                quant_hook = FakeQuantHook(self.quant_type, self.force_py)
                hooks.append(layer.register_forward_hook(quant_hook))

            # Prepare inputs
            inputs = self.tokenizer(
                self.test_prompts[0],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # Baseline forward
            with torch.no_grad():
                baseline_outputs = []

                def capture_baseline(m, inp, out):
                    if isinstance(out, tuple):
                        baseline_outputs.append(out[0].clone())
                    else:
                        baseline_outputs.append(out.clone())
                    return out

                h = layer.register_forward_hook(capture_baseline)
                model(**inputs)
                h.remove()

                # Noisy forward (inject noise at layer input)
                noise_std = noise_scale * baseline_outputs[0].std().item()
                noisy_outputs = []

                def inject_noise_and_capture(m, inp, out):
                    if isinstance(out, tuple):
                        hidden = out[0]
                    else:
                        hidden = out

                    # Add noise proportional to activation scale
                    noise = torch.randn_like(hidden) * noise_std
                    hidden_noisy = hidden + noise
                    noisy_outputs.append(hidden_noisy.clone())

                    if isinstance(out, tuple):
                        return (hidden_noisy,) + out[1:]
                    return hidden_noisy

                h = layer.register_forward_hook(inject_noise_and_capture)
                model(**inputs)
                h.remove()

            # Compute gain
            if len(baseline_outputs) > 0 and len(noisy_outputs) > 0:
                output_diff_std = (noisy_outputs[0] - baseline_outputs[0]).std().item()
                gain = output_diff_std / noise_std if noise_std > 0 else 1.0
            else:
                gain = 1.0

            gains[layer_id] = gain

            # Remove hooks
            for h in hooks:
                h.remove()

        return gains

    def run_kurtosis_scan(
        self,
        model,
        with_quant: bool = False,
    ) -> Dict[int, float]:
        """Run kurtosis scan to detect saturation issues"""
        print(f"\n--- Kurtosis Scan (with_quant={with_quant}) ---")
        kurtosis_values = {}

        for layer_id in range(self.num_layers):
            hooks = []
            layer = model.model.layers[layer_id]

            # Add quantization hook if requested
            if with_quant:
                quant_hook = FakeQuantHook(self.quant_type, self.force_py)
                hooks.append(layer.register_forward_hook(quant_hook))

            # Capture outputs
            captured = []

            def capture_output(m, inp, out):
                if isinstance(out, tuple):
                    captured.append(out[0].clone())
                else:
                    captured.append(out.clone())
                return out

            h = layer.register_forward_hook(capture_output)

            # Forward pass
            inputs = self.tokenizer(
                self.test_prompts[0],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                model(**inputs)

            h.remove()
            for hook in hooks:
                hook.remove()

            # Compute kurtosis
            if len(captured) > 0:
                values = captured[0].float().flatten().cpu().numpy()
                k = stats.kurtosis(values, fisher=True)
                kurtosis_values[layer_id] = float(k)
            else:
                kurtosis_values[layer_id] = 0.0

        return kurtosis_values

    def run_instability_scan(
        self,
        model,
        with_quant: bool = False,
        num_trials: int = 3,
    ) -> Dict[int, float]:
        """Run instability scan (for stochastic rounding detection)"""
        print(f"\n--- Instability Scan (with_quant={with_quant}) ---")
        instability = {}

        inputs = self.tokenizer(
            self.test_prompts[0],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        for layer_id in range(self.num_layers):
            layer = model.model.layers[layer_id]
            trial_outputs = []

            for trial in range(num_trials):
                hooks = []

                # Add quantization hook if requested
                if with_quant:
                    quant_hook = FakeQuantHook(self.quant_type, self.force_py)
                    hooks.append(layer.register_forward_hook(quant_hook))

                captured = []

                def capture_output(m, inp, out):
                    if isinstance(out, tuple):
                        captured.append(out[0].clone())
                    else:
                        captured.append(out.clone())
                    return out

                h = layer.register_forward_hook(capture_output)

                with torch.no_grad():
                    model(**inputs)

                h.remove()
                for hook in hooks:
                    hook.remove()

                if len(captured) > 0:
                    trial_outputs.append(captured[0])

            # Compute instability (std across trials)
            if len(trial_outputs) >= 2:
                stacked = torch.stack(trial_outputs, dim=0)
                inst = stacked.std(dim=0).mean().item()
            else:
                inst = 0.0

            instability[layer_id] = inst

        return instability

    def compute_layer_loss(
        self,
        model,
        with_quant_layers: Optional[List[int]] = None,
    ) -> float:
        """Compute cross-entropy loss with optional per-layer quantization"""
        total_loss = 0.0

        # Register quantization hooks for specified layers
        hooks = []
        if with_quant_layers is not None:
            for layer_id in with_quant_layers:
                quant_hook = FakeQuantHook(self.quant_type, self.force_py)
                h = model.model.layers[layer_id].register_forward_hook(quant_hook)
                hooks.append(h)

        for text in self.test_prompts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs.input_ids)
                total_loss += outputs.loss.item()

        # Remove hooks
        for h in hooks:
            h.remove()

        return total_loss / len(self.test_prompts)

    def compare_bf16_vs_mxfp4(self) -> Dict[str, Any]:
        """Compare SRDD metrics between BF16 and MXFP4"""
        print("\n" + "=" * 60)
        print("PHASE 1: BF16 vs MXFP4 SRDD Comparison")
        print("=" * 60)

        results = {
            'bf16': {'gains': {}, 'kurtosis': {}, 'instability': {}},
            'mxfp4': {'gains': {}, 'kurtosis': {}, 'instability': {}},
        }

        # BF16 baseline scans
        print("\n[BF16 Baseline Scans]")
        results['bf16']['gains'] = self.run_gain_scan(self.model_bf16, with_quant=False)
        results['bf16']['kurtosis'] = self.run_kurtosis_scan(self.model_bf16, with_quant=False)
        results['bf16']['instability'] = self.run_instability_scan(self.model_bf16, with_quant=False)

        # MXFP4 scans (with fake quantization)
        print("\n[MXFP4 Quantized Scans]")
        results['mxfp4']['gains'] = self.run_gain_scan(self.model_bf16, with_quant=True)
        results['mxfp4']['kurtosis'] = self.run_kurtosis_scan(self.model_bf16, with_quant=True)
        results['mxfp4']['instability'] = self.run_instability_scan(self.model_bf16, with_quant=True)

        # Compute loss
        print("\n[Loss Comparison]")
        results['bf16']['loss'] = self.compute_layer_loss(self.model_bf16, with_quant_layers=None)
        results['mxfp4']['loss'] = self.compute_layer_loss(self.model_bf16, with_quant_layers=list(range(self.num_layers)))

        print(f"  BF16 loss: {results['bf16']['loss']:.4f}")
        print(f"  MXFP4 loss: {results['mxfp4']['loss']:.4f}")
        print(f"  Degradation: {(results['mxfp4']['loss'] - results['bf16']['loss']) / results['bf16']['loss'] * 100:+.2f}%")

        return results

    def identify_problematic_layers(
        self,
        results: Dict[str, Any],
        gain_threshold: float = 0.9,
        kurtosis_threshold: float = 0.9,
    ) -> List[int]:
        """Identify layers most affected by quantization"""
        print("\n" + "=" * 60)
        print("PHASE 2: Identify Problematic Layers")
        print("=" * 60)

        problematic = []
        layer_issues = {}

        for layer_id in range(self.num_layers):
            issues = []

            # Check gain drop (deadzone indicator)
            bf16_gain = results['bf16']['gains'].get(layer_id, 1.0)
            mxfp4_gain = results['mxfp4']['gains'].get(layer_id, 1.0)
            gain_ratio = mxfp4_gain / bf16_gain if bf16_gain > 0 else 1.0

            if gain_ratio < gain_threshold:
                issues.append(f"gain_drop={gain_ratio:.3f}")

            # Check kurtosis drop (saturation indicator)
            bf16_kurt = results['bf16']['kurtosis'].get(layer_id, 3500)
            mxfp4_kurt = results['mxfp4']['kurtosis'].get(layer_id, 3500)
            kurt_ratio = mxfp4_kurt / bf16_kurt if bf16_kurt > 0 else 1.0

            if kurt_ratio < kurtosis_threshold:
                issues.append(f"kurtosis_drop={kurt_ratio:.3f}")

            # Check instability increase
            bf16_inst = results['bf16']['instability'].get(layer_id, 0.0)
            mxfp4_inst = results['mxfp4']['instability'].get(layer_id, 0.0)

            if mxfp4_inst > bf16_inst + 0.1:  # Significant increase
                issues.append(f"instability_increase={mxfp4_inst - bf16_inst:.3f}")

            if issues:
                problematic.append(layer_id)
                layer_issues[layer_id] = issues

        # Print summary
        print(f"\nFound {len(problematic)} problematic layers:")
        for layer_id in problematic:
            print(f"  Layer {layer_id}: {', '.join(layer_issues[layer_id])}")

        if not problematic:
            print("  No significant issues detected!")
            print("  All layers appear robust to MXFP4 quantization.")

        return problematic

    def run_aqn_comparison(
        self,
        problematic_layers: List[int],
        aqn_gamma: float = 0.01,
        num_runs: int = 5,
    ) -> Dict[str, Any]:
        """Compare Global AQN vs SRDD-guided AQN"""
        print("\n" + "=" * 60)
        print("PHASE 3: AQN Strategy Comparison")
        print("=" * 60)

        if not problematic_layers:
            print("No problematic layers identified. Skipping AQN comparison.")
            return {}

        all_layers = list(range(self.num_layers))
        healthy_layers = [l for l in all_layers if l not in problematic_layers]

        configs = {
            'baseline': [],  # MXFP4 only, no AQN
            'global_aqn': all_layers,  # AQN on all layers
            'targeted_aqn': problematic_layers,  # AQN only on problematic
            'healthy_aqn': healthy_layers,  # AQN only on healthy (control)
        }

        results = {}

        for config_name, aqn_layers in configs.items():
            print(f"\n[{config_name}] AQN on {len(aqn_layers)} layers")
            losses = []

            for run in range(num_runs):
                torch.manual_seed(42 + run)
                loss = self._compute_loss_with_quant_and_aqn(
                    quant_layers=all_layers,  # Always quantize all
                    aqn_layers=aqn_layers,
                    aqn_gamma=aqn_gamma,
                )
                losses.append(loss)

            mean_loss = np.mean(losses)
            std_loss = np.std(losses)
            results[config_name] = {
                'losses': losses,
                'mean': mean_loss,
                'std': std_loss,
                'aqn_layers': aqn_layers,
            }
            print(f"  Loss: {mean_loss:.4f} +/- {std_loss:.4f}")

        # Statistical comparison
        if 'global_aqn' in results and 'targeted_aqn' in results:
            t_stat, p_value = stats.ttest_ind(
                results['global_aqn']['losses'],
                results['targeted_aqn']['losses'],
            )
            results['comparison'] = {
                't_stat': t_stat,
                'p_value': p_value,
                'targeted_better': results['targeted_aqn']['mean'] < results['global_aqn']['mean'],
            }
            print(f"\n[Statistical Comparison]")
            print(f"  Targeted vs Global: p={p_value:.4f}")
            if results['targeted_aqn']['mean'] < results['global_aqn']['mean']:
                improvement = (results['global_aqn']['mean'] - results['targeted_aqn']['mean']) / results['global_aqn']['mean'] * 100
                print(f"  SRDD-guided AQN is {improvement:.2f}% better!")
            else:
                print(f"  Global AQN performs better or equal")

        return results

    def _compute_loss_with_quant_and_aqn(
        self,
        quant_layers: List[int],
        aqn_layers: List[int],
        aqn_gamma: float,
    ) -> float:
        """Compute loss with both quantization and AQN hooks"""
        hooks = []

        # Register combined hooks
        for layer_id in range(self.num_layers):
            apply_quant = layer_id in quant_layers
            apply_aqn = layer_id in aqn_layers

            if apply_quant or apply_aqn:
                hook = CombinedQuantAQNHook(
                    qtype=self.quant_type if apply_quant else None,
                    aqn_gamma=aqn_gamma if apply_aqn else 0.0,
                    force_py=self.force_py,
                )
                h = self.model_bf16.model.layers[layer_id].register_forward_hook(hook)
                hooks.append(h)

        # Compute loss
        total_loss = 0.0
        for text in self.test_prompts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model_bf16(**inputs, labels=inputs.input_ids)
                total_loss += outputs.loss.item()

        # Remove hooks
        for h in hooks:
            h.remove()

        return total_loss / len(self.test_prompts)

    def run_full_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment"""
        print("=" * 60)
        print("SRDD + MXFP4 Quantization Experiment")
        print("=" * 60)
        print(f"Model: {self.model_path}")
        print(f"Quantization: {self.quant_type}")
        print(f"Device: {self.device}")
        print(f"Force Python: {self.force_py}")

        # Load model
        self.load_model()

        # Phase 1: Compare BF16 vs MXFP4
        comparison_results = self.compare_bf16_vs_mxfp4()

        # Phase 2: Identify problematic layers
        problematic_layers = self.identify_problematic_layers(comparison_results)

        # Phase 3: AQN comparison (if issues found)
        aqn_results = {}
        if problematic_layers:
            aqn_results = self.run_aqn_comparison(problematic_layers)

        # Summary
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Total layers: {self.num_layers}")
        print(f"Problematic layers: {len(problematic_layers)} - {problematic_layers}")
        print(f"BF16 loss: {comparison_results['bf16']['loss']:.4f}")
        print(f"MXFP4 loss: {comparison_results['mxfp4']['loss']:.4f}")

        if aqn_results and 'comparison' in aqn_results:
            print(f"\nAQN Comparison (p={aqn_results['comparison']['p_value']:.4f}):")
            print(f"  Global AQN: {aqn_results['global_aqn']['mean']:.4f}")
            print(f"  Targeted AQN: {aqn_results['targeted_aqn']['mean']:.4f}")

        return {
            'config': {
                'model_path': self.model_path,
                'quant_type': self.quant_type,
                'num_layers': self.num_layers,
            },
            'comparison': comparison_results,
            'problematic_layers': problematic_layers,
            'aqn_results': aqn_results,
            'timestamp': datetime.now().isoformat(),
        }


class CombinedQuantAQNHook:
    """Combined hook for quantization and AQN"""

    def __init__(
        self,
        qtype: Optional[str] = None,
        aqn_gamma: float = 0.0,
        force_py: bool = True,
    ):
        self.qtype = qtype
        self.aqn_gamma = aqn_gamma
        self.force_py = force_py

        if qtype and QUANT_AVAILABLE:
            self.Q = QType(qtype)
        else:
            self.Q = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        result = hidden

        # Step 1: Apply AQN (before quantization)
        if self.aqn_gamma > 0:
            noise = torch.randn_like(result) * self.aqn_gamma * result.abs()
            result = result + noise

        # Step 2: Apply quantization
        if self.qtype:
            if self.Q is not None:
                result = quant_dequant_float(result, self.Q, force_py=self.force_py)
            else:
                # Fallback simulation
                result = self._simulated_mxfp4(result)

        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result

    def _simulated_mxfp4(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback MXFP4 simulation"""
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

        return x_dequant.view(shape)


def main():
    parser = argparse.ArgumentParser(description="SRDD + MXFP4 Quantization Experiment")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--quant_type", type=str, default="mxfp4",
                        choices=['mxfp4', 'mxfp4_sr', 'mxfp4_2d', 'mxfp4_v2'],
                        help="Quantization type")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--force_py", action="store_true", default=True,
                        help="Force Python quantization (for non-NPU)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    experiment = MXFP4SRDDExperiment(
        model_path=args.model_path,
        quant_type=args.quant_type,
        device=args.device,
        force_py=args.force_py,
    )

    results = experiment.run_full_experiment()

    if args.output:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        results = convert_numpy(results)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
