#!/usr/bin/env python3
"""
Self-Referential Differential Diagnosis (SRDD) for Error Source Finding.

This method finds hardware error sources WITHOUT a reference system (no GPU needed).
It uses controllable noise injection as a probe to detect ANOMALOUS layer behavior.

Key Insight:
- We cannot compare absolute sensitivity (L27=92% vs L10=0.2% is meaningless)
- Instead, we detect ANOMALIES in how each layer responds to controlled perturbations
- Hardware errors are DISCRETE (affect specific layers)
- Architectural sensitivity is CONTINUOUS (smooth across layers)

Three Probes:
1. Linearity Probe: Correct layers show linear response to noise scale
   - Faulty layers show saturation, dead zones, or non-monotonic behavior

2. Neighborhood Smoothness Probe: Find discontinuities in sensitivity curve
   - Use second derivative to detect "spikes"
   - Hardware faults create sudden jumps in otherwise smooth curves

3. Input Invariance Probe: Detect data-dependent faults
   - Faulty layers show inconsistent behavior across different inputs
   - "Neurotic" response: normal for 99 inputs, explodes on 1
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from verl.utils.noisy_ops import (
    enable_noisy_ops,
    disable_noisy_ops,
    set_selective_layers,
    set_selective_operators,
    register_layer_hooks,
    reset_injection_stats,
    reset_layer_injection_stats,
)


class SRDDErrorFinder:
    """
    Self-Referential Differential Diagnosis for finding hardware error sources.

    Works WITHOUT a reference system - uses statistical anomaly detection
    to find layers with abnormal response to controlled noise injection.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Get number of layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.num_layers = len(model.model.layers)
        else:
            self.num_layers = 28  # Default for Qwen2.5

        # Register layer hooks
        self.num_hooks = register_layer_hooks(model)
        print(f"[SRDD] Registered {self.num_hooks} layer hooks")
        print(f"[SRDD] Model has {self.num_layers} layers")

    def get_output_logits(self, prompt: str) -> torch.Tensor:
        """Get model output logits for a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :].clone()

        return logits

    def compute_kl_divergence(self, p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
        """Compute KL divergence between two logit distributions."""
        p = torch.softmax(p_logits, dim=-1)
        q = torch.softmax(q_logits, dim=-1)

        eps = 1e-10
        kl = torch.sum(p * torch.log((p + eps) / (q + eps)))
        return kl.item()

    def probe_linearity(
        self,
        prompts: List[str],
        noise_scales: List[float] = [0.01, 0.02, 0.05, 0.10, 0.15],
        num_trials: int = 3,
    ) -> Dict[int, Dict]:
        """
        Probe A: Linearity Test

        Measures how linearly each layer responds to increasing noise levels.

        - Correct layer: noise scale vs divergence shows linear/monotonic relationship (R² ≈ 1)
        - Faulty layer: shows saturation, dead zones, or non-monotonic behavior (R² << 1)

        Returns:
            Dict mapping layer_id to {linearity_r2, sensitivities, is_anomaly}
        """
        print(f"\n{'='*60}")
        print("PROBE A: LINEARITY TEST")
        print(f"{'='*60}")
        print(f"Noise scales: {noise_scales}")
        print(f"Trials per scale: {num_trials}")
        print(f"Testing {self.num_layers} layers...")

        # Step 1: Get baseline (no noise)
        print("\n[Step 1] Getting baseline outputs...")
        disable_noisy_ops()
        set_selective_layers(None)

        baseline_logits = []
        for prompt in prompts:
            logits = self.get_output_logits(prompt)
            baseline_logits.append(logits)

        # Step 2: For each layer, measure divergence at each noise scale
        print("\n[Step 2] Measuring noise response for each layer...")

        layer_results = {}

        for layer_id in range(self.num_layers):
            sensitivities = []

            for scale in noise_scales:
                # Run multiple trials and average
                trial_divergences = []

                for trial in range(num_trials):
                    set_selective_layers([layer_id])
                    enable_noisy_ops(error_scale=scale, error_type='relative_gaussian')

                    # Set different seed for each trial
                    torch.manual_seed(42 + layer_id * 1000 + trial * 100 + int(scale * 1000))

                    prompt_divergences = []
                    for i, prompt in enumerate(prompts):
                        noisy_logits = self.get_output_logits(prompt)
                        div = self.compute_kl_divergence(baseline_logits[i], noisy_logits)
                        prompt_divergences.append(div)

                    trial_divergences.append(np.mean(prompt_divergences))
                    disable_noisy_ops()

                sensitivities.append(np.mean(trial_divergences))

            # Compute linearity (Pearson correlation coefficient)
            # R² = correlation^2
            if len(set(sensitivities)) > 1:  # Need variance to compute correlation
                r, _ = stats.pearsonr(noise_scales, sensitivities)
                r_squared = r ** 2
            else:
                r_squared = 0.0  # No variance = suspicious

            layer_results[layer_id] = {
                'sensitivities': sensitivities,
                'linearity_r2': r_squared,
                'mean_sensitivity': np.mean(sensitivities),
            }

            if (layer_id + 1) % 5 == 0 or layer_id == 0:
                print(f"  Layer {layer_id:2d}: R² = {r_squared:.4f}, mean_sens = {np.mean(sensitivities):.4f}")

        set_selective_layers(None)

        # Step 3: Detect anomalies (layers with low R²)
        print("\n[Step 3] Detecting linearity anomalies...")

        r2_values = [r['linearity_r2'] for r in layer_results.values()]
        r2_mean = np.mean(r2_values)
        r2_std = np.std(r2_values)

        # Z-score based anomaly detection
        for layer_id, result in layer_results.items():
            z_score = (result['linearity_r2'] - r2_mean) / (r2_std + 1e-10)
            result['z_score'] = z_score
            # Anomaly: significantly lower R² than average (z < -2)
            result['is_anomaly'] = z_score < -2.0

        # Report anomalies
        anomalies = [(lid, r) for lid, r in layer_results.items() if r['is_anomaly']]
        if anomalies:
            print(f"\n  Found {len(anomalies)} anomalous layers (R² significantly below average):")
            for lid, r in sorted(anomalies, key=lambda x: x[1]['linearity_r2']):
                print(f"    Layer {lid}: R² = {r['linearity_r2']:.4f} (z = {r['z_score']:.2f})")
        else:
            print("\n  No significant linearity anomalies detected.")

        # Rank by R² (lowest = most suspicious)
        sorted_by_r2 = sorted(layer_results.items(), key=lambda x: x[1]['linearity_r2'])
        print("\n  Top 5 most suspicious layers (lowest R²):")
        for rank, (lid, r) in enumerate(sorted_by_r2[:5], 1):
            print(f"    {rank}. Layer {lid}: R² = {r['linearity_r2']:.4f}")

        return layer_results

    def probe_neighborhood_smoothness(
        self,
        linearity_results: Dict[int, Dict] = None,
        prompts: List[str] = None,
        noise_scale: float = 0.05,
    ) -> Dict[int, Dict]:
        """
        Probe B: Neighborhood Smoothness Test

        Detects discontinuities in the sensitivity curve across layers.
        Uses second derivative to find "spikes" that indicate hardware faults.

        - Hardware errors are DISCRETE (affect specific layers)
        - Architectural sensitivity is CONTINUOUS (smooth across layers)
        - Sudden jumps in sensitivity indicate faults

        Args:
            linearity_results: Results from probe_linearity (reuse sensitivity data)
            prompts: Test prompts (only needed if linearity_results not provided)
            noise_scale: Noise scale for sensitivity measurement

        Returns:
            Dict mapping layer_id to {local_anomaly, second_derivative, is_spike}
        """
        print(f"\n{'='*60}")
        print("PROBE B: NEIGHBORHOOD SMOOTHNESS TEST")
        print(f"{'='*60}")

        # Get sensitivity values
        if linearity_results:
            print("Using sensitivity data from Linearity Probe...")
            sensitivities = [linearity_results[i]['mean_sensitivity'] for i in range(self.num_layers)]
        else:
            # Compute sensitivities if not provided
            print(f"Computing sensitivities at scale {noise_scale}...")
            disable_noisy_ops()
            set_selective_layers(None)

            baseline_logits = []
            for prompt in prompts:
                logits = self.get_output_logits(prompt)
                baseline_logits.append(logits)

            sensitivities = []
            for layer_id in range(self.num_layers):
                set_selective_layers([layer_id])
                enable_noisy_ops(error_scale=noise_scale, error_type='relative_gaussian')

                divs = []
                for i, prompt in enumerate(prompts):
                    noisy_logits = self.get_output_logits(prompt)
                    div = self.compute_kl_divergence(baseline_logits[i], noisy_logits)
                    divs.append(div)

                sensitivities.append(np.mean(divs))
                disable_noisy_ops()

            set_selective_layers(None)

        print(f"Analyzing smoothness across {len(sensitivities)} layers...")

        # Compute second derivative (local anomaly)
        # Local anomaly = |S[i] - (S[i-1] + S[i+1])/2|
        # This measures how much layer i deviates from the linear interpolation of its neighbors

        layer_results = {}
        local_anomalies = []

        for i in range(self.num_layers):
            if i == 0:
                # First layer: compare with next two
                local_anomaly = abs(sensitivities[i] - sensitivities[i+1])
            elif i == self.num_layers - 1:
                # Last layer: compare with previous two
                local_anomaly = abs(sensitivities[i] - sensitivities[i-1])
            else:
                # Middle layers: second derivative
                expected = (sensitivities[i-1] + sensitivities[i+1]) / 2
                local_anomaly = abs(sensitivities[i] - expected)

            local_anomalies.append(local_anomaly)
            layer_results[i] = {
                'sensitivity': sensitivities[i],
                'local_anomaly': local_anomaly,
            }

        # Compute second derivative more formally
        first_derivative = np.diff(sensitivities)
        second_derivative = np.diff(first_derivative)

        # Pad to match layer count
        second_derivative = np.concatenate([[0], second_derivative, [0]])

        for i in range(self.num_layers):
            layer_results[i]['second_derivative'] = abs(second_derivative[i])

        # Detect spikes using Z-score
        anomaly_mean = np.mean(local_anomalies)
        anomaly_std = np.std(local_anomalies)

        print("\n[Analysis] Local anomaly statistics:")
        print(f"  Mean: {anomaly_mean:.6f}")
        print(f"  Std:  {anomaly_std:.6f}")

        for i, result in layer_results.items():
            z_score = (result['local_anomaly'] - anomaly_mean) / (anomaly_std + 1e-10)
            result['z_score'] = z_score
            # Spike: significantly higher local anomaly (z > 2)
            result['is_spike'] = z_score > 2.0

        # Report spikes
        spikes = [(lid, r) for lid, r in layer_results.items() if r['is_spike']]
        if spikes:
            print(f"\n  Found {len(spikes)} smoothness spikes:")
            for lid, r in sorted(spikes, key=lambda x: x[1]['local_anomaly'], reverse=True):
                print(f"    Layer {lid}: local_anomaly = {r['local_anomaly']:.6f} (z = {r['z_score']:.2f})")
        else:
            print("\n  No significant smoothness spikes detected.")

        # Rank by local anomaly (highest = most suspicious)
        sorted_by_anomaly = sorted(layer_results.items(), key=lambda x: x[1]['local_anomaly'], reverse=True)
        print("\n  Top 5 most suspicious layers (highest local anomaly):")
        for rank, (lid, r) in enumerate(sorted_by_anomaly[:5], 1):
            print(f"    {rank}. Layer {lid}: anomaly = {r['local_anomaly']:.6f}")

        return layer_results

    def probe_input_invariance(
        self,
        diverse_prompts: List[str],
        noise_scale: float = 0.05,
        num_trials: int = 3,
    ) -> Dict[int, Dict]:
        """
        Probe C: Input Invariance Test

        Detects data-dependent faults - layers that behave inconsistently across inputs.

        - Correct layer: consistent noise response across different inputs
        - Faulty layer: "neurotic" - normal for most inputs, explodes on some

        Args:
            diverse_prompts: Diverse set of prompts (simple to complex)
            noise_scale: Noise scale for testing
            num_trials: Number of trials per input

        Returns:
            Dict mapping layer_id to {variance_across_inputs, cv, is_neurotic}
        """
        print(f"\n{'='*60}")
        print("PROBE C: INPUT INVARIANCE TEST")
        print(f"{'='*60}")
        print(f"Testing with {len(diverse_prompts)} diverse prompts")
        print(f"Noise scale: {noise_scale}")

        # Step 1: Get baseline for all prompts
        print("\n[Step 1] Getting baseline outputs for all prompts...")
        disable_noisy_ops()
        set_selective_layers(None)

        baseline_logits = []
        for prompt in diverse_prompts:
            logits = self.get_output_logits(prompt)
            baseline_logits.append(logits)

        # Step 2: For each layer, measure divergence variance across inputs
        print("\n[Step 2] Measuring input-dependent sensitivity for each layer...")

        layer_results = {}

        for layer_id in range(self.num_layers):
            input_sensitivities = []

            for prompt_idx, prompt in enumerate(diverse_prompts):
                trial_divs = []

                for trial in range(num_trials):
                    set_selective_layers([layer_id])
                    enable_noisy_ops(error_scale=noise_scale, error_type='relative_gaussian')
                    torch.manual_seed(42 + layer_id * 1000 + prompt_idx * 100 + trial)

                    noisy_logits = self.get_output_logits(prompt)
                    div = self.compute_kl_divergence(baseline_logits[prompt_idx], noisy_logits)
                    trial_divs.append(div)

                    disable_noisy_ops()

                input_sensitivities.append(np.mean(trial_divs))

            # Compute variance statistics
            mean_sens = np.mean(input_sensitivities)
            std_sens = np.std(input_sensitivities)
            # Coefficient of Variation (CV) = std/mean - normalized measure of variability
            cv = std_sens / (mean_sens + 1e-10)

            layer_results[layer_id] = {
                'input_sensitivities': input_sensitivities,
                'mean': mean_sens,
                'std': std_sens,
                'cv': cv,  # Higher CV = more inconsistent across inputs
            }

            if (layer_id + 1) % 5 == 0 or layer_id == 0:
                print(f"  Layer {layer_id:2d}: CV = {cv:.4f}, mean = {mean_sens:.4f}, std = {std_sens:.4f}")

        set_selective_layers(None)

        # Step 3: Detect "neurotic" layers (high CV)
        print("\n[Step 3] Detecting neurotic layers...")

        cv_values = [r['cv'] for r in layer_results.values()]
        cv_mean = np.mean(cv_values)
        cv_std = np.std(cv_values)

        for layer_id, result in layer_results.items():
            z_score = (result['cv'] - cv_mean) / (cv_std + 1e-10)
            result['z_score'] = z_score
            # Neurotic: significantly higher CV than average (z > 2)
            result['is_neurotic'] = z_score > 2.0

        # Report neurotic layers
        neurotic = [(lid, r) for lid, r in layer_results.items() if r['is_neurotic']]
        if neurotic:
            print(f"\n  Found {len(neurotic)} neurotic layers (inconsistent across inputs):")
            for lid, r in sorted(neurotic, key=lambda x: x[1]['cv'], reverse=True):
                print(f"    Layer {lid}: CV = {r['cv']:.4f} (z = {r['z_score']:.2f})")
        else:
            print("\n  No significantly neurotic layers detected.")

        # Rank by CV (highest = most suspicious)
        sorted_by_cv = sorted(layer_results.items(), key=lambda x: x[1]['cv'], reverse=True)
        print("\n  Top 5 most suspicious layers (highest CV):")
        for rank, (lid, r) in enumerate(sorted_by_cv[:5], 1):
            print(f"    {rank}. Layer {lid}: CV = {r['cv']:.4f}")

        return layer_results

    def run_full_diagnosis(
        self,
        prompts: List[str],
        diverse_prompts: List[str] = None,
        noise_scales: List[float] = [0.01, 0.02, 0.05, 0.10, 0.15],
        ground_truth_layer: int = None,  # For validation only
    ) -> Dict:
        """
        Run all three probes and aggregate results.

        Args:
            prompts: Basic test prompts
            diverse_prompts: Diverse prompts for input invariance test (optional)
            noise_scales: Noise scales for linearity test
            ground_truth_layer: If provided, used to validate results (simulation mode)

        Returns:
            Aggregated diagnosis results with ranked suspects
        """
        print(f"\n{'='*70}")
        print("SELF-REFERENTIAL DIFFERENTIAL DIAGNOSIS (SRDD)")
        print(f"{'='*70}")
        if ground_truth_layer is not None:
            print(f"Validation mode: Ground truth layer = {ground_truth_layer}")
        print(f"{'='*70}")

        # Use same prompts for all probes if diverse_prompts not provided
        if diverse_prompts is None:
            diverse_prompts = prompts

        # Run Probe A: Linearity
        linearity_results = self.probe_linearity(
            prompts=prompts,
            noise_scales=noise_scales,
        )

        # Run Probe B: Neighborhood Smoothness
        smoothness_results = self.probe_neighborhood_smoothness(
            linearity_results=linearity_results,
        )

        # Run Probe C: Input Invariance
        invariance_results = self.probe_input_invariance(
            diverse_prompts=diverse_prompts,
        )

        # Aggregate results using weighted voting
        print(f"\n{'='*60}")
        print("AGGREGATED DIAGNOSIS")
        print(f"{'='*60}")

        layer_scores = {}

        for layer_id in range(self.num_layers):
            # Collect anomaly indicators from each probe
            lin_anomaly = linearity_results[layer_id]['is_anomaly']
            lin_z = linearity_results[layer_id]['z_score']

            smooth_spike = smoothness_results[layer_id]['is_spike']
            smooth_z = smoothness_results[layer_id]['z_score']

            inv_neurotic = invariance_results[layer_id]['is_neurotic']
            inv_z = invariance_results[layer_id]['z_score']

            # Composite score: weighted sum of Z-scores (inverted for linearity since low is bad)
            # Higher composite score = more suspicious
            composite_score = (
                max(0, -lin_z) * 1.0 +  # Linearity: lower is worse, so invert
                max(0, smooth_z) * 1.5 +  # Smoothness: higher anomaly is worse
                max(0, inv_z) * 1.0  # Invariance: higher CV is worse
            )

            # Count how many probes flagged this layer
            flags = sum([lin_anomaly, smooth_spike, inv_neurotic])

            layer_scores[layer_id] = {
                'composite_score': composite_score,
                'flags': flags,
                'linearity_r2': linearity_results[layer_id]['linearity_r2'],
                'local_anomaly': smoothness_results[layer_id]['local_anomaly'],
                'cv': invariance_results[layer_id]['cv'],
            }

        # Sort by composite score
        sorted_layers = sorted(
            layer_scores.items(),
            key=lambda x: (x[1]['flags'], x[1]['composite_score']),
            reverse=True
        )

        print("\nTop 10 suspicious layers (by composite score):")
        print("-" * 70)
        print(f"{'Rank':<6}{'Layer':<8}{'Flags':<7}{'Score':<10}{'R²':<10}{'LocalAnom':<12}{'CV':<10}")
        print("-" * 70)

        for rank, (lid, scores) in enumerate(sorted_layers[:10], 1):
            marker = " <-- GT" if lid == ground_truth_layer else ""
            print(f"{rank:<6}{lid:<8}{scores['flags']:<7}{scores['composite_score']:<10.3f}"
                  f"{scores['linearity_r2']:<10.4f}{scores['local_anomaly']:<12.6f}"
                  f"{scores['cv']:<10.4f}{marker}")

        # Final diagnosis
        diagnosed_layer = sorted_layers[0][0]

        print(f"\n{'='*50}")
        print(f"DIAGNOSIS: Layer {diagnosed_layer} is the most likely error source")

        if ground_truth_layer is not None:
            top_5 = [l[0] for l in sorted_layers[:5]]
            if diagnosed_layer == ground_truth_layer:
                result = "EXACT_MATCH"
                print(f"Validation: ✅ EXACT MATCH")
            elif ground_truth_layer in top_5:
                rank = top_5.index(ground_truth_layer) + 1
                result = f"IN_TOP_5_RANK_{rank}"
                print(f"Validation: ⚠️ Ground truth in top 5 (rank {rank})")
            else:
                result = "MISMATCH"
                print(f"Validation: ❌ MISMATCH (GT layer {ground_truth_layer} not in top 5)")
        else:
            result = "NO_VALIDATION"

        print(f"{'='*50}")

        return {
            'diagnosed_layer': diagnosed_layer,
            'result': result,
            'layer_scores': layer_scores,
            'sorted_layers': sorted_layers,
            'linearity_results': linearity_results,
            'smoothness_results': smoothness_results,
            'invariance_results': invariance_results,
        }


def load_model(model_path: str, device: str = "cuda"):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


class HardwareFaultSimulator:
    """
    Simulates a hardware fault by adding a persistent error hook to a specific layer.

    This creates more realistic fault behavior:
    - Saturation: values get clamped to a range (simulates FP overflow)
    - Bias: systematic offset added (simulates computation error)
    - Noise: random perturbation (simulates numerical instability)
    """

    def __init__(self, model, fault_layer: int, fault_type: str = "saturation",
                 fault_magnitude: float = 0.1):
        self.model = model
        self.fault_layer = fault_layer
        self.fault_type = fault_type
        self.fault_magnitude = fault_magnitude
        self.hook_handle = None

        # Find the target layer
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.target_module = model.model.layers[fault_layer]
        else:
            raise ValueError(f"Cannot find layer {fault_layer} in model")

    def _fault_hook(self, module, input, output):
        """Hook that injects the fault into layer output."""
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        if self.fault_type == "saturation":
            # Clamp values to simulate FP overflow/saturation
            max_val = hidden_states.abs().max() * (1.0 - self.fault_magnitude)
            hidden_states = hidden_states.clamp(-max_val, max_val)

        elif self.fault_type == "bias":
            # Add systematic offset
            bias = self.fault_magnitude * hidden_states.std()
            hidden_states = hidden_states + bias

        elif self.fault_type == "noise":
            # Add random noise (different from AQN probe noise)
            noise = torch.randn_like(hidden_states) * hidden_states.abs() * self.fault_magnitude
            hidden_states = hidden_states + noise

        elif self.fault_type == "dead_zone":
            # Small values become zero (simulates underflow)
            threshold = hidden_states.abs().max() * self.fault_magnitude
            mask = hidden_states.abs() < threshold
            hidden_states = hidden_states.masked_fill(mask, 0.0)

        elif self.fault_type == "spike":
            # Random large values (simulates bit flip)
            spike_prob = self.fault_magnitude * 0.01  # 1% of magnitude
            spike_mask = torch.rand_like(hidden_states) < spike_prob
            spike_values = hidden_states.abs().max() * torch.randn_like(hidden_states) * 10
            hidden_states = torch.where(spike_mask, spike_values, hidden_states)

        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states

    def enable(self):
        """Enable the fault injection."""
        if self.hook_handle is None:
            self.hook_handle = self.target_module.register_forward_hook(self._fault_hook)
            print(f"[FAULT SIMULATOR] Enabled {self.fault_type} fault on layer {self.fault_layer}")

    def disable(self):
        """Disable the fault injection."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            print(f"[FAULT SIMULATOR] Disabled fault on layer {self.fault_layer}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SRDD Error Source Finder")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--ground_truth_layer", type=int, default=None,
                       help="Layer with simulated HW error (for validation)")
    parser.add_argument("--fault_type", type=str, default="saturation",
                       choices=["saturation", "bias", "noise", "dead_zone", "spike"],
                       help="Type of hardware fault to simulate")
    parser.add_argument("--fault_magnitude", type=float, default=0.1,
                       help="Magnitude of simulated fault (0.0-1.0)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.model_path, args.device)

    # Test prompts
    basic_prompts = [
        "What is 2 + 2? Answer:",
        "The capital of France is",
        "def fibonacci(n):",
        "Water boils at",
    ]

    # Diverse prompts for input invariance test
    diverse_prompts = [
        # Simple
        "Hello",
        "1 + 1 =",
        "The color of the sky is",
        # Medium
        "Explain photosynthesis in one sentence:",
        "What programming language is Python similar to?",
        "The largest ocean on Earth is",
        # Complex
        "Write a haiku about artificial intelligence:",
        "Explain the theory of relativity briefly:",
        "What are the implications of quantum computing for cryptography?",
        "Describe the process of neural network training in technical terms:",
    ]

    # Create finder
    finder = SRDDErrorFinder(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
    )

    # If ground_truth_layer is specified, simulate hardware error for validation
    fault_simulator = None
    if args.ground_truth_layer is not None:
        print(f"\n[VALIDATION MODE] Simulating {args.fault_type} fault on layer {args.ground_truth_layer}")
        print(f"Fault magnitude: {args.fault_magnitude}")

        # Create fault simulator - this adds a hook that corrupts the target layer's output
        fault_simulator = HardwareFaultSimulator(
            model=model,
            fault_layer=args.ground_truth_layer,
            fault_type=args.fault_type,
            fault_magnitude=args.fault_magnitude,
        )
        fault_simulator.enable()

    # Run full diagnosis
    try:
        results = finder.run_full_diagnosis(
            prompts=basic_prompts,
            diverse_prompts=diverse_prompts,
            ground_truth_layer=args.ground_truth_layer,
        )
    finally:
        # Clean up fault simulator
        if fault_simulator is not None:
            fault_simulator.disable()

    # Exit code based on results
    if args.ground_truth_layer is not None:
        success = results['result'] in ['EXACT_MATCH', 'IN_TOP_5_RANK_1', 'IN_TOP_5_RANK_2']
        sys.exit(0 if success else 1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
