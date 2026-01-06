#!/usr/bin/env python3
"""
Self-Referential Differential Diagnosis (SRDD) v3.2 - Robust & Efficient

This method finds hardware error sources WITHOUT a reference system (no GPU needed).
It uses controllable noise injection as a probe to detect ANOMALOUS layer behavior.

v3.2 Fixes (Gemini collaboration):
1. Independent RNG: Simulator noise uses separate Generator, not affected by global seed
2. Dynamic MAD Floor: Use max(median*0.01, 1e-4) to prevent Z-score explosion

v3.1 Improvements:
1. Ambient Instability: Measure baseline noise to avoid "noisy baseline" trap
2. Trial Instability Probe: Detect Gaussian noise faults via non-determinism
3. High-Energy Concavity Probe: Detect saturation via extended noise scales [0.05-1.0]
4. MAD Statistics: Use Median Absolute Deviation for outlier-robust Z-scores
5. Decoupled Probing: Efficiency - instability (5 trials @ 1 scale) + dynamics (1 trial @ 5 scales)

Key Insight:
- Gaussian noise fault: Non-deterministic even with fixed probe seed
- Saturation fault: Amplification factor drops at high noise scales (concave curve)
- Dead zone fault: Non-monotonic response, low sensitivity
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
    Self-Referential Differential Diagnosis v3.2 for finding hardware error sources.

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
        print(f"[SRDD v3.2] Registered {self.num_hooks} layer hooks")
        print(f"[SRDD v3.2] Model has {self.num_layers} layers")

    def get_output_logits(self, prompt: str) -> torch.Tensor:
        """Get model output logits for a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use float32 immediately to preserve precision
            logits = outputs.logits[:, -1, :].float().clone()

        return logits

    def compute_kl_divergence(self, p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
        """Compute KL divergence safely in float32."""
        # p and q are already float32 from get_output_logits
        p = torch.softmax(p_logits, dim=-1)
        q = torch.softmax(q_logits, dim=-1)
        eps = 1e-6
        kl = torch.sum(p * torch.log((p + eps) / (q + eps)))
        return max(0.0, kl.item())

    def measure_ambient_instability(
        self,
        prompts: List[str],
        num_trials: int = 5,
    ) -> float:
        """
        Measure inherent hardware noise without any probes active.

        v3.1 IMPROVEMENT: If hardware is faulty, even 'disable_noisy_ops' outputs jitter.
        We measure this "ambient noise" to subtract from instability measurements.
        """
        print("\n[Calibration] Measuring Ambient Instability...")
        disable_noisy_ops()
        set_selective_layers(None)

        prompt_variances = []

        for prompt in prompts[:2]:  # Use first 2 prompts for calibration
            logits_list = []
            for _ in range(num_trials):
                logits_list.append(self.get_output_logits(prompt))

            # Calculate pairwise KL between first run and others
            ref_logits = logits_list[0]
            divergences = []
            for i in range(1, num_trials):
                div = self.compute_kl_divergence(ref_logits, logits_list[i])
                divergences.append(div)

            prompt_variances.append(np.mean(divergences))

        ambient_noise = np.mean(prompt_variances)
        print(f"  Ambient Noise Level (KL): {ambient_noise:.6f}")
        return ambient_noise

    def probe_instability(
        self,
        prompts: List[str],
        baseline_logits: List[torch.Tensor],
        noise_scale: float = 0.10,
        num_trials: int = 5,
    ) -> Dict[int, float]:
        """
        Probe A: Trial Instability Test (v3.1 for Gaussian Noise Detection)

        Detects NOISE faults where hardware adds random perturbations.
        - Normal layer: Fixed seed + fixed input = deterministic output (variance ~ 0)
        - Noisy layer: Fixed seed + fixed input = jittery output (variance > 0)
        """
        print(f"\n{'='*60}")
        print("PROBE A: TRIAL INSTABILITY (Gaussian Noise Detection)")
        print(f"{'='*60}")
        print(f"Scale: {noise_scale}, Trials: {num_trials}")

        layer_instability = {}

        try:
            for layer_id in range(self.num_layers):
                trial_divs = []

                for trial in range(num_trials):
                    set_selective_layers([layer_id])
                    enable_noisy_ops(error_scale=noise_scale, error_type='relative_gaussian')

                    # CRITICAL v3.1: Fix the seed for the PROBE
                    # Normal layers: same seed = identical output = 0 variance
                    # Noisy layers: hardware noise causes jitter despite fixed seed
                    torch.manual_seed(42 + layer_id * 1000 + trial)

                    prompt_divs = []
                    for i, prompt in enumerate(prompts):
                        noisy_logits = self.get_output_logits(prompt)
                        div = self.compute_kl_divergence(baseline_logits[i], noisy_logits)
                        prompt_divs.append(div)

                    trial_divs.append(np.mean(prompt_divs))
                    disable_noisy_ops()

                # Instability = std dev across trials (should be ~0 for normal layers)
                layer_instability[layer_id] = np.std(trial_divs)

                if (layer_id + 1) % 7 == 0 or layer_id == 0:
                    print(f"  Layer {layer_id:2d}: instability = {layer_instability[layer_id]:.6f}")

        finally:
            disable_noisy_ops()
            set_selective_layers(None)

        return layer_instability

    def probe_dynamics(
        self,
        prompts: List[str],
        baseline_logits: List[torch.Tensor],
        noise_scales: List[float] = [0.05, 0.10, 0.20, 0.50, 1.00],
    ) -> Dict[int, Dict]:
        """
        Probe B: Dynamics Test (v3.1 for Saturation and Dead Zone Detection)

        Extended noise scales [0.05 - 1.0] to hit saturation ceiling.
        - Normal layer: KL divergence grows rapidly with noise (high amplification)
        - Saturated layer: KL plateaus at high noise (low amplification, concave curve)
        - Dead zone layer: Non-monotonic or flat response
        """
        print(f"\n{'='*60}")
        print("PROBE B: DYNAMICS (Saturation & Dead Zone Detection)")
        print(f"{'='*60}")
        print(f"Scales: {noise_scales}")

        layer_dynamics = {}

        try:
            for layer_id in range(self.num_layers):
                scale_responses = []

                for scale in noise_scales:
                    set_selective_layers([layer_id])
                    enable_noisy_ops(error_scale=scale, error_type='relative_gaussian')
                    torch.manual_seed(42 + layer_id)  # Consistent seed for curve shape

                    prompt_divs = []
                    for i, prompt in enumerate(prompts):
                        noisy_logits = self.get_output_logits(prompt)
                        div = self.compute_kl_divergence(baseline_logits[i], noisy_logits)
                        prompt_divs.append(div)

                    # Use Median to be robust against single-prompt outliers
                    scale_responses.append(np.median(prompt_divs))
                    disable_noisy_ops()

                # Analysis
                # 1. Amplification Factor: response at 1.0 / response at 0.1
                resp_low = scale_responses[1]   # 0.10
                resp_high = scale_responses[-1]  # 1.00

                if resp_low > 1e-9:
                    amp_factor = resp_high / resp_low
                else:
                    amp_factor = 0.0  # Dead layer

                # 2. Monotonicity (Spearman)
                rho, _ = stats.spearmanr(noise_scales, scale_responses)
                monotonicity = rho ** 2 if not np.isnan(rho) else 0.0

                # 3. Log-space smoothness (for dead zone detection)
                log_resp = np.log(np.array(scale_responses) + 1e-9)
                # Simple measure: variance of increments (should be consistent)
                increments = np.diff(log_resp)
                increment_var = np.var(increments) if len(increments) > 1 else 0.0

                layer_dynamics[layer_id] = {
                    'responses': scale_responses,
                    'amp_factor': amp_factor,
                    'monotonicity': monotonicity,
                    'sensitivity_low': resp_low,
                    'sensitivity_high': resp_high,
                    'increment_var': increment_var,
                }

                if (layer_id + 1) % 7 == 0 or layer_id == 0:
                    print(f"  Layer {layer_id:2d}: amp={amp_factor:.1f}, mono={monotonicity:.3f}, sens_low={resp_low:.4f}")

        finally:
            disable_noisy_ops()
            set_selective_layers(None)

        return layer_dynamics

    def diagnose(
        self,
        instability_results: Dict[int, float],
        dynamics_results: Dict[int, Dict],
        ambient_noise: float,
        ground_truth_layer: int = None,
    ) -> Dict:
        """
        Aggregate results and diagnose fault type using MAD-based Z-scores.

        v3.1 IMPROVEMENT: Use Median Absolute Deviation (MAD) instead of StdDev
        for outlier-robust Z-score calculation.
        """
        print(f"\n{'='*60}")
        print("DIAGNOSIS REPORT (v3.2 MAD-based)")
        print(f"{'='*60}")

        # Collect metrics
        instabilities = []
        amp_factors = []
        monotonicities = []
        sensitivities_low = []

        for lid in range(self.num_layers):
            # Subtract ambient noise from instability
            net_instability = max(0, instability_results[lid] - ambient_noise)
            instabilities.append(net_instability)
            amp_factors.append(dynamics_results[lid]['amp_factor'])
            monotonicities.append(dynamics_results[lid]['monotonicity'])
            sensitivities_low.append(dynamics_results[lid]['sensitivity_low'])

        instabilities = np.array(instabilities)
        amp_factors = np.array(amp_factors)
        monotonicities = np.array(monotonicities)
        sensitivities_low = np.array(sensitivities_low)

        # MAD-based Z-score (robust to outliers)
        def mad_zscore(values, x):
            """Calculate Z-score using Median Absolute Deviation.

            v3.2 FIX: Use dynamic floor to prevent Z-score explosion
            when MAD approaches zero (all values nearly identical).
            """
            med = np.median(values)
            abs_diff = np.abs(values - med)
            mad = np.median(abs_diff)

            # v3.2 FIX: Dynamic floor based on median value
            # Prevents division by near-zero when all values are similar
            min_mad = max(abs(med) * 0.01, 1e-4)
            effective_mad = max(mad, min_mad)

            # 1.4826 scales MAD to Sigma for normal distribution
            return (x - med) / (effective_mad * 1.4826)

        # Print statistics
        print(f"\nStatistics:")
        print(f"  Instability: median={np.median(instabilities):.6f}, MAD={np.median(np.abs(instabilities - np.median(instabilities))):.6f}")
        print(f"  Amplification: median={np.median(amp_factors):.1f}, MAD={np.median(np.abs(amp_factors - np.median(amp_factors))):.1f}")

        # Classify each layer
        candidates = []

        for lid in range(self.num_layers):
            z_inst = mad_zscore(instabilities, instabilities[lid])
            z_amp = mad_zscore(amp_factors, amp_factors[lid])
            z_mono = mad_zscore(monotonicities, monotonicities[lid])

            score = 0
            reasons = []

            # 1. NOISE FAULT: High Instability (z > 3)
            if z_inst > 3.0:
                score += z_inst * 2.0  # High weight
                reasons.append(f"UNSTABLE(z={z_inst:.1f})")

            # 2. SATURATION FAULT: Low Amplification (z < -2) but not dead
            if z_amp < -2.0 and sensitivities_low[lid] > 1e-4:
                score += abs(z_amp) * 1.5
                reasons.append(f"SATURATED(z={z_amp:.1f})")

            # 3. DEAD ZONE FAULT: Low Monotonicity or very low sensitivity
            if z_mono < -2.0:
                score += abs(z_mono) * 1.5
                reasons.append(f"NON-MONO(z={z_mono:.1f})")

            if sensitivities_low[lid] < 1e-5:
                score += 5.0  # High penalty for dead layers
                reasons.append("DEAD")

            # Store results
            candidates.append({
                'layer': lid,
                'score': score,
                'reasons': reasons,
                'z_inst': z_inst,
                'z_amp': z_amp,
                'z_mono': z_mono,
                'instability': instabilities[lid],
                'amp_factor': amp_factors[lid],
                'monotonicity': monotonicities[lid],
            })

        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Print results
        print(f"\n{'Layer':<6}{'Score':<8}{'Z_Inst':<10}{'Z_Amp':<10}{'Z_Mono':<10}{'Diagnosis'}")
        print("-" * 70)

        for c in candidates[:10]:
            marker = " <-- GT" if c['layer'] == ground_truth_layer else ""
            diagnosis = ", ".join(c['reasons']) if c['reasons'] else "Normal"
            print(f"{c['layer']:<6}{c['score']:<8.2f}{c['z_inst']:<10.2f}{c['z_amp']:<10.2f}{c['z_mono']:<10.2f}{diagnosis}{marker}")

        # Final diagnosis
        top_suspect = candidates[0] if candidates[0]['score'] > 0 else None

        print(f"\n{'='*50}")
        if top_suspect:
            diagnosis = ", ".join(top_suspect['reasons'])
            print(f"DIAGNOSIS: Layer {top_suspect['layer']} - {diagnosis}")
        else:
            print("DIAGNOSIS: No significant faults detected")

        # Validation
        result = "NO_FAULT_DETECTED"
        if ground_truth_layer is not None and top_suspect:
            if top_suspect['layer'] == ground_truth_layer:
                result = "EXACT_MATCH"
                print(f"Validation: EXACT MATCH")
            else:
                top_5_layers = [c['layer'] for c in candidates[:5] if c['score'] > 0]
                if ground_truth_layer in top_5_layers:
                    rank = top_5_layers.index(ground_truth_layer) + 1
                    result = f"IN_TOP_5_RANK_{rank}"
                    print(f"Validation: Ground truth in top 5 (rank {rank})")
                else:
                    result = "MISMATCH"
                    print(f"Validation: MISMATCH (GT layer {ground_truth_layer} not in top 5)")

        print(f"{'='*50}")

        return {
            'diagnosed_layer': top_suspect['layer'] if top_suspect else None,
            'diagnosis': top_suspect['reasons'] if top_suspect else [],
            'result': result,
            'candidates': candidates,
        }

    def run_full_diagnosis(
        self,
        prompts: List[str],
        ground_truth_layer: int = None,
    ) -> Dict:
        """
        Run complete v3.2 diagnosis pipeline.
        """
        print(f"\n{'='*70}")
        print("SELF-REFERENTIAL DIFFERENTIAL DIAGNOSIS (SRDD v3.2)")
        print(f"{'='*70}")
        if ground_truth_layer is not None:
            print(f"Validation mode: Ground truth layer = {ground_truth_layer}")
        print(f"{'='*70}")

        # Step 1: Measure ambient instability (calibration)
        ambient_noise = self.measure_ambient_instability(prompts)

        # Step 2: Get baseline
        print("\n[Baseline] Capturing clean outputs...")
        disable_noisy_ops()
        set_selective_layers(None)
        baseline_logits = [self.get_output_logits(p) for p in prompts]

        # Step 3: Run instability probe (for noise detection)
        instability_results = self.probe_instability(
            prompts=prompts,
            baseline_logits=baseline_logits,
            noise_scale=0.10,
            num_trials=5,
        )

        # Step 4: Run dynamics probe (for saturation/dead zone detection)
        dynamics_results = self.probe_dynamics(
            prompts=prompts,
            baseline_logits=baseline_logits,
            noise_scales=[0.05, 0.10, 0.20, 0.50, 1.00],
        )

        # Step 5: Diagnose
        results = self.diagnose(
            instability_results=instability_results,
            dynamics_results=dynamics_results,
            ambient_noise=ambient_noise,
            ground_truth_layer=ground_truth_layer,
        )

        return results


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

    Supported fault types:
    - noise: Random Gaussian noise (non-deterministic)
    - saturation: Hard clamp values
    - dead_zone: Small values become zero
    - bias: Systematic offset
    - spike: Random large values
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

        # v3.2 FIX: Create independent Generator for "physical" randomness
        # This Generator is NOT affected by global torch.manual_seed()
        # Real hardware noise is independent of software seeds
        self.rng = torch.Generator(device=model.device)
        self.rng.seed()  # Seed from system entropy

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
            # v3.2 FIX: Use independent RNG for "physical" noise
            # Even when main program resets global seed, this noise stays random
            noise = torch.randn(
                hidden_states.size(),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                generator=self.rng,  # Independent of global seed
            ) * hidden_states.abs().mean() * self.fault_magnitude
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

    parser = argparse.ArgumentParser(description="SRDD Error Source Finder v3.2")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--ground_truth_layer", type=int, default=None,
                       help="Layer with simulated HW error (for validation)")
    parser.add_argument("--fault_type", type=str, default="dead_zone",
                       choices=["saturation", "bias", "noise", "dead_zone", "spike"],
                       help="Type of hardware fault to simulate")
    parser.add_argument("--fault_magnitude", type=float, default=0.3,
                       help="Magnitude of simulated fault (0.0-1.0)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.model_path, args.device)

    # Test prompts
    prompts = [
        "What is 2 + 2? Answer:",
        "The capital of France is",
        "def fibonacci(n):",
        "Water boils at",
        "Explain photosynthesis in one sentence:",
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
            prompts=prompts,
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
