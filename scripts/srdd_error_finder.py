#!/usr/bin/env python3
"""
Self-Referential Differential Diagnosis (SRDD) v4.0 - DC Bias Probe

This method finds hardware error sources WITHOUT a reference system (no GPU needed).
It uses controllable noise injection as a probe to detect ANOMALOUS layer behavior.

v4.0 New Feature (Gemini collaboration):
- DC Bias Probe for SATURATION detection
- Instead of random noise (variance), inject constant DC offset (bias)
- Normal layer: output mean shifts proportionally
- Saturated layer: output mean shifts LESS (hit the ceiling)

v3.3 Key Insight:
- Fault localization is an EDGE DETECTION problem
- Use LOCAL comparison (layer i vs i-1) instead of GLOBAL (layer i vs median)
- First derivative (jump) in metrics pinpoints the fault onset

v3.2 Fixes:
1. Independent RNG: Simulator noise uses separate Generator, not affected by global seed
2. Dynamic MAD Floor: Use max(median*0.01, 1e-4) to prevent Z-score explosion
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
    Self-Referential Differential Diagnosis v4.0 for finding hardware error sources.

    Works WITHOUT a reference system - uses edge detection and DC bias probing
    to find where fault behavior STARTS (the transition point).
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
        print(f"[SRDD v4.0] Registered {self.num_hooks} layer hooks")
        print(f"[SRDD v4.0] Model has {self.num_layers} layers")

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

    def probe_saturation_dc(
        self,
        prompts: List[str],
        baseline_logits: List[torch.Tensor],
        bias_scales: List[float] = [1.0, 5.0, 10.0, 20.0, 50.0],
    ) -> Dict[int, Dict]:
        """
        Probe C: DC Bias Test (v4.0 for Saturation Detection)

        Instead of random noise (variance), inject constant DC offset (bias).
        - Normal layer: output mean shifts proportionally to input bias
        - Saturated layer: output mean shifts LESS (hit the ceiling)

        Key insight: Saturation is a signal DAMPER, not corruptor.
        Random noise gets clipped but the effect averages out.
        DC bias creates a consistent shift that saturated layers absorb.
        """
        print(f"\n{'='*60}")
        print("PROBE C: DC BIAS (Saturation Detection)")
        print(f"{'='*60}")
        print(f"Bias scales: {bias_scales}")

        layer_dc_results = {}

        # Get target modules for hook injection
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            print("  Warning: Cannot find model layers for DC bias probe")
            return {}

        try:
            for layer_id in range(self.num_layers):
                bias_responses = []
                target_module = layers[layer_id]

                for bias_scale in bias_scales:
                    # Create DC bias hook
                    def dc_bias_hook(module, input, output, bias=bias_scale):
                        if isinstance(output, tuple):
                            hidden_states = output[0]
                            rest = output[1:]
                        else:
                            hidden_states = output
                            rest = None

                        # Add constant DC bias (scaled by activation magnitude for stability)
                        dc_offset = bias * hidden_states.abs().mean()
                        hidden_states = hidden_states + dc_offset

                        if rest is not None:
                            return (hidden_states,) + rest
                        return hidden_states

                    # Register hook
                    hook_handle = target_module.register_forward_hook(dc_bias_hook)

                    try:
                        # Measure output shift
                        prompt_shifts = []
                        for i, prompt in enumerate(prompts):
                            biased_logits = self.get_output_logits(prompt)
                            # Measure mean shift in logits
                            mean_shift = (biased_logits - baseline_logits[i]).abs().mean().item()
                            prompt_shifts.append(mean_shift)

                        bias_responses.append(np.median(prompt_shifts))
                    finally:
                        hook_handle.remove()

                # Analysis: Gain = (response at high bias) / (response at low bias)
                # Normal: high gain (output scales with input)
                # Saturated: low gain (output clamped, doesn't scale)
                resp_low = bias_responses[0]   # bias=1.0
                resp_high = bias_responses[-1]  # bias=50.0

                if resp_low > 1e-9:
                    dc_gain = resp_high / resp_low
                    # Expected gain for linear response: 50.0/1.0 = 50
                    # Saturated layer: gain << 50
                    saturation_ratio = dc_gain / (bias_scales[-1] / bias_scales[0])
                else:
                    dc_gain = 0.0
                    saturation_ratio = 0.0

                layer_dc_results[layer_id] = {
                    'responses': bias_responses,
                    'dc_gain': dc_gain,
                    'saturation_ratio': saturation_ratio,  # 1.0 = linear, <1.0 = saturated
                }

                if (layer_id + 1) % 7 == 0 or layer_id == 0:
                    print(f"  Layer {layer_id:2d}: dc_gain={dc_gain:.1f}, sat_ratio={saturation_ratio:.3f}")

        except Exception as e:
            print(f"  Error in DC bias probe: {e}")

        return layer_dc_results

    def diagnose(
        self,
        instability_results: Dict[int, float],
        dynamics_results: Dict[int, Dict],
        dc_results: Dict[int, Dict],
        ambient_noise: float,
        ground_truth_layer: int = None,
    ) -> Dict:
        """
        Aggregate results and diagnose fault type using EDGE DETECTION + DC BIAS.

        v4.0 NEW: DC Bias Probe for saturation detection
        v3.3 KEY INSIGHT: Find where the fault STARTS (the edge/transition),
        not just which layers look abnormal. Use first derivative (jump from
        previous layer) to detect the onset of fault behavior.
        """
        print(f"\n{'='*60}")
        print("DIAGNOSIS REPORT (v4.0 Edge Detection + DC Bias)")
        print(f"{'='*60}")

        # Collect metrics
        instabilities = []
        amp_factors = []
        monotonicities = []
        sensitivities_low = []
        saturation_ratios = []

        for lid in range(self.num_layers):
            # Subtract ambient noise from instability
            net_instability = max(0, instability_results[lid] - ambient_noise)
            instabilities.append(net_instability)
            amp_factors.append(dynamics_results[lid]['amp_factor'])
            monotonicities.append(dynamics_results[lid]['monotonicity'])
            sensitivities_low.append(dynamics_results[lid]['sensitivity_low'])
            # v4.0: DC saturation ratio (1.0 = linear, <1.0 = saturated)
            if dc_results and lid in dc_results:
                saturation_ratios.append(dc_results[lid]['saturation_ratio'])
            else:
                saturation_ratios.append(1.0)  # Default to linear

        instabilities = np.array(instabilities)
        amp_factors = np.array(amp_factors)
        monotonicities = np.array(monotonicities)
        sensitivities_low = np.array(sensitivities_low)
        saturation_ratios = np.array(saturation_ratios)

        # MAD-based Z-score (robust to outliers)
        def mad_zscore(values, x):
            """Calculate Z-score using Median Absolute Deviation."""
            med = np.median(values)
            abs_diff = np.abs(values - med)
            mad = np.median(abs_diff)
            min_mad = max(abs(med) * 0.01, 1e-4)
            effective_mad = max(mad, min_mad)
            return (x - med) / (effective_mad * 1.4826)

        def mad_zscore_array(values):
            """Calculate Z-scores for entire array."""
            return np.array([mad_zscore(values, v) for v in values])

        # ============================================================
        # v3.3 EDGE DETECTION: Calculate JUMPS (first derivative)
        # The fault ONSET creates a sharp transition in metrics
        # ============================================================

        # Log-transform for scale-invariant comparison (handles magnitude differences)
        log_mono = np.log(monotonicities + 1e-9)
        log_amp = np.log(amp_factors + 1e-4)
        log_sens = np.log(sensitivities_low + 1e-9)
        log_sat = np.log(saturation_ratios + 1e-9)  # v4.0: DC saturation

        # First derivative: jump from previous layer
        # Prepend first value so array length matches
        mono_jumps = np.diff(log_mono, prepend=log_mono[0])
        amp_jumps = np.diff(log_amp, prepend=log_amp[0])
        sens_jumps = np.diff(log_sens, prepend=log_sens[0])
        sat_jumps = np.diff(log_sat, prepend=log_sat[0])  # v4.0

        # Z-score the JUMPS to find anomalous transitions
        z_mono_jump = mad_zscore_array(mono_jumps)
        z_amp_jump = mad_zscore_array(amp_jumps)
        z_sens_jump = mad_zscore_array(sens_jumps)
        z_sat_jump = mad_zscore_array(sat_jumps)  # v4.0

        # Print statistics
        print(f"\nStatistics:")
        print(f"  Monotonicity: median={np.median(monotonicities):.4f}")
        print(f"  Amplification: median={np.median(amp_factors):.1f}")
        print(f"  DC Saturation Ratio: median={np.median(saturation_ratios):.4f}")
        print(f"  Mono Jump Z: max={np.max(np.abs(z_mono_jump)):.2f} at L{np.argmax(np.abs(z_mono_jump))}")
        print(f"  Sat Jump Z: max={np.max(np.abs(z_sat_jump)):.2f} at L{np.argmax(np.abs(z_sat_jump))}")

        # Classify each layer
        candidates = []

        for lid in range(self.num_layers):
            # Global Z-scores (for reference)
            z_inst = mad_zscore(instabilities, instabilities[lid])
            z_amp = mad_zscore(amp_factors, amp_factors[lid])
            z_mono = mad_zscore(monotonicities, monotonicities[lid])

            # v3.3: Local EDGE scores (jump from previous layer)
            edge_mono = abs(z_mono_jump[lid])  # Sudden drop in monotonicity
            edge_amp = abs(z_amp_jump[lid])    # Sudden change in amplification
            edge_sens = abs(z_sens_jump[lid])  # Sudden change in sensitivity

            score = 0
            reasons = []

            # v3.3 PRIMARY: Edge Detection (where fault STARTS)
            # A large ABSOLUTE jump in monotonicity indicates fault onset
            # (can be drop OR rise depending on fault type interaction)
            if abs(z_mono_jump[lid]) > 2.0:
                edge_score = abs(z_mono_jump[lid]) * 2.0
                score += edge_score
                reasons.append(f"EDGE(mono={z_mono_jump[lid]:.1f})")

            # Large jump in amplification (saturation onset)
            if abs(z_amp_jump[lid]) > 2.0:
                edge_score = abs(z_amp_jump[lid]) * 1.5
                score += edge_score
                reasons.append(f"EDGE(amp={z_amp_jump[lid]:.1f})")

            # v4.0 NEW: DC Saturation Edge Detection
            # A sudden DROP in saturation ratio indicates saturation onset
            if z_sat_jump[lid] < -2.0:  # Negative = DROP in linearity
                edge_score = abs(z_sat_jump[lid]) * 2.5  # High weight for DC probe
                score += edge_score
                reasons.append(f"DC_SAT(z={z_sat_jump[lid]:.1f})")

            # v3.2 SECONDARY: Global anomaly detection (backup)
            # 1. NOISE FAULT: High Instability
            if z_inst > 3.0:
                score += z_inst * 1.0  # Reduced weight vs edge detection
                reasons.append(f"UNSTABLE(z={z_inst:.1f})")

            # 2. SATURATION FAULT: Low DC Saturation Ratio (v4.0)
            z_sat = mad_zscore(saturation_ratios, saturation_ratios[lid])
            if z_sat < -2.0:  # Below median = more saturated
                score += abs(z_sat) * 1.0
                reasons.append(f"SAT_RATIO(z={z_sat:.1f})")

            # 3. DEAD ZONE: Very low sensitivity
            if sensitivities_low[lid] < 1e-5:
                score += 5.0
                reasons.append("DEAD")

            # Store results
            candidates.append({
                'layer': lid,
                'score': score,
                'reasons': reasons,
                'z_inst': z_inst,
                'z_amp': z_amp,
                'z_mono': z_mono,
                'z_mono_jump': z_mono_jump[lid],
                'z_amp_jump': z_amp_jump[lid],
                'z_sat_jump': z_sat_jump[lid],  # v4.0
                'instability': instabilities[lid],
                'amp_factor': amp_factors[lid],
                'monotonicity': monotonicities[lid],
                'saturation_ratio': saturation_ratios[lid],  # v4.0
            })

        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Print results with edge detection columns (v4.0: added SatJump)
        print(f"\n{'Layer':<6}{'Score':<8}{'MonoJump':<10}{'SatJump':<10}{'SatRatio':<10}{'Diagnosis'}")
        print("-" * 80)

        for c in candidates[:10]:
            marker = " <-- GT" if c['layer'] == ground_truth_layer else ""
            diagnosis = ", ".join(c['reasons']) if c['reasons'] else "Normal"
            print(f"{c['layer']:<6}{c['score']:<8.2f}{c['z_mono_jump']:<10.2f}{c['z_sat_jump']:<10.2f}{c['saturation_ratio']:<10.4f}{diagnosis}{marker}")

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
        Run complete v4.0 diagnosis pipeline with edge detection + DC bias probe.
        """
        print(f"\n{'='*70}")
        print("SELF-REFERENTIAL DIFFERENTIAL DIAGNOSIS (SRDD v4.0)")
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

        # Step 4: Run dynamics probe (for dead zone detection)
        dynamics_results = self.probe_dynamics(
            prompts=prompts,
            baseline_logits=baseline_logits,
            noise_scales=[0.05, 0.10, 0.20, 0.50, 1.00],
        )

        # Step 5: v4.0 NEW - Run DC bias probe (for saturation detection)
        dc_results = self.probe_saturation_dc(
            prompts=prompts,
            baseline_logits=baseline_logits,
            bias_scales=[1.0, 5.0, 10.0, 20.0, 50.0],
        )

        # Step 6: Diagnose
        results = self.diagnose(
            instability_results=instability_results,
            dynamics_results=dynamics_results,
            dc_results=dc_results,
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

    parser = argparse.ArgumentParser(description="SRDD Error Source Finder v4.0 (Edge Detection + DC Bias)")
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
