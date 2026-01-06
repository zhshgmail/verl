#!/usr/bin/env python3
"""
Self-Referential Differential Diagnosis (SRDD) v6.0 - Sparse Saturation Detection

This method finds hardware error sources WITHOUT a reference system (no GPU needed).

v6.0 NEW: Min Gain Scan for Sparse Saturation (Gemini "Weakest Link" collaboration)
- Problem: Global stats (kurtosis) average out sparse faults
- Solution: Inject DC bias, measure MINIMUM element response
- Normal neuron: min_gain ≈ 1.0 (output shifts proportionally)
- Saturated neuron: min_gain ≈ 0.0 (output clamped, can't shift)
- Key advantage: Finds the "weakest link" regardless of healthy neighbors

v5.3: Sparse Fault Simulation (Gemini collaboration)
- Dense faults affect ALL tensor elements (default, sparsity=1.0)
- Sparse faults affect only a FRACTION of elements (e.g., sparsity=0.01 = 1%)
- Simulates local hardware faults (e.g., single bad core, bad memory bank)
- Use --sparsity flag to test detection limits

v5.2: Kurtosis Scan for Dense Saturation Detection (Gemini collaboration)
- Why: LLM activations are naturally 'spiky' (High Kurtosis >> 0)
- Saturation/clamping cuts off these spikes, causing massive Kurtosis DROP
- Key advantage: Passive measurement, bypasses LayerNorm invariance issues
- Limitation: Averages out sparse faults (use Min Gain scan instead)

v5.0.1 FIX: True Local Gain Measurement
- Use forward_pre_hook to perturb INPUT to layer, measure OUTPUT change
- This gives the true "local transfer function" (gain = output_change / input_noise)

v5.0 KEY BREAKTHROUGH (Gemini collaboration):
- Problem: E2E probing fails due to "propagation masking"
- Solution: LOCAL MEASUREMENT at the layer itself, not at final output

Three Local Scan Methods:
1. Instability Scan (Noise Detection): Run same input multiple times
   - Normal layer: output identical across trials
   - Noise fault: output differs (hardware adds random noise)

2. Gain Scan (Dead Zone Detection): Perturb input, measure output change
   - Normal layer: gain ≈ 1.0 (linear transfer)
   - Dead zone layer: gain ≈ 0.0 (signal lost)

3. Kurtosis Scan (Saturation Detection): Measure distribution shape
   - Normal layer: kurtosis >> 0 (spiky distribution)
   - Saturated layer: kurtosis ~ 0 (flattened, spikes clipped)

v3.2-v4.0 Fixes (retained):
1. Independent RNG for simulator noise
2. Dynamic MAD floor for numerical stability
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
    Self-Referential Differential Diagnosis v5.0 for finding hardware error sources.

    Works WITHOUT a reference system - uses LOCAL SCAN to measure layer behavior
    directly at the layer output, bypassing propagation masking.
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
        print(f"[SRDD v5.0] Registered {self.num_hooks} layer hooks")
        print(f"[SRDD v5.0] Model has {self.num_layers} layers")

        # Get layer modules for local scanning
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.layers = model.model.layers
        else:
            self.layers = None

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

    # ================================================================
    # v5.0 LOCAL SCAN METHODS - Bypass Propagation Masking
    # ================================================================

    def local_ambient_scan(
        self,
        prompts: List[str],
        num_trials: int = 3,
    ) -> Dict[int, float]:
        """
        v5.0.1 Probe: Ambient Stethoscope (Noise Fault Detection)

        Measures if a layer's OUTPUT CHANGES between trials with the SAME input.
        - Normal layer: output is identical across trials (deterministic)
        - Noise fault layer: output differs across trials (hardware adds random noise)

        Key fix (v5.0.1): Compare actual OUTPUT VALUES between trials,
        not just the variance of the output tensor.
        """
        print(f"\n{'='*60}")
        print("v5.0.1 LOCAL SCAN: AMBIENT (Noise Detection)")
        print(f"{'='*60}")

        if self.layers is None:
            print("  Error: Cannot access model layers")
            return {}

        layer_instabilities = {}
        prompt = prompts[0]  # Use single prompt for consistency

        for layer_id in range(self.num_layers):
            trial_outputs = []

            for trial in range(num_trials):
                # Hook to capture this layer's output
                captured = []

                def capture_hook(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0]
                    else:
                        out = output
                    # Store the actual output tensor
                    captured.append(out.detach().clone())

                hook = self.layers[layer_id].register_forward_hook(capture_hook)

                try:
                    # Run inference with same input
                    self.get_output_logits(prompt)
                finally:
                    hook.remove()

                if captured:
                    trial_outputs.append(captured[0])

            # Measure instability: how much does output change across trials?
            if len(trial_outputs) >= 2:
                # Compare consecutive trials
                diffs = []
                for i in range(1, len(trial_outputs)):
                    diff = trial_outputs[i] - trial_outputs[0]
                    diff_std = torch.std(diff.float()).item()
                    diffs.append(diff_std)
                instability = np.mean(diffs)
            else:
                instability = 0.0

            layer_instabilities[layer_id] = instability

            # Print every 5th layer or layer 10 specifically for debugging
            if (layer_id + 1) % 5 == 0 or layer_id == 0 or layer_id == 10:
                print(f"  Layer {layer_id:2d}: instability = {instability:.6f}")

        return layer_instabilities

    def local_gain_scan(
        self,
        prompts: List[str],
        noise_scale: float = 0.1,
    ) -> Dict[int, float]:
        """
        v5.0 Probe: Local Gain Analysis (Saturation/Dead Zone Detection)

        TRUE LOCAL MEASUREMENT: Perturb INPUT to layer, measure OUTPUT change.
        - Normal layer: output changes proportionally to input perturbation
        - Saturated layer: output barely changes (signal compressed at ceiling)
        - Dead zone layer: output barely changes (small values zeroed out)

        Key fix (v5.0.1): Use forward_pre_hook to add noise to INPUT,
        then measure OUTPUT difference. This is true "local gain".
        """
        print(f"\n{'='*60}")
        print("v5.0.1 LOCAL SCAN: GAIN (True Local Measurement)")
        print(f"{'='*60}")
        print(f"Input perturbation scale: {noise_scale}")

        if self.layers is None:
            print("  Error: Cannot access model layers")
            return {}

        layer_gains = {}

        for layer_id in range(self.num_layers):
            # Step 1: Get baseline output (no perturbation)
            baseline_outputs = []

            def capture_baseline(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                baseline_outputs.append(out.detach().clone())

            hook_base = self.layers[layer_id].register_forward_hook(capture_baseline)
            try:
                self.get_output_logits(prompts[0])
            finally:
                hook_base.remove()

            if not baseline_outputs:
                layer_gains[layer_id] = 0.0
                continue

            baseline_out = baseline_outputs[0]
            input_std_estimate = torch.std(baseline_out.float()).item()

            # Step 2: Add perturbation to INPUT and capture OUTPUT
            perturbed_outputs = []
            perturbation_magnitude = []

            def perturb_input(module, args):
                """Forward pre-hook: Add noise to the input of this layer."""
                if isinstance(args, tuple) and len(args) > 0:
                    hidden_states = args[0]
                    # Add Gaussian noise scaled to input magnitude
                    noise = torch.randn_like(hidden_states) * noise_scale * torch.std(hidden_states)
                    perturbation_magnitude.append(torch.std(noise.float()).item())
                    perturbed = hidden_states + noise
                    # Return modified args
                    return (perturbed,) + args[1:]
                return args

            def capture_perturbed(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                perturbed_outputs.append(out.detach().clone())

            hook_pre = self.layers[layer_id].register_forward_pre_hook(perturb_input)
            hook_post = self.layers[layer_id].register_forward_hook(capture_perturbed)

            try:
                self.get_output_logits(prompts[0])
            finally:
                hook_pre.remove()
                hook_post.remove()

            if not perturbed_outputs or not perturbation_magnitude:
                layer_gains[layer_id] = 0.0
                continue

            perturbed_out = perturbed_outputs[0]
            input_noise_std = perturbation_magnitude[0]

            # Step 3: Calculate gain = output_change / input_perturbation
            # This is the TRUE local transfer function
            output_diff = perturbed_out - baseline_out
            output_change_std = torch.std(output_diff.float()).item()

            if input_noise_std > 1e-9:
                gain = output_change_std / input_noise_std
            else:
                gain = 0.0

            layer_gains[layer_id] = gain

            # Print every 5th layer or layer 10/15 specifically for debugging
            if (layer_id + 1) % 5 == 0 or layer_id == 0 or layer_id in [10, 15]:
                print(f"  Layer {layer_id:2d}: in_noise={input_noise_std:.4f}, out_change={output_change_std:.4f}, gain={gain:.4f}")

        return layer_gains

    def local_kurtosis_scan(
        self,
        prompts: List[str],
    ) -> Dict[int, float]:
        """
        v5.2 Probe: Passive Kurtosis Scan (Saturation Detection)

        Why Kurtosis works for saturation:
        - LLM activations are naturally 'spiky' (High Kurtosis >> 0)
        - Saturation/Clamping cuts off these spikes
        - This causes massive Kurtosis DROP (from ~50 to ~1)

        Key advantage: Passive measurement, bypasses LayerNorm invariance issues.
        """
        print(f"\n{'='*60}")
        print("v5.2 LOCAL SCAN: KURTOSIS (Saturation Detection)")
        print(f"{'='*60}")

        if self.layers is None:
            print("  Error: Cannot access model layers")
            return {}

        from scipy.stats import kurtosis

        layer_kurtosis = {}
        prompt = prompts[0]  # Single prompt is enough

        for layer_id in range(self.num_layers):
            captured_outputs = []

            def capture_hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                captured_outputs.append(out.detach().float().cpu().numpy())

            hook = self.layers[layer_id].register_forward_hook(capture_hook)

            try:
                self.get_output_logits(prompt)
            finally:
                hook.remove()

            if captured_outputs:
                # Calculate Kurtosis of the flat array
                # Fisher=True means normal distribution has kurtosis=0
                # LLM activations are typically Super-Gaussian (>> 0)
                data = captured_outputs[0].flatten()
                k = kurtosis(data, fisher=True)
                layer_kurtosis[layer_id] = k

            if (layer_id + 1) % 5 == 0 or layer_id == 0 or layer_id in [10, 15]:
                print(f"  Layer {layer_id:2d}: kurtosis = {layer_kurtosis.get(layer_id, 0):.2f}")

        return layer_kurtosis

    def local_min_gain_scan(
        self,
        prompts: List[str],
        bias_magnitude: float = 50.0,
    ) -> Dict[int, float]:
        """
        v6.0 Probe: Sparse Saturation Detection (The 'Weakest Link' Scan)

        Why: Global stats (Kurtosis) average out sparse faults.
        If 1% of neurons are saturated, Mean/Kurtosis barely change.

        Solution: Inject massive bias and check the MINIMUM element response.
        - Normal neuron: Output shifts by ~bias_magnitude
        - Saturated neuron: Output shifts by < 1.0 (clamped)
        - Metric: min(Output_Shift) / Input_Bias

        This finds the "weakest link" in the tensor, regardless of how many
        healthy neurons surround it.
        """
        print(f"\n{'='*60}")
        print("v6.0 LOCAL SCAN: MINIMUM GAIN (Sparse Saturation)")
        print(f"{'='*60}")
        print(f"Bias magnitude: {bias_magnitude}")

        if self.layers is None:
            print("  Error: Cannot access model layers")
            return {}

        layer_min_gains = {}

        for layer_id in range(self.num_layers):
            # Step 1: Capture baseline output (no bias)
            baseline_outputs = []

            def capture_baseline(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                baseline_outputs.append(out.detach().clone())

            hook_base = self.layers[layer_id].register_forward_hook(capture_baseline)
            try:
                self.get_output_logits(prompts[0])
            finally:
                hook_base.remove()

            if not baseline_outputs:
                layer_min_gains[layer_id] = 1.0
                continue

            # Step 2: Inject DC bias to INPUT and capture OUTPUT
            biased_outputs = []

            def bias_input(module, args):
                """Forward pre-hook: Add constant DC bias to input."""
                if isinstance(args, tuple) and len(args) > 0:
                    hidden_states = args[0]
                    # Add large DC offset (constant, not random)
                    biased = hidden_states + bias_magnitude
                    return (biased,) + args[1:]
                return args

            def capture_biased(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                biased_outputs.append(out.detach().clone())

            hook_pre = self.layers[layer_id].register_forward_pre_hook(bias_input)
            hook_post = self.layers[layer_id].register_forward_hook(capture_biased)

            try:
                self.get_output_logits(prompts[0])
            finally:
                hook_pre.remove()
                hook_post.remove()

            if not biased_outputs:
                layer_min_gains[layer_id] = 1.0
                continue

            # Step 3: Calculate per-element gain
            # Shift = Biased - Baseline (expected to be ~bias_magnitude for healthy neurons)
            shift = biased_outputs[0] - baseline_outputs[0]

            # Normalize by the injected bias
            element_gains = shift / bias_magnitude

            # THE KEY: Look at BOTH low and high percentile gains
            # Saturation affects HIGH values (clips them), so saturated neurons
            # will have LOWER gain at high percentiles (output can't increase past ceiling)
            #
            # - min_gain (0.1%): affected by dead zone (small values zeroed)
            # - max_gain (99.9%): affected by saturation (large values clipped)
            flat_gains = element_gains.float().flatten()
            min_gain = torch.quantile(flat_gains, 0.001).item()
            max_gain = torch.quantile(flat_gains, 0.999).item()

            layer_min_gains[layer_id] = {
                'min': min_gain,
                'max': max_gain,
                'mean': flat_gains.mean().item(),
            }

            if (layer_id + 1) % 5 == 0 or layer_id == 0 or layer_id in [10, 15]:
                print(f"  Layer {layer_id:2d}: min_gain={min_gain:.4f}, max_gain={max_gain:.4f}")

        return layer_min_gains

    def local_discrete_scan(
        self,
        prompts: List[str],
        bias_magnitude: float = 50.0,
        threshold_ratio: float = 0.85,
    ) -> Dict[int, Dict]:
        """
        v7.0 Probe: Discrete Outlier Counting (Sparse Fault Detection)

        PARADIGM SHIFT: Stop measuring statistics, start counting failures.

        Why previous methods failed:
        - Kurtosis/percentiles aggregate data - sparse faults get averaged out
        - Looking at 99.9th percentile still selects a HEALTHY neuron

        v7.0 Method:
        1. Inject DC Bias (+50)
        2. Calculate per-neuron shift
        3. Determine expected shift (median of layer)
        4. COUNT neurons shifting < 85% of median (outliers)

        Key insight: Even 1 stuck neuron gives count=1, not gain=0.9999
        """
        print(f"\n{'='*60}")
        print("v7.0 LOCAL SCAN: DISCRETE OUTLIER COUNT")
        print(f"{'='*60}")
        print(f"Bias: {bias_magnitude}, Threshold ratio: {threshold_ratio}")

        if self.layers is None:
            print("  Error: Cannot access model layers")
            return {}

        layer_results = {}

        for layer_id in range(self.num_layers):
            # Step 1: Capture baseline output
            baseline_outputs = []

            def capture_baseline(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                baseline_outputs.append(out.detach().clone())

            hook_base = self.layers[layer_id].register_forward_hook(capture_baseline)
            try:
                self.get_output_logits(prompts[0])
            finally:
                hook_base.remove()

            if not baseline_outputs:
                layer_results[layer_id] = {'failure_rate': 0.0, 'failures': 0, 'median_shift': 0.0}
                continue

            # Step 2: Inject DC bias and capture output
            biased_outputs = []

            def bias_input(module, args):
                if isinstance(args, tuple) and len(args) > 0:
                    return (args[0] + bias_magnitude,) + args[1:]
                return args

            def capture_biased(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                biased_outputs.append(out.detach().clone())

            hook_pre = self.layers[layer_id].register_forward_pre_hook(bias_input)
            hook_post = self.layers[layer_id].register_forward_hook(capture_biased)

            try:
                self.get_output_logits(prompts[0])
            finally:
                hook_pre.remove()
                hook_post.remove()

            if not biased_outputs:
                layer_results[layer_id] = {'failure_rate': 0.0, 'failures': 0, 'median_shift': 0.0}
                continue

            # Step 3: Calculate per-neuron shift
            shift = biased_outputs[0] - baseline_outputs[0]
            flat_shift = shift.float().flatten()

            # Step 4: Determine expected shift (median - robust to outliers)
            median_shift = torch.median(flat_shift).item()

            # Skip unresponsive layers (dead or heavily normalized)
            if abs(median_shift) < bias_magnitude * 0.1:
                layer_results[layer_id] = {
                    'failure_rate': 0.0,
                    'failures': 0,
                    'median_shift': median_shift,
                    'note': 'unresponsive'
                }
                continue

            # Step 5: COUNT OUTLIERS
            # Neurons shifting significantly LESS than expected
            cutoff = median_shift * threshold_ratio
            failures = (flat_shift < cutoff).sum().item()
            total_neurons = flat_shift.numel()
            failure_rate = failures / total_neurons

            layer_results[layer_id] = {
                'failure_rate': failure_rate,
                'failures': int(failures),
                'total': total_neurons,
                'median_shift': median_shift,
                'cutoff': cutoff,
            }

            # Print layers with failures or periodic updates
            if failure_rate > 0.001 or (layer_id + 1) % 5 == 0 or layer_id == 0:
                marker = " <-- OUTLIERS!" if failure_rate > 0.001 else ""
                print(f"  Layer {layer_id:2d}: median_shift={median_shift:.1f}, failures={failures}/{total_neurons} ({failure_rate*100:.3f}%){marker}")

        return layer_results

    def local_histogram_scan(
        self,
        prompts: List[str],
        num_bins: int = 200,
    ) -> Dict[int, Dict]:
        """
        v8.0 Probe: Histogram Pile-up Detection (Sparse/Dense Saturation)

        CRITICAL FIX over Gemini's proposal:
        - Gemini looks at top 1% of MAX value (0.99 * max)
        - But saturation pile-up is at THRESHOLD, not MAX
        - With threshold = 30% of max, pile-up is at 0.3*max, not 0.99*max!

        v8.0 Method:
        1. Build full histogram over entire value range
        2. Find peaks by comparing each bin to its neighbors
        3. Look for unusual peaks in TAIL regions (not center)
        4. Saturation creates pile-up at threshold value (anywhere in tail)

        Key insight: Normal distribution has smooth exponential tail decay.
        Saturation creates a spike/pile-up at the clamp threshold.
        """
        print(f"\n{'='*60}")
        print("v8.0 LOCAL SCAN: HISTOGRAM PILE-UP DETECTION")
        print(f"{'='*60}")
        print(f"Scanning for unusual peaks in activation histograms...")

        if self.layers is None:
            print("  Error: Cannot access model layers")
            return {}

        layer_results = {}

        for layer_id in range(self.num_layers):
            # Capture activations
            outputs = []

            def capture_output(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                outputs.append(out.detach().float().cpu().numpy().flatten())

            hook = self.layers[layer_id].register_forward_hook(capture_output)
            try:
                self.get_output_logits(prompts[0])
            finally:
                hook.remove()

            if not outputs:
                layer_results[layer_id] = {'max_tail_peak_ratio': 0.0}
                continue

            data = outputs[0]

            # Handle edge cases
            if len(data) < 100 or np.std(data) < 1e-6:
                layer_results[layer_id] = {'max_tail_peak_ratio': 0.0}
                continue

            # Build histogram over full range
            hist, bin_edges = np.histogram(data, bins=num_bins)

            # Find peaks: bins with count >> neighbors (local anomaly)
            # Compare each bin to average of its 4 neighbors
            peak_ratio = np.zeros(num_bins)
            for i in range(2, num_bins - 2):
                local_avg = (hist[i-2] + hist[i-1] + hist[i+1] + hist[i+2]) / 4
                if local_avg > 10:  # Minimum neighbor count to avoid noise
                    peak_ratio[i] = hist[i] / local_avg

            # Focus on TAIL regions (saturation pile-up is in tails, not center)
            # Exclude center 50% where normal distribution peak is
            tail_boundary = num_bins // 4
            left_tail_idx = range(2, tail_boundary)
            right_tail_idx = range(num_bins - tail_boundary, num_bins - 2)

            left_tail_max = np.max(peak_ratio[2:tail_boundary]) if tail_boundary > 2 else 0
            right_tail_max = np.max(peak_ratio[num_bins - tail_boundary:num_bins - 2]) if tail_boundary > 2 else 0
            max_tail_peak = max(left_tail_max, right_tail_max)

            # Find location of max peak
            if max_tail_peak > 1.5:
                if left_tail_max >= right_tail_max:
                    peak_idx = np.argmax(peak_ratio[2:tail_boundary]) + 2
                else:
                    peak_idx = np.argmax(peak_ratio[num_bins - tail_boundary:num_bins - 2]) + (num_bins - tail_boundary)
                peak_value = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
                peak_count = hist[peak_idx]
            else:
                peak_value = 0
                peak_count = 0

            layer_results[layer_id] = {
                'max_tail_peak_ratio': max_tail_peak,
                'peak_value': peak_value,
                'peak_count': peak_count,
                'data_range': (np.min(data), np.max(data)),
            }

            # Print significant pile-ups
            if max_tail_peak > 2.0 or (layer_id + 1) % 7 == 0 or layer_id == 0:
                marker = " <-- PILE-UP!" if max_tail_peak > 2.0 else ""
                print(f"  Layer {layer_id:2d}: max_tail_peak_ratio={max_tail_peak:.2f}, peak_at={peak_value:.1f}{marker}")

        return layer_results

    def diagnose_local(
        self,
        ambient_results: Dict[int, float],
        gain_results: Dict[int, float],
        kurtosis_results: Dict[int, float] = None,
        min_gain_results: Dict[int, float] = None,
        discrete_results: Dict[int, Dict] = None,
        histogram_results: Dict[int, Dict] = None,
        ground_truth_layer: int = None,
    ) -> Dict:
        """
        v8.0 Diagnosis using Local Scan results.

        Six detection methods:
        - Noise: Edge detection on instability (first spike = source)
        - Dead zone: Local gain (gain << 1.0)
        - Dense saturation: Kurtosis drop (spiky distribution flattened)
        - Sparse saturation (v6.0): Min gain drop (weakest link blocked)
        - Sparse saturation (v7.0): Discrete outlier counting (failed neurons)
        - Sparse saturation (v8.0): Histogram pile-up at saturation threshold
        """
        print(f"\n{'='*60}")
        print("v8.0 LOCAL DIAGNOSIS")
        print(f"{'='*60}")

        # Convert to arrays
        instability_arr = np.array([ambient_results.get(i, 0) for i in range(self.num_layers)])
        gain_arr = np.array([gain_results.get(i, 0) for i in range(self.num_layers)])

        # MAD-based Z-score
        def mad_zscore(values):
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            min_mad = max(abs(med) * 0.01, 1e-6)
            effective_mad = max(mad, min_mad)
            return (values - med) / (effective_mad * 1.4826)

        z_instability = mad_zscore(instability_arr)
        z_gain = mad_zscore(gain_arr)

        # v5.2: Process kurtosis if available
        z_kurt = None
        kurt_arr = None
        z_kurt_drop = None
        first_saturation_layer = None
        if kurtosis_results:
            kurt_arr = np.array([kurtosis_results.get(i, 0) for i in range(self.num_layers)])
            # Log-space helps because kurtosis varies by orders of magnitude
            log_kurt = np.log(np.maximum(kurt_arr, 1e-4))
            z_kurt = mad_zscore(log_kurt)

            # EDGE DETECTION: Find where kurtosis DROPS (first derivative)
            # Saturation at layer i causes kurtosis DROP at layer i and propagates downstream
            kurt_diff = np.diff(log_kurt, prepend=log_kurt[0])
            z_kurt_drop = mad_zscore(kurt_diff)

            # Find FIRST layer with significant kurtosis DROP (excluding L0/L1 which are embedding layers)
            for lid in range(2, self.num_layers):
                if z_kurt_drop[lid] < -2.0:  # Negative = DROP
                    first_saturation_layer = lid
                    break

        # v6.0: Process min/max_gain if available (sparse fault detection)
        # IMPORTANT: Exclude boundary layers (L0, L1, L26, L27) due to natural anomalies
        # - min_gain: for dead zone detection (small values zeroed)
        # - max_gain: for saturation detection (large values clipped)
        min_gain_arr = None
        max_gain_arr = None
        z_min_gain = None
        z_max_gain = None
        first_sparse_sat_layer = None
        min_gain_valid_range = (2, self.num_layers - 2)  # Exclude first/last 2 layers
        z_min_gain_drop = None
        z_max_gain_drop = None

        if min_gain_results:
            # Extract min and max gains from dictionary
            min_gain_arr = np.array([min_gain_results.get(i, {'min': 1.0})['min'] if isinstance(min_gain_results.get(i), dict) else min_gain_results.get(i, 1.0) for i in range(self.num_layers)])
            max_gain_arr = np.array([min_gain_results.get(i, {'max': 1.0})['max'] if isinstance(min_gain_results.get(i), dict) else 1.0 for i in range(self.num_layers)])

            valid_start, valid_end = min_gain_valid_range

            # Z-scores for min_gain (dead zone detection)
            valid_min_gains = min_gain_arr[valid_start:valid_end]
            z_min_gain_valid = mad_zscore(valid_min_gains)
            z_min_gain = np.zeros(self.num_layers)
            z_min_gain[valid_start:valid_end] = z_min_gain_valid

            # Z-scores for max_gain (saturation detection) - LOW max_gain = saturation
            valid_max_gains = max_gain_arr[valid_start:valid_end]
            z_max_gain_valid = mad_zscore(valid_max_gains)
            z_max_gain = np.zeros(self.num_layers)
            z_max_gain[valid_start:valid_end] = z_max_gain_valid

            # EDGE DETECTION for max_gain: Find where max_gain DROPS (saturation)
            max_gain_diff = np.diff(max_gain_arr, prepend=max_gain_arr[0])
            valid_max_diffs = max_gain_diff[valid_start:valid_end]
            z_max_diff_valid = mad_zscore(valid_max_diffs)

            z_max_gain_drop = np.zeros(self.num_layers)
            z_max_gain_drop[valid_start:valid_end] = z_max_diff_valid

            # Find FIRST layer with significant max_gain DROP (sparse saturation)
            for i, lid in enumerate(range(valid_start, valid_end)):
                if z_max_diff_valid[i] < -2.0:  # Negative = DROP in max_gain
                    first_sparse_sat_layer = lid
                    break

            # Also compute min_gain edge for debugging
            min_gain_diff = np.diff(min_gain_arr, prepend=min_gain_arr[0])
            valid_min_diffs = min_gain_diff[valid_start:valid_end]
            z_min_diff_valid = mad_zscore(valid_min_diffs)
            z_min_gain_drop = np.zeros(self.num_layers)
            z_min_gain_drop[valid_start:valid_end] = z_min_diff_valid

        # v7.0: Process discrete counting results
        failure_rate_arr = None
        z_failure = None
        first_discrete_fault_layer = None
        discrete_valid_range = (2, self.num_layers - 2)
        if discrete_results:
            failure_rate_arr = np.array([
                discrete_results.get(i, {}).get('failure_rate', 0)
                for i in range(self.num_layers)
            ])

            # Z-score on valid range only (exclude boundary layers)
            valid_s, valid_e = discrete_valid_range
            valid_failures = failure_rate_arr[valid_s:valid_e]
            z_failure_valid = mad_zscore(valid_failures)
            z_failure = np.zeros(self.num_layers)
            z_failure[valid_s:valid_e] = z_failure_valid

            # Find FIRST layer with significant failure rate SPIKE
            # Using edge detection on failure_rate
            failure_diff = np.diff(failure_rate_arr, prepend=0)
            valid_diffs = failure_diff[valid_s:valid_e]
            z_diff_valid = mad_zscore(valid_diffs)

            for i, lid in enumerate(range(valid_s, valid_e)):
                # A spike in failure rate indicates fault source
                if z_diff_valid[i] > 2.0 and failure_rate_arr[lid] > 0.001:  # >0.1% failures
                    first_discrete_fault_layer = lid
                    break

        # v8.0: Process histogram pile-up results
        pileup_arr = None
        z_pileup = None
        first_pileup_layer = None
        histogram_valid_range = (2, self.num_layers - 2)
        if histogram_results:
            pileup_arr = np.array([
                histogram_results.get(i, {}).get('max_tail_peak_ratio', 0)
                for i in range(self.num_layers)
            ])

            # Z-score on valid range only (exclude boundary layers)
            valid_s, valid_e = histogram_valid_range
            valid_pileup = pileup_arr[valid_s:valid_e]
            z_pileup_valid = mad_zscore(valid_pileup)
            z_pileup = np.zeros(self.num_layers)
            z_pileup[valid_s:valid_e] = z_pileup_valid

            # Find FIRST layer with significant pile-up SPIKE
            pileup_diff = np.diff(pileup_arr, prepend=0)
            valid_diffs = pileup_diff[valid_s:valid_e]
            z_diff_valid = mad_zscore(valid_diffs)

            for i, lid in enumerate(range(valid_s, valid_e)):
                # A spike in pile-up ratio indicates saturation source
                if z_diff_valid[i] > 2.0 and pileup_arr[lid] > 2.0:  # Ratio > 2 = significant
                    first_pileup_layer = lid
                    break

        # Print statistics
        print(f"\nStatistics:")
        print(f"  Instability: max={np.max(instability_arr):.6f} at L{np.argmax(instability_arr)}")
        print(f"  Gain: min={np.min(gain_arr):.4f} at L{np.argmin(gain_arr)}")
        if max_gain_arr is not None:
            valid_s, valid_e = min_gain_valid_range
            valid_max = max_gain_arr[valid_s:valid_e]
            print(f"  Max Gain: min={np.min(valid_max):.4f} at L{np.argmin(valid_max)+valid_s}, median={np.median(valid_max):.4f}")
            if first_sparse_sat_layer is not None:
                print(f"  MAX_GAIN EDGE: Layer {first_sparse_sat_layer} drop_z={z_max_gain_drop[first_sparse_sat_layer]:.2f}")
        if kurt_arr is not None:
            print(f"  Kurtosis: min={np.min(kurt_arr[2:]):.2f} at L{np.argmin(kurt_arr[2:])+2}, median={np.median(kurt_arr[2:]):.2f}")
            if first_saturation_layer is not None:
                print(f"  KURTOSIS EDGE: Layer {first_saturation_layer} drop_z={z_kurt_drop[first_saturation_layer]:.2f}")
        if failure_rate_arr is not None:
            valid_s, valid_e = discrete_valid_range
            valid_fail = failure_rate_arr[valid_s:valid_e]
            max_fail_idx = np.argmax(valid_fail) + valid_s
            print(f"  Failure Rate: max={np.max(valid_fail)*100:.3f}% at L{max_fail_idx}")
            if first_discrete_fault_layer is not None:
                print(f"  DISCRETE EDGE: Layer {first_discrete_fault_layer} rate={failure_rate_arr[first_discrete_fault_layer]*100:.3f}%")
        if pileup_arr is not None:
            valid_s, valid_e = histogram_valid_range
            valid_pileup = pileup_arr[valid_s:valid_e]
            max_pileup_idx = np.argmax(valid_pileup) + valid_s
            print(f"  Pile-up Ratio: max={np.max(valid_pileup):.2f} at L{max_pileup_idx}")
            if first_pileup_layer is not None:
                print(f"  PILEUP EDGE: Layer {first_pileup_layer} ratio={pileup_arr[first_pileup_layer]:.2f}")

        # EDGE DETECTION: Find where instability JUMPS (first derivative)
        # Layers before fault have instability ≈ 0
        # Fault layer has sudden spike in instability
        instability_jumps = np.diff(instability_arr, prepend=0)
        z_inst_jump = mad_zscore(instability_jumps)

        # Find first layer with significant instability JUMP (edge detection)
        first_noisy_layer = None
        for lid in range(self.num_layers):
            if z_inst_jump[lid] > 2.0 and instability_arr[lid] > 0.1:
                first_noisy_layer = lid
                print(f"  NOISE EDGE: Layer {lid} jump_z={z_inst_jump[lid]:.2f}, instab={instability_arr[lid]:.4f}")
                break

        # Find layer with lowest gain (saturation/dead zone source)
        min_gain_layer = int(np.argmin(gain_arr))

        # Score each layer
        candidates = []
        for lid in range(self.num_layers):
            score = 0
            reasons = []

            # FIRST layer with high instability = noise source
            if lid == first_noisy_layer:
                score += 100.0  # High score for first noisy layer
                reasons.append(f"NOISE_SOURCE(z={z_instability[lid]:.1f})")
            elif z_instability[lid] > 2.0:
                # Secondary: downstream noise propagation
                score += z_instability[lid] * 0.5  # Lower score for downstream
                reasons.append(f"NOISE_PROP(z={z_instability[lid]:.1f})")

            # Low gain = dead zone (signal lost)
            if z_gain[lid] < -2.0:
                score += abs(z_gain[lid]) * 2.0
                reasons.append(f"DEAD_ZONE(z={z_gain[lid]:.1f})")

            # v5.2: Kurtosis DROP = dense saturation (spikes clipped)
            # Only score kurtosis if:
            # 1. No noise fault found (first_noisy_layer is None)
            # 2. No dead zone fault found (min gain z > -2)
            # 3. Within valid range (exclude boundary layers L0/L1 and L26/L27)
            # This prevents kurtosis from interfering with primary detection methods
            has_dead_zone = np.min(z_gain) < -2.0
            has_noise = first_noisy_layer is not None
            kurt_valid_range = (2, self.num_layers - 2)  # Exclude first/last 2 layers

            in_kurt_valid = kurt_valid_range[0] <= lid < kurt_valid_range[1]
            if not has_dead_zone and not has_noise and z_kurt_drop is not None and in_kurt_valid:
                if lid == first_saturation_layer:
                    score += 100.0  # High score for first saturation layer
                    reasons.append(f"SAT_SOURCE(drop_z={z_kurt_drop[lid]:.1f})")
                elif z_kurt_drop[lid] < -5.0:  # Higher threshold for secondary
                    score += abs(z_kurt_drop[lid]) * 0.3
                    reasons.append(f"SAT_PROP(drop_z={z_kurt_drop[lid]:.1f})")

            # v6.0: Max Gain DROP = sparse saturation (high values clipped)
            # This detects sparse faults that kurtosis misses
            # Saturation clips HIGH values, so max_gain drops at saturated layers
            # Only apply to valid range (exclude boundary layers with natural anomalies)
            in_valid_range = min_gain_valid_range[0] <= lid < min_gain_valid_range[1]
            if not has_dead_zone and not has_noise and z_max_gain is not None and in_valid_range:
                if lid == first_sparse_sat_layer:
                    score += 100.0  # High score for first sparse saturation layer
                    reasons.append(f"SPARSE_SAT(max_g={max_gain_arr[lid]:.3f})")
                elif z_max_gain[lid] < -3.0:  # Low max gain = high values clipped
                    score += abs(z_max_gain[lid]) * 1.5
                    reasons.append(f"SPARSE_SAT_PROP(z={z_max_gain[lid]:.1f})")

            # v7.0: Discrete Outlier Counting = sparse saturation (failed neurons)
            # COUNT neurons failing to shift - works even with sparse faults
            # Uses same exclusion rules: no dead zone, no noise, valid range
            in_discrete_valid = discrete_valid_range[0] <= lid < discrete_valid_range[1]
            if not has_dead_zone and not has_noise and failure_rate_arr is not None and in_discrete_valid:
                if lid == first_discrete_fault_layer:
                    score += 100.0  # High score for first layer with significant failures
                    reasons.append(f"DISCRETE_SAT(rate={failure_rate_arr[lid]*100:.2f}%)")
                elif failure_rate_arr[lid] > 0.01:  # >1% failure rate (secondary)
                    score += failure_rate_arr[lid] * 500  # Scale for scoring
                    reasons.append(f"DISCRETE_PROP(rate={failure_rate_arr[lid]*100:.2f}%)")

            # v8.0: Histogram Pile-up = sparse/dense saturation (values stuck at ceiling)
            # Looks for unusual peaks in activation histogram at saturation threshold
            # This is the most direct physical evidence of saturation
            in_histogram_valid = histogram_valid_range[0] <= lid < histogram_valid_range[1]
            if not has_dead_zone and not has_noise and pileup_arr is not None and in_histogram_valid:
                if lid == first_pileup_layer:
                    score += 100.0  # High score for first layer with pile-up
                    reasons.append(f"PILEUP_SAT(ratio={pileup_arr[lid]:.2f})")
                elif pileup_arr[lid] > 3.0:  # Significant pile-up (secondary)
                    score += pileup_arr[lid] * 10  # Scale for scoring
                    reasons.append(f"PILEUP_PROP(ratio={pileup_arr[lid]:.2f})")

            candidates.append({
                'layer': lid,
                'score': score,
                'reasons': reasons,
                'instability': instability_arr[lid],
                'gain': gain_arr[lid],
                'max_gain': max_gain_arr[lid] if max_gain_arr is not None else 1.0,
                'kurtosis': kurt_arr[lid] if kurt_arr is not None else 0,
                'failure_rate': failure_rate_arr[lid] if failure_rate_arr is not None else 0,
                'pileup_ratio': pileup_arr[lid] if pileup_arr is not None else 0,
                'z_instability': z_instability[lid],
                'z_gain': z_gain[lid],
                'z_max_gain': z_max_gain[lid] if z_max_gain is not None else 0,
                'z_kurt': z_kurt[lid] if z_kurt is not None else 0,
            })

        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Print results
        if max_gain_arr is not None:
            print(f"\n{'Layer':<6}{'Score':<8}{'Instab':<10}{'Gain':<8}{'MaxGain':<10}{'Diagnosis'}")
            print("-" * 75)
        elif kurt_arr is not None:
            print(f"\n{'Layer':<6}{'Score':<8}{'Instab':<10}{'Gain':<8}{'Kurt':<10}{'Diagnosis'}")
            print("-" * 70)
        else:
            print(f"\n{'Layer':<6}{'Score':<8}{'Instab':<12}{'Gain':<10}{'Diagnosis'}")
            print("-" * 60)

        for c in candidates[:10]:
            marker = " <-- GT" if c['layer'] == ground_truth_layer else ""
            diagnosis = ", ".join(c['reasons']) if c['reasons'] else "Normal"
            if max_gain_arr is not None:
                print(f"{c['layer']:<6}{c['score']:<8.2f}{c['instability']:<10.4f}{c['gain']:<8.4f}{c['max_gain']:<10.4f}{diagnosis}{marker}")
            elif kurt_arr is not None:
                print(f"{c['layer']:<6}{c['score']:<8.2f}{c['instability']:<10.4f}{c['gain']:<8.4f}{c['kurtosis']:<10.2f}{diagnosis}{marker}")
            else:
                print(f"{c['layer']:<6}{c['score']:<8.2f}{c['instability']:<12.6f}{c['gain']:<10.4f}{diagnosis}{marker}")

        # Final diagnosis
        top = candidates[0] if candidates[0]['score'] > 0 else None

        print(f"\n{'='*50}")
        if top:
            print(f"DIAGNOSIS: Layer {top['layer']} - {', '.join(top['reasons'])}")
        else:
            print("DIAGNOSIS: No significant faults detected")

        # Validation
        result = "NO_FAULT_DETECTED"
        if ground_truth_layer is not None and top:
            if top['layer'] == ground_truth_layer:
                result = "EXACT_MATCH"
                print("Validation: EXACT MATCH")
            elif abs(top['layer'] - ground_truth_layer) == 1:
                result = "OFF_BY_ONE"
                print(f"Validation: Off by 1 (GT={ground_truth_layer})")
            else:
                top_5 = [c['layer'] for c in candidates[:5] if c['score'] > 0]
                if ground_truth_layer in top_5:
                    rank = top_5.index(ground_truth_layer) + 1
                    result = f"IN_TOP_5_RANK_{rank}"
                    print(f"Validation: In top 5 (rank {rank})")
                else:
                    result = "MISMATCH"
                    print(f"Validation: MISMATCH (GT={ground_truth_layer})")

        print(f"{'='*50}")

        return {
            'diagnosed_layer': top['layer'] if top else None,
            'diagnosis': top['reasons'] if top else [],
            'result': result,
            'candidates': candidates,
        }

    def run_local_scan(
        self,
        prompts: List[str],
        ground_truth_layer: int = None,
    ) -> Dict:
        """
        Run v8.0 Local Scan diagnosis pipeline.

        Six detection methods:
        - Instability Scan: Noise detection (output changes between trials)
        - Gain Scan: Dead zone detection (gain << 1.0)
        - Kurtosis Scan: Dense saturation detection (spiky distribution flattened)
        - Min Gain Scan (v6.0): Sparse saturation detection (weakest link blocked)
        - Discrete Scan (v7.0): Sparse saturation via outlier counting
        - Histogram Scan (v8.0): Sparse saturation via pile-up detection
        """
        print(f"\n{'='*70}")
        print("SRDD v8.0 LOCAL SCAN DIAGNOSIS")
        print(f"{'='*70}")
        if ground_truth_layer is not None:
            print(f"Validation mode: Ground truth layer = {ground_truth_layer}")
        print(f"{'='*70}")

        # Step 1: Ambient scan (noise detection)
        ambient_results = self.local_ambient_scan(prompts, num_trials=3)

        # Step 2: Gain scan (dead zone detection)
        gain_results = self.local_gain_scan(prompts, noise_scale=0.1)

        # Step 3: Kurtosis scan (dense saturation detection) - v5.2
        kurtosis_results = self.local_kurtosis_scan(prompts)

        # Step 4: Min Gain scan (sparse saturation detection) - v6.0
        min_gain_results = self.local_min_gain_scan(prompts, bias_magnitude=50.0)

        # Step 5: Discrete scan (sparse saturation via outlier counting) - v7.0
        discrete_results = self.local_discrete_scan(prompts, bias_magnitude=50.0)

        # Step 6: Histogram scan (sparse saturation via pile-up detection) - v8.0
        histogram_results = self.local_histogram_scan(prompts)

        # Step 7: Diagnose with all six methods
        results = self.diagnose_local(
            ambient_results=ambient_results,
            gain_results=gain_results,
            kurtosis_results=kurtosis_results,
            min_gain_results=min_gain_results,
            discrete_results=discrete_results,
            histogram_results=histogram_results,
            ground_truth_layer=ground_truth_layer,
        )

        return results

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

    v5.3 NEW: Sparsity parameter for simulating local/sparse faults.
    - sparsity=1.0: Dense fault (affects ALL elements) - default
    - sparsity=0.01: Sparse fault (affects only 1% of elements)

    v6.0 FIX: Fixed sparse mask (deterministic)
    - Real hardware faults are deterministic (same neurons always affected)
    - The sparse mask is generated once and reused across all forward passes
    - This prevents false "noise" detection from changing sparse patterns
    """

    def __init__(self, model, fault_layer: int, fault_type: str = "saturation",
                 fault_magnitude: float = 0.1, sparsity: float = 1.0,
                 absolute_saturation: bool = False):
        self.model = model
        self.fault_layer = fault_layer
        self.fault_type = fault_type
        self.fault_magnitude = fault_magnitude
        self.sparsity = sparsity  # v5.3: Fraction of elements affected (1.0 = all)
        self.absolute_saturation = absolute_saturation  # v7.1: Use fixed threshold
        self.hook_handle = None

        # v6.0: Fixed sparse mask (created lazily on first forward pass)
        # This ensures the SAME neurons are always affected (deterministic fault)
        self._fixed_sparse_mask = None

        # v7.1: Absolute saturation threshold (computed once, cached)
        # Tests hypothesis: dynamic threshold defeats detection methods
        self._absolute_threshold = None

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

        # v6.0: Use FIXED sparsity mask for deterministic sparse faults
        # Real hardware faults affect the SAME neurons every time
        if self.sparsity < 1.0:
            # Create fixed mask on first forward pass (lazy initialization)
            # Use only the last dimension (hidden_dim) for the mask - this simulates
            # "certain neurons are always faulty" regardless of batch/sequence
            hidden_dim = hidden_states.shape[-1]
            if self._fixed_sparse_mask is None or self._fixed_sparse_mask.shape[0] != hidden_dim:
                self._fixed_sparse_mask = torch.rand(
                    hidden_dim,
                    device=hidden_states.device,
                    generator=self.rng,
                ) < self.sparsity
                num_faulty = self._fixed_sparse_mask.sum().item()
                print(f"  [SPARSE FAULT] Created fixed mask: {num_faulty}/{hidden_dim} neurons ({100*num_faulty/hidden_dim:.1f}%) affected")

            # Broadcast mask to match hidden_states shape
            sparse_mask = self._fixed_sparse_mask.expand_as(hidden_states)
        else:
            sparse_mask = None  # Dense fault - affect all elements

        # Store original for sparse blending
        original_states = hidden_states.clone() if sparse_mask is not None else None

        if self.fault_type == "saturation":
            # Clamp values to simulate FP overflow/saturation
            if self.absolute_saturation:
                # v7.1: Absolute saturation - compute threshold ONCE and cache
                # This tests hypothesis: dynamic threshold defeats detection
                if self._absolute_threshold is None:
                    self._absolute_threshold = hidden_states.abs().max().item() * (1.0 - self.fault_magnitude)
                    print(f"  [ABSOLUTE SAT] Fixed threshold: {self._absolute_threshold:.4f}")
                max_val = self._absolute_threshold
            else:
                # Default: Dynamic threshold (70% of current max)
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
            fault_mask = hidden_states.abs() < threshold
            hidden_states = hidden_states.masked_fill(fault_mask, 0.0)

        elif self.fault_type == "spike":
            # Random large values (simulates bit flip)
            spike_prob = self.fault_magnitude * 0.01  # 1% of magnitude
            spike_mask = torch.rand_like(hidden_states) < spike_prob
            spike_values = hidden_states.abs().max() * torch.randn_like(hidden_states) * 10
            hidden_states = torch.where(spike_mask, spike_values, hidden_states)

        # v5.3: Apply sparsity - blend faulty and original
        if sparse_mask is not None:
            hidden_states = torch.where(sparse_mask, hidden_states, original_states)

        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states

    def enable(self):
        """Enable the fault injection."""
        if self.hook_handle is None:
            self.hook_handle = self.target_module.register_forward_hook(self._fault_hook)
            sparsity_str = f" (sparsity={self.sparsity*100:.1f}%)" if self.sparsity < 1.0 else ""
            print(f"[FAULT SIMULATOR] Enabled {self.fault_type} fault on layer {self.fault_layer}{sparsity_str}")

    def disable(self):
        """Disable the fault injection."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            print(f"[FAULT SIMULATOR] Disabled fault on layer {self.fault_layer}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SRDD Error Source Finder v6.0 (Sparse Saturation Detection)")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--ground_truth_layer", type=int, default=None,
                       help="Layer with simulated HW error (for validation)")
    parser.add_argument("--fault_type", type=str, default="dead_zone",
                       choices=["saturation", "bias", "noise", "dead_zone", "spike"],
                       help="Type of hardware fault to simulate")
    parser.add_argument("--fault_magnitude", type=float, default=0.3,
                       help="Magnitude of simulated fault (0.0-1.0)")
    parser.add_argument("--sparsity", type=float, default=1.0,
                       help="v5.3: Fraction of elements affected by fault (1.0=dense, 0.01=1%% sparse)")
    parser.add_argument("--absolute_saturation", action="store_true",
                       help="v7.1: Use fixed saturation threshold (tests dynamic threshold hypothesis)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--local-scan", action="store_true",
                       help="Use v5.0 Local Scan instead of v4.0 E2E probing")
    parser.add_argument("--method", type=str, default="auto",
                       choices=["auto", "local", "e2e"],
                       help="Diagnosis method: local (v5.0), e2e (v4.0), or auto")

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
        if args.sparsity < 1.0:
            print(f"Sparsity: {args.sparsity*100:.1f}% of elements affected (SPARSE FAULT)")
        if args.absolute_saturation:
            print(f"[v7.1] Using ABSOLUTE saturation threshold (tests dynamic threshold hypothesis)")

        fault_simulator = HardwareFaultSimulator(
            model=model,
            fault_layer=args.ground_truth_layer,
            fault_type=args.fault_type,
            fault_magnitude=args.fault_magnitude,
            sparsity=args.sparsity,
            absolute_saturation=args.absolute_saturation,
        )
        fault_simulator.enable()

    # Determine which method to use
    use_local = getattr(args, 'local_scan', False) or args.method == "local"
    if args.method == "auto":
        # Default to local scan in v5.0
        use_local = True

    # Run diagnosis
    try:
        if use_local:
            print("\n[Using v5.0 LOCAL SCAN method]")
            results = finder.run_local_scan(
                prompts=prompts,
                ground_truth_layer=args.ground_truth_layer,
            )
        else:
            print("\n[Using v4.0 E2E method]")
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
