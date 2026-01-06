#!/usr/bin/env python3
"""
Error Source Finder: Identify which layer has persistent hardware error.

This script implements the "Differential Sensitivity" methodology:

Theory:
- If layer X has persistent HW error (noise), adding MORE diagnostic noise to X
  should show LESS additional degradation than adding noise to clean layers
- This is because layer X is already degraded, so relative impact is smaller

Algorithm:
1. Simulate HW error: Enable persistent noise on ground truth layer
2. Get "degraded baseline": Run inference with HW error active
3. For each candidate layer:
   - Keep HW error ON (persistent)
   - Add diagnostic noise to candidate layer
   - Measure ADDITIONAL divergence from degraded baseline
4. The layer with LEAST additional divergence is the error source

Success criteria:
- The diagnosed layer should match or be within 1-2 of ground truth
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from verl.utils.noisy_ops import (
    enable_noisy_ops,
    disable_noisy_ops,
    set_selective_layers,
    set_selective_operators,
    register_layer_hooks,
    get_layer_injection_stats,
    reset_layer_injection_stats,
    reset_injection_stats,
)


class ErrorSourceFinder:
    """Find which layer has persistent hardware error.

    Two methods available:
    1. Differential Sensitivity: Layer with LEAST additional divergence is error source
       (already noisy, so adding more noise has less relative impact)
    2. Fingerprint Correlation: Layer whose noise pattern MATCHES the failure pattern
       (correct layer produces similar output deviation as HW error)
    """

    def __init__(
        self,
        model,
        tokenizer,
        hw_error_scale: float = 0.05,
        diagnostic_scale: float = 0.05,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hw_error_scale = hw_error_scale
        self.diagnostic_scale = diagnostic_scale
        self.device = device

        # Get number of layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.num_layers = len(model.model.layers)
        else:
            self.num_layers = 28  # Default for Qwen2.5

        # Register layer hooks
        self.num_hooks = register_layer_hooks(model)
        print(f"[ErrorSourceFinder] Registered {self.num_hooks} layer hooks")

    def get_output_logits(self, prompt: str) -> torch.Tensor:
        """Get model output logits for a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get last token logits
            logits = outputs.logits[:, -1, :].clone()

        return logits

    def compute_kl_divergence(self, p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
        """Compute KL divergence between two logit distributions."""
        p = torch.softmax(p_logits, dim=-1)
        q = torch.softmax(q_logits, dim=-1)

        eps = 1e-10
        kl = torch.sum(p * torch.log((p + eps) / (q + eps)))
        return kl.item()

    def compute_cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity between two tensors."""
        a_flat = a.flatten().float()
        b_flat = b.flatten().float()
        return torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()

    def compute_deviation_pattern(self, clean: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        """Compute the deviation pattern (fingerprint) from clean to noisy output."""
        return noisy - clean

    def find_error_source(
        self,
        ground_truth_layer: int,
        prompts: List[str],
        search_range: Tuple[int, int] = None,
    ) -> Dict:
        """
        Find the error source layer using differential sensitivity.

        Args:
            ground_truth_layer: The layer with simulated HW error (for validation)
            prompts: Test prompts
            search_range: (start, end) layer range to search, None for all

        Returns:
            Dict with diagnosis results
        """
        if search_range is None:
            search_range = (0, self.num_layers)

        print(f"\n{'='*60}")
        print("ERROR SOURCE FINDER - Differential Sensitivity Method")
        print(f"{'='*60}")
        print(f"Ground Truth Layer: {ground_truth_layer}")
        print(f"HW Error Scale: {self.hw_error_scale}")
        print(f"Diagnostic Scale: {self.diagnostic_scale}")
        print(f"Search Range: layers {search_range[0]}-{search_range[1]-1}")
        print(f"{'='*60}")

        # Step 1: Get clean baseline (no noise)
        print("\n[Step 1] Getting clean baseline...")
        disable_noisy_ops()
        set_selective_layers(None)

        clean_logits = []
        for prompt in prompts:
            logits = self.get_output_logits(prompt)
            clean_logits.append(logits)

        # Step 2: Get degraded baseline (HW error on ground truth layer)
        print(f"\n[Step 2] Getting degraded baseline (HW error on layer {ground_truth_layer})...")
        enable_noisy_ops(error_scale=self.hw_error_scale, error_type='relative_gaussian')
        set_selective_layers([ground_truth_layer])

        degraded_logits = []
        for prompt in prompts:
            logits = self.get_output_logits(prompt)
            degraded_logits.append(logits)

        # Measure degradation from clean
        clean_to_degraded_div = []
        for i in range(len(prompts)):
            div = self.compute_kl_divergence(clean_logits[i], degraded_logits[i])
            clean_to_degraded_div.append(div)
        avg_degradation = np.mean(clean_to_degraded_div)
        print(f"Average degradation (clean → degraded): {avg_degradation:.4f}")

        # Step 3: Differential sensitivity sweep
        print(f"\n[Step 3] Running differential sensitivity sweep...")
        print("(Lower additional divergence = more likely error source)")

        additional_divergences = {}

        for layer_id in range(search_range[0], search_range[1]):
            # Keep HW error ON (ground truth layer)
            # Add diagnostic noise to candidate layer
            if layer_id == ground_truth_layer:
                # Both HW error and diagnostic on same layer
                layers_to_noise = [ground_truth_layer]
            else:
                # HW error on GT layer + diagnostic on candidate
                layers_to_noise = [ground_truth_layer, layer_id]

            set_selective_layers(layers_to_noise)
            reset_injection_stats()
            reset_layer_injection_stats()

            # Get output with both HW error + diagnostic noise
            combined_logits = []
            for prompt in prompts:
                logits = self.get_output_logits(prompt)
                combined_logits.append(logits)

            # Measure ADDITIONAL divergence from degraded baseline
            additional_divs = []
            for i in range(len(prompts)):
                div = self.compute_kl_divergence(degraded_logits[i], combined_logits[i])
                additional_divs.append(div)

            avg_additional = np.mean(additional_divs)
            additional_divergences[layer_id] = avg_additional

            marker = " <-- GT" if layer_id == ground_truth_layer else ""
            if (layer_id + 1) % 5 == 0 or layer_id == ground_truth_layer:
                print(f"  Layer {layer_id:2d}: additional_div = {avg_additional:.4f}{marker}")

        disable_noisy_ops()
        set_selective_layers(None)

        # Step 4: Identify error source (layer with LEAST additional divergence)
        print(f"\n[Step 4] Analyzing results...")

        # Sort by additional divergence (ascending - lower = more likely error source)
        sorted_layers = sorted(additional_divergences.items(), key=lambda x: x[1])

        print("\nTop 5 candidates (lowest additional divergence):")
        for rank, (layer_id, div) in enumerate(sorted_layers[:5], 1):
            marker = " <-- GROUND TRUTH" if layer_id == ground_truth_layer else ""
            print(f"  {rank}. Layer {layer_id}: {div:.4f}{marker}")

        # Diagnosis
        diagnosed_layer = sorted_layers[0][0]

        print(f"\n{'='*50}")
        print(f"Ground Truth Layer: {ground_truth_layer}")
        print(f"Diagnosed Layer:    {diagnosed_layer}")

        # Check if ground truth is in top 3
        top_3_ids = [l[0] for l in sorted_layers[:3]]

        if diagnosed_layer == ground_truth_layer:
            result = "EXACT_MATCH"
            print("Result: EXACT MATCH")
        elif ground_truth_layer in top_3_ids:
            rank = top_3_ids.index(ground_truth_layer) + 1
            result = f"IN_TOP_3_RANK_{rank}"
            print(f"Result: In top 3 (rank {rank})")
        elif abs(diagnosed_layer - ground_truth_layer) <= 2:
            result = "ADJACENT"
            print(f"Result: Adjacent layer (within 2)")
        else:
            result = "MISMATCH"
            print(f"Result: MISMATCH")

        print(f"{'='*50}")

        return {
            "ground_truth": ground_truth_layer,
            "diagnosed": diagnosed_layer,
            "result": result,
            "clean_to_degraded_div": avg_degradation,
            "additional_divergences": additional_divergences,
            "sorted_candidates": sorted_layers[:10],
        }

    def find_error_source_fingerprint(
        self,
        ground_truth_layer: int,
        prompts: List[str],
        search_range: Tuple[int, int] = None,
    ) -> Dict:
        """
        Find error source using fingerprint correlation method.

        This method compares the "deviation pattern" caused by HW error with
        the pattern caused by injecting noise into each candidate layer.
        The layer whose pattern best matches the HW error pattern is the source.

        Algorithm:
        1. Get clean baseline (no noise)
        2. Get degraded output (HW error on GT layer) - this is the "failure fingerprint"
        3. For each candidate layer:
           - Inject noise ONLY into that layer
           - Compute the "candidate fingerprint" (deviation from clean)
           - Compute similarity between candidate and failure fingerprint
        4. Layer with HIGHEST similarity is the error source
        """
        if search_range is None:
            search_range = (0, self.num_layers)

        print(f"\n{'='*60}")
        print("ERROR SOURCE FINDER - Fingerprint Correlation Method")
        print(f"{'='*60}")
        print(f"Ground Truth Layer: {ground_truth_layer}")
        print(f"HW Error Scale: {self.hw_error_scale}")
        print(f"Diagnostic Scale: {self.diagnostic_scale}")
        print(f"Search Range: layers {search_range[0]}-{search_range[1]-1}")
        print(f"{'='*60}")

        # Step 1: Get clean baseline
        print("\n[Step 1] Getting clean baseline...")
        disable_noisy_ops()
        set_selective_layers(None)

        clean_logits = []
        for prompt in prompts:
            logits = self.get_output_logits(prompt)
            clean_logits.append(logits)

        # Step 2: Get failure fingerprint (HW error on GT layer)
        print(f"\n[Step 2] Getting failure fingerprint (HW error on layer {ground_truth_layer})...")
        enable_noisy_ops(error_scale=self.hw_error_scale, error_type='relative_gaussian')
        set_selective_layers([ground_truth_layer])

        failure_logits = []
        failure_fingerprints = []
        for i, prompt in enumerate(prompts):
            logits = self.get_output_logits(prompt)
            failure_logits.append(logits)
            fingerprint = self.compute_deviation_pattern(clean_logits[i], logits)
            failure_fingerprints.append(fingerprint)

        disable_noisy_ops()

        # Step 3: Fingerprint matching sweep
        print(f"\n[Step 3] Running fingerprint correlation sweep...")
        print("(Higher similarity = more likely error source)")

        correlations = {}

        for layer_id in range(search_range[0], search_range[1]):
            set_selective_layers([layer_id])
            enable_noisy_ops(error_scale=self.diagnostic_scale, error_type='relative_gaussian')
            reset_injection_stats()
            reset_layer_injection_stats()

            # Get candidate fingerprint
            candidate_similarities = []
            for i, prompt in enumerate(prompts):
                logits = self.get_output_logits(prompt)
                candidate_fingerprint = self.compute_deviation_pattern(clean_logits[i], logits)

                # Compute similarity between candidate and failure fingerprint
                similarity = self.compute_cosine_similarity(
                    failure_fingerprints[i], candidate_fingerprint
                )
                candidate_similarities.append(similarity)

            avg_similarity = np.mean(candidate_similarities)
            correlations[layer_id] = avg_similarity

            disable_noisy_ops()

            marker = " <-- GT" if layer_id == ground_truth_layer else ""
            if (layer_id + 1) % 5 == 0 or layer_id == ground_truth_layer:
                print(f"  Layer {layer_id:2d}: similarity = {avg_similarity:.4f}{marker}")

        set_selective_layers(None)

        # Step 4: Identify error source (layer with HIGHEST similarity)
        print(f"\n[Step 4] Analyzing results...")

        # Sort by similarity (descending - higher = more likely error source)
        sorted_layers = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

        print("\nTop 5 candidates (highest fingerprint similarity):")
        for rank, (layer_id, sim) in enumerate(sorted_layers[:5], 1):
            marker = " <-- GROUND TRUTH" if layer_id == ground_truth_layer else ""
            print(f"  {rank}. Layer {layer_id}: {sim:.4f}{marker}")

        # Diagnosis
        diagnosed_layer = sorted_layers[0][0]

        print(f"\n{'='*50}")
        print(f"Ground Truth Layer: {ground_truth_layer}")
        print(f"Diagnosed Layer:    {diagnosed_layer}")

        # Check result
        top_3_ids = [l[0] for l in sorted_layers[:3]]

        if diagnosed_layer == ground_truth_layer:
            result = "EXACT_MATCH"
            print("Result: EXACT MATCH")
        elif ground_truth_layer in top_3_ids:
            rank = top_3_ids.index(ground_truth_layer) + 1
            result = f"IN_TOP_3_RANK_{rank}"
            print(f"Result: In top 3 (rank {rank})")
        elif abs(diagnosed_layer - ground_truth_layer) <= 2:
            result = "ADJACENT"
            print(f"Result: Adjacent layer (within 2)")
        else:
            result = "MISMATCH"
            print(f"Result: MISMATCH")

        print(f"{'='*50}")

        return {
            "ground_truth": ground_truth_layer,
            "diagnosed": diagnosed_layer,
            "result": result,
            "correlations": correlations,
            "sorted_candidates": sorted_layers[:10],
            "method": "fingerprint_correlation",
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


def run_multi_trial(
    finder: ErrorSourceFinder,
    ground_truth_layer: int,
    prompts: List[str],
    search_range: Tuple[int, int],
    num_trials: int = 3,
) -> Dict:
    """Run multiple trials and aggregate results using voting."""
    print(f"\n{'='*60}")
    print(f"MULTI-TRIAL ERROR SOURCE FINDER ({num_trials} trials)")
    print(f"{'='*60}")

    all_results = []
    layer_votes = {}

    for trial in range(num_trials):
        print(f"\n>>> Trial {trial + 1}/{num_trials}")
        # Set different seed for each trial
        torch.manual_seed(42 + trial * 100)
        np.random.seed(42 + trial * 100)

        result = finder.find_error_source(
            ground_truth_layer=ground_truth_layer,
            prompts=prompts,
            search_range=search_range,
        )
        all_results.append(result)

        # Count votes for top 3 candidates
        for rank, (layer_id, div) in enumerate(result["sorted_candidates"][:3]):
            weight = 3 - rank  # Rank 1 gets 3 votes, rank 2 gets 2, rank 3 gets 1
            layer_votes[layer_id] = layer_votes.get(layer_id, 0) + weight

    # Final decision based on voting
    sorted_votes = sorted(layer_votes.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*60}")
    print("MULTI-TRIAL AGGREGATED RESULTS")
    print(f"{'='*60}")
    print(f"Ground Truth Layer: {ground_truth_layer}")
    print(f"\nVoting results (weighted by rank):")
    for layer_id, votes in sorted_votes[:5]:
        marker = " <-- GROUND TRUTH" if layer_id == ground_truth_layer else ""
        print(f"  Layer {layer_id}: {votes} votes{marker}")

    final_diagnosed = sorted_votes[0][0]
    print(f"\nFinal Diagnosis: Layer {final_diagnosed}")

    # Check result
    top_3_ids = [l[0] for l in sorted_votes[:3]]
    if final_diagnosed == ground_truth_layer:
        final_result = "EXACT_MATCH"
        print("Result: EXACT MATCH")
    elif ground_truth_layer in top_3_ids:
        rank = top_3_ids.index(ground_truth_layer) + 1
        final_result = f"IN_TOP_3_RANK_{rank}"
        print(f"Result: In top 3 (rank {rank})")
    elif abs(final_diagnosed - ground_truth_layer) <= 2:
        final_result = "ADJACENT"
        print(f"Result: Adjacent layer (within 2)")
    else:
        final_result = "MISMATCH"
        print(f"Result: MISMATCH")

    print(f"{'='*60}")

    return {
        "ground_truth": ground_truth_layer,
        "final_diagnosed": final_diagnosed,
        "final_result": final_result,
        "voting_results": sorted_votes,
        "all_trial_results": all_results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Error Source Finder")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--ground_truth_layer", type=int, default=10,
                       help="Layer with simulated HW error (ground truth)")
    parser.add_argument("--hw_error_scale", type=float, default=0.05,
                       help="HW error noise scale")
    parser.add_argument("--diagnostic_scale", type=float, default=0.05,
                       help="Diagnostic noise scale")
    parser.add_argument("--search_start", type=int, default=0,
                       help="Start layer for search")
    parser.add_argument("--search_end", type=int, default=None,
                       help="End layer for search (exclusive)")
    parser.add_argument("--num_trials", type=int, default=1,
                       help="Number of trials for multi-trial mode (1=single trial)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--method", type=str, default="differential",
                       choices=["differential", "fingerprint", "both"],
                       help="Method: differential (least additional divergence), "
                            "fingerprint (pattern correlation), or both")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")

    args = parser.parse_args()

    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to {args.seed}")

    # Load model
    model, tokenizer = load_model(args.model_path, args.device)

    # Test prompts
    prompts = [
        "What is 2 + 2? Answer:",
        "The capital of France is",
        "def fibonacci(n):",
        "The largest planet in our solar system is",
        "Water boils at",
    ]

    # Create finder
    finder = ErrorSourceFinder(
        model=model,
        tokenizer=tokenizer,
        hw_error_scale=args.hw_error_scale,
        diagnostic_scale=args.diagnostic_scale,
        device=args.device,
    )

    # Determine search range
    search_end = args.search_end if args.search_end else finder.num_layers
    search_range = (args.search_start, search_end)

    # Run methods based on selection
    all_results = []

    if args.method in ["differential", "both"]:
        print("\n" + "="*70)
        print("METHOD 1: DIFFERENTIAL SENSITIVITY")
        print("="*70)
        if args.num_trials > 1:
            results = run_multi_trial(
                finder=finder,
                ground_truth_layer=args.ground_truth_layer,
                prompts=prompts,
                search_range=search_range,
                num_trials=args.num_trials,
            )
            results["method"] = "differential"
            all_results.append(results)
        else:
            results = finder.find_error_source(
                ground_truth_layer=args.ground_truth_layer,
                prompts=prompts,
                search_range=search_range,
            )
            results["method"] = "differential"
            all_results.append(results)

    if args.method in ["fingerprint", "both"]:
        print("\n" + "="*70)
        print("METHOD 2: FINGERPRINT CORRELATION")
        print("="*70)
        results = finder.find_error_source_fingerprint(
            ground_truth_layer=args.ground_truth_layer,
            prompts=prompts,
            search_range=search_range,
        )
        all_results.append(results)

    # Summary if both methods were run
    if args.method == "both":
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        for r in all_results:
            method = r.get("method", "unknown")
            diagnosed = r.get("diagnosed") or r.get("final_diagnosed")
            result = r.get("result") or r.get("final_result")
            print(f"  {method:20s}: Diagnosed layer {diagnosed}, {result}")

    # Exit code based on results
    success = False
    for r in all_results:
        result = r.get("result") or r.get("final_result")
        if result in ["EXACT_MATCH", "IN_TOP_3_RANK_1", "IN_TOP_3_RANK_2", "IN_TOP_3_RANK_3"]:
            success = True
            break

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
