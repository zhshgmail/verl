#!/usr/bin/env python3
"""
SRDD-Guided AQN vs Global AQN Experiment

This experiment validates whether SRDD-guided targeted AQN outperforms global AQN
when dealing with MXFP4 deadzone errors in specific layers.

Experiment Design:
1. Inject deadzone error in layer N (simulating MXFP4 quantization loss)
2. Run SRDD to detect which layer has the fault
3. Compare training approaches:
   - Baseline: No deadzone, no AQN (clean reference)
   - Deadzone only: Deadzone error without AQN (shows degradation)
   - Global AQN: Deadzone + AQN on ALL layers (current approach)
   - SRDD-guided AQN: Deadzone + AQN ONLY on detected layer (proposed approach)

Expected Results:
- SRDD-guided AQN should achieve better metrics than Global AQN because:
  - Less noise in healthy layers = better gradient signal
  - Targeted noise where needed = same protection against deadzone
  - More efficient use of noise budget

Usage:
    # Run on A100
    python scripts/srdd_aqn_experiment.py \
        --model_path /path/to/Qwen2.5-1.5B-Instruct \
        --faulty_layer 10 \
        --deadzone_threshold 0.01

    # Quick test (just SRDD detection)
    python scripts/srdd_aqn_experiment.py \
        --model_path /path/to/model \
        --mode detect_only
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_srdd_detection(model_path: str, faulty_layer: int, deadzone_threshold: float, device: str = "cuda"):
    """Run SRDD to detect deadzone fault in the model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("Step 1: SRDD Deadzone Detection")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Injected fault layer: {faulty_layer}")
    print(f"  Deadzone threshold: {deadzone_threshold}")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = len(model.model.layers)
    print(f"  Model has {num_layers} layers")

    # Import SRDD components
    from scripts.srdd_error_finder import SRDDErrorFinder, HWFaultSimulator

    # Create SRDD finder
    finder = SRDDErrorFinder(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # Create fault simulator for deadzone
    fault_simulator = HWFaultSimulator(
        model=model,
        fault_type="dead_zone",
        fault_layer=faulty_layer,
        fault_magnitude=deadzone_threshold,
        sparsity=1.0,  # Dense fault (all elements affected)
    )

    # Test prompts
    prompts = [
        "What is 2 + 2? Answer:",
        "The capital of France is",
        "def fibonacci(n):",
        "Water boils at",
        "Explain photosynthesis in one sentence:",
    ]

    print("\nRunning SRDD local gain scan...")
    # Run local scan (gain scan for deadzone detection)
    results = finder.run_local_scan(
        prompts=prompts,
        method="gain",  # Gain scan detects deadzone (signal loss)
        verbose=True,
    )

    # Analyze results
    print("\n" + "=" * 60)
    print("SRDD Detection Results")
    print("=" * 60)

    detected_layer = None
    min_gain = float('inf')
    layer_gains = {}

    for layer_idx, layer_result in results.items():
        gain = layer_result.get('mean_gain', 1.0)
        layer_gains[layer_idx] = gain
        if gain < min_gain:
            min_gain = gain
            detected_layer = layer_idx

    print(f"\n  Injected fault layer: {faulty_layer}")
    print(f"  SRDD detected layer:  {detected_layer}")
    print(f"  Minimum gain value:   {min_gain:.4f}")

    # Check if detection is correct
    detection_correct = (detected_layer == faulty_layer)
    print(f"\n  Detection correct: {'YES' if detection_correct else 'NO'}")

    # Clean up
    fault_simulator.disable()
    del model
    torch.cuda.empty_cache()

    return {
        'faulty_layer': faulty_layer,
        'detected_layer': detected_layer,
        'detection_correct': detection_correct,
        'min_gain': min_gain,
        'layer_gains': layer_gains,
    }


def run_inference_comparison(
    model_path: str,
    faulty_layer: int,
    deadzone_threshold: float,
    device: str = "cuda",
):
    """Compare inference quality with different configurations."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 60)
    print("Step 2: Inference Quality Comparison")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test prompts for GSM8K-style math problems
    test_prompts = [
        "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 to bake. How many eggs does she sell daily?",
        "A store has 50 apples. If 20 are sold and 10 more arrive, how many apples are there?",
        "Tom has $100. He spends $30 on books and $25 on food. How much money does he have left?",
    ]

    results = {}

    # Configuration 1: Clean (no deadzone)
    print("\n--- Config 1: Clean (no deadzone) ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    clean_outputs = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        clean_outputs.append(response)
        print(f"  Q: {prompt[:50]}...")
        print(f"  A: {response[len(prompt):].strip()[:100]}...")

    results['clean'] = clean_outputs
    del model
    torch.cuda.empty_cache()

    # Configuration 2: Deadzone only (no AQN)
    print("\n--- Config 2: Deadzone on layer {} (no AQN) ---".format(faulty_layer))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Inject deadzone using hook
    from verl.utils.hw_error_injection import HWErrorConfig, HWErrorInjector

    deadzone_config = HWErrorConfig(
        enabled=True,
        error_type='deadzone',
        deadzone_threshold=deadzone_threshold,
        target_layers=[faulty_layer],
        injection_point='output',
        apply_during='both',
    )
    injector = HWErrorInjector(deadzone_config)
    injector.set_phase('both')  # 'inference' not valid, use 'both'
    num_hooks = injector.register_hooks(model, verbose=True)
    print(f"  Registered {num_hooks} deadzone hooks")

    deadzone_outputs = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        deadzone_outputs.append(response)
        print(f"  Q: {prompt[:50]}...")
        print(f"  A: {response[len(prompt):].strip()[:100]}...")

    results['deadzone_only'] = deadzone_outputs
    injector.remove_hooks()
    del model
    torch.cuda.empty_cache()

    # Compare outputs
    print("\n" + "=" * 60)
    print("Output Comparison")
    print("=" * 60)

    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: {prompt[:60]}...")
        print(f"  Clean:    {clean_outputs[i][len(prompt):].strip()[:80]}...")
        print(f"  Deadzone: {deadzone_outputs[i][len(prompt):].strip()[:80]}...")

        # Check if outputs match
        clean_answer = clean_outputs[i][len(prompt):].strip()
        deadzone_answer = deadzone_outputs[i][len(prompt):].strip()
        match = clean_answer == deadzone_answer
        print(f"  Match: {'YES' if match else 'NO - DEGRADATION DETECTED'}")

    return results


def run_aqn_comparison_simple(
    model_path: str,
    faulty_layer: int,
    deadzone_threshold: float,
    aqn_gamma: float = 0.01,
    device: str = "cuda",
):
    """
    Simple AQN comparison using forward pass loss.

    Compares:
    1. Clean model (reference)
    2. Deadzone only (shows degradation)
    3. Deadzone + Global AQN
    4. Deadzone + Targeted AQN (SRDD-guided)
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 60)
    print("Step 3: AQN Comparison (Forward Pass)")
    print("=" * 60)
    print(f"  AQN gamma: {aqn_gamma}")
    print(f"  Deadzone threshold: {deadzone_threshold}")
    print(f"  Target layer: {faulty_layer}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test inputs
    test_texts = [
        "The answer to 2 + 2 is 4.",
        "Paris is the capital of France.",
        "Water freezes at 0 degrees Celsius.",
        "The sun rises in the east.",
        "Python is a programming language.",
    ]

    results = {}

    def compute_loss(model, texts, tokenizer, device):
        """Compute average cross-entropy loss on texts."""
        total_loss = 0
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs.input_ids)
                total_loss += outputs.loss.item()
        return total_loss / len(texts)

    def add_aqn_noise(module, input, output):
        """Add AQN (Adaptive Quantization Noise) to output."""
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        noise = torch.randn_like(hidden) * aqn_gamma * hidden.abs().mean()
        if isinstance(output, tuple):
            return (hidden + noise,) + output[1:]
        return hidden + noise

    # Config 1: Clean
    print("\n--- Config 1: Clean ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()
    clean_loss = compute_loss(model, test_texts, tokenizer, device)
    print(f"  Loss: {clean_loss:.4f}")
    results['clean'] = {'loss': clean_loss}
    del model
    torch.cuda.empty_cache()

    # Config 2: Deadzone only
    print("\n--- Config 2: Deadzone only ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()

    from verl.utils.hw_error_injection import HWErrorConfig, HWErrorInjector
    deadzone_config = HWErrorConfig(
        enabled=True, error_type='deadzone', deadzone_threshold=deadzone_threshold,
        target_layers=[faulty_layer], injection_point='output', apply_during='both',
    )
    dz_injector = HWErrorInjector(deadzone_config)
    dz_injector.set_phase('inference')
    dz_injector.register_hooks(model, verbose=False)

    deadzone_loss = compute_loss(model, test_texts, tokenizer, device)
    print(f"  Loss: {deadzone_loss:.4f} (degradation: {(deadzone_loss - clean_loss) / clean_loss * 100:.1f}%)")
    results['deadzone_only'] = {'loss': deadzone_loss, 'degradation': (deadzone_loss - clean_loss) / clean_loss}

    dz_injector.remove_hooks()
    del model
    torch.cuda.empty_cache()

    # Config 3: Deadzone + Global AQN
    print("\n--- Config 3: Deadzone + Global AQN (all layers) ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()

    # Add deadzone
    dz_injector = HWErrorInjector(deadzone_config)
    dz_injector.set_phase('inference')
    dz_injector.register_hooks(model, verbose=False)

    # Add global AQN (all decoder layers)
    aqn_hooks = []
    for i, layer in enumerate(model.model.layers):
        hook = layer.register_forward_hook(add_aqn_noise)
        aqn_hooks.append(hook)
    print(f"  Registered {len(aqn_hooks)} AQN hooks (all layers)")

    global_aqn_loss = compute_loss(model, test_texts, tokenizer, device)
    print(f"  Loss: {global_aqn_loss:.4f} (vs deadzone: {(global_aqn_loss - deadzone_loss) / deadzone_loss * 100:+.1f}%)")
    results['global_aqn'] = {
        'loss': global_aqn_loss,
        'vs_deadzone': (global_aqn_loss - deadzone_loss) / deadzone_loss,
        'vs_clean': (global_aqn_loss - clean_loss) / clean_loss,
    }

    for hook in aqn_hooks:
        hook.remove()
    dz_injector.remove_hooks()
    del model
    torch.cuda.empty_cache()

    # Config 4: Deadzone + Targeted AQN (SRDD-guided)
    print(f"\n--- Config 4: Deadzone + Targeted AQN (layer {faulty_layer} only) ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()

    # Add deadzone
    dz_injector = HWErrorInjector(deadzone_config)
    dz_injector.set_phase('inference')
    dz_injector.register_hooks(model, verbose=False)

    # Add targeted AQN (only on faulty layer)
    layer = model.model.layers[faulty_layer]
    targeted_hook = layer.register_forward_hook(add_aqn_noise)
    print(f"  Registered 1 AQN hook (layer {faulty_layer} only)")

    targeted_aqn_loss = compute_loss(model, test_texts, tokenizer, device)
    print(f"  Loss: {targeted_aqn_loss:.4f} (vs deadzone: {(targeted_aqn_loss - deadzone_loss) / deadzone_loss * 100:+.1f}%)")
    results['targeted_aqn'] = {
        'loss': targeted_aqn_loss,
        'vs_deadzone': (targeted_aqn_loss - deadzone_loss) / deadzone_loss,
        'vs_clean': (targeted_aqn_loss - clean_loss) / clean_loss,
    }

    targeted_hook.remove()
    dz_injector.remove_hooks()
    del model
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Clean loss:        {clean_loss:.4f}")
    print(f"  Deadzone loss:     {deadzone_loss:.4f} ({results['deadzone_only']['degradation']*100:+.1f}%)")
    print(f"  Global AQN loss:   {global_aqn_loss:.4f} ({results['global_aqn']['vs_clean']*100:+.1f}% vs clean)")
    print(f"  Targeted AQN loss: {targeted_aqn_loss:.4f} ({results['targeted_aqn']['vs_clean']*100:+.1f}% vs clean)")
    print()

    # Determine winner
    if targeted_aqn_loss < global_aqn_loss:
        improvement = (global_aqn_loss - targeted_aqn_loss) / global_aqn_loss * 100
        print(f"  CONCLUSION: SRDD-guided AQN is {improvement:.1f}% BETTER than Global AQN")
        results['conclusion'] = 'srdd_guided_wins'
        results['improvement'] = improvement
    else:
        print(f"  CONCLUSION: Global AQN performs similar or better")
        results['conclusion'] = 'global_aqn_wins'
        results['improvement'] = 0

    return results


def main():
    parser = argparse.ArgumentParser(description="SRDD-Guided AQN vs Global AQN Experiment")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model")
    parser.add_argument("--faulty_layer", type=int, default=10,
                       help="Layer to inject deadzone fault")
    parser.add_argument("--deadzone_threshold", type=float, default=0.01,
                       help="Deadzone threshold (fraction of max value)")
    parser.add_argument("--aqn_gamma", type=float, default=0.01,
                       help="AQN noise gamma parameter")
    parser.add_argument("--mode", type=str, default="full",
                       choices=["detect_only", "compare_only", "full"],
                       help="Experiment mode")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")

    args = parser.parse_args()

    print("=" * 60)
    print("SRDD-Guided AQN vs Global AQN Experiment")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Faulty layer: {args.faulty_layer}")
    print(f"Deadzone threshold: {args.deadzone_threshold}")
    print(f"AQN gamma: {args.aqn_gamma}")
    print(f"Mode: {args.mode}")
    print()

    all_results = {
        'config': {
            'model_path': args.model_path,
            'faulty_layer': args.faulty_layer,
            'deadzone_threshold': args.deadzone_threshold,
            'aqn_gamma': args.aqn_gamma,
        },
        'timestamp': datetime.now().isoformat(),
    }

    # Step 1: SRDD Detection (optional)
    if args.mode in ["detect_only", "full"]:
        try:
            detection_results = run_srdd_detection(
                model_path=args.model_path,
                faulty_layer=args.faulty_layer,
                deadzone_threshold=args.deadzone_threshold,
                device=args.device,
            )
            all_results['detection'] = detection_results
        except Exception as e:
            print(f"SRDD detection failed: {e}")
            print("Continuing with known faulty layer...")
            all_results['detection'] = {'error': str(e)}

    # Step 2: Inference comparison
    if args.mode in ["compare_only", "full"]:
        inference_results = run_inference_comparison(
            model_path=args.model_path,
            faulty_layer=args.faulty_layer,
            deadzone_threshold=args.deadzone_threshold,
            device=args.device,
        )
        all_results['inference'] = inference_results

    # Step 3: AQN comparison
    if args.mode in ["compare_only", "full"]:
        aqn_results = run_aqn_comparison_simple(
            model_path=args.model_path,
            faulty_layer=args.faulty_layer,
            deadzone_threshold=args.deadzone_threshold,
            aqn_gamma=args.aqn_gamma,
            device=args.device,
        )
        all_results['aqn_comparison'] = aqn_results

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    main()
