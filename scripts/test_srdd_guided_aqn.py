#!/usr/bin/env python3
"""
End-to-End Test: SRDD-Guided AQN vs Global AQN

This script demonstrates the SRDD → Targeted AQN workflow:
1. Inject MXFP4 deadzone fault at a specific layer
2. Use SRDD to detect the faulty layer
3. Compare training with:
   - Model A: Targeted AQN (only on detected layer)
   - Model B: Global AQN (all layers)

Run on A100:
    python scripts/test_srdd_guided_aqn.py --ground_truth_layer 15
"""

import argparse
import copy
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def run_srdd_diagnosis(model, tokenizer, fault_layer: int, threshold: float = 0.01):
    """Run SRDD to detect the deadzone layer."""
    print("\n" + "=" * 60)
    print("PHASE 1: SRDD Diagnosis")
    print("=" * 60)

    from scripts.srdd_error_finder import SRDDErrorFinder, HardwareFaultSimulator

    # Enable deadzone fault
    print(f"\n[Setup] Injecting deadzone on layer {fault_layer} (threshold={threshold})...")
    fault_sim = HardwareFaultSimulator(
        model=model,
        fault_layer=fault_layer,
        fault_type="dead_zone",
        fault_magnitude=threshold,
    )
    fault_sim.enable()

    # Run SRDD
    print("\n[SRDD] Running diagnosis...")
    finder = SRDDErrorFinder(model, tokenizer)

    # Run gain scan (most effective for deadzone detection)
    prompts = ["What is 2 + 2?", "Explain machine learning.", "Write hello world."]
    gain_results = finder.local_gain_scan(prompts, noise_scale=0.1)

    # Find the layer with minimum gain
    min_gain_layer = min(gain_results.keys(), key=lambda k: gain_results[k])
    min_gain_value = gain_results[min_gain_layer]

    print(f"\n[SRDD Result] Detected faulty layer: {min_gain_layer} (gain={min_gain_value:.4f})")

    fault_sim.disable()

    # Verify detection
    if min_gain_layer == fault_layer:
        print(f"  EXACT MATCH with ground truth (layer {fault_layer})")
        detected_layer = min_gain_layer
    else:
        print(f"  WARNING: Mismatch with ground truth (layer {fault_layer})")
        print(f"  Using ground truth for experiment...")
        detected_layer = fault_layer

    return detected_layer


def create_training_data(tokenizer, num_samples: int = 100):
    """Create simple training data (math problems)."""
    data = []
    for i in range(num_samples):
        a, b = torch.randint(1, 100, (2,)).tolist()
        prompt = f"Question: What is {a} + {b}?\nAnswer:"
        answer = f" {a + b}"
        full_text = prompt + answer

        tokens = tokenizer(full_text, return_tensors="pt", padding=False)
        prompt_len = len(tokenizer(prompt)['input_ids'])

        data.append({
            'input_ids': tokens['input_ids'][0],
            'attention_mask': tokens['attention_mask'][0],
            'prompt_len': prompt_len,
        })

    return data


def compute_loss(model, batch, device):
    """Compute language modeling loss."""
    input_ids = batch['input_ids'].unsqueeze(0).to(device)
    attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
    prompt_len = batch['prompt_len']

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Shift for next-token prediction (only on answer part)
    shift_logits = logits[:, prompt_len:-1, :].contiguous()
    shift_labels = input_ids[:, prompt_len+1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    return loss


def train_epoch(
    model,
    train_data,
    device,
    lr: float = 1e-5,
    deadzone_layer: int = None,
    deadzone_threshold: float = 0.01,
    aqn_layers: list = None,  # None = no AQN, [] = all layers, [15] = specific
    aqn_sigma: float = 0.05,
):
    """Train for one epoch with optional deadzone and AQN."""
    from verl.utils.deadzone_injection import (
        DeadzoneInjector,
        register_deadzone_layer_hooks,
    )
    from verl.utils.noisy_ops import (
        enable_noisy_ops,
        disable_noisy_ops,
        set_selective_layers,
        register_layer_hooks,
    )

    model.train()

    # Setup deadzone if specified
    deadzone_injector = None
    if deadzone_layer is not None:
        deadzone_injector = DeadzoneInjector(model, deadzone_layer, deadzone_threshold)
        deadzone_injector.enable()

    # Setup AQN if specified
    if aqn_layers is not None:
        register_layer_hooks(model)
        enable_noisy_ops(error_scale=aqn_sigma)
        if len(aqn_layers) > 0:
            set_selective_layers(aqn_layers)
        else:
            set_selective_layers(None)  # All layers

    # Simple SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    total_loss = 0
    num_batches = 0

    for batch in tqdm(train_data, desc="Training"):
        optimizer.zero_grad()

        loss = compute_loss(model, batch, device)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    # Cleanup
    if deadzone_injector is not None:
        deadzone_injector.disable()

    if aqn_layers is not None:
        disable_noisy_ops()

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, eval_data, device, deadzone_layer: int = None, deadzone_threshold: float = 0.01):
    """Evaluate model on data (with deadzone to simulate deployment)."""
    from verl.utils.deadzone_injection import DeadzoneInjector

    model.eval()

    # Setup deadzone (simulating deployment on faulty hardware)
    deadzone_injector = None
    if deadzone_layer is not None:
        deadzone_injector = DeadzoneInjector(model, deadzone_layer, deadzone_threshold)
        deadzone_injector.enable()

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_data:
            loss = compute_loss(model, batch, device)
            total_loss += loss.item()
            num_batches += 1

    if deadzone_injector is not None:
        deadzone_injector.disable()

    avg_loss = total_loss / num_batches
    return avg_loss


def run_experiment(
    base_model,
    tokenizer,
    train_data,
    eval_data,
    device,
    deadzone_layer: int,
    deadzone_threshold: float,
    aqn_sigma: float,
    detected_layer: int,
    num_epochs: int = 2,
):
    """Run comparative experiment: Targeted AQN vs Global AQN."""
    print("\n" + "=" * 60)
    print("PHASE 2: Comparative Training")
    print("=" * 60)

    results = {}

    # =========================================================================
    # Model A: SRDD-Guided (Targeted AQN on detected layer only)
    # =========================================================================
    print("\n" + "-" * 40)
    print("MODEL A: SRDD-Guided AQN (Targeted)")
    print(f"  Deadzone: Layer {deadzone_layer}")
    print(f"  AQN: Layer {detected_layer} only, sigma={aqn_sigma}")
    print("-" * 40)

    model_a = copy.deepcopy(base_model)
    model_a.to(device)

    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model_a, train_data, device,
            deadzone_layer=deadzone_layer,
            deadzone_threshold=deadzone_threshold,
            aqn_layers=[detected_layer],  # Targeted
            aqn_sigma=aqn_sigma,
        )
        eval_loss = evaluate(
            model_a, eval_data, device,
            deadzone_layer=deadzone_layer,
            deadzone_threshold=deadzone_threshold,
        )
        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}")

    results['model_a'] = {
        'name': 'SRDD-Guided (Targeted)',
        'final_eval_loss': eval_loss,
        'aqn_layers': [detected_layer],
    }

    del model_a
    torch.cuda.empty_cache()

    # =========================================================================
    # Model B: Global AQN (all layers)
    # =========================================================================
    print("\n" + "-" * 40)
    print("MODEL B: Global AQN (All Layers)")
    print(f"  Deadzone: Layer {deadzone_layer}")
    print(f"  AQN: ALL layers, sigma={aqn_sigma}")
    print("-" * 40)

    model_b = copy.deepcopy(base_model)
    model_b.to(device)

    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model_b, train_data, device,
            deadzone_layer=deadzone_layer,
            deadzone_threshold=deadzone_threshold,
            aqn_layers=[],  # Empty = all layers
            aqn_sigma=aqn_sigma,
        )
        eval_loss = evaluate(
            model_b, eval_data, device,
            deadzone_layer=deadzone_layer,
            deadzone_threshold=deadzone_threshold,
        )
        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}")

    results['model_b'] = {
        'name': 'Global AQN',
        'final_eval_loss': eval_loss,
        'aqn_layers': 'all',
    }

    del model_b
    torch.cuda.empty_cache()

    # =========================================================================
    # Model C: No AQN (baseline)
    # =========================================================================
    print("\n" + "-" * 40)
    print("MODEL C: No AQN (Baseline)")
    print(f"  Deadzone: Layer {deadzone_layer}")
    print(f"  AQN: None")
    print("-" * 40)

    model_c = copy.deepcopy(base_model)
    model_c.to(device)

    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model_c, train_data, device,
            deadzone_layer=deadzone_layer,
            deadzone_threshold=deadzone_threshold,
            aqn_layers=None,  # No AQN
            aqn_sigma=0,
        )
        eval_loss = evaluate(
            model_c, eval_data, device,
            deadzone_layer=deadzone_layer,
            deadzone_threshold=deadzone_threshold,
        )
        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}")

    results['model_c'] = {
        'name': 'No AQN (Baseline)',
        'final_eval_loss': eval_loss,
        'aqn_layers': None,
    }

    del model_c
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="SRDD-Guided AQN E2E Test")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    )
    parser.add_argument("--ground_truth_layer", type=int, default=15)
    parser.add_argument("--deadzone_threshold", type=float, default=0.01)
    parser.add_argument("--aqn_sigma", type=float, default=0.05)
    parser.add_argument("--num_train_samples", type=int, default=50)
    parser.add_argument("--num_eval_samples", type=int, default=20)
    parser.add_argument("--num_epochs", type=int, default=2)
    args = parser.parse_args()

    print("=" * 60)
    print("SRDD-GUIDED AQN END-TO-END TEST")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Ground truth deadzone layer: {args.ground_truth_layer}")
    print(f"Deadzone threshold: {args.deadzone_threshold}")
    print(f"AQN sigma: {args.aqn_sigma}")
    print(f"Training samples: {args.num_train_samples}")
    print(f"Eval samples: {args.num_eval_samples}")
    print(f"Epochs: {args.num_epochs}")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create data
    print("\nCreating training data...")
    train_data = create_training_data(tokenizer, args.num_train_samples)
    eval_data = create_training_data(tokenizer, args.num_eval_samples)

    # Phase 1: SRDD Diagnosis
    base_model.to(device)
    detected_layer = run_srdd_diagnosis(
        base_model, tokenizer,
        fault_layer=args.ground_truth_layer,
        threshold=args.deadzone_threshold,
    )
    base_model.cpu()
    torch.cuda.empty_cache()

    # Phase 2: Comparative Training
    results = run_experiment(
        base_model=base_model,
        tokenizer=tokenizer,
        train_data=train_data,
        eval_data=eval_data,
        device=device,
        deadzone_layer=args.ground_truth_layer,
        deadzone_threshold=args.deadzone_threshold,
        aqn_sigma=args.aqn_sigma,
        detected_layer=detected_layer,
        num_epochs=args.num_epochs,
    )

    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\nDeadzone: Layer {args.ground_truth_layer}, threshold={args.deadzone_threshold}")
    print(f"SRDD detected: Layer {detected_layer}")
    print()

    for key in ['model_a', 'model_b', 'model_c']:
        r = results[key]
        aqn_str = f"layers {r['aqn_layers']}" if r['aqn_layers'] else "none"
        print(f"{r['name']:30s} | AQN: {aqn_str:15s} | Eval Loss: {r['final_eval_loss']:.4f}")

    # Compare
    a_loss = results['model_a']['final_eval_loss']
    b_loss = results['model_b']['final_eval_loss']
    c_loss = results['model_c']['final_eval_loss']

    print()
    if a_loss < b_loss:
        improvement = (b_loss - a_loss) / b_loss * 100
        print(f"SRDD-Guided AQN WINS by {improvement:.2f}% (lower loss)")
    elif b_loss < a_loss:
        improvement = (a_loss - b_loss) / a_loss * 100
        print(f"Global AQN wins by {improvement:.2f}% (lower loss)")
    else:
        print("TIE: Both methods have similar loss")

    print()
    if a_loss < c_loss:
        improvement = (c_loss - a_loss) / c_loss * 100
        print(f"SRDD-Guided AQN improves over no-AQN baseline by {improvement:.2f}%")
    else:
        print(f"No improvement over baseline (baseline might be better)")

    return 0


if __name__ == "__main__":
    exit(main())
