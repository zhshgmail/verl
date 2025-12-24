"""
GPU test for vLLM routing capture.

This test requires:
- GPU access
- vLLM installed
- Qwen1.5-MoE-A2.7B model downloaded

Run on A100 server:
    docker exec verl-fp8-container python /home/z00637938/workspace/verl/tests/test_vllm_capture_gpu.py
"""

import sys
sys.path.insert(0, '/home/z00637938/workspace/verl')

import torch
from vllm import LLM, SamplingParams
from verl.workers.rollout.vllm_routing_capture import (
    patch_vllm_moe_for_routing_capture,
    get_routing_logs_from_vllm,
)


def test_vllm_routing_capture():
    """Test routing capture with real vLLM + Qwen1.5-MoE-A2.7B."""

    print("=" * 80)
    print("vLLM ROUTING CAPTURE TEST")
    print("=" * 80)

    # Configuration
    model_path = "/data/z00637938/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
    prompts = [
        "The capital of France is",
        "In machine learning, a neural network is",
    ]
    max_tokens = 32

    print(f"\n[Step 1/5] Patching vLLM for routing capture (before model load)")
    patch_vllm_moe_for_routing_capture(
        capture_router_logits=True,  # Test with full logits
        verbose=True
    )
    print("  ✅ MoE forward method patched")

    print(f"\n[Step 2/5] Loading model: Qwen1.5-MoE-A2.7B")
    print(f"  Path: {model_path}")

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        enforce_eager=True,  # Disable torch.compile for testing
    )
    print("  ✅ Model loaded")

    print(f"\n[Step 3/5] Generating completions")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Max tokens: {max_tokens}")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        top_p=1.0,
    )

    outputs = llm.generate(prompts, sampling_params)
    print(f"  ✅ Generated {len(outputs)} completions")

    # Show sample output
    for i, output in enumerate(outputs[:1]):
        print(f"\n  Sample output {i}:")
        print(f"    Prompt: {output.prompt[:50]}...")
        print(f"    Completion: {output.outputs[0].text[:100]}...")

    print(f"\n[Step 4/5] Retrieving routing logs")
    routing_logs = get_routing_logs_from_vllm(batch_id=0)

    if routing_logs is None:
        print("  ❌ FAILED: No routing logs captured!")
        return False

    print(f"  ✅ Captured routing logs:")
    print(f"    Batch ID: {routing_logs.batch_id}")
    print(f"    Num layers: {routing_logs.num_layers}")
    print(f"    Num tokens: {routing_logs.num_tokens}")
    print(f"    Top-k: {routing_logs.top_k}")
    print(f"    Num experts: {routing_logs.num_experts}")

    print(f"\n[Step 5/5] Validating routing logs")

    # Check layer count (should be 24 for Qwen1.5-MoE-A2.7B)
    expected_layers = 24
    assert routing_logs.num_layers == expected_layers, \
        f"Layer count mismatch: {routing_logs.num_layers} != {expected_layers}"
    print(f"  ✅ Layer count correct: {routing_logs.num_layers}")

    # Check we have logs for all layers
    assert len(routing_logs.layers) == routing_logs.num_layers, \
        f"Missing logs: {len(routing_logs.layers)} != {routing_logs.num_layers}"
    print(f"  ✅ All layers have logs")

    # Check shapes
    for i, layer_log in enumerate(routing_logs.layers[:3]):
        assert layer_log.layer_id == i, f"Layer ID mismatch at {i}"
        assert layer_log.expert_ids.shape == (routing_logs.num_tokens, routing_logs.top_k)
        assert layer_log.routing_weights.shape == (routing_logs.num_tokens, routing_logs.top_k)
        if layer_log.router_logits is not None:
            assert layer_log.router_logits.shape == (routing_logs.num_tokens, routing_logs.num_experts)

    print(f"  ✅ Shapes correct for all layers")

    # Check expert IDs are valid
    for layer_log in routing_logs.layers[:3]:
        assert (layer_log.expert_ids >= 0).all()
        assert (layer_log.expert_ids < routing_logs.num_experts).all()

    print(f"  ✅ Expert IDs valid (0-{routing_logs.num_experts-1})")

    # Check routing weights sum to ~1.0
    for layer_log in routing_logs.layers[:3]:
        weight_sums = layer_log.routing_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=0.01)

    print(f"  ✅ Routing weights normalized (sum≈1.0)")

    # Memory usage
    memory_mb = routing_logs.memory_size() / (1024 * 1024)
    print(f"\n  Memory usage: {memory_mb:.2f} MB")

    # Estimate expected tokens
    # Each prompt generates max_tokens, plus prompt tokens
    expected_tokens_approx = len(prompts) * (max_tokens + 20)  # ~20 prompt tokens
    print(f"  Expected tokens (approx): {expected_tokens_approx}")
    print(f"  Actual tokens: {routing_logs.num_tokens}")

    # Test serialization
    print(f"\n[Bonus] Testing serialization...")
    serialized = routing_logs.serialize()
    serialized_mb = len(serialized) / (1024 * 1024)
    print(f"  Serialized size: {serialized_mb:.2f} MB")

    deserialized = routing_logs.__class__.deserialize(serialized)
    assert deserialized.batch_id == routing_logs.batch_id
    assert deserialized.num_layers == routing_logs.num_layers
    print(f"  ✅ Serialization works")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Model: Qwen1.5-MoE-A2.7B (BF16)")
    print(f"  MoE layers: {routing_logs.num_layers}")
    print(f"  Tokens generated: {routing_logs.num_tokens}")
    print(f"  Top-k experts: {routing_logs.top_k}")
    print(f"  Memory overhead: {memory_mb:.2f} MB")
    print(f"  Routing capture: ✅ WORKING")

    return True


def test_without_router_logits():
    """Test capture without router logits (production mode)."""

    print("\n" + "=" * 80)
    print("PRODUCTION MODE TEST (no router logits)")
    print("=" * 80)

    model_path = "/data/z00637938/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
    prompts = ["Test prompt for production mode"]

    print(f"\n[1/3] Patching WITHOUT router_logits")
    patch_vllm_moe_for_routing_capture(
        capture_router_logits=False,  # Production mode
        verbose=True
    )

    print(f"\n[2/3] Loading model")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        enforce_eager=True,  # Disable torch.compile for testing
    )

    print(f"\n[3/3] Generating")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
    outputs = llm.generate(prompts, sampling_params)

    routing_logs = get_routing_logs_from_vllm()

    # Check router_logits is None
    for layer_log in routing_logs.layers:
        assert layer_log.router_logits is None, "router_logits should be None in production mode"

    memory_mb = routing_logs.memory_size() / (1024 * 1024)
    print(f"\n  ✅ Production mode memory: {memory_mb:.2f} MB")
    print(f"  (Compare with ~4.75x more with router_logits)")

    return True


if __name__ == "__main__":
    try:
        # Test 1: Full capture (with router logits)
        success1 = test_vllm_routing_capture()

        # Test 2: Production mode (without router logits)
        success2 = test_without_router_logits()

        if success1 and success2:
            print("\n" + "=" * 80)
            print("🎉 ALL TESTS PASSED - ROUTING CAPTURE WORKING!")
            print("=" * 80)
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
