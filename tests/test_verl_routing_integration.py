"""
Integration test for routing capture within veRL's vLLM worker.

⚠️  IMPORTANT: This test uses LLM() API (standalone mode) which does NOT match
    veRL's actual usage pattern. With vLLM V1 multiprocessing, this test will FAIL
    because the class registry is not accessible across process boundaries.

    veRL's actual usage (which WORKS):
    - _load_model() runs IN worker process
    - Has direct access to class registry
    - No cross-process communication needed

    This test is kept for reference but is expected to fail with vLLM V1.
    Real validation should be done with actual veRL training on GPU.

Run on A100 server (for reference only, will fail with vLLM V1):
    docker exec verl-fp8-container python /home/z00637938/workspace/verl/tests/test_verl_routing_integration.py
"""

import sys
sys.path.insert(0, '/home/z00637938/workspace/verl')

import os
import torch
from typing import Any

print("=" * 80)
print("veRL ROUTING INTEGRATION TEST")
print("=" * 80)

# Step 1: Initialize environment
print("\n[Step 1/6] Setting up environment")
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
print("  ✅ Environment configured")

# Step 2: Import veRL and vLLM
print("\n[Step 2/6] Importing veRL and vLLM")
try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    from vllm.v1.worker.worker_base import WorkerWrapperBase

from vllm import LLM, SamplingParams

from verl.utils.vllm.patch import patch_vllm_for_routing_capture
from verl.workers.rollout.vllm_routing_capture import get_routing_logs_from_vllm

print("  ✅ Imports successful")

# Step 3: Create vLLM engine with routing capture
print("\n[Step 3/6] Creating vLLM engine")
model_path = "/data/z00637938/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"

# Enable routing capture BEFORE creating engine
print("  [3a] Enabling routing capture patch")
num_layers = patch_vllm_for_routing_capture(
    enable_capture=True,
    capture_router_logits=True
)
print(f"  ✅ Routing capture enabled (patched forward method)")

# Create LLM engine (simpler than WorkerWrapperBase for testing)
print("  [3b] Creating LLM engine")
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.5,
    enforce_eager=True,  # Disable torch.compile for testing
)
print("  ✅ LLM engine created")

# Step 4: Generate sequences
print("\n[Step 4/6] Generating sequences")
prompts = [
    "The capital of France is",
    "In machine learning, a neural network is",
]
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=32,
    top_p=1.0,
)

outputs = llm.generate(prompts, sampling_params)
print(f"  ✅ Generated {len(outputs)} completions")

# Show sample output
for i, output in enumerate(outputs[:1]):
    print(f"\n  Sample output {i}:")
    print(f"    Prompt: {output.prompt[:50]}...")
    print(f"    Completion: {output.outputs[0].text[:80]}...")

# Step 5: Retrieve routing logs
print("\n[Step 5/6] Retrieving routing logs")
routing_logs = get_routing_logs_from_vllm(batch_id=0, clear_buffer=True)

if routing_logs is None:
    print("  ❌ FAILED: No routing logs captured!")
    sys.exit(1)

print(f"  ✅ Captured routing logs:")
print(f"    Batch ID: {routing_logs.batch_id}")
print(f"    Num layers: {routing_logs.num_layers}")
print(f"    Num tokens: {routing_logs.num_tokens}")
print(f"    Top-k: {routing_logs.top_k}")
print(f"    Num experts: {routing_logs.num_experts}")

# Step 6: Validate data format
print("\n[Step 6/6] Validating routing logs")

# Check layer count (should be 24 for Qwen1.5-MoE-A2.7B)
expected_layers = 24
assert routing_logs.num_layers == expected_layers, \
    f"Layer count mismatch: {routing_logs.num_layers} != {expected_layers}"
print(f"  ✅ Layer count correct: {routing_logs.num_layers}")

# Check we have logs for all layers
assert len(routing_logs.layers) == routing_logs.num_layers, \
    f"Missing logs: {len(routing_logs.layers)} != {routing_logs.num_layers}"
print(f"  ✅ All layers have logs")

# Check shapes for each layer
print(f"  [6a] Validating layer shapes")
for i, layer_log in enumerate(routing_logs.layers):
    assert layer_log.layer_id == i, f"Layer ID mismatch at {i}"

    expected_shape_ids = (routing_logs.num_tokens, routing_logs.top_k)
    expected_shape_weights = (routing_logs.num_tokens, routing_logs.top_k)

    assert layer_log.expert_ids.shape == expected_shape_ids, \
        f"Layer {i}: expert_ids shape {layer_log.expert_ids.shape} != {expected_shape_ids}"
    assert layer_log.routing_weights.shape == expected_shape_weights, \
        f"Layer {i}: routing_weights shape {layer_log.routing_weights.shape} != {expected_shape_weights}"

    if layer_log.router_logits is not None:
        expected_shape_logits = (routing_logs.num_tokens, routing_logs.num_experts)
        assert layer_log.router_logits.shape == expected_shape_logits, \
            f"Layer {i}: router_logits shape {layer_log.router_logits.shape} != {expected_shape_logits}"

print(f"  ✅ Shapes correct for all {routing_logs.num_layers} layers")

# Check expert IDs are valid
print(f"  [6b] Validating expert IDs")
for layer_log in routing_logs.layers:
    assert (layer_log.expert_ids >= 0).all(), "Found negative expert ID"
    assert (layer_log.expert_ids < routing_logs.num_experts).all(), \
        f"Found expert ID >= {routing_logs.num_experts}"
print(f"  ✅ Expert IDs valid (0-{routing_logs.num_experts-1})")

# Check routing weights sum to ~1.0
print(f"  [6c] Validating routing weights")
for layer_log in routing_logs.layers[:3]:  # Check first 3 layers
    weight_sums = layer_log.routing_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=0.01), \
        "Routing weights do not sum to 1.0"
print(f"  ✅ Routing weights normalized (sum≈1.0)")

# Check router logits present
print(f"  [6d] Validating router logits")
has_logits = all(layer.router_logits is not None for layer in routing_logs.layers)
assert has_logits, "Missing router_logits in some layers"
print(f"  ✅ Router logits captured for all layers")

# Memory usage
memory_mb = routing_logs.memory_size() / (1024 * 1024)
print(f"\n  Memory usage: {memory_mb:.2f} MB")

# Test serialization
print(f"\n  [6e] Testing serialization")
serialized = routing_logs.serialize()
serialized_mb = len(serialized) / (1024 * 1024)
print(f"  Serialized size: {serialized_mb:.2f} MB")

deserialized = routing_logs.__class__.deserialize(serialized)
assert deserialized.batch_id == routing_logs.batch_id
assert deserialized.num_layers == routing_logs.num_layers
assert deserialized.num_tokens == routing_logs.num_tokens
print(f"  ✅ Serialization works")

# Step 7: Test with multiple generations (routing log clearing)
print("\n[Step 7/7] Testing routing log cleanup")
print("  [7a] Generating second batch")
outputs2 = llm.generate(["Test prompt"], SamplingParams(temperature=0.0, max_tokens=16))
print("  ✅ Second batch generated")

print("  [7b] Retrieving second batch logs")
routing_logs2 = get_routing_logs_from_vllm(batch_id=1, clear_buffer=True)
assert routing_logs2 is not None, "Second batch logs not captured"
print(f"  ✅ Second batch logs captured ({routing_logs2.num_tokens} tokens)")

# Verify token counts are different (different prompt lengths)
assert routing_logs.num_tokens != routing_logs2.num_tokens, \
    "Token counts should differ between batches"
print(f"  ✅ Batch separation working (batch1: {routing_logs.num_tokens} tokens, batch2: {routing_logs2.num_tokens} tokens)")

# Summary
print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED - veRL ROUTING INTEGRATION WORKING!")
print("=" * 80)
print(f"\nSummary:")
print(f"  Model: Qwen1.5-MoE-A2.7B (BF16)")
print(f"  MoE layers: {routing_logs.num_layers}")
print(f"  Tokens per batch: {routing_logs.num_tokens} (batch 1), {routing_logs2.num_tokens} (batch 2)")
print(f"  Top-k experts: {routing_logs.top_k}")
print(f"  Total experts: {routing_logs.num_experts}")
print(f"  Memory overhead: {memory_mb:.2f} MB/batch")
print(f"  Router logits: ✅ CAPTURED")
print(f"  Serialization: ✅ WORKING")
print(f"  Multi-batch: ✅ WORKING")
print(f"\n✅ Ready for Phase 3 (Megatron playback)!")

sys.exit(0)
