"""
Simple CPU tests for routing playback without pytest dependency.
"""

import sys
sys.path.insert(0, '/home/zheng/workspace/verl')

import torch
from verl.utils.routing_playback import RoutingLog, BatchRoutingLogs, merge_routing_logs


def test_routing_log_creation():
    """Test basic RoutingLog creation."""
    print("\n[Test 1] RoutingLog creation...")

    num_tokens = 100
    top_k = 8

    log = RoutingLog(
        layer_id=0,
        expert_ids=torch.randint(0, 60, (num_tokens, top_k)),
        routing_weights=torch.rand(num_tokens, top_k),
    )

    assert log.layer_id == 0
    assert log.expert_ids.shape == (num_tokens, top_k)
    assert log.routing_weights.shape == (num_tokens, top_k)
    print(f"  ✅ Created RoutingLog: {num_tokens} tokens, top-{top_k}")


def test_routing_log_memory():
    """Test memory size estimation."""
    print("\n[Test 2] Memory size estimation...")

    num_tokens = 1000
    top_k = 8
    num_experts = 60

    # Without router_logits
    log_small = RoutingLog(
        layer_id=0,
        expert_ids=torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32),
        routing_weights=torch.rand(num_tokens, top_k, dtype=torch.float32),
    )

    size_small_kb = log_small.memory_size() / 1024
    print(f"  ✅ Memory (no logits): {size_small_kb:.2f} KB")

    # With router_logits
    log_large = RoutingLog(
        layer_id=0,
        expert_ids=torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32),
        routing_weights=torch.rand(num_tokens, top_k, dtype=torch.float32),
        router_logits=torch.randn(num_tokens, num_experts, dtype=torch.float32),
    )

    size_large_kb = log_large.memory_size() / 1024
    print(f"  ✅ Memory (with logits): {size_large_kb:.2f} KB")
    print(f"  ✅ Ratio: {size_large_kb / size_small_kb:.2f}x")


def test_batch_routing_logs():
    """Test BatchRoutingLogs creation."""
    print("\n[Test 3] BatchRoutingLogs creation...")

    num_layers = 24
    num_tokens = 100
    top_k = 8
    num_experts = 60

    layers = []
    for layer_id in range(num_layers):
        layers.append(
            RoutingLog(
                layer_id=layer_id,
                expert_ids=torch.randint(0, num_experts, (num_tokens, top_k)),
                routing_weights=torch.rand(num_tokens, top_k),
            )
        )

    batch_logs = BatchRoutingLogs(
        batch_id=0,
        num_layers=num_layers,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        layers=layers,
    )

    assert batch_logs.num_layers == num_layers
    assert len(batch_logs.layers) == num_layers
    print(f"  ✅ Created BatchRoutingLogs: {num_layers} layers, {num_tokens} tokens")


def test_serialization():
    """Test serialization and deserialization."""
    print("\n[Test 4] Serialization...")

    num_layers = 3
    num_tokens = 100
    top_k = 8

    layers = []
    for layer_id in range(num_layers):
        layers.append(
            RoutingLog(
                layer_id=layer_id,
                expert_ids=torch.randint(0, 60, (num_tokens, top_k)),
                routing_weights=torch.rand(num_tokens, top_k),
                router_logits=torch.randn(num_tokens, 60),
            )
        )

    batch_logs = BatchRoutingLogs(
        batch_id=42,
        num_layers=num_layers,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=60,
        layers=layers,
    )

    # Serialize
    serialized = batch_logs.serialize()
    size_kb = len(serialized) / 1024
    print(f"  ✅ Serialized size: {size_kb:.2f} KB")

    # Deserialize
    deserialized = BatchRoutingLogs.deserialize(serialized)

    # Verify
    assert deserialized.batch_id == 42
    assert deserialized.num_layers == num_layers

    for i in range(num_layers):
        assert torch.equal(deserialized.layers[i].expert_ids, batch_logs.layers[i].expert_ids)

    print(f"  ✅ Deserialized and verified")


def test_realistic_batch():
    """Test with realistic batch size."""
    print("\n[Test 5] Realistic batch size (batch=32, seq=512, layers=24)...")

    num_layers = 24
    batch_size = 32
    seq_len = 512
    num_tokens = batch_size * seq_len
    top_k = 8
    num_experts = 60

    # Without router_logits (production mode)
    layers_small = []
    for layer_id in range(num_layers):
        layers_small.append(
            RoutingLog(
                layer_id=layer_id,
                expert_ids=torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32),
                routing_weights=torch.rand(num_tokens, top_k, dtype=torch.float32),
            )
        )

    batch_logs_small = BatchRoutingLogs(
        batch_id=0,
        num_layers=num_layers,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        layers=layers_small,
    )

    size_small_mb = batch_logs_small.memory_size() / (1024 * 1024)
    print(f"  ✅ Memory (no logits): {size_small_mb:.2f} MB")

    # With router_logits (training mode with router LoRA)
    layers_large = []
    for layer_id in range(num_layers):
        layers_large.append(
            RoutingLog(
                layer_id=layer_id,
                expert_ids=torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32),
                routing_weights=torch.rand(num_tokens, top_k, dtype=torch.float32),
                router_logits=torch.randn(num_tokens, num_experts, dtype=torch.float32),
            )
        )

    batch_logs_large = BatchRoutingLogs(
        batch_id=0,
        num_layers=num_layers,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        layers=layers_large,
    )

    size_large_mb = batch_logs_large.memory_size() / (1024 * 1024)
    print(f"  ✅ Memory (with logits): {size_large_mb:.2f} MB")
    print(f"  ✅ Ratio: {size_large_mb / size_small_mb:.2f}x")


def test_merge():
    """Test merging routing logs."""
    print("\n[Test 6] Merging routing logs...")

    logs = [
        RoutingLog(
            layer_id=5,
            expert_ids=torch.randint(0, 60, (100, 8)),
            routing_weights=torch.rand(100, 8),
            router_logits=torch.randn(100, 60),
        ),
        RoutingLog(
            layer_id=5,
            expert_ids=torch.randint(0, 60, (50, 8)),
            routing_weights=torch.rand(50, 8),
            router_logits=torch.randn(50, 60),
        ),
        RoutingLog(
            layer_id=5,
            expert_ids=torch.randint(0, 60, (75, 8)),
            routing_weights=torch.rand(75, 8),
            router_logits=torch.randn(75, 60),
        ),
    ]

    merged = merge_routing_logs(logs)

    assert merged.expert_ids.shape == (100 + 50 + 75, 8)
    assert merged.routing_weights.shape == (100 + 50 + 75, 8)
    assert merged.router_logits.shape == (100 + 50 + 75, 60)

    print(f"  ✅ Merged 3 logs: {logs[0].expert_ids.shape[0]} + {logs[1].expert_ids.shape[0]} + {logs[2].expert_ids.shape[0]} = {merged.expert_ids.shape[0]} tokens")


def main():
    print("=" * 80)
    print("ROUTING PLAYBACK CPU TESTS")
    print("=" * 80)

    try:
        test_routing_log_creation()
        test_routing_log_memory()
        test_batch_routing_logs()
        test_serialization()
        test_realistic_batch()
        test_merge()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
