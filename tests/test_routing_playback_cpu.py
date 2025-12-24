"""
CPU-only unit tests for routing playback.

These tests can run on CPU without GPU, useful for quick verification.
"""

import pytest
import torch
import pickle
from verl.utils.routing_playback import RoutingLog, BatchRoutingLogs, merge_routing_logs


class TestRoutingLog:
    """Test RoutingLog data structure."""

    def test_creation_basic(self):
        """Test basic RoutingLog creation."""
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
        assert log.router_logits is None

    def test_creation_with_logits(self):
        """Test RoutingLog creation with router logits."""
        num_tokens = 100
        top_k = 8
        num_experts = 60

        log = RoutingLog(
            layer_id=0,
            expert_ids=torch.randint(0, num_experts, (num_tokens, top_k)),
            routing_weights=torch.rand(num_tokens, top_k),
            router_logits=torch.randn(num_tokens, num_experts),
        )

        assert log.router_logits.shape == (num_tokens, num_experts)

    def test_shape_validation(self):
        """Test that shape mismatches are caught."""
        num_tokens = 100
        top_k = 8

        # Mismatched routing_weights shape
        with pytest.raises(AssertionError):
            RoutingLog(
                layer_id=0,
                expert_ids=torch.randint(0, 60, (num_tokens, top_k)),
                routing_weights=torch.rand(num_tokens, top_k + 1),  # Wrong shape
            )

        # Mismatched router_logits shape
        with pytest.raises(AssertionError):
            RoutingLog(
                layer_id=0,
                expert_ids=torch.randint(0, 60, (num_tokens, top_k)),
                routing_weights=torch.rand(num_tokens, top_k),
                router_logits=torch.randn(num_tokens + 1, 60),  # Wrong shape
            )

    def test_device_transfer(self):
        """Test moving RoutingLog to different device."""
        log = RoutingLog(
            layer_id=0,
            expert_ids=torch.randint(0, 60, (100, 8)),
            routing_weights=torch.rand(100, 8),
            router_logits=torch.randn(100, 60),
        )

        # Test CPU (no-op on CPU-only system)
        log_cpu = log.cpu()
        assert log_cpu.expert_ids.device == torch.device('cpu')

        # Test to() with CPU device
        log_cpu2 = log.to(torch.device('cpu'))
        assert torch.equal(log_cpu.expert_ids, log_cpu2.expert_ids)

    def test_memory_size(self):
        """Test memory size estimation."""
        num_tokens = 1000
        top_k = 8
        num_experts = 60

        # Without router_logits
        log_small = RoutingLog(
            layer_id=0,
            expert_ids=torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32),
            routing_weights=torch.rand(num_tokens, top_k, dtype=torch.float32),
        )

        # expert_ids: 1000 * 8 * 4 = 32KB
        # routing_weights: 1000 * 8 * 4 = 32KB
        # Total: 64KB
        expected_small = 1000 * 8 * 4 + 1000 * 8 * 4
        assert log_small.memory_size() == expected_small

        # With router_logits
        log_large = RoutingLog(
            layer_id=0,
            expert_ids=torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32),
            routing_weights=torch.rand(num_tokens, top_k, dtype=torch.float32),
            router_logits=torch.randn(num_tokens, num_experts, dtype=torch.float32),
        )

        # Additional: 1000 * 60 * 4 = 240KB
        expected_large = expected_small + 1000 * 60 * 4
        assert log_large.memory_size() == expected_large

        print(f"Memory - Small (no logits): {log_small.memory_size() / 1024:.2f} KB")
        print(f"Memory - Large (with logits): {log_large.memory_size() / 1024:.2f} KB")


class TestBatchRoutingLogs:
    """Test BatchRoutingLogs data structure."""

    def test_creation(self):
        """Test basic BatchRoutingLogs creation."""
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

    def test_layer_id_validation(self):
        """Test that layer_id consistency is enforced."""
        layers = [
            RoutingLog(
                layer_id=0,
                expert_ids=torch.randint(0, 60, (100, 8)),
                routing_weights=torch.rand(100, 8),
            ),
            RoutingLog(
                layer_id=2,  # Wrong! Should be 1
                expert_ids=torch.randint(0, 60, (100, 8)),
                routing_weights=torch.rand(100, 8),
            ),
        ]

        with pytest.raises(AssertionError):
            BatchRoutingLogs(
                batch_id=0,
                num_layers=2,
                num_tokens=100,
                top_k=8,
                num_experts=60,
                layers=layers,
            )

    def test_serialization(self):
        """Test serialization and deserialization."""
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
        print(f"Serialized size: {len(serialized) / 1024:.2f} KB")

        # Deserialize
        deserialized = BatchRoutingLogs.deserialize(serialized)

        # Verify
        assert deserialized.batch_id == 42
        assert deserialized.num_layers == num_layers
        assert len(deserialized.layers) == num_layers

        for i in range(num_layers):
            assert torch.equal(deserialized.layers[i].expert_ids, batch_logs.layers[i].expert_ids)
            assert torch.allclose(deserialized.layers[i].routing_weights, batch_logs.layers[i].routing_weights)

    def test_memory_size(self):
        """Test total memory size calculation."""
        num_layers = 24
        num_tokens = 16 * 512  # batch_size=16, seq_len=512
        top_k = 8
        num_experts = 60

        # Without router_logits
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

        # Per layer: (num_tokens * top_k) * 4 * 2 = 8192 * 8 * 8 = 524KB
        # Total: 524KB * 24 = 12.6 MB
        size_small_mb = batch_logs_small.memory_size() / (1024 * 1024)
        print(f"Batch memory (no logits): {size_small_mb:.2f} MB")
        assert size_small_mb < 20  # Should be < 20 MB

        # With router_logits
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
        print(f"Batch memory (with logits): {size_large_mb:.2f} MB")

        # Ratio should be ~10x (router_logits are much larger)
        ratio = size_large_mb / size_small_mb
        print(f"Size ratio (with/without logits): {ratio:.2f}x")
        assert ratio > 5  # Should be significantly larger


class TestMergeRoutingLogs:
    """Test merging multiple RoutingLog captures."""

    def test_merge_single(self):
        """Test that merging single log returns same log."""
        log = RoutingLog(
            layer_id=0,
            expert_ids=torch.randint(0, 60, (100, 8)),
            routing_weights=torch.rand(100, 8),
        )

        merged = merge_routing_logs([log])
        assert torch.equal(merged.expert_ids, log.expert_ids)

    def test_merge_multiple(self):
        """Test merging multiple logs from same layer."""
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

        # Check shapes
        assert merged.expert_ids.shape == (100 + 50 + 75, 8)
        assert merged.routing_weights.shape == (100 + 50 + 75, 8)
        assert merged.router_logits.shape == (100 + 50 + 75, 60)

        # Check content (first 100 tokens should match first log)
        assert torch.equal(merged.expert_ids[:100], logs[0].expert_ids)

    def test_merge_different_layers_fails(self):
        """Test that merging logs from different layers fails."""
        logs = [
            RoutingLog(
                layer_id=0,
                expert_ids=torch.randint(0, 60, (100, 8)),
                routing_weights=torch.rand(100, 8),
            ),
            RoutingLog(
                layer_id=1,  # Different layer
                expert_ids=torch.randint(0, 60, (100, 8)),
                routing_weights=torch.rand(100, 8),
            ),
        ]

        with pytest.raises(AssertionError):
            merge_routing_logs(logs)

    def test_merge_empty_fails(self):
        """Test that merging empty list fails."""
        with pytest.raises(ValueError):
            merge_routing_logs([])


if __name__ == "__main__":
    # Can run directly for quick testing
    import sys

    print("Running routing playback CPU tests...")
    print("=" * 80)

    # Run with pytest if available, otherwise run manually
    try:
        pytest.main([__file__, "-v", "-s"])
    except SystemExit:
        pass
