# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for replica factory functions.

These tests verify:
1. get_async_rl_replica_class() factory function
2. Backward compatibility with enable_async_rl flag
3. SGLang NotImplementedError behavior
4. CPU-only (no Ray/GPU dependencies)
"""

import pytest
from unittest.mock import patch, MagicMock
import sys


class TestGetAsyncRLReplicaClass:
    """Test replica factory function."""

    def test_vllm_async_rl_enabled(self):
        """Test that vLLM with AsyncRL returns AsyncRLvLLMReplica."""
        # Mock the imports to avoid actual module loading
        with patch.dict('sys.modules', {
            'recipe.async_rl.vllm_engine_patches': MagicMock(),
            'recipe.async_rl.extended_vllm_server': MagicMock(),
        }):
            from recipe.async_rl.replica_factory import get_async_rl_replica_class

            # Create mock replica class
            mock_replica = type('AsyncRLvLLMReplica', (), {})
            sys.modules['recipe.async_rl.extended_vllm_server'].AsyncRLvLLMReplica = mock_replica

            replica_cls = get_async_rl_replica_class("vllm", enable_async_rl=True)

            assert replica_cls == mock_replica

    def test_vllm_async_rl_disabled(self):
        """Test that vLLM with AsyncRL disabled returns standard vLLMReplica."""
        # Mock the standard vLLM replica
        with patch.dict('sys.modules', {
            'verl.workers.rollout.vllm_rollout.vllm_async_server': MagicMock(),
        }):
            from recipe.async_rl.replica_factory import get_async_rl_replica_class

            # Create mock standard replica
            mock_standard_replica = type('vLLMReplica', (), {})
            sys.modules['verl.workers.rollout.vllm_rollout.vllm_async_server'].vLLMReplica = mock_standard_replica

            replica_cls = get_async_rl_replica_class("vllm", enable_async_rl=False)

            assert replica_cls == mock_standard_replica

    def test_sglang_async_rl_enabled_raises_error(self):
        """Test that SGLang with AsyncRL enabled raises NotImplementedError."""
        from recipe.async_rl.replica_factory import get_async_rl_replica_class

        with pytest.raises(NotImplementedError, match="AsyncRL is not implemented for SGLang"):
            get_async_rl_replica_class("sglang", enable_async_rl=True)

    def test_sglang_async_rl_disabled(self):
        """Test that SGLang with AsyncRL disabled returns standard SGLangReplica."""
        # Mock required modules for SGLang
        with patch.dict('sys.modules', {
            'vllm': MagicMock(),
            'verl.workers.rollout.sglang_rollout.http_server_engine': MagicMock(),
        }):
            from recipe.async_rl.replica_factory import get_async_rl_replica_class

            # Create mock SGLang replica
            mock_sglang_replica = type('SGLangReplica', (), {})
            sys.modules['verl.workers.rollout.sglang_rollout.http_server_engine'].SGLangReplica = mock_sglang_replica

            replica_cls = get_async_rl_replica_class("sglang", enable_async_rl=False)

            assert replica_cls == mock_sglang_replica

    def test_unknown_backend_raises_error(self):
        """Test that unknown backend raises ValueError."""
        from recipe.async_rl.replica_factory import get_async_rl_replica_class

        with pytest.raises(ValueError, match="Unknown rollout backend"):
            get_async_rl_replica_class("unknown_backend", enable_async_rl=False)

    def test_default_enable_async_rl(self):
        """Test default value of enable_async_rl parameter."""
        with patch.dict('sys.modules', {
            'recipe.async_rl.vllm_engine_patches': MagicMock(),
            'recipe.async_rl.extended_vllm_server': MagicMock(),
        }):
            from recipe.async_rl.replica_factory import get_async_rl_replica_class

            # Create mock replica class
            mock_replica = type('AsyncRLvLLMReplica', (), {})
            sys.modules['recipe.async_rl.extended_vllm_server'].AsyncRLvLLMReplica = mock_replica

            # Default should be True
            replica_cls = get_async_rl_replica_class("vllm")

            assert replica_cls == mock_replica


class TestPatchGetRolloutReplicaClass:
    """Test monkey-patching function."""

    def test_patch_modifies_verl_function(self):
        """Test that patch modifies verl's get_rollout_replica_class()."""
        # This test verifies the patch actually modifies the verl module
        # We need to import the actual module to test this
        import verl.workers.rollout.replica
        from recipe.async_rl.replica_factory import patch_get_rollout_replica_class

        # Save original function
        original_func = verl.workers.rollout.replica.get_rollout_replica_class

        try:
            # Apply patch
            patch_get_rollout_replica_class(enable_async_rl=True)

            # Check that function was replaced
            assert verl.workers.rollout.replica.get_rollout_replica_class != original_func
            # Check that original was saved
            assert hasattr(verl.workers.rollout.replica, '_original_get_rollout_replica_class')
            assert verl.workers.rollout.replica._original_get_rollout_replica_class == original_func
        finally:
            # Restore original function
            verl.workers.rollout.replica.get_rollout_replica_class = original_func
            if hasattr(verl.workers.rollout.replica, '_original_get_rollout_replica_class'):
                delattr(verl.workers.rollout.replica, '_original_get_rollout_replica_class')

    def test_patch_backward_compatible_signature(self):
        """Test that patched function maintains backward-compatible signature."""
        import verl.workers.rollout.replica
        from recipe.async_rl.replica_factory import patch_get_rollout_replica_class
        import inspect

        # Save original function
        original_func = verl.workers.rollout.replica.get_rollout_replica_class

        try:
            # Apply patch
            patch_get_rollout_replica_class(enable_async_rl=False)

            # Check signature: should accept single 'rollout' parameter
            patched_func = verl.workers.rollout.replica.get_rollout_replica_class
            sig = inspect.signature(patched_func)
            assert 'rollout' in sig.parameters
            # Should NOT require enable_async_rl (backward compatible)
            assert len(sig.parameters) == 1
        finally:
            # Restore original function
            verl.workers.rollout.replica.get_rollout_replica_class = original_func
            if hasattr(verl.workers.rollout.replica, '_original_get_rollout_replica_class'):
                delattr(verl.workers.rollout.replica, '_original_get_rollout_replica_class')


class TestSGLangErrorMessage:
    """Test that SGLang error message is helpful."""

    def test_sglang_error_message_content(self):
        """Test that SGLang NotImplementedError has helpful message."""
        from recipe.async_rl.replica_factory import get_async_rl_replica_class

        try:
            get_async_rl_replica_class("sglang", enable_async_rl=True)
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError as e:
            error_msg = str(e)

            # Check that error message contains key information
            assert "SGLang" in error_msg
            assert "vllm" in error_msg.lower()
            assert "enable_async_weight_updates" in error_msg or "enable_async_rl" in error_msg

            # Check that it suggests alternatives
            assert "rollout_backend: vllm" in error_msg or "use 'vllm'" in error_msg.lower()


class TestBackwardCompatibility:
    """Test backward compatibility with standard verl behavior."""

    def test_vllm_standard_mode_matches_verl(self):
        """Test that disable AsyncRL gives exact verl behavior for vLLM."""
        with patch.dict('sys.modules', {
            'verl.workers.rollout.vllm_rollout.vllm_async_server': MagicMock(),
        }):
            from recipe.async_rl.replica_factory import get_async_rl_replica_class

            # Create mock standard replica (what verl returns)
            mock_verl_replica = type('vLLMReplica', (), {})
            sys.modules['verl.workers.rollout.vllm_rollout.vllm_async_server'].vLLMReplica = mock_verl_replica

            # Our factory with AsyncRL disabled should return same class
            replica_cls = get_async_rl_replica_class("vllm", enable_async_rl=False)

            assert replica_cls == mock_verl_replica

    def test_sglang_standard_mode_sets_cpu_engine(self):
        """Test that SGLang mode sets SGLANG_USE_CPU_ENGINE environment variable."""
        import os

        with patch.dict('sys.modules', {
            'vllm': MagicMock(),
            'verl.workers.rollout.sglang_rollout.http_server_engine': MagicMock(),
        }):
            from recipe.async_rl.replica_factory import get_async_rl_replica_class

            # Create mock SGLang replica
            mock_sglang_replica = type('SGLangReplica', (), {})
            sys.modules['verl.workers.rollout.sglang_rollout.http_server_engine'].SGLangReplica = mock_sglang_replica

            # Clear environment variable first
            if 'SGLANG_USE_CPU_ENGINE' in os.environ:
                del os.environ['SGLANG_USE_CPU_ENGINE']

            replica_cls = get_async_rl_replica_class("sglang", enable_async_rl=False)

            # Should set environment variable (verl compatibility)
            assert os.environ.get('SGLANG_USE_CPU_ENGINE') == "1"

    def test_sglang_vllm_import_error_fallback(self):
        """Test that SGLang handles vLLM ImportError by mocking it."""
        import os

        # Remove vllm from sys.modules if it exists
        vllm_existed = 'vllm' in sys.modules
        if vllm_existed:
            vllm_backup = sys.modules['vllm']
            del sys.modules['vllm']

        try:
            with patch.dict('sys.modules', {
                'verl.workers.rollout.sglang_rollout.http_server_engine': MagicMock(),
            }):
                # Clear vllm to force ImportError
                if 'vllm' in sys.modules:
                    del sys.modules['vllm']

                from recipe.async_rl.replica_factory import get_async_rl_replica_class

                # Create mock SGLang replica
                mock_sglang_replica = type('SGLangReplica', (), {})
                sys.modules['verl.workers.rollout.sglang_rollout.http_server_engine'].SGLangReplica = mock_sglang_replica

                os.environ['SGLANG_USE_CPU_ENGINE'] = "1"

                # This should trigger the ImportError path and create mock vllm
                replica_cls = get_async_rl_replica_class("sglang", enable_async_rl=False)

                # Should have created mock vllm module
                assert 'vllm' in sys.modules
                assert hasattr(sys.modules['vllm'], '_custom_ops')

                assert replica_cls == mock_sglang_replica
        finally:
            # Restore vllm if it existed
            if vllm_existed:
                sys.modules['vllm'] = vllm_backup
            elif 'vllm' in sys.modules:
                del sys.modules['vllm']


class TestPatchConfigReading:
    """Test patch_get_rollout_replica_class config reading paths."""

    def test_patch_config_exception_fallback(self):
        """Test that patch handles config read exceptions gracefully."""
        import verl.workers.rollout.replica
        from recipe.async_rl.replica_factory import patch_get_rollout_replica_class

        # Save original function
        original_func = verl.workers.rollout.replica.get_rollout_replica_class

        try:
            # Apply patch - should handle exception and default to False
            patch_get_rollout_replica_class(enable_async_rl=None)

            # Get the patched function
            patched_func = verl.workers.rollout.replica.get_rollout_replica_class

            # Mock the replica_factory to avoid actual imports
            with patch.dict('sys.modules', {
                'verl.workers.rollout.vllm_rollout.vllm_async_server': MagicMock(),
            }):
                mock_replica = type('vLLMReplica', (), {})
                sys.modules['verl.workers.rollout.vllm_rollout.vllm_async_server'].vLLMReplica = mock_replica

                # Call patched function - should use default False since config reading failed
                replica_cls = patched_func("vllm")

                # Should return standard replica (AsyncRL disabled by default)
                assert replica_cls == mock_replica
        finally:
            # Restore original function
            verl.workers.rollout.replica.get_rollout_replica_class = original_func
            if hasattr(verl.workers.rollout.replica, '_original_get_rollout_replica_class'):
                delattr(verl.workers.rollout.replica, '_original_get_rollout_replica_class')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
