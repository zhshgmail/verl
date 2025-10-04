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
Unit tests for AsyncRLTrainer.

Tests:
1. __init__ signature matches parent RayPPOTrainer
2. sync_weights_to_rollout() doesn't call non-existent parent method
3. training_step() doesn't call non-existent parent method
4. PartialRolloutManager receives correct parameters

Note: These tests use source code inspection to avoid Ray initialization overhead.
"""

import pytest
import inspect
import ast


class TestAsyncRLTrainerSignature:
    """Test AsyncRLTrainer method signatures and parent compatibility."""

    def test_init_signature_matches_parent(self):
        """Test that AsyncRLTrainer.__init__ accepts all parent parameters."""
        # Read source file to avoid Ray initialization
        import os
        trainer_file = os.path.join(
            os.path.dirname(__file__), '..', 'ray_trainer.py'
        )

        with open(trainer_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # Parse AST to find __init__ method
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'AsyncRLTrainer':
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == '__init__':
                        params = [arg.arg for arg in method.args.args]

                        # Check required parameters
                        required_params = [
                            'self', 'config', 'tokenizer', 'role_worker_mapping',
                            'resource_pool_manager'
                        ]
                        for param in required_params:
                            assert param in params, f"Missing required parameter: {param}"

                        # Check optional parameters
                        optional_params = [
                            'ray_worker_group_cls', 'processor', 'reward_fn', 'val_reward_fn',
                            'train_dataset', 'val_dataset', 'collate_fn', 'train_sampler',
                            'device_name'
                        ]
                        for param in optional_params:
                            assert param in params, f"Missing optional parameter: {param}"
                        return

        raise AssertionError("Could not find AsyncRLTrainer.__init__ in source code")

    def test_sync_weights_no_super_call(self):
        """Test that sync_weights_to_rollout doesn't call non-existent parent method."""
        import os
        trainer_file = os.path.join(
            os.path.dirname(__file__), '..', 'ray_trainer.py'
        )

        with open(trainer_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # Check that it doesn't call super().sync_rollout_weights()
        assert 'super().sync_rollout_weights()' not in source, \
            "sync_weights_to_rollout should not call non-existent parent method"

    def test_training_step_no_super_call(self):
        """Test that training_step doesn't call non-existent parent method."""
        import os
        trainer_file = os.path.join(
            os.path.dirname(__file__), '..', 'ray_trainer.py'
        )

        with open(trainer_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # Check that it doesn't call super().training_step()
        assert 'super().training_step(' not in source, \
            "training_step should not call non-existent parent method"

    def test_training_step_implementation(self):
        """Test that training_step has its own implementation."""
        import os
        trainer_file = os.path.join(
            os.path.dirname(__file__), '..', 'ray_trainer.py'
        )

        with open(trainer_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # Find training_step method
        in_training_step = False
        for line in source.split('\n'):
            if 'def training_step(self' in line:
                in_training_step = True
            elif in_training_step and line.strip().startswith('def '):
                break  # Next method starts

            if in_training_step:
                # Check for actual implementation
                if 'metrics' in line or 'update_critic' in line or 'update_actor' in line:
                    return  # Found implementation

        raise AssertionError("training_step should have implementation")


class TestPartialRolloutManagerParameters:
    """Test PartialRolloutManager initialization."""

    def test_partial_rollout_manager_signature(self):
        """Test that PartialRolloutManager doesn't accept invalid parameters."""
        import os
        manager_file = os.path.join(
            os.path.dirname(__file__), '..', 'partial_rollout_manager.py'
        )

        with open(manager_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # Parse AST to find __init__ method
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'PartialRolloutManager':
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == '__init__':
                        params = [arg.arg for arg in method.args.args]

                        # Check valid parameters
                        assert 'config' in params
                        assert 'worker_group' in params
                        assert 'rm_wg' in params
                        assert 'weight_coordinator' in params

                        # Check invalid parameters are not present
                        assert 'new_tokens_per_chunk' not in params, \
                            "PartialRolloutManager should not accept new_tokens_per_chunk parameter"
                        return

        raise AssertionError("Could not find PartialRolloutManager.__init__ in source code")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
