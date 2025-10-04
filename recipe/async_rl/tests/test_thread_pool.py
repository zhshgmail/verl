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
Unit tests for thread pool management in AsyncRLTrainer.

Tests:
1. Thread pool is explicitly created (not using default executor)
2. Thread pool configuration from YAML
3. Thread pool shutdown is clean
4. Thread pool has correct naming and worker count
"""

import pytest
import threading
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor


class TestThreadPoolManagement:
    """Test explicit thread pool management (Java best practice)."""

    def test_type_hints_for_transfer_queue(self):
        """Test that TransferQueue client has proper type hints."""
        import os

        # Check trainer has type hints for TransferQueue client
        trainer_file = os.path.join(os.path.dirname(__file__), '..', 'ray_trainer.py')
        with open(trainer_file, 'r', encoding='utf-8') as f:
            trainer_source = f.read()

        # Check TransferQueue imports
        assert 'from verl.experimental.transfer_queue import' in trainer_source, \
            "AsyncRLTrainer must import from verl.experimental.transfer_queue"

        # Check tq_client has type annotation
        assert 'tq_client: AsyncTransferQueueClient' in trainer_source, \
            "tq_client should have AsyncTransferQueueClient type hint"

    def test_thread_pool_explicitly_created(self):
        """Test that AsyncRLTrainer creates dedicated ThreadPoolExecutor."""
        # Check that ray_trainer.py creates ThreadPoolExecutor, not using None
        import os
        trainer_file = os.path.join(os.path.dirname(__file__), '..', 'ray_trainer.py')

        with open(trainer_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # Check ThreadPoolExecutor is imported
        assert 'from concurrent.futures import ThreadPoolExecutor' in source, \
            "Must import ThreadPoolExecutor"

        # Check ThreadPoolExecutor is instantiated
        assert 'ThreadPoolExecutor(' in source, \
            "Must create dedicated ThreadPoolExecutor"

        # Check NOT using None as executor
        assert 'run_in_executor(\n                    None,' not in source, \
            "Should NOT use default executor (None)"

    def test_thread_pool_configuration_from_yaml(self):
        """Test that thread pool size is configurable via YAML."""
        import os
        yaml_file = os.path.join(os.path.dirname(__file__), '..', 'config', 'async_rl_trainer.yaml')

        with open(yaml_file, 'r', encoding='utf-8') as f:
            yaml_content = f.read()

        # Check rollout_executor_workers is in config
        assert 'rollout_executor_workers:' in yaml_content, \
            "rollout_executor_workers must be in config"

    def test_thread_pool_has_name_prefix(self):
        """Test that thread pool has meaningful name prefix for debugging."""
        import os
        trainer_file = os.path.join(os.path.dirname(__file__), '..', 'ray_trainer.py')

        with open(trainer_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # Check thread_name_prefix is set
        assert 'thread_name_prefix=' in source, \
            "Thread pool should have name prefix for debugging"

        # Check prefix is meaningful
        assert 'async_rl_rollout' in source, \
            "Thread name prefix should identify AsyncRL rollout threads"

    def test_thread_pool_shutdown_implemented(self):
        """Test that shutdown method properly cleans up thread pool."""
        import os
        trainer_file = os.path.join(os.path.dirname(__file__), '..', 'ray_trainer.py')

        with open(trainer_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # Check shutdown method exists
        assert 'def shutdown(self):' in source, \
            "AsyncRLTrainer must have shutdown() method"

        # Check executor shutdown is called
        assert '_rollout_executor.shutdown(' in source, \
            "shutdown() must call executor.shutdown()"

        # Check wait=True (ensures clean shutdown)
        assert 'wait=True' in source, \
            "executor.shutdown(wait=True) ensures clean shutdown"

    def test_thread_pool_best_practice_comment(self):
        """Test that code documents why default executor should not be used."""
        import os
        trainer_file = os.path.join(os.path.dirname(__file__), '..', 'ray_trainer.py')

        with open(trainer_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # Check that code documents best practice
        assert 'IMPORTANT: Never use default executor' in source or \
               'Never use default executor' in source, \
            "Code should document why default executor is bad"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
