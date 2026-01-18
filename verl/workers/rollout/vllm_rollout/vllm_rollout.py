# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import getpass
import logging
import os
from dataclasses import asdict
from types import MethodType
from typing import Any, Generator

import cloudpickle as pickle
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from torch.distributed.device_mesh import DeviceMesh
from vllm.config import LoRAConfig

from verl.utils.ray_utils import get_event_loop

try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    # https://github.com/vllm-project/vllm/commit/6a113d9aed8221a9c234535958e70e34ab6cac5b
    from vllm.v1.worker.worker_base import WorkerWrapperBase

from packaging import version as vs

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL, get_version
from verl.utils.device import is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_vllm_max_lora_rank,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


if is_version_ge(pkg="vllm", minver="0.7.3"):
    VLLMHijack.hijack()


def _check_vllm_version_for_sleep_level():
    # https://github.com/vllm-project/vllm/issues/25171
    minver = "0.11.0"
    current_version = get_version("vllm")
    if not current_version:
        logger.warning("Could not determine vLLM version, assuming an older version for sleep_level configuration.")
        return False
    return vs.parse(current_version) >= vs.parse(minver)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout(BaseRollout):
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase, which is engine in single worker process."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = self.model_config.tokenizer
        self.inference_engine: WorkerWrapperBase = None
        self.address = self._init_zeromq()
        self.lora_config = (
            {"max_loras": 1, "max_lora_rank": get_vllm_max_lora_rank(self.model_config.lora_rank)}
            if self.model_config.lora_rank > 0
            else {}
        )

        if config.layered_summon or (config.expert_parallel_size > 1 and not _check_vllm_version_for_sleep_level()):
            logger.warning("Setting the sleep level to 1 may cause a memory overflow.")
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        # Initialize noise injection config (AQN - Adaptive Quantization Noise)
        # Use getattr for OmegaConf compatibility
        epoch_ranges_raw = getattr(config, 'noise_injection_epoch_ranges', [])
        # Convert epoch_ranges from list of lists to list of tuples
        epoch_ranges = [tuple(r) for r in epoch_ranges_raw] if epoch_ranges_raw else []

        # AQN Fix: Cache for clean RMSNorm weights to restore before applying fresh noise each step.
        #
        # Problem: With LoRA, base weights are frozen so verl only syncs LoRA weights after step 0.
        # But AQN modifies RMSNorm weights in-place (module.weight.add_(noise)), and these changes
        # persist because base weights are never re-synced. This causes noise from step 0 to stay
        # permanently, breaking the sigma decay schedule.
        #
        # Solution: Cache clean RMSNorm weights at step 0, restore before applying fresh noise.
        # Memory cost is minimal (~0.2-1MB for RMSNorm weights only).
        #
        # Future optimization (Option 2): Use forward hooks instead of weight modification.
        # This would add noise to outputs during forward pass without polluting weights,
        # similar to hw_error_injection.py approach. This eliminates the need for caching entirely.
        self._aqn_clean_weights_cache = {}  # {module_name: clean_weight_tensor}
        self._aqn_weights_cached = False

        self.noise_injection_config = {
            'enabled': getattr(config, 'noise_injection_enabled', False),
            'sigma_trend': list(getattr(config, 'noise_injection_sigma_trend', [])),
            'total_steps': getattr(config, 'noise_injection_total_steps', 1000),
            'current_step': 0,  # Will be updated from trainer
            # Use None for target_modules/exclude_patterns to enable model type auto-detection
            # (Dense models: ALL RMSNorm, MoE models: post_attention_layernorm only)
            'target_modules': list(getattr(config, 'noise_injection_target_modules', [])) or None,
            'exclude_patterns': list(getattr(config, 'noise_injection_exclude_patterns', [])) or None,
            # Layer types to target: ['rmsnorm'], ['linear'], or ['rmsnorm', 'linear']
            'layer_types': list(getattr(config, 'noise_injection_layer_types', [])) or None,
            # Epoch-aware config (Option C)
            'epoch_aware': getattr(config, 'noise_injection_epoch_aware', False),
            'epoch_ranges': epoch_ranges,
            'stages_per_epoch': getattr(config, 'noise_injection_stages_per_epoch', 5),
            'steps_per_epoch': getattr(config, 'noise_injection_steps_per_epoch', 0),
            # SRDD-guided layer-specific sigma config (optional)
            'layer_sigma_config': self._parse_layer_sigma_config(getattr(config, 'noise_injection_layer_sigma_config', None)),
        }
        if self.noise_injection_config['enabled']:
            # Use print() to ensure visibility regardless of logging level
            targets_str = self.noise_injection_config['target_modules'] or 'AUTO (Dense=ALL, MoE=post_attn)'
            if self.noise_injection_config['epoch_aware']:
                print(f"[AQN] Epoch-aware noise injection enabled: "
                      f"{len(self.noise_injection_config['epoch_ranges'])} epochs, "
                      f"{self.noise_injection_config['stages_per_epoch']} stages/epoch, "
                      f"targets={targets_str}")
            else:
                print(f"[AQN] Noise injection enabled: {len(self.noise_injection_config['sigma_trend'])} stages, "
                      f"total_steps={self.noise_injection_config['total_steps']}, "
                      f"targets={targets_str}")

        # Initialize HW error injection config (simulate GPU/NPU heterogeneous errors)
        self.hw_error_injection_enabled = getattr(config, 'hw_error_injection_enabled', False)
        self.hw_error_injector = None
        if self.hw_error_injection_enabled:
            hw_config = getattr(config, 'hw_error_injection_config', {})
            # Convert OmegaConf to dict if needed
            if hasattr(hw_config, 'items'):
                hw_config = dict(hw_config)
            print(f"[HW Error] HW error injection enabled: {hw_config}")

    def _parse_layer_sigma_config(self, config):
        """Parse layer_sigma_config from OmegaConf or dict format."""
        if config is None:
            return None

        # Convert OmegaConf to dict if needed (recursive for nested configs)
        def _to_dict(obj):
            if hasattr(obj, 'items'):
                return {k: _to_dict(v) for k, v in dict(obj).items()}
            elif isinstance(obj, list):
                return [_to_dict(item) for item in obj]
            return obj

        config = _to_dict(config)

        if not isinstance(config, dict):
            return None

        # Ensure layer_multipliers is a regular dict with string keys
        if 'layer_multipliers' in config and isinstance(config['layer_multipliers'], dict):
            config['layer_multipliers'] = {str(k): v for k, v in config['layer_multipliers'].items()}

        # Parse zone_schedule if present (E14a)
        if 'zone_schedule' in config and isinstance(config['zone_schedule'], dict):
            zone_config = config['zone_schedule']
            # Ensure edge_layers and middle_layers are lists of ints
            if 'edge_layers' in zone_config:
                zone_config['edge_layers'] = [int(x) for x in zone_config['edge_layers']]
            if 'middle_layers' in zone_config:
                zone_config['middle_layers'] = [int(x) for x in zone_config['middle_layers']]

        return config

    def _aqn_cache_clean_weights(self, model):
        """Cache clean RMSNorm weights before applying AQN noise.

        This is called once during the first full weight sync (step 0).
        The cached weights are used to restore clean state before applying
        fresh noise in subsequent steps (when only LoRA weights are synced).

        Memory cost: ~0.2-1MB for typical models (RMSNorm weights only).
        """
        if self._aqn_weights_cached:
            return  # Already cached

        layer_types = self.noise_injection_config.get('layer_types') or ['rmsnorm']
        cached_count = 0

        for name, module in model.named_modules():
            # Check if this is a target layer type for AQN
            class_name = module.__class__.__name__.lower()
            is_target = False

            if 'rmsnorm' in layer_types and 'rmsnorm' in class_name:
                is_target = True
            if 'linear' in layer_types and 'linear' in class_name.lower():
                # Only cache Linear if explicitly targeted (unusual for AQN)
                is_target = True

            if is_target and hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                # Clone and store on same device (no CPU transfer to avoid latency)
                self._aqn_clean_weights_cache[name] = module.weight.data.clone()
                cached_count += 1

        self._aqn_weights_cached = True
        cache_size_bytes = sum(t.numel() * t.element_size() for t in self._aqn_clean_weights_cache.values())
        print(f"[AQN] Cached {cached_count} clean layer weights ({cache_size_bytes / 1024:.1f} KB) for restore-before-noise")

    def _aqn_restore_clean_weights(self, model):
        """Restore RMSNorm weights to clean state before applying fresh noise.

        This ensures each step gets fresh noise at the current sigma level,
        rather than accumulating noise from previous steps.
        """
        if not self._aqn_weights_cached:
            logger.warning("[AQN] Cannot restore - no clean weights cached")
            return

        restored_count = 0
        for name, module in model.named_modules():
            if name in self._aqn_clean_weights_cache:
                module.weight.data.copy_(self._aqn_clean_weights_cache[name])
                restored_count += 1

        if restored_count > 0:
            current_step = self.noise_injection_config.get('current_step', 0)
            print(f"[AQN-Fix] Restored {restored_count} layers to clean weights (step {current_step}, LoRA-only sync)")

    def _aqn_apply_noise(self, model):
        """Apply AQN noise to model with current sigma from decay schedule.

        This is the common noise application logic used by both:
        - Full weight sync path (after caching clean weights)
        - LoRA-only sync path (after restoring clean weights)
        """
        if not self.noise_injection_config.get('enabled', False):
            return

        is_validation = self.noise_injection_config.get('is_validation', False)
        if is_validation:
            print(f"[AQN] Skipping noise injection during validation (QeRL behavior)")
            return

        current_step = self.noise_injection_config.get('current_step', 0)
        total_steps = self.noise_injection_config.get('total_steps', 1000)
        epoch_aware = self.noise_injection_config.get('epoch_aware', False)

        if epoch_aware:
            # Epoch-aware mode: sigma decays within each epoch
            from verl.utils.noise_injection import get_sigma_by_step_epoch_aware
            epoch_ranges = self.noise_injection_config.get('epoch_ranges', [])
            stages_per_epoch = self.noise_injection_config.get('stages_per_epoch', 5)
            steps_per_epoch = self.noise_injection_config.get('steps_per_epoch', 1)

            if epoch_ranges and steps_per_epoch > 0:
                sigma_id, sigma = get_sigma_by_step_epoch_aware(
                    current_step, steps_per_epoch, epoch_ranges, stages_per_epoch
                )
                current_epoch = current_step // steps_per_epoch
                step_in_epoch = current_step % steps_per_epoch
                print(f"[AQN-EpochAware] step={current_step} (epoch {current_epoch+1}, step {step_in_epoch}), "
                      f"sigma_id={sigma_id}, sigma={sigma:.6f}")

                if sigma > 0:
                    from verl.utils.noise_injection import generate_expert_gaussian_noise
                    generate_expert_gaussian_noise(
                        model=model,
                        step=0,
                        total_step=1,
                        sigma_trend=[sigma],
                        target_modules=self.noise_injection_config.get('target_modules'),
                        exclude_patterns=self.noise_injection_config.get('exclude_patterns'),
                        layer_types=self.noise_injection_config.get('layer_types'),
                        layer_sigma_config=self.noise_injection_config.get('layer_sigma_config'),
                        verbose=True
                    )
        else:
            # Original mode: sigma decays globally across all steps
            from verl.utils.noise_injection import generate_expert_gaussian_noise, get_sigma_by_step
            sigma_trend = self.noise_injection_config.get('sigma_trend', [])

            if sigma_trend and total_steps > 0:
                sigma_id, sigma = get_sigma_by_step(current_step, total_steps, sigma_trend)
                print(f"[AQN] Applying noise injection: step={current_step}/{total_steps}, sigma_id={sigma_id}, sigma={sigma:.6f}")
                generate_expert_gaussian_noise(
                    model=model,
                    step=current_step,
                    total_step=total_steps,
                    sigma_trend=sigma_trend,
                    target_modules=self.noise_injection_config.get('target_modules'),
                    exclude_patterns=self.noise_injection_config.get('exclude_patterns'),
                    layer_types=self.noise_injection_config.get('layer_types'),
                    layer_sigma_config=self.noise_injection_config.get('layer_sigma_config'),
                    verbose=True
                )

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip = ray.util.get_node_ip_address().strip("[]")
                port, sock = get_free_port(ip)
                if is_valid_ipv6_address(ip):
                    address = f"tcp://[{ip}]:{port}"
                    self.socket.setsockopt(zmq.IPV6, 1)
                else:
                    address = f"tcp://{ip}:{port}"
            self.socket.bind(address)

        loop = get_event_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    async def _loop_forever(self):
        while True:
            try:
                message = await self.socket.recv()
                method, args, kwargs = pickle.loads(message)
                result = await self._execute_method(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")
                await self.socket.send(pickle.dumps(e))
                break

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        device_name = "NPU" if is_npu_available else "GPU"
        all_kwargs[0]["local_rank"] = (
            0
            if not ray_noset_visible_devices()
            else int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        )
        self.vllm_config = all_kwargs[0]["vllm_config"]
        if self.lora_config:
            lora_dtype = getattr(torch, self.config.dtype)
            self.vllm_config.lora_config = LoRAConfig(lora_dtype=lora_dtype, **self.lora_config)
        if self.config.quantization is not None:
            _SUPPORTED_QUANTIZATION = ["fp8", "torchao"]
            if self.config.quantization not in _SUPPORTED_QUANTIZATION:
                raise ValueError(
                    f"Currently only support {_SUPPORTED_QUANTIZATION} quantization, got: {self.config.quantization}"
                )

            if self.config.quantization == "fp8":
                # Apply vllm fp8 patches
                # Will remove the patch after vllm support on-the-fly quant for rollout natively.
                apply_vllm_fp8_patches()

        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine:
            self.inference_engine.wake_up(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            # In async mode, make sure the old lora is removed before adding the new one
            self.inference_engine.worker.remove_lora(VLLM_LORA_INT_ID)
            weights = dict(weights)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=asdict(peft_config),
                lora_tensors=weights,
            )
            self.inference_engine.worker.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")

            # AQN Fix: For LoRA-only sync, restore clean weights and apply fresh noise.
            # Without this fix, noise from step 0 would persist permanently because
            # base weights are never re-synced in LoRA mode.
            if hasattr(self, 'noise_injection_config') and self.noise_injection_config.get('enabled', False):
                model = self.inference_engine.worker.model_runner.model
                self._aqn_restore_clean_weights(model)
                self._aqn_apply_noise(model)
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model_runner = self.inference_engine.worker.model_runner
            model = model_runner.model
            patch_vllm_moe_model_weight_loader(model)

            # Add the FP8 related logic here as sharding manager has been deprecated.
            # Check if FP8 quantization is enabled and apply appropriate weight loading
            if is_fp8_model(model_runner.vllm_config):
                logger.info(f"FP8 model detected (async): {model_runner.vllm_config.quant_config}")
                # Convert bf16 weights to fp8 format before loading
                loaded_params = load_quanted_weights(weights, model_runner)
                logger.info(f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}")
            else:
                logger.info("Loading standard weights (non-FP8, async)")
                model.load_weights(weights)

            # AQN (Adaptive Quantization Noise): Cache clean weights, then apply noise.
            # For full weight sync (step 0 or non-LoRA), we cache clean weights first,
            # then apply noise. This cache is used by LoRA-only syncs to restore clean
            # weights before applying fresh noise at the current sigma level.
            if hasattr(self, 'noise_injection_config') and self.noise_injection_config.get('enabled', False):
                current_step = self.noise_injection_config.get('current_step', 0)
                print(f"[AQN-Fix] Full weight sync path (step {current_step}) - caching clean weights")
                self._aqn_cache_clean_weights(model)
                self._aqn_apply_noise(model)

            # HW ERROR INJECTION: Register hooks on model for simulating GPU/NPU errors
            # Supports deadzone injection for MXFP4 simulation (SRDD-guided AQN)
            if hasattr(self, 'hw_error_injection_enabled') and self.hw_error_injection_enabled:
                from verl.utils.hw_error_injection import HWErrorConfig, HWErrorInjector

                # Remove old hooks if they exist
                if self.hw_error_injector is not None:
                    self.hw_error_injector.remove_hooks()

                # Get config from rollout config
                hw_config_dict = getattr(self.config, 'hw_error_injection_config', {})
                if hasattr(hw_config_dict, 'items'):
                    hw_config_dict = dict(hw_config_dict)

                # Create injector and register hooks
                # v2.0: Support target_layers and deadzone_threshold for SRDD-guided AQN
                target_layers = hw_config_dict.get('target_layers', None)
                if target_layers is not None:
                    target_layers = list(target_layers)

                hw_config = HWErrorConfig(
                    enabled=True,
                    error_scale=hw_config_dict.get('error_scale', 1e-5),
                    error_type=hw_config_dict.get('error_type', 'relative_gaussian'),
                    injection_point=hw_config_dict.get('injection_point', 'output'),  # output for deadzone
                    target_modules=list(hw_config_dict.get('target_modules', ['rmsnorm'])),
                    exclude_modules=list(hw_config_dict.get('exclude_modules') or []) or None,
                    target_layers=target_layers,  # e.g., [15] for SRDD-guided injection
                    apply_during=hw_config_dict.get('apply_during', 'rollout'),
                    deadzone_threshold=hw_config_dict.get('deadzone_threshold', 0.01),
                    # STE mode: doesn't matter for inference (no backward pass), but keep for consistency
                    use_ste=hw_config_dict.get('use_ste', True),
                )
                self.hw_error_injector = HWErrorInjector(hw_config)
                self.hw_error_injector.set_phase('rollout')
                num_hooks = self.hw_error_injector.register_hooks(model, verbose=True)
                print(f"[HW Error] Registered {num_hooks} hooks on vLLM model after weight sync: {hw_config}")

    async def update_noise_injection_step(self, current_step: int, is_validation: bool = False):
        """Update current training step for noise injection schedule.

        Args:
            current_step: Current training step
            is_validation: If True, noise injection will be skipped during next weight sync
                          (QeRL only applies noise during training, not validation)
        """
        if hasattr(self, 'noise_injection_config'):
            self.noise_injection_config['current_step'] = current_step
            self.noise_injection_config['is_validation'] = is_validation
            logger.debug(f"Updated noise injection step to {current_step}, is_validation={is_validation}")

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode."""
        raise NotImplementedError

    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address
