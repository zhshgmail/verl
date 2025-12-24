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
Routing Playback Format Converter

Converts vLLM's BatchRoutingLogs format to Megatron's tensor format for R3 mode.
"""

from typing import List, Optional

import torch

from verl.utils.routing_playback import BatchRoutingLogs


def convert_routing_logs_to_tensor(
    routing_logs_list: List[Optional[BatchRoutingLogs]],
    max_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """
    Convert a list of BatchRoutingLogs (one per sequence) to Megatron format tensor.

    Args:
        routing_logs_list: List of BatchRoutingLogs, one per sequence in batch.
                          None entries indicate sequences without routing logs (non-MoE models).
        max_seq_len: Maximum sequence length to pad to. If None, uses the longest sequence.

    Returns:
        torch.Tensor: Shape [bs, max_seq_len, num_layers, topk]
                     Contains expert indices for routing replay.

    Raises:
        ValueError: If routing logs are inconsistent (different num_layers, top_k, etc.)

    Example:
        >>> # Batch of 4 sequences
        >>> routing_logs = [logs1, logs2, logs3, logs4]
        >>> tensor = convert_routing_logs_to_tensor(routing_logs)
        >>> assert tensor.shape == (4, max_len, num_layers, topk)
    """
    # Filter out None entries
    valid_logs = [log for log in routing_logs_list if log is not None]

    if not valid_logs:
        raise ValueError("No valid routing logs in batch")

    batch_size = len(routing_logs_list)

    # Validate consistency across batch
    ref_log = valid_logs[0]
    num_layers = ref_log.num_layers
    top_k = ref_log.top_k

    for i, log in enumerate(valid_logs):
        if log.num_layers != num_layers:
            raise ValueError(
                f"Inconsistent num_layers: sequence 0 has {num_layers}, "
                f"but sequence {i} has {log.num_layers}"
            )
        if log.top_k != top_k:
            raise ValueError(
                f"Inconsistent top_k: sequence 0 has {top_k}, " f"but sequence {i} has {log.top_k}"
            )
        if len(log.layers) != num_layers:
            raise ValueError(
                f"Sequence {i}: len(layers)={len(log.layers)} != num_layers={num_layers}"
            )

    # Determine max sequence length
    if max_seq_len is None:
        max_seq_len = max(log.num_tokens for log in valid_logs)

    # Initialize output tensor with zeros (padding value)
    # Shape: [bs, max_seq_len, num_layers, topk]
    output_tensor = torch.zeros(
        (batch_size, max_seq_len, num_layers, top_k),
        dtype=torch.int64,  # Expert IDs are indices
    )

    # Fill tensor with routing logs
    for batch_idx, routing_log in enumerate(routing_logs_list):
        if routing_log is None:
            continue  # Leave as zeros for padded/invalid sequences

        seq_len = routing_log.num_tokens

        # Check sequence length
        if seq_len > max_seq_len:
            raise ValueError(
                f"Sequence {batch_idx} has length {seq_len} > max_seq_len {max_seq_len}"
            )

        # Process each layer
        for layer_log in routing_log.layers:
            layer_id = layer_log.layer_id

            # Validate layer_id
            if layer_id >= num_layers:
                raise ValueError(
                    f"Invalid layer_id {layer_id} (num_layers={num_layers})"
                )

            # expert_ids shape: (num_tokens, top_k)
            expert_ids = layer_log.expert_ids

            # Validate shape
            if expert_ids.shape != (seq_len, top_k):
                raise ValueError(
                    f"Layer {layer_id} expert_ids shape {expert_ids.shape} != "
                    f"expected ({seq_len}, {top_k})"
                )

            # Fill tensor: [batch_idx, :seq_len, layer_id, :]
            output_tensor[batch_idx, :seq_len, layer_id, :] = expert_ids

    return output_tensor


def convert_routing_logs_to_tensor_with_response_only(
    routing_logs_list: List[Optional[BatchRoutingLogs]],
    prompt_lengths: List[int],
    response_lengths: List[int],
) -> torch.Tensor:
    """
    Convert BatchRoutingLogs to tensor, keeping only response tokens.

    In RL training, we typically only need routing logs for generated response tokens,
    not for the input prompt tokens. This function extracts only the response portion.

    Args:
        routing_logs_list: List of BatchRoutingLogs, one per sequence.
        prompt_lengths: List of prompt lengths for each sequence.
        response_lengths: List of response lengths for each sequence.

    Returns:
        torch.Tensor: Shape [bs, max_response_len, num_layers, topk]
                     Contains expert indices for response tokens only.

    Example:
        >>> # Batch with prompts of length [10, 15] and responses of length [20, 18]
        >>> logs = [logs1, logs2]
        >>> prompt_lens = [10, 15]
        >>> response_lens = [20, 18]
        >>> tensor = convert_routing_logs_to_tensor_with_response_only(
        ...     logs, prompt_lens, response_lens
        ... )
        >>> assert tensor.shape == (2, 20, num_layers, topk)
    """
    # Filter out None entries
    valid_logs = [log for log in routing_logs_list if log is not None]

    if not valid_logs:
        raise ValueError("No valid routing logs in batch")

    batch_size = len(routing_logs_list)

    # Validate inputs
    if len(prompt_lengths) != batch_size:
        raise ValueError(
            f"prompt_lengths length {len(prompt_lengths)} != batch_size {batch_size}"
        )
    if len(response_lengths) != batch_size:
        raise ValueError(
            f"response_lengths length {len(response_lengths)} != batch_size {batch_size}"
        )

    # Validate consistency
    ref_log = valid_logs[0]
    num_layers = ref_log.num_layers
    top_k = ref_log.top_k

    for log in valid_logs:
        if log.num_layers != num_layers:
            raise ValueError(f"Inconsistent num_layers across batch")
        if log.top_k != top_k:
            raise ValueError(f"Inconsistent top_k across batch")

    # Determine max response length
    max_response_len = max(response_lengths)

    # Initialize output tensor
    # Shape: [bs, max_response_len, num_layers, topk]
    output_tensor = torch.zeros(
        (batch_size, max_response_len, num_layers, top_k),
        dtype=torch.int64,
    )

    # Fill tensor with routing logs (response portion only)
    for batch_idx, routing_log in enumerate(routing_logs_list):
        if routing_log is None:
            continue

        prompt_len = prompt_lengths[batch_idx]
        response_len = response_lengths[batch_idx]
        seq_len = routing_log.num_tokens

        # Validate lengths
        expected_len = prompt_len + response_len
        if seq_len != expected_len:
            raise ValueError(
                f"Sequence {batch_idx}: routing_log.num_tokens={seq_len} != "
                f"prompt_len + response_len = {prompt_len} + {response_len} = {expected_len}"
            )

        # Process each layer, extracting response tokens only
        for layer_log in routing_log.layers:
            layer_id = layer_log.layer_id

            # expert_ids shape: (num_tokens, top_k)
            # Extract response portion: [prompt_len:prompt_len+response_len, :]
            expert_ids = layer_log.expert_ids
            response_expert_ids = expert_ids[prompt_len : prompt_len + response_len, :]

            # Validate shape
            if response_expert_ids.shape[0] != response_len:
                raise ValueError(
                    f"Layer {layer_id} response_expert_ids length {response_expert_ids.shape[0]} "
                    f"!= response_len {response_len}"
                )

            # Fill tensor: [batch_idx, :response_len, layer_id, :]
            output_tensor[batch_idx, :response_len, layer_id, :] = response_expert_ids

    return output_tensor


def extract_routed_experts_from_dataproto(data):
    """
    Extract routed_experts tensor from DataProto batch.

    This is a helper function for cases where routing logs are stored in DataProto
    and need to be converted to tensor format.

    Args:
        data: DataProto containing routed_experts in batch

    Returns:
        torch.Tensor or None: Converted tensor if routed_experts present, else None
    """
    if "routed_experts" not in data.batch:
        return None

    routed_experts = data.batch["routed_experts"]

    # If already a tensor, return as-is
    if isinstance(routed_experts, torch.Tensor):
        return routed_experts

    # If list of BatchRoutingLogs, convert
    if isinstance(routed_experts, list):
        return convert_routing_logs_to_tensor(routed_experts)

    raise ValueError(f"Unexpected routed_experts type: {type(routed_experts)}")
