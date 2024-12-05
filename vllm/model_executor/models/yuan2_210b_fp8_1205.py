# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Yuan model compatible with HuggingFace weights."""
from operator import index
import functools
import json
import os
from urllib import request
import pdb
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import copy
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import broadcast
from torch import einsum, nn, Tensor
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm.model_executor.models.configuration_yuan import YuanConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.config import LoRAConfig, CacheConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from transformers.activations import ACT2FN
from vllm.model_executor.utils import set_weight_attrs
from vllm.attention import Attention, AttentionMetadata
#from apex.normalization import MixedFusedRMSNorm as RMSNorm
from vllm.model_executor.layers.fused_moe import *
from vllm.model_executor.layers.linear import  (LinearMethodBase,
                                               ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead, DEFAULT_VOCAB_PADDING_SIZE)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader, kv_cache_scales_loader,) # hf_model_weights_iterator)
from vllm.sequence import SamplerOutput, IntermediateTensors, ExecuteModelRequest
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)

from vllm.model_executor.utils import set_weight_attrs
from vllm.logger import init_logger
from vllm.utils import is_hip

from vllm.engine.llm_engine import LLMEngine
from vllm.engine.async_llm_engine import AsyncLLMEngine, _AsyncLLMEngine # for api_server
from vllm.outputs import EmbeddingRequestOutput, RequestOutput # for api_server
from vllm.sequence import SequenceData
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.distributed.parallel_state import get_tensor_model_parallel_group
from transformer_engine.pytorch import RMSNorm 
### add lf1_caches and lf2_caches in SequenceData for Yuan model
setattr(SequenceData, "lf1_caches", [])
setattr(SequenceData, "lf2_caches", [])

global_seq_list = []
class YuanLLMEngine(LLMEngine):

    def step(self):
        if self.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is only supported through AsyncLLMEngine "
                "as performance will be severely degraded otherwise.")

        seq_group_metadata_list, scheduler_outputs = self.scheduler[0].schedule()
        global global_seq_list
        # each step needs to be cleared
        global_seq_list.clear()
        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                global_seq_list.append(seq_data)

        if not scheduler_outputs.is_empty():
            finished_requests_ids = self.scheduler[
                0].get_and_reset_finished_requests_ids()
            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids)
            output = self.model_executor.execute_model(
                execute_model_req=execute_model_req)
        else:
            output = []
        request_outputs = self._process_model_outputs(
            output, scheduler_outputs.scheduled_seq_groups,
            scheduler_outputs.ignored_seq_groups, seq_group_metadata_list)

        # Log stats.
        self.do_log_stats(scheduler_outputs, output)

        # Tracing
        self.do_tracing(scheduler_outputs)

        if not self.has_unfinished_requests():
            # Stop the execute model loop in parallel workers until there are
            # more requests to process. This avoids waiting indefinitely in
            # torch.distributed ops which may otherwise timeout, and unblocks
            # the RPC thread in the workers so that they can process any other
            # queued control plane messages, such as add/remove lora adapters.
            self.model_executor.stop_remote_worker_execution_loop()
        return request_outputs

class YuanAsyncLLMEngine(LLMEngine):
    async def step_async(
        self, virtual_engine: int
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        seq_group_metadata_list, scheduler_outputs = self.scheduler[
            virtual_engine].schedule()

        global global_seq_list
        # each step needs to be cleared
        global_seq_list.clear()
        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                global_seq_list.append(seq_data)

        if not scheduler_outputs.is_empty():
            # Execute the model.
            finished_requests_ids = self.scheduler[
                virtual_engine].get_and_reset_finished_requests_ids()
            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                virtual_engine=virtual_engine,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids)
            output = await self.model_executor.execute_model_async(
                execute_model_req)
        else:
            output = []

        request_outputs = self._process_model_outputs(
            output, scheduler_outputs.scheduled_seq_groups,
            scheduler_outputs.ignored_seq_groups, seq_group_metadata_list)

        # Log stats.
        self.do_log_stats(scheduler_outputs, output)

        # Tracing
        self.do_tracing(scheduler_outputs)

        return request_outputs


# hijack LLMEngine.step with YuanLLMEngine.step, use record global_seq_list
LLMEngine.step = YuanLLMEngine.step
_AsyncLLMEngine.step_async = YuanAsyncLLMEngine.step_async # api_server use

logger = init_logger(__name__)
LFCache_type = List[torch.Tensor]

'''
class RMSNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states
'''
class ParallelAttention_router(nn.Module):
    def __init__(self, config):
        super(ParallelAttention_router, self).__init__()

        self.hidden_size = config.hidden_size
        self.projection_size = config.moe_config['moe_num_experts']
        self.query_key_value = ReplicatedLinear(self.hidden_size, self.projection_size*3, bias=False)

    def forward(self, hidden_states, attn_metadata):
        mix_layer, _ = self.query_key_value(hidden_states)
        (query_layer, key_layer, value_layer) = torch.chunk(mix_layer, 3, dim=-1)
        
        query_layer = query_layer.view(*query_layer.shape, 1).float()
        key_layer = key_layer.view(*key_layer.shape, 1).float()
        value_layer = value_layer.view(*value_layer.shape, 1).float()

        attn_weights = torch.matmul(query_layer, key_layer.transpose(1,2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value_layer)
        router_output = attn_output.squeeze(2)
        return router_output

class MoEDroplessTokenDispatcher:
    def __init__(self, num_experts: int, config: YuanConfig) -> None:
        
        self.num_experts = num_experts
        assert self.num_experts > 0, "Expected at least one expert"
        self.router_topk = config.moe_config['moe_top_k']

    def token_permutation(
        self, hidden_states: torch.Tensor, max_prob: torch.Tensor, max_ind: torch.Tensor
    ):
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        if self.router_topk > 1:
            global_local_map = torch.ones_like(max_ind).bool()
            local_indices = max_ind.masked_select(global_local_map)
            local_probs = max_prob.masked_select(global_local_map)
            global_local_map = global_local_map.nonzero()[:, 0]
            global_local_map = global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
            local_hidden_states = torch.gather(hidden_states, 0, global_local_map)

        indices = torch.argsort(local_indices, dim=0)
        tokens_per_expert = torch.histc(
            local_indices,
            bins=self.num_experts,
            min=0,
            max=self.num_experts - 1,
        )
        tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

        indices = indices.view(-1, 1).expand(-1, hidden_states.shape[-1])
        permuted_local_hidden_states = torch.gather(local_hidden_states, 0, indices)
        return (permuted_local_hidden_states, tokens_per_expert, local_probs, indices, global_local_map)

    def token_unpermutation(
        self,
        hidden_states: torch.Tensor,
        scores: torch.Tensor,
        indices: torch.Tensor,
        global_local_map: torch.Tensor = None,
    ):
        scores = scores.to(dtype=hidden_states.dtype)
        unpermuted_local_hidden = torch.zeros_like(hidden_states)
        assert indices.shape == hidden_states.shape, f'{indices.shape}, {hidden_states.shape}'
        unpermuted_local_hidden = unpermuted_local_hidden.scatter(0, indices, hidden_states)

        if self.router_topk > 1:
            unpermuted_local_hidden = unpermuted_local_hidden * scores.view(-1, 1)

        unpermuted_local_bias = None
        output_total = unpermuted_local_hidden
        output_bias_total = unpermuted_local_bias

        if self.router_topk > 1:
            global_num_tokens = self.hidden_shape[0]
            global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
            unpermuted_global_hidden = torch.zeros(
                global_hidden_shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            output_total = unpermuted_global_hidden.scatter_add(
                0, global_local_map, unpermuted_local_hidden
            )
        output_total = output_total.view(self.hidden_shape)

        return output_total

class GroupedMLP(nn.Module):
    """An efficient implementation of the Experts layer using CUTLASS GroupedGEMM.

    This class is designed to execute multiple experts in parallel, thereby maximizing computational efficiency.
    """

    def __init__(self, num_experts, config: YuanConfig, layer_num: int,  params_dtype: Optional[torch.dtype] = None, cache_config: Optional[CacheConfig] = None, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.intermediate_size = (config.moe_config['ffn_hidden_size'] //
                                  self.tp_size)
        def glu(x):
            x = torch.chunk(x, 2, dim=-1)
            return torch.nn.functional.silu(x[0]) * x[1]

        self.activation_func = glu
        self.ffn_hidden_size = config.moe_config['ffn_hidden_size']
        
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.w1 = nn.ModuleList([ReplicatedLinear(config.hidden_size, self.intermediate_size * 2, bias=False, quant_config=quant_config) for _ in range(self.num_experts)])
        self.w2 = nn.ModuleList([ReplicatedLinear(self.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config) for _ in range(self.num_experts)])

        set_weight_attrs(
            self.w1,
            {
                "weight_loader": self.weight_loader,
            },
        )
        set_weight_attrs(
            self.w2,
            {
                "weight_loader": self.weight_loader,
            },
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        if weight_name.endswith("w1"):
            chunk_size = loaded_weight.shape[2] // 2
            chunk0 = torch.split(loaded_weight, chunk_size, dim=2)[0].clone().detach()
            chunk1 = torch.split(loaded_weight, chunk_size, dim=2)[1].clone().detach()
            sub_chunk_size = param_data.shape[2] // 2
            sub_chunk0 = torch.split(chunk0, sub_chunk_size, dim=2)[tp_rank].clone().detach()
            sub_chunk1 = torch.split(chunk1, sub_chunk_size, dim=2)[tp_rank].clone().detach()
            param_data.copy_(torch.cat([sub_chunk0, sub_chunk1], dim=2))#.permute(0, 2, 1))

        if weight_name.endswith("w2"):
            chunk_size = loaded_weight.shape[1] // self.tp_size
            sub_chunk = torch.split(loaded_weight, chunk_size, dim=1)[tp_rank].clone().detach()
            param_data.copy_(sub_chunk)#.permute(0, 2, 1))

    def forward(self, permuted_hidden_states, tokens_per_expert):
        torch.cuda.set_device(permuted_hidden_states.device)
        permuted_hidden_states = permuted_hidden_states#.to('cuda:0')

        fc2_outputs = []
        start_idx = 0
        for i in range(self.num_experts):
            if tokens_per_expert[i] == 0:
                continue
            end_idx = start_idx + tokens_per_expert[i]
            # Use custom attributes for each expert's Linear layers

            fc1_output, _ = self.w1[i](permuted_hidden_states[start_idx:end_idx])
            intermediate_parallel = self.activation_func(fc1_output)
            fc2_output, _ = self.w2[i](intermediate_parallel)
            fc2_outputs.append(fc2_output)
            start_idx = end_idx
        fc2_output = torch.cat(fc2_outputs, dim=0)
        if self.tp_size > 1:
            fc2_output = tensor_model_parallel_all_reduce(fc2_output.float()).to(permuted_hidden_states.dtype)
        return fc2_output#.to('cuda:1')

class YuanMoeLayer(nn.Module):
    def __init__(self, config:YuanConfig, num_layer: int, num_experts, quant_config: Optional[QuantizationConfig] = None): # Quant Use
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.top_k = config.moe_config['moe_top_k']
        self.hidden_size = config.hidden_size
        self.num_layer = num_layer

        expert_indices_offset = (0)

        if config.moe_config['router_type'] == 'attn_router':
            self.gate = ParallelAttention_router(config)
        else:
            self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.token_dispatcher = MoEDroplessTokenDispatcher(self.num_experts, self.config)
        self.experts = GroupedMLP(self.num_experts, self.config, num_layer, quant_config=quant_config) # Quant Use


    def routing(self, logits: torch.Tensor) -> torch.Tensor:
        top_logits, indices = torch.topk(logits, k=self.top_k, dim=1)
        scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)
        return scores, indices

    def forward(self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata) -> torch.Tensor:
        sequence_length, hidden_dim = hidden_states.shape
        logits = self.gate(hidden_states)  # not use ParallelAttn_router
        scores, indices = self.routing(logits)
        scores = scores.to(hidden_states.dtype)
        (dispatched_input, tokens_per_expert, scores, indices, global_local_map, ) = self.token_dispatcher.token_permutation(hidden_states, scores, indices)
        
        expert_output = self.experts(dispatched_input, tokens_per_expert)
        output = self.token_dispatcher.token_unpermutation(expert_output, scores, indices, global_local_map)
        return output

def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(_yarn_find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)

def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def _yarn_get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0

class YuanYaRNScaledRotaryEmbedding(nn.Module):
    def __init__(self,
                 dim,
                 rotary_base=10000,
                 max_position_embeddings=2048,
                 scale=1,
                 original_max_position_embeddings=2048,
                 extrapolation_factor=1,
                 attn_factor=1,
                 beta_fast=32,
                 beta_slow=1,
                 dtype=torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = rotary_base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        
        self.revised_yarn()
        self.max_seq_len_cached = max_position_embeddings
        t = np.arange(self.max_seq_len_cached)
        t = torch.tensor(t, device=self.inv_freq.device,dtype=torch.float)
        freqs = torch.outer(t, self.inv_freq)
        self.emb = torch.cat((freqs, freqs), dim=-1)

    def forward(self, x, seq_len=None):
        return self.emb[:, None, None, :]

    def yarn(self, device):
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)
        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.base,
            self.original_max_position_embeddings
        )
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.dim // 2).float().to(device)) \
            * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def revised_yarn(self):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.base,
            self.original_max_position_embeddings
        )
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.dim // 2).float()) * self.extrapolation_factor
        inv_freq = inv_freq / ((1-inv_freq_mask)*self.scale + inv_freq_mask)
        self.register_buffer("inv_freq", inv_freq)

class YuanRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, dtype=torch.float32, rotary_interleaved=False, seq_len_interpolation_factor=None):
        super().__init__()
        self.base = base
        self.dim = dim
        self.rotary_interleaved = rotary_interleaved
        self.seq_len_interpolation_factor = seq_len_interpolation_factor

    def forward(self, x, max_seq_len, offset=0):

        """Forward pass of RoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): _description_. Defaults to 0.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        inv_freq = (1.0 / ( self.base**(torch.arange(0, self.dim, 2, dtype=torch.float32, device=torch.cuda.current_device()) / self.dim))).to(torch.float32)
        
        #max_seq_len_int = max_seq_len.item() if max_seq_len.numel() == 1 else max_seq_len.max().item()
        seq = (
            torch.arange(max_seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
            + offset
        )

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = torch.outer(seq, inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
                freqs.shape[0], -1
            )
        # emb [seq_length, .., dim]
        emb = emb[:, None, None, :]
        #emb = emb[:, None, :] 
        return emb

def _rotate_half_bshd(x: Tensor, rotary_interleaved: bool):
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)

def apply_rotary_pos_emb_bshd(t: Tensor, freqs: Tensor, position_ids: Tensor ,rotary_interleaved: bool = False):
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    dtype = t.dtype
    rot_dim = freqs.shape[-1]
    #if position_ids.shape[1] > 1:
    freqs = freqs[position_ids]
    freqs = freqs.view(t.shape[0],freqs.shape[1],freqs.shape[3])
    #freqs = freqs.view(t.shape[1],freqs.shape[1],freqs.shape[2],freqs.shape[4]).transpose(0,1)
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t_type = t.dtype
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    t = (t * cos_) + (_rotate_half_bshd(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)

def apply_rotary_pos_emb_thd(
        t: Tensor, cu_seqlens: Tensor, freqs: Tensor, position_ids: Tensor, rotary_interleaved: bool = False,
):

    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """

    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    return torch.cat(
        [
            apply_rotary_pos_emb_bshd(x.unsqueeze(1), freqs[: x.size(0)], position_ids)
            for x in torch.split(t, seqlens)
        ]
    ).squeeze(1)

def apply_rotary_pos_emb(t: Tensor, freqs: Tensor, position_ids: Tensor, apply_rope_fusion: bool = True, cu_seqlens: Optional[Tensor] = None):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd (conventional) / thd (packed seq) format
    """
    return apply_rotary_pos_emb_bshd(t, freqs, position_ids)
    '''
    if apply_rope_fusion:
        if cu_seqlens is None:
            return fused_apply_rotary_pos_emb(t, freqs, transpose_output_memory=True)
        else:
            return fused_apply_rotary_pos_emb_thd(t, cu_seqlens, freqs)
    else:
        if cu_seqlens is None:
            return apply_rotary_pos_emb_bshd(t, freqs, position_ids, rotary_interleaved=rotary_interleaved)
        else:
            return apply_rotary_pos_emb_thd(
                t, cu_seqlens, position_ids, freqs, rotary_interleaved=config.rotary_interleaved
            )
    '''

class LocalizedFiltering(torch.nn.Module):
    """
    Mega's Exponential Moving Average layer, largely left unmodified from the original repo with the exception of
    variable names and moving away from the stateful representation of incremental decoding state. See
    "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(self, config, hidden_size, lf_conv2d_group, lf_conv2d_num_pad):
        super().__init__()

        self.embed_dim = hidden_size
        self.lf_conv2d_group = lf_conv2d_group
        self.lf_conv2d_num_pad = lf_conv2d_num_pad
        self.conv1 = torch.nn.Conv2d(
            self.embed_dim,
            self.embed_dim // 2,
            (2, 1),
            stride=(1, 1),
            padding=(self.lf_conv2d_num_pad, 0),
            groups=self.lf_conv2d_group
        )
        self.conv2 = torch.nn.Conv2d(
            self.embed_dim // 2,
            self.embed_dim,
            (2, 1),
            stride=(1, 1),
            padding=(self.lf_conv2d_num_pad, 0),
            groups=self.lf_conv2d_group
        )
        self.output_layernorm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)

    def forward(self, inputs, lf1_cache, lf2_cache, attn_metadata):
        bsz = attn_metadata.num_prefills + attn_metadata.num_decode_tokens
        result, lf1_, lf2_ = [], [], []
        if attn_metadata.prefill_metadata != None:
            for b in range(bsz):
                inputs_ = inputs[attn_metadata.prefill_metadata.seq_start_loc[b]:attn_metadata.prefill_metadata.seq_start_loc[b+1]]
                lf1_cache_ = lf1_cache[b:b+1]
                lf2_cache_ = lf2_cache[b:b+1]

                inputs_ = inputs_.view(lf1_cache_.shape[0], -1, inputs_.shape[-1]) # [b, s, h]
                inputs_ = inputs_.permute([1, 0, 2])  # [ s, b, h]
                residual = inputs_
                old_shape = inputs_.shape
                new_shape = inputs_.view(inputs_.shape[0], 1, inputs_.shape[1], inputs_.shape[2]).shape  # [s, 1, b, h]
                inputs_ = inputs_.view(new_shape).permute([2, 3, 0, 1])  # [b, h, s, 1]
                inputs_ = torch.cat([lf1_cache_, inputs_], dim=2)  # [b, h, s+1, 1]
                output1 = self.conv1(inputs_)
                output1 = torch.cat([lf2_cache_, output1], dim=2)
                output2 = self.conv2(output1).permute([2, 3, 0, 1])
                output2 = output2.view(old_shape)
               
                assert list(output2.shape) == list(residual.shape), f'{output2.shape}, {residual.shape}'
                output3 = output2 + residual
                lf_output = self.output_layernorm(output3)
                lf_output = lf_output.permute([1, 0, 2])

                lf1 = inputs_[:, :, -1:, :].contiguous()
                lf2 = output1[:, :, -1:, :].contiguous()
                #lf_output = lf_output.contiguous().view(-1, lf_output.shape[-1])
                lf1_.append(lf1)
                lf2_.append(lf2)
                result.append(lf_output.view(-1, *lf_output.shape[2:]))
            lf_output = torch.cat(result, dim=0)
            lf1 = torch.cat(lf1_)
            lf2 = torch.cat(lf2_)
        else:
            inputs = inputs.view(lf1_cache.shape[0], -1, inputs.shape[-1]) # [b, s, h]
            inputs = inputs.permute([1, 0, 2])  # [ s, b, h]
            residual = inputs
            old_shape = inputs.shape
            new_shape = inputs.view(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2]).shape  # [s, 1, b, h]
            inputs = inputs.view(new_shape).permute([2, 3, 0, 1])  # [b, h, s, 1]
            inputs = torch.cat([lf1_cache, inputs], dim=2)  # [b, h, s+1, 1]
            output1 = self.conv1(inputs)
            output1 = torch.cat([lf2_cache, output1], dim=2)
            output2 = self.conv2(output1).permute([2, 3, 0, 1])
            output2 = output2.view(old_shape)

            assert list(output2.shape) == list(residual.shape), f'{output2.shape}, {residual.shape}'
            output3 = output2 + residual
            lf_output = self.output_layernorm(output3)
            lf_output = lf_output.permute([1, 0, 2])

            lf1 = inputs[:, :, -1:, :].contiguous()
            lf2 = output1[:, :, -1:, :].contiguous()
        return lf_output, lf1, lf2

class YuanMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.up_proj = ColumnParallelLinear(hidden_size,
                                            intermediate_size,
                                            bias=False,)
        self.gate_proj= ColumnParallelLinear(hidden_size,
                                            intermediate_size,
                                            bias=False,)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x, attn_metadata: AttentionMetadata):
        x1, _ = self.up_proj(x)
        x3 = self.act_fn(x1)
        x2, _ = self.gate_proj(x)
        x, _ = self.down_proj(x2 * x3)
        return x


class YuanAttention(nn.Module):

    def __init__(
        self,
        config: YuanConfig,
        hidden_size: int,
        attention_projection_size: int,
        num_heads: int,
        num_kv_heads: int,
        attn_head_size=None,
        rope_theta: float = 500000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 4096,
        linear_method: Optional[LinearMethodBase] = None,
        bias: bool = False,
        sliding_window: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None, # Quant Use
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        self.lf_conv2d_group = config.lf_conv2d_group
        self.lf_conv2d_num_pad = config.lf_conv2d_num_pad

        self.attn_head_size = attention_projection_size // num_heads if attn_head_size is None else attn_head_size
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        
        self.eps = 1e-6
        self.get_query_key = ColumnParallelLinear(
            hidden_size,
            2 * attention_projection_size,
            bias=False,
            quant_config=quant_config,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            attention_projection_size,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            attention_projection_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        
        self.model_type = getattr(config, 'model_type', 'yuan')
        self.lf_gate = LocalizedFiltering(self.config, self.hidden_size, self.lf_conv2d_group, self.lf_conv2d_num_pad)
        self.attn = Attention(self.num_kv_heads,
                              self.attn_head_size,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              #sliding_window=sliding_window
                              ) 

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        kv_cache: torch.Tensor,
        lf1_cache: torch.Tensor,
        lf2_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        use_yarn: bool=False,
        yarn_scale_factor: float=1.0,
        attn_factor: float=1.0,
    ) -> torch.Tensor:
        v, _ = self.v_proj(hidden_states)
        hidden_states, lf1, lf2 = self.lf_gate(hidden_states, lf1_cache, lf2_cache, attn_metadata)
        hidden_states = hidden_states.contiguous().view(-1, hidden_states.shape[-1])
        qk, _ = self.get_query_key(hidden_states)
        qk = qk.view(*qk.shape[:-1], self.num_heads, int(qk.shape[-1] // self.num_heads))
        (q, k) = torch.chunk(qk, 2, dim=-1)
        
        q = apply_rotary_pos_emb(q, rotary_pos_emb, positions, attn_metadata)
        k = apply_rotary_pos_emb(k, rotary_pos_emb, positions, attn_metadata)
        q = q.view(-1, self.num_heads * self.attn_head_size)
        k = k.view(-1, self.num_heads * self.attn_head_size)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output, lf1, lf2


class YuanDecoderLayer(nn.Module):

    def __init__(
        self,
        config: YuanConfig,
        num_layer,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_projection_size = getattr(config, 'attention_projection_size', config.hidden_size)
        self.self_attn = YuanAttention(
            config=config,
            hidden_size=self.hidden_size,
            attention_projection_size=self.attention_projection_size,
            num_heads=config.num_attention_heads,
            linear_method=linear_method,
            num_kv_heads=config.num_attention_heads,
            cache_config=cache_config,
            quant_config=quant_config,
        )
        self.num_layer = num_layer
        self.use_moe = getattr(config, "use_moe", False)
        if self.use_moe:
            assert config.moe_config['per_layer_experts_blocks'] != None or config.moe_config['num_experts'] != None
            if config.moe_config['per_layer_experts_blocks'] != None:
                self.num_experts = config.moe_config['per_layer_experts_blocks'][num_layer]
            else:
                self.num_experts = config.moe_config['num_experts']
            self.mlp = YuanMoeLayer(config, self.num_layer, self.num_experts, quant_config=quant_config)
        else:
            self.mlp = YuanMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                linear_method=linear_method,
            )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        kv_cache: torch.Tensor,
        lf1_cache: torch.Tensor,
        lf2_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        use_yarn: bool=False,
        yarn_scale_factor: float=1.0,
        attn_factor: float=1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, lf1, lf2 = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            kv_cache=kv_cache,
            lf1_cache=lf1_cache,
            lf2_cache=lf2_cache,
            attn_metadata=attn_metadata,
            use_yarn=use_yarn,
            yarn_scale_factor=yarn_scale_factor,
            attn_factor=attn_factor
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, attn_metadata)
        hidden_states = residual + hidden_states
        return hidden_states, lf1, lf2


class YuanModel(nn.Module):

    def __init__(
        self,
        config: YuanConfig,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
 
        self.vocab_size = config.vocab_size + lora_vocab
        num_heads = getattr(config, "kv_channels", config.num_attention_heads)
        rotary_percent = getattr(config, "rotary_percent", 1.0)
        rotary_dim = getattr(config, "rotary_dim", config.hidden_size // config.num_attention_heads)
        self.use_yarn = getattr(config, "use_yarn", False)
        rotary_interleaved = getattr(config, "rotary_interleaved", False)
        rotary_base = getattr(config, "rotary_base", 500000)
        seq_len_interpolation_factor = getattr(config, "seq_len_interpolation_factor", None)
        self.yarn_scale_factor = getattr(config, "yarn_scale_factor", 128)
        max_position_embeddings = getattr(config, "max_position_embeddings", 4096)
        self.attn_factor = getattr(config, "attn_factor", 1.0)
        scaled_max_position_embeddings = getattr(config, "scaled_max_position_embeddings", max_position_embeddings)
        torch_dtype = getattr(config, "torch_dtype", torch.bfloat16)

        if self.use_yarn:
            self.rotary_emb = YuanYaRNScaledRotaryEmbedding(
                rotary_dim,
                max_position_embeddings=scaled_max_position_embeddings,
                scale=self.yarn_scale_factor,
                original_max_position_embeddings=max_position_embeddings,
                attn_factor=self.attn_factor,
                dtype=torch_dtype
            )
            self.seq_len = scaled_max_position_embeddings
        else:
            self.rotary_emb = YuanRotaryEmbedding(rotary_dim, base=config.rotary_base, dtype=config.torch_dtype, rotary_interleaved=rotary_interleaved, seq_len_interpolation_factor=seq_len_interpolation_factor)
            self.seq_len = max_position_embeddings
        self.layers = nn.ModuleList([
            YuanDecoderLayer(config, i, linear_method, cache_config, quant_config)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.hidden_size = getattr(config, "hidden_size", 3072)
        self.lf_cache_init = False

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        rotary_pos_emb = self.rotary_emb(hidden_states, self.seq_len)
        
        global global_seq_list
        lf1_cache_list = []
        lf2_cache_list = []
        # profile lf_cache: init
        #if attn_metadata.is_prompt and not self.lf_cache_init:
        if attn_metadata.prefill_metadata != None and not self.lf_cache_init:
            ### profile
            ### LLMEngine.step has not been hijacked, init lf1_cache and lf2_cache
            for j, layer in enumerate(self.layers):
                bs = attn_metadata.block_tables.shape[0]
                lf1_cache = torch.zeros((bs, self.hidden_size, 1, 1),
                                            dtype=hidden_states.dtype, device=hidden_states.device)
                lf2_cache = torch.zeros((bs, self.hidden_size // 2, 1, 1),
                                            dtype=hidden_states.dtype, device=hidden_states.device)
                lf1_cache_list.append(lf1_cache)
                lf2_cache_list.append(lf2_cache)
            self.lf_cache_init = True
        elif global_seq_list:
            ### is_driver_worker == True
            #if attn_metadata.is_prompt:
            if attn_metadata.prefill_metadata != None:
                for i, seq_data in enumerate(global_seq_list):
                    lf1_layer = torch.zeros((1, self.hidden_size, 1, 1),
                                                dtype=hidden_states.dtype, device=hidden_states.device)
                    lf2_layer = torch.zeros((1, self.hidden_size // 2, 1, 1),
                                                dtype=hidden_states.dtype, device=hidden_states.device)
                    lf1_caches = [lf1_layer.clone() for _ in range(len(self.layers))]
                    lf2_caches = [lf2_layer.clone() for _ in range(len(self.layers))]
                    seq_data.lf1_caches = lf1_caches
                    seq_data.lf2_caches = lf2_caches
            
            for i, layer in enumerate(self.layers):
                lf1_sub = []
                lf2_sub = []
                for j, seq_data in enumerate(global_seq_list):
                    lf1_sub.append(seq_data.lf1_caches[i])
                    lf2_sub.append(seq_data.lf2_caches[i])
                lf1_cache_list.append(torch.cat(lf1_sub, 0))
                lf2_cache_list.append(torch.cat(lf2_sub, 0))
            
            metadata_dict = {}
            for i, layer in enumerate(self.layers):
                metadata_dict.update({ str(i) : lf1_cache_list[i],
                                       str(i + len(self.layers)) : lf2_cache_list[i]})
            broadcast_tensor_dict(metadata_dict, src=0)
        else:
            ### is_driver_worker == False
            metadata_dict = broadcast_tensor_dict(src=0)
            for i, layer in enumerate(self.layers):
                lf1_cache_list.append(metadata_dict[str(i)])
                lf2_cache_list.append(metadata_dict[str(i+len(self.layers))])
        for i, layer in enumerate(self.layers):
            hidden_states, lf1, lf2 = layer(
                positions,
                hidden_states,
                rotary_pos_emb,
                kv_caches[i],
                lf1_cache_list[i],
                lf2_cache_list[i],
                attn_metadata,
                self.use_yarn,
                self.yarn_scale_factor,
                self.attn_factor
            )

            ### is_driver_worker == True, will update lf_cache
            if global_seq_list:
                for j, seq_data in enumerate(global_seq_list):
                    seq_data.lf1_caches[i].copy_(lf1[j])
                    seq_data.lf2_caches[i].copy_(lf2[j])

        hidden_states = self.norm(hidden_states)
        return hidden_states


class YuanForCausalLM(nn.Module):

    packed_modules_mapping = {
        "up_gate_proj": [
            "up_proj",
            "gate_proj",
        ],
    }

    def __init__(
        self,
        config: YuanConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.use_moe = getattr(config, "use_moe", False)
        self.linear_method = linear_method
        self.model = YuanModel(config, linear_method, lora_config, quant_config, cache_config)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
            quant_config=quant_config,
        )

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        for name, loaded_weight in weights:
            if "rotary_pos_emb" in name:
                continue
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            param_data = param.data
            if 'w1' in name:
                    chunk_size = loaded_weight.shape[0] // 2
                    chunk0 = torch.split(loaded_weight, chunk_size, dim=0)[0].clone().detach()
                    chunk1 = torch.split(loaded_weight, chunk_size, dim=0)[1].clone().detach()
                    sub_chunk_size = param_data.shape[0] // 2
                    sub_chunk0 = torch.split(chunk0, sub_chunk_size, dim=0)[tp_rank].clone().detach()
                    sub_chunk1 = torch.split(chunk1, sub_chunk_size, dim=0)[tp_rank].clone().detach()
                    param_data.copy_(torch.cat([sub_chunk0, sub_chunk1], dim=0))
            elif 'w2' in name:
                if 'weight_scale' in name:
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                else:
                    chunk_size = loaded_weight.shape[1] // tp_size
                    sub_chunk = torch.split(loaded_weight, chunk_size, dim=1)[tp_rank].clone().detach()
                    param_data.copy_(sub_chunk)
            else:
                if 'get_query_key.weight_scale' in name or 'v_proj.weight_scale' in name:
                    chunk_size = loaded_weight.shape[0] // tp_size
                    sub_chunk = torch.split(loaded_weight, chunk_size, dim=0)[tp_rank].clone().detach()
                    param_data.copy_(sub_chunk)
                else:
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight) 
