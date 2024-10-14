"""CacheEngine class for managing the KV cache."""
from typing import List, Tuple

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size,
                        is_pin_memory_available)

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)
        self.total_num_heads = model_config.hf_config.num_attention_heads
        try:
            self.attn_head_size = model_config.hf_config.attention_projection_size // self.total_num_heads
        except:
            self.attn_head_size = model_config.hf_config.hidden_size // self.total_num_heads

        self.head_size = model_config.get_head_size()
        self.hidden_size = self.head_size * self.total_num_heads
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(
            model_config.get_num_attention_heads(parallel_config),
            self.attn_head_size, # 这里修改
            self.num_kv_heads,
            model_config.get_sliding_window(),
            model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
        )

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.attn_head_size)
        #print("----------------", kv_cache_shape)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            kv_cache.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache
    ''' 
    def get_lf_cache_shape(self, bsz, hidden_size) -> Tuple[int, int, int, int]:
        return (bsz, hidden_size, 1, 1)

    def _allocate_lf_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates LF cache on the specified device."""
        LF_gpu_cache: List[LFCache] = []
        lf1_cache_shape = self.get_lf_cache_shape(num_blocks, self.hidden_size)
        lf2_cache_shape = self.get_lf_cache_shape(num_blocks, self.hidden_size // 2)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        for _ in range(self.num_layers):
            lf1_cache = torch.empty(lf1_cache_shape, dtype=self.dtype, pin_memory=pin_memory, device=device)
            lf2_cache = torch.empty(lf2_cache_shape, dtype=self.dtype, pin_memory=pin_memory, device=device)
            LF_gpu_cache.append((lf1_cache, lf2_cache))
        return LF_gpu_cache
    '''
    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)
        total_num_heads = model_config.hf_config.num_attention_heads
        try:
            attn_head_size = model_config.hf_config.attention_projection_size // total_num_heads
        except:
            attn_head_size = model_config.hf_config.hidden_size // total_num_heads

        key_cache_block = cache_config.block_size * num_heads * attn_head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total