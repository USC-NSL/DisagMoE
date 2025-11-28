import torch

from torch import Tensor
import numpy as np

from typing import override, Tuple, List, Union, Dict, Optional
from enum import Enum
import time

from vllm.config import CacheConfig as VllmCacheConfig
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from disagmoe.env import ENV_VARS
from disagmoe.models.attention import MoEAttention
from disagmoe.models.experts import MoEExperts, MoEExpertsSerial, MoEExpertsDeepGemmFP8
from disagmoe.config import ModelConfig, CacheConfig as DmoeCacheConfig
from disagmoe.utils.utils import nvtx_range, _log_memory_usage
from disagmoe.utils.logger import get_logger
from disagmoe.models.utils import make_attention_dummy_batch, make_prefill_meta
from disagmoe.block_manager.block_manager import GPUBlockManager, CPUBlockManager, BaseBlockManager
from disagmoe.block_manager.mem_pool import MHATokenToKVPool
from disagmoe.frontend.datatypes import AttentionForwardBatch, AttentionForwardResult
from disagmoe.frontend.engine_utils import get_global_engine_config
from disagmoe.executor.cuda_graph import CUDAGraphAttnExecutor

def get_module_param_memory(module, unit='GB'):
    unit_scale = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    scale = unit_scale[unit.upper()]
    
    param_mem = sum(p.numel() * p.element_size() for p in module.parameters())
    buffer_mem = sum(b.numel() * b.element_size() for b in module.buffers())
    
    total_mem = (param_mem + buffer_mem) / scale
    return total_mem

class ExecutorType(Enum):
    ATTENTION_EXEC = 1
    EXPERTS_EXEC = 2
    
class Executor:
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.num_layers = len(model_config.layer_ids)
        self.layer_mappings = [0 for _ in range(max(model_config.layer_ids) + 1)]
        for i, id in enumerate(model_config.layer_ids):
            self.layer_mappings[id] = i
        self.operators: List[torch.nn.Module] = None
    
    def execute(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def initialize_cache(self, num_blocks: int) -> None:
        raise NotImplementedError()
    
    def print_model_memory_usage(self) -> None:
        # sum up all memory used by the operators
        total_memory_gb = 0
        for operator in self.operators:
            total_memory_gb += get_module_param_memory(operator, unit='GB')
        print(f"Model weights total memory usage: {total_memory_gb:.1f} GB")

class AttnExecutor(Executor):

    def __init__(self, model_config: ModelConfig, cache_config: DmoeCacheConfig, gate_profile_bytes: Optional[bytes] = None):
        super().__init__(model_config)
        self.type = ExecutorType.ATTENTION_EXEC
        self.cache_config = cache_config
        self.vllm_cache_config = VllmCacheConfig(
            cache_dtype="auto",
            block_size=cache_config.block_size,
            gpu_memory_utilization=0,
            swap_space=0,
        )
        self.enable_cuda_graph = get_global_engine_config().enable_cuda_graph_attn
        self.cuda_graph_executor = None
        self.device = "cuda"
        self.block_mgr: BaseBlockManager = None
        self.gate_profile_bytes: Optional[bytes] = gate_profile_bytes
        
        self.init_model()
        self.init_kv_cache()
        
    def init_model(self):
        _, total_memory = torch.cuda.mem_get_info()
        self.init_gpu_memory = total_memory
        
        # Build quantization config for attention QKV if requested
        qkv_quant_config = None
        try:
            method = getattr(self.model_config, "attn_qkv_quant", None)
            if method and method != "none":
                if method == "fp8":
                    # Use defaults; requires activation_scheme at construction time
                    qkv_quant_config = Fp8Config(activation_scheme="dynamic")
                    get_logger().info(f"Successfully built FP8 quant config for QKV.")
                else:
                    # Handle other methods as needed
                    qkv_quant_config = None
        except Exception as e:
            get_logger().warning(
                f"Failed to build QKV quantization config '{getattr(self.model_config, 'attn_qkv_quant', None)}': {e}. Falling back to unquantized."
            )
            qkv_quant_config = None
        
        self.operators = [
            MoEAttention(
                layer_id,
                self.model_config.hidden_size, 
                self.model_config.num_heads, 
                self.model_config.num_kv_heads, 
                self.model_config.num_experts,
                self.model_config.top_k,
                cache_config=self.vllm_cache_config,
                quant_config_qkv=qkv_quant_config,
                gate_profile_bytes=self.gate_profile_bytes,
            ) for layer_id in range(self.num_layers)
        ]
        _log_memory_usage("After allocate attention parameters")

        # DisagMoE hacks:
        # 1. for vllm's fp8, use randn dummy weights rather than empty weights
        # 2. call process_weights_after_loading to match quantization kernel layouts
        for operator in self.operators:
            for _, module in operator.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is None:
                    continue
                # Dummy init for original FP8 methods if no checkpoint populated them.
                if quant_method.__class__.__name__ in (
                        "Fp8LinearMethod",
                        "PTPCFp8LinearMethod",
                        "ModelOptFp8LinearMethod",
                ):
                    weight = getattr(module, "weight", None)
                    # Scales may be per-tensor or block-wise (inv). Either might be present.
                    weight_scale = getattr(module, "weight_scale",
                                           getattr(module, "weight_scale_inv", None))
                    input_k = getattr(module, "input_size_per_partition", None)
                    output_n = getattr(module, "output_size_per_partition", None)
                    if weight is not None and input_k is not None and output_n is not None:
                        try:
                            with torch.no_grad():
                                # If scales exist and are still sentinel-min, or if scales don't exist,
                                # initialize weights with a stable random tensor instead of empty memory.
                                need_init = False
                                if weight_scale is None:
                                    need_init = True
                                else:
                                    try:
                                        need_init = torch.all(
                                            weight_scale == torch.finfo(torch.float32).min
                                        ).item()
                                    except Exception:
                                        # If comparison fails for any reason, be conservative and skip
                                        need_init = False
                                if need_init:
                                    # weight currently has shape [N, K] prior to post-load processing
                                    rand_w = torch.randn((output_n, input_k),
                                                         dtype=torch.float32,
                                                         device=weight.device)
                                    rand_w.clamp_(-2.0, 2.0)
                                    weight.copy_(rand_w.to(weight.dtype))
                                    if weight_scale is not None:
                                        weight_scale.fill_(1.0)
                        except Exception as e:
                            get_logger().warning(
                                f"FP8 dummy init failed for {module.__class__.__name__}: {e}"
                            )
                if isinstance(quant_method, QuantizeMethodBase) and hasattr(
                        quant_method, "process_weights_after_loading"):
                    quant_method.process_weights_after_loading(module)
                    
    def init_kv_cache(self, use_gpu_block_mgr: bool=False):
        assert not self.cache_config.cache_dtype.startswith("fp8") # flash attn supports only fp16 & bf16
        if self.cache_config.num_gpu_blocks is None:
            self.num_cache_blocks = self.determine_kv_cache_blocks()
            self.cache_config.num_gpu_blocks = self.num_cache_blocks
            get_logger().info(f"kv cache num_gpu_blocks: {self.cache_config.num_gpu_blocks}")
        else:
            self.num_cache_blocks = self.cache_config.num_gpu_blocks
            
        self.kv_cache = MHATokenToKVPool(
            self.num_cache_blocks,
            self.cache_config.block_size,
            self.model_config.dtype,
            self.model_config.num_kv_heads,
            self.model_config.hidden_size // self.model_config.num_heads,
            self.num_layers,
            self.device,
        )
        
        _log_memory_usage("After initializing kv cache")
        
        # TODO: fix this magic number
        self.max_running_reqs = self.num_cache_blocks * self.cache_config.block_size // 100 + 1
        
        if use_gpu_block_mgr:
            self.block_mgr = GPUBlockManager(self.model_config, self.cache_config, self.max_running_reqs, self.device)
        else:
            self.block_mgr = CPUBlockManager(self.model_config, self.cache_config, self.max_running_reqs, self.device)
    
    def get_num_cache_blocks(self):
        return self.num_cache_blocks
    
    def get_block_mgr(self):
        return self.block_mgr

    def memory_profile(self, batch_size: int):
        # use prefill to simulate a batch decoding to get memory profile
        attn_metadata = make_prefill_meta(batch_size, self.cache_config.block_size)
        kv_cache = torch.tensor([])
        for layer_id in range(self.num_layers):
            positions = torch.ones(batch_size, dtype=torch.long, device=self.device)
            hidden_states = torch.randn((batch_size, self.model_config.hidden_size), dtype=self.model_config.dtype)
            operator = self.operators[layer_id]
            # Use dummy request IDs to satisfy profile-driven gating during profiling.
            dummy_request_ids = list(range(batch_size))
            operator.forward(positions, hidden_states, kv_cache, attn_metadata, request_ids=dummy_request_ids)
            
    def determine_kv_cache_blocks(self) -> int:
        torch.cuda.empty_cache()
                
        self.memory_profile(get_global_engine_config().max_batch_size_attn)      
        torch.cuda.synchronize()
        
        _log_memory_usage("After profile run")
        
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        
        peak_memory = self.init_gpu_memory - free_gpu_memory
        cache_block_size = self.model_config.hidden_size // self.model_config.num_heads \
                            * self.model_config.num_kv_heads * self.cache_config.block_size * 2 * 2 # 2 for kv, 2 for fp16/bf16
        
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization - peak_memory) 
            // cache_block_size // len(self.model_config.layer_ids)
        )
        
        return num_gpu_blocks
    
    def build_cuda_graph_executor(self):
        if self.enable_cuda_graph:
            self.cuda_graph_executor = CUDAGraphAttnExecutor(self.model_config, self.cache_config, self)
            self.cuda_graph_executor.create_cuda_graph_buffers()
            self.cuda_graph_executor.capture()
            _log_memory_usage("After build CUDA graphs")
            
    def warmup(self, batch_size: int):
        get_logger().info(f"Attention warmup start, batch size {batch_size}")
        batch = make_attention_dummy_batch(0, batch_size, self.model_config.hidden_size, self.model_config.max_seq_len)
        meta = self.block_mgr.pack_flash_attn_metadata(batch.to_metadata_c(), batch, dummy_cache=True)
        get_logger().info(f"Attention warmup meta block table shape: {meta.block_tables.shape}")
        for layer_id in self.model_config.layer_ids:
            # get_logger().info(f"Attention warmup layer {layer_id} start")
            for _ in range(2):
                # Pass batch.req_ids so profile-driven gating receives request IDs.
                self.execute_eager(layer_id, batch.seq_lens_tensor.to(torch.long), batch.data, meta, request_ids=batch.req_ids)
            # get_logger().info(f"Attention warmup layer {layer_id} done")
                
        get_logger().info("Attention warmup done")
    
    def execute_eager(
        self,
        layer_id: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        request_ids: Optional[List[int]] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        vid = self.layer_mappings[layer_id]
        outputs, topk_weights, topk_ids = self.operators[vid].forward(
            positions, 
            hidden_states, 
            self.kv_cache.get_kv_buffer(vid), 
            attn_metadata,
            request_ids=request_ids,
        )
        return outputs, topk_weights, topk_ids
        
    @nvtx_range("AttnExecutor.execute")
    def execute(self, batch: AttentionForwardBatch) -> AttentionForwardResult:
        if self.enable_cuda_graph and batch.metadata.num_decode_tokens <= get_global_engine_config().max_attn_graph_bsz:
            outputs, topk_weights, topk_ids = self.cuda_graph_executor.run(batch.layer_id, batch.positions, batch.data, batch.metadata)
            # TODO: if overlap schedule is enabled, we need to copy results out to leave the output buffer free for the next batch
            # The copy process can be optimized by using a buffer pool
            outputs = outputs.clone()
            topk_weights = topk_weights.clone()
            topk_ids = topk_ids.clone()
        else:
            outputs, topk_weights, topk_ids = self.execute_eager(batch.layer_id, batch.positions, batch.data, batch.metadata, request_ids=batch.req_ids)
            
        return AttentionForwardResult(
            hiddens=outputs,
            expert_weights=topk_weights,
            expert_ids=topk_ids,
            sync_event=None,
        )
    
    @staticmethod
    def build(model_config: ModelConfig, cache_config: DmoeCacheConfig, gate_profile_bytes: Optional[bytes] = None) -> "Executor":
        if model_config.tp_size > 1:
            return ParallelAttnExecutor(model_config, cache_config, gate_profile_bytes=gate_profile_bytes)
        else:
            return AttnExecutor(model_config, cache_config, gate_profile_bytes=gate_profile_bytes)
        
class ExpertsExecutor(Executor):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        expert_cls = MoEExperts if get_global_engine_config().enable_grouped_gemm else MoEExpertsSerial
        self.type = ExecutorType.EXPERTS_EXEC
        # Build quantization config for MoE experts (Serial only) if requested
        moe_quant_config = None
        use_deep_gemm_fp8 = False
        
        try:
            method = getattr(self.model_config, "moe_linear_quant", None)
            if method and method != "none":
                if method == "fp8":
                    if expert_cls is MoEExpertsSerial:
                        moe_quant_config = Fp8Config(activation_scheme="dynamic")
                        get_logger().info(f"Successfully built FP8 quant config for MoE experts (Serial).")
                    else:
                        use_deep_gemm_fp8 = True
                        get_logger().info(f"Enabled deep_gemm FP8 for MoE experts (Grouped).")
                else:
                    moe_quant_config = None
        except Exception as e:
            get_logger().warning(
                f"Failed to build MoE quantization config '{getattr(self.model_config, 'moe_linear_quant', None)}': {e}. Falling back to unquantized."
            )
            moe_quant_config = None
        # Create operators
        self.operators = []
        for _ in range(self.num_layers):
            if expert_cls is MoEExpertsSerial:
                self.operators.append(
                    MoEExpertsSerial(
                        self.model_config.hidden_size,
                        self.model_config.intermediate_size,
                        self.model_config.num_experts_per_rank,
                        max_batch_size=get_global_engine_config().max_batch_size_expert,
                        quant_config=moe_quant_config,
                    )
                )
            else:
                grouped_cls = MoEExpertsDeepGemmFP8 if use_deep_gemm_fp8 else MoEExperts
                self.operators.append(
                    grouped_cls(
                        self.model_config.hidden_size,
                        self.model_config.intermediate_size,
                        self.model_config.num_experts_per_rank,
                        max_batch_size=get_global_engine_config().max_batch_size_expert,
                    )
                )
        # DisagMoE hacks:
        # 1. for vllm's fp8, use randn dummy weights rather than empty weights
        # 2. call process_weights_after_loading to match quantization kernel layouts
        if moe_quant_config is not None:
            for operator in self.operators:
                for _, module in operator.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is None:
                        continue
                    if quant_method.__class__.__name__ in (
                            "Fp8LinearMethod",
                            "PTPCFp8LinearMethod",
                            "ModelOptFp8LinearMethod",
                    ):
                        weight = getattr(module, "weight", None)
                        weight_scale = getattr(module, "weight_scale",
                                               getattr(module, "weight_scale_inv", None))
                        input_k = getattr(module, "input_size_per_partition", None)
                        output_n = getattr(module, "output_size_per_partition", None)
                        if weight is not None and input_k is not None and output_n is not None:
                            with torch.no_grad():
                                need_init = False
                                if weight_scale is None:
                                    need_init = True
                                else:
                                    try:
                                        need_init = torch.all(
                                            weight_scale == torch.finfo(torch.float32).min
                                        ).item()
                                    except Exception:
                                        need_init = False
                                if need_init:
                                    rand_w = torch.randn((output_n, input_k),
                                                            dtype=torch.float32,
                                                            device=weight.device)
                                    rand_w.clamp_(-2.0, 2.0)
                                    weight.copy_(rand_w.to(weight.dtype))
                                    if weight_scale is not None:
                                        weight_scale.fill_(1.0)
                    if isinstance(quant_method, QuantizeMethodBase) and hasattr(
                            quant_method, "process_weights_after_loading"):
                        quant_method.process_weights_after_loading(module)
        
    
    def warmup(self, batch_size: int):
        self._static_bs_cuda = torch.zeros((self.model_config.num_experts_per_rank, ), dtype=torch.int64, device="cuda")
        
        input = torch.zeros((batch_size, self.model_config.hidden_size), device="cuda")
        batch_sizes = torch.tensor([batch_size // self.model_config.num_experts_per_rank] * self.model_config.num_experts_per_rank,
            dtype=torch.int64,
            # NOTE(hogura|20241014): cuBLAS grouped_gemm requires batch_sizes to be on cpu
            device="cuda" if ENV_VARS["GROUPED_GEMM_CUTLASS"] else "cpu")
        for layer_id in self.model_config.layer_ids:
            for _ in range(2):
                _ = self.execute(layer_id, batch_size, input, batch_sizes)

    @nvtx_range("ExpertsExecutor.execute")
    def execute(self, layer_id: int, num_tokens: int, hidden_states: Tensor, batch_sizes: Tensor) -> Tensor:
        vid = self.layer_mappings[layer_id]
        operator = self.operators[vid]
        outputs = operator.forward(num_tokens, hidden_states, batch_sizes)
        return outputs
    
class ParallelAttnExecutor(AttnExecutor):
    
    def __init__(self, model_config: ModelConfig, cache_config: DmoeCacheConfig, gate_profile_bytes: Optional[bytes] = None):
        Executor.__init__(self, model_config)
        self.type = ExecutorType.ATTENTION_EXEC
        self.cache_config = cache_config
        self.gate_profile_bytes: Optional[bytes] = gate_profile_bytes
        # Build quantization config for attention QKV if requested
        qkv_quant_config = None
        try:
            method = getattr(self.model_config, "attn_qkv_quant", None)
            if method and method != "none":
                if method == "fp8":
                    # Use defaults; requires activation_scheme at construction time
                    qkv_quant_config = Fp8Config(activation_scheme="dynamic")
                    get_logger().info(f"Successfully built FP8 quant config for QKV.")
                else:
                    # Handle other methods as needed
                    qkv_quant_config = None
        except Exception as e:
            get_logger().warning(
                f"Failed to build QKV quantization config '{getattr(self.model_config, 'attn_qkv_quant', None)}': {e}. Falling back to unquantized."
            )
            qkv_quant_config = None
        self.operators = [
            MoEAttention(
                layer_id,
                self.model_config.hidden_size, 
                self.model_config.num_heads, 
                self.model_config.num_kv_heads, 
                self.model_config.num_experts,
                tp_size=model_config.tp_size,
                tp_rank=model_config.rank,
                quant_config_qkv=qkv_quant_config,
                gate_profile_bytes=self.gate_profile_bytes,
            ) for layer_id in range(self.num_layers)
        ]
        assert not cache_config.cache_dtype.startswith("fp8") # flash attn supports only fp16 & bf16
