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
from disagmoe.models.experts import (
    MoEExpertsCUTLASS,
    MoEExpertsCUTLASSFP8,
    MoEExpertsDeepGemmBF16,
    MoEExpertsDeepGemmFP8,
    MoEExpertsSerial,
)
from disagmoe.config import ModelConfig, CacheConfig as DmoeCacheConfig
from disagmoe.utils.utils import nvtx_range, _log_memory_usage
from disagmoe.utils.logger import get_logger
from disagmoe.models.utils import make_attention_dummy_batch, make_prefill_meta, make_expert_dummy_inputs
from disagmoe.block_manager.block_manager import GPUBlockManager, CPUBlockManager, BaseBlockManager
from disagmoe.block_manager.mem_pool import MHATokenToKVPool
from disagmoe.frontend.datatypes import AttentionForwardBatch, AttentionForwardResult, ExpertForwardBatch
from disagmoe.frontend.engine_utils import get_global_engine_config
from disagmoe.executor.cuda_graph import CUDAGraphAttnExecutor, CUDAGraphExpertsExecutor
from disagmoe.ops.cuda_graph import copy_graph_results_cuda
from disagmoe.frontend.datatypes import BatchMetadata
from disagmoe.utils.gdr_context import use_gdrcopy_optimization, GdrDoubleBuffer

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
        self.has_profile_gating = gate_profile_bytes is not None and len(gate_profile_bytes) > 0
        
        self.init_model()
        self.init_kv_cache()
        
    def init_model(self):
        free_memory, total_memory = torch.cuda.mem_get_info()
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
        
        # Build quantization config for shared experts
        shared_quant_config = None
        try:
            # Shared experts use the same quant as attention QKV (they run on attn GPU)
            shared_method = getattr(self.model_config, "attn_qkv_quant", None)
            if shared_method and shared_method != "none":
                if shared_method == "fp8":
                    shared_quant_config = Fp8Config(activation_scheme="dynamic")
                    get_logger().info(f"Successfully built FP8 quant config for shared experts.")
        except Exception as e:
            get_logger().warning(
                f"Failed to build shared expert quantization config: {e}. Falling back to unquantized."
            )
            shared_quant_config = None

        self.operators = [
            MoEAttention(
                layer_id,
                self.model_config.hidden_size, 
                self.model_config.head_dim,
                self.model_config.num_heads, 
                self.model_config.num_kv_heads, 
                self.model_config.num_experts,
                self.model_config.top_k,
                cache_config=self.vllm_cache_config,
                quant_config_qkv=qkv_quant_config,
                gate_profile_bytes=self.gate_profile_bytes,
                num_shared_experts=getattr(self.model_config, "num_shared_experts", 0),
                shared_expert_intermediate_size=getattr(self.model_config, "shared_expert_intermediate_size", None),
                quant_config_shared=shared_quant_config,
                intermediate_size=getattr(self.model_config, "intermediate_size", None),
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
            self.model_config.head_dim,
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
            dummy_request_ids = torch.arange(batch_size, dtype=torch.int64, device=self.device) if self.has_profile_gating else None
            operator.forward(positions, hidden_states, kv_cache, attn_metadata, request_ids=dummy_request_ids)
            
    def determine_kv_cache_blocks(self) -> int:
        torch.cuda.empty_cache()
                
        self.memory_profile(get_global_engine_config().max_batch_size_attn)      
        torch.cuda.synchronize()
        
        _log_memory_usage("After profile run")
        
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        
        peak_memory = self.init_gpu_memory - free_gpu_memory
        cache_block_size = self.model_config.head_dim \
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
            _log_memory_usage("After build attn CUDA graphs")
            
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
        positions: Tensor, 
        hidden_states: Tensor, 
        attn_metadata: FlashAttentionMetadata, 
        request_ids = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.has_profile_gating and request_ids is not None and not isinstance(request_ids, Tensor):
            request_ids = torch.tensor(request_ids, dtype=torch.int64, device="cuda")
        
        vid = self.layer_mappings[layer_id]
        outputs, topk_weights, topk_ids = self.operators[vid].forward(
            positions, 
            hidden_states, 
            self.kv_cache.get_kv_buffer(vid), 
            attn_metadata,
            request_ids=request_ids,
        )
        return outputs, topk_weights, topk_ids
    
    def execute_normal(self, batch: AttentionForwardBatch) -> AttentionForwardResult:
        outputs, topk_weights, topk_ids = self.execute_eager(batch.layer_id, batch.positions, batch.data, batch.metadata, request_ids=batch.req_ids)
        
        return AttentionForwardResult(
            hiddens=outputs,
            expert_weights=topk_weights,
            expert_ids=topk_ids,
            sync_event=None,
        )
    
    def execute_graph(self, batch: AttentionForwardBatch) -> AttentionForwardResult:
        req_ids_tensor = None
        if self.has_profile_gating and batch.req_ids is not None:
            req_ids_tensor = torch.tensor(batch.req_ids, dtype=torch.int64, device="cuda")
        staging_outputs, staging_topk_weights, staging_topk_ids = self.cuda_graph_executor.run(batch.layer_id, batch.positions, batch.data, batch.metadata, request_ids=req_ids_tensor)
        outputs = torch.empty_like(staging_outputs)
        
        if batch.expert_ids_buffer is not None:
            expert_ids = batch.expert_ids_buffer
            expert_weights = batch.expert_weights_buffer
        else:
            expert_ids = torch.empty_like(staging_topk_ids)
            expert_weights = torch.empty_like(staging_topk_weights)
            
        copy_graph_results_cuda(
            staging_outputs, 
            staging_topk_ids, 
            staging_topk_weights, 
            outputs, 
            expert_ids, 
            expert_weights, 
            batch.num_tokens,
        )
        
        return AttentionForwardResult(
            hiddens=outputs,
            expert_weights=expert_weights,
            expert_ids=expert_ids,
            sync_event=None,
        )

    @nvtx_range("AttnExecutor.execute")
    def execute(self, batch: AttentionForwardBatch) -> AttentionForwardResult:
        if self.enable_cuda_graph and batch.metadata.num_decode_tokens <= get_global_engine_config().max_attn_graph_bsz:
            return self.execute_graph(batch)
        else:
            return self.execute_normal(batch)
    
    @staticmethod
    def build(model_config: ModelConfig, cache_config: DmoeCacheConfig, gate_profile_bytes: Optional[bytes] = None) -> "Executor":
        if model_config.tp_size > 1:
            return ParallelAttnExecutor(model_config, cache_config, gate_profile_bytes=gate_profile_bytes)
        else:
            return AttnExecutor(model_config, cache_config, gate_profile_bytes=gate_profile_bytes)
        
class ExpertsExecutor(Executor):

    def __init__(self, model_config: ModelConfig, local_to_global_expert_rank: List[int], global_to_local_expert_rank: List[int]):
        super().__init__(model_config)
        self.type = ExecutorType.EXPERTS_EXEC
        self.local_num_experts = len(local_to_global_expert_rank)
        # Build quantization config for MoE experts (Serial only) if requested
        self.expert_ids = torch.arange(self.local_num_experts, device="cpu", dtype=torch.int32)
        self.local_to_global_expert_rank = local_to_global_expert_rank
        self.global_to_local_expert_rank = global_to_local_expert_rank
        self.cuda_graph_executor: CUDAGraphExpertsExecutor = None
        
        self.token_m_indices_gdr = GdrDoubleBuffer(get_global_engine_config().max_batch_size_expert, dtype=torch.int32, device="cuda")
        self.batch_sizes_gdr = GdrDoubleBuffer(self.local_num_experts, dtype=torch.int64, device="cuda")
        
        self.static_batch_sizes = torch.zeros((self.local_num_experts,), dtype=torch.int64, device="cuda")
        self.static_m_indices = torch.zeros((get_global_engine_config().max_batch_size_expert,), dtype=torch.int32, device="cuda")
        
        self.quant_method = getattr(self.model_config, "moe_linear_quant", None) or "none"
        if self.quant_method == "none":
            get_logger().info("Using unquantized (BF16) MoE experts.")
        elif self.quant_method == "fp8":
            if getattr(get_global_engine_config(), "less_than_sm90", False):
                get_logger().info("MoE FP8 experts: CUTLASS grouped GEMM (pre-sm90).")
            else:
                get_logger().info("MoE FP8 experts: DeepGEMM (sm90+).")
        else:
            raise ValueError(f"Invalid MoE linear quantization method: {self.quant_method}")
            
        # Create operators
        self.operators = []
        if get_global_engine_config().enable_cuda_graph_expert:
            get_logger().info(f"Enabled CUDA graphs for experts, need to capture graphs.")
            
        self.expert_cls = self.get_expert_cls()
        
        for _ in range(self.num_layers):
            if self.expert_cls is MoEExpertsSerial:
                self.operators.append(
                    MoEExpertsSerial(
                        self.model_config.hidden_size,
                        self.model_config.intermediate_size,
                        self.local_num_experts,
                        max_batch_size=get_global_engine_config().max_batch_size_expert,
                        quant_config=None,
                    )
                )
            else:
                self.operators.append(
                    self.expert_cls(
                        self.model_config.hidden_size,
                        self.model_config.intermediate_size,
                        self.local_num_experts,
                        max_batch_size=get_global_engine_config().max_batch_size_expert,
                    )
                )
                        
    def get_expert_cls(self):
        cfg = get_global_engine_config()
        if not cfg.enable_grouped_gemm:
            expert_cls = MoEExpertsSerial
        elif getattr(cfg, "less_than_sm90", False):
            # Pre-SM90: DeepGEMM is unavailable; use CUTLASS grouped GEMM (bf16 or fp8).
            expert_cls = (
                MoEExpertsCUTLASSFP8 if self.quant_method == "fp8" else MoEExpertsCUTLASS
            )
        elif self.quant_method == "fp8":
            expert_cls = MoEExpertsDeepGemmFP8
        else:
            expert_cls = MoEExpertsDeepGemmBF16
        
        get_logger().info(f"Using expert class: {expert_cls.__name__}")
        return expert_cls
    
    def build_cuda_graph_executor(self):
        self.cuda_graph_executor = CUDAGraphExpertsExecutor(
            self.model_config,
            self.local_to_global_expert_rank,
            self.global_to_local_expert_rank,
            self)
        self.cuda_graph_executor.create_cuda_graph_buffers()
        self.cuda_graph_executor.capture()
        _log_memory_usage("After build CUDA experts graphs")
    
    def prepare_bsz_and_indices(self, meta_c: BatchMetadata) -> Tuple[Optional[Union[Tensor, List[int]]], Optional[Tensor]]:
        batch_sizes = self.static_batch_sizes
        m_indices = self.static_m_indices[:meta_c.num_tokens()]
        
        if self.expert_cls is MoEExpertsSerial:
            batch_sizes = list(meta_c.get_expert_batch_sizes(self.model_config.num_experts))
            batch_sizes = [batch_sizes[i] for i in self.local_to_global_expert_rank]
            
        if self.expert_cls in (MoEExpertsCUTLASS, MoEExpertsCUTLASSFP8):
            batch_sizes_list = list(meta_c.get_expert_batch_sizes(self.model_config.num_experts))
            batch_sizes_list = [batch_sizes_list[i] for i in self.local_to_global_expert_rank]
            if use_gdrcopy_optimization:
                batch_sizes_gdr_handle = self.batch_sizes_gdr.get_one_handle()
                batch_sizes_gdr_handle.copy_from_host_int64(batch_sizes_list)
                batch_sizes = batch_sizes_gdr_handle.tensor[:len(batch_sizes_list)]
            else:
                batch_sizes = torch.tensor(
                    batch_sizes_list,
                    dtype=torch.int64, device="cuda"
                )
        
        # DeepGEMM-based experts (both BF16 and FP8) expect m_indices
        if self.expert_cls in [MoEExpertsDeepGemmBF16, MoEExpertsDeepGemmFP8]:
            m_indices_list = meta_c.get_token_expert_indices(self.model_config.num_experts, self.global_to_local_expert_rank)
            if use_gdrcopy_optimization:
                token_m_indices_buffer_gdr = self.token_m_indices_gdr.get_one_handle()
                token_m_indices_buffer_gdr.copy_from_host_int32(m_indices_list)
                m_indices = token_m_indices_buffer_gdr.tensor[:len(m_indices_list)]
            else:
                m_indices = torch.tensor(m_indices_list, dtype=torch.int32, device="cuda")

        return batch_sizes, m_indices
    
    def warmup(self, batch_size: int):
        hiddens, batch_sizes, m_indices = make_expert_dummy_inputs(
            batch_size=batch_size,
            hidden_size=self.model_config.hidden_size,
            num_experts_per_rank=self.local_num_experts,
            expert_ids=self.expert_ids,
        )

        for layer_id in self.model_config.layer_ids:
            batch = ExpertForwardBatch(
                layer_id=layer_id,
                data=hiddens,
                num_tokens=batch_size,
                meta_c=None,
                proc_func=None,
                post_proc_func=None,
                batch_sizes=batch_sizes,
                m_indices=m_indices
            )
            for _ in range(2):
                _ = self.execute(batch)

    @nvtx_range("ExpertsExecutor.execute")
    def execute(self, batch: ExpertForwardBatch) -> Tensor:
        assert batch.num_tokens <= get_global_engine_config().max_batch_size_expert, f"batch size {batch.num_tokens} exceeds max batch size {get_global_engine_config().max_batch_size_expert}"
        vid = self.layer_mappings[batch.layer_id]
        
        # CUDA graph path for DeepGEMM and CUTLASS grouped-GEMM experts
        if self.expert_cls in [
            MoEExpertsDeepGemmBF16,
            MoEExpertsDeepGemmFP8,
            MoEExpertsCUTLASS,
            MoEExpertsCUTLASSFP8,
        ]:
            if get_global_engine_config().enable_cuda_graph_expert:
                outputs = self.cuda_graph_executor.run(vid, batch.data, batch.batch_sizes, batch.m_indices)
            else:
                outputs = self.execute_eager(batch)
        else:
            # Serial expert fallback (no CUDA graph)
            operator = self.operators[vid]
            outputs = operator.forward(batch.num_tokens, batch.data, batch.batch_sizes)
        return outputs

    @nvtx_range("ExpertsExecute.execute_eager")
    def execute_eager(self, batch: ExpertForwardBatch) -> Tensor:
        # used for capturing CUDA graph for the classes that doesn't do graph at model level
        vid = self.layer_mappings[batch.layer_id]
        
        # Dispatch: CUTLASS (bf16/fp8) uses batch_sizes; DeepGEMM uses m_indices
        if self.expert_cls in (MoEExpertsCUTLASS, MoEExpertsCUTLASSFP8):
            outputs = self.operators[vid].forward(batch.num_tokens, batch.data, batch.batch_sizes)
        elif self.expert_cls in [MoEExpertsDeepGemmBF16, MoEExpertsDeepGemmFP8]:
            outputs = self.operators[vid].forward(batch.num_tokens, batch.data, batch.m_indices)
        else:
            raise ValueError(f"Unsupported expert class for CUDA graph: {self.expert_cls}")
        
        return outputs

    
class ParallelAttnExecutor(AttnExecutor):
    
    def __init__(self, model_config: ModelConfig, cache_config: DmoeCacheConfig, gate_profile_bytes: Optional[bytes] = None):
        Executor.__init__(self, model_config)
        self.type = ExecutorType.ATTENTION_EXEC
        self.cache_config = cache_config
        self.gate_profile_bytes: Optional[bytes] = gate_profile_bytes
        self.has_profile_gating = gate_profile_bytes is not None and len(gate_profile_bytes) > 0
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
        # Build quantization config for shared experts (parallel path)
        shared_quant_config = None
        try:
            shared_method = getattr(self.model_config, "attn_qkv_quant", None)
            if shared_method and shared_method != "none":
                if shared_method == "fp8":
                    shared_quant_config = Fp8Config(activation_scheme="dynamic")
        except Exception:
            shared_quant_config = None

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
                num_shared_experts=getattr(self.model_config, "num_shared_experts", 0),
                shared_expert_intermediate_size=getattr(self.model_config, "shared_expert_intermediate_size", None),
                quant_config_shared=shared_quant_config,
                intermediate_size=getattr(self.model_config, "intermediate_size", None),
            ) for layer_id in range(self.num_layers)
        ]
        assert not cache_config.cache_dtype.startswith("fp8") # flash attn supports only fp16 & bf16
