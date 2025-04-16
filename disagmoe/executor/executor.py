import torch
import torch.distributed as dist

from torch import Tensor

from typing import override, Tuple, List, Union, Dict
from time import sleep
from enum import Enum

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig as VllmCacheConfig

from disagmoe.models.attention import MoEAttention
from disagmoe.models.experts import MoEExperts, MoEExpertsSerial
from disagmoe.config import ModelConfig, CacheConfig as DmoeCacheConfig
from disagmoe.utils.utils import nvtx_range
from disagmoe.models.utils import make_dummy_meta, make_prefill_meta
from disagmoe.frontend.datatypes import AttentionBatchMetadata
from vllm.attention.backends.flash_attn import FlashAttentionMetadata

from disagmoe_c import prepare_batch_infos
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
    
    def execute(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def initialize_cache(self, num_blocks: int) -> None:
        raise NotImplementedError()
    


class AttnExecutor(Executor):

    def __init__(self, model_config: ModelConfig, cache_config: DmoeCacheConfig):
        super().__init__(model_config)
        self.type = ExecutorType.ATTENTION_EXEC
        self.cache_config = cache_config
        self.vllm_cache_config = VllmCacheConfig(
            cache_dtype="auto",
            block_size=cache_config.block_size,
            gpu_memory_utilization=0, # useless in our case
            swap_space=0, #useless in our case
        )
        self.operators = [
            MoEAttention(
                layer_id,
                self.model_config.hidden_size, 
                self.model_config.num_heads, 
                self.model_config.num_kv_heads, 
                self.model_config.num_experts,
                self.model_config.top_k,
                cache_config=self.vllm_cache_config,
            ) for layer_id in range(self.num_layers)
        ]
        assert not cache_config.cache_dtype.startswith("fp8") # flash attn supports only fp16 & bf16
        
    @override
    def initialize_cache(self, num_blocks):
        self._make_kv_cache(
            self.num_layers,
            num_blocks,
            self.cache_config.block_size, 
            self.model_config.num_kv_heads, 
            self.model_config.hidden_size // self.model_config.num_heads,
        )
    
    def _make_kv_cache(self, num_layers, num_blocks, block_size, num_heads, head_size):
        data_type = self.model_config.dtype
        self.cache = torch.randn((num_layers, 2, num_blocks, block_size, num_heads, head_size), dtype=data_type)
    
    def profile_execute(self, batch_size: int):
        attn_metadata = make_prefill_meta(batch_size, self.cache_config.block_size)
        kv_cache = None
        for layer_id in range(self.num_layers):
            positions = torch.ones(batch_size, dtype=torch.long, device="cuda")
            hidden_states = torch.randn((batch_size, self.model_config.hidden_size), dtype=self.model_config.dtype)
            operator = self.operators[layer_id]
            operator.forward(positions, hidden_states, kv_cache, attn_metadata)

    @override
    @nvtx_range("AttnExecutor.execute")
    def execute(self,
                layer_id: int,
                positions: torch.Tensor,
                hidden_states: torch.Tensor,
                attn_metadata: AttentionMetadata) -> Tuple[Tensor, Tensor, Tensor]:
        vid = self.layer_mappings[layer_id]
        operator = self.operators[vid]
        outputs, topk_weights, topk_ids = operator.forward(
            positions, 
            hidden_states, 
            self.cache[vid], 
            attn_metadata
        )
        return outputs, topk_weights, topk_ids
    
    @staticmethod
    def build(model_config: ModelConfig, cache_config: DmoeCacheConfig) -> "Executor":
        if model_config.tp_size > 1:
            return ParallelAttnExecutor(model_config, cache_config)
        else:
            return AttnExecutor(model_config, cache_config)

class CUDAGraphAttnExecutor:
    
    def __init__(self, model_config: ModelConfig, cache_config: DmoeCacheConfig, attn_executor: AttnExecutor):
        self.model_config = model_config
        self.cache_config = cache_config
        self.attn_executor = attn_executor
        
    def create_cuda_graph_buffers(self):
        assert self.model_config.enable_cuda_graph_attn
        batch_size = self.model_config.max_batch_size_attn
        self.graphs: Dict[int, List[torch.cuda.CUDAGraph]] = {}
        self.static_outputs: Dict[int, List[Tuple[Tensor]]] = {}

        self.static_input = torch.zeros((batch_size, self.model_config.hidden_size), device="cuda")
        self.static_positions = torch.zeros(batch_size, dtype=torch.long, device="cuda")
        self.static_block_table = torch.zeros(
            (batch_size, self.model_config.max_seq_len // self.cache_config.block_size), 
            dtype=torch.int32, device="cuda")
        self.static_slot_mapping = torch.zeros((batch_size, ), dtype=torch.long, device="cuda")

        self.static_batch_info = torch.zeros((batch_size + batch_size + (batch_size + 1)), dtype=torch.int32, device="cuda")
        self.static_seq_lens = self.static_batch_info[ : batch_size]
        self.static_context_lens = self.static_batch_info[batch_size : batch_size + batch_size]
        self.static_seq_start_loc = self.static_batch_info[batch_size + batch_size : ]

        self.static_batch_infos: Dict[int, Tensor] = {}

        self.graph_batch_sizes = list(range(max(self.model_config.graph_stride, self.model_config.ep_size),
                                            batch_size + 1,
                                            self.model_config.graph_stride))
        self.graph_batch_sizes = [1] + self.graph_batch_sizes

        for layer_id in self.model_config.layer_ids:
            self.graphs[layer_id] = [torch.cuda.CUDAGraph() for _ in self.graph_batch_sizes]
            self.static_outputs[layer_id] = []
        
        for bs in self.graph_batch_sizes:
            self.static_batch_infos[bs] = torch.zeros((bs + bs + (bs + 1)), dtype=torch.int32, device="cuda")

    def _prepare_dummy_flash_attn_metadata(
            self,
            meta_c: AttentionBatchMetadata,
            meta_py: AttentionBatchMetadata,
            decode_seq_lens: List[int],
            mocking: bool = False,
        ) -> FlashAttentionMetadata:
        
        num_tokens = meta_py.num_decode_tokens + meta_py.num_prefill_tokens
        num_seqs = meta_py.num_prefill_seqs + meta_py.num_decode_tokens

        # print(f"meta_py {meta_py}, decode_seq_lens {self.decode_seq_lens}")

        max_num_blocks_per_seq = max(decode_seq_lens) // self.cache_config.block_size
        block_table_1d = torch.zeros(
            (num_seqs, max_num_blocks_per_seq), 
            dtype=torch.int32, device="cuda")
        
        self.static_slot_mapping[ : num_tokens].copy_(torch.arange(num_tokens, dtype=torch.long, device="cuda"))
        self.static_block_table[ : num_seqs, : max_num_blocks_per_seq].copy_(block_table_1d)
        slot_mapping_cuda = self.static_slot_mapping[ : num_tokens]
        block_table_cuda = self.static_block_table[ : num_seqs, : max_num_blocks_per_seq]

        # 2. prepare seqlens and start_locs
        # pack (seq_lens, context_lens, seq_start_loc) in the same tensor
        batch_info_cuda = prepare_batch_infos(meta_c, decode_seq_lens)


        # batch_info_len = batch_info_cuda.shape[0]

        # batch_size = num_tokens
        # static_batch_info = self.static_batch_infos[batch_size]
        # static_batch_info[ : batch_info_len].copy_(batch_info_cuda)
        # seq_lens_cuda = static_batch_info[ : num_seqs]
        # context_lens_cuda = static_batch_info[num_seqs : num_seqs + num_seqs]
        # seq_start_loc_cuda = static_batch_info[num_seqs + num_seqs : ]
        
        max_decode_seq_len = max(decode_seq_lens) if len(decode_seq_lens) > 0 else 0
        seq_lens_cuda = self.static_seq_lens[ : num_seqs]
        context_lens_cuda = self.static_context_lens[ : num_seqs]
        seq_start_loc_cuda = self.static_seq_start_loc[ : num_seqs + 1]
        seq_lens_cuda[ : num_seqs].copy_(batch_info_cuda[ : num_seqs])
        context_lens_cuda[ : num_seqs].copy_(batch_info_cuda[num_seqs : num_seqs + num_seqs])
        seq_start_loc_cuda[ : num_seqs + 1].copy_(batch_info_cuda[num_seqs + num_seqs : ])

        seq_lens = decode_seq_lens

        return FlashAttentionMetadata(
            0,
            0,
            num_tokens,
            slot_mapping_cuda,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_cuda,
            max_query_len=0,
            max_prefill_seq_len=0,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=None,
            seq_start_loc=seq_start_loc_cuda,
            context_lens_tensor=context_lens_cuda,
            block_tables=block_table_cuda,
            use_cuda_graph=True,
        )

    def capture(self):
        assert self.model_config.enable_cuda_graph_attn, "Attention CUDA Graph is not enabled."
        for layer_id in self.model_config.layer_ids:
            for graph, graph_batch_size in zip(self.graphs[layer_id], self.graph_batch_sizes):
                meta_py = make_dummy_meta(0, graph_batch_size)
                meta = self._prepare_dummy_flash_attn_metadata(meta_py.to_c(), meta_py, [self.model_config.max_seq_len] * graph_batch_size)

                def run_once() -> Tuple[Tensor, Tensor, Tensor]:
                    return self.attn_executor.execute(
                        layer_id, self.static_positions[ : graph_batch_size], 
                        self.static_input[ : graph_batch_size], meta
                    )

                for _ in range(2):
                    # warmup
                    torch.cuda.synchronize()
                    run_once()
                    torch.cuda.synchronize()

                with torch.cuda.graph(graph):
                    outputs = run_once()
                    
                torch.cuda.synchronize()

                self.static_outputs[layer_id].append(outputs)
                
                # warmup for the actual execution
                graph.replay()
                    
                torch.cuda.synchronize()

        print("cuda graph captured")

        self.test_graph()
        
        print("cuda graph tested")

    def _prepare_test_flash_attn_metadata(
            self,
            meta_c: AttentionBatchMetadata,
            meta_py: AttentionBatchMetadata,
            decode_seq_lens: List[int],
        ) -> FlashAttentionMetadata:
        
        num_tokens = meta_py.num_decode_tokens + meta_py.num_prefill_tokens
        num_seqs = meta_py.num_prefill_seqs + meta_py.num_decode_tokens

        # print(f"meta_py {meta_py}, decode_seq_lens {self.decode_seq_lens}")
        
        # 1. prepare block table
        block_table_1d = torch.zeros(
            (num_tokens + num_seqs * self.model_config.max_seq_len // self.cache_config.block_size, ), 
            dtype=torch.int32, device="cuda")

        slot_mapping_cuda = block_table_1d[-num_tokens : ].to(torch.int64)
        block_table_cuda = block_table_1d[ : -num_tokens].view(num_tokens, -1)

        # 2. prepare seqlens and start_locs
        # pack (seq_lens, context_lens, seq_start_loc) in the same tensor
        batch_infos_cuda = prepare_batch_infos(meta_c, decode_seq_lens)

        seq_lens_cuda, context_lens_cuda, seq_start_loc_cuda = \
            torch.split(batch_infos_cuda, [num_seqs, num_seqs, num_seqs + 1], dim=0)

        seq_lens = decode_seq_lens
            
        max_decode_seq_len = max(decode_seq_lens) if len(decode_seq_lens) > 0 else 0

        return FlashAttentionMetadata(
            0,
            0,
            num_tokens,
            slot_mapping_cuda,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_cuda,
            max_query_len=0,
            max_prefill_seq_len=0,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=None,
            seq_start_loc=seq_start_loc_cuda,
            context_lens_tensor=context_lens_cuda,
            block_tables=block_table_cuda,
            use_cuda_graph=self.model_config.enable_cuda_graph_attn,
        )

    def test_graph(self):
        for layer_id in self.model_config.layer_ids:
            for bs in range(1, self.model_config.max_batch_size_attn + 1):
                meta_py = make_dummy_meta(0, bs)
                meta = self._prepare_test_flash_attn_metadata(meta_py.to_c(), meta_py, [200] * bs)
                hiddens, expert_weights, expert_ids = self.run(layer_id, torch.zeros(bs, dtype=torch.long, device="cuda"), torch.randn(bs, self.model_config.hidden_size, device="cuda"), meta)
                torch.cuda.synchronize()
                _, reorder_ids = torch.sort(expert_ids.view(-1), stable=True)

    def _get_graph_by_batch_size(self, batch_size: int):
        for i, size in enumerate(self.graph_batch_sizes):
            if size >= batch_size:
                return i, size
        assert False, f"No available graph for batch size={batch_size}"

    def run(self, layer_id: int, positions: torch.Tensor, hidden_states: torch.Tensor, meta: FlashAttentionMetadata) -> Tuple[Tensor, Tensor, Tensor]:
        num_tokens = hidden_states.shape[0]
        graph_id, batch_size = self._get_graph_by_batch_size(num_tokens)
        self.static_input[ : num_tokens].copy_(hidden_states)
        self.static_positions[ : num_tokens].copy_(positions)
        self.static_slot_mapping[ : num_tokens].copy_(meta.slot_mapping)

        max_num_blocks = meta.block_tables.shape[1]
        self.static_block_table[ : num_tokens, : max_num_blocks].copy_(meta.block_tables)

        # static_batch_info = self.static_batch_infos[batch_size]

        # seq_lens_cuda  = static_batch_info[ : batch_size]
        # context_lens_cuda = static_batch_info[batch_size : batch_size + batch_size]
        # seq_start_loc_cuda = static_batch_info[batch_size + batch_size : ]
        # seq_lens_cuda[ : num_tokens].copy_(meta.seq_lens_tensor)
        # context_lens_cuda[ : num_tokens].copy_(meta.context_lens_tensor)
        # seq_start_loc_cuda[ : num_tokens + 1].copy_(meta.seq_start_loc)
        self.static_seq_lens[ : num_tokens].copy_(meta.seq_lens_tensor)
        self.static_context_lens[ : num_tokens].copy_(meta.context_lens_tensor)
        self.static_seq_start_loc[ : num_tokens + 1].copy_(meta.seq_start_loc)

        self.graphs[layer_id][graph_id].replay()

        outputs, topk_weights, topk_ids = self.static_outputs[layer_id][graph_id]

        return outputs[ : num_tokens], topk_weights[ : num_tokens], topk_ids[ : num_tokens]
        
class ExpertsExecutor(Executor):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        expert_cls = MoEExperts if model_config.enable_grouped_gemm else MoEExpertsSerial
        self.type = ExecutorType.EXPERTS_EXEC
        self.operators = [
            expert_cls(
                self.model_config.hidden_size,
                self.model_config.intermediate_size,
                self.model_config.num_experts_per_rank,
                max_batch_size=self.model_config.max_batch_size_expert
            ) for _ in range(self.num_layers)
        ]

    @override
    @nvtx_range("ExpertsExecutor.execute")
    def execute(self, layer_id: int, num_tokens: int, hidden_states: Tensor, batch_sizes: Tensor) -> Tensor:
        vid = self.layer_mappings[layer_id]
        operator = self.operators[vid]
        outputs = operator.forward(num_tokens, hidden_states, batch_sizes)
        return outputs
    
class ParallelAttnExecutor(AttnExecutor):
    
    def __init__(self, model_config: ModelConfig, cache_config: DmoeCacheConfig):
        Executor.__init__(self, model_config)
        self.type = ExecutorType.ATTENTION_EXEC
        self.cache_config = cache_config
        self.operators = [
            MoEAttention(
                layer_id,
                self.model_config.hidden_size, 
                self.model_config.num_heads, 
                self.model_config.num_kv_heads, 
                self.model_config.num_experts,
                tp_size=model_config.tp_size,
                tp_rank=model_config.rank,
            ) for layer_id in range(self.num_layers)
        ]
        assert not cache_config.cache_dtype.startswith("fp8") # flash attn supports only fp16 & bf16
