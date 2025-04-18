import torch
from typing import override
from grouped_gemm.backend import gmm
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from disagmoe.utils.constants import MAX_BATCH_SIZE

class MoEExperts(torch.nn.Module):
    
    def __init__(self, 
                 hidden_size: int, 
                 intermediate_size: int,
                 num_experts: int, 
                 tp_size: int = 1,
                 enable_cutlass_cache: bool = True,
                 max_batch_size: int = MAX_BATCH_SIZE
                ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.tp_size = tp_size
        assert tp_size == 1, "Not implemented TP for experts yet"
        if self.tp_size > 1:
            assert self.intermediate_size % self.tp_size == 0
            self.intermediate_size //= self.tp_size
        self.create_weights(enable_cutlass_cache=enable_cutlass_cache,
                            max_batch_size=max_batch_size)
        self.gmm_with_cache = None
        self.gmm = gmm
        self.gmm_cache_max_batch_size = max_batch_size
        
    def create_weights(self, params_dtype: torch.dtype = None, 
                       enable_cutlass_cache: bool = False,
                       max_batch_size: int = MAX_BATCH_SIZE):
        if params_dtype == None:
            # FIXME(hogura|20241014): maybe use torch.get_default_dtype
            params_dtype = torch.bfloat16
            
        # Fused gate_up_proj (column parallel)
        self.w13_weight = torch.nn.Parameter(torch.randn(self.num_experts,
                                                    self.hidden_size,
                                                    self.intermediate_size * 2,
                                                    dtype=params_dtype).cuda(),
                                        requires_grad=False)
        self.register_parameter("w13_weight", self.w13_weight)

        # down_proj (row parallel)
        self.w2_weight = torch.nn.Parameter(torch.randn(self.num_experts,
                                                    self.intermediate_size,
                                                    self.hidden_size,
                                                    dtype=params_dtype).cuda(),
                                        requires_grad=False)
        self.register_parameter("w2_weight", self.w2_weight)
        
        # self.w3_weight = torch.nn.Parameter(torch.randn(self.num_experts,
        #                                             self.hidden_size,
        #                                             self.intermediate_size,
        #                                             dtype=params_dtype).cuda(),
        #                                 requires_grad=False)
        # self.register_parameter("w3_weight", self.w3_weight)
        
        self.act_fn = torch.nn.SiLU(inplace=True)

        self.cache_up = torch.empty((max_batch_size, self.intermediate_size * 2), dtype=params_dtype, device=torch.device("cuda"))
        self.cache_down = torch.empty((max_batch_size, self.hidden_size), dtype=params_dtype, device=torch.device("cuda"))
        if enable_cutlass_cache:
            from grouped_gemm.backend import get_arguments, gmm_with_arguments

            self.cutlass_workspace_size, self.arguments_ptr = get_arguments(
                self.num_experts, torch.device("cuda"))
            self.cutlass_workspace = torch.empty(
                [self.cutlass_workspace_size], dtype=torch.uint8, device=torch.device("cuda"))

            def _gmm(hiddens, weight, batch_sizes, **kwargs):
                return gmm_with_arguments(hiddens, weight, batch_sizes, self.cutlass_workspace, self.arguments_ptr, **kwargs)
            
            self.gmm_with_cache = _gmm
        
    def forward(self, bs: int, hiddens: torch.Tensor, batch_sizes: torch.Tensor):
        output = None
        if bs < self.gmm_cache_max_batch_size and self.gmm_with_cache is not None:
            up = self.gmm(hiddens, self.w13_weight, batch_sizes, c=self.cache_up)
            up_1 = up[:bs, :self.intermediate_size]
            up_3 = up[:bs, self.intermediate_size:]
            up_1 = self.act_fn(up_1)
            up_1 *= up_3
            down = self.gmm(up_1, self.w2_weight, batch_sizes, c=self.cache_down)
            output = down[:bs]
        else:
            up = self.gmm(hiddens, self.w13_weight, batch_sizes)
            up = self.act_fn(up[:, :self.intermediate_size]) * up[:, self.intermediate_size:]
            down = self.gmm(up, self.w2_weight, batch_sizes)
            output = down
        return output

class MoEExpertsSerial(MoEExperts):
    
    def __init__(self, hidden_size, intermediate_size, num_experts, tp_size = 1, 
                 max_batch_size: int = MAX_BATCH_SIZE):
        super().__init__(hidden_size, intermediate_size, num_experts, tp_size)
        
    @override
    def forward(self, num_tokens: int, hiddens: torch.Tensor, batch_sizes: torch.Tensor):
        results = []
        s = 0
        single = batch_sizes.shape[0] == 1
        for i, bs in enumerate(batch_sizes):
            if single:
                cur_hiddens = hiddens
            else:
                cur_hiddens = hiddens[s: s + bs]
            
            result = self.act_fn(torch.matmul(cur_hiddens, self.w1_weight[i])) * torch.matmul(cur_hiddens, self.w3_weight[i])
            result = torch.matmul(result, self.w2_weight[i])
            results.append(result)
            
            s += bs
        return torch.cat(results)