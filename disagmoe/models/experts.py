import torch
from typing import override, List
from grouped_gemm.backend import gmm
from disagmoe.utils.constants import MAX_BATCH_SIZE

class MoEExperts(torch.nn.Module):
    
    def __init__(
        self, 
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
            
        params_dtype = torch.get_default_dtype()
        assert params_dtype == torch.bfloat16, "Only bf16 is supported for now"
        self.create_weights(params_dtype)
        self.gmm_with_cache = None
        self.gmm = gmm
        self.gmm_cache_max_batch_size = max_batch_size
        self.create_grouped_gemm_cache(params_dtype, enable_cutlass_cache, max_batch_size)
        
    def create_weights(self, params_dtype: torch.dtype):
        self.w13_weight = torch.nn.Parameter(torch.randn(self.num_experts,
                                                    self.hidden_size,
                                                    self.intermediate_size * 2,
                                                    dtype=params_dtype).cuda(),
                                        requires_grad=False)
        self.register_parameter("w13_weight", self.w13_weight)
        
        self.w2_weight = torch.nn.Parameter(torch.randn(self.num_experts,
                                                    self.intermediate_size,
                                                    self.hidden_size,
                                                    dtype=params_dtype).cuda(),
                                        requires_grad=False)
        self.register_parameter("w2_weight", self.w2_weight)
        
        self.act_fn = torch.nn.SiLU(inplace=True)

    def create_grouped_gemm_cache(self, params_dtype, enable_cutlass_cache, max_batch_size):
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
            up = self.gmm_with_cache(hiddens, self.w13_weight, batch_sizes, c=self.cache_up)
            up = self.act_fn(up[:bs, :self.intermediate_size]) * up[:bs, self.intermediate_size:]
            down = self.gmm_with_cache(up, self.w2_weight, batch_sizes, c=self.cache_down)
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
        super().__init__(hidden_size, intermediate_size, num_experts, tp_size, enable_cutlass_cache=False)
        
    @override
    def forward(self, num_tokens: int, hiddens: torch.Tensor, batch_sizes: List[int]):
        
        def calc(input, local_expert_id: int):
            up = torch.matmul(input, self.w13_weight[local_expert_id])
            up = self.act_fn(up[:, :self.intermediate_size]) * up[:, self.intermediate_size:]
            down = torch.matmul(up, self.w2_weight[local_expert_id])
            return down
        
        if len(batch_sizes) == 1:
            return calc(hiddens, 0)
        
        s = 0
        results = []
        for i, bs in enumerate(batch_sizes):
            if bs == 0:
                continue
            cur_hiddens = hiddens[s: s + bs]
            results.append(calc(cur_hiddens, i))
            s += bs
        
        if len(results) == 1:
            return results[0]
        return torch.cat(results)