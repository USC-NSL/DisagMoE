import torch
from grouped_gemm.backend import gmm
from disagmoe.third_party.vllm.vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)

class MoEExperts(torch.nn.Module):
    
    def __init__(self, 
                 hidden_size: int, 
                 intermediate_size: int,
                 num_experts: int, 
                 tp_size: int = 1,
                ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.tp_size = tp_size
        if self.tp_size > 1:
            assert self.intermediate_size % self.tp_size == 0
            self.intermediate_size //= self.tp_size
        
    def create_weights(self, params_dtype: torch.dtype = None):
        if params_dtype == None:
            params_dtype = torch.get_default_dtype()
            
        # Fused gate_up_proj (column parallel)
        w1_weight = torch.nn.Parameter(torch.empty(self.num_experts,
                                                    self.hidden_size,
                                                    self.intermediate_size,
                                                    dtype=params_dtype),
                                        requires_grad=False)
        self.register_parameter("w1_weight", w1_weight)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(self.num_experts,
                                                    self.intermediate_size,
                                                    self.hidden_size,
                                                    dtype=params_dtype),
                                        requires_grad=False)
        self.register_parameter("w2_weight", w2_weight)
        
    def forward(self, hiddens: torch.Tensor, batch_sizes: torch.Tensor):
        # Here use grouped gemm, tokens must be permuted by corresponding expert_id
        up = gmm(hiddens, self.w1_weight, batch_sizes)
        down = gmm(up, self.w2_weight, batch_sizes)
        if self.tp_size > 1:
            down = tensor_model_parallel_all_reduce(down)
        return down
        
        
    