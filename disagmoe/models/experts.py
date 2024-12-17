import torch
from grouped_gemm.backend import gmm
from vllm.distributed import (get_tensor_model_parallel_rank,
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
        self.create_weights()
        
    def create_weights(self, params_dtype: torch.dtype = None):
        if params_dtype == None:
            # FIXME(hogura|20241014): maybe use torch.get_default_dtype
            params_dtype = torch.bfloat16
            
        # Fused gate_up_proj (column parallel)
        self.w1_weight = torch.nn.Parameter(torch.randn(self.num_experts,
                                                    self.hidden_size,
                                                    self.intermediate_size,
                                                    dtype=params_dtype).cuda(),
                                        requires_grad=False)
        self.register_parameter("w1_weight", self.w1_weight)

        # down_proj (row parallel)
        self.w2_weight = torch.nn.Parameter(torch.randn(self.num_experts,
                                                    self.intermediate_size,
                                                    self.hidden_size,
                                                    dtype=params_dtype).cuda(),
                                        requires_grad=False)
        self.register_parameter("w2_weight", self.w2_weight)
        
        self.w3_weight = torch.nn.Parameter(torch.randn(self.num_experts,
                                                    self.hidden_size,
                                                    self.intermediate_size,
                                                    dtype=params_dtype).cuda(),
                                        requires_grad=False)
        self.register_parameter("w3_weight", self.w3_weight)
        
        self.act_fn = torch.nn.SiLU()
        
    def forward(self, hiddens: torch.Tensor, batch_sizes: torch.Tensor):
        # Here use grouped gemm, tokens must be permuted by corresponding expert_id
        up = gmm(hiddens, self.w1_weight, batch_sizes)
        up = self.act_fn(up)
        up_r = gmm(hiddens, self.w3_weight, batch_sizes)
        up = up * up_r
        
        down = gmm(up, self.w2_weight, batch_sizes)
        if self.tp_size > 1:
            down = tensor_model_parallel_all_reduce(down)
        return down