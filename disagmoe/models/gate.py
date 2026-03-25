from typing import Optional, Tuple, Dict, List

import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time


class ProfileDrivenRouter:

    def __init__(self, profile_bytes: bytes, num_experts_expected: int, top_k: int, layer_id: int = None) -> None:
        self.num_experts = int(num_experts_expected)
        self.top_k = top_k
        self.layer_id = layer_id
        if len(profile_bytes) == 0:
            raise ValueError("ProfileDrivenRouter requires non-empty profile bytes at init")
        self._load_profile_from_bytes(profile_bytes, top_k)

    def _load_profile_from_bytes(self, data: bytes, top_k: int) -> None:
        try:
            reader = pa.BufferReader(data)
            table = pq.read_table(reader)
        except Exception as e:
            raise RuntimeError(f"Failed to parse Parquet profile bytes: {e}")

        start_time = time.perf_counter()

        column_names = {name: table[name] for name in table.column_names}
        for required_column in ("rid", "token_index", "layer"):
            if required_column not in column_names:
                raise ValueError(f"Required column {required_column} not found in profile")

        expert_columns = [c for c in table.column_names if c.startswith("expert_logical_k")]
        if not expert_columns:
            raise ValueError("No expert columns found in profile")

        if len(expert_columns) != top_k:
            raise ValueError(
                f"Number of expert columns {len(expert_columns)} in the profile "
                f"does not match system's K = {top_k}"
            )

        expert_df = table.select(expert_columns).to_pandas(types_mapper=pd.ArrowDtype)
        expert_vals_np = expert_df.to_numpy(dtype=np.int32, copy=False)
        unique_experts = np.unique(expert_vals_np)
        unique_experts = unique_experts[unique_experts >= 0]
        num_unique_experts = int(unique_experts.size)
        project_group_size: Optional[int] = None
        if num_unique_experts == self.num_experts:
            pass
        elif num_unique_experts > self.num_experts and (num_unique_experts % self.num_experts == 0):
            project_group_size = num_unique_experts // self.num_experts
            print(
                f"\033[33m[ProfileDrivenRouter] Profile has {num_unique_experts} experts; "
                f"system expects {self.num_experts}. Projecting by grouping {project_group_size} "
                f"profiled experts per system expert.\033[0m"
            )
        else:
            raise ValueError(
                f"Profile contains {num_unique_experts} unique experts, "
                f"but system expects {self.num_experts}."
            )

        use_cols = ["rid", "token_index", "layer"] + expert_columns
        df = table.select(use_cols).to_pandas(types_mapper=pd.ArrowDtype)

        if self.layer_id is not None:
            df = df[df["layer"] == self.layer_id].reset_index(drop=True)

        routing_outcomes = df[expert_columns].to_numpy(dtype=np.int32, copy=True)
        if project_group_size is not None:
            mask = routing_outcomes >= 0
            routing_outcomes[mask] = routing_outcomes[mask] // int(project_group_size)

        rid_array = df["rid"].to_numpy(dtype=np.int32, copy=False)
        self.num_profiled_requests = int(np.unique(rid_array).size)

        layer_array = df["layer"].to_numpy(dtype=np.int32, copy=False)
        self.num_layers = 1 if self.layer_id is not None else int(layer_array.max()) + 1

        token_counts = df.groupby("rid")["token_index"].nunique()
        tokens_per_req_np = np.zeros(self.num_profiled_requests, dtype=np.int64)
        for rid, count in token_counts.items():
            tokens_per_req_np[int(rid)] = int(count)
        self.tokens_per_request = torch.as_tensor(
            tokens_per_req_np, dtype=torch.int64, device="cuda"
        )

        self.token_prefix_sum = torch.zeros(
            self.num_profiled_requests + 1, dtype=torch.int64, device="cuda"
        )
        self.token_prefix_sum[1:] = torch.cumsum(self.tokens_per_request, dim=0)
        self.total_tokens_per_layer = int(self.token_prefix_sum[-1].item())

        df["_row_idx"] = np.arange(len(df))
        df_sorted = df.sort_values(["layer", "rid", "token_index"]).reset_index(drop=True)
        sorted_indices = df_sorted["_row_idx"].to_numpy()
        routing_sorted = routing_outcomes[sorted_indices]

        self.routing_data = torch.as_tensor(routing_sorted, dtype=torch.int32, device="cuda")

        self.uniform_weight = 1.0 / float(top_k) if top_k > 1 else 1.0

        end_time = time.perf_counter()
        print(f"Time used to load and process the profile: {end_time - start_time:.3f} seconds")
        gpu_mb = self.routing_data.nelement() * self.routing_data.element_size() / (1024 ** 2)
        print(
            f"[ProfileDrivenRouter] GPU routing table: {self.routing_data.shape}, "
            f"{gpu_mb:.1f} MB"
        )

    def route(
        self,
        request_ids: torch.Tensor,
        token_indices: torch.Tensor,
        layer_id: int,
        top_k: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if top_k != self.top_k:
            raise ValueError(
                f"Requested top_k {top_k} does not match ProfileDrivenRouter's top_k {self.top_k}"
            )

        mapped_rid = request_ids % self.num_profiled_requests
        mapped_tok = token_indices % self.tokens_per_request[mapped_rid]

        if self.layer_id is not None:
            flat_idx = self.token_prefix_sum[mapped_rid] + mapped_tok
        else:
            flat_idx = (
                layer_id * self.total_tokens_per_layer
                + self.token_prefix_sum[mapped_rid]
                + mapped_tok
            )

        topk_ids = self.routing_data[flat_idx].to(dtype=torch.int32)
        topk_weights = torch.full(
            (topk_ids.shape[0], top_k), self.uniform_weight, device=device, dtype=torch.float32
        )

        return topk_weights, topk_ids


__all__ = ["ProfileDrivenRouter"]
