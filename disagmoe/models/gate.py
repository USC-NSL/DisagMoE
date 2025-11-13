from typing import Optional, Tuple, Dict, List

import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time


class ProfileDrivenRouter:
    """
    Skeleton for a profile-driven simulated routing module.
    """

    # all args are required
    def __init__(self, profile_bytes: bytes, num_experts_expected: int, top_k: int) -> None:
        self.num_experts = int(num_experts_expected)
        self.top_k = top_k
        # If using ProfileDriven routing, bytes must be provided and non-empty.
        if len(profile_bytes) == 0:
            raise ValueError("ProfileDrivenRouter requires non-empty profile bytes at init")

        
        self._load_profile_from_bytes(profile_bytes, top_k)

    def _load_profile_from_bytes(self, data: bytes, top_k: int) -> None:
        """Load routing profile from raw bytes (delivered via Ray object store).

        Tries pyarrow first (best for in-memory Parquet), falls back to pandas.
        Stores a pandas.DataFrame on self.profile_df for later use by forward().
        """
        # Try pyarrow parquet in-memory read
        try:
            reader = pa.BufferReader(data)
            table = pq.read_table(reader)
        except Exception as e:
            raise RuntimeError(f"Failed to parse Parquet profile bytes: {e}")

        # Record the time used to load and process the profile
        start_time = time.perf_counter()
        
        # Check required columns
        column_names = {name: table[name] for name in table.column_names}
        for required_column in ("rid", "token_index", "layer"):
            if required_column not in column_names:
                raise ValueError(f"Required column {required_column} not found in profile")
        
        # Extract expert columns: expert_logical_k0...kN
        expert_columns = [c for c in table.column_names if c.startswith("expert_logical_k")]
        if not expert_columns:
            raise ValueError("No expert columns found in profile")
        
        # Verify the number of expert columns matches the K
        if len(expert_columns) != top_k:
            raise ValueError(f"Number of expert columns {len(expert_columns)} in the profile does not match system's K = {top_k}")

        # Verify the number of total experts is compatible with the number of experts in the system
        expert_df = table.select(expert_columns).to_pandas(types_mapper=pd.ArrowDtype)
        expert_vals_np = expert_df.to_numpy(dtype=np.int32, copy=False)
        unique_experts = np.unique(expert_vals_np)
        unique_experts = unique_experts[unique_experts >= 0]  # ignore any negative placeholders
        num_unique_experts = int(unique_experts.size)
        project_group_size: Optional[int] = None
        if num_unique_experts == self.num_experts:
            pass
        elif num_unique_experts > self.num_experts and (num_unique_experts % self.num_experts == 0):
            project_group_size = num_unique_experts // self.num_experts
            # Print a yellow warning to notify projection is applied
            print(
                f"\033[33m[ProfileDrivenRouter] Profile has {num_unique_experts} experts; "
                f"system expects {self.num_experts}. Projecting by grouping {project_group_size} "
                f"profiled experts per system expert.\033[0m"
            )
        else:
            raise ValueError(f"Profile contains {num_unique_experts} unique experts, but system expects {self.num_experts}.")
        
        # Extract the data to pandas and numpy and build the mapping, so later routing is efficient
        use_cols = ["rid", "token_index", "layer"] + expert_columns
        df = table.select(use_cols).to_pandas(types_mapper=pd.ArrowDtype)

        keys = list(
            zip(
                df["rid"].to_numpy(dtype=np.int32, copy=False),
                df["token_index"].to_numpy(dtype=np.int32, copy=False),
                df["layer"].to_numpy(dtype=np.int32, copy=False),
            )
        )

        self.routing_outcomes = df[expert_columns].to_numpy(dtype=np.int32, copy=False)
        # If projection is required, map profiled expert ids to expected expert ids
        
        if project_group_size is not None:
            # Only project non-negative expert ids; keep negative placeholders intact
            mask = self.routing_outcomes >= 0
            # Integer division groups ids: [0..g-1]->0, [g..2g-1]->1, ...
            self.routing_outcomes[mask] = self.routing_outcomes[mask] // int(project_group_size)

        # Map key -> routing_outcomes row's index 
        self.token_meta_to_routing_outcomes_index: Dict[Tuple[int, int, int], int] = {k: i for i, k in enumerate(keys)}

        # Count how many requests are there in the profile
        rid_array = df["rid"].to_numpy(dtype=np.int32, copy=False)
        self.num_profiled_requests = int(np.unique(rid_array).size)

        # Find the maximum layer id in the profile
        layer_array = df["layer"].to_numpy(dtype=np.int32, copy=False)
        self.num_layers = int(layer_array.max())

        # Count how many tokens are there for every profiled request
        # Make a dict request id -> number of profiled tokens
        token_counts = df.groupby("rid")["token_index"].nunique() # Count unique token_index per rid (tokens may repeat across layers)
        self.profiled_tokens_per_request: Dict[int, int] = {
            int(rid): int(count) for rid, count in token_counts.items()
        }

        end_time = time.perf_counter()
        print(f"Time used to load and process the profile: {end_time - start_time:.3f} seconds")

    def route(self, request_ids: List[int], token_indices: torch.Tensor, layer_id: int, top_k: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        # Validate inputs
        if layer_id > self.num_layers:
            raise ValueError(f"Requested layer_id {layer_id} exceeds profiled max layer {self.num_layers}. Profile/model mismatch.")
        token_indicies_np = token_indices.detach().to(dtype=torch.int64, device="cpu").numpy()
        assert len(request_ids) == len(token_indicies_np), "request_ids and token_indices must have the same length"
        
        # Ensure top_k matches the profile's K dimension
        if top_k != self.top_k:
            raise ValueError(f"Requested top_k {top_k} does not match ProfileDrivenRouter's top_k {self.top_k}")
        
        # Build list of expert ids for each (rid, token_index) after wraparound
        gathered_ids: List[np.ndarray] = []
        for rid_in, tok_idx_in in zip(request_ids, token_indicies_np):
            mapped_rid = int(rid_in) % int(self.num_profiled_requests)
            num_tokens_for_rid = self.profiled_tokens_per_request[mapped_rid]
            if num_tokens_for_rid <= 0:
                raise ValueError(f"Profiled request {mapped_rid} has no tokens")
            mapped_token_index = int(tok_idx_in) % int(num_tokens_for_rid)
            key = (int(mapped_rid), int(mapped_token_index), int(layer_id))
            row_idx = self.token_meta_to_routing_outcomes_index.get(key, None)
            if row_idx is None:
                raise KeyError(f"No routing outcome found for key={key}")
            gathered_ids.append(self.routing_outcomes[row_idx])
        
        # Convert to torch tensors
        if len(gathered_ids) == 0:
            # No tokens -> return empty tensors with correct shapes
            topk_ids = torch.empty((0, top_k), device=device, dtype=torch.int64)
            topk_weights = torch.empty((0, top_k), device=device, dtype=dtype)
            return topk_weights, topk_ids
        
        ids_np = np.stack(gathered_ids, axis=0)  # [num_tokens, top_k], int32
        topk_ids = torch.as_tensor(ids_np, device=device, dtype=torch.int64)
        
        # Use uniform weights for profiled routing outcomes
        if top_k == 1:
            topk_weights = torch.ones((topk_ids.shape[0], 1), device=device, dtype=dtype)
        else:
            topk_weights = torch.full((topk_ids.shape[0], top_k), 1.0 / float(top_k), device=device, dtype=dtype)
        
        return topk_weights, topk_ids

__all__ = ["ProfileDrivenRouter"]


