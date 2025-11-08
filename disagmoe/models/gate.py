from typing import Optional, Tuple

import torch
from torch import nn


class ProfileDrivenGate(nn.Module):
    """
    Skeleton for a profile-driven simulated gating module.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,  # ignored
        params_dtype: Optional[torch.dtype] = None,  # ignored
        quant_config: Optional[object] = None,  # ignored
        prefix: str = "",  # ignored
        profile_bytes: Optional[bytes] = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        # If caller chose ProfileDrivenGate, bytes must be provided and non-empty.
        if profile_bytes is None or len(profile_bytes) == 0:
            raise ValueError("ProfileDrivenGate requires non-empty profile bytes at init")
        self._load_profile_from_bytes(profile_bytes)

    def forward(self, hidden_states: torch.Tensor, request_ids, layer_id: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        This is not a real FFN. Gating is decided here by the profile.
        """
        raise NotImplementedError()

    def _load_profile_from_bytes(self, data: bytes) -> None:
        """Load routing profile from raw bytes (delivered via Ray object store)."""
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("Profile data must be bytes or bytearray")
        # Minimal skeleton: record that bytes were received; actual parsing TBD.
        self._profile_loaded = True
        self._profile_size = len(data)

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, output_size={self.output_size}"


__all__ = ["ProfileDrivenGate"]


