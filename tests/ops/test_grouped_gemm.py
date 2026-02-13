"""Standalone unit test for disagmoe_c.grouped_gemm."""
import torch
import disagmoe_c

torch.set_default_dtype(torch.bfloat16)


def _ref_grouped_gemm(a, b, batch_sizes):
    """Reference: per-expert torch.matmul."""
    num_experts = b.size(0)
    N = b.size(2)
    outputs = []
    offset = 0
    for i in range(num_experts):
        M = int(batch_sizes[i].item())
        if M > 0:
            ai = a[offset : offset + M]     # [M, K]
            bi = b[i]                         # [K, N]
            outputs.append(ai @ bi)           # [M, N]
        offset += M
    return torch.cat(outputs, dim=0) if outputs else torch.empty(0, N, device=a.device, dtype=a.dtype)


def test_basic():
    """Basic correctness: compare disagmoe_c.grouped_gemm vs torch.matmul."""
    E, K, N = 8, 128, 256
    tokens_per_expert = 64
    total_tokens = E * tokens_per_expert

    a = torch.randn(total_tokens, K, device="cuda")
    b = torch.randn(E, K, N, device="cuda")
    c = torch.empty(total_tokens, N, device="cuda")
    batch_sizes = torch.full((E,), tokens_per_expert, dtype=torch.int64, device="cuda")

    disagmoe_c.grouped_gemm(a, b, c, batch_sizes)
    ref = _ref_grouped_gemm(a, b, batch_sizes)

    # bf16 tolerance
    assert torch.allclose(c, ref, atol=1e-1, rtol=1e-1), \
        f"Max diff: {(c - ref).abs().max().item()}"
    print(f"[PASS] test_basic  (max diff = {(c - ref).abs().max().item():.6f})")


def test_uneven_batch_sizes():
    """Experts with different batch sizes."""
    E, K, N = 4, 64, 128
    sizes = [10, 0, 30, 20]  # one expert has 0 tokens
    total_tokens = sum(sizes)

    a = torch.randn(total_tokens, K, device="cuda")
    b = torch.randn(E, K, N, device="cuda")
    c = torch.empty(total_tokens, N, device="cuda")
    batch_sizes = torch.tensor(sizes, dtype=torch.int64, device="cuda")

    disagmoe_c.grouped_gemm(a, b, c, batch_sizes)
    ref = _ref_grouped_gemm(a, b, batch_sizes)

    assert torch.allclose(c[:total_tokens], ref, atol=1e-1, rtol=1e-1), \
        f"Max diff: {(c[:total_tokens] - ref).abs().max().item()}"
    print(f"[PASS] test_uneven_batch_sizes  (max diff = {(c[:total_tokens] - ref).abs().max().item():.6f})")


def test_cpu_batch_sizes():
    """batch_sizes on CPU should also work."""
    E, K, N = 4, 64, 128
    tokens_per_expert = 16
    total_tokens = E * tokens_per_expert

    a = torch.randn(total_tokens, K, device="cuda")
    b = torch.randn(E, K, N, device="cuda")
    c = torch.empty(total_tokens, N, device="cuda")
    batch_sizes = torch.full((E,), tokens_per_expert, dtype=torch.int64)  # CPU

    disagmoe_c.grouped_gemm(a, b, c, batch_sizes)
    ref = _ref_grouped_gemm(a, b, batch_sizes)

    assert torch.allclose(c, ref, atol=1e-1, rtol=1e-1), \
        f"Max diff: {(c - ref).abs().max().item()}"
    print(f"[PASS] test_cpu_batch_sizes  (max diff = {(c - ref).abs().max().item():.6f})")


def test_moe_pattern():
    """Mimics the MoEExpertsCUTLASS up/down projection pattern."""
    E = 8
    hidden = 256
    inter = 512
    tokens_per_expert = 32
    total_tokens = E * tokens_per_expert

    # Up projection
    hiddens = torch.randn(total_tokens, hidden, device="cuda")
    w13 = torch.randn(E, hidden, inter * 2, device="cuda")
    cache_up = torch.empty(total_tokens, inter * 2, device="cuda")
    batch_sizes = torch.full((E,), tokens_per_expert, dtype=torch.int64, device="cuda")

    disagmoe_c.grouped_gemm(hiddens, w13, cache_up, batch_sizes)
    ref_up = _ref_grouped_gemm(hiddens, w13, batch_sizes)
    assert torch.allclose(cache_up, ref_up, atol=1e-1, rtol=1e-1)

    # SiLU + gate
    act = torch.nn.SiLU(inplace=True)
    up = act(cache_up[:, :inter]) * cache_up[:, inter:]

    # Down projection
    w2 = torch.randn(E, inter, hidden, device="cuda")
    cache_down = torch.empty(total_tokens, hidden, device="cuda")
    disagmoe_c.grouped_gemm(up, w2, cache_down, batch_sizes)
    ref_down = _ref_grouped_gemm(up, w2, batch_sizes)
    assert torch.allclose(cache_down, ref_down, atol=1e-1, rtol=1e-1)

    print(f"[PASS] test_moe_pattern  (up diff = {(cache_up - ref_up).abs().max().item():.6f}, "
          f"down diff = {(cache_down - ref_down).abs().max().item():.6f})")


if __name__ == "__main__":
    # Print device info
    dev = torch.cuda.current_device()
    cap = torch.cuda.get_device_capability(dev)
    print(f"Device: {torch.cuda.get_device_name(dev)}, SM {cap[0]}{cap[1]}")

    # One-time init — probes hardware and selects optimal tile config
    desc = disagmoe_c.init_grouped_gemm(dev)
    print(f"Config: {desc}")
    print()

    test_basic()
    test_uneven_batch_sizes()
    test_cpu_batch_sizes()
    test_moe_pattern()

    print("\nAll tests passed!")
