import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt


try:
    import deep_gemm as dg
except ImportError:
    dg = None


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for DeepGemm BF16 benchmark.")

# DeepGemm BF16 kernels are intended for Hopper (SM90) or later.
major, _ = torch.cuda.get_device_capability()
if major < 9:
    raise RuntimeError(
        f"DeepGemm BF16 benchmark requires Hopper (SM90) or later, got SM{major}x."
    )

if dg is None:
    raise ImportError("deep_gemm is not available. Please build/install it before running this benchmark.")


DTYPE = torch.bfloat16

torch.cuda.set_device(device)
torch.set_default_device(device)
torch.set_default_dtype(DTYPE)
print(f"DeepGemm BF16 dtype fixed to: {DTYPE}")


@torch.inference_mode()
def alloc_deepgemm_bf16_weights(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    device,
):
    """Allocate BF16 weights directly in DeepGemm's preferred layout.

    w13: [E, 2I, H]  (N = 2I, K = H)
    w2:  [E, H, I]   (N = H,  K = I)
    """
    # w13
    k_w13 = hidden_size
    n_w13 = intermediate_size * 2
    w13 = torch.randn(
        num_experts,
        n_w13,
        k_w13,
        device=device,
        dtype=DTYPE,
    )

    # w2
    k_w2 = intermediate_size
    n_w2 = hidden_size
    w2 = torch.randn(
        num_experts,
        n_w2,
        k_w2,
        device=device,
        dtype=DTYPE,
    )

    return w13, w2


@torch.inference_mode()
def benchmark_deepgemm_bf16_grouped_moe(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
):
    """Benchmark DeepGemm BF16 grouped experts with CUDA graphs.

    Layout and math follow `MoEExpertsDeepGemmBF16` in `disagmoe/models/experts.py`.
    Batch sizes are linear: [1, 32, 64, ..., 1024].

    We mimic a 2-layer MoE block:
      - Layer 1: BF16 DeepGemm grouped experts (not timed).
      - Layer 2: BF16 DeepGemm grouped experts (timed via CUDA graph).

    For each measured iteration, we first run Layer 1, then time
    the CUDA-graph replay for Layer 2. This avoids always having a
    \"cold\" layer's data resident in SM caches at the start of timing.
    """

    # Linear per-expert batch sizes
    row_sizes = np.array([1] + list(range(32, 1025, 32)), dtype=np.int32)
    num_repeats = 5

    # Two layers: separate weights but same dimensions.
    w13_l1, w2_l1 = alloc_deepgemm_bf16_weights(
        hidden_size, intermediate_size, num_experts, device
    )
    w13_l2, w2_l2 = alloc_deepgemm_bf16_weights(
        hidden_size, intermediate_size, num_experts, device
    )

    act_fn = torch.nn.SiLU(inplace=True)

    results_deep_bf16 = []

    for n_rows in row_sizes:
        results_this_batch_size = []

        # Per-expert variable batch sizes
        ratios_local = (
            expert_batch_size_ratios
            if len(expert_batch_size_ratios) == num_experts
            else [1.0] * num_experts
        )
        batch_sizes_i = [
            max(1, int(round(float(n_rows) * float(r)))) for r in ratios_local
        ]

        # Input activations per expert, then flattened to [M, H]
        As_list = [
            torch.randn(bs, hidden_size, device=device, dtype=DTYPE).contiguous()
            for bs in batch_sizes_i
        ]
        A_flat = torch.cat(As_list, dim=0).contiguous()
        M = A_flat.shape[0]

        # m_indices maps each row to its expert group: [M]
        m_indices = torch.empty(M, device=device, dtype=torch.int32)
        start_idx = 0
        for e, bs in enumerate(batch_sizes_i):
            end_idx = start_idx + bs
            m_indices[start_idx:end_idx] = e
            start_idx = end_idx

        # Buffers for layer 1 (L1)
        up_buf_l1 = torch.empty(
            M,
            intermediate_size * 2,
            device=device,
            dtype=DTYPE,
        )
        up_activated_l1 = torch.empty(
            M,
            intermediate_size,
            device=device,
            dtype=DTYPE,
        )
        hiddens_l2 = torch.empty(
            M,
            hidden_size,
            device=device,
            dtype=DTYPE,
        )

        # Buffers for layer 2 (L2, timed)
        up_buf_l2 = torch.empty(
            M,
            intermediate_size * 2,
            device=device,
            dtype=DTYPE,
        )
        up_activated_l2 = torch.empty(
            M,
            intermediate_size,
            device=device,
            dtype=DTYPE,
        )
        down_buf_l2 = torch.empty(
            M,
            hidden_size,
            device=device,
            dtype=DTYPE,
        )

        def run_layer1():
            # Layer 1 GEMM 1: A_flat @ w13_l1 -> up_buf_l1
            dg.m_grouped_bf16_gemm_nt_contiguous(
                A_flat,
                w13_l1,
                up_buf_l1,
                m_indices,
            )

            # Activation + gating: SiLU(up[:, :I]) * up[:, I:]
            gate = up_buf_l1[:, :intermediate_size]
            val = up_buf_l1[:, intermediate_size:]
            act_fn(gate)
            torch.mul(gate, val, out=up_activated_l1)

            # Layer 1 GEMM 2: up_activated_l1 @ w2_l1 -> hiddens_l2
            dg.m_grouped_bf16_gemm_nt_contiguous(
                up_activated_l1,
                w2_l1,
                hiddens_l2,
                m_indices,
            )

        def run_layer2():
            # Layer 2 GEMM 1: hiddens_l2 @ w13_l2 -> up_buf_l2
            dg.m_grouped_bf16_gemm_nt_contiguous(
                hiddens_l2,
                w13_l2,
                up_buf_l2,
                m_indices,
            )

            # Activation + gating
            gate2 = up_buf_l2[:, :intermediate_size]
            val2 = up_buf_l2[:, intermediate_size:]
            act_fn(gate2)
            torch.mul(gate2, val2, out=up_activated_l2)

            # Layer 2 GEMM 2: up_activated_l2 @ w2_l2 -> down_buf_l2
            dg.m_grouped_bf16_gemm_nt_contiguous(
                up_activated_l2,
                w2_l2,
                down_buf_l2,
                m_indices,
            )

        # Warmup
        for _ in range(2):
            run_layer1()
            run_layer2()

        # Capture CUDA graph for this batch size
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph, stream=stream):
            run_layer2()
        torch.cuda.synchronize(device)

        # Warmup graph
        for _ in range(2):
            run_layer1()
            graph.replay()

        # Timed replays
        for _ in range(num_repeats):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            # Run layer 1 (un-timed) to \"prime\" caches and inputs
            run_layer1()
            torch.cuda.synchronize(device)
            start_evt.record()
            graph.replay()
            end_evt.record()
            torch.cuda.synchronize(device)
            results_this_batch_size.append(start_evt.elapsed_time(end_evt))

        results_deep_bf16.append(float(np.mean(results_this_batch_size)))

    torch.cuda.empty_cache()
    return row_sizes, results_deep_bf16


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepGemm BF16 grouped MoE benchmark with CUDA graphs",
    )
    parser.add_argument(
        "--hidden-sizes-k",
        type=int,
        nargs="+",
        default=[2, 4],
        help="Hidden sizes in k (e.g., 2 -> 2048). Default: 2 4",
    )
    parser.add_argument(
        "--intermediate-sizes",
        type=int,
        nargs="+",
        default=[768, 1536],
        help="Intermediate sizes. Default: 768 1536",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        nargs="+",
        default=[8, 8],
        help="Number of experts per model. Default: 8 8",
    )
    args = parser.parse_args()

    hidden_sizes_k = np.array(args.hidden_sizes_k, dtype=np.int32)
    hidden_sizes = hidden_sizes_k * 1024
    intermediate_sizes = args.intermediate_sizes
    num_experts_list = args.num_experts

    if not (
        len(hidden_sizes) == len(intermediate_sizes) == len(num_experts_list)
    ):
        raise ValueError(
            "hidden_sizes_k, intermediate_sizes, and num_experts must have the same length."
        )

    models = [
        f"Model-{i}: H={h}k, I={i_size}"
        for i, (h, i_size) in enumerate(
            zip(hidden_sizes_k, intermediate_sizes)
        )
    ]

    global expert_batch_size_ratios
    # Default: uniform per-expert batch sizes.
    max_num_experts = max(num_experts_list)
    expert_batch_size_ratios = [1.0] * max_num_experts

    fig, ax = plt.subplots(figsize=(9, 6))

    for hidden_size, intermediate_size, num_experts, model in zip(
        hidden_sizes, intermediate_sizes, num_experts_list, models
    ):
        row_sizes, times_ms = benchmark_deepgemm_bf16_grouped_moe(
            hidden_size, intermediate_size, num_experts
        )

        # Print numeric results
        print(
            f"\n{model}, H={hidden_size}, I={intermediate_size}, E={num_experts}"
        )
        for bs, t in zip(row_sizes, times_ms):
            print(f"  per-expert batch {int(bs):4d}: {t:.4f} ms")

        ax.plot(
            row_sizes,
            times_ms,
            marker="o",
            linestyle="-",
            label=f"{model} (E={num_experts})",
        )

    ax.set_xlabel("Per-expert batch size")
    ax.set_ylabel("Avg execution time (ms)")
    ax.set_title("DeepGemm BF16 Grouped MoE (CUDA Graphs)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = "deepgemm_bf16_grouped_moe.png"
    fig.savefig(out_path, dpi=300)
    print(f"\nSaved DeepGemm BF16 grouped MoE plot to {out_path}")


if __name__ == "__main__":
    main()
