#!/usr/bin/env python3
"""
Microbenchmark: GPT-OSS-120B vs GLM-4.5-Air-106B
CUTLASS Grouped GEMM, 8 experts per rank.

Per-expert batch sizes: 8, 16, 32, 48, 64.
Each data point = uniform distribution across all 8 experts.

Usage:
    python bench.py
"""

import os
import sys

import torch

# Must set default dtype before MoEExpertsCUTLASS construction (it asserts bf16).
torch.set_default_dtype(torch.bfloat16)

import pandas as pd
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, *([".."] * 5)))
sys.path.insert(0, _ROOT)

from disagmoe.utils.logger import initialize_logger

initialize_logger("bench")

from disagmoe.models.experts import MoEExpertsCUTLASS
from disagmoe.config import gptoss_120b_config, glm45air_106b_config

NUM_EXPERTS = 8
PER_EXPERT_BS = [8, 16, 32, 48, 64, 96, 128]
WARMUP_ITERS = 20
BENCH_ITERS = 100

CONFIGS = {
    "GPT-OSS-120B": {
        "hidden_size": gptoss_120b_config.hidden_size,  # 2880
        "intermediate_size": gptoss_120b_config.intermediate_size,  # 2880
    },
    "GLM-4.5-Air-106B": {
        "hidden_size": glm45air_106b_config.hidden_size,  # 4096
        "intermediate_size": glm45air_106b_config.intermediate_size,  # 1408
    },
}


def bench_model(name: str, hidden_size: int, intermediate_size: int) -> pd.DataFrame:
    max_total_bs = max(PER_EXPERT_BS) * NUM_EXPERTS

    print(f"\n{'=' * 60}")
    print(
        f"  {name}: hidden={hidden_size}, inter={intermediate_size}, "
        f"experts={NUM_EXPERTS}"
    )
    print(
        f"  w13 GEMM shape per expert: "
        f"[bs, {hidden_size}] x [{hidden_size}, {intermediate_size * 2}]"
    )
    print(
        f"  w2  GEMM shape per expert: "
        f"[bs, {intermediate_size}] x [{intermediate_size}, {hidden_size}]"
    )
    print(f"{'=' * 60}")

    experts = MoEExpertsCUTLASS(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=NUM_EXPERTS,
        tp_size=1,
        max_batch_size=max_total_bs,
    )
    experts.eval()

    rows = []
    for pbs in PER_EXPERT_BS:
        total_bs = pbs * NUM_EXPERTS
        hiddens = torch.randn(
            total_bs, hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        batch_sizes = torch.full((NUM_EXPERTS,), pbs, dtype=torch.int64, device="cuda")

        with torch.no_grad():
            for _ in range(WARMUP_ITERS):
                experts(total_bs, hiddens, batch_sizes)
        torch.cuda.synchronize()

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            t0.record()
            for _ in range(BENCH_ITERS):
                experts(total_bs, hiddens, batch_sizes)
            t1.record()
        torch.cuda.synchronize()

        avg_ms = t0.elapsed_time(t1) / BENCH_ITERS
        per_expert_ms = avg_ms / NUM_EXPERTS

        rows.append(
            dict(
                per_expert_bs=pbs,
                total_bs=total_bs,
                total_ms=avg_ms,
                per_expert_ms=per_expert_ms,
            )
        )
        print(
            f"  per_expert_bs={pbs:>3d}  total_bs={total_bs:>4d}  "
            f"total={avg_ms:.4f} ms  per_expert={per_expert_ms:.4f} ms"
        )

    del experts
    torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def plot_results(results: dict[str, pd.DataFrame], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "GPT-OSS-120B vs GLM-4.5-Air-106B  |  CUTLASS Grouped GEMM  |  8 experts/rank",
        fontsize=13,
    )

    colors = {"GPT-OSS-120B": "#2196F3", "GLM-4.5-Air-106B": "#FF5722"}

    ax = axes[0]
    for name, df in results.items():
        ax.plot(
            df["per_expert_bs"],
            df["total_ms"],
            "s-",
            color=colors[name],
            label=name,
        )
    ax.set_xlabel("Per-Expert Batch Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Total Grouped GEMM Latency (8 experts)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(PER_EXPERT_BS)

    ax = axes[1]
    for name, df in results.items():
        throughput = df["total_bs"] / df["total_ms"]
        ax.plot(
            df["per_expert_bs"],
            throughput,
            "^-",
            color=colors[name],
            label=name,
        )
    ax.set_xlabel("Per-Expert Batch Size")
    ax.set_ylabel("Throughput (tokens / ms)")
    ax.set_title("Throughput")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(PER_EXPERT_BS)

    plt.tight_layout()
    path = os.path.join(out_dir, "gptoss_vs_glm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {path}")


def main():
    out_dir = os.path.join(_SCRIPT_DIR, "plots")
    os.makedirs(out_dir, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    for name, cfg in CONFIGS.items():
        df = bench_model(name, **cfg)
        csv_path = os.path.join(out_dir, f"{name.replace(' ', '_')}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  CSV: {csv_path}")
        results[name] = df

    plot_results(results, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
