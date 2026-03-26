#!/usr/bin/env python3
"""
Generalized advanced-logging processor and plotter for asyncmoe experiments.

Usage:
  python experiments/process_and_plot_advanced_logging.py <advanced_logs_dir> [output_dir]

  <advanced_logs_dir>  Directory containing device_* subdirs with moe_steps.json,
                       queuing_delays.json, queue_snapshots.json
  [output_dir]         Where to write plots (default: <advanced_logs_dir>/plots)

Examples:
  python experiments/process_and_plot_advanced_logging.py experiments/amoe-064/advanced_logs
  python experiments/process_and_plot_advanced_logging.py experiments/amoe-064/advanced_logs experiments/amoe-064/plots
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


N_EXPERTS_PER_RANK = 8  # gptoss_120b: 128 experts / 16 GPUs


def load_device_data(adv_log_dir: Path):
    data = {}
    for dev_dir in sorted(adv_log_dir.glob("device_*")):
        dev_id = int(dev_dir.name.split("_")[1])
        moe_path = dev_dir / "moe_steps.json"
        q_path = dev_dir / "queuing_delays.json"
        qs_path = dev_dir / "queue_snapshots.json"
        if not moe_path.exists():
            continue
        with open(moe_path) as f:
            moe = json.load(f)
        queuing = {}
        queue_snapshots = {}
        if q_path.exists():
            with open(q_path) as f:
                queuing = json.load(f)
        if qs_path.exists():
            with open(qs_path) as f:
                queue_snapshots = json.load(f)
        data[dev_id] = {
            "moe_steps": moe,
            "queuing_delays": queuing,
            "queue_snapshots": queue_snapshots,
        }
    return data


def cdf(values):
    arr = np.sort(values)
    p = np.arange(1, len(arr) + 1) / len(arr)
    return arr, p


def write_summary(data: dict, out_path: Path):
    lines = []
    lines.append("=" * 70)
    lines.append("Advanced Logging Summary")
    lines.append("=" * 70)

    all_bsz = []
    all_times = []
    for dev_id, d in sorted(data.items()):
        bsz = d["moe_steps"].get("batch_sizes", [])
        times = d["moe_steps"].get("execution_times_ms", [])
        all_bsz.extend(bsz)
        all_times.extend(times)
        if bsz:
            lines.append(
                f"  rank {dev_id:2d}: {len(bsz):6d} MoE steps, "
                f"bsz mean={np.mean(bsz):.1f} p50={np.median(bsz):.0f} p99={np.percentile(bsz, 99):.0f} max={max(bsz)}, "
                f"time mean={np.mean(times):.3f}ms p99={np.percentile(times, 99):.3f}ms"
            )

    if all_bsz:
        lines.append("")
        lines.append(f"  ALL RANKS: {len(all_bsz)} total MoE steps")
        lines.append(f"    batch size:  mean={np.mean(all_bsz):.1f}  p50={np.median(all_bsz):.0f}  p99={np.percentile(all_bsz, 99):.0f}  max={max(all_bsz)}")
        lines.append(f"    exec time:   mean={np.mean(all_times):.3f}ms  p50={np.median(all_times):.3f}ms  p99={np.percentile(all_times, 99):.3f}ms")

    txt = "\n".join(lines)
    out_path.write_text(txt)
    print(txt)
    print(f"\n  saved: {out_path}")


# ── CDF Plots ─────────────────────────────────────────────────────────

def plot_gemm_time_cdf(data: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab20.colors
    plotted = 0
    for dev_id, d in sorted(data.items()):
        times = d["moe_steps"].get("execution_times_ms", [])
        if not times:
            continue
        x, y = cdf(times)
        ax.plot(x, y, label=f"rank {dev_id}", color=colors[dev_id % len(colors)], lw=1.2)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("groupedGEMM execution time (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Per-MoE-step groupedGEMM time CDF (per rank)")
    ax.legend(fontsize=7, ncol=4, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_gemm_bsz_cdf(data: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab20.colors
    plotted = 0
    for dev_id, d in sorted(data.items()):
        bsz = d["moe_steps"].get("batch_sizes", [])
        if not bsz:
            continue
        x, y = cdf(bsz)
        ax.plot(x, y, label=f"rank {dev_id}", color=colors[dev_id % len(colors)], lw=1.2)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("groupedGEMM batch size (tokens)")
    ax.set_ylabel("CDF")
    ax.set_title("Per-MoE-step groupedGEMM batch size CDF (per rank)")
    ax.legend(fontsize=7, ncol=4, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── Batch-size vs Time Plots ──────────────────────────────────────────

def plot_bsz_vs_time(data: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab20.colors
    plotted = 0
    for dev_id, d in sorted(data.items()):
        bsz_list = d["moe_steps"].get("batch_sizes", [])
        time_list = d["moe_steps"].get("execution_times_ms", [])
        if not bsz_list or not time_list:
            continue
        groups = defaultdict(list)
        for bsz, t in zip(bsz_list, time_list):
            groups[bsz].append(t)
        xs = sorted(groups.keys())
        ys = [np.mean(groups[x]) for x in xs]
        ax.plot(xs, ys, label=f"rank {dev_id}", color=colors[dev_id % len(colors)],
                lw=1.2, marker=".", markersize=3)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("groupedGEMM batch size (tokens)")
    ax.set_ylabel("mean execution time (ms)")
    ax.set_title("Per-batch-size averaged groupedGEMM compute time (per rank)")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_bsz_vs_time_avg(data: dict, out_path: Path):
    all_groups = defaultdict(list)
    for d in data.values():
        bsz_list = d["moe_steps"].get("batch_sizes", [])
        time_list = d["moe_steps"].get("execution_times_ms", [])
        for bsz, t in zip(bsz_list, time_list):
            all_groups[bsz].append(t)
    if not all_groups:
        return
    xs = sorted(all_groups.keys())
    ys = np.array([np.mean(all_groups[x]) for x in xs])
    ys_std = np.array([np.std(all_groups[x]) for x in xs])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, lw=1.8, color="#3b82f6", label="mean across all ranks")
    ax.fill_between(xs, ys - ys_std, ys + ys_std, alpha=0.2, color="#3b82f6", label="±1 std")
    ax.set_xlabel("groupedGEMM batch size (tokens)")
    ax.set_ylabel("mean execution time (ms)")
    ax.set_title("Per-batch-size averaged groupedGEMM compute time (all ranks)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── Queuing Delay Heatmaps ────────────────────────────────────────────

def build_queue_matrix(data: dict):
    sums = defaultdict(float)
    counts = defaultdict(int)

    for d in data.values():
        for key, entry in d["queuing_delays"].items():
            lid = entry["layer_id"]
            eid = entry["expert_id"]
            mean = entry["mean_ms"]
            n = entry["count"]
            if n == 0:
                continue
            sums[(lid, eid)] += mean * n
            counts[(lid, eid)] += n

    if not counts:
        return None, [], []

    layer_ids = sorted({k[0] for k in counts})
    expert_ids = sorted({k[1] for k in counts})

    mat = np.full((len(expert_ids), len(layer_ids)), np.nan)
    for li, lid in enumerate(layer_ids):
        for ei, eid in enumerate(expert_ids):
            if counts[(lid, eid)] > 0:
                mat[ei, li] = sums[(lid, eid)] / counts[(lid, eid)]

    return mat, layer_ids, expert_ids


def plot_heatmap_expert(data: dict, out_path: Path):
    mat, layer_ids, expert_ids = build_queue_matrix(data)
    if mat is None:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(layer_ids) * 0.35), max(6, len(expert_ids) * 0.18)))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="YlOrRd", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="avg queuing delay (ms)")

    ax.set_xticks(range(len(layer_ids)))
    ax.set_xticklabels(layer_ids, fontsize=6, rotation=90)
    ax.set_yticks(range(len(expert_ids)))
    ax.set_yticklabels(expert_ids, fontsize=6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Expert ID")
    ax.set_title("Queuing delay heatmap (per expert)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_heatmap_rank(data: dict, out_path: Path, n_experts_per_rank: int = N_EXPERTS_PER_RANK):
    mat_expert, layer_ids, expert_ids = build_queue_matrix(data)
    if mat_expert is None:
        return

    max_rank = (max(expert_ids) // n_experts_per_rank) + 1
    rank_mat = np.full((max_rank, len(layer_ids)), np.nan)
    for rank in range(max_rank):
        eid_lo = rank * n_experts_per_rank
        eid_hi = eid_lo + n_experts_per_rank - 1
        rows = [i for i, eid in enumerate(expert_ids) if eid_lo <= eid <= eid_hi]
        if rows:
            slice_ = mat_expert[rows, :]
            with np.errstate(all="ignore"):
                rank_mat[rank, :] = np.nanmean(slice_, axis=0)

    rank_labels = [f"rank {r}\n(exp {r * n_experts_per_rank}–{r * n_experts_per_rank + n_experts_per_rank - 1})"
                   for r in range(max_rank)]

    fig, ax = plt.subplots(figsize=(max(8, len(layer_ids) * 0.35), max(4, max_rank * 0.5)))
    im = ax.imshow(rank_mat, aspect="auto", origin="lower", cmap="YlOrRd", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="avg queuing delay (ms)")

    ax.set_xticks(range(len(layer_ids)))
    ax.set_xticklabels(layer_ids, fontsize=6, rotation=90)
    ax.set_yticks(range(max_rank))
    ax.set_yticklabels(rank_labels, fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rank (experts averaged)")
    ax.set_title("Queuing delay heatmap (per rank, experts averaged)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── Queue Depth Timeseries ────────────────────────────────────────────

def _render_rank_queue_png(
    dev_id: int, timestamps: np.ndarray, layer_depths: list,
    out_path: Path, t_lo: float = None, t_hi: float = None,
    scheduled_layer_ids: list = None,
):
    snapshot_len = max(len(x) for x in layer_depths)
    num_expert = (snapshot_len - 1) // 2
    num_attn = snapshot_len - num_expert
    sampler_idx = num_expert

    if t_lo is not None or t_hi is not None:
        lo = t_lo if t_lo is not None else timestamps[0]
        hi = t_hi if t_hi is not None else timestamps[-1]
        mask = (timestamps >= lo) & (timestamps <= hi)
        timestamps = timestamps[mask]
        layer_depths = [layer_depths[i] for i in range(len(mask)) if mask[i]]
        if scheduled_layer_ids is not None:
            scheduled_layer_ids = [scheduled_layer_ids[i] for i in range(len(mask)) if mask[i]]
        if len(timestamps) < 2:
            return

    ncols = len(timestamps)

    mat = np.array(layer_depths, dtype=float).T
    attn_mat = mat[:num_attn, :]
    expert_mat = mat[num_attn:, :]
    n_pairs = num_expert
    total = n_pairs * 2

    attn_real = attn_mat[:num_expert, :]
    vmax_attn = max(np.nanmax(attn_real), 1) if attn_real.size else 1
    vmax_expert = max(np.nanmax(expert_mat), 1) if expert_mat.size else 1

    rgba = np.ones((total, ncols, 4), dtype=float)
    for i in range(n_pairs):
        t = np.clip(attn_real[i, :] / vmax_attn, 0, 1)
        row_a = i * 2
        rgba[row_a, :, 0] = 1.0 - t
        rgba[row_a, :, 1] = 1.0 - t * 0.6
        rgba[row_a, :, 2] = 1.0

        t = np.clip(expert_mat[i, :] / vmax_expert, 0, 1)
        row_e = i * 2 + 1
        rgba[row_e, :, 0] = 1.0
        rgba[row_e, :, 1] = 1.0 - t * 0.7
        rgba[row_e, :, 2] = 1.0 - t

    fig, ax = plt.subplots(figsize=(20, 10))
    extent = [0, ncols, 0, total]
    ax.imshow(rgba, aspect="auto", origin="lower", extent=extent, interpolation="nearest")

    if scheduled_layer_ids is not None:
        from matplotlib.collections import LineCollection
        segs = []
        for step_idx, sched_lid in enumerate(scheduled_layer_ids):
            if sched_lid == sampler_idx:
                continue
            if sched_lid >= num_attn:
                row = (sched_lid - num_attn) * 2 + 1
            else:
                row = sched_lid * 2
            if row < 0 or row >= total:
                continue
            segs.append([(step_idx + 1, row), (step_idx + 1, row + 1)])
        if segs:
            lc = LineCollection(segs, colors="lime", linewidths=0.5, alpha=0.9)
            ax.add_collection(lc)

    sm_attn = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0, vmax=vmax_attn))
    sm_expert = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=0, vmax=vmax_expert))
    cb_a = fig.colorbar(sm_attn, ax=ax, pad=0.01, aspect=30, fraction=0.02)
    cb_a.set_label("attn queued tokens", fontsize=8)
    cb_e = fig.colorbar(sm_expert, ax=ax, pad=0.01, aspect=30, fraction=0.02)
    cb_e.set_label("expert queued tokens", fontsize=8)

    tick_count = 20
    tick_indices = np.linspace(0, ncols - 1, min(tick_count, ncols), dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f"{timestamps[i] - timestamps[0]:.2f}s" for i in tick_indices],
                       fontsize=6, rotation=45)

    ytick_step = max(1, n_pairs // 12)
    yticks = []
    ylabels = []
    for i in range(0, n_pairs, ytick_step):
        yticks.extend([i * 2, i * 2 + 1])
        ylabels.extend([f"A{i}", f"E{i}"])
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=6)
    ax.set_ylabel("Layer (A=attn, E=expert, interleaved)")
    ax.set_xlabel("Time (step index)")
    ax.set_title(f"Rank {dev_id} — queue depth (blue=attn, red=expert, sampler excluded)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_rank_queue_timeseries(data: dict, out_dir: Path):
    full_dir = out_dir / "rank_queue_timeseries"
    zoom10_dir = out_dir / "rank_queue_timeseries_mid10s"
    zoom1_dir = out_dir / "rank_queue_timeseries_mid1s"
    full_dir.mkdir(parents=True, exist_ok=True)
    zoom10_dir.mkdir(parents=True, exist_ok=True)
    zoom1_dir.mkdir(parents=True, exist_ok=True)

    for dev_id, d in sorted(data.items()):
        qs = d.get("queue_snapshots", {})
        ts_list = qs.get("timestamps_s", [])
        depths = qs.get("layer_depths", [])
        sched = qs.get("scheduled_layer_ids", None)
        if not ts_list or not depths:
            continue
        timestamps = np.array(ts_list, dtype=float)

        _render_rank_queue_png(dev_id, timestamps, depths, full_dir / f"rank_{dev_id:02d}.png",
                               scheduled_layer_ids=sched)

        t_mid = (timestamps[0] + timestamps[-1]) / 2.0
        _render_rank_queue_png(dev_id, timestamps, depths, zoom10_dir / f"rank_{dev_id:02d}.png",
                               t_lo=t_mid - 5.0, t_hi=t_mid + 5.0, scheduled_layer_ids=sched)
        _render_rank_queue_png(dev_id, timestamps, depths, zoom1_dir / f"rank_{dev_id:02d}.png",
                               t_lo=t_mid - 0.5, t_hi=t_mid + 0.5, scheduled_layer_ids=sched)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    adv_log_dir = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        out_dir = Path(sys.argv[2])
    else:
        out_dir = adv_log_dir / "plots"

    if not adv_log_dir.exists():
        print(f"Advanced log dir not found: {adv_log_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Processing advanced logs: {adv_log_dir}")
    print(f"Output directory:         {out_dir}")
    print(f"{'=' * 60}")

    data = load_device_data(adv_log_dir)
    if not data:
        print("No device data found!")
        sys.exit(1)
    print(f"Loaded {len(data)} device(s): {sorted(data.keys())}\n")

    write_summary(data, out_dir / "summary.txt")
    print()

    plot_gemm_time_cdf(data, out_dir / "cdf_gemm_time.png")
    plot_gemm_bsz_cdf(data, out_dir / "cdf_gemm_batchsize.png")
    plot_bsz_vs_time(data, out_dir / "bsz_vs_time.png")
    plot_bsz_vs_time_avg(data, out_dir / "bsz_vs_time_avg.png")
    plot_heatmap_expert(data, out_dir / "heatmap_queue_per_expert.png")
    plot_heatmap_rank(data, out_dir / "heatmap_queue_per_rank.png")
    plot_rank_queue_timeseries(data, out_dir)

    print(f"\nAll outputs saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
