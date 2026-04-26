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
import re
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
            try:
                with open(q_path) as f:
                    queuing = json.load(f)
            except json.JSONDecodeError:
                print(f"  WARNING: corrupt JSON, skipping {q_path}")
        if qs_path.exists():
            try:
                with open(qs_path) as f:
                    queue_snapshots = json.load(f)
            except json.JSONDecodeError:
                print(f"  WARNING: corrupt JSON, skipping {qs_path}")
        dispatcher_puts = {}
        dp_path = dev_dir / "dispatcher_puts.json"
        if dp_path.exists():
            try:
                with open(dp_path) as f:
                    dispatcher_puts = json.load(f)
            except json.JSONDecodeError:
                print(f"  WARNING: corrupt JSON, skipping {dp_path}")
        pool_puts = {}
        pp_path = dev_dir / "pool_puts.json"
        if pp_path.exists():
            try:
                with open(pp_path) as f:
                    pool_puts = json.load(f)
            except json.JSONDecodeError:
                print(f"  WARNING: corrupt JSON, skipping {pp_path}")
        recv_completions = {}
        rc_path = dev_dir / "recv_completions.json"
        if rc_path.exists():
            try:
                with open(rc_path) as f:
                    recv_completions = json.load(f)
            except json.JSONDecodeError:
                print(f"  WARNING: corrupt JSON, skipping {rc_path}")
        pending_send_stalls = {}
        pss_path = dev_dir / "pending_send_stalls.json"
        if pss_path.exists():
            try:
                with open(pss_path) as f:
                    pending_send_stalls = json.load(f)
            except json.JSONDecodeError:
                print(f"  WARNING: corrupt JSON, skipping {pss_path}")
        data[dev_id] = {
            "moe_steps": moe,
            "queuing_delays": queuing,
            "queue_snapshots": queue_snapshots,
            "dispatcher_puts": dispatcher_puts,
            "pool_puts": pool_puts,
            "recv_completions": recv_completions,
            "pending_send_stalls": pending_send_stalls,
        }
    return data


def cdf(values):
    arr = np.sort(values)
    p = np.arange(1, len(arr) + 1) / len(arr)
    return arr, p


def _get_windowed_moe_steps(d: dict, t_lo: float = None, t_hi: float = None):
    moe = d.get("moe_steps", {})
    bsz = moe.get("batch_sizes", [])
    times = moe.get("execution_times_ms", [])
    ts = moe.get("timestamps_s", [])
    if not bsz or not times:
        return [], [], []
    if not ts or (t_lo is None and t_hi is None):
        return bsz, times, ts

    ts_arr = np.array(ts, dtype=float)
    t0 = float(ts_arr.min())
    rel = ts_arr - t0
    mask = np.ones(len(ts_arr), dtype=bool)
    if t_lo is not None:
        mask &= rel >= t_lo
    if t_hi is not None:
        mask &= rel <= t_hi
    idx = np.nonzero(mask)[0]
    return [bsz[i] for i in idx], [times[i] for i in idx], [ts[i] for i in idx]


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

def plot_gemm_time_cdf(data: dict, out_path: Path, t_lo: float = None, t_hi: float = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab20.colors
    plotted = 0
    for dev_id, d in sorted(data.items()):
        _, times, _ = _get_windowed_moe_steps(d, t_lo=t_lo, t_hi=t_hi)
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
    if t_lo is None and t_hi is None:
        ax.set_title("Per-MoE-step groupedGEMM time CDF (per rank)")
    else:
        ax.set_title(f"Per-MoE-step groupedGEMM time CDF (per rank, {t_lo:.0f}-{t_hi:.0f}s)")
    ax.legend(fontsize=7, ncol=4, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_gemm_bsz_cdf(data: dict, out_path: Path, t_lo: float = None, t_hi: float = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab20.colors
    plotted = 0
    for dev_id, d in sorted(data.items()):
        bsz, _, _ = _get_windowed_moe_steps(d, t_lo=t_lo, t_hi=t_hi)
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
    if t_lo is None and t_hi is None:
        ax.set_title("Per-MoE-step groupedGEMM batch size CDF (per rank)")
    else:
        ax.set_title(f"Per-MoE-step groupedGEMM batch size CDF (per rank, {t_lo:.0f}-{t_hi:.0f}s)")
    ax.legend(fontsize=7, ncol=4, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── Batch-size vs Time Plots ──────────────────────────────────────────

def plot_bsz_vs_time(data: dict, out_path: Path, t_lo: float = None, t_hi: float = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab20.colors
    plotted = 0
    for dev_id, d in sorted(data.items()):
        bsz_list, time_list, _ = _get_windowed_moe_steps(d, t_lo=t_lo, t_hi=t_hi)
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
    if t_lo is None and t_hi is None:
        ax.set_title("Per-batch-size averaged groupedGEMM compute time (per rank)")
    else:
        ax.set_title(f"Per-batch-size averaged groupedGEMM compute time (per rank, {t_lo:.0f}-{t_hi:.0f}s)")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_bsz_vs_time_avg(data: dict, out_path: Path, t_lo: float = None, t_hi: float = None):
    all_groups = defaultdict(list)
    for d in data.values():
        bsz_list, time_list, _ = _get_windowed_moe_steps(d, t_lo=t_lo, t_hi=t_hi)
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
    if t_lo is None and t_hi is None:
        ax.set_title("Per-batch-size averaged groupedGEMM compute time (all ranks)")
    else:
        ax.set_title(f"Per-batch-size averaged groupedGEMM compute time (all ranks, {t_lo:.0f}-{t_hi:.0f}s)")
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


# ── Dispatcher Put Latency Plots ──────────────────────────────────────

def plot_dispatcher_latency_cdf(data: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab20.colors
    plotted = 0
    for dev_id, d in sorted(data.items()):
        latencies = d.get("dispatcher_puts", {}).get("latencies_ms", [])
        if not latencies:
            continue
        x, y = cdf(latencies)
        ax.plot(x, y, label=f"rank {dev_id}", color=colors[dev_id % len(colors)], lw=1.2)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("dispatcher.put() latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Dispatcher put latency CDF (per rank) — higher = backpressure from max_pending_sends")
    ax.legend(fontsize=7, ncol=4, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_dispatcher_bsz_vs_latency(data: dict, out_path: Path):
    all_groups = defaultdict(list)
    for d in data.values():
        dp = d.get("dispatcher_puts", {})
        tokens = dp.get("num_tokens", [])
        latencies = dp.get("latencies_ms", [])
        for tok, lat in zip(tokens, latencies):
            all_groups[tok].append(lat)
    if not all_groups:
        return
    xs = sorted(all_groups.keys())
    ys = np.array([np.mean(all_groups[x]) for x in xs])
    ys_std = np.array([np.std(all_groups[x]) for x in xs])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, lw=1.8, color="#ef4444", label="mean across all ranks")
    ax.fill_between(xs, ys - ys_std, ys + ys_std, alpha=0.2, color="#ef4444", label="±1 std")
    ax.set_xlabel("batch size (tokens)")
    ax.set_ylabel("dispatcher.put() latency (ms)")
    ax.set_title("Dispatcher put latency vs batch size — reveals bandwidth utilization")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_dispatcher_timeline(data: dict, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = plt.cm.tab20.colors
    plotted = 0
    for dev_id, d in sorted(data.items()):
        dp = d.get("dispatcher_puts", {})
        ts = dp.get("timestamps_s", [])
        latencies = dp.get("latencies_ms", [])
        tokens = dp.get("num_tokens", [])
        if not ts:
            continue
        ts_arr = np.array(ts) - ts[0]
        axes[0].scatter(ts_arr, latencies, s=2, alpha=0.4,
                        color=colors[dev_id % len(colors)], label=f"rank {dev_id}")
        axes[1].scatter(ts_arr, tokens, s=2, alpha=0.4,
                        color=colors[dev_id % len(colors)], label=f"rank {dev_id}")
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    axes[0].set_ylabel("dispatcher.put() latency (ms)")
    axes[0].set_title("Dispatcher behavior over time")
    axes[0].legend(fontsize=6, ncol=4, loc="upper right")
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("batch size (tokens)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_pool_admission_rate(data: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab20.colors
    plotted = 0
    for dev_id, d in sorted(data.items()):
        pp = d.get("pool_puts", {})
        ts = pp.get("timestamps_s", [])
        if not ts or len(ts) < 2:
            continue
        ts_arr = np.array(ts)
        t_start = ts_arr[0]
        ts_rel = ts_arr - t_start
        bin_width = 0.5
        bins = np.arange(0, ts_rel[-1] + bin_width, bin_width)
        counts, edges = np.histogram(ts_rel, bins=bins)
        rate = counts / bin_width
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, rate, label=f"rank {dev_id}",
                color=colors[dev_id % len(colors)], lw=1.2)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pool admission rate (reqs/s)")
    ax.set_title("New request admission rate into pool over time")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved: {out_path}")


def write_dispatcher_pool_summary(data: dict, out_path: Path):
    lines = []
    lines.append("=" * 70)
    lines.append("Dispatcher & Pool Summary")
    lines.append("=" * 70)

    all_lat = []
    all_tok = []
    for dev_id, d in sorted(data.items()):
        dp = d.get("dispatcher_puts", {})
        latencies = dp.get("latencies_ms", [])
        tokens = dp.get("num_tokens", [])
        pp = d.get("pool_puts", {})
        pool_ts = pp.get("timestamps_s", [])
        all_lat.extend(latencies)
        all_tok.extend(tokens)
        if latencies:
            lines.append(
                f"  rank {dev_id:2d}: {len(latencies):6d} dispatcher puts, "
                f"latency mean={np.mean(latencies):.3f}ms p50={np.median(latencies):.3f}ms "
                f"p99={np.percentile(latencies, 99):.3f}ms max={max(latencies):.3f}ms, "
                f"bsz mean={np.mean(tokens):.0f}, "
                f"pool_puts={len(pool_ts)}"
            )

    if all_lat:
        lines.append("")
        lines.append(f"  ALL RANKS: {len(all_lat)} total dispatcher puts")
        lines.append(f"    latency:   mean={np.mean(all_lat):.3f}ms  p50={np.median(all_lat):.3f}ms  p99={np.percentile(all_lat, 99):.3f}ms  max={max(all_lat):.3f}ms")
        lines.append(f"    batch size: mean={np.mean(all_tok):.0f}  p50={np.median(all_tok):.0f}  max={max(all_tok)}")
        high_lat = [l for l in all_lat if l > 1.0]
        lines.append(f"    high-latency puts (>1ms): {len(high_lat)} ({100*len(high_lat)/len(all_lat):.1f}%)")

    txt = "\n".join(lines)
    out_path.write_text(txt)
    print(txt)
    print(f"\n  saved: {out_path}")


def _build_recv_events(data: dict, include_local: bool = False):
    events = []
    for dev_id, d in sorted(data.items()):
        rc = d.get("recv_completions", {})
        peer_ids = rc.get("peer_ids", [])
        layer_ids = rc.get("layer_ids", [])
        num_tokens = rc.get("num_tokens", [])
        num_bytes = rc.get("num_bytes", [])
        posted_ts = rc.get("posted_timestamps_s", [])
        completed_ts = rc.get("completed_timestamps_s", [])
        is_local = rc.get("is_local", [])
        for peer_id, layer_id, ntok, nbytes, ts0, ts1, local in zip(
            peer_ids, layer_ids, num_tokens, num_bytes, posted_ts, completed_ts, is_local
        ):
            if (not include_local) and local:
                continue
            start = float(min(ts0, ts1))
            end = float(max(ts0, ts1))
            if end <= start:
                end = start + 1e-6
            events.append({
                "device_id": dev_id,
                "node_id": dev_id // 2,
                "peer_id": int(peer_id),
                "layer_id": int(layer_id),
                "num_tokens": int(ntok),
                "num_bytes": int(nbytes),
                "start_s": start,
                "end_s": end,
                "is_local": bool(local),
            })
    return events


def _build_node_bandwidth_series(data: dict, bin_width: float = 0.5):
    events = _build_recv_events(data, include_local=False)
    if not events:
        return None, None
    t0 = min(e["start_s"] for e in events)
    t1 = max(e["end_s"] for e in events)
    nbins = max(1, int(np.ceil((t1 - t0) / bin_width)))
    node_bins = defaultdict(lambda: np.zeros(nbins, dtype=float))
    for e in events:
        start = (e["start_s"] - t0) / bin_width
        end = (e["end_s"] - t0) / bin_width
        b0 = max(0, int(np.floor(start)))
        b1 = min(nbins - 1, int(np.floor(end)))
        duration = e["end_s"] - e["start_s"]
        for b in range(b0, b1 + 1):
            bin_start = t0 + b * bin_width
            bin_end = bin_start + bin_width
            overlap = max(0.0, min(e["end_s"], bin_end) - max(e["start_s"], bin_start))
            if overlap <= 0:
                continue
            node_bins[e["node_id"]][b] += e["num_bytes"] * (overlap / duration)
    centers = (np.arange(nbins) + 0.5) * bin_width
    return centers, node_bins


def write_recv_bandwidth_summary(data: dict, out_path: Path):
    events = _build_recv_events(data, include_local=False)
    lines = []
    lines.append("=" * 70)
    lines.append("Receive Bandwidth & Backpressure Summary")
    lines.append("=" * 70)

    if not events:
        txt = "No receive-completion data found."
        out_path.write_text(txt)
        print(txt)
        print(f"\n  saved: {out_path}")
        return

    durations_ms = np.array([(e["end_s"] - e["start_s"]) * 1000.0 for e in events], dtype=float)
    op_bw = np.array([e["num_bytes"] / (e["end_s"] - e["start_s"]) / 1e9 for e in events], dtype=float)
    centers, node_bins = _build_node_bandwidth_series(data)

    lines.append(f"  remote recv completions: {len(events)}")
    lines.append(
        f"  recv duration: mean={np.mean(durations_ms):.3f}ms p50={np.median(durations_ms):.3f}ms "
        f"p99={np.percentile(durations_ms, 99):.3f}ms max={np.max(durations_ms):.3f}ms"
    )
    lines.append(
        f"  per-op recv bw: mean={np.mean(op_bw):.3f}GB/s p50={np.median(op_bw):.3f}GB/s "
        f"p99={np.percentile(op_bw, 99):.3f}GB/s max={np.max(op_bw):.3f}GB/s"
    )
    lines.append("")
    lines.append("  Per-node receive bandwidth timeline (2 ranks share one 200 Gbps NIC):")
    all_node_gbps = []
    for node_id in sorted(node_bins):
        gbps = node_bins[node_id] / 0.5 / 1e9
        all_node_gbps.extend(gbps.tolist())
        lines.append(
            f"    node {node_id}: mean={np.mean(gbps):.3f}GB/s p95={np.percentile(gbps, 95):.3f}GB/s "
            f"max={np.max(gbps):.3f}GB/s util_p95={100*np.percentile(gbps,95)/25:.1f}% util_max={100*np.max(gbps)/25:.1f}%"
        )

    all_node_gbps = np.array(all_node_gbps, dtype=float)
    if len(all_node_gbps):
        lines.append("")
        lines.append(
            f"  ALL NODES: mean={np.mean(all_node_gbps):.3f}GB/s p95={np.percentile(all_node_gbps,95):.3f}GB/s "
            f"max={np.max(all_node_gbps):.3f}GB/s"
        )

    all_stalls = []
    for dev_id, d in sorted(data.items()):
        ps = d.get("pending_send_stalls", {})
        starts = ps.get("start_timestamps_s", [])
        ends = ps.get("end_timestamps_s", [])
        yields = ps.get("yield_counts", [])
        pending_before = ps.get("pending_before", [])
        max_pending = ps.get("max_pending", [])
        if not starts:
            continue
        stall_ms = [(e - s) * 1000.0 for s, e in zip(starts, ends)]
        all_stalls.extend(stall_ms)
        lines.append(
            f"  rank {dev_id:2d}: pending-send stalls={len(starts)} mean={np.mean(stall_ms):.3f}ms "
            f"p95={np.percentile(stall_ms,95):.3f}ms max={np.max(stall_ms):.3f}ms "
            f"pending_before_mean={np.mean(pending_before):.2f} max_pending={np.mean(max_pending):.2f} yields_mean={np.mean(yields):.1f}"
        )
    if all_stalls:
        all_stalls = np.array(all_stalls, dtype=float)
        lines.append("")
        lines.append(
            f"  ALL RANKS pending-send stalls: count={len(all_stalls)} mean={np.mean(all_stalls):.3f}ms "
            f"p95={np.percentile(all_stalls,95):.3f}ms max={np.max(all_stalls):.3f}ms"
        )
    else:
        lines.append("")
        lines.append("  ALL RANKS pending-send stalls: count=0")

    txt = "\n".join(lines)
    out_path.write_text(txt)
    print(txt)
    print(f"\n  saved: {out_path}")


def plot_node_receive_bandwidth(data: dict, out_path: Path, bin_width: float = 0.5):
    centers, node_bins = _build_node_bandwidth_series(data, bin_width=bin_width)
    if centers is None:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors
    for i, node_id in enumerate(sorted(node_bins)):
        gbps = node_bins[node_id] / bin_width / 1e9
        ax.plot(centers, gbps, lw=1.5, color=colors[i % len(colors)], label=f"node {node_id}")
    ax.axhline(25.0, color="red", linestyle="--", lw=1.0, label="200 Gbps NIC")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("recv bandwidth (GB/s)")
    ax.set_title("Per-node receive bandwidth over time (2 ranks / NIC)")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_pending_send_stalls(data: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab20.colors
    plotted = 0
    for dev_id, d in sorted(data.items()):
        ps = d.get("pending_send_stalls", {})
        starts = ps.get("start_timestamps_s", [])
        ends = ps.get("end_timestamps_s", [])
        if not starts:
            continue
        starts = np.array(starts, dtype=float)
        ends = np.array(ends, dtype=float)
        dur_ms = (ends - starts) * 1000.0
        t0 = starts.min()
        ax.scatter(starts - t0, dur_ms, s=8, alpha=0.7,
                   color=colors[dev_id % len(colors)], label=f"rank {dev_id}")
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("time since first stall in rank (s)")
    ax.set_ylabel("stall duration (ms)")
    ax.set_title("Pending-send stall events (blocked by max_pending_sends)")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved: {out_path}")


GLOBAL_REQ_RE = re.compile(
    r"(?P<ts>\d+\.\d+) - \[INFO\].*Global DP scheduler: #running requests: (?P<running>\d+), #waiting requests: (?P<waiting>\d+)"
)


def parse_global_running_requests(server_log_path: Path):
    if not server_log_path.exists():
        return None
    timestamps = []
    running = []
    waiting = []
    with open(server_log_path, "r", errors="ignore") as f:
        for line in f:
            m = GLOBAL_REQ_RE.search(line)
            if not m:
                continue
            timestamps.append(float(m.group("ts")))
            running.append(int(m.group("running")))
            waiting.append(int(m.group("waiting")))
    if not timestamps:
        return None
    t0 = timestamps[0]
    return {
        "time_s": np.array(timestamps, dtype=float) - t0,
        "running": np.array(running, dtype=float),
        "waiting": np.array(waiting, dtype=float),
    }


def plot_global_running_requests(server_log_path: Path, out_path: Path):
    series = parse_global_running_requests(server_log_path)
    if series is None:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series["time_s"], series["running"], lw=1.8, color="#2563eb", label="running requests")
    ax.plot(series["time_s"], series["waiting"], lw=1.5, color="#dc2626", label="waiting requests")
    ax.set_xlabel("time since first DP-scheduler sample (s)")
    ax.set_ylabel("requests")
    ax.set_title("Global DP scheduler request timeline")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved: {out_path}")


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
    plot_gemm_time_cdf(data, out_dir / "cdf_gemm_time_20_40s.png", t_lo=20.0, t_hi=40.0)
    plot_gemm_bsz_cdf(data, out_dir / "cdf_gemm_batchsize_20_40s.png", t_lo=20.0, t_hi=40.0)
    plot_bsz_vs_time(data, out_dir / "bsz_vs_time_20_40s.png", t_lo=20.0, t_hi=40.0)
    plot_bsz_vs_time_avg(data, out_dir / "bsz_vs_time_avg_20_40s.png", t_lo=20.0, t_hi=40.0)
    plot_heatmap_expert(data, out_dir / "heatmap_queue_per_expert.png")
    plot_heatmap_rank(data, out_dir / "heatmap_queue_per_rank.png")
    plot_rank_queue_timeseries(data, out_dir)

    server_log_path = adv_log_dir.parent / "server.log"
    plot_global_running_requests(server_log_path, out_dir / "global_running_requests.png")

    has_dispatcher = any(d.get("dispatcher_puts", {}).get("latencies_ms") for d in data.values())
    if has_dispatcher:
        print("\n  Dispatcher/pool data found — generating dispatcher/pool plots...")
        write_dispatcher_pool_summary(data, out_dir / "dispatcher_pool_summary.txt")
        plot_dispatcher_latency_cdf(data, out_dir / "cdf_dispatcher_latency.png")
        plot_dispatcher_bsz_vs_latency(data, out_dir / "dispatcher_bsz_vs_latency.png")
        plot_dispatcher_timeline(data, out_dir / "dispatcher_timeline.png")
        plot_pool_admission_rate(data, out_dir / "pool_admission_rate.png")
    else:
        print("\n  No dispatcher/pool data found (run with --enable-advanced-logging to collect)")

    has_recv = any(d.get("recv_completions", {}).get("num_bytes") for d in data.values())
    has_stalls = any(d.get("pending_send_stalls", {}).get("start_timestamps_s") for d in data.values())
    if has_recv or has_stalls:
        print("\n  Receive/backpressure data found — generating corrected bandwidth/backpressure outputs...")
        write_recv_bandwidth_summary(data, out_dir / "recv_bandwidth_summary.txt")
        plot_node_receive_bandwidth(data, out_dir / "node_receive_bandwidth.png")
        plot_pending_send_stalls(data, out_dir / "pending_send_stalls.png")
    else:
        print("\n  No receive/backpressure data found in advanced logs")

    print(f"\nAll outputs saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
