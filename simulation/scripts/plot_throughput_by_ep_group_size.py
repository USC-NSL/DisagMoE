#!/usr/bin/env python3
"""
Plot throughput vs ep_group_size for async and sync logs.

Usage:
  python simulation/scripts/plot_throughput_by_ep_group_size.py \
    --async-csv /path/to/experiment_results_async.csv \
    --sync-csv /path/to/experiment_results_sync.csv \
    --output throughput_vs_ep_group_size.png

Notes:
- X axis: ep_group_size
- Y axis: avg_throughput_req_per_sec
- One colored line per unique configuration of non-output columns (e.g., global_request_max_batch_size, attn_service_t, net_delay)
- Line style encodes mode: async (dashed), sync (solid)
"""

import argparse
import os
from typing import List, Tuple, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Columns that are NOT configuration (excluded from config grouping)
OUTPUT_COLUMNS = {
    "avg_token_latency_ms",
    "makespan_ms",
    "avg_throughput_req_per_sec",
    "avg_request_latency_ms",
    "p90_request_latency_ms",
    "p99_request_latency_ms",
    "avg_layer_runtime_ms",
    "avg_layer_wait_imbalance_ms",
    "avg_per_expert_batch_size",
    "avg_layer_worker_queue_stddev",
}
NON_CONFIG_COLUMNS = {"mode", "ep_group_size"} | OUTPUT_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot throughput vs ep_group_size for async and sync CSV logs."
    )
    parser.add_argument(
        "--async-csv", required=True, type=str, help="Path to async CSV log."
    )
    parser.add_argument(
        "--sync-csv", required=True, type=str, help="Path to sync CSV log."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="throughput_vs_ep_group_size.png",
        help="Output image file. If not provided, defaults to throughput_vs_ep_group_size.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Throughput vs Expert Group Size",
        help="Plot title.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="auto",
        help="Colormap name or 'auto'. If 'auto', uses tab20 (<=20 configs), "
             "tab20/tab20b/tab20c combined (<=60), else 'nipy_spectral'.",
    )
    parser.add_argument(
        "--config-cols",
        type=str,
        default=None,
        help="Comma-separated list of columns to define configuration (overrides auto-detection).",
    )
    return parser.parse_args()


def read_csv_enforced(path: str, expected_mode: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # If mode column exists, keep only the expected mode (robustness if file contains multiple modes)
    if "mode" in df.columns:
        df = df[df["mode"] == expected_mode].copy()
    return df


def detect_config_columns(async_cols: List[str], sync_cols: List[str]) -> List[str]:
    union_cols = set(async_cols) | set(sync_cols)
    config_cols = sorted([c for c in union_cols if c not in NON_CONFIG_COLUMNS])
    return config_cols


def build_config_id_and_label(row: pd.Series, config_cols: List[str]) -> Tuple[Tuple[Any, ...], str]:
    values: List[Any] = []
    parts: List[str] = []
    for col in config_cols:
        val = row.get(col)
        # Represent NaN consistently in labels/ids
        if pd.isna(val):
            val_repr = "nan"
        else:
            # Normalize floats with minimal representation
            if isinstance(val, float):
                val_repr = f"{val:.6g}"
            else:
                val_repr = str(val)
        values.append(val if not pd.isna(val) else "nan")
        parts.append(f"{col}={val_repr}")
    label = ", ".join(parts) if parts else "(default)"
    return tuple(values), label


def prepare_series_by_config(df: pd.DataFrame, config_cols: List[str]) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    """
    Returns:
        mapping: config_id -> {
            'label': str,
            'series': pd.Series indexed by ep_group_size with mean throughput values
        }
    """
    if not {"ep_group_size", "avg_throughput_req_per_sec"}.issubset(df.columns):
        missing = {"ep_group_size", "avg_throughput_req_per_sec"} - set(df.columns)
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Compute config_id and label
    if config_cols:
        ids_labels = df.apply(lambda r: build_config_id_and_label(r, config_cols), axis=1, result_type="expand")
        df = df.copy()
        df["__config_id"] = ids_labels[0]
        df["__config_label"] = ids_labels[1]
    else:
        df = df.copy()
        df["__config_id"] = [tuple()] * len(df)
        df["__config_label"] = ["(default)"] * len(df)

    mapping: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for cfg_id, g in df.groupby("__config_id"):
        label = g["__config_label"].iloc[0]
        # Aggregate throughput by ep_group_size, mean across duplicates
        series = (
            g.groupby("ep_group_size")["avg_throughput_req_per_sec"]
            .mean()
            .sort_index()
        )
        mapping[cfg_id] = {"label": label, "series": series}
    return mapping


def main() -> None:
    args = parse_args()

    async_df = read_csv_enforced(args.async_csv, expected_mode="async")
    sync_df = read_csv_enforced(args.sync_csv, expected_mode="sync")

    # Decide which columns define a configuration
    if args.config_cols:
        config_cols = [c.strip() for c in args.config_cols.split(",") if c.strip()]
    else:
        config_cols = detect_config_columns(async_df.columns.tolist(), sync_df.columns.tolist())

    # Prepare per-config series
    async_map = prepare_series_by_config(async_df, config_cols)
    sync_map = prepare_series_by_config(sync_df, config_cols)

    # Union of config ids across both
    all_cfg_ids = sorted(set(async_map.keys()) | set(sync_map.keys()), key=lambda t: (len(t), tuple(t)))

    # Create a stable, non-repeating color assignment
    num_configs = max(1, len(all_cfg_ids))
    color_for_cfg: Dict[Tuple[Any, ...], Any] = {}
    colors_list = None
    if args.cmap != "auto":
        cmap = plt.get_cmap(args.cmap, num_configs)
        for idx, cfg_id in enumerate(all_cfg_ids):
            color_for_cfg[cfg_id] = cmap(idx)
    else:
        if num_configs <= 20:
            cmap = plt.get_cmap("tab20", num_configs)
            for idx, cfg_id in enumerate(all_cfg_ids):
                color_for_cfg[cfg_id] = cmap(idx)
        elif num_configs <= 60:
            # Combine tab20, tab20b, tab20c to get up to 60 distinct discrete colors
            colors_list = list(plt.get_cmap("tab20").colors) \
                        + list(plt.get_cmap("tab20b").colors) \
                        + list(plt.get_cmap("tab20c").colors)
            for idx, cfg_id in enumerate(all_cfg_ids):
                color_for_cfg[cfg_id] = colors_list[idx]
        else:
            # Fall back to a continuous map with distinct sampling
            cmap = plt.get_cmap("nipy_spectral", num_configs)
            for idx, cfg_id in enumerate(all_cfg_ids):
                color_for_cfg[cfg_id] = cmap(idx)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each config for async and sync
    labeled_cfg_ids = set()
    for cfg_id in all_cfg_ids:
        color = color_for_cfg[cfg_id]
        # async line (dashed)
        if cfg_id in async_map:
            s_async = async_map[cfg_id]["series"]
            # Do not add a separate legend entry for async; share the entry with sync/config legend
            ax.plot(s_async.index.values, s_async.values, linestyle="--", marker="o", color=color, linewidth=1.8, markersize=4, alpha=0.95, label=None)
        # sync line (solid)
        if cfg_id in sync_map:
            s_sync = sync_map[cfg_id]["series"]
            # Add one legend entry per configuration using the sync line (solid) for display
            label_sync = sync_map[cfg_id]["label"]
            add_label = label_sync if cfg_id not in labeled_cfg_ids else None
            ax.plot(s_sync.index.values, s_sync.values, linestyle="-", marker="o", color=color, linewidth=1.8, markersize=4, alpha=0.95, label=add_label)
            labeled_cfg_ids.add(cfg_id)

    ax.set_xlabel("ep_group_size")
    ax.set_ylabel("avg_throughput_req_per_sec")
    ax.set_title(args.title)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Reserve narrower space on the right for legends
    fig.subplots_adjust(right=0.8)

    # 1) Mode legend (line style), placed in reserved right margin (top)
    mode_handles = [
        Line2D([0], [0], color="black", linestyle="--", label="async"),
        Line2D([0], [0], color="black", linestyle="-", label="sync"),
    ]
    mode_legend = ax.legend(
        handles=mode_handles,
        title="Mode (line style)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize="small",
    )
    ax.add_artist(mode_legend)

    # 2) Configuration legend with one entry per configuration color (bottom of reserved area)
    ax.legend(
        title="Configurations (color)",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
        fontsize="xx-small",
        frameon=True,
    )

    # Keep tight layout within the left area only
    plt.tight_layout(rect=(0, 0, 0.8, 1))

    out_path = args.output if args.output else "throughput_vs_ep_group_size.png"
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    # Do not call plt.show() to avoid blocking in headless environments
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()


