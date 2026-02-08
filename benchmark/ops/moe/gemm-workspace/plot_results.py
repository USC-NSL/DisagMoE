#!/usr/bin/env python3
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

def _display_model_name(basename: str) -> str:
    parts = basename.split('_')
    if len(parts) >= 2:
        tail = parts[-1]
        if tail.isupper() and any(ch.isdigit() for ch in tail):
            return '_'.join(parts[:-1])
    return basename

def plot_model_results(detailed_csv_path, output_dir):
    basename = os.path.basename(detailed_csv_path)
    if basename.endswith('_rotated_detailed.csv'):
        basename = basename.replace('_rotated_detailed.csv', '')
    else:
        basename = basename.replace('_detailed.csv', '')
    
    df = pd.read_csv(detailed_csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'MoE GEMM Profiling: {basename}', fontsize=14)
    
    ax1 = axes[0, 0]
    ax1.plot(df['batch_size'], df['plain_per_expert_ms'], label='Plain GEMM', alpha=0.8)
    ax1.plot(df['batch_size'], df['grouped_per_expert_ms'], label='Grouped GEMM', alpha=0.8)
    ax1.set_xlabel('Per-Expert Batch Size (tokens)')
    ax1.set_ylabel('Time per Expert (ms)')
    ax1.set_title('Per-Expert Latency vs Batch Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    speedup = df['plain_per_expert_ms'] / df['grouped_per_expert_ms']
    ax2.plot(df['batch_size'], speedup, color='green', alpha=0.8)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax2.set_xlabel('Per-Expert Batch Size (tokens)')
    ax2.set_ylabel('Speedup (Plain / Grouped)')
    ax2.set_title('Grouped GEMM Speedup over Plain GEMM')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.plot(df['batch_size'], df['plain_per_expert_ms'], label='Plain GEMM', alpha=0.8)
    ax3.plot(df['batch_size'], df['grouped_per_expert_ms'], label='Grouped GEMM', alpha=0.8)
    ax3.set_xlabel('Per-Expert Batch Size (tokens)')
    ax3.set_ylabel('Time per Expert (ms)')
    ax3.set_title('Per-Expert Latency (Log Scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    ax3.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    ax3.get_xaxis().set_major_formatter(ScalarFormatter())
    
    ax4 = axes[1, 1]
    throughput_plain = df['batch_size'] / (df['plain_per_expert_ms'])
    throughput_grouped = df['batch_size'] / (df['grouped_per_expert_ms'])
    ax4.plot(df['batch_size'], throughput_plain, label='Plain GEMM', alpha=0.8)
    ax4.plot(df['batch_size'], throughput_grouped, label='Grouped GEMM', alpha=0.8)
    ax4.set_xlabel('Per-Expert Batch Size (tokens)')
    ax4.set_ylabel('Throughput (tokens/ms/expert)')
    ax4.set_title('Throughput vs Batch Size')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{basename}_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_all_models_comparison(plots_dir):
    detailed_csvs = glob.glob(os.path.join(plots_dir, '*_rotated_detailed.csv'))
    if len(detailed_csvs) < 2:
        return

    detailed_csvs = sorted(detailed_csvs)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle('Cross-Model Comparison (Plain vs Grouped GEMM)', fontsize=14)

    for idx, csv_path in enumerate(detailed_csvs):
        basename = os.path.basename(csv_path).replace('_rotated_detailed.csv', '')
        model_name = _display_model_name(basename)
        df = pd.read_csv(csv_path)
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None

        ax.plot(
            df['batch_size'],
            df['grouped_per_expert_ms'],
            label=f'{model_name} (Grouped)',
            linestyle='-',
            color=color,
            alpha=0.9,
        )
        ax.plot(
            df['batch_size'],
            df['plain_per_expert_ms'],
            label=f'{model_name} (Plain)',
            linestyle='--',
            color=color,
            alpha=0.9,
        )

    ax.set_xlabel('Per-Expert Batch Size (tokens)')
    ax.set_ylabel('Time per Expert (ms)')
    ax.set_title('Per-Expert Latency (Grouped = solid, Plain = dashed)')
    ax.set_ylim(bottom=0)
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(plots_dir, 'all_models_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_workspace_comparison_all_models(plots_dir):
    rotated_csvs = glob.glob(os.path.join(plots_dir, '*_rotated_detailed.csv'))
    if not rotated_csvs:
        return

    rotated_csvs = sorted(rotated_csvs)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle('Workspace Rotation Effect (All Models, REAL)', fontsize=14)

    for idx, rot_path in enumerate(rotated_csvs):
        base = os.path.basename(rot_path).replace('_rotated_detailed.csv', '')
        no_rot_path = os.path.join(plots_dir, base + '_no_rotation_detailed.csv')
        if not os.path.exists(no_rot_path):
            raise FileNotFoundError(f"Missing no-rotation detailed CSV for {base}: {no_rot_path}")

        model_name = _display_model_name(base)
        df_rot = pd.read_csv(rot_path)
        df_no = pd.read_csv(no_rot_path)

        bs_rot = df_rot['batch_size'].to_numpy()
        bs_no = df_no['batch_size'].to_numpy()
        if not np.array_equal(bs_rot, bs_no):
            raise ValueError(f"Batch sizes mismatch between rotated and no-rotation for {base}")

        color = color_cycle[idx % len(color_cycle)] if color_cycle else None

        ax.plot(
            df_rot['batch_size'],
            df_rot['grouped_per_expert_ms'],
            label=f'{model_name} (rotation)',
            linestyle='-',
            color=color,
            alpha=0.9,
        )
        ax.plot(
            df_no['batch_size'],
            df_no['grouped_per_expert_ms'],
            label=f'{model_name} (no rotation, ws=1)',
            linestyle='--',
            color=color,
            alpha=0.9,
        )

    ax.set_xlabel('Per-Expert Batch Size (tokens)')
    ax.set_ylabel('Time per Expert (ms)')
    ax.set_title('Grouped GEMM Per-Expert Latency')
    ax.set_ylim(bottom=0)
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(plots_dir, 'workspace_rotation_all_models.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    
    if not os.path.exists(plots_dir):
        print(f"No plots directory found at {plots_dir}")
        return
    
    detailed_csvs = glob.glob(os.path.join(plots_dir, '*_rotated_detailed.csv'))
    
    if not detailed_csvs:
        print("No detailed CSV files found in plots directory")
        return
    
    print(f"Found {len(detailed_csvs)} detailed CSV files")
    
    for csv_path in detailed_csvs:
        print(f"Processing: {csv_path}")
        plot_model_results(csv_path, plots_dir)
        # Workspace-rotation comparison is generated once across models.
    
    plot_all_models_comparison(plots_dir)
    plot_workspace_comparison_all_models(plots_dir)
    
    print("\nPlot generation complete!")

if __name__ == '__main__':
    main()
