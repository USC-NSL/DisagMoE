# Benchmark Plot Styling Guide

Consistent styling conventions for all benchmark comparison plots in this project.

## Script Structure

```python
#!/usr/bin/env python3
"""<one-line description>"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
```

- Always use `matplotlib.use("Agg")` for headless rendering.
- Always derive output path from `__file__` so the script works from any working directory.
- Close figure with `plt.close(fig)` after saving.

## Layout

- **1×3 subplots** for the standard triple (throughput, median ITL, p99 ITL).
- `figsize=(20, 6)` for 3 panels.
- `fig.tight_layout()` before saving.

## Data Organization

Systems go on the **x-axis** (one bar group per system). Workload levels are distinguished by bar position and alpha, not by separate colors.

```python
SYSTEMS = ["System A", "System B\n(variant)", ...]

# Indexed by system, not by workload
output_tput_high = [sys_a_high, sys_b_high, ...]
output_tput_low  = [sys_a_low,  sys_b_low,  ...]
```

- Use `\n` in system names for multi-line x-tick labels when a variant label is needed.
- High workload = more prompts at higher rate. Low workload = fewer prompts at lower rate.

## Colors

One color per system. Workload differentiation uses **alpha** (0.85 for high, 0.50 for low).

```python
COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#27ae60", ...]
```

Palette reference:
| Hex | Use for |
|-----|---------|
| `#e74c3c` | Red — baseline / competitor |
| `#3498db` | Blue — our system (config A) |
| `#2980b9` | Darker blue — our system (config A variant) |
| `#2ecc71` | Green — our system (config B) |
| `#27ae60` | Darker green — our system (config B variant) |

Add more colors from the same flat-UI palette if needed. Avoid pastels that look washed out on projectors.

## Bars

```python
w = 0.35

bars_high = ax.bar(x - w/2, data_high, w,
                   label="High (2000 req @ 500/s)",
                   color=COLORS, edgecolor="black", linewidth=0.5, alpha=0.85)
bars_low  = ax.bar(x + w/2, data_low,  w,
                   label="Low (1000 req @ 250/s)",
                   color=COLORS, edgecolor="black", linewidth=0.5, alpha=0.50)
```

- Bar width `w = 0.35`.
- Black edge, `linewidth=0.5`.
- High workload on the left (`x - w/2`), low on the right (`x + w/2`).
- Legend describes workload, not system (systems are already on x-axis).

## Value Annotations

```python
for bar in bars:
    h = bar.get_height()
    ax.annotate(fmt.format(h),
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=fontsize_val)
```

- Use `ax.annotate` with `xytext=(0, 3)` offset — not `ax.text`.
- Default format `"{:.0f}"` (no decimals). Use `"{:.1f}"` for sub-millisecond values.
- `fontsize=7` for value labels (small enough not to overlap).

## Font Sizes

| Element | Size | Weight |
|---------|------|--------|
| `suptitle` | 14 | bold |
| Subplot `title` | 12 | bold |
| `ylabel` | 10 | normal |
| x-tick labels | 8 | normal |
| Legend | 8 | normal |
| Value annotations | 7 | normal |

## Grid & Axes

```python
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
```

- Y-axis grid only, `alpha=0.3`.
- Grid behind bars (`set_axisbelow`).

## Suptitle

```python
fig.suptitle("<Model> on <Hardware>: <System A> vs <System B> vs ...",
             fontsize=14, fontweight="bold", y=1.02)
```

- Single line preferred. Mention model and hardware.
- `y=1.02` to sit above the subplots with `tight_layout`.

## Saving

```python
out_path = os.path.join(OUT_DIR, FILENAME)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_path}")
```

- `dpi=150` — good for slides and docs without bloating file size.
- `bbox_inches="tight"` — no whitespace clipping.
- Always `print` the output path on success.

## Complete Template

```python
#!/usr/bin/env python3
"""Generate benchmark comparison plot."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

SYSTEMS = ["System A", "System B"]

# High workload (indexed by system)
output_tput_high = [1000, 2000]
median_itl_high  = [100,  50]
p99_itl_high     = [500,  200]

# Low workload (indexed by system)
output_tput_low  = [800,  1500]
median_itl_low   = [120,  60]
p99_itl_low      = [600,  250]

COLORS = ["#e74c3c", "#3498db"]
FILENAME = "benchmark_comparison.png"


def plot_subplot(ax, title, ylabel, data_high, data_low, fmt="{:.0f}", fontsize_val=7):
    x = np.arange(len(SYSTEMS))
    w = 0.35

    bars_high = ax.bar(x - w/2, data_high, w, label="High (2000 req @ 500/s)",
                       color=COLORS, edgecolor="black", linewidth=0.5, alpha=0.85)
    bars_low  = ax.bar(x + w/2, data_low,  w, label="Low (1000 req @ 250/s)",
                       color=COLORS, edgecolor="black", linewidth=0.5, alpha=0.50)

    for bars in [bars_high, bars_low]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(fmt.format(h), xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=fontsize_val)

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(SYSTEMS, fontsize=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    plot_subplot(axes[0], "Output Token Throughput", "Tokens/s",
                 output_tput_high, output_tput_low)
    plot_subplot(axes[1], "Median Inter-Token Latency", "Latency (ms)",
                 median_itl_high, median_itl_low)
    plot_subplot(axes[2], "P99 Inter-Token Latency", "Latency (ms)",
                 p99_itl_high, p99_itl_low)

    fig.suptitle("<Model> on <Hardware>: <System A> vs <System B>",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, FILENAME)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
```
