#!/usr/bin/env python3
"""Plot in-flight request timelines from AsyncMoE server logs.

Usage: python plot_asyncmoe_inflight_timeline.py <results_dir>
  results_dir should contain asyncmoe-*/ subdirectories, each with server.log
"""
import re, os, sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <results_dir>")
    sys.exit(1)

BASE = sys.argv[1]
DIRS = sorted(glob.glob(os.path.join(BASE, "asyncmoe-*")))

LOG_PATTERN = re.compile(
    r"(\d+\.\d+) - \[INFO\].*#running requests: (\d+),\s*#waiting requests: (\d+)"
)
LOG_PATTERN_OLD = re.compile(
    r"(\d+\.\d+) - \[INFO\].*#running requests: (\d+)"
)

def label_from_dir(d):
    name = os.path.basename(d)
    parts = name.replace("asyncmoe-", "").split("_")
    return f"{parts[0]} ({parts[1]})"

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for i, d in enumerate(DIRS):
    log_path = os.path.join(d, "server.log")
    if not os.path.isfile(log_path):
        continue

    with open(log_path) as f:
        content = f.read()

    timestamps, running, waiting = [], [], []
    for m in LOG_PATTERN.finditer(content):
        timestamps.append(float(m.group(1)))
        running.append(int(m.group(2)))
        waiting.append(int(m.group(3)))

    if not timestamps:
        for m in LOG_PATTERN_OLD.finditer(content):
            timestamps.append(float(m.group(1)))
            running.append(int(m.group(2)))
        waiting = None

    if not timestamps:
        continue

    t0 = timestamps[0]
    rel = [t - t0 for t in timestamps]
    c = colors[i % len(colors)]
    label = label_from_dir(d)

    axes[0].plot(rel, running, label=label, linewidth=1.5, color=c)
    if waiting is not None:
        axes[1].plot(rel, waiting, label=label, linewidth=1.5, color=c)

axes[0].set_ylabel("Running Requests", fontsize=11)
axes[0].set_title("AsyncMoE In-Flight Request Timelines", fontsize=13)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Time since first log entry (s)", fontsize=11)
axes[1].set_ylabel("Waiting Requests", fontsize=11)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
out = os.path.join(BASE, "asyncmoe_inflight_timeline.png")
fig.savefig(out, dpi=180)
print(f"Saved to {out}")
