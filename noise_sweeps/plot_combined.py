import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

NOISE_DIR = Path(__file__).parent
OUT_DIR = NOISE_DIR / "plots" / "combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALGO_MAP = {
    "earniebest": "ernie",
    "m3ddpgbest": "m3ddpg",
    "maddpgbest": "maddpg",
    "rmaacbest": "rmaac",
}
COLORS = {
    "ernie": "tab:blue",
    "m3ddpg": "tab:orange",
    "maddpg": "tab:green",
    "rmaac": "tab:red",
}
ALGO_ORDER = ["earniebest", "m3ddpgbest", "maddpgbest", "rmaacbest"]

# ── load noise sweep data ─────────────────────────────────────────────────────
NOISE_RE = re.compile(r"^(simple_.+?)__(.+?)(?:_mu[-\d.]+)?_actstd_tstart_sweep\.csv$")
noise_data = defaultdict(lambda: defaultdict(list))  # [scenario][algo] -> list[df]

for f in sorted(NOISE_DIR.glob("seed0/*.csv")):
    m = NOISE_RE.match(f.name)
    if not m or m.group(2) not in ALGO_MAP:
        continue
    df = pd.read_csv(f)
    if "noise_mu" not in df.columns:
        df.insert(0, "noise_mu", 0)
    df["noise_mu"] = df["noise_mu"].round().astype(int)
    noise_data[m.group(1)][m.group(2)].append(df)

# ── load delay sweep data ─────────────────────────────────────────────────────
DELAY_RE = re.compile(r"^(simple_.+?)__(.+?)_delay_sweep\.csv$")
delay_data = defaultdict(dict)  # [scenario][algo] -> df

for f in sorted((NOISE_DIR / "delay_sweeps").glob("*.csv")):
    m = DELAY_RE.match(f.name)
    if not m or m.group(2) not in ALGO_MAP:
        continue
    delay_data[m.group(1)][m.group(2)] = pd.read_csv(f).sort_values("delay_k")

# ── load actuation fault data ─────────────────────────────────────────────────
FAULT_RE = re.compile(r"^(simple_.+?)__(.+?)_stuck_at_sweep\.csv$")
fault_data = defaultdict(dict)  # [scenario][algo] -> df

for f in sorted((NOISE_DIR / "actuation_fault").glob("*.csv")):
    m = FAULT_RE.match(f.name)
    if not m or m.group(2) not in ALGO_MAP:
        continue
    fault_data[m.group(1)][m.group(2)] = pd.read_csv(f).sort_values("sp_prob")

# ── collect all scenarios ─────────────────────────────────────────────────────
scenarios = sorted(set(noise_data) | set(delay_data) | set(fault_data))


def plot_lines(ax, algo_dfs, x_col, y_nd_col, y_wd_col, legend_handles, legend_labels):
    """Plot no-diffusion (solid) and with-diffusion (dotted) lines for each algo."""
    for algo_key in ALGO_ORDER:
        if algo_key not in algo_dfs:
            continue
        df = algo_dfs[algo_key]
        display = ALGO_MAP[algo_key]
        color = COLORS[display]

        (line_nd,) = ax.plot(df[x_col], df[y_nd_col], color=color, linestyle="-", linewidth=1.8)
        (line_wd,) = ax.plot(df[x_col], df[y_wd_col], color=color, linestyle=":", linewidth=1.8)

        label_nd = f"{display}: no diffusion"
        label_wd = f"{display}: with diffusion"
        if label_nd not in legend_labels:
            legend_handles += [line_nd, line_wd]
            legend_labels += [label_nd, label_wd]


# ── plot ──────────────────────────────────────────────────────────────────────
for scenario in scenarios:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.suptitle(scenario, fontsize=13, fontweight="bold")

    legend_handles, legend_labels = [], []
    all_y = []

    # subplot 0: delay sweep
    ax = axes[0]
    ax.set_title("Delay Sweep")
    ax.set_xlabel("delay_k")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    if scenario in delay_data:
        plot_lines(ax, delay_data[scenario], "delay_k", "reward_noisy",
                   "best_reward_with_diffusion", legend_handles, legend_labels)
    for line in ax.get_lines():
        all_y.extend(line.get_ydata().tolist())

    # subplot 1: actuation fault
    ax = axes[1]
    ax.set_title("Actuation Fault")
    ax.set_xlabel("sp_prob")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    if scenario in fault_data:
        plot_lines(ax, fault_data[scenario], "sp_prob", "reward_noisy",
                   "best_reward_with_diffusion", legend_handles, legend_labels)
    for line in ax.get_lines():
        all_y.extend(line.get_ydata().tolist())

    # subplots 2-4: noise sweep for mu = -1, 0, 1
    for ax_idx, mu_val in enumerate([-1, 0, 1]):
        ax = axes[2 + ax_idx]
        ax.set_title(f"Noise (μ={mu_val})")
        ax.set_xlabel("action_noise_std")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)

        if scenario not in noise_data:
            continue

        # build per-algo dict filtered to this mu
        mu_dfs = {}
        for algo_key, dfs in noise_data[scenario].items():
            combined = pd.concat(dfs, ignore_index=True)
            subset = combined[combined["noise_mu"] == mu_val].sort_values("action_noise_std")
            if not subset.empty:
                mu_dfs[algo_key] = subset

        plot_lines(ax, mu_dfs, "action_noise_std", "reward_noise_no_diffusion",
                   "best_reward_with_diffusion", legend_handles, legend_labels)
        for line in ax.get_lines():
            all_y.extend(line.get_ydata().tolist())

    # apply shared y-axis limits across all 5 subplots
    if all_y:
        y_min, y_max = min(all_y), max(all_y)
        pad = (y_max - y_min) * 0.05 or 0.1
        for ax in axes:
            ax.set_ylim(y_min - pad, y_max + pad)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.14),
        fontsize=9,
    )
    fig.tight_layout()
    out_path = OUT_DIR / f"{scenario}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
