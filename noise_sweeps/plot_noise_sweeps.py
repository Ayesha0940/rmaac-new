import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

NOISE_DIR = Path(__file__).parent
PLOTS_DIR = NOISE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

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
MU_VALUES = [-1, 0, 1]

FILE_RE = re.compile(r"^(simple_.+?)__(.+?)(?:_mu[-\d.]+)?_actstd_tstart_sweep\.csv$")

# data[scenario][algo] = list of DataFrames
data = defaultdict(lambda: defaultdict(list))

for csv_file in sorted(NOISE_DIR.glob("*.csv")):
    m = FILE_RE.match(csv_file.name)
    if not m:
        continue
    scenario, algo_key = m.group(1), m.group(2)
    if algo_key not in ALGO_MAP:
        continue

    df = pd.read_csv(csv_file)
    if "noise_mu" not in df.columns:
        df.insert(0, "noise_mu", 0)
    # Normalise to integer mu values to avoid float comparison issues
    df["noise_mu"] = df["noise_mu"].round().astype(int)
    data[scenario][algo_key].append(df)

for scenario, algo_dfs in sorted(data.items()):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(scenario, fontsize=14, fontweight="bold")

    legend_handles, legend_labels = [], []

    for ax_idx, mu_val in enumerate(MU_VALUES):
        ax = axes[ax_idx]
        ax.set_title(f"noise_mu = {mu_val}")
        ax.set_xlabel("action_noise_std")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)

        for algo_key in ALGO_ORDER:
            if algo_key not in algo_dfs:
                continue
            combined = pd.concat(algo_dfs[algo_key], ignore_index=True)
            subset = combined[combined["noise_mu"] == mu_val].sort_values("action_noise_std")
            if subset.empty:
                continue

            display = ALGO_MAP[algo_key]
            color = COLORS[display]
            x = subset["action_noise_std"]

            (line_nd,) = ax.plot(x, subset["reward_noise_no_diffusion"], color=color, linestyle="-", linewidth=1.8)
            (line_wd,) = ax.plot(x, subset["best_reward_with_diffusion"], color=color, linestyle=":", linewidth=1.8)

            # Add to shared legend only on first occurrence
            label_nd = f"{display}: no diffusion"
            label_wd = f"{display}: with diffusion"
            if label_nd not in legend_labels:
                legend_handles += [line_nd, line_wd]
                legend_labels += [label_nd, label_wd]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.14),
        fontsize=9,
    )
    fig.tight_layout()
    out_path = PLOTS_DIR / f"{scenario}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
