import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

DATA_DIR = Path(__file__).parent / "guassian_noise" / "seed49" / "predprey"
OUT_DIR  = Path(__file__).parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "grid.linewidth": 0.4,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ALGO_MAP = {
    "earniebest": "ERNIE",
    "m3ddpgbest": "M3DDPG",
    "maddpgbest": "MADDPG",
    "rmaacbest":  "RMAAC",
}
COLORS = {
    "ERNIE":  "#0072B2",
    "M3DDPG": "#E69F00",
    "MADDPG": "#009E73",
    "RMAAC":  "#D55E00",
}
ALGO_ORDER = ["earniebest", "m3ddpgbest", "maddpgbest", "rmaacbest"]

FILE_RE = re.compile(r"^(simple_.+?)__(.+?)(?:_mu[-\d.]+)?_actstd_tstart_sweep\.csv$")

# ── load ──────────────────────────────────────────────────────────────────────
algo_dfs = defaultdict(list)
for f in sorted(DATA_DIR.glob("*.csv")):
    m = FILE_RE.match(f.name)
    if not m or m.group(2) not in ALGO_MAP:
        continue
    df = pd.read_csv(f)[["noise_mu", "action_noise_std",
                          "reward_prey_noise_no_diffusion",
                          "best_reward_prey_with_diffusion"]]
    df["noise_mu"] = df["noise_mu"].round().astype(int)
    algo_dfs[m.group(2)].append(df)

# merge mu files per algo
merged = {}
for algo_key, dfs in algo_dfs.items():
    merged[algo_key] = pd.concat(dfs, ignore_index=True).sort_values(
        ["noise_mu", "action_noise_std"])

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6), sharex=True)
fig.suptitle("Predator–Prey (simple_tag) — prey reward", fontsize=8, fontweight="bold")

legend_handles, legend_labels = [], []

for ax_idx, mu_val in enumerate([-1, 0, 1]):
    ax = axes[ax_idx]
    ax.set_title(f"Biased noise $\\mu={'+' if mu_val > 0 else ''}{mu_val}$",
                 fontweight="bold", pad=3)
    ax.set_xlabel(r"$\alpha$")
    ax.grid(True, alpha=0.3)
    ax.tick_params(length=2, pad=1.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    if ax_idx == 0:
        ax.set_ylabel("Prey reward")

    for algo_key in ALGO_ORDER:
        if algo_key not in merged:
            continue
        subset = merged[algo_key][merged[algo_key]["noise_mu"] == mu_val] \
                     .sort_values("action_noise_std")
        if subset.empty:
            continue
        display = ALGO_MAP[algo_key]
        color = COLORS[display]
        x = subset["action_noise_std"]

        (line_nd,) = ax.plot(x, subset["reward_prey_noise_no_diffusion"],
                             color=color, linestyle="-", linewidth=1.0)
        (line_wd,) = ax.plot(x, subset["best_reward_prey_with_diffusion"],
                             color=color, linestyle=":", linewidth=1.1)

        label_nd = f"{display}: no diffusion"
        label_wd = f"{display}: with diffusion"
        if label_nd not in legend_labels:
            legend_handles += [line_nd, line_wd]
            legend_labels  += [label_nd, label_wd]

color_handles = [Line2D([], [], color=COLORS[ALGO_MAP[k]], lw=1.5,
                        label=ALGO_MAP[k]) for k in ALGO_ORDER]
style_handles = [
    Line2D([], [], color="black", ls="-",  lw=1.2, label="no diffusion (baseline)"),
    Line2D([], [], color="black", ls=":",  lw=1.4, label="with diffusion (ours)"),
]
fig.legend(handles=color_handles + style_handles,
           loc="lower center", ncol=6, frameon=False,
           bbox_to_anchor=(0.5, -0.02), columnspacing=1.2, handlelength=1.8)

fig.tight_layout(rect=(0, 0.08, 1, 1))

for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
    out_path = OUT_DIR / f"predprey_seed49.{ext}"
    fig.savefig(out_path, bbox_inches="tight", **kw)
    print(f"Saved {out_path}")
plt.close(fig)
