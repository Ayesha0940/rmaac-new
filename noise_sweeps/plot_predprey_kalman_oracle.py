"""
plot_predprey_kalman_oracle.py
================================
Predator-prey (simple_tag) Kalman hyperparameter-oracle results -- mirrors
plot_predprey.py's exact 4-row (Predator reward / Prey reward / Captures /
Survival) x 5-column (Delay / Actuation fault / Biased mu=-1 / mu=0 / mu=+1)
layout, but plots the Kalman per-point oracle (see
RMA-AC/experiments/run_predprey_kalman_oracle_sweep.py) instead of the DDPM
diffusion denoiser: solid = no-recovery baseline, dotted = best achievable
with Kalman filtering (each of the 4 metrics independently maximized across
the swept (q,r) grid at that exact point -- same convention plot_predprey.py
already uses for best_reward_pred_with_diffusion etc.).

First-pass scope is maddpg-none only (see ALGORITHMS in the sweep script),
so only one color line appears for now; the legend/loading code already
generalizes to more algorithms once they're added.

Reads:
    noise_sweeps/predprey_kalman_oracle/{delay,actuation_fault,noise,
        biased_noise_pos,biased_noise_neg}/seed*/simple_tag__*_oracle_sweep.csv

Writes:
    noise_sweeps/plots/predprey_kalman_oracle.{pdf,png}
"""
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

NOISE_DIR = Path(__file__).parent
ORACLE_DIR = NOISE_DIR / "predprey_kalman_oracle"
OUT_DIR = NOISE_DIR / "plots"
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

METRIC_COLS = {
    "pred_reward": ("pred_reward_baseline", "best_pred_reward"),
    "prey_reward": ("prey_reward_baseline", "best_prey_reward"),
    "captures":    ("captures_baseline",    "best_captures"),
    "survival":    ("survival_baseline",    "best_survival"),
}
ROWS = [
    ("Predator reward", "pred_reward"),
    ("Prey reward",      "prey_reward"),
    ("Captures",          "captures"),
    ("Survival",          "survival"),
]

# (column title, output subdir, oracle-sweep label, x_col, xlabel)
COLUMNS = [
    ("Delay ($k$ steps)",       "delay",            "delay",            "delay_k",          "$k$"),
    ("Actuation fault ($p$)",   "actuation_fault",  "stuck_at",         "sp_prob",          "$p_\\mathrm{stuck}$"),
    ("Biased noise $\\mu=-1$",  "biased_noise_neg", "biased_gauss_neg", "action_noise_std", r"$\alpha$"),
    ("Biased noise $\\mu=0$",   "noise",            "gauss",            "action_noise_std", r"$\alpha$"),
    ("Biased noise $\\mu=+1$",  "biased_noise_pos", "biased_gauss_pos", "action_noise_std", r"$\alpha$"),
]


def _load(subdir, label, x_col):
    regex = re.compile(r"^(simple_tag)__(.+?)_{}_oracle_sweep\.csv$".format(re.escape(label)))
    raw = defaultdict(list)
    for f in sorted((ORACLE_DIR / subdir).glob("seed*/*.csv")):
        m = regex.match(f.name)
        if not m or m.group(2) not in ALGO_MAP:
            continue
        raw[m.group(2)].append(pd.read_csv(f))
    return raw


def _agg(dfs, x_col):
    combined = pd.concat(dfs, ignore_index=True)
    y_cols = [c for pair in METRIC_COLS.values() for c in pair if c in combined.columns]
    return combined.groupby(x_col)[y_cols].agg(["mean", "std"]).reset_index()


DATA_BY_COL = []
for _title, _subdir, _label, _x_col, _xlabel in COLUMNS:
    raw = _load(_subdir, _label, _x_col)
    agg = {a: _agg(dfs, _x_col) for a, dfs in raw.items()}
    DATA_BY_COL.append(agg)

HAVE_DATA = any(DATA_BY_COL)


def plot_lines(ax, algo_dfs, x_col, baseline_col, best_col):
    for algo_key in ALGO_ORDER:
        if algo_key not in algo_dfs:
            continue
        df = algo_dfs[algo_key]
        color = COLORS[ALGO_MAP[algo_key]]
        x = df[x_col]
        for y_col, ls in [(baseline_col, "-"), (best_col, ":")]:
            if (y_col, "mean") not in df.columns:
                continue
            mean = df[(y_col, "mean")]
            std = df[(y_col, "std")].fillna(0)
            ax.plot(x, mean, color=color, linestyle=ls, linewidth=1.2)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)


if not HAVE_DATA or not any(d for d in DATA_BY_COL):
    print("No predator-prey Kalman oracle CSVs found under noise_sweeps/predprey_kalman_oracle/ "
          "-- run run_predprey_kalman_oracle_sweep.py first.")
else:
    n_rows = len(ROWS)
    n_cols = len(COLUMNS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.16, 4.5), sharex=False)
    fig.suptitle("Predator–Prey (simple_tag) — Kalman oracle", fontsize=8, fontweight="bold", y=1.005)

    for row_idx, (row_label, metric) in enumerate(ROWS):
        baseline_col, best_col = METRIC_COLS[metric]
        for col_idx, (col_title, _subdir, _label, x_col, xlabel) in enumerate(COLUMNS):
            ax = axes[row_idx, col_idx]
            ax.grid(True, alpha=0.3)
            ax.tick_params(length=2, pad=1.5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

            if row_idx == 0:
                ax.set_title(col_title, fontweight="bold", pad=3)
            if row_idx == n_rows - 1:
                ax.set_xlabel(xlabel)
            if col_idx == 0:
                ax.set_ylabel(row_label, rotation=0, ha="right", va="center", labelpad=8)

            algo_dfs = DATA_BY_COL[col_idx]
            plot_lines(ax, algo_dfs, x_col, baseline_col, best_col)

    present_algos = {k for d in DATA_BY_COL for k in d}
    color_handles = [Line2D([], [], color=COLORS[ALGO_MAP[k]], lw=1.5, label=ALGO_MAP[k])
                      for k in ALGO_ORDER if k in present_algos]
    style_handles = [
        Line2D([], [], color="black", ls="-", lw=1.2, label="no recovery (baseline)"),
        Line2D([], [], color="black", ls=":", lw=1.4, label="Kalman (oracle)"),
    ]
    fig.legend(handles=color_handles + style_handles,
               loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.015), columnspacing=1.2, handlelength=1.8)

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.subplots_adjust(wspace=0.35, hspace=0.18)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out_path = OUT_DIR / f"predprey_kalman_oracle.{ext}"
        fig.savefig(out_path, bbox_inches="tight", **kw)
        print(f"Saved {out_path}")
    plt.close(fig)
