import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

NOISE_DIR = Path(__file__).parent
OUT_DIR   = NOISE_DIR / "plots"
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

NOISE_RE = re.compile(r"^(simple_.+?)__(.+?)(?:_mu[-\d.]+)?_actstd_tstart_sweep\.csv$")
DELAY_RE = re.compile(r"^(simple_.+?)__(.+?)_delay_sweep\.csv$")
FAULT_RE = re.compile(r"^(simple_.+?)__(.+?)_stuck_at_sweep\.csv$")

# Columns to keep from delay/fault CSVs (t20/t40 intermediates dropped after computing best)
_DF_COLS = [
    "reward_pred_noisy", "reward_prey_noisy",
    "captures_noisy", "survival_noisy",
    "captures_diff_t20", "captures_diff_t40",
    "survival_diff_t20", "survival_diff_t40",
    "best_reward_pred_with_diffusion", "best_reward_prey_with_diffusion",
]
# Columns to keep from noise CSVs
_NOISE_COLS = [
    "noise_mu", "action_noise_std",
    "reward_pred_noise_no_diffusion", "reward_prey_noise_no_diffusion",
    "captures_noisy", "survival_noisy",
    "captures_diff_t20", "captures_diff_t40",
    "survival_diff_t20", "survival_diff_t40",
    "best_reward_pred_with_diffusion", "best_reward_prey_with_diffusion",
]


def _add_best_cap_surv(df):
    df["best_captures_with_diffusion"] = df[["captures_diff_t20", "captures_diff_t40"]].max(axis=1)
    df["best_survival_with_diffusion"]  = df[["survival_diff_t20",  "survival_diff_t40"]].max(axis=1)
    return df.drop(columns=["captures_diff_t20", "captures_diff_t40",
                             "survival_diff_t20",  "survival_diff_t40"])


# ── load all seeds ────────────────────────────────────────────────────────────

_noise_raw = defaultdict(list)
for f in sorted((NOISE_DIR / "guassian_noise").glob("*/predprey/*.csv")):
    m = NOISE_RE.match(f.name)
    if not m or m.group(2) not in ALGO_MAP:
        continue
    df = pd.read_csv(f)[_NOISE_COLS]
    df["noise_mu"] = df["noise_mu"].round().astype(int)
    _noise_raw[m.group(2)].append(_add_best_cap_surv(df))

_delay_raw = defaultdict(list)
for f in sorted((NOISE_DIR / "delay_sweeps").glob("*/predprey/*.csv")):
    m = DELAY_RE.match(f.name)
    if not m or m.group(2) not in ALGO_MAP:
        continue
    df = pd.read_csv(f)[["delay_k"] + _DF_COLS]
    _delay_raw[m.group(2)].append(_add_best_cap_surv(df))

_fault_raw = defaultdict(list)
for f in sorted((NOISE_DIR / "actuation_fault").glob("*/predprey/*.csv")):
    m = FAULT_RE.match(f.name)
    if not m or m.group(2) not in ALGO_MAP:
        continue
    df = pd.read_csv(f)[["sp_prob"] + _DF_COLS]
    _fault_raw[m.group(2)].append(_add_best_cap_surv(df))

# ── aggregate across seeds ────────────────────────────────────────────────────

_METRICS = [
    "reward_pred_noisy", "reward_prey_noisy",
    "captures_noisy", "survival_noisy",
    "best_reward_pred_with_diffusion", "best_reward_prey_with_diffusion",
    "best_captures_with_diffusion", "best_survival_with_diffusion",
]
_NOISE_METRICS = [
    "reward_pred_noise_no_diffusion", "reward_prey_noise_no_diffusion",
    "captures_noisy", "survival_noisy",
    "best_reward_pred_with_diffusion", "best_reward_prey_with_diffusion",
    "best_captures_with_diffusion", "best_survival_with_diffusion",
]


def _agg(dfs, x_col, y_cols):
    combined = pd.concat(dfs, ignore_index=True)
    return combined.groupby(x_col)[y_cols].agg(["mean", "std"]).reset_index()


delay_agg = {k: _agg(v, "delay_k", _METRICS) for k, v in _delay_raw.items()}
fault_agg = {k: _agg(v, "sp_prob", _METRICS) for k, v in _fault_raw.items()}

noise_agg = {}   # mu_val -> {algo_key: agg_df}
for mu_val in [-1, 0, 1]:
    mu_map = {}
    for algo_key, dfs in _noise_raw.items():
        combined = pd.concat(dfs, ignore_index=True)
        subset = combined[combined["noise_mu"] == mu_val]
        if subset.empty:
            continue
        agg = subset.groupby("action_noise_std")[_NOISE_METRICS].agg(["mean", "std"]).reset_index()
        mu_map[algo_key] = agg
    if mu_map:
        noise_agg[mu_val] = mu_map

# ── row/column specs ──────────────────────────────────────────────────────────

# (row label, y_nd for delay/fault, y_nd for noise, y_wd shared)
ROWS = [
    ("Predator reward", "reward_pred_noisy",           "reward_pred_noise_no_diffusion", "best_reward_pred_with_diffusion"),
    ("Prey reward",     "reward_prey_noisy",           "reward_prey_noise_no_diffusion", "best_reward_prey_with_diffusion"),
    ("Captures",        "captures_noisy",              "captures_noisy",                 "best_captures_with_diffusion"),
    ("Survival",        "survival_noisy",              "survival_noisy",                 "best_survival_with_diffusion"),
]

# (col title, algo_dfs dict, x_col, xlabel, is_noise)
COLUMNS = [
    ("Delay ($k$ steps)",      delay_agg,          "delay_k",          "$k$",                       False),
    ("Actuation fault ($p$)",  fault_agg,          "sp_prob",          "$p_\\mathrm{stuck}$",        False),
    ("Biased noise $\\mu=-1$", noise_agg.get(-1, {}), "action_noise_std", r"$\alpha$",               True),
    ("Biased noise $\\mu=0$",  noise_agg.get(0,  {}), "action_noise_std", r"$\alpha$",               True),
    ("Biased noise $\\mu=+1$", noise_agg.get(1,  {}), "action_noise_std", r"$\alpha$",               True),
]

# ── plot helper ───────────────────────────────────────────────────────────────

def plot_lines(ax, algo_dfs, x_col, y_nd_col, y_wd_col):
    for algo_key in ALGO_ORDER:
        if algo_key not in algo_dfs:
            continue
        df = algo_dfs[algo_key]
        color = COLORS[ALGO_MAP[algo_key]]
        x = df[x_col]
        for y_col, ls in [(y_nd_col, "-"), (y_wd_col, ":")]:
            mean = df[(y_col, "mean")]
            std  = df[(y_col, "std")].fillna(0)
            ax.plot(x, mean, color=color, linestyle=ls, linewidth=1.0)
            ax.fill_between(x, mean - std, mean + std,
                            color=color, alpha=0.12, linewidth=0)

# ── figure ────────────────────────────────────────────────────────────────────

n_rows = len(ROWS)
fig, axes = plt.subplots(n_rows, 5, figsize=(7.16, 4.5), sharex=False)
fig.suptitle("Predator–Prey (simple_tag)", fontsize=8, fontweight="bold", y=1.005)

for row_idx, (row_label, y_nd_df, y_nd_noise, y_wd) in enumerate(ROWS):
    for col_idx, (col_title, algo_dfs, x_col, xlabel, is_noise) in enumerate(COLUMNS):
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

        y_nd = y_nd_noise if is_noise else y_nd_df
        plot_lines(ax, algo_dfs, x_col, y_nd, y_wd)

# ── compact legend ────────────────────────────────────────────────────────────

color_handles = [Line2D([], [], color=COLORS[ALGO_MAP[k]], lw=1.5,
                        label=ALGO_MAP[k]) for k in ALGO_ORDER]
style_handles = [
    Line2D([], [], color="black", ls="-",  lw=1.2, label="no diffusion (baseline)"),
    Line2D([], [], color="black", ls=":",  lw=1.4, label="with diffusion (ours)"),
]
fig.legend(handles=color_handles + style_handles,
           loc="lower center", ncol=6, frameon=False,
           bbox_to_anchor=(0.5, -0.015), columnspacing=1.2, handlelength=1.8)

fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.subplots_adjust(wspace=0.35, hspace=0.18)

for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
    out_path = OUT_DIR / f"predprey_all_metrics.{ext}"
    fig.savefig(out_path, bbox_inches="tight", **kw)
    print(f"Saved {out_path}")
plt.close(fig)
