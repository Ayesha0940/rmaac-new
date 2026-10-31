"""
plot_predprey_diffusion_and_kalman.py
=======================================
Predator-prey (simple_tag) combined comparison: DDPM diffusion denoiser vs.
Kalman filter, same design as plot_diffusion_and_filters_combined.py (the
4-scenario combined plot) but adapted to plot_predprey.py's 4-row (Predator
reward / Prey reward / Captures / Survival) x 5-column (Delay / Actuation
fault / Biased mu=-1 / mu=0 / mu=+1) predator-prey layout.

Three lines per panel, one shared no-recovery baseline:
    solid    = no recovery (baseline)   -- from the DDPM pipeline's own data
    dotted   = with diffusion (ours)    -- from the DDPM pipeline's own data
    long-dash= Kalman (oracle)          -- from run_predprey_kalman_oracle_sweep.py

Baseline and diffusion both come from the DDPM pipeline (same source/columns
plot_predprey.py itself uses) rather than mixing in the Kalman sweep's own
baseline -- avoids overlaying two separately-run baselines for the same
nominal condition, same lesson learned building
plot_diffusion_and_filters_combined.py. Kalman's own *_hparam columns are
loaded nowhere here (not plotted), matching that script's convention too.

Reads:
    DDPM data:
        noise_sweeps/delay_sweeps/seed*/predprey/*_delay_sweep.csv
        noise_sweeps/actuation_fault/seed*/predprey/*_stuck_at_sweep.csv
        noise_sweeps/guassian_noise/seed*/predprey/*_actstd_tstart_sweep.csv (mu=-1,0,1)
    Kalman oracle data:
        noise_sweeps/predprey_kalman_oracle/{delay,actuation_fault,noise,
            biased_noise_pos,biased_noise_neg}/seed*/*_oracle_sweep.csv

Writes:
    noise_sweeps/plots/predprey_diffusion_and_kalman.{pdf,png}
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

LINESTYLES = {
    "no_filter": (0, ()),            # matches plot_predprey.py's "-" baseline
    "diffusion": (0, (1, 1)),        # matches plot_predprey.py's ":" diffusion line
    "kalman":    (0, (6, 2, 1, 2)),  # coarse long-dash+dot, same as plot_diffusion_and_filters_combined.py
}
LINEWIDTHS = {"no_filter": 1.2, "diffusion": 1.2, "kalman": 1.3}
METHOD_LABEL = {
    "no_filter": "no recovery (baseline)",
    "diffusion": "with diffusion (ours)",
    "kalman":    "Kalman (oracle)",
}

# ── DDPM predator-prey data (same loading as plot_predprey.py) ───────────────
NOISE_RE = re.compile(r"^(simple_.+?)__(.+?)(?:_mu[-\d.]+)?_actstd_tstart_sweep\.csv$")
DELAY_RE = re.compile(r"^(simple_.+?)__(.+?)_delay_sweep\.csv$")
FAULT_RE = re.compile(r"^(simple_.+?)__(.+?)_stuck_at_sweep\.csv$")

_DF_COLS = [
    "reward_pred_noisy", "reward_prey_noisy",
    "captures_noisy", "survival_noisy",
    "captures_diff_t20", "captures_diff_t40",
    "survival_diff_t20", "survival_diff_t40",
    "best_reward_pred_with_diffusion", "best_reward_prey_with_diffusion",
]
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
    df["best_survival_with_diffusion"] = df[["survival_diff_t20", "survival_diff_t40"]].max(axis=1)
    return df.drop(columns=["captures_diff_t20", "captures_diff_t40",
                             "survival_diff_t20", "survival_diff_t40"])


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

_DDPM_METRICS = [
    "reward_pred_noisy", "reward_prey_noisy",
    "captures_noisy", "survival_noisy",
    "best_reward_pred_with_diffusion", "best_reward_prey_with_diffusion",
    "best_captures_with_diffusion", "best_survival_with_diffusion",
]
_DDPM_NOISE_METRICS = [
    "reward_pred_noise_no_diffusion", "reward_prey_noise_no_diffusion",
    "captures_noisy", "survival_noisy",
    "best_reward_pred_with_diffusion", "best_reward_prey_with_diffusion",
    "best_captures_with_diffusion", "best_survival_with_diffusion",
]


def _agg(dfs, x_col, y_cols):
    combined = pd.concat(dfs, ignore_index=True)
    return combined.groupby(x_col)[y_cols].agg(["mean", "std"]).reset_index()


delay_ddpm = {k: _agg(v, "delay_k", _DDPM_METRICS) for k, v in _delay_raw.items()}
fault_ddpm = {k: _agg(v, "sp_prob", _DDPM_METRICS) for k, v in _fault_raw.items()}

noise_ddpm = {}  # mu_val -> {algo_key: agg_df}
for mu_val in [-1, 0, 1]:
    mu_map = {}
    for algo_key, dfs in _noise_raw.items():
        combined = pd.concat(dfs, ignore_index=True)
        subset = combined[combined["noise_mu"] == mu_val]
        if subset.empty:
            continue
        mu_map[algo_key] = subset.groupby("action_noise_std")[_DDPM_NOISE_METRICS].agg(
            ["mean", "std"]).reset_index()
    if mu_map:
        noise_ddpm[mu_val] = mu_map

# ── Kalman oracle data (same loading as plot_predprey_kalman_oracle.py) ──────
_KALMAN_METRIC_COLS = {
    "pred_reward": "best_pred_reward",
    "prey_reward": "best_prey_reward",
    "captures":    "best_captures",
    "survival":    "best_survival",
}


def _load_kalman(subdir, label, x_col):
    regex = re.compile(r"^(simple_tag)__(.+?)_{}_oracle_sweep\.csv$".format(re.escape(label)))
    raw = defaultdict(list)
    for f in sorted((ORACLE_DIR / subdir).glob("seed*/*.csv")):
        m = regex.match(f.name)
        if not m or m.group(2) not in ALGO_MAP:
            continue
        raw[m.group(2)].append(pd.read_csv(f))
    return raw


def _agg_kalman(dfs, x_col):
    combined = pd.concat(dfs, ignore_index=True)
    y_cols = [c for c in _KALMAN_METRIC_COLS.values() if c in combined.columns]
    return combined.groupby(x_col)[y_cols].agg(["mean", "std"]).reset_index()


delay_kalman = {a: _agg_kalman(dfs, "delay_k") for a, dfs in _load_kalman("delay", "delay", "delay_k").items()}
fault_kalman = {a: _agg_kalman(dfs, "sp_prob") for a, dfs in _load_kalman("actuation_fault", "stuck_at", "sp_prob").items()}
noise_neg_kalman = {a: _agg_kalman(dfs, "action_noise_std")
                     for a, dfs in _load_kalman("biased_noise_neg", "biased_gauss_neg", "action_noise_std").items()}
noise_0_kalman = {a: _agg_kalman(dfs, "action_noise_std")
                   for a, dfs in _load_kalman("noise", "gauss", "action_noise_std").items()}
noise_pos_kalman = {a: _agg_kalman(dfs, "action_noise_std")
                     for a, dfs in _load_kalman("biased_noise_pos", "biased_gauss_pos", "action_noise_std").items()}

# ── row/column specs ──────────────────────────────────────────────────────────
# (row label, ddpm baseline col (delay/fault), ddpm baseline col (noise), ddpm diffusion col, kalman metric key)
ROWS = [
    ("Predator reward", "reward_pred_noisy", "reward_pred_noise_no_diffusion",
     "best_reward_pred_with_diffusion", "pred_reward"),
    ("Prey reward", "reward_prey_noisy", "reward_prey_noise_no_diffusion",
     "best_reward_prey_with_diffusion", "prey_reward"),
    ("Captures", "captures_noisy", "captures_noisy",
     "best_captures_with_diffusion", "captures"),
    ("Survival", "survival_noisy", "survival_noisy",
     "best_survival_with_diffusion", "survival"),
]

# (col title, ddpm_algo_dfs, kalman_algo_dfs, x_col, xlabel, is_noise)
COLUMNS = [
    ("Delay ($k$ steps)",       delay_ddpm,               delay_kalman,      "delay_k",          "$k$",                    False),
    ("Actuation fault ($p$)",   fault_ddpm,               fault_kalman,      "sp_prob",          "$p_\\mathrm{stuck}$",    False),
    ("Biased noise $\\mu=-1$",  noise_ddpm.get(-1, {}),   noise_neg_kalman,  "action_noise_std", r"$\alpha$",              True),
    ("Biased noise $\\mu=0$",   noise_ddpm.get(0, {}),    noise_0_kalman,    "action_noise_std", r"$\alpha$",              True),
    ("Biased noise $\\mu=+1$",  noise_ddpm.get(1, {}),    noise_pos_kalman,  "action_noise_std", r"$\alpha$",              True),
]


def plot_lines(ax, ddpm_algo_dfs, kalman_algo_dfs, x_col, y_nd_df, y_nd_noise, y_wd, kalman_key, is_noise):
    y_nd = y_nd_noise if is_noise else y_nd_df
    for algo_key in ALGO_ORDER:
        color = COLORS[ALGO_MAP[algo_key]]

        if algo_key in ddpm_algo_dfs:
            df = ddpm_algo_dfs[algo_key]
            x = df[x_col]
            for method, y_col in [("no_filter", y_nd), ("diffusion", y_wd)]:
                if (y_col, "mean") not in df.columns:
                    continue
                mean = df[(y_col, "mean")]
                std = df[(y_col, "std")].fillna(0)
                ax.plot(x, mean, color=color, linestyle=LINESTYLES[method], linewidth=LINEWIDTHS[method])
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10, linewidth=0)

        if algo_key in kalman_algo_dfs:
            df = kalman_algo_dfs[algo_key]
            x = df[x_col]
            y_col = _KALMAN_METRIC_COLS[kalman_key]
            if (y_col, "mean") in df.columns:
                mean = df[(y_col, "mean")]
                std = df[(y_col, "std")].fillna(0)
                ax.plot(x, mean, color=color, linestyle=LINESTYLES["kalman"], linewidth=LINEWIDTHS["kalman"])
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10, linewidth=0)


n_rows = len(ROWS)
n_cols = len(COLUMNS)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.16, 4.5), sharex=False)
fig.suptitle("Predator–Prey (simple_tag) — diffusion vs. Kalman", fontsize=8, fontweight="bold", y=1.01)

for row_idx, (row_label, y_nd_df, y_nd_noise, y_wd, kalman_key) in enumerate(ROWS):
    for col_idx, (col_title, ddpm_data, kalman_data, x_col, xlabel, is_noise) in enumerate(COLUMNS):
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

        plot_lines(ax, ddpm_data, kalman_data, x_col, y_nd_df, y_nd_noise, y_wd, kalman_key, is_noise)

present_algos = {k for _, d, kd, *_ in COLUMNS for k in (set(d) | set(kd))}
color_handles = [Line2D([], [], color=COLORS[ALGO_MAP[k]], lw=1.5, label=ALGO_MAP[k])
                  for k in ALGO_ORDER if k in present_algos]
style_handles = [Line2D([], [], color="black", ls=LINESTYLES[m], lw=1.3, label=METHOD_LABEL[m])
                  for m in ["no_filter", "diffusion", "kalman"]]
fig.legend(handles=color_handles + style_handles,
           loc="lower center", ncol=len(color_handles) + len(style_handles), frameon=False,
           bbox_to_anchor=(0.5, -0.02), columnspacing=1.2, handlelength=2.0)

fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.subplots_adjust(wspace=0.35, hspace=0.18)

for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
    out_path = OUT_DIR / f"predprey_diffusion_and_kalman.{ext}"
    fig.savefig(out_path, bbox_inches="tight", **kw)
    print(f"Saved {out_path}")
plt.close(fig)
