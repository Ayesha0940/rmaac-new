"""
plot_filter_combined.py
========================
Combined robustness grid for the classical action-recovery filters (EMA,
median, Kalman — see RMA-AC/experiments/action_filters.py /
filter_eval.py), structurally mirroring plot_combined.py (which does the
same thing for the DDPM diffusion denoiser) but with one line per filter
method instead of a single "with diffusion" line.

Reads:
    noise_sweeps/filters/delay/seed*/*_delay_filter_sweep.csv
    noise_sweeps/filters/actuation_fault/seed*/*_stuck_at_filter_sweep.csv
    noise_sweeps/filters/noise/seed*/*_mu0.0_actstd_filter_sweep.csv
    noise_sweeps/filters/biased_noise_pos/seed*/*_mu1.0_actstd_filter_sweep.csv
    noise_sweeps/filters/biased_noise_neg/seed*/*_mu-1.0_actstd_filter_sweep.csv

Writes:
    noise_sweeps/plots/filters_all_scenarios.{pdf,png}
    noise_sweeps/plots/aggregated_filters_{delay,fault,noise,biased_pos,biased_neg}.csv
"""
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

FIG_W, FIG_H = 9.5, 5.6
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

NOISE_DIR = Path(__file__).parent
OUT_DIR = NOISE_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALGO_MAP = {
    "earniebest": "ERNIE",
    "m3ddpgbest": "M3DDPG",
    "maddpgbest": "MADDPG",
    "rmaacbest": "RMAAC",
}
COLORS = {
    "ERNIE":  "#0072B2",
    "M3DDPG": "#E69F00",
    "MADDPG": "#009E73",
    "RMAAC":  "#D55E00",
}
ALGO_ORDER = ["earniebest", "m3ddpgbest", "maddpgbest", "rmaacbest"]

SCENARIO_LABEL = {
    "simple_adversary":        "Physical\nDeception",
    "simple_push":             "Keep-away",
    "simple_speaker_listener": "Cooperative\nCommunication",
    "simple_spread":           "Cooperative\nNavigation",
    "simple_tag":              "Predator–Prey",
}

FILTER_METHODS = ["no_filter", "ema", "median", "kalman"]
LINESTYLES = {"no_filter": "-", "ema": "--", "median": ":", "kalman": "-."}
METHOD_LABEL = {"no_filter": "no filter (baseline)", "ema": "EMA", "median": "median", "kalman": "Kalman"}

REWARD_COLS = ["reward_" + m for m in FILTER_METHODS]

COLUMNS = [
    ("Delay ($k$ steps)",        "delay_k",          "$k$"),
    ("Actuation fault ($p$)",    "sp_prob",          "$p_\\mathrm{stuck}$"),
    ("Action noise ($\\sigma$)", "action_noise_std", r"$\sigma$"),
    ("Biased noise $\\mu=+1$",   "action_noise_std", r"$\sigma$"),
    ("Biased noise $\\mu=-1$",   "action_noise_std", r"$\sigma$"),
]


def _load(glob_pattern, regex, x_col):
    raw = defaultdict(lambda: defaultdict(list))
    for f in sorted(NOISE_DIR.glob(glob_pattern)):
        m = regex.match(f.name)
        if not m or m.group(2) not in ALGO_MAP:
            continue
        df = pd.read_csv(f)
        cols = [x_col] + [c for c in REWARD_COLS if c in df.columns]
        raw[m.group(1)][m.group(2)].append(df[cols])
    return raw


def _agg(dfs, x_col):
    combined = pd.concat(dfs, ignore_index=True)
    y_cols = [c for c in REWARD_COLS if c in combined.columns]
    return combined.groupby(x_col)[y_cols].agg(["mean", "std"]).reset_index()


DELAY_RE = re.compile(r"^(simple_.+?)__(.+?)_delay_filter_sweep\.csv$")
FAULT_RE = re.compile(r"^(simple_.+?)__(.+?)_stuck_at_filter_sweep\.csv$")
NOISE_RE = re.compile(r"^(simple_.+?)__(.+?)_mu0\.0_actstd_filter_sweep\.csv$")
BIASED_POS_RE = re.compile(r"^(simple_.+?)__(.+?)_mu1\.0_actstd_filter_sweep\.csv$")
BIASED_NEG_RE = re.compile(r"^(simple_.+?)__(.+?)_mu-1\.0_actstd_filter_sweep\.csv$")

_delay_raw = _load("filters/delay/*/*.csv", DELAY_RE, "delay_k")
_fault_raw = _load("filters/actuation_fault/*/*.csv", FAULT_RE, "sp_prob")
_noise_raw = _load("filters/noise/*/*.csv", NOISE_RE, "action_noise_std")
_biased_pos_raw = _load("filters/biased_noise_pos/*/*.csv", BIASED_POS_RE, "action_noise_std")
_biased_neg_raw = _load("filters/biased_noise_neg/*/*.csv", BIASED_NEG_RE, "action_noise_std")

delay_data = {s: {a: _agg(dfs, "delay_k") for a, dfs in m.items()} for s, m in _delay_raw.items()}
fault_data = {s: {a: _agg(dfs, "sp_prob") for a, dfs in m.items()} for s, m in _fault_raw.items()}
noise_data = {s: {a: _agg(dfs, "action_noise_std") for a, dfs in m.items()} for s, m in _noise_raw.items()}
biased_pos_data = {s: {a: _agg(dfs, "action_noise_std") for a, dfs in m.items()} for s, m in _biased_pos_raw.items()}
biased_neg_data = {s: {a: _agg(dfs, "action_noise_std") for a, dfs in m.items()} for s, m in _biased_neg_raw.items()}

SCENARIOS = sorted(set(delay_data) | set(fault_data) | set(noise_data)
                    | set(biased_pos_data) | set(biased_neg_data))


def _dump_agg(data_dict, out_path):
    rows = []
    for scenario, algo_map in data_dict.items():
        for algo_key, df in algo_map.items():
            flat = df.copy()
            flat.columns = [c if isinstance(c, str) else "_".join(p for p in c if p) for c in flat.columns]
            base_cols = {c[:-5] for c in flat.columns if c.endswith("_mean")}
            for base in sorted(base_cols):
                mean_col, std_col = f"{base}_mean", f"{base}_std"
                flat[base] = flat.apply(lambda r, m=mean_col, s=std_col: f"{r[m]:.2f}±{r[s]:.2f}", axis=1)
                flat.drop(columns=[mean_col, std_col], inplace=True)
            flat.insert(0, "algo", ALGO_MAP[algo_key])
            flat.insert(0, "scenario", scenario)
            rows.append(flat)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out_path, index=False)
        print(f"Saved {out_path}")


_dump_agg(delay_data, OUT_DIR / "aggregated_filters_delay.csv")
_dump_agg(fault_data, OUT_DIR / "aggregated_filters_fault.csv")
_dump_agg(noise_data, OUT_DIR / "aggregated_filters_noise.csv")
_dump_agg(biased_pos_data, OUT_DIR / "aggregated_filters_biased_pos.csv")
_dump_agg(biased_neg_data, OUT_DIR / "aggregated_filters_biased_neg.csv")


def plot_lines(ax, algo_dfs, x_col):
    for algo_key in ALGO_ORDER:
        if algo_key not in algo_dfs:
            continue
        df = algo_dfs[algo_key]
        color = COLORS[ALGO_MAP[algo_key]]
        x = df[x_col]
        for method in FILTER_METHODS:
            y_col = "reward_" + method
            if (y_col, "mean") not in df.columns:
                continue
            mean = df[(y_col, "mean")]
            std = df[(y_col, "std")].fillna(0)
            ax.plot(x, mean, color=color, linestyle=LINESTYLES[method], linewidth=1.0)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10, linewidth=0)


if not SCENARIOS:
    print("No filter sweep CSVs found under noise_sweeps/filters/ — run the "
          "run_*_sweep_filters.sh scripts first.")
else:
    n_rows = len(SCENARIOS)
    n_cols = len(COLUMNS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_W, FIG_H * n_rows / 5.0), sharex="col", squeeze=False)

    DATA_BY_COL = [delay_data, fault_data, noise_data, biased_pos_data, biased_neg_data]

    for row_idx, scenario in enumerate(SCENARIOS):
        for col_idx, (col_title, x_col, xlabel) in enumerate(COLUMNS):
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
                ax.set_ylabel(SCENARIO_LABEL.get(scenario, scenario),
                              rotation=0, ha="right", va="center", labelpad=8)

            algo_dfs = DATA_BY_COL[col_idx].get(scenario, {})
            plot_lines(ax, algo_dfs, x_col)

    color_handles = [Line2D([], [], color=COLORS[ALGO_MAP[k]], lw=1.5, label=ALGO_MAP[k]) for k in ALGO_ORDER]
    style_handles = [Line2D([], [], color="black", ls=LINESTYLES[m], lw=1.2, label=METHOD_LABEL[m])
                      for m in FILTER_METHODS]
    fig.legend(handles=color_handles + style_handles,
               loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.2, handlelength=1.8)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.subplots_adjust(wspace=0.30, hspace=0.18)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out_path = OUT_DIR / f"filters_all_scenarios.{ext}"
        fig.savefig(out_path, bbox_inches="tight", **kw)
        print(f"Saved {out_path}")
    plt.close(fig)
