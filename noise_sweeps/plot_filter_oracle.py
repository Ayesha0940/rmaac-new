"""
plot_filter_oracle.py
======================
Per-point hyperparameter-oracle sweep results (see
RMA-AC/experiments/run_hparam_oracle_sweep.py) for the EMA/median/Kalman
action filters: at every (scenario, attack type, severity) point, each
filter's coarse hyperparameter grid was searched independently and only the
best-achievable ("oracle") reward at that exact point was kept.

This plot ONLY reads/draws the already-maximized reward_no_filter /
reward_ema / reward_median / reward_kalman columns from each CSV -- the
*_best_hparam columns (which hyperparameter won at each point) are metadata,
not plotted here.

Reads:
    noise_sweeps/filters_oracle/{delay,actuation_fault,noise,
        biased_noise_pos,biased_noise_neg}/seed*/*_oracle_sweep.csv

Writes:
    noise_sweeps/plots/filter_oracle_all_scenarios.{pdf,png}
    noise_sweeps/plots/aggregated_oracle_{delay,stuck_at,gauss,biased_gauss_pos,biased_gauss_neg}.csv
"""
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

FIG_W = 8.5
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
ORACLE_DIR = NOISE_DIR / "filters_oracle"
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
}

FILTER_METHODS = ["no_filter", "ema", "median", "kalman"]
LINESTYLES = {"no_filter": "-", "ema": "--", "median": ":", "kalman": "-."}
METHOD_LABEL = {"no_filter": "no filter (baseline)", "ema": "EMA (oracle)",
                 "median": "median (oracle)", "kalman": "Kalman (oracle)"}
REWARD_COLS = ["reward_" + m for m in FILTER_METHODS]

# (column title, output subdir, x_col, oracle-sweep label suffix, xlabel)
COLUMNS = [
    ("Delay ($k$ steps)",        "delay",            "delay_k",          "delay",            "$k$"),
    ("Actuation fault ($p$)",    "actuation_fault",  "sp_prob",          "stuck_at",         "$p_\\mathrm{stuck}$"),
    ("Action noise ($\\sigma$)", "noise",            "action_noise_std", "gauss",            r"$\sigma$"),
    ("Biased noise $\\mu=+1$",   "biased_noise_pos", "action_noise_std", "biased_gauss_pos", r"$\sigma$"),
    ("Biased noise $\\mu=-1$",   "biased_noise_neg", "action_noise_std", "biased_gauss_neg", r"$\sigma$"),
]


def _load(subdir, label, x_col):
    regex = re.compile(r"^(simple_.+?)__(.+?)_{}_oracle_sweep\.csv$".format(re.escape(label)))
    raw = defaultdict(lambda: defaultdict(list))
    for f in sorted((ORACLE_DIR / subdir).glob("seed*/*.csv")):
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


DATA_BY_COL = []
for _title, _subdir, _x_col, _label, _xlabel in COLUMNS:
    raw = _load(_subdir, _label, _x_col)
    agg = {s: {a: _agg(dfs, _x_col) for a, dfs in m.items()} for s, m in raw.items()}
    DATA_BY_COL.append(agg)

SCENARIOS = sorted(set().union(*[set(d) for d in DATA_BY_COL]))


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


for (_, _, _, label, _), data in zip(COLUMNS, DATA_BY_COL):
    _dump_agg(data, OUT_DIR / f"aggregated_oracle_{label}.csv")


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
    print("No oracle sweep CSVs found under noise_sweeps/filters_oracle/ — run "
          "run_hparam_oracle_sweep.py first.")
else:
    n_rows = len(SCENARIOS)
    n_cols = len(COLUMNS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_W, 1.15 * n_rows + 0.6),
                              sharex="col", squeeze=False)

    for row_idx, scenario in enumerate(SCENARIOS):
        for col_idx, (col_title, _subdir, x_col, _label, xlabel) in enumerate(COLUMNS):
            ax = axes[row_idx, col_idx]
            ax.grid(True, alpha=0.3)
            ax.tick_params(length=2, pad=1.5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

            if row_idx == 0:
                ax.set_title(col_title, fontweight="bold", pad=3)
            if row_idx == n_rows - 1:
                ax.set_xlabel(xlabel)
            if col_idx == 0:
                ax.set_ylabel(SCENARIO_LABEL.get(scenario, scenario),
                              rotation=0, ha="right", va="center", labelpad=8)

            algo_dfs = DATA_BY_COL[col_idx].get(scenario, {})
            plot_lines(ax, algo_dfs, x_col)

    present_algos = {k for d in DATA_BY_COL for algo_map in d.values() for k in algo_map}
    color_handles = [Line2D([], [], color=COLORS[ALGO_MAP[k]], lw=1.5, label=ALGO_MAP[k])
                      for k in ALGO_ORDER if k in present_algos]
    style_handles = [Line2D([], [], color="black", ls=LINESTYLES[m], lw=1.2, label=METHOD_LABEL[m])
                      for m in FILTER_METHODS]
    fig.legend(handles=color_handles + style_handles,
               loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.05), columnspacing=1.2, handlelength=1.8)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.subplots_adjust(wspace=0.30, hspace=0.18)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out_path = OUT_DIR / f"filter_oracle_all_scenarios.{ext}"
        fig.savefig(out_path, bbox_inches="tight", **kw)
        print(f"Saved {out_path}")
    plt.close(fig)
