"""
Publication-ready version of the 5x5 robustness grid.

Key changes vs. the original (all aimed at print legibility in a two-column
IEEE `figure*`):
  * Figure authored at final print size (~\textwidth) with final font sizes,
    so nothing is scaled down 3x at include time.
  * Vector PDF output (+ a 300-dpi PNG for previews). Lines stay crisp.
  * sharex='col' + MaxNLocator: 5 sets of x-ticks instead of 25.
  * Okabe-Ito colorblind-safe palette (no red/green clash).
  * Compact 6-entry legend: colour = algorithm, linestyle = diffusion on/off.
  * Rows relabelled with the environment names used in the paper; noise
    magnitude shown as alpha to match the text.
Include in LaTeX as:
    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{images/all_scenarios.pdf}
      \caption{...}
    \end{figure*}
"""
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

# ── print / style setup ───────────────────────────────────────────────────────
# Full text width of a two-column IEEE paper is ~7.16in. Author at that width;
# do NOT shrink at \includegraphics time.
FIG_W, FIG_H = 7.16, 5.6   # flatter panels -> ~1/3 less height; ratio is what LaTeX preserves
plt.rcParams.update({
    "font.family": "serif",          # matches IEEE Times body text
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
    "pdf.fonttype": 42,              # editable/embeddable fonts in the PDF
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
# Okabe-Ito colourblind-safe palette
COLORS = {
    "ERNIE":  "#0072B2",   # blue
    "M3DDPG": "#E69F00",   # orange
    "MADDPG": "#009E73",   # bluish green
    "RMAAC":  "#D55E00",   # vermillion
}
ALGO_ORDER = ["earniebest", "m3ddpgbest", "maddpgbest", "rmaacbest"]

# Scenario -> label used in the paper body (relabel rows so readers don't have
# to translate simple_* names).
SCENARIO_LABEL = {
    "simple_adversary":        "Physical\nDeception",
    "simple_push":             "Keep-away",
    "simple_speaker_listener": "Cooperative\nCommunication",
    "simple_spread":           "Cooperative\nNavigation",
    "simple_tag":              "Predator\u2013Prey",
}

# (column title, x column, y-no-diffusion column, y-with-diffusion column, xlabel)
COLUMNS = [
    ("Delay ($k$ steps)",       "delay_k",          "reward_noisy",              "best_reward_with_diffusion", "$k$"),
    ("Actuation fault ($p$)",   "sp_prob",          "reward_noisy",              "best_reward_with_diffusion", "$p_\\mathrm{stuck}$"),
    ("Biased noise $\\mu=-1$",  "action_noise_std", "reward_noise_no_diffusion", "best_reward_with_diffusion", r"$\alpha$"),
    ("Biased noise $\\mu=0$",   "action_noise_std", "reward_noise_no_diffusion", "best_reward_with_diffusion", r"$\alpha$"),
    ("Biased noise $\\mu=+1$",  "action_noise_std", "reward_noise_no_diffusion", "best_reward_with_diffusion", r"$\alpha$"),
]
MU_BY_COL = [None, None, -1, 0, 1]

# ── load gaussian noise sweep data (all seeds) ───────────────────────────────
NOISE_RE = re.compile(r"^(simple_.+?)__(.+?)(?:_mu[-\d.]+)?_actstd_tstart_sweep\.csv$")
_noise_raw = defaultdict(lambda: defaultdict(list))  # [scenario][algo] -> list[df]
for f in sorted((NOISE_DIR / "guassian_noise").glob("*/*.csv")):
    m = NOISE_RE.match(f.name)
    if not m or m.group(2) not in ALGO_MAP:
        continue
    df = pd.read_csv(f)[["noise_mu", "action_noise_std",
                          "reward_noise_no_diffusion", "best_reward_with_diffusion"]] \
         if "noise_mu" in pd.read_csv(f, nrows=0).columns \
         else pd.read_csv(f)[["action_noise_std",
                               "reward_noise_no_diffusion", "best_reward_with_diffusion"]]
    if "noise_mu" not in df.columns:
        df.insert(0, "noise_mu", 0)
    df["noise_mu"] = df["noise_mu"].round().astype(int)
    _noise_raw[m.group(1)][m.group(2)].append(df)

# ── load delay sweep data (all seeds) ────────────────────────────────────────
DELAY_RE = re.compile(r"^(simple_.+?)__(.+?)_delay_sweep\.csv$")
_delay_raw = defaultdict(lambda: defaultdict(list))
for f in sorted((NOISE_DIR / "delay_sweeps").glob("*/*.csv")):
    m = DELAY_RE.match(f.name)
    if not m or m.group(2) not in ALGO_MAP:
        continue
    _delay_raw[m.group(1)][m.group(2)].append(
        pd.read_csv(f)[["delay_k", "reward_noisy", "best_reward_with_diffusion"]])

# ── load actuation fault data (all seeds) ────────────────────────────────────
FAULT_RE = re.compile(r"^(simple_.+?)__(.+?)_stuck_at_sweep\.csv$")
_fault_raw = defaultdict(lambda: defaultdict(list))
for f in sorted((NOISE_DIR / "actuation_fault").glob("*/*.csv")):
    m = FAULT_RE.match(f.name)
    if not m or m.group(2) not in ALGO_MAP:
        continue
    _fault_raw[m.group(1)][m.group(2)].append(
        pd.read_csv(f)[["sp_prob", "reward_noisy", "best_reward_with_diffusion"]])


def _agg(dfs, x_col, y_cols):
    """Concatenate seed DataFrames and return mean±std per x value."""
    combined = pd.concat(dfs, ignore_index=True)
    agg = combined.groupby(x_col)[y_cols].agg(["mean", "std"]).reset_index()
    return agg


# ── aggregate delay and actuation fault ──────────────────────────────────────
delay_data = {
    scenario: {algo: _agg(dfs, "delay_k", ["reward_noisy", "best_reward_with_diffusion"])
               for algo, dfs in algo_map.items()}
    for scenario, algo_map in _delay_raw.items()
}
fault_data = {
    scenario: {algo: _agg(dfs, "sp_prob", ["reward_noisy", "best_reward_with_diffusion"])
               for algo, dfs in algo_map.items()}
    for scenario, algo_map in _fault_raw.items()
}

# ── build merged noise dicts per (scenario, mu): aggregate across seeds ───────
noise_merged = {}
for scenario, algo_map in _noise_raw.items():
    for mu_val in [-1, 0, 1]:
        mu_dfs = {}
        for algo_key, dfs in algo_map.items():
            combined = pd.concat(dfs, ignore_index=True)
            subset = combined[combined["noise_mu"] == mu_val]
            if subset.empty:
                continue
            agg = subset.groupby("action_noise_std")[
                ["reward_noise_no_diffusion", "best_reward_with_diffusion"]
            ].agg(["mean", "std"]).reset_index()
            mu_dfs[algo_key] = agg
        if mu_dfs:
            noise_merged[(scenario, mu_val)] = mu_dfs

SCENARIOS = sorted(set(_noise_raw) | set(delay_data) | set(fault_data))


def _dump_agg(data_dict, out_path):
    rows = []
    for key, algo_map in data_dict.items():
        if isinstance(key, tuple):      # noise_merged: (scenario, mu)
            scenario, mu = key
        else:                           # delay/fault: scenario string
            scenario, mu = key, None
        for algo_key, df in algo_map.items():
            flat = df.copy()
            flat.columns = [
                c if isinstance(c, str) else "_".join(p for p in c if p)
                for c in flat.columns
            ]
            # merge _mean/_std pairs into "mean±std" strings
            base_cols = {c[:-5] for c in flat.columns if c.endswith("_mean")}
            for base in sorted(base_cols):
                mean_col, std_col = f"{base}_mean", f"{base}_std"
                flat[base] = flat.apply(
                    lambda r, m=mean_col, s=std_col:
                        f"{r[m]:.2f}±{r[s]:.2f}", axis=1)
                flat.drop(columns=[mean_col, std_col], inplace=True)
            flat.insert(0, "algo", ALGO_MAP[algo_key])
            flat.insert(0, "scenario", scenario)
            if mu is not None:
                flat.insert(2, "noise_mu", mu)
            rows.append(flat)
    pd.concat(rows, ignore_index=True).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


_dump_agg(delay_data,   OUT_DIR / "aggregated_delay.csv")
_dump_agg(fault_data,   OUT_DIR / "aggregated_fault.csv")
_dump_agg(noise_merged, OUT_DIR / "aggregated_noise.csv")


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
n_rows = len(SCENARIOS)
fig, axes = plt.subplots(
    n_rows, 5, figsize=(FIG_W, FIG_H),
    sharex="col",            # 5 sets of x-ticks instead of 25
)

for row_idx, scenario in enumerate(SCENARIOS):
    for col_idx, (col_title, x_col, y_nd, y_wd, xlabel) in enumerate(COLUMNS):
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

        mu = MU_BY_COL[col_idx]
        if col_idx == 0:
            algo_dfs = delay_data.get(scenario, {})
        elif col_idx == 1:
            algo_dfs = fault_data.get(scenario, {})
        else:
            algo_dfs = noise_merged.get((scenario, mu), {})

        plot_lines(ax, algo_dfs, x_col, y_nd, y_wd)

# ── compact 6-entry legend: colour = algorithm, style = diffusion on/off ──────
color_handles = [Line2D([], [], color=COLORS[ALGO_MAP[k]], lw=1.5,
                        label=ALGO_MAP[k]) for k in ALGO_ORDER]
style_handles = [
    Line2D([], [], color="black", ls="-", lw=1.2, label="no diffusion (baseline)"),
    Line2D([], [], color="black", ls=":", lw=1.4, label="with diffusion (ours)"),
]
fig.legend(handles=color_handles + style_handles,
           loc="lower center", ncol=6, frameon=False,
           bbox_to_anchor=(0.5, -0.015), columnspacing=1.2, handlelength=1.8)

fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.subplots_adjust(wspace=0.30, hspace=0.18)

for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
    out_path = OUT_DIR / f"all_scenarios.{ext}"
    fig.savefig(out_path, bbox_inches="tight", **kw)
    print(f"Saved {out_path}")
plt.close(fig)