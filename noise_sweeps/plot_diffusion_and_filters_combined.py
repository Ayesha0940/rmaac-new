"""
plot_diffusion_and_filters_combined.py
=======================================
Combined comparison: the DDPM diffusion denoiser (train.py's
--use-denoiser sweep path) vs. the classical EMA/median/Kalman
action-recovery filters (action_filters.py / filter_eval.py), on one grid
(5 perturbation-type columns x 4-scenario rows), all overlaid against a
single no-recovery baseline.

Reads:
    DDPM data -- baseline AND diffusion line, same source/columns/multi-seed
    aggregation plot_combined.py's all_scenarios.pdf uses, so both lines here
    are identical to that plot, not a separately-run baseline:
        noise_sweeps/delay_sweeps/seed*/*_delay_sweep.csv       (reward_noisy, best_reward_with_diffusion)
        noise_sweeps/actuation_fault/seed*/*_stuck_at_sweep.csv (reward_noisy, best_reward_with_diffusion)
        noise_sweeps/guassian_noise/seed*/*_actstd_tstart_sweep.csv (reward_noise_no_diffusion, best_reward_with_diffusion; mu=-1,0,1)
    Filter data -- ONLY reward_kalman is taken from here (EMA/median are
    loaded/plotted nowhere in this version; this pipeline's own
    reward_no_filter baseline is intentionally not plotted either, since the
    DDPM baseline above already covers that role):
        noise_sweeps/filters/{delay,actuation_fault,noise,
            biased_noise_pos,biased_noise_neg}/seed*/*.csv

Note: the DDPM noise sweep used a coarser action_noise_std grid (0,1,2,3)
than the filter sweep (9-point 0.0-3.0 grid) -- both are drawn on the same
x-axis but are plotted independently (not merged/joined row-by-row), so the
Kalman line will visibly show more sampled points than the baseline/diffusion
lines in the 3 noise-related columns.

Writes:
    noise_sweeps/plots/diffusion_and_filters_all_scenarios.{pdf,png}
"""
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

FIG_W, FIG_H = 7.16, 5.6   # matches plot_combined.py's all_scenarios.pdf sizing
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
}

LINESTYLES = {
    "no_filter": (0, ()),          # matches plot_combined.py's "-" baseline
    "diffusion": (0, (1, 1)),      # matches plot_combined.py's ":" diffusion line
    "kalman":    (0, (6, 2, 1, 2)),  # long dash + dot, coarser than a tight dash-dot so it reads at panel scale
    "mlp":       (0, (4, 1.5)),      # plain dash, the one style not used above
}
LINEWIDTHS = {"no_filter": 1.2, "diffusion": 1.2, "kalman": 1.3, "mlp": 1.2}
METHOD_LABEL = {
    "no_filter": "no diffusion (baseline)",  # exact label text from plot_combined.py
    "diffusion": "with diffusion (ours)",    # exact label text from plot_combined.py
    "kalman":    "Kalman",
    "mlp":       "reconstruction",
}


def _agg(dfs, x_col, y_cols):
    combined = pd.concat(dfs, ignore_index=True)
    return combined.groupby(x_col)[y_cols].agg(["mean", "std"]).reset_index()


# ── DDPM diffusion data (baseline + diffusion, byte-for-byte the same source
# and aggregation plot_combined.py uses for its "no diffusion" / "with
# diffusion" lines -- this is what all_scenarios.pdf's baseline/diffusion
# lines are drawn from, so this plot's baseline/diffusion must match exactly) ──
DELAY_DDPM_RE = re.compile(r"^(simple_.+?)__(.+?)_delay_sweep\.csv$")
_delay_ddpm_raw = defaultdict(lambda: defaultdict(list))
for f in sorted((NOISE_DIR / "delay_sweeps").glob("*/*.csv")):
    m = DELAY_DDPM_RE.match(f.name)
    if not m or m.group(2) not in ALGO_MAP:
        continue
    df = pd.read_csv(f)[["delay_k", "reward_noisy", "best_reward_with_diffusion"]]
    df = df.rename(columns={"reward_noisy": "baseline"})
    _delay_ddpm_raw[m.group(1)][m.group(2)].append(df)
delay_ddpm = {s: {a: _agg(dfs, "delay_k", ["baseline", "best_reward_with_diffusion"]) for a, dfs in m.items()}
               for s, m in _delay_ddpm_raw.items()}

FAULT_DDPM_RE = re.compile(r"^(simple_.+?)__(.+?)_stuck_at_sweep\.csv$")
_fault_ddpm_raw = defaultdict(lambda: defaultdict(list))
for f in sorted((NOISE_DIR / "actuation_fault").glob("*/*.csv")):
    m = FAULT_DDPM_RE.match(f.name)
    if not m or m.group(2) not in ALGO_MAP:
        continue
    df = pd.read_csv(f)[["sp_prob", "reward_noisy", "best_reward_with_diffusion"]]
    df = df.rename(columns={"reward_noisy": "baseline"})
    _fault_ddpm_raw[m.group(1)][m.group(2)].append(df)
fault_ddpm = {s: {a: _agg(dfs, "sp_prob", ["baseline", "best_reward_with_diffusion"]) for a, dfs in m.items()}
               for s, m in _fault_ddpm_raw.items()}

NOISE_DDPM_RE = re.compile(r"^(simple_.+?)__(.+?)(?:_mu[-\d.]+)?_actstd_tstart_sweep\.csv$")
_noise_ddpm_raw = defaultdict(lambda: defaultdict(list))
for f in sorted((NOISE_DIR / "guassian_noise").glob("*/*.csv")):
    m = NOISE_DDPM_RE.match(f.name)
    if not m or m.group(2) not in ALGO_MAP:
        continue
    cols0 = pd.read_csv(f, nrows=0).columns
    df = (pd.read_csv(f)[["noise_mu", "action_noise_std", "reward_noise_no_diffusion", "best_reward_with_diffusion"]]
          if "noise_mu" in cols0
          else pd.read_csv(f)[["action_noise_std", "reward_noise_no_diffusion", "best_reward_with_diffusion"]])
    df = df.rename(columns={"reward_noise_no_diffusion": "baseline"})
    if "noise_mu" not in df.columns:
        df.insert(0, "noise_mu", 0)
    df["noise_mu"] = df["noise_mu"].round().astype(int)
    _noise_ddpm_raw[m.group(1)][m.group(2)].append(df)

noise_ddpm_by_mu = {}
for mu_val in [0, 1, -1]:
    merged = {}
    for scenario, algo_map in _noise_ddpm_raw.items():
        algo_dfs = {}
        for algo_key, dfs in algo_map.items():
            combined = pd.concat(dfs, ignore_index=True)
            subset = combined[combined["noise_mu"] == mu_val]
            if subset.empty:
                continue
            algo_dfs[algo_key] = subset.groupby("action_noise_std")[
                ["baseline", "best_reward_with_diffusion"]].agg(["mean", "std"]).reset_index()
        if algo_dfs:
            merged[scenario] = algo_dfs
    noise_ddpm_by_mu[mu_val] = merged

# ── classical filter data (only Kalman is drawn from this source; its own
# reward_no_filter baseline is intentionally not plotted -- baseline is drawn
# from the DDPM pipeline's data above instead, see plot_lines) ───────────────
FILTER_Y = ["reward_kalman"]


def _load_filter(glob_pattern, regex, x_col):
    raw = defaultdict(lambda: defaultdict(list))
    for f in sorted(NOISE_DIR.glob(glob_pattern)):
        m = regex.match(f.name)
        if not m or m.group(2) not in ALGO_MAP:
            continue
        df = pd.read_csv(f)
        cols = [x_col] + [c for c in FILTER_Y if c in df.columns]
        raw[m.group(1)][m.group(2)].append(df[cols])
    return raw


def _agg_filter(dfs, x_col):
    combined = pd.concat(dfs, ignore_index=True)
    y = [c for c in FILTER_Y if c in combined.columns]
    return combined.groupby(x_col)[y].agg(["mean", "std"]).reset_index()


DELAY_F_RE = re.compile(r"^(simple_.+?)__(.+?)_delay_filter_sweep\.csv$")
FAULT_F_RE = re.compile(r"^(simple_.+?)__(.+?)_stuck_at_filter_sweep\.csv$")
NOISE_F_RE = re.compile(r"^(simple_.+?)__(.+?)_mu0\.0_actstd_filter_sweep\.csv$")
BIASPOS_F_RE = re.compile(r"^(simple_.+?)__(.+?)_mu1\.0_actstd_filter_sweep\.csv$")
BIASNEG_F_RE = re.compile(r"^(simple_.+?)__(.+?)_mu-1\.0_actstd_filter_sweep\.csv$")

_delay_f_raw = _load_filter("filters/delay/*/*.csv", DELAY_F_RE, "delay_k")
_fault_f_raw = _load_filter("filters/actuation_fault/*/*.csv", FAULT_F_RE, "sp_prob")
_noise_f_raw = _load_filter("filters/noise/*/*.csv", NOISE_F_RE, "action_noise_std")
_biaspos_f_raw = _load_filter("filters/biased_noise_pos/*/*.csv", BIASPOS_F_RE, "action_noise_std")
_biasneg_f_raw = _load_filter("filters/biased_noise_neg/*/*.csv", BIASNEG_F_RE, "action_noise_std")

delay_f = {s: {a: _agg_filter(dfs, "delay_k") for a, dfs in m.items()} for s, m in _delay_f_raw.items()}
fault_f = {s: {a: _agg_filter(dfs, "sp_prob") for a, dfs in m.items()} for s, m in _fault_f_raw.items()}
noise_f = {s: {a: _agg_filter(dfs, "action_noise_std") for a, dfs in m.items()} for s, m in _noise_f_raw.items()}
biaspos_f = {s: {a: _agg_filter(dfs, "action_noise_std") for a, dfs in m.items()} for s, m in _biaspos_f_raw.items()}
biasneg_f = {s: {a: _agg_filter(dfs, "action_noise_std") for a, dfs in m.items()} for s, m in _biasneg_f_raw.items()}

# ── MLP joint-action denoiser ("reconstruction"): MADDPG only, trained on the
# episode's initial state like the DDPM, evaluated at low/high severity only
# (run_mlp_denoiser_sweep.py), so each panel has two points per scenario ────
MLP_RE = re.compile(r"^(simple_.+?)__maddpgbest_mlp_sweep\.csv$")
_mlp_raw = defaultdict(list)
for f in sorted((NOISE_DIR / "mlp_denoiser_initial").glob("seed*/*_mlp_sweep.csv")):
    m = MLP_RE.match(f.name)
    if m:
        _mlp_raw[m.group(1)].append(pd.read_csv(f))


def _mlp_for(attack, mu, x_col):
    out = {}
    for scenario, dfs in _mlp_raw.items():
        df = pd.concat(dfs, ignore_index=True)
        df = df[df["attack"] == attack]
        if mu is not None:
            df = df[df["noise_mu"] == mu]
        df = df.rename(columns={"severity": x_col})[[x_col, "reward_mlp"]]
        out[scenario] = {"maddpgbest": _agg([df], x_col, ["reward_mlp"])}
    return out


# (title, x_col, xlabel, ddpm_data, filter_data, mlp_data) -- same order/titles as
# plot_combined.py's COLUMNS: Delay, Actuation fault, Biased noise mu=-1/0/+1.
COLUMNS = [
    ("Delay ($k$ steps)",      "delay_k",          "$k$",                 delay_ddpm,           delay_f,   _mlp_for("delay", None, "delay_k")),
    ("Actuation fault ($p$)",  "sp_prob",          "$p_\\mathrm{stuck}$", fault_ddpm,           fault_f,   _mlp_for("stuck_at", None, "sp_prob")),
    ("Biased noise $\\mu=-1$", "action_noise_std", r"$\alpha$",           noise_ddpm_by_mu[-1], biasneg_f, _mlp_for("biased_gauss", -1.0, "action_noise_std")),
    ("Biased noise $\\mu=0$",  "action_noise_std", r"$\alpha$",           noise_ddpm_by_mu[0],  noise_f,   _mlp_for("gauss", 0.0, "action_noise_std")),
    ("Biased noise $\\mu=+1$", "action_noise_std", r"$\alpha$",           noise_ddpm_by_mu[1],  biaspos_f, _mlp_for("biased_gauss", 1.0, "action_noise_std")),
]

SCENARIOS = sorted(set(delay_f) | set(fault_f) | set(noise_f) | set(biaspos_f) | set(biasneg_f))


def plot_lines(ax, ddpm_algo_dfs, filter_algo_dfs, x_col, mlp_algo_dfs=None):
    mlp_algo_dfs = mlp_algo_dfs or {}
    for algo_key in ALGO_ORDER:
        color = COLORS[ALGO_MAP[algo_key]]

        # baseline + diffusion: both from the DDPM pipeline's own data, same
        # source/aggregation as plot_combined.py's all_scenarios.pdf.
        if algo_key in ddpm_algo_dfs:
            df = ddpm_algo_dfs[algo_key]
            x = df[x_col]
            for method, y_col in [("no_filter", "baseline"), ("diffusion", "best_reward_with_diffusion")]:
                if (y_col, "mean") not in df.columns:
                    continue
                mean = df[(y_col, "mean")]
                std = df[(y_col, "std")].fillna(0)
                ax.plot(x, mean, color=color, linestyle=LINESTYLES[method], linewidth=LINEWIDTHS[method])
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.08, linewidth=0)

        # Kalman: from the classical-filter sweep pipeline, aggregated over
        # seed0/5/10 (the DDPM sweep has no equivalent of this method). Uses
        # a coarser dash pattern than a tight dash-dot so it's traceable
        # where it sits close to baseline, without extra weight (no markers,
        # linewidth close to the other lines) that would dominate the plot.
        if algo_key in filter_algo_dfs:
            df = filter_algo_dfs[algo_key]
            x = df[x_col]
            method, y_col = "kalman", "reward_kalman"
            if (y_col, "mean") in df.columns:
                mean = df[(y_col, "mean")]
                std = df[(y_col, "std")].fillna(0)
                ax.plot(x, mean, color=color, linestyle=LINESTYLES[method], linewidth=LINEWIDTHS[method])
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.08, linewidth=0)

        # MLP reconstruction: MADDPG only, two severities per panel, so it gets
        # small markers to stay visible as a short segment.
        if algo_key in mlp_algo_dfs:
            df = mlp_algo_dfs[algo_key]
            x = df[x_col]
            method, y_col = "mlp", "reward_mlp"
            mean = df[(y_col, "mean")]
            std = df[(y_col, "std")].fillna(0)
            ax.plot(x, mean, color=color, linestyle=LINESTYLES[method], linewidth=LINEWIDTHS[method],
                    marker="o", markersize=2.5)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.08, linewidth=0)


if not SCENARIOS:
    print("No filter-sweep CSVs found under noise_sweeps/filters/ — run "
          "run_full_filter_sweeps.py / run_full_filter_sweeps_biased.py first.")
else:
    n_rows = len(SCENARIOS)
    n_cols = len(COLUMNS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_W, FIG_H * n_rows / 5.0),
                              sharex="col", squeeze=False)

    for row_idx, scenario in enumerate(SCENARIOS):
        for col_idx, (col_title, x_col, xlabel, ddpm_data, filter_data, mlp_data) in enumerate(COLUMNS):
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

            ddpm_algo_dfs = ddpm_data.get(scenario, {})
            filter_algo_dfs = filter_data.get(scenario, {})
            plot_lines(ax, ddpm_algo_dfs, filter_algo_dfs, x_col, mlp_data.get(scenario, {}))

    fig.supylabel("Reward")

    color_handles = [Line2D([], [], color=COLORS[ALGO_MAP[k]], lw=1.5, label=ALGO_MAP[k])
                      for k in ALGO_ORDER]
    style_handles = [Line2D([], [], color="black", ls=LINESTYLES[m], lw=1.3, label=METHOD_LABEL[m])
                      for m in ["no_filter", "diffusion", "kalman"]]
    style_handles.append(Line2D([], [], color="black", ls=LINESTYLES["mlp"], lw=1.3,
                                marker="o", markersize=2.5, label=METHOD_LABEL["mlp"]))
    fig.legend(handles=color_handles + style_handles,
               loc="lower center", ncol=len(color_handles) + len(style_handles), frameon=False,
               bbox_to_anchor=(0.5, -0.015), columnspacing=1.2, handlelength=2.2)

    fig.tight_layout(rect=(0.01, 0.030, 1, 1))
    fig.subplots_adjust(wspace=0.30, hspace=0.18)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out_path = OUT_DIR / f"diffusion_and_filters_all_scenarios.{ext}"
        fig.savefig(out_path, bbox_inches="tight", **kw)
        print(f"Saved {out_path}")
    plt.close(fig)
