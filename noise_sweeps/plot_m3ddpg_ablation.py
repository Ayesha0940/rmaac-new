"""Two 4×3 figures: preyonly and predatoronly ablation, all 4 algorithms.
M3DDPG uses its _preyonly / _predatoronly CSV; the other 3 use their standard CSV.
Color = algorithm, solid = no diffusion, dotted = with diffusion.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

NOISE_DIR = Path(__file__).parent
DATA_DIR  = NOISE_DIR / "guassian_noise" / "seed49" / "predprey"
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

ALGO_MAP   = {"earniebest": "ERNIE", "m3ddpgbest": "M3DDPG",
              "maddpgbest": "MADDPG", "rmaacbest": "RMAAC"}
COLORS     = {"ERNIE": "#0072B2", "M3DDPG": "#E69F00",
              "MADDPG": "#009E73", "RMAAC": "#D55E00"}
ALGO_ORDER = ["earniebest", "m3ddpgbest", "maddpgbest", "rmaacbest"]

ROWS = [
    ("Predator reward", "reward_pred_noise_no_diffusion", "best_reward_pred_with_diffusion"),
    ("Prey reward",     "reward_prey_noise_no_diffusion", "best_reward_prey_with_diffusion"),
    ("Captures",        "captures_noisy",                 "best_captures_with_diffusion"),
    ("Survival",        "survival_noisy",                 "best_survival_with_diffusion"),
]

MU_VALUES = [-1.0, 0.0, 1.0]
MU_LABELS = [r"Biased noise $\mu=-1$", r"Biased noise $\mu=0$", r"Biased noise $\mu=+1$"]

# ── load ──────────────────────────────────────────────────────────────────────

def _load(algo_key, mu_val, suffix=""):
    fname = f"simple_tag__{algo_key}_mu{mu_val:.1f}_actstd_tstart_sweep{suffix}.csv"
    df = pd.read_csv(DATA_DIR / fname)
    df["best_captures_with_diffusion"] = df[["captures_diff_t20", "captures_diff_t40"]].max(axis=1)
    df["best_survival_with_diffusion"]  = df[["survival_diff_t20",  "survival_diff_t40"]].max(axis=1)
    return df.sort_values("action_noise_std")


def _load_all(m3ddpg_suffix):
    """Load all 4 algos for all mu values. M3DDPG uses m3ddpg_suffix; others use ''."""
    return {
        mu: {
            algo: _load(algo, mu, m3ddpg_suffix if algo == "m3ddpgbest" else "")
            for algo in ALGO_ORDER
        }
        for mu in MU_VALUES
    }


# ── shared figure builder ──────────────────────────────────────────────────────

def make_figure(data, title):
    fig, axes = plt.subplots(4, 3, figsize=(7.16, 4.5), sharex="col")
    fig.suptitle(title, fontsize=8, fontweight="bold", y=1.005)

    for row_idx, (row_label, y_nd, y_wd) in enumerate(ROWS):
        for col_idx, (mu_val, mu_label) in enumerate(zip(MU_VALUES, MU_LABELS)):
            ax = axes[row_idx, col_idx]
            ax.grid(True, alpha=0.3)
            ax.tick_params(length=2, pad=1.5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

            if row_idx == 0:
                ax.set_title(mu_label, fontweight="bold", pad=3)
            if row_idx == len(ROWS) - 1:
                ax.set_xlabel(r"$\alpha$")
            if col_idx == 0:
                ax.set_ylabel(row_label, rotation=0, ha="right", va="center", labelpad=8)

            for algo_key in ALGO_ORDER:
                df    = data[mu_val][algo_key]
                color = COLORS[ALGO_MAP[algo_key]]
                x     = df["action_noise_std"]
                ax.plot(x, df[y_nd], color=color, linestyle="-",  linewidth=1.0)
                ax.plot(x, df[y_wd], color=color, linestyle=":",  linewidth=1.1)

    color_handles = [Line2D([], [], color=COLORS[ALGO_MAP[k]], lw=1.5,
                            label=ALGO_MAP[k]) for k in ALGO_ORDER]
    style_handles = [
        Line2D([], [], color="black", ls="-",  lw=1.2, label="no diffusion (baseline)"),
        Line2D([], [], color="black", ls=":",  lw=1.4, label="with diffusion (ours)"),
    ]
    fig.legend(handles=color_handles + style_handles,
               loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.015), columnspacing=1.2, handlelength=1.8)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.subplots_adjust(wspace=0.35, hspace=0.18)
    return fig


# ── generate both figures ─────────────────────────────────────────────────────

FIGURES = [
    ("_preyonly",     "M3DDPG prey-only vs others — Predator–Prey (seed 49)",     "m3ddpg_preyonly"),
    ("_predatoronly", "M3DDPG predator-only vs others — Predator–Prey (seed 49)", "m3ddpg_predatoronly"),
]

for m3ddpg_suffix, title, stem in FIGURES:
    data = _load_all(m3ddpg_suffix)
    fig  = make_figure(data, title)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out_path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", **kw)
        print(f"Saved {out_path}")
    plt.close(fig)
