"""
plot_decentralized_reduced_dummy.py -- analysis-only (dummy) plot, same layout/styling as
plots/diffusion_and_filters_all_scenarios (4 scenario rows x 5 perturbation columns), 2 lines per panel:
    reduced-observation decentralized denoiser (best t, selected after the fact)   [dotted, as the diffusion line]
    BC on the same reduced input, one FLAT line at its no-fault value               [solid]
M3DDPG, single seed (seed0). Level per row: speaker goal colour removed / Physical Deception goal removed /
Keep-away goal+colour removed / Navigation other-agents removed.
Reads noise_sweeps/decentralized_reduced/<scenario>__<role>__<level>/...; writes plots/decentralized_reduced_dummy_all_scenarios.{pdf,png}
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

FIG_W, FIG_H = 7.16, 5.6
plt.rcParams.update({
    "font.family": "serif", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    "lines.linewidth": 1.0, "grid.linewidth": 0.4, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

NOISE_DIR = Path(__file__).parent
DATA = NOISE_DIR / "decentralized_reduced"
OUT_DIR = NOISE_DIR / "plots"
COLOR = "#E69F00"
ROWS = [("simple_adversary", "good", "removed", "Physical\nDeception"),
        ("simple_push", "good", "goal-hidden", "Keep-away"),
        ("simple_speaker_listener", "speaker", "removed", "Cooperative\nCommunication"),
        ("simple_spread", "shared", "others-removed", "Cooperative\nNavigation")]
LINESTYLES = {"bc": (0, ()), "diffusion": (0, (1, 1))}
COLUMNS = [
    ("Delay ($k$ steps)", "delay_k", "$k$", "delay_sweeps/seed0/{s}__m3ddpgbest_delay_sweep.csv"),
    ("Actuation fault ($p$)", "sp_prob", "$p_\\mathrm{stuck}$", "actuation_fault/seed0/{s}__m3ddpgbest_stuck_at_sweep.csv"),
    ("Biased noise $\\mu=-1$", "action_noise_std", r"$\alpha$", "guassian_noise/seed0/{s}__m3ddpgbest_mu-1.0_actstd_tstart_sweep.csv"),
    ("Biased noise $\\mu=0$", "action_noise_std", r"$\alpha$", "guassian_noise/seed0/{s}__m3ddpgbest_mu0.0_actstd_tstart_sweep.csv"),
    ("Biased noise $\\mu=+1$", "action_noise_std", r"$\alpha$", "guassian_noise/seed0/{s}__m3ddpgbest_mu1.0_actstd_tstart_sweep.csv"),
]

fig, axes = plt.subplots(len(ROWS), len(COLUMNS), figsize=(FIG_W, FIG_H * len(ROWS) / 5.0), sharex="col", squeeze=False)
for r, (scen, role, lvl, label) in enumerate(ROWS):
    base = DATA / "{}__{}__{}".format(scen, role, lvl)
    # flat BC line: BC at the no-fault cell (delay 0 of the delay sweep)
    d0 = pd.read_csv(base / COLUMNS[0][3].format(s=scen)).sort_values("delay_k")
    bc_flat = float(d0.loc[d0.delay_k == 0, "reward_bc_reduced"].iloc[0])
    for c, (title, x_col, xlabel, pat) in enumerate(COLUMNS):
        ax = axes[r, c]
        ax.grid(True, alpha=0.3)
        ax.tick_params(length=2, pad=1.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if r == 0:
            ax.set_title(title, fontweight="bold", pad=3)
        if r == len(ROWS) - 1:
            ax.set_xlabel(xlabel)
        if c == 0:
            ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=8)
        df = pd.read_csv(base / pat.format(s=scen)).sort_values(x_col)
        ax.plot(df[x_col], df["best_reward_with_diffusion"], color=COLOR, linestyle=LINESTYLES["diffusion"], linewidth=1.2)
        ax.plot(df[x_col], [bc_flat] * len(df), color=COLOR, linestyle=LINESTYLES["bc"], linewidth=1.2)

fig.supylabel("Reward")
handles = [Line2D([], [], color=COLOR, lw=1.5, label="M3DDPG (seed 0)"),
           Line2D([], [], color="black", ls=LINESTYLES["bc"], lw=1.3, label="BC, same reduced input (flat)"),
           Line2D([], [], color="black", ls=LINESTYLES["diffusion"], lw=1.3, label="reduced-obs denoiser (best t, after the fact)")]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
           bbox_to_anchor=(0.5, -0.035), columnspacing=1.2, handlelength=2.2)
fig.tight_layout(rect=(0.01, 0.008, 1, 1), pad=0.15)
fig.subplots_adjust(wspace=0.30, hspace=0.18)
for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
    out = OUT_DIR / "decentralized_reduced_dummy_all_scenarios.{}".format(ext)
    fig.savefig(out, bbox_inches="tight", **kw)
    print("Saved", out)
plt.close(fig)
