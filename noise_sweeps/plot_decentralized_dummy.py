"""
plot_decentralized_dummy.py
===========================
Analysis-only (dummy) version of plots/diffusion_and_filters_all_scenarios, same layout and styling
(5 perturbation columns x 4 scenario rows), but only 2 lines per panel:
    noisy policy (no diffusion)  vs  decentralized per-agent diffusion denoiser (best t, selected after the fact)
M3DDPG only, single seed (seed0), so no std bands. No Kalman.

Reads:  noise_sweeps/decentralized/{delay_sweeps,actuation_fault,guassian_noise}/seed0/*.csv
Writes: noise_sweeps/plots/decentralized_dummy_all_scenarios.{pdf,png}
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
DATA = NOISE_DIR / "decentralized"
OUT_DIR = NOISE_DIR / "plots"
COLOR = "#E69F00"                     # M3DDPG colour in the original plot
SCENARIOS = ["simple_adversary", "simple_push", "simple_speaker_listener", "simple_spread"]
SCENARIO_LABEL = {
    "simple_adversary": "Physical\nDeception", "simple_push": "Keep-away",
    "simple_speaker_listener": "Cooperative\nCommunication", "simple_spread": "Cooperative\nNavigation",
}
LINESTYLES = {"no_filter": (0, ()), "diffusion": (0, (1, 1))}
LINEWIDTHS = {"no_filter": 1.2, "diffusion": 1.2}
METHOD_LABEL = {"no_filter": "no diffusion (noisy policy)",
                "diffusion": "decentralized diffusion (best t, after the fact)"}

# (title, x_col, xlabel, file pattern, baseline column)
COLUMNS = [
    ("Delay ($k$ steps)", "delay_k", "$k$", "delay_sweeps/seed0/{}__m3ddpgbest_delay_sweep.csv", "reward_noisy"),
    ("Actuation fault ($p$)", "sp_prob", "$p_\\mathrm{stuck}$", "actuation_fault/seed0/{}__m3ddpgbest_stuck_at_sweep.csv", "reward_noisy"),
    ("Biased noise $\\mu=-1$", "action_noise_std", r"$\alpha$", "guassian_noise/seed0/{}__m3ddpgbest_mu-1.0_actstd_tstart_sweep.csv", "reward_noise_no_diffusion"),
    ("Biased noise $\\mu=0$", "action_noise_std", r"$\alpha$", "guassian_noise/seed0/{}__m3ddpgbest_mu0.0_actstd_tstart_sweep.csv", "reward_noise_no_diffusion"),
    ("Biased noise $\\mu=+1$", "action_noise_std", r"$\alpha$", "guassian_noise/seed0/{}__m3ddpgbest_mu1.0_actstd_tstart_sweep.csv", "reward_noise_no_diffusion"),
]

n_rows, n_cols = len(SCENARIOS), len(COLUMNS)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_W, FIG_H * n_rows / 5.0), sharex="col", squeeze=False)
for r, scen in enumerate(SCENARIOS):
    for c, (title, x_col, xlabel, pat, base_col) in enumerate(COLUMNS):
        ax = axes[r, c]
        ax.grid(True, alpha=0.3)
        ax.tick_params(length=2, pad=1.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if r == 0:
            ax.set_title(title, fontweight="bold", pad=3)
        if r == n_rows - 1:
            ax.set_xlabel(xlabel)
        if c == 0:
            ax.set_ylabel(SCENARIO_LABEL[scen], rotation=0, ha="right", va="center", labelpad=8)
        df = pd.read_csv(DATA / pat.format(scen)).sort_values(x_col)
        for method, y_col in (("no_filter", base_col), ("diffusion", "best_reward_with_diffusion")):
            ax.plot(df[x_col], df[y_col], color=COLOR, linestyle=LINESTYLES[method], linewidth=LINEWIDTHS[method])

fig.supylabel("Reward")
handles = [Line2D([], [], color=COLOR, lw=1.5, label="M3DDPG (seed 0)")] + \
          [Line2D([], [], color="black", ls=LINESTYLES[m], lw=1.3, label=METHOD_LABEL[m]) for m in ("no_filter", "diffusion")]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
           bbox_to_anchor=(0.5, -0.035), columnspacing=1.2, handlelength=2.2)
fig.tight_layout(rect=(0.01, 0.008, 1, 1), pad=0.15)
fig.subplots_adjust(wspace=0.30, hspace=0.18)
for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
    out = OUT_DIR / "decentralized_dummy_all_scenarios.{}".format(ext)
    fig.savefig(out, bbox_inches="tight", **kw)
    print("Saved", out)
plt.close(fig)
