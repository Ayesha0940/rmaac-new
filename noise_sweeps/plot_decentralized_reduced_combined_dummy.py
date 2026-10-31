"""
plot_decentralized_reduced_combined_dummy.py -- analysis-only (dummy) plot, same layout/styling as
plots/diffusion_and_filters_all_scenarios (4 scenario rows x 5 perturbation columns), 4 lines per panel, M3DDPG, seed 0 (orange = noisy / full-obs denoiser, green = reduced-obs denoiser / BC, yellow = BC on the full local observation, flat, purple = state-only diffusion (full: dark, reduced: light), flat,
blue = supervised reconstructor C1, unscaled (full: dark dotted, reduced: light dash-dot)):
    noisy policy (no diffusion)                              solid
    decentralized denoiser, FULL local observation           dotted  (best t, selected after the fact)
    decentralized denoiser, REDUCED observation              dash-dot (best t, selected after the fact)
    BC on the same reduced input, one FLAT line              dashed  (value at the no-fault cell)
Level per row: Physical Deception good goal position removed / Keep-away goal+colour removed /
Cooperative Communication speaker goal colour removed / Navigation other agents removed.
Reads noise_sweeps/decentralized_reduced/<scenario>__<role>__<level>/ (noisy, full-obs and reduced denoiser, BC columns);
writes plots/decentralized_reduced_combined_dummy_all_scenarios.{pdf,png}
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
COLOR = "#E69F00"            # M3DDPG colour (noisy, full-observation denoiser)
GREEN = "#009E73"            # reduced-observation denoiser and BC
YELLOW = "#F0E442"           # BC on the full local observation
PURPLE = "#7B3FA0"           # state-only diffusion, full local observation
PURPLE_L = "#B07CC6"         # state-only diffusion, reduced observation
BLUE = "#0072B2"             # supervised reconstructor C1, full local observation
BLUE_L = "#56B4E9"           # supervised reconstructor C1, reduced observation
C1_DIR = NOISE_DIR / "decentralized" / "c1"
C1_PAT = ["delay_sweeps/seed0/{s}__m3ddpgbest_delay_sweep.csv", "actuation_fault/seed0/{s}__m3ddpgbest_stuck_at_sweep.csv",
          "guassian_noise/seed0/{s}__m3ddpgbest_mu-1.0_actstd_c1.csv", "guassian_noise/seed0/{s}__m3ddpgbest_mu0.0_actstd_c1.csv",
          "guassian_noise/seed0/{s}__m3ddpgbest_mu1.0_actstd_c1.csv"]
LCOLOR = {"noisy": COLOR, "full": COLOR, "reduced": GREEN, "bc": GREEN, "bcfull": YELLOW, "sofull": PURPLE, "sored": PURPLE_L,
          "c1full": BLUE, "c1red": BLUE_L}
ROWS = [("simple_adversary", "good", "removed", "Physical\nDeception"),
        ("simple_push", "good", "goal-hidden", "Keep-away"),
        ("simple_speaker_listener", "speaker", "removed", "Cooperative\nCommunication"),
        ("simple_spread", "shared", "others-removed", "Cooperative\nNavigation")]
STYLE = {"noisy": (0, ()), "full": (0, (1, 1)), "reduced": (0, (6, 2, 1, 2)), "bc": (0, (4, 2)), "bcfull": (0, (4, 2)),
         "sofull": (0, (7, 2, 1, 2, 1, 2)), "sored": (0, (2, 2)),
         "c1full": (0, (1, 1)), "c1red": (0, (6, 2, 1, 2))}
LABEL = {"noisy": "no diffusion (noisy policy)", "full": "denoiser, full local obs",
         "reduced": "denoiser, reduced obs", "bc": "BC, reduced obs (flat)",
         "bcfull": "BC, full local obs (flat)", "sofull": "state-only diffusion, full obs (flat)",
         "sored": "state-only diffusion, reduced obs (flat)",
         "c1full": "reconstructor C1, full local obs", "c1red": "reconstructor C1, reduced obs"}
COLUMNS = [
    ("Delay ($k$ steps)", "delay_k", "$k$", "delay_sweeps/seed0/{s}__m3ddpgbest_delay_sweep.csv", "reward_noisy"),
    ("Actuation fault ($p$)", "sp_prob", "$p_\\mathrm{stuck}$", "actuation_fault/seed0/{s}__m3ddpgbest_stuck_at_sweep.csv", "reward_noisy"),
    ("Biased noise $\\mu=-1$", "action_noise_std", r"$\alpha$", "guassian_noise/seed0/{s}__m3ddpgbest_mu-1.0_actstd_tstart_sweep.csv", "reward_noise_no_diffusion"),
    ("Biased noise $\\mu=0$", "action_noise_std", r"$\alpha$", "guassian_noise/seed0/{s}__m3ddpgbest_mu0.0_actstd_tstart_sweep.csv", "reward_noise_no_diffusion"),
    ("Biased noise $\\mu=+1$", "action_noise_std", r"$\alpha$", "guassian_noise/seed0/{s}__m3ddpgbest_mu1.0_actstd_tstart_sweep.csv", "reward_noise_no_diffusion"),
]

fig, axes = plt.subplots(len(ROWS), len(COLUMNS), figsize=(FIG_W, FIG_H * len(ROWS) / 5.0), sharex="col", squeeze=False)
for r, (scen, role, lvl, label) in enumerate(ROWS):
    base = DATA / "{}__{}__{}".format(scen, role, lvl)
    d0 = pd.read_csv(base / COLUMNS[0][3].format(s=scen)).sort_values("delay_k")
    bc_flat = float(d0.loc[d0.delay_k == 0, "reward_bc_reduced"].iloc[0])      # BC at the no-fault cell
    bcf = pd.read_csv(NOISE_DIR / "decentralized" / "bc_full" / "seed0" / "{}__bc_full_obs.csv".format(scen))
    bc_full_flat = float(bcf.loc[bcf.method == "bc", "reward_mean"].iloc[0])  # BC, full local obs, all agents (fault-independent)
    so = {}
    for mode in ("full", "reduced"):                                           # state-only diffusion (fault-independent)
        t_ = pd.read_csv(NOISE_DIR / "decentralized" / "state_only" / "seed0" / "{}__state_only_{}.csv".format(scen, mode))
        so[mode] = float(t_.loc[t_.method == "state_only", "reward_mean"].iloc[0])
    for c, (title, x_col, xlabel, pat, base_col) in enumerate(COLUMNS):
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
        x = df[x_col]
        ax.plot(x, df[base_col], color=COLOR, linestyle=STYLE["noisy"], linewidth=1.2)
        ax.plot(x, df["best_reward_with_diffusion_full_obs"], color=COLOR, linestyle=STYLE["full"], linewidth=1.2)
        ax.plot(x, df["best_reward_with_diffusion"], color=GREEN, linestyle=STYLE["reduced"], linewidth=1.3)
        ax.plot(x, [bc_full_flat] * len(df), color=YELLOW, linestyle=STYLE["bcfull"], linewidth=1.4)
        ax.plot(x, [bc_flat] * len(df), color=GREEN, linestyle=STYLE["bc"], linewidth=1.2)
        ax.plot(x, [so["full"]] * len(df), color=PURPLE, linestyle=STYLE["sofull"], linewidth=1.3)
        ax.plot(x, [so["reduced"]] * len(df), color=PURPLE_L, linestyle=STYLE["sored"], linewidth=1.5)
        c1 = pd.read_csv(C1_DIR / C1_PAT[c].format(s=scen)).sort_values(x_col)
        assert list(c1[x_col].round(4)) == list(x.round(4))
        ax.plot(c1[x_col], c1["reward_c1_full_obs"], color=BLUE, linestyle=STYLE["c1full"], linewidth=1.2)
        ax.plot(c1[x_col], c1["reward_c1_reduced_obs"], color=BLUE_L, linestyle=STYLE["c1red"], linewidth=1.3)

fig.supylabel("Reward")
handles = [Line2D([], [], color=LCOLOR[m], ls=STYLE[m], lw=1.4, label=LABEL[m]) for m in ("noisy", "full", "reduced", "bc", "bcfull", "sofull", "sored", "c1full", "c1red")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.11), columnspacing=1.0, handlelength=2.2)
fig.tight_layout(rect=(0.01, 0.008, 1, 1), pad=0.15)
fig.subplots_adjust(wspace=0.30, hspace=0.18)
for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
    out = OUT_DIR / "decentralized_reduced_combined_dummy_all_scenarios.{}".format(ext)
    fig.savefig(out, bbox_inches="tight", **kw)
    print("Saved", out)
plt.close(fig)
