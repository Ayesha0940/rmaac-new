"""
plot_decentralized_split.py -- the decentralized analysis plot split into two figures (same 4 scenario rows x 5 perturbation
columns layout as plots/diffusion_and_filters_all_scenarios), M3DDPG, seed 0:
  plots/decentralized_full_obs_all_scenarios.{pdf,png}    : every corrector uses the full local observation
  plots/decentralized_reduced_obs_all_scenarios.{pdf,png} : the reduced agent(s) use the reduced observation
Each figure: noisy policy (black, solid) + diffusion denoiser (best t, selected after the fact), supervised reconstructor C1
(unscaled, one fixed variant), BC (flat) and state-only diffusion (flat), all dotted; one colour per method (denoiser = black dotted like the noisy policy it corrects), the same in both
figures (palette checked with the dataviz validator: CVD dE >= 10.6, normal-vision dE >= 21.8, contrast >= 3:1).
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

FIG_W, FIG_H = 7.16, 5.6
plt.rcParams.update({
    "font.family": "serif", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "lines.linewidth": 1.0, "grid.linewidth": 0.4, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

NOISE_DIR = Path(__file__).parent
RED_DIR = NOISE_DIR / "decentralized_reduced"
DEC = NOISE_DIR / "decentralized"
OUT_DIR = NOISE_DIR / "plots"
COLOR = {"noisy": "#000000", "den": "#000000", "c1": "#3366CC", "bc": "#00A08A", "so": "#B8007A"}
DOT = (0, (1, 1.3))
ORDER = ["noisy", "den", "c1", "bc", "so"]
ROWS = [("simple_adversary", "good", "removed", "Physical\nDeception", "goal position removed"),
        ("simple_push", "good", "goal-hidden", "Keep-away", "goal position + colour removed"),
        ("simple_speaker_listener", "speaker", "removed", "Cooperative\nCommunication", "speaker goal colour removed"),
        ("simple_spread", "shared", "others-removed", "Cooperative\nNavigation", "others' positions removed")]
COLUMNS = [
    ("Delay ($k$ steps)", "delay_k", "$k$", "delay_sweeps/seed0/{s}__m3ddpgbest_delay_sweep.csv",
     "delay_sweeps/seed0/{s}__m3ddpgbest_delay_sweep.csv", "reward_noisy"),
    ("Actuation fault ($p$)", "sp_prob", "$p_\\mathrm{stuck}$", "actuation_fault/seed0/{s}__m3ddpgbest_stuck_at_sweep.csv",
     "actuation_fault/seed0/{s}__m3ddpgbest_stuck_at_sweep.csv", "reward_noisy")] + [
    ("Biased noise $\\mu={}$".format(t), "action_noise_std", r"$\alpha$",
     "guassian_noise/seed0/{{s}}__m3ddpgbest_mu{}_actstd_tstart_sweep.csv".format(m),
     "guassian_noise/seed0/{{s}}__m3ddpgbest_mu{}_actstd_c1.csv".format(m), "reward_noise_no_diffusion")
    for t, m in (("-1", "-1.0"), ("0", "0.0"), ("+1", "1.0"))]


def flat_values(scen, base):
    d0 = pd.read_csv(base / COLUMNS[0][3].format(s=scen))
    bcf = pd.read_csv(DEC / "bc_full" / "seed0" / "{}__bc_full_obs.csv".format(scen))
    so = {}
    for mode in ("full", "reduced"):
        t = pd.read_csv(DEC / "state_only" / "seed0" / "{}__state_only_{}.csv".format(scen, mode))
        so[mode] = float(t.loc[t.method == "state_only", "reward_mean"].iloc[0])
    return {"full": {"bc": float(bcf.loc[bcf.method == "bc", "reward_mean"].iloc[0]), "so": so["full"]},
            "reduced": {"bc": float(d0.loc[d0.delay_k == 0, "reward_bc_reduced"].iloc[0]), "so": so["reduced"]}}


def make(mode):
    lab = {"noisy": "no correction (noisy policy)",
           "den": "diffusion denoiser, {} obs (best $t$)".format("full local" if mode == "full" else "reduced"),
           "c1": "supervised reconstructor C1, {} obs".format("full local" if mode == "full" else "reduced"),
           "bc": "BC, {} obs (flat)".format("full local" if mode == "full" else "reduced"),
           "so": "state-only diffusion, {} obs (flat)".format("full local" if mode == "full" else "reduced")}
    den_col = "best_reward_with_diffusion_full_obs" if mode == "full" else "best_reward_with_diffusion"
    c1_col = "reward_c1_full_obs" if mode == "full" else "reward_c1_reduced_obs"
    fig, axes = plt.subplots(len(ROWS), len(COLUMNS), figsize=(FIG_W, FIG_H * len(ROWS) / 5.0), sharex="col", squeeze=False)
    for r, (scen, role, lvl, label, what) in enumerate(ROWS):
        base = RED_DIR / "{}__{}__{}".format(scen, role, lvl)
        flat = flat_values(scen, base)[mode]
        for c, (title, x_col, xlabel, pat, c1pat, noisy_col) in enumerate(COLUMNS):
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
                ylab = label if mode == "full" else "{}\n({})".format(label, what.replace(" removed", "\nremoved"))
                ax.set_ylabel(ylab, rotation=0, ha="right", va="center", labelpad=8)
            df = pd.read_csv(base / pat.format(s=scen)).sort_values(x_col)
            c1 = pd.read_csv(DEC / "c1" / c1pat.format(s=scen)).sort_values(x_col)
            assert list(c1[x_col].round(4)) == list(df[x_col].round(4))
            x = df[x_col]
            ax.plot(x, [flat["bc"]] * len(x), color=COLOR["bc"], ls=DOT, lw=1.6)
            ax.plot(x, [flat["so"]] * len(x), color=COLOR["so"], ls=DOT, lw=1.6)
            ax.plot(x, c1[c1_col], color=COLOR["c1"], ls=DOT, lw=1.6)
            ax.plot(x, df[den_col], color=COLOR["den"], ls=DOT, lw=1.6)
            ax.plot(x, df[noisy_col], color=COLOR["noisy"], ls="-", lw=1.2)
    fig.supylabel("Reward")
    fig.suptitle("{} observation{}".format("Full local" if mode == "full" else "Reduced",
                 "" if mode == "full" else " (reduced agent(s) only; every other agent keeps its full observation)"),
                 fontsize=8, fontweight="bold", y=1.0)
    handles = [Line2D([], [], color=COLOR[m], ls="-" if m == "noisy" else DOT, lw=1.8, label=lab[m]) for m in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.075),
               columnspacing=1.4, handlelength=2.6)
    fig.tight_layout(rect=(0.01, 0.008, 1, 0.985), pad=0.15)
    fig.subplots_adjust(wspace=0.30, hspace=0.18)
    name = "decentralized_{}_obs_all_scenarios".format(mode)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = OUT_DIR / "{}.{}".format(name, ext)
        fig.savefig(out, bbox_inches="tight", **kw)
        print("Saved", out)
    plt.close(fig)


for m in ("full", "reduced"):
    make(m)
