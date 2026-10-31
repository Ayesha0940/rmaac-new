"""Report for the information-level sweep. Reads results/il.csv (or argv[1]); writes results/il_report.txt.

Per (task, varied agent): rows = information levels, columns = faults. Denoiser cell = best reward over the 16 variants
(t in {5,..,99} x {1step, chain}), "best t (selected after the fact, not deployable)", shown as value [t, mode].
RECONSTRUCTS: among the 7 faulted cells where noisy is worse than clean by > 2 SE, the best-t denoiser recovers >= half of
  (clean - noisy) AND best t < 60, in MORE THAN HALF of those cells.
NEEDS-ACTION: BC (fault-independent) < clean - 0.2 * (clean - worst noisy reward over the 7 faulted cells).
Reward: good-agent reward for simple_adversary, summed over agents otherwise. SE = reward_std / sqrt(n_episodes).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import info_levels as IL

CELLS = [("gauss", 0.0, "gauss 0"), ("gauss", 1.0, "gauss 1"), ("gauss", 3.0, "gauss 3"),
         ("biased_gauss_mu+1", 1.0, "biased+1"), ("delay", 3.0, "delay 3"), ("delay", 8.0, "delay 8"),
         ("stuck", 0.5, "stuck .5"), ("stuck", 1.0, "stuck 1")]
GROUPS = [("simple_speaker_listener", "speaker"), ("simple_speaker_listener", "listener"),
          ("simple_adversary", "good"), ("simple_spread", "shared")]
RECOVER, BEST_T_MAX, NEEDS_FRAC = 0.5, 60, 0.2


def cols(task):
    return ("reward_good_mean", "reward_good_std") if task == "simple_adversary" else ("reward_mean", "reward_std")


def cell_rows(d, task, varied, level, cell):
    return d[(d.scenario == task) & (d.varied == varied) & (d.info_level == level) & (d.fault == cell[0]) & (d.level == cell[1])]


def pick(rows, method):
    x = rows[rows.method == method]
    return None if len(x) == 0 else x.iloc[0]


def mse_mean(s, idx):
    v = [float(z) for z in str(s).split("|")]
    return float(np.mean([v[i] for i in idx]))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "il.csv")
    d = pd.read_csv(path)
    out = []
    P = out.append
    summary = []
    for task, varied in GROUPS:
        t = d[(d.scenario == task) & (d.varied == varied)]
        if len(t) == 0:
            continue
        mc, sc = cols(task)
        levels = [lv for lv in IL.LEVELS_USED[(task, varied)] if lv in set(t.info_level)]
        CS = IL.CORRECTED[(task, varied)]
        P("=" * 130)
        P("TASK {}   varied agent: {}   corrected+faulted agents: {}   reward = {}".format(
            task, varied, CS, "good-agent reward" if task == "simple_adversary" else "summed over agents"))
        P("=" * 130)
        # reference rows (identical across levels; taken from the first level that has the cell)
        ref = {}
        for cell in CELLS:
            for lv in levels:
                rr = cell_rows(t, task, varied, lv, cell)
                if len(rr):
                    ref[cell] = (pick(rr, "clean"), pick(rr, "noisy"))
                    break
        clean = list(ref.values())[0][0]
        hdr = "{:<16}".format("level") + "".join("{:>17}".format(c[2]) for c in CELLS) + "{:>12}".format("BC")
        P(hdr)
        P("{:<16}".format("clean policy") + "{:>17}".format("{:.1f}".format(clean[mc])))
        P("{:<16}".format("noisy policy") + "".join("{:>17}".format("{:.1f}".format(ref[c][1][mc]) if c in ref else "-") for c in CELLS))
        worst_noisy = min(ref[c][1][mc] for c in CELLS[1:] if c in ref)
        P("")
        P("denoiser = best t (selected after the fact, not deployable): value [t, mode: s=1step c=chain]")
        labels = []
        sec = {"mse": [], "cos": [], "acc": []}
        for lv in levels:
            line = "{:<16}".format(lv)
            bc_row = None
            rec_ok, qual = 0, 0
            for k, cell in enumerate(CELLS):
                rr = cell_rows(t, task, varied, lv, cell)
                if len(rr) == 0:
                    line += "{:>17}".format("-")
                    continue
                if pick(rr, "bc") is not None:
                    bc_row = pick(rr, "bc")
                den = rr[rr.method == "den"]
                bi = den[mc].values.argmax()
                b = den.iloc[bi]
                noisy = pick(rr, "noisy")
                line += "{:>17}".format("{:.1f} [{},{}]".format(b[mc], int(b.t), "c" if b["mode"] == "chain" else "s"))
                sec["mse"].append((lv, cell[2], mse_mean(b.action_mse_per_agent, CS), mse_mean(noisy.action_mse_per_agent, CS)))
                sec["cos"].append((lv, cell[2], mse_mean(b.action_cos_per_agent, CS), mse_mean(noisy.action_cos_per_agent, CS)))
                if task == "simple_speaker_listener" and varied == "speaker":
                    sec["acc"].append((lv, cell[2], float(b.msg_acc), float(noisy.msg_acc)))
                if k > 0:
                    cl = ref[cell][0]
                    gap = cl[mc] - noisy[mc]
                    se = np.sqrt((cl[sc] ** 2 + noisy[sc] ** 2) / cl["n_episodes"])
                    if gap > 2 * se:
                        qual += 1
                        if (b[mc] - noisy[mc]) >= RECOVER * gap and b.t < BEST_T_MAX:
                            rec_ok += 1
            bcv = None if bc_row is None else bc_row[mc]
            line += "{:>12}".format("-" if bcv is None else "{:.1f}".format(bcv))
            P(line)
            reconstructs = qual > 0 and rec_ok > qual / 2.0
            thr = clean[mc] - NEEDS_FRAC * (clean[mc] - worst_noisy)
            needs = bcv is not None and bcv < thr
            labels.append((lv, reconstructs, rec_ok, qual, needs, bcv, thr))
        P("")
        P("{:<16}{:>16}{:>30}".format("level", "RECONSTRUCTS", "NEEDS-ACTION (BC < threshold)"))
        for lv, rc, ok, q, na, bcv, thr in labels:
            P("{:<16}{:>16}{:>30}".format(lv, "{} ({}/{} cells)".format("YES" if rc else "no", ok, q),
                                          "{} (BC {:.1f} vs {:.1f})".format("YES" if na else "no", bcv, thr) if bcv is not None else "-"))
        both = [lv for lv, rc, ok, q, na, bcv, thr in labels if rc and na]
        P("\nCANDIDATE NARROW SET ({} / {}): {}".format(task, varied, ", ".join(both) if both else "none"))
        summary.append((task, varied, both))
        for nm, rows, title in (("mse", sec["mse"], "Action MSE vs clean (mean over corrected agents): best-t denoiser / noisy"),
                                ("cos", sec["cos"], "Action cosine vs clean (mean over corrected agents): best-t denoiser / noisy"),
                                ("acc", sec["acc"], "Message accuracy (speaker): best-t denoiser / noisy")):
            if not rows:
                continue
            P("\n" + title)
            P("{:<16}".format("level") + "".join("{:>17}".format(c[2]) for c in CELLS))
            for lv in levels:
                line = "{:<16}".format(lv)
                for cell in CELLS:
                    x = [r for r in rows if r[0] == lv and r[1] == cell[2]]
                    line += "{:>17}".format("-" if not x else "{:.3f}/{:.3f}".format(x[0][2], x[0][3]))
                P(line)
            if nm == "mse":
                P("BC action MSE / cosine (corrected agents), fault-independent:")
                for lv in levels:
                    br = None
                    for cc in CELLS:
                        br = br if br is not None else pick(cell_rows(t, task, varied, lv, cc), "bc")
                    if br is not None:
                        P("  {:<14} mse {:.3f}  cos {:.3f}".format(lv, mse_mean(br.action_mse_per_agent, CS), mse_mean(br.action_cos_per_agent, CS)))
        P("")
    P("=" * 130)
    P("SUMMARY: levels that are both RECONSTRUCTS and NEEDS-ACTION")
    for task, varied, both in summary:
        P("  {:<26} {:<9}: {}".format(task, varied, ", ".join(both) if both else "none"))
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(HERE, "results", "il_report.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
