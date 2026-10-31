"""MPE Config-B confirmation report (seeds 0-4).  MPE defines NO success metric: every verdict, margin and G below is
computed from the reward column only (good-agent reward for simple_adversary, reward summed over agents otherwise);
the `success` column of confirm_cells.csv is left empty and is never read.

Sources: results/il.csv (seed 0 clean/noisy/den/bc), results/step2/parts (seed 0 bmean/c1; seeds 1-2 everything),
<confirm>/parts (seeds 0-2 c1lin; seeds 3-4 everything incl. c1lin; random seeds 3-4), results/gap (random seeds 0-2).
Usage: python report_confirm.py [--confirm-dir results/confirm] [--out-prefix results/confirm]
"""
import argparse
from collections import OrderedDict
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = [("gauss", 0.0, "gauss 0"), ("gauss", 1.0, "gauss 1"), ("gauss", 3.0, "gauss 3"),
         ("biased_gauss_mu+1", 1.0, "biased+1"), ("delay", 3.0, "delay 3"), ("delay", 8.0, "delay 8"),
         ("stuck", 0.5, "stuck .5"), ("stuck", 1.0, "stuck 1")]
LEVELS = [("simple_speaker_listener", "speaker", "removed", "P1 Speaker-Listener, target removed", "win"),
          ("simple_adversary", "good", "removed", "P2 Physical Deception, target removed", "win"),
          ("simple_speaker_listener", "listener", "bottleneck-1", "P3 control: SL listener bottleneck-1", "control"),
          ("simple_adversary", "good", "bottleneck-1", "P3 control: adversary bottleneck-1", "control"),
          ("simple_spread", "shared", "bottleneck-2", "P3 control: spread bottleneck-2", "control"),
          ("simple_spread", "shared", "bottleneck-4", "P3 control: spread bottleneck-4", "control")]
N_REQ = {("simple_speaker_listener", "speaker", "removed"): 3, ("simple_adversary", "good", "removed"): 4}
SEEDS = [0, 1, 2, 3, 4]
CORR = {("simple_speaker_listener", "speaker"): [0], ("simple_speaker_listener", "listener"): [1],
        ("simple_adversary", "good"): [1, 2], ("simple_spread", "shared"): [0, 1, 2]}


def rcol(task):
    return "reward_good_mean" if task == "simple_adversary" else "reward_mean"


def scol(task):
    return "reward_good_std" if task == "simple_adversary" else "reward_std"


def per_agent_mean(s, idx):
    v = [float(z) for z in str(s).split("|")]
    return float(np.mean([v[i] for i in idx]))


def load(confirm_dir):
    il = pd.read_csv(os.path.join(HERE, "results", "il.csv")).assign(seed=0)
    s2 = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(HERE, "results", "step2", "parts", "*.csv"))
                    if "_only-clean-noisy-den" not in f])
    cf = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(confirm_dir, "parts", "*.csv"))])
    gap = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(HERE, "results", "gap", "*.csv"))])
    parts = []
    for task, varied, lv, _, _ in LEVELS:
        sel = lambda d: d[(d.scenario == task) & (d.varied == varied) & (d.info_level == lv)]
        parts.append(sel(il)[sel(il).method.isin(["clean", "noisy", "den", "bc"])])
        x = sel(s2)
        parts.append(x[(x.seed == 0) & x.method.isin(["bmean", "c1"])])
        parts.append(x[x.seed.isin([1, 2])])
        y = sel(cf)
        parts.append(y[(y.seed.isin([0, 1, 2])) & (y.method == "c1lin")])
        parts.append(y[y.seed.isin([3, 4]) & (y.method != "random")])
    rnd = pd.concat([gap[gap.method == "random"], cf[cf.method == "random"]])
    return pd.concat(parts), rnd, il


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-dir", default=os.path.join(HERE, "results", "confirm"))
    ap.add_argument("--out-prefix", default=os.path.join(HERE, "results", "confirm"))
    a = ap.parse_args()
    d, rnd, il = load(a.confirm_dir)

    # ---- confirm_cells.csv (success column intentionally empty: MPE has no success metric)
    rows = []
    for _, r in d.iterrows():
        task = r.scenario
        cs = CORR[(task, r.varied)]
        rows.append({"task": task, "varied": r.varied, "level": r.info_level, "seed": int(r.seed),
                     "cell": "{} {}".format(r.fault, r.level), "fault": r.fault, "fault_level": r.level,
                     "method": r.method, "mode": r["mode"] if isinstance(r["mode"], str) else "",
                     "t": "" if pd.isnull(r.t) else int(r.t), "success": "",
                     "reward": float(r[rcol(task)]), "reward_episode_std": float(r[scol(task)]),
                     "action_mse_corrected": per_agent_mean(r.action_mse_per_agent, cs),
                     "action_cos_corrected": per_agent_mean(r.action_cos_per_agent, cs)})
    cells = pd.DataFrame(rows)
    if cells.reward.isnull().any():
        raise SystemExit("NaN reward in confirm_cells -- stopping")
    cells.to_csv(a.out_prefix + "_cells.csv", index=False)

    out = []
    P = out.append
    P("MPE Config-B confirmation run -- seeds {}; metric = REWARD (MPE defines no success; success column is empty).".format(SEEDS))
    P("den / B_mean / C1-t / C1-linear-t / reconstructor = best variant per seed, selected after the fact.")
    P("reconstructor = best of {C1 unscaled, C1-t, C1-linear unscaled, C1-linear-t} per cell and seed.")
    P("v2: WIN if den beats noisy, BC, B_mean and reconstructor each by > max(sd_den, sd_X) (sd over seeds); LOSE if any "
      "beats den by that margin; else TIE.")
    # ---------------------------------------------------------------- C1 sanity (no-fault cell), printed first
    P("\n" + "=" * 120)
    P("C1 SANITY at the no-fault cell (gauss 0), mean over seeds; success = n/a (MPE)")
    P("{:<44}{:>20}{:>20}{:>24}{:>34}".format("task / level", "BC mse / cos", "C1 mse / cos", "C1-linear mse / cos", "C1-t (best) mse / cos [t per seed]"))
    sanity = {}
    for task, varied, lv, name, _ in LEVELS:
        x = cells[(cells.task == task) & (cells.varied == varied) & (cells.level == lv) & (cells.cell == "gauss 0.0")]
        def mc(m, mode=None):
            y = x[(x.method == m) & ((x["mode"] == mode) if mode else True)]
            return y.action_mse_corrected.mean(), y.action_cos_corrected.mean()
        ct = x[(x.method == "c1") & (x["mode"] == "scaled")]
        best = ct.loc[ct.groupby("seed").reward.idxmax()]
        bc, c1u, c1l = mc("bc"), mc("c1", "unscaled"), mc("c1lin", "unscaled")
        sanity[(task, varied, lv)] = c1u[0]
        P("{:<44}{:>20}{:>20}{:>24}{:>34}".format(
            "{} / {} / {}".format(task.replace("simple_", ""), varied, lv),
            "{:.4f} / {:.3f}".format(*bc), "{:.4f} / {:.3f}".format(*c1u), "{:.4f} / {:.3f}".format(*c1l),
            "{:.4f} / {:.3f} [{}]".format(best.action_mse_corrected.mean(), best.action_cos_corrected.mean(),
                                         ",".join(str(int(t)) for t in best.t))))
    # ---------------------------------------------------------------- per level
    checks = []
    for task, varied, lv, name, kind in LEVELS:
        x = cells[(cells.task == task) & (cells.varied == varied) & (cells.level == lv)]
        P("\n" + "=" * 120)
        P("{}   ({} / {} / {}; reward = {})".format(name, task, varied, lv,
          "good-agent reward" if task == "simple_adversary" else "summed over agents"))
        P("=" * 120)
        bc_seed = {}
        for s in SEEDS:
            y = x[(x.seed == s) & (x.method == "bc")]
            if len(y) != 1:
                raise SystemExit("missing / duplicate BC row: {} seed {} ({} rows)".format(lv, s, len(y)))
            bc_seed[s] = float(y.reward.iloc[0])
        # qualifying cells: il.csv seed 0, noisy worse than clean by > 2 SE
        qual = {}
        for f, l, lab in CELLS[1:]:
            z = il[(il.scenario == task) & (il.varied == varied) & (il.info_level == lv) & (il.fault == f) & (il.level == l)]
            cl, no = z[z.method == "clean"].iloc[0], z[z.method == "noisy"].iloc[0]
            qual[lab] = (cl[rcol(task)] - no[rcol(task)]) > 2 * np.hypot(cl[scol(task)], no[scol(task)]) / np.sqrt(cl.n_episodes)
        table = OrderedDict((k, []) for k in ["clean", "noisy", "BC", "den (best t)", "B_mean (best t)", "C1", "C1-t",
                                              "C1-linear", "C1-linear-t", "reconstructor", "verdict v2",
                                              "den - recon / margin", "qualifying"])
        verdicts = {}
        for f, l, lab in CELLS:
            y = x[(x.fault == f) & (x.fault_level == l)]
            per = {k: [] for k in ["clean", "noisy", "BC", "den", "bm", "c1", "c1t", "c1l", "c1lt", "rec"]}
            for s in SEEDS:
                z = y[y.seed == s]
                def one(m, mode=None):
                    w = z[(z.method == m) & ((z["mode"] == mode) if mode else True)]
                    if len(w) == 0:
                        raise SystemExit("missing rows: {} {} seed {} method {} {}".format(lv, lab, s, m, mode))
                    return float(w.reward.max())
                vals = dict(clean=one("clean"), noisy=one("noisy"), BC=bc_seed[s], den=one("den"), bm=one("bmean"),
                            c1=one("c1", "unscaled"), c1t=one("c1", "scaled"), c1l=one("c1lin", "unscaled"),
                            c1lt=one("c1lin", "scaled"))
                vals["rec"] = max(vals["c1"], vals["c1t"], vals["c1l"], vals["c1lt"])
                for k in per:
                    per[k].append(vals[k])
            ms = dict((k, (np.mean(v), np.std(v, ddof=1))) for k, v in per.items())
            if any(np.isnan(m) or np.isnan(sd) for m, sd in ms.values()):
                raise SystemExit("NaN in {} {}".format(lv, lab))
            md, sd = ms["den"]
            others = [ms["noisy"], ms["BC"], ms["bm"], ms["rec"]]
            if all(md - mo > max(sd, so) for mo, so in others):
                v = "WIN"
            elif any(mo - md > max(sd, so) for mo, so in others):
                v = "LOSE"
            else:
                v = "TIE"
            verdicts[lab] = v
            f2 = lambda k: "{:.2f}±{:.2f}".format(*ms[k])
            for k, kk in [("clean", "clean"), ("noisy", "noisy"), ("BC", "BC"), ("den (best t)", "den"),
                          ("B_mean (best t)", "bm"), ("C1", "c1"), ("C1-t", "c1t"), ("C1-linear", "c1l"),
                          ("C1-linear-t", "c1lt"), ("reconstructor", "rec")]:
                table[k].append(f2(kk))
            table["verdict v2"].append(v if lab != "gauss 0" else v + "*")
            table["den - recon / margin"].append("{:+.2f}/{:.2f}".format(md - ms["rec"][0], max(sd, ms["rec"][1])))
            table["qualifying"].append("-" if lab == "gauss 0" else ("yes" if qual[lab] else "no"))
        P("{:<22}".format("") + "".join("{:>16}".format(c[2]) for c in CELLS))
        for k, v in table.items():
            P("{:<22}".format(k) + "".join("{:>16}".format(z) for z in v))
        P("(* gauss 0 = no fault, not counted)")
        qcells = [lab for lab in qual if qual[lab]]
        w = sum(verdicts[c] == "WIN" for c in qcells)
        t_ = sum(verdicts[c] == "TIE" for c in qcells)
        lo = sum(verdicts[c] == "LOSE" for c in qcells)
        P("v2 counts over qualifying cells ({}): WIN {} / TIE {} / LOSE {}   (all 7 faulted: WIN {} / TIE {} / LOSE {})".format(
            len(qcells), w, t_, lo, sum(verdicts[c[2]] == "WIN" for c in CELLS[1:]),
            sum(verdicts[c[2]] == "TIE" for c in CELLS[1:]), sum(verdicts[c[2]] == "LOSE" for c in CELLS[1:])))
        # G
        jr = [float(rnd[(rnd.scenario == task) & (rnd.varied == varied) & (rnd.seed == s)][rcol(task)].iloc[0]) for s in SEEDS]
        jc = np.mean([float(x[(x.seed == s) & (x.method == "clean") & (x.cell == "gauss 0.0")].reward.iloc[0]) for s in SEEDS])
        jb = np.mean([bc_seed[s] for s in SEEDS])
        G = (jc - jb) / (jc - np.mean(jr))
        P("G = [J(clean) - J(BC)] / [J(clean) - J(random)] = ({:.3f} - {:.3f}) / ({:.3f} - {:.3f}) = {:.3f}"
          "   [J(random) per seed: {}]".format(jc, jb, jc, np.mean(jr), G, ", ".join("{:.2f}".format(z) for z in jr)))
        # prediction check
        if kind == "win":
            need = N_REQ[(task, varied, lv)]
            held = w >= need and lo == 0
            line = "WIN {} of {} qualifying cells (needs >= {}), LOSE {}".format(w, len(qcells), need, lo)
            if sanity[(task, varied, lv)] < 0.01 and any(verdicts[c] != "WIN" for c in qcells):
                line += "; note: C1 no-fault MSE {:.4f} is near zero yet C1 does not lose everywhere".format(sanity[(task, varied, lv)])
        else:
            ok = 0
            for f, l, lab in CELLS[1:]:
                if qual[lab]:
                    y = x[(x.fault == f) & (x.fault_level == l)]
                    den_v = [float(y[(y.seed == s) & (y.method == "den")].reward.max()) for s in SEEDS]
                    rec_v = [float(y[(y.seed == s) & y.method.isin(["c1", "c1lin"])].reward.max()) for s in SEEDS]
                    if np.mean(den_v) - np.mean(rec_v) <= max(np.std(den_v, ddof=1), np.std(rec_v, ddof=1)):
                        ok += 1
            held = ok > len(qcells) / 2.0
            line = "reconstructor ties or beats the denoiser in {} of {} qualifying cells (needs > {:.1f})".format(ok, len(qcells), len(qcells) / 2.0)
        checks.append((name, kind, held, line))
    P("\n" + "=" * 120)
    P("PREDICTION CHECK (registered in results/predictions_confirm.txt)")
    for name, kind, held, line in checks:
        P("  {:<44} {:<7} {}".format(name, "HELD" if held else "BROKEN", line))
    ctrl = [h for n, k, h, l in checks if k == "control"]
    P("  {:<44} {:<7} {} of 4 bottleneck levels held".format("P3 control overall", "HELD" if all(ctrl) else "BROKEN", sum(ctrl)))
    txt = "\n".join(out)
    print(txt)
    with open(a.out_prefix + "_summary.txt", "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
