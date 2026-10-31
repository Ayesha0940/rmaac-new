"""Step 2 report: denoiser vs B_mean vs C1 vs BC on the 7 candidate levels, seeds 0-2.
Seed-0 denoiser / BC / clean / noisy rows come from results/il.csv (reused); everything else from results/step2/parts.
Rewards: good-agent reward for simple_adversary, summed over agents otherwise. sd = across-seed sample sd (ddof=1).
Verdict per cell: WIN if mean(den) - mean(X) > max(sd_den, sd_X) for every X in {BC, B_mean, C1};
LOSE if some X beats the denoiser by that margin; otherwise TIE.
  (A) per-seed best (best variant chosen per seed, after the fact), seeds 0-2
  (B) selected on seed 0: the variant that won on seed 0, evaluated on seeds 1-2 only (BC, C1 on seeds 1-2 too)
C1 = the better (by mean) of C1-unscaled and C1-scaled (scaled: best over 8 scalings per seed in (A), seed-0 choice in (B)).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = [("gauss", 0.0, "gauss 0"), ("gauss", 1.0, "gauss 1"), ("gauss", 3.0, "gauss 3"),
         ("biased_gauss_mu+1", 1.0, "biased+1"), ("delay", 3.0, "delay 3"), ("delay", 8.0, "delay 8"),
         ("stuck", 0.5, "stuck .5"), ("stuck", 1.0, "stuck 1")]
CANDS = [("simple_speaker_listener", "speaker", "removed"), ("simple_speaker_listener", "listener", "bottleneck-1"),
         ("simple_adversary", "good", "removed"), ("simple_adversary", "good", "bottleneck-1"),
         ("simple_spread", "shared", "coarse-4"), ("simple_spread", "shared", "bottleneck-2"),
         ("simple_spread", "shared", "bottleneck-4")]


def mcol(task):
    return "reward_good_mean" if task == "simple_adversary" else "reward_mean"


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def verdict(den, others):
    """den, others: (mean, sd). returns WIN / LOSE / TIE"""
    md, sd = den
    if all(md - mo > max(sd, so) for mo, so in others):
        return "WIN"
    if any(mo - md > max(sd, so) for mo, so in others):
        return "LOSE"
    return "TIE"


def load():
    il = pd.read_csv(os.path.join(HERE, "results", "il.csv"))
    parts = [f for f in glob.glob(os.path.join(HERE, "results", "step2", "parts", "*.csv")) if "_only-clean-noisy-den" not in f]
    s2 = pd.concat([pd.read_csv(f) for f in parts]) if parts else il.iloc[:0]
    keep = []
    for task, varied, lv in CANDS:
        keep.append(il[(il.scenario == task) & (il.varied == varied) & (il.info_level == lv) & (il.method.isin(["clean", "noisy", "den", "bc"]))].assign(seed=0))
    return pd.concat(keep + [s2]), il


def main():
    d, il = load()
    out = []
    P = out.append
    counts = []
    for task, varied, lv in CANDS:
        c = mcol(task)
        t = d[(d.scenario == task) & (d.varied == varied) & (d.info_level == lv)]
        if len(t) == 0:
            continue
        seeds = sorted(t.seed.unique())
        P("=" * 150)
        P("CANDIDATE {} / {} / {}   seeds present {}   reward = {}".format(task, varied, lv, seeds,
          "good-agent reward" if task == "simple_adversary" else "summed over agents"))
        P("=" * 150)
        bc = {}
        for s in seeds:
            x = t[(t.seed == s) & (t.method == "bc")]
            if len(x):
                bc[s] = float(x[c].iloc[0])
        rows = {k: [] for k in ["clean", "noisy", "BC", "den best", "den sel-s0", "B_mean best", "B_mean sel-s0",
                                "C1 unscaled", "C1 scaled best", "C1 scaled sel-s0", "C1 (better)", "C1 (better) sel",
                                "verdict A", "verdict B", "den best t", "B_mean best t"]}
        vA, vB = [], []
        for cell in CELLS:
            x = t[(t.fault == cell[0]) & (t.level == cell[1])]
            per = {}
            for s in seeds:
                xs = x[x.seed == s]
                g = lambda m: xs[xs.method == m]
                den, bm, c1 = g("den"), g("bmean"), g("c1")
                per[s] = dict(
                    clean=float(g("clean")[c].iloc[0]) if len(g("clean")) else np.nan,
                    noisy=float(g("noisy")[c].iloc[0]) if len(g("noisy")) else np.nan,
                    den=den, bm=bm,
                    c1u=float(c1[c1["mode"] == "unscaled"][c].iloc[0]) if len(c1[c1["mode"] == "unscaled"]) else np.nan,
                    c1s=c1[c1["mode"] == "scaled"])
            def best(df):
                if len(df) == 0:
                    return np.nan, ""
                b = df.iloc[df[c].values.argmax()]
                return float(b[c]), "{}{}".format(int(b.t), "c" if b["mode"] == "chain" else ("s" if b["mode"] == "1step" else ""))
            def sel(df0, dfs, key_cols):
                if len(df0) == 0:
                    return np.nan
                b = df0.iloc[df0[c].values.argmax()]
                m = dfs
                for k in key_cols:
                    m = m[m[k] == b[k]]
                return float(m[c].iloc[0]) if len(m) else np.nan
            A_seeds = [s for s in seeds if len(per[s]["den"]) and len(per[s]["bm"]) and s in bc]
            B_seeds = [s for s in A_seeds if s != 0]
            den_b = [best(per[s]["den"])[0] for s in A_seeds]
            bm_b = [best(per[s]["bm"])[0] for s in A_seeds]
            c1s_b = [best(per[s]["c1s"])[0] for s in A_seeds]
            c1u = [per[s]["c1u"] for s in A_seeds]
            bcA = [bc[s] for s in A_seeds]
            have0 = 0 in A_seeds
            den_sel = [sel(per[0]["den"], per[s]["den"], ["t", "mode"]) for s in B_seeds] if have0 else []
            bm_sel = [sel(per[0]["bm"], per[s]["bm"], ["t", "mode"]) for s in B_seeds] if have0 else []
            c1s_sel = [sel(per[0]["c1s"], per[s]["c1s"], ["t"]) for s in B_seeds] if have0 else []
            c1u_B = [per[s]["c1u"] for s in B_seeds]
            bcB = [bc[s] for s in B_seeds]
            fm = lambda v: "-" if len(v) == 0 or np.any(np.isnan(v)) else "{:.1f}±{:.1f}".format(*ms(v))
            rows["clean"].append(fm([per[s]["clean"] for s in seeds]))
            rows["noisy"].append(fm([per[s]["noisy"] for s in seeds]))
            rows["BC"].append(fm(bcA))
            rows["den best"].append(fm(den_b)); rows["den sel-s0"].append(fm(den_sel))
            rows["B_mean best"].append(fm(bm_b)); rows["B_mean sel-s0"].append(fm(bm_sel))
            rows["C1 unscaled"].append(fm(c1u)); rows["C1 scaled best"].append(fm(c1s_b)); rows["C1 scaled sel-s0"].append(fm(c1s_sel))
            c1A = max([ms(c1u), ms(c1s_b)], key=lambda z: z[0]) if len(A_seeds) else None
            c1B = max([ms(c1u_B), ms(c1s_sel)], key=lambda z: z[0]) if len(B_seeds) and have0 else None
            rows["C1 (better)"].append("-" if c1A is None else "{:.1f}±{:.1f}".format(*c1A))
            rows["C1 (better) sel"].append("-" if c1B is None else "{:.1f}±{:.1f}".format(*c1B))
            rows["den best t"].append(" ".join("s{}:{}".format(s, best(per[s]["den"])[1]) for s in A_seeds))
            rows["B_mean best t"].append(" ".join("s{}:{}".format(s, best(per[s]["bm"])[1]) for s in A_seeds))
            a = verdict(ms(den_b), [ms(bcA), ms(bm_b), c1A]) if len(A_seeds) >= 2 else "-"
            b = verdict(ms(den_sel), [ms(bcB), ms(bm_sel), c1B]) if len(B_seeds) >= 2 and have0 else "-"
            rows["verdict A"].append(a); rows["verdict B"].append(b)
            if cell != CELLS[0]:
                vA.append(a); vB.append(b)
        P("{:<18}".format("") + "".join("{:>15}".format(cl[2]) for cl in CELLS))
        for k in ["clean", "noisy", "BC", "den best", "B_mean best", "C1 unscaled", "C1 scaled best", "C1 (better)",
                  "verdict A", "den sel-s0", "B_mean sel-s0", "C1 scaled sel-s0", "C1 (better) sel", "verdict B"]:
            if k in ("verdict A", "den sel-s0"):
                P("")
            P("{:<18}".format(k) + "".join("{:>15}".format(v) for v in rows[k]))
        P("\nper-seed best variant (t + s=1step / c=chain):")
        for k in ["den best t", "B_mean best t"]:
            P("  {:<14}".format(k) + " | ".join("{}: {}".format(cl[2], v) for cl, v in zip(CELLS, rows[k])))
        cA = dict((z, vA.count(z)) for z in ("WIN", "TIE", "LOSE"))
        cB = dict((z, vB.count(z)) for z in ("WIN", "TIE", "LOSE"))
        P("\nCOUNTS over the 7 faulted cells: (A) per-seed best, seeds 0-2: WIN {WIN} / TIE {TIE} / LOSE {LOSE};  "
          "(B) selected on seed 0, seeds 1-2: WIN {0[WIN]} / TIE {0[TIE]} / LOSE {0[LOSE]}".format(cB, **cA))
        counts.append((task, varied, lv, cA, cB))
        P("")
    P("=" * 150)
    P("SUMMARY (7 faulted cells per candidate)")
    P("{:<26}{:<10}{:<14}{:>28}{:>34}".format("task", "varied", "level", "(A) per-seed best W/T/L", "(B) selected on seed 0 W/T/L"))
    for task, varied, lv, a, b in counts:
        P("{:<26}{:<10}{:<14}{:>28}{:>34}".format(task, varied, lv, "{}/{}/{}".format(a["WIN"], a["TIE"], a["LOSE"]),
                                                  "{}/{}/{}".format(b["WIN"], b["TIE"], b["LOSE"])))
    # regression check (seed-0 cell rerun in full vs il.csv)
    reg = glob.glob(os.path.join(HERE, "results", "step2", "parts", "*_only-clean-noisy-den.csv"))
    for f in reg:
        r = pd.read_csv(f)
        o = il[(il.scenario == r.scenario.iloc[0]) & (il.varied == r.varied.iloc[0]) & (il.info_level == r.info_level.iloc[0])
               & (il.fault == r.fault.iloc[0]) & (il.level == r.level.iloc[0]) & (il.method.isin(["clean", "noisy", "den"]))]
        k = ["method", "mode", "t"]
        m = r.merge(o, on=k, suffixes=("_new", "_il"))
        P("\nREGRESSION ({}): {} rows matched, reward identical: {}, action MSE identical: {}".format(
            os.path.basename(f), len(m), bool((m.reward_mean_new == m.reward_mean_il).all()),
            bool((m.action_mse_per_agent_new == m.action_mse_per_agent_il).all())))
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(HERE, "results", "step2_report.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
