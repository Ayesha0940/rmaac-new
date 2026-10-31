"""Build results/report_extract_mpe.csv and results/levels_manifest_mpe.txt from EXISTING result files only
(nothing is re-run or retrained). Sources: results/il.csv, results/step2/parts, results/confirm/parts,
results/shift/parts, results/speaker_hardening/parts (the *_only-clean-noisy-den regression part and the random-policy parts excluded).
metric = reward: good-agent reward for simple_adversary, reward summed over agents otherwise.
Per (onboard_level, cell, seed): denoiser / B_mean / reconstructor = best variant (after the fact); mean / sd(ddof=1) across the
seeds that have data for that method; n_seeds reported. BC is fault-independent (its value is replicated to every cell).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
CELL = {("gauss", 0.0): ("clean", 0.0, 1), ("gauss", 1.0): ("gaussian", 1.0, 0), ("gauss", 3.0): ("gaussian", 3.0, 0),
        ("biased_gauss_mu+1", 1.0): ("bias", 1.0, 0), ("delay", 3.0): ("delay", 3.0, 0), ("delay", 8.0): ("delay", 8.0, 0),
        ("stuck", 0.5): ("stuck", 0.5, 0), ("stuck", 1.0): ("stuck", 1.0, 0)}
RECON_NAME = {("c1", "unscaled"): "C1", ("c1", "scaled"): "C1-t", ("c1lin", "unscaled"): "C1-linear",
              ("c1lin", "scaled"): "C1-linear-t", ("c1g", "unscaled"): "C1-gauss", ("c1g", "scaled"): "C1-gauss-t"}


def load():
    keys = ["scenario", "varied", "info_level", "fault", "level", "seed", "method", "mode", "t"]
    parts = [pd.read_csv(os.path.join(R, "il.csv"))]
    srcs = ["il.csv"]
    for sub in ("step2", "confirm", "shift", "speaker_hardening"):
        for f in sorted(glob.glob(os.path.join(R, sub, "parts", "*.csv"))):
            if "_only-clean-noisy-den" in f or os.path.basename(f).startswith("random_"):
                continue            # regression rerun / random-policy parts (clean + random only) are not extract sources
            parts.append(pd.read_csv(f))
            srcs.append(os.path.relpath(f, R))
    d = pd.concat(parts, ignore_index=True, sort=False)
    d = d[d.method != "random"].copy()
    d["mode"] = d["mode"].fillna("")
    d["t"] = d["t"].fillna(-1)
    dup = d.duplicated(keys, keep=False)
    if dup.any():
        raise SystemExit("duplicate rows across sources:\n" + d[dup][keys].head().to_string())
    d["reward"] = np.where(d.scenario == "simple_adversary", d.reward_good_mean, d.reward_mean)
    if d.reward.isnull().any():
        raise SystemExit("NaN reward")
    d["level_name"] = d.scenario + "/" + d.varied + "/" + d.info_level
    return d, len(srcs)


def agg(vals, notes=None):
    v = np.asarray(vals, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan")), len(v)


def main():
    d, nsrc = load()
    rows = []
    for lvl, dl in d.groupby("level_name"):
        # BC: one value per seed (fault-independent), taken from the no-fault job and replicated to every cell
        bc = dl[dl.method == "bc"].groupby("seed").reward.first()
        for (f, l), (ptype, strength, isclean) in CELL.items():
            c = dl[(dl.fault == f) & (dl.level == l)]
            if len(c) == 0:
                continue
            per = {"clean": {}, "noisy": {}, "BC": {}, "B_mean": {}, "denoiser": {}, "reconstructor": {}}
            note = {"B_mean": {}, "denoiser": {}, "reconstructor": {}}
            for s, cs in c.groupby("seed"):
                x = cs[cs.method == "clean"]
                y = cs[cs.method == "noisy"]
                if len(x) == 1 and len(y) == 1:
                    per["clean"][s], per["noisy"][s] = float(x.reward.iloc[0]), float(y.reward.iloc[0])
                if s in bc.index:
                    per["BC"][s] = float(bc[s])
                for key, meth in (("B_mean", "bmean"), ("denoiser", "den")):
                    z = cs[cs.method == meth]
                    if len(z):
                        b = z.iloc[z.reward.values.argmax()]
                        per[key][s] = float(b.reward)
                        note[key][s] = "{}@{}".format(b["mode"], int(b.t))
                z = cs[cs.method.isin(["c1", "c1lin", "c1g"])]
                if len(z):
                    b = z.iloc[z.reward.values.argmax()]
                    per["reconstructor"][s] = float(b.reward)
                    nm = RECON_NAME[(b.method, b["mode"])]
                    note["reconstructor"][s] = nm + ("@{}".format(int(b.t)) if b["mode"] == "scaled" else "")
            for m, vals in per.items():
                if not vals:
                    continue
                mean, sd, n = agg(list(vals.values()))
                nt = ";".join("s{}={}".format(int(s), v) for s, v in sorted(note[m].items())) if m in note else ""
                rows.append(dict(testbed="MPE", onboard_level=lvl, agent_count="n/a", perturbation_type=ptype,
                                 strength_value=strength, is_clean=isclean, metric="reward",
                                 method={"BC": "BC"}.get(m, m), mean=mean, sd=sd, n_seeds=n, best_variant_note=nt))
    out = pd.DataFrame(rows, columns=["testbed", "onboard_level", "agent_count", "perturbation_type", "strength_value",
                                      "is_clean", "metric", "method", "mean", "sd", "n_seeds", "best_variant_note"])
    if out["mean"].isnull().any():
        raise SystemExit("NaN mean in extract")
    order = {"clean": 0, "gaussian": 1, "bias": 2, "delay": 3, "stuck": 4}
    morder = {"clean": 0, "noisy": 1, "BC": 2, "B_mean": 3, "denoiser": 4, "reconstructor": 5}
    out["_p"] = out.perturbation_type.map(order)
    out["_m"] = out.method.map(morder)
    out = out.sort_values(["onboard_level", "_p", "strength_value", "_m"]).drop(["_p", "_m"], axis=1)
    out.to_csv(os.path.join(R, "report_extract_mpe.csv"), index=False)

    # ------------------------------------------------------------------ manifest
    L = []
    P = L.append
    P("levels_manifest_mpe.txt -- built from report_extract_mpe.csv ({} rows) and {} existing result files; nothing re-run.".format(len(out), nsrc))
    P("metric = reward (good-agent reward for simple_adversary, summed over agents otherwise); sd = across-seed sd (blank when n_seeds = 1).")
    P("\nDISTINCT onboard_levels ({}), with the methods present and the seeds available:".format(out.onboard_level.nunique()))
    for lvl, g in out.groupby("onboard_level"):
        ms = [m for m in ["clean", "noisy", "BC", "B_mean", "denoiser", "reconstructor"] if m in set(g.method)]
        P("  {:<52} n_seeds={}  methods={}".format(lvl, sorted(set(g.n_seeds)), ",".join(ms)))
    P("\nPERTURBATION TYPES and strength values (sorted), with suggested tag:")
    for pt in ["clean", "gaussian", "bias", "delay", "stuck"]:
        vals = sorted(out[out.perturbation_type == pt].strength_value.unique())
        if pt == "clean":
            tags = "{} (no fault)".format(vals)
        elif len(vals) == 1:
            tags = "{} -> single value (mild = severe)".format(vals)
        else:
            tags = "{} -> mild = {}, severe = {}".format(vals, vals[0], vals[-1])
        unit = {"gaussian": "action-noise std", "bias": "mean offset +1 with std 1 (fixed)", "delay": "steps k", "stuck": "probability p per dim"}.get(pt, "")
        P("  {:<9} {}   [{}]".format(pt, tags, unit))
    # ------------------------------------------------------------------ WINs over the reconstructor
    P("\nWINS OVER THE RECONSTRUCTOR per onboard_level (7 faulted cells; WIN if mean(den) - mean(rec) > max(sd_den, sd_rec),")
    P("LOSE if mean(rec) - mean(den) > that margin, else TIE). Rankable = reconstructor present and n_seeds >= 3.")
    stats = []
    for lvl, g in out.groupby("onboard_level"):
        den, rec = g[g.method == "denoiser"], g[g.method == "reconstructor"]
        if len(rec) == 0 or g[g.method == "reconstructor"].n_seeds.min() < 3:
            stats.append((lvl, None))
            continue
        w = t = lo = 0
        marg = []
        for _, r in den[den.is_clean == 0].iterrows():
            q = rec[(rec.perturbation_type == r.perturbation_type) & (rec.strength_value == r.strength_value)].iloc[0]
            m = max(r.sd, q.sd)
            diff = r["mean"] - q["mean"]
            if diff > m:
                w += 1
            elif -diff > m:
                lo += 1
            else:
                t += 1
            marg.append(diff)
        stats.append((lvl, (w, t, lo, float(np.mean(marg)), int(rec.n_seeds.min()), sorted(set(rec.best_variant_note.str.split(";").explode().str.split("=").str[1]))[:0])))
    rk = sorted([s for s in stats if s[1] is not None], key=lambda s: (-s[1][0], s[1][2], -s[1][3]))
    P("{:<52}{:>5}{:>5}{:>6}{:>10}{:>9}".format("onboard_level", "WIN", "TIE", "LOSE", "mean(d-r)", "n_seeds"))
    for lvl, (w, t, lo, md, n, _) in rk:
        P("{:<52}{:>5}{:>5}{:>6}{:>10.2f}{:>9}".format(lvl, w, t, lo, md, n))
    na = [s[0] for s in stats if s[1] is None]
    if na:
        P("not rankable (no reconstructor rows or n_seeds < 3): " + "; ".join(na))
    top = [s[0] for s in rk[:3]]
    P("\nSUGGESTED TO FEATURE (most WINs over the reconstructor, ties broken by fewer LOSEs then larger mean margin): " + "; ".join(top))
    P("Reconstructor family per level (best-of over what exists): C1, C1-t where available; C1-linear, C1-linear-t for the six confirm-run levels;")
    P("C1-gauss for adversary/good/removed, spread/shared/bottleneck-2 and speaker_listener/speaker/removed. Levels differ in family size.")
    open(os.path.join(R, "levels_manifest_mpe.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nrows:", len(out), "| levels:", out.onboard_level.nunique())


if __name__ == "__main__":
    main()
