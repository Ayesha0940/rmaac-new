"""MPE corruption-shift report (Config B, seeds 0-4). Metric = REWARD only (MPE defines no success metric).
Diffusion / B_mean / BC / clean / noisy rows: existing confirm-run rows (report_confirm.load), already seen before the
predictions were registered. C1-gauss rows: <shift-dir>/parts (new), reconstructor trained only on clip(a+sigma*eps),
sigma ~ U(0,3) (train_recon method 'c2'), best of {unscaled, 8 sqrt(abar_t)-scaled} per cell and seed.
Usage: python report_shift.py [--shift-dir results/shift] [--out-prefix results/shift] [--model-dir models]
"""
import argparse
import glob
import hashlib
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report_confirm as RC
import common as C
import info_levels as IL

LEVELS = [("simple_adversary", "good", "removed", "Physical Deception, target removed"),
          ("simple_spread", "shared", "bottleneck-2", "spread bottleneck-2 (unimodal control)")]
GAUSS = [("gauss", 1.0, "gauss 1"), ("gauss", 3.0, "gauss 3")]
NONG = [("stuck", 0.5, "stuck .5"), ("stuck", 1.0, "stuck 1"), ("biased_gauss_mu+1", 1.0, "biased+1"),
        ("delay", 3.0, "delay 3"), ("delay", 8.0, "delay 8")]
NOFAULT = [("gauss", 0.0, "no fault")]
SEEDS = [0, 1, 2, 3, 4]


def ckpt_manifest(model_dir):
    out = []
    for task, varied, lv, _ in LEVELS:
        tag, info_f = C.agent_tag_and_info(task, IL.CORRECTED[(task, varied)][0], "full")
        for s in SEEDS:
            p = C.model_path("c2" + IL.SUFFIX[lv], task, info_f, tag, s)
            import torch
            ck = torch.load(p, map_location="cpu")
            with open(p, "rb") as fh:
                sha = hashlib.sha256(fh.read()).hexdigest()[:16]
            out.append(dict(task=task, level=lv, seed=s, path=p, method=ck["method"], corruption=ck["corruption"],
                            max_std=ck["max_std"], level_ck=ck["level"], sha=sha, mtime=os.path.getmtime(p)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift-dir", default=os.path.join(HERE, "results", "shift"))
    ap.add_argument("--out-prefix", default=os.path.join(HERE, "results", "shift"))
    ap.add_argument("--model-dir", default=os.path.join(HERE, "models"))
    ap.add_argument("--confirm-dir", default=os.path.join(HERE, "results", "confirm"))
    a = ap.parse_args()
    C.MODEL_DIR = a.model_dir
    d, _, _ = RC.load(a.confirm_dir)
    pf = glob.glob(os.path.join(a.shift_dir, "parts", "*.csv"))
    g = pd.concat([pd.read_csv(f).assign(_part=f, _mtime=os.path.getmtime(f)) for f in pf])
    if g.reward_mean.isnull().any() or d.reward_mean.isnull().any():
        raise SystemExit("NaN reward -- stopping")

    # ---- training-fault manifest + not-retrained assertion
    man = ckpt_manifest(a.model_dir)
    for m in man:
        assert m["method"] == "c2" and m["corruption"].startswith("clip(a+sigma*eps,-1,1)") and m["max_std"] == 3.0, m
        assert not any(k in m["corruption"].lower() for k in ("stuck", "delay", "bias")), m
    first_part = g._mtime.min()
    assert all(m["mtime"] < first_part for m in man), "a C1-gauss checkpoint is newer than the first evaluation part"
    for (task, lv, seed), x in g.groupby(["scenario", "info_level", "seed"]):
        shas = set(x.model_sha)
        want = [m["sha"] for m in man if m["task"] == task and m["level"] == lv and m["seed"] == seed]
        assert shas == set(want), (task, lv, seed, shas, want)

    rows, out = [], []
    HASH = {"checked": 0, "skipped": 0}
    P = out.append
    blocks, checks = [], []
    for task, varied, lv, name in LEVELS:
        col = RC.rcol(task)
        x = d[(d.scenario == task) & (d.varied == varied) & (d.info_level == lv)]
        y = g[(g.scenario == task) & (g.varied == varied) & (g.info_level == lv)]
        bc = {s: float(x[(x.seed == s) & (x.method == "bc")][col].iloc[0]) for s in SEEDS}
        res = OrderedDict()
        for f, l, lab in NOFAULT + GAUSS + NONG:
            per = OrderedDict((k, []) for k in ["clean", "noisy", "BC", "B_mean", "diffusion", "C1-gauss"])
            best_var = []
            for s in SEEDS:
                xs = x[(x.seed == s) & (x.fault == f) & (x.level == l)]
                ys = y[(y.seed == s) & (y.fault == f) & (y.level == l)]
                if len(ys) != 9:
                    raise SystemExit("missing C1-gauss rows: {} {} seed {} ({} rows)".format(lv, lab, s, len(ys)))
                ref = xs[xs.method.isin(["clean", "noisy", "den"])]
                if int(ys.n_episodes.iloc[0]) == int(ref.n_episodes.iloc[0]):
                    assert set(int(z) for z in ys.start_hash) <= set(int(z) for z in ref.start_hash), ("start hash mismatch", lv, lab, s)
                    fh = set(int(z) for z in xs[xs.method.isin(["noisy", "den"])].fault_state_hash.dropna())
                    assert set(int(z) for z in ys.fault_state_hash) <= fh, ("fault hash mismatch", lv, lab, s)
                    HASH["checked"] += 1
                else:
                    HASH["skipped"] += 1
                v = dict(clean=float(xs[xs.method == "clean"][col].iloc[0]), noisy=float(xs[xs.method == "noisy"][col].iloc[0]),
                         BC=bc[s], B_mean=float(xs[xs.method == "bmean"][col].max()), diffusion=float(xs[xs.method == "den"][col].max()),
                         **{"C1-gauss": float(ys[col].max())})
                b = ys.iloc[ys[col].values.argmax()]
                best_var.append("s{}:{}".format(s, "unscaled" if b["mode"] == "unscaled" else "t{}".format(int(b.t))))
                for k in per:
                    per[k].append(v[k])
                for m, val in v.items():
                    rows.append({"task": task, "level": lv, "seed": s, "cell": lab, "cell_type":
                                 "no-fault" if lab == "no fault" else ("gaussian" if (f, l, lab) in GAUSS else "non-gaussian"),
                                 "method": m + (" (best t, selected after the fact)" if m in ("diffusion", "B_mean") else
                                                " (best variant, selected after the fact)" if m == "C1-gauss" else ""),
                                 "success": "", "reward": val})
            ms = OrderedDict((k, (np.mean(v), np.std(v, ddof=1))) for k, v in per.items())
            md, sd = ms["diffusion"]
            mc, sc = ms["C1-gauss"]
            marg = max(sd, sc)
            others = [ms["noisy"], ms["BC"], ms["B_mean"], ms["C1-gauss"]]
            verdict = "WIN" if all(md - mo > max(sd, so) for mo, so in others) else (
                "LOSE" if any(mo - md > max(sd, so) for mo, so in others) else "TIE")
            res[lab] = dict(ms=ms, dc=md - mc, marg=marg, verdict=verdict, best=best_var)
        blocks.append((name, task, varied, lv, res))
        s1 = all(not (res[c[2]]["dc"] > res[c[2]]["marg"]) for c in GAUSS)
        wins = sum(res[c[2]]["dc"] > res[c[2]]["marg"] for c in NONG)
        loss = sum(-res[c[2]]["dc"] > res[c[2]]["marg"] for c in NONG)
        s2 = wins >= 4 and loss == 0
        checks.append((name, lv, s1, s2, wins, loss, [round(res[c[2]]["dc"], 2) for c in GAUSS]))

    pd.DataFrame(rows).to_csv(a.out_prefix + "_cells.csv", index=False)
    P("PREDICTION CHECK (registered in results/predictions_shift.txt; margin m = max(sd_diffusion, sd_C1-gauss), 5 seeds)")
    for name, lv, s1, s2, wins, loss, gdc in checks:
        P("  S1 Gaussian cells, {:<40} {:<7} diffusion - C1-gauss on gauss 1 / gauss 3 = {} (HELD if neither > m)".format(
            name, "HELD" if s1 else "BROKEN", " / ".join("{:+.2f}".format(z) for z in gdc)))
        P("  S2 non-Gaussian cells, {:<36} {:<7} diffusion beats C1-gauss by > m in {} of 5 cells, C1-gauss beats diffusion by > m in {} (HELD if >= 4 and 0)".format(
            name, "HELD" if s2 else "BROKEN", wins, loss))
    sp = [c for c in checks if c[1] == "bottleneck-2"][0]
    P("  S3 spread bottleneck-2 (unimodal control): {}".format("HELD" if sp[3] else "BROKEN"))
    P("")
    P("Notes: metric = reward (MPE has no success; success column empty). Diffusion, B_mean, BC, clean and noisy rows are the")
    P("existing confirm-run rows (seeds 0-4), seen before the predictions were registered; C1-gauss rows are new.")
    P("The diffusion denoiser is trained on clean actions with Gaussian forward-process noise (never sees a fault model).")
    P("C1-gauss manifest (all 10 checkpoints): method c2, corruption '{}', max_std 3.0, Gaussian-only; checkpoint hashes".format(man[0]["corruption"]))
    P("identical across all evaluation parts of the same (level, seed); all checkpoints older than the first evaluation part.")
    P("Start-state / fault hashes equal to the existing rows: checked in {} (level, cell, seed) groups, skipped (episode count differs) in {}.".format(HASH["checked"], HASH["skipped"]))
    for name, task, varied, lv, res in blocks:
        P("\n" + "=" * 126)
        P("{}   ({} / {} / {}; reward = {})".format(name, task, varied, lv,
          "good-agent reward" if task == "simple_adversary" else "summed over agents"))
        P("=" * 126)
        for title, cells in (("NO FAULT (reported, not counted)", NOFAULT), ("GAUSSIAN cells (matched to C1-gauss training)", GAUSS),
                             ("NON-GAUSSIAN cells (unmatched: C1-gauss never trained on these)", NONG)):
            P("-- " + title)
            P("{:<34}".format("") + "".join("{:>18}".format(c[2]) for c in cells))
            for k in ["clean", "noisy", "BC", "B_mean", "diffusion", "C1-gauss"]:
                lab = {"diffusion": "diffusion (best t, after the fact)", "B_mean": "B_mean (best t, after the fact)",
                       "C1-gauss": "C1-gauss (best variant, after fact)"}.get(k, k)
                P("{:<34}".format(lab) + "".join("{:>18}".format("{:.2f}±{:.2f}".format(*res[c[2]]["ms"][k])) for c in cells))
            P("{:<34}".format("diffusion - C1-gauss / margin") + "".join("{:>18}".format("{:+.2f}/{:.2f}".format(res[c[2]]["dc"], res[c[2]]["marg"])) for c in cells))
            P("{:<34}".format("v2 verdict (vs noisy,BC,B_mean,C1g)") + "".join("{:>18}".format(res[c[2]]["verdict"]) for c in cells))
            P("{:<34}".format("C1-gauss best variant per seed") + "  " + " | ".join("{}: {}".format(c[2], ",".join(res[c[2]]["best"])) for c in cells))
    txt = "\n".join(out)
    print(txt)
    with open(a.out_prefix + "_summary.txt", "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
