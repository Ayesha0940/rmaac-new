"""Speaker-Listener reconstructor-hardening report (Config B, reward only; MPE has no success metric).
Reused rows (NOT recomputed): diffusion, BC, B_mean, clean, noisy, C1, C1-t, C1-linear, C1-linear-t for SL speaker
'removed' via report_confirm.load(). New: C1-gauss parts in <hard-dir>/parts.
Usage: python report_speaker_hardening.py [--hard-dir results/speaker_hardening] [--out results/speaker_hardening_summary.txt]
                                         [--model-dir models]
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

TASK, VARIED, LV = "simple_speaker_listener", "speaker", "removed"
CELLS = [("gauss", 0.0, "no fault"), ("gauss", 1.0, "gauss 1"), ("gauss", 3.0, "gauss 3"),
         ("biased_gauss_mu+1", 1.0, "biased+1"), ("delay", 3.0, "delay 3"), ("delay", 8.0, "delay 8"),
         ("stuck", 0.5, "stuck .5"), ("stuck", 1.0, "stuck 1")]
QUAL = ["gauss 1", "gauss 3", "biased+1"]
SEEDS = [0, 1, 2, 3, 4]
RECON = ["C1", "C1-t", "C1-linear", "C1-linear-t", "C1-gauss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-dir", default=os.path.join(HERE, "results", "speaker_hardening"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "speaker_hardening_summary.txt"))
    ap.add_argument("--model-dir", default=os.path.join(HERE, "models"))
    a = ap.parse_args()
    pred = os.path.join(HERE, "results", "predictions_speaker_hardening.txt")
    t_pred = os.path.getmtime(pred)

    # ---- reused rows: loaded from existing files, all older than the registered prediction (never recomputed)
    src = [os.path.join(HERE, "results", "il.csv")]
    src += glob.glob(os.path.join(HERE, "results", "step2", "parts", "simple_speaker_listener_speaker_removed_*.csv"))
    src += glob.glob(os.path.join(HERE, "results", "confirm", "parts", "*_simple_speaker_listener_speaker_removed_*.csv"))
    newest_src = max(os.path.getmtime(f) for f in src)
    assert newest_src < t_pred, "a reused source file is newer than the registered prediction"
    d, _, _ = RC.load(os.path.join(HERE, "results", "confirm"))
    d = d[(d.scenario == TASK) & (d.varied == VARIED) & (d.info_level == LV)]
    pf = glob.glob(os.path.join(a.hard_dir, "parts", "*.csv"))
    g = pd.concat([pd.read_csv(f).assign(_mtime=os.path.getmtime(f)) for f in pf])
    assert set(g.method) == {"c1g"}, "new parts must contain only C1-gauss rows (nothing else recomputed)"
    if d.reward_mean.isnull().any() or g.reward_mean.isnull().any():
        raise SystemExit("NaN reward -- stopping")

    # ---- C1-gauss manifest + not-retrained check
    C.MODEL_DIR = a.model_dir
    import torch
    shas = {}
    for s in SEEDS:
        p = C.model_path("c2" + IL.SUFFIX[LV], TASK, "full", "speaker", s)
        ck = torch.load(p, map_location="cpu")
        assert ck["method"] == "c2" and ck["corruption"].startswith("clip(a+sigma*eps,-1,1)") and ck["max_std"] == 3.0
        assert ck["level"] == LV and ck["seed"] == s
        assert os.path.getmtime(p) < g._mtime.min()
        with open(p, "rb") as fh:
            shas[s] = hashlib.sha256(fh.read()).hexdigest()[:16]
        manifest = ck["corruption"]
    for s, x in g.groupby("seed"):
        assert set(x.model_sha) == {shas[s]}, ("checkpoint hash mismatch", s)

    res = OrderedDict()
    nhash = 0
    nskip = [0]
    for f, l, lab in CELLS:
        per = OrderedDict((k, []) for k in ["clean", "noisy", "BC", "B_mean", "diffusion"] + RECON + ["reconstructor"])
        bestname, c1g_var = [], []
        for s in SEEDS:
            xs = d[(d.seed == s) & (d.fault == f) & (d.level == l)]
            ys = g[(g.seed == s) & (g.fault == f) & (g.level == l)]
            if len(ys) != 9:
                raise SystemExit("missing C1-gauss rows: {} seed {} ({} rows)".format(lab, s, len(ys)))
            ref = xs[xs.method.isin(["clean", "noisy", "den"])]
            if int(ys.n_episodes.iloc[0]) == int(ref.n_episodes.iloc[0]):
                assert set(int(z) for z in ys.start_hash) <= set(int(z) for z in ref.start_hash), ("start hash", lab, s)
                assert set(int(z) for z in ys.fault_state_hash) <= set(int(z) for z in xs[xs.method.isin(["noisy", "den"])].fault_state_hash.dropna()), ("fault hash", lab, s)
                nhash += 1
            else:
                nskip[0] += 1
            bc = d[(d.seed == s) & (d.method == "bc")].reward_mean
            assert len(bc) == 1
            q = lambda m, mode=None: float(xs[(xs.method == m) & ((xs["mode"] == mode) if mode else True)].reward_mean.max())
            v = OrderedDict([("clean", q("clean")), ("noisy", q("noisy")), ("BC", float(bc.iloc[0])), ("B_mean", q("bmean")),
                             ("diffusion", q("den")), ("C1", q("c1", "unscaled")), ("C1-t", q("c1", "scaled")),
                             ("C1-linear", q("c1lin", "unscaled")), ("C1-linear-t", q("c1lin", "scaled")),
                             ("C1-gauss", float(ys.reward_mean.max()))])
            if any(np.isnan(z) for z in v.values()):
                raise SystemExit("NaN / missing reused row: {} seed {}".format(lab, s))
            b = ys.iloc[ys.reward_mean.values.argmax()]
            c1g_var.append("unscaled" if b["mode"] == "unscaled" else "t{}".format(int(b.t)))
            rn = max(RECON, key=lambda k: v[k])
            v["reconstructor"] = v[rn]
            bestname.append(rn)
            for k in per:
                per[k].append(v[k])
        ms = OrderedDict((k, (np.mean(z), np.std(z, ddof=1))) for k, z in per.items())
        md, sd = ms["diffusion"]
        oth = [ms["noisy"], ms["BC"], ms["B_mean"], ms["reconstructor"]]
        ver = "WIN" if all(md - mo > max(sd, so) for mo, so in oth) else (
            "LOSE" if any(mo - md > max(sd, so) for mo, so in oth) else "TIE")
        res[lab] = dict(ms=ms, ver=ver, best=bestname, c1gv=c1g_var,
                        dr=md - ms["reconstructor"][0], mr=max(sd, ms["reconstructor"][1]),
                        dg=md - ms["C1-gauss"][0], mg=max(sd, ms["C1-gauss"][1]))

    w = sum(res[c]["ver"] == "WIN" for c in QUAL)
    lo = sum(res[c]["ver"] == "LOSE" for c in QUAL)
    over = [c for c in QUAL if -res[c]["dg"] > res[c]["mg"]]
    out = []
    P = out.append
    P("PREDICTION CHECK (registered in results/predictions_speaker_hardening.txt)")
    P("  Speaker win vs expanded reconstructor {{C1, C1-t, C1-linear, C1-linear-t, C1-gauss}}: {}  -- WIN {} / TIE {} / LOSE {} of 3 qualifying cells (HELD if WIN >= 2 and LOSE 0)".format(
        "HELD" if (w >= 2 and lo == 0) else "BROKEN", w, 3 - w - lo, lo))
    P("  Overturn check (C1-gauss beats diffusion by > margin in any qualifying cell): {}".format(
        "OVERTURNED in " + ", ".join(over) if over else "not overturned (C1-gauss beats diffusion in 0 of 3 qualifying cells)"))
    P("")
    P("Notes: metric = reward summed over agents (MPE has no success metric). Only the speaker is faulted and corrected.")
    P("Reused rows (diffusion, BC, B_mean, clean, noisy, C1, C1-t, C1-linear, C1-linear-t) were loaded from {} existing files, all older".format(len(src)))
    P("than the registered prediction; the new parts contain only C1-gauss rows. Start-state / fault hashes of every C1-gauss row equal the")
    P("reused rows of the same cell and seed ({} cell-seed groups checked, {} skipped because the episode count differs).".format(nhash, nskip[0]))
    P("C1-gauss manifest (5 checkpoints): method c2, corruption '{}', max_std 3.0, Gaussian-only; checkpoint".format(manifest))
    P("hashes identical across all evaluation parts of a seed; all checkpoints older than the first evaluation part.")
    P("diffusion / B_mean / C1-t / C1-linear-t / C1-gauss / reconstructor = best variant per seed, selected after the fact.")
    P("")
    P("{:<30}".format("") + "".join("{:>15}".format(c[2]) for c in CELLS))
    for k in ["clean", "noisy", "BC", "B_mean", "diffusion"] + RECON + ["reconstructor"]:
        P("{:<30}".format(k + (" (best-of-5)" if k == "reconstructor" else "")) +
          "".join("{:>15}".format("{:.2f}±{:.2f}".format(*res[c[2]]["ms"][k])) for c in CELLS))
    P("{:<30}".format("diffusion - recon / margin") + "".join("{:>15}".format("{:+.2f}/{:.2f}".format(res[c[2]]["dr"], res[c[2]]["mr"])) for c in CELLS))
    P("{:<30}".format("diffusion - C1-gauss / margin") + "".join("{:>15}".format("{:+.2f}/{:.2f}".format(res[c[2]]["dg"], res[c[2]]["mg"])) for c in CELLS))
    P("{:<30}".format("v2 verdict") + "".join("{:>15}".format(res[c[2]]["ver"] + ("" if c[2] in QUAL else "*")) for c in CELLS))
    P("(* not a qualifying cell; qualifying = gauss 1, gauss 3, biased+1)")
    P("\nBest reconstructor variant per seed (s0..s4):")
    for c in CELLS:
        P("  {:<10} {}   | C1-gauss best variant: {}".format(c[2], ", ".join(res[c[2]]["best"]), ", ".join(res[c[2]]["c1gv"])))
    txt = "\n".join(out)
    print(txt)
    with open(a.out, "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
