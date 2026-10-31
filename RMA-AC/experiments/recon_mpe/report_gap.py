"""Gap G = [J(clean) - J(BC)] / [J(clean) - J(random)] per information level.
J(clean), J(BC) from results/il.csv (seed 0); J(random) = mean over seeds 0-2 from results/gap/*.csv (same protocol).
"random" = uniform over the projection bounds [-1, 1] per action dim for the protocol's agents (the policy's own
actions lie in [0, 1]); other agents act on the clean policy; no fault.
For the 7 Step-2 levels also G3 from 3-seed means of clean / BC (Step 2 seeds 1-2 + il seed 0) / random.
Adversary: good-agent reward; otherwise reward summed over agents."""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import info_levels as IL

GROUPS = [("simple_speaker_listener", "speaker"), ("simple_speaker_listener", "listener"),
          ("simple_adversary", "good"), ("simple_spread", "shared")]
STEP2 = {("simple_speaker_listener", "speaker", "removed"), ("simple_speaker_listener", "listener", "bottleneck-1"),
         ("simple_adversary", "good", "removed"), ("simple_adversary", "good", "bottleneck-1"),
         ("simple_spread", "shared", "coarse-4"), ("simple_spread", "shared", "bottleneck-2"),
         ("simple_spread", "shared", "bottleneck-4")}


def col(task):
    return "reward_good_mean" if task == "simple_adversary" else "reward_mean"


def main():
    il = pd.read_csv(os.path.join(HERE, "results", "il.csv"))
    rnd = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(HERE, "results", "gap", "*.csv"))])
    s2p = glob.glob(os.path.join(HERE, "results", "step2", "parts", "*_s[12].csv"))
    s2 = pd.concat([pd.read_csv(f) for f in s2p])
    out = []
    P = out.append
    P('"random" = uniform over the projection bounds [-1, 1] per action dim (the policy\'s own actions lie in [0, 1]), '
      'for the protocol\'s agents only; other agents clean; no fault; 800 episodes; seeds 0-2.')
    P("")
    P("{:<26}{:<10}{:<14}{:>10}{:>10}{:>24}{:>8}{:>8}".format("task", "varied", "level", "J(clean)", "J(BC)",
                                                                "J(random) s0/s1/s2 mean", "G", "G3"))
    for task, varied in GROUPS:
        c = col(task)
        r = rnd[(rnd.scenario == task) & (rnd.varied == varied)]
        jr = [float(r[(r.seed == s) & (r.method == "random")][c].iloc[0]) for s in (0, 1, 2)]
        jr_m = float(np.mean(jr))
        for lv in IL.LEVELS_USED[(task, varied)]:
            t = il[(il.scenario == task) & (il.varied == varied) & (il.info_level == lv)]
            jc = float(t[t.method == "clean"][c].iloc[0])
            jb = float(t[t.method == "bc"][c].iloc[0])
            g = (jc - jb) / (jc - jr_m)
            g3 = ""
            if (task, varied, lv) in STEP2:
                x = s2[(s2.scenario == task) & (s2.varied == varied) & (s2.info_level == lv)]
                cl3 = [jc] + [float(x[(x.seed == s) & (x.method == "clean")][c].iloc[0]) for s in (1, 2)]
                bc3 = [jb] + [float(x[(x.seed == s) & (x.method == "bc")][c].iloc[0]) for s in (1, 2)]
                cr3 = [float(r[(r.seed == s) & (r.method == "clean")][c].iloc[0]) for s in (0, 1, 2)]
                assert np.allclose(cl3, cr3), (cl3, cr3)       # clean in the gap jobs equals il / step2 clean
                g3 = "{:.3f}".format((np.mean(cl3) - np.mean(bc3)) / (np.mean(cl3) - jr_m))
            P("{:<26}{:<10}{:<14}{:>10.2f}{:>10.2f}{:>24}{:>8.3f}{:>8}".format(
                task, varied, lv, jc, jb, "{:.1f}/{:.1f}/{:.1f} {:.1f}".format(jr[0], jr[1], jr[2], jr_m), g, g3))
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(HERE, "results", "gap_report.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
