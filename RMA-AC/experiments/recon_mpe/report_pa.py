"""Report for the per-agent denoiser stage. Reads results/pa.csv (or argv[1]); writes results/pa_report.txt.
WORKS = variant beats the noisy policy by > 2 standard errors of the difference
        (se = reward_std / sqrt(n_episodes), combined in quadrature) in that cell.
simple_adversary: WORKS and the std-0 gap use the GOOD-agent reward; adversary reward is shown separately.
Other tasks: reward summed over agents. Oracle rows (t_match uses knowledge of the fault) are labelled."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = [("gauss", 0.0, "gauss 0"), ("gauss", 1.0, "gauss 1"), ("gauss", 3.0, "gauss 3"),
         ("biased_gauss_mu+1", 1.0, "biased+1"), ("delay", 3.0, "delay 3"), ("delay", 8.0, "delay 8"),
         ("stuck", 0.5, "stuck .5"), ("stuck", 1.0, "stuck 1")]
TASKS = ["simple_speaker_listener", "simple_adversary", "simple_spread"]
ORACLE_TAG = "oracle (upper bound, not deployable)"


def is_oracle(name):
    return "ORACLE" in name


def disp(name):
    return name.replace("ORACLE_", "oracle ") + ("  [" + ORACLE_TAG + "]" if is_oracle(name) else "")


def cols(task):
    return ("reward_good_mean", "reward_good_std") if task == "simple_adversary" else ("reward_mean", "reward_std")


def lookup(d, task, method, cell):
    x = d[(d.scenario == task) & (d.method == method) & (d.fault == cell[0]) & (d.level == cell[1])]
    return None if len(x) == 0 else x.iloc[0]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "pa.csv")
    d = pd.read_csv(path)
    out = []
    P = out.append
    tm = pd.read_csv(os.path.join(HERE, "results", "pa_tmatch.csv"))
    for task in TASKS:
        t = d[d.scenario == task]
        if len(t) == 0:
            continue
        mc, sc = cols(task)
        methods = [m for m in t.method.unique() if m not in ("clean", "noisy")]
        dep = sorted([m for m in methods if not is_oracle(m)], key=lambda m: (m[:2], m))
        orc = sorted([m for m in methods if is_oracle(m)], key=lambda m: (m[:2], m))
        P("=" * 118)
        P("TASK {}   reward = {}".format(task, "GOOD-agent reward (adversary reported separately)" if task == "simple_adversary" else "summed over agents"))
        P("=" * 118)
        hdr = "{:<100}".format("variant") + "".join("{:>11}".format(c[2]) for c in CELLS) + "  #WORKS/7  std0-gap"
        P(hdr)
        clean = lookup(t, task, "clean", CELLS[0])
        n_cells_work = {}
        for block, names in (("reference", ["clean", "noisy"]), ("deployable (fixed t)", dep), ("oracle", orc)):
            if not names:
                continue
            P("-- " + (block if block != "oracle" else ORACLE_TAG))
            for m in names:
                line = "{:<100}".format(disp(m) if block != "reference" else m)
                nw = 0
                for k, cell in enumerate(CELLS):
                    r = lookup(t, task, m, cell)
                    nz = lookup(t, task, "noisy", cell)
                    if r is None:
                        line += "{:>11}".format("-")
                        continue
                    mark = ""
                    if m not in ("clean", "noisy") and nz is not None:
                        se = np.sqrt((r[sc] ** 2 + nz[sc] ** 2) / r["n_episodes"])
                        if r[mc] - nz[mc] > 2 * se:
                            mark = "*"
                            if k > 0:
                                nw += 1
                    line += "{:>11}".format("{:.1f}{}".format(r[mc], mark))
                gap = ""
                r0 = lookup(t, task, m, CELLS[0])
                if m not in ("clean", "noisy") and r0 is not None and clean is not None:
                    gap = "{:+.1f}".format(r0[mc] - clean[mc])
                if m in ("clean", "noisy"):
                    line += "          -   {:>8}".format("-")
                else:
                    n_cells_work[m] = nw
                    line += "{:>9d}   {:>8}".format(nw, gap)
                P(line)
        P("(* = WORKS: beats noisy by > 2 SE; std0-gap = reward at gauss 0 minus clean policy; #WORKS counts the 7 faulted cells)")
        # separate adversary reward
        if task == "simple_adversary":
            P("\nAdversary reward (agent 0), no WORKS marks")
            P("{:<100}".format("variant") + "".join("{:>11}".format(c[2]) for c in CELLS))
            for m in ["clean", "noisy"] + dep + orc:
                line = "{:<100}".format(disp(m) if m not in ("clean", "noisy") else m)
                for cell in CELLS:
                    r = lookup(t, task, m, cell)
                    line += "{:>11}".format("-" if r is None else "{:.2f}".format(r["reward_adv_mean"]))
                P(line)
            P("(summed-over-agents reward is in the CSV column reward_mean)")
        # action MSE
        P("\nAction MSE vs clean policy sample, mean over agents")
        P("{:<100}".format("variant") + "".join("{:>11}".format(c[2]) for c in CELLS))
        for m in ["noisy"] + dep + orc:
            line = "{:<100}".format(disp(m) if m != "noisy" else m)
            for cell in CELLS:
                r = lookup(t, task, m, cell)
                line += "{:>11}".format("-" if r is None else "{:.3f}".format(np.mean([float(x) for x in str(r["action_mse_per_agent"]).split("|")])))
            P(line)
        if task == "simple_speaker_listener":
            P("\nMessage accuracy (argmax of executed speaker message == argmax of clean message)")
            P("{:<100}".format("variant") + "".join("{:>11}".format(c[2]) for c in CELLS))
            for m in ["noisy"] + dep + orc:
                line = "{:<100}".format(disp(m) if m != "noisy" else m)
                for cell in CELLS:
                    r = lookup(t, task, m, cell)
                    line += "{:>11}".format("-" if r is None else "{:.3f}".format(float(r["msg_acc"])))
                P(line)
        # t_match
        P("\nt_match per fault level (oracle variants only), per agent: S0 / S1")
        tt = tm[tm.scenario == task]
        for _, cell in enumerate(CELLS):
            fk = {"biased_gauss_mu+1": "biased"}.get(cell[0], cell[0])
            x = tt[(tt.fault == fk) & (tt.level == cell[1])]
            P("  {:<10} ".format(cell[2]) + "   ".join("agent{}: {}/{} (err {:.2f})".format(int(r.agent), int(r.t_match_S0), int(r.t_match_S1), r.injected_error_norm) for r in x.itertuples()))
        # summary lines
        most = 4
        w1 = [(m, n) for m, n in n_cells_work.items() if not is_oracle(m) and n >= most]
        w2 = [(m, n) for m, n in n_cells_work.items() if is_oracle(m) and n >= most]
        fmt = lambda L: "; ".join("{} ({}/7)".format(m, n) for m, n in sorted(L, key=lambda z: -z[1])) or "none"
        P("\nSUMMARY {}".format(task))
        P("  (1) deployable variants (fixed t only) that work across most faults (>=4 of 7): {}".format(fmt(w1)))
        P("  (2) oracle variants ({}) that work across most faults: {}".format(ORACLE_TAG, fmt(w2)))
        P("  The WORKS verdict for the method is line (1) only.\n")
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(HERE, "results", "pa_report.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
