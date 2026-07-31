"""
action_filters.py
==================
Stateful action-recovery filters (EMA, median, Kalman) applied to actions
after they have passed through `ActionAttack.perturb()` (noise/delay/actuator
fault injection) and before `env.step()`.

Mirrors the lifecycle of `ActionAttack` (`action_attacks.py`) so it can be
dropped into the same test loop:

    filt = ActionFilter(args, kind="ema")
    ...
    for ep in range(num_test_episodes):
        filt.reset(n_agents=len(action_n))
        ...
        for step in range(horizon):
            action_n_filtered = [filt.apply(a, i) for i, a in enumerate(action_n_noisy)]
            ...

argparse additions (already present in train.py:parse_args)
-------------------------------------------------------------
    --filter-list {ema,median,kalman}+  (which filters to evaluate in a sweep)
    --ema-alpha       float   EMA smoothing factor (higher = less smoothing)
    --median-window   int     sliding window length for the median filter
    --kalman-q        float   Kalman process-noise variance
    --kalman-r        float   Kalman measurement-noise variance
"""

import numpy as np


class ActionFilter:
    def __init__(self, args, kind):
        self.a = args
        self.kind = kind
        self._buf = {}  # per-agent state: {agent_idx: {...}}

    # ---- lifecycle --------------------------------------------------------
    def reset(self, n_agents):
        self._buf = {i: {} for i in range(n_agents)}

    # ---- main entry -------------------------------------------------------
    def apply(self, action, agent_idx=0):
        x = np.asarray(action, dtype=np.float32)
        st = self._buf.setdefault(agent_idx, {})

        if self.kind == "ema":
            out = self._ema(x, st)
        elif self.kind == "median":
            out = self._median(x, st)
        elif self.kind == "kalman":
            out = self._kalman(x, st)
        else:
            raise ValueError("unknown filter: {}".format(self.kind))

        return out

    # ---- EMA ----------------------------------------------------------
    def _ema(self, x, st):
        alpha = self.a.ema_alpha
        if "y" not in st:
            st["y"] = x.copy()
        else:
            st["y"] = alpha * x + (1.0 - alpha) * st["y"]
        return st["y"].copy()

    # ---- median ---------------------------------------------------------
    def _median(self, x, st):
        w = max(1, self.a.median_window)
        hist = st.setdefault("hist", [])
        hist.append(x.copy())
        if len(hist) > w:
            hist.pop(0)
        return np.median(np.stack(hist, axis=0), axis=0)

    # ---- Kalman (independent scalar filter per action dimension) --------
    def _kalman(self, x, st):
        q, r = self.a.kalman_q, self.a.kalman_r
        if "x_hat" not in st:
            st["x_hat"] = x.copy()
            st["P"] = np.ones_like(x)
            return st["x_hat"].copy()

        # predict (constant-position model)
        x_pred = st["x_hat"]
        p_pred = st["P"] + q

        # update
        k_gain = p_pred / (p_pred + r)
        x_hat = x_pred + k_gain * (x - x_pred)
        p = (1.0 - k_gain) * p_pred

        st["x_hat"], st["P"] = x_hat, p
        return x_hat.copy()
