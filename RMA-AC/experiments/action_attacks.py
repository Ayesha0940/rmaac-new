"""
action_attacks.py
=================
Drop-in extension of `apply_action_disruption` for IROS rebuttal point #2
(diverse / realistic action uncertainties).

Design
------
All attacks are exposed through a single stateful `ActionAttack` object so that
attacks needing memory (delay, OU, hold-last dropout, non-stationary schedules)
work the same way as memoryless ones.

Integration (in your test loop in train.py)
-------------------------------------------
    from action_attacks import ActionAttack

    attack = ActionAttack(args)          # once, before the episode loop
    ...
    for ep in range(num_test_episodes):
        attack.reset(n_agents=len(action_n))     # at episode start
        ...
        for step in range(horizon):
            action_n_noisy = [attack.perturb(a, i) for i, a in enumerate(action_n)]
            attack.step()                        # once per env step
            ... (your existing diffusion_denoise_action call is unchanged) ...

Backward compatibility
----------------------
`apply_action_disruption(action, reward, env, args)` is kept as a thin wrapper
so existing call sites keep working for the memoryless attacks.

argparse additions (paste into parse_args)
------------------------------------------
    p.add_argument("--attack", type=str, default="gauss",
        choices=["none","gauss","biased_gauss",        # already validated
                 "timevarying_gauss","heavy_tail","colored_ou","dist_shift",  # Tier 1
                 "stuck_at","dropout","saturate","gain","deadzone","signflip", # Tier 2
                 "delay","quantize"])                   # Tier 3
    p.add_argument("--attack-sigma", type=float, default=0.5)   # base std / scale
    p.add_argument("--attack-mu",    type=float, default=0.0)   # bias
    p.add_argument("--tv-profile",   type=str, default="sin",
                   choices=["ramp","sin","randwalk"])           # timevarying_gauss
    p.add_argument("--tv-period",    type=int,  default=25)     # sin period (steps)
    p.add_argument("--heavy-dist",   type=str, default="laplace",
                   choices=["laplace","student_t","saltpepper"])
    p.add_argument("--heavy-df",     type=float, default=2.0)   # student-t dof
    p.add_argument("--sp-prob",      type=float, default=0.1)   # salt&pepper / dropout / stuck prob
    p.add_argument("--ou-theta",     type=float, default=0.15)  # OU mean reversion
    p.add_argument("--ou-sigma",     type=float, default=0.3)   # OU volatility
    p.add_argument("--shift-low",    type=float, default=-0.5)  # dist_shift uniform bounds
    p.add_argument("--shift-high",   type=float, default=0.5)
    p.add_argument("--dropout-fill", type=str, default="zero",
                   choices=["zero","hold"])                     # zero-fill vs hold-last
    p.add_argument("--gain-factor",  type=float, default=1.5)   # multiplicative gain error
    p.add_argument("--deadzone",     type=float, default=0.1)   # |a|<dz -> 0
    p.add_argument("--delay-k",      type=int,  default=3)      # comms delay (steps)
    p.add_argument("--quant-levels", type=int,  default=8)      # quantization bins
    p.add_argument("--act-low",      type=float, default=None)  # optional clip (Pi_A); None = no clip
    p.add_argument("--act-high",     type=float, default=None)
"""

import numpy as np


class ActionAttack:
    def __init__(self, args):
        self.a = args
        self.t = 0
        self._buf = {}          # per-agent histories: {agent_idx: {...}}

    # ---- lifecycle --------------------------------------------------------
    def reset(self, n_agents):
        self.t = 0
        self._buf = {i: {"hist": [], "ou": None, "last": None} for i in range(n_agents)}

    def step(self):
        self.t += 1

    def _clip(self, x):
        lo, hi = self.a.act_low, self.a.act_high
        if lo is not None and hi is not None:
            return np.clip(x, lo, hi)
        return x

    # ---- main entry -------------------------------------------------------
    def perturb(self, action, agent_idx=0):
        x = np.asarray(action, dtype=np.float32).copy()
        kind = getattr(self.a, "attack", "gauss")
        st = self._buf.setdefault(agent_idx, {"hist": [], "ou": None, "last": None})

        if kind == "none":
            out = x

        # ---------- already validated -------------------------------------
        elif kind == "gauss":
            out = x + np.random.normal(0.0, self.a.attack_sigma, size=x.shape)
        elif kind == "biased_gauss":
            out = x + np.random.normal(self.a.attack_mu, self.a.attack_sigma, size=x.shape)

        # ---------- Tier 1: HIGH ------------------------------------------
        elif kind == "timevarying_gauss":
            sigma_t = self._tv_sigma(st)
            out = x + np.random.normal(self.a.attack_mu, sigma_t, size=x.shape)

        elif kind == "heavy_tail":
            out = x + self._heavy(x.shape)

        elif kind == "colored_ou":
            if st["ou"] is None:
                st["ou"] = np.zeros_like(x)
            # OU: e_{t} = e_{t-1} + theta*(mu - e_{t-1}) + sigma*N(0,1)
            st["ou"] = (st["ou"]
                        + self.a.ou_theta * (self.a.attack_mu - st["ou"])
                        + self.a.ou_sigma * np.random.normal(size=x.shape))
            out = x + st["ou"]

        elif kind == "dist_shift":
            out = x + np.random.uniform(self.a.shift_low, self.a.shift_high, size=x.shape)

        # ---------- Tier 2: MEDIUM ----------------------------------------
        elif kind == "stuck_at":
            # each dim independently freezes (prob sp_prob) at its first-seen value
            if st["last"] is None:
                st["last"] = x.copy()
                st["frozen_mask"] = np.random.rand(*x.shape) < self.a.sp_prob
                st["frozen_val"] = x.copy()
            out = np.where(st["frozen_mask"], st["frozen_val"], x)

        elif kind == "dropout":
            mask = np.random.rand(*x.shape) < self.a.sp_prob
            if self.a.dropout_fill == "zero":
                out = np.where(mask, 0.0, x)
            else:  # hold last executed value
                base = st["last"] if st["last"] is not None else x
                out = np.where(mask, base, x)
            st["last"] = x.copy()

        elif kind == "saturate":
            # clip to a tighter actuator limit than the env's own bound
            lim = self.a.attack_sigma  # reuse as saturation magnitude
            out = np.clip(x, -lim, lim)

        elif kind == "gain":
            out = x * self.a.gain_factor

        elif kind == "deadzone":
            out = np.where(np.abs(x) < self.a.deadzone, 0.0, x)

        elif kind == "signflip":
            # flip a fixed random subset of dims for the whole episode
            if "flip_mask" not in st:
                st["flip_mask"] = np.random.rand(*x.shape) < self.a.sp_prob
            out = np.where(st["flip_mask"], -x, x)

        # ---------- Tier 3: LOW -------------------------------------------
        elif kind == "delay":
            st["hist"].append(x.copy())
            k = self.a.delay_k
            out = st["hist"][-1 - k] if len(st["hist"]) > k else st["hist"][0]

        elif kind == "quantize":
            lo = self.a.act_low if self.a.act_low is not None else float(x.min())
            hi = self.a.act_high if self.a.act_high is not None else float(x.max())
            hi = hi if hi > lo else lo + 1e-6
            levels = max(2, self.a.quant_levels)
            q = np.round((x - lo) / (hi - lo) * (levels - 1)) / (levels - 1)
            out = q * (hi - lo) + lo

        else:
            raise ValueError("unknown attack: {}".format(kind))

        return self._clip(out)

    # ---- helpers ----------------------------------------------------------
    def _tv_sigma(self, st):
        s0, prof = self.a.attack_sigma, self.a.tv_profile
        if prof == "ramp":
            return s0 * (1.0 + self.t / max(1, self.a.tv_period))
        if prof == "sin":
            return s0 * (1.0 + 0.9 * np.sin(2 * np.pi * self.t / max(1, self.a.tv_period)))
        if prof == "randwalk":
            if st["ou"] is None:
                st["ou"] = s0
            st["ou"] = max(0.0, st["ou"] + 0.1 * s0 * np.random.normal())
            return st["ou"]
        return s0

    def _heavy(self, shape):
        d, s = self.a.heavy_dist, self.a.attack_sigma
        if d == "laplace":
            return np.random.laplace(self.a.attack_mu, s / np.sqrt(2), size=shape)
        if d == "student_t":
            return s * np.random.standard_t(self.a.heavy_df, size=shape)
        if d == "saltpepper":
            mask = np.random.rand(*shape) < self.a.sp_prob
            spikes = np.random.choice([-1.0, 1.0], size=shape) * (5.0 * s)
            return np.where(mask, spikes, 0.0)
        raise ValueError(d)


# -------------------------------------------------------------------------
# Backward-compatible wrapper for existing memoryless call sites.
# (Stateful attacks should use ActionAttack.perturb directly.)
# -------------------------------------------------------------------------
_GLOBAL_ATTACK = None


def apply_action_disruption(action, reward, env, args):
    global _GLOBAL_ATTACK
    if _GLOBAL_ATTACK is None or getattr(_GLOBAL_ATTACK, "a", None) is not args:
        _GLOBAL_ATTACK = ActionAttack(args)
        _GLOBAL_ATTACK.reset(n_agents=64)  # generous; per-agent buffers are lazy
    return _GLOBAL_ATTACK.perturb(action, agent_idx=0)
