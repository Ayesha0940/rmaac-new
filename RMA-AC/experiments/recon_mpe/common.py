"""
common.py -- shared pieces for the per-agent reconstructor experiment on MPE
(simple_speaker_listener, simple_spread). Torch only (no TF), so the training
script does not need the environment.

Layouts (verified against the scenario files and diffusion_data/*.npz):
  simple_speaker_listener: joint obs 14 = speaker 3 (goal colour) | listener 11
                           (vel 2, landmark rel pos 6, received comm 3);
                           joint action 8 = speaker 3 (comm) | listener 5.
  simple_spread:           3 homogeneous agents, obs 18 each, action 5 each.
"""
import os

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(EXP, "diffusion_data")
MODEL_DIR = os.path.join(HERE, "models")
RESULT_DIR = os.path.join(HERE, "results")
LOG_DIR = os.path.join(HERE, "logs")

HIDDEN = 256
T_DIFF = 100
BETA_START, BETA_END = 1e-4, 2e-2
ACT_LOW, ACT_HIGH = -1.0, 1.0

SCENARIOS = {
    "simple_speaker_listener": {"obs_dims": [3, 11], "act_dims": [3, 5]},
    "simple_spread": {"obs_dims": [18, 18, 18], "act_dims": [5, 5, 5]},
    # agent 0 = adversary (obs 8: landmarks 4 + others 4, no goal); agents 1, 2 = good (obs 10: goal 2 + landmarks 4 + others 4)
    "simple_adversary": {"obs_dims": [8, 10, 10], "act_dims": [5, 5, 5]},
    # agent 0 = adversary (obs 8: vel 2 + landmarks 4 + other 2); agent 1 = good
    # (obs 19: vel 2 + goal 2 + colour 3 + landmarks 4 + landmark colours 6 + other 2)
    "simple_push": {"obs_dims": [8, 19], "act_dims": [5, 5]},
}
METHODS = ["bc", "c1", "c2", "den"]


def beta_schedule(T=T_DIFF):
    """Same as train.make_beta_schedule (numpy float64 -> float32 tensors)."""
    betas = torch.linspace(BETA_START, BETA_END, T)
    alphas = 1.0 - betas
    return betas, alphas, torch.cumprod(alphas, dim=0)


# ---------------------------------------------------------------- conditioning
def cond_spec(scenario, info, agent):
    """How the corrector input for `agent` is built from the agent's own obs.
    kind: 'obs'      -> z-scored own observation
          'id_only'  -> constant one-hot agent id (goal colour removed)
          'obs_id'   -> z-scored own observation + one-hot agent id (shared net)
    """
    if scenario == "simple_speaker_listener":
        if agent == 0:
            return {"kind": "id_only" if info == "hidden" else "obs", "n_ids": 2}
        return {"kind": "obs", "n_ids": 2}
    if scenario == "simple_push":
        return {"kind": "obs_id", "n_ids": 2}
    return {"kind": "obs_id", "n_ids": 3}


def cond_dim(scenario, info, agent):
    spec = cond_spec(scenario, info, agent)
    od = SCENARIOS[scenario]["obs_dims"][agent]
    return {"obs": od, "id_only": spec["n_ids"], "obs_id": od + spec["n_ids"]}[spec["kind"]]


def build_cond(spec, obs_norm, agent_idx):
    """obs_norm: [B, obs_dim] already z-scored. Returns [B, cond_dim] float32."""
    B = obs_norm.shape[0]
    if spec["kind"] == "obs":
        return obs_norm.astype(np.float32)
    onehot = np.zeros((B, spec["n_ids"]), dtype=np.float32)
    onehot[:, agent_idx] = 1.0
    if spec["kind"] == "id_only":
        return onehot
    return np.concatenate([obs_norm, onehot], axis=1).astype(np.float32)


# ------------------------------------------------------------------------ data
def load_agent_data(scenario):
    """Flatten the 25 steps of every episode, split into per-agent samples.
    Returns list over agents of (obs [N*25, od], act [N*25, ad])."""
    d = np.load(os.path.join(DATA_DIR, "{}_m3ddpg.npz".format(scenario)))
    S, A = d["states"], d["actions"]
    cfg = SCENARIOS[scenario]
    so = np.cumsum([0] + cfg["obs_dims"])
    ao = np.cumsum([0] + cfg["act_dims"])
    assert S.shape[2] == so[-1] and A.shape[2] == ao[-1], (S.shape, A.shape)
    out = []
    for i in range(len(cfg["obs_dims"])):
        o = S[:, :, so[i]:so[i + 1]].reshape(-1, cfg["obs_dims"][i])
        a = A[:, :, ao[i]:ao[i + 1]].reshape(-1, cfg["act_dims"][i])
        out.append((o.astype(np.float32), a.astype(np.float32)))
    return out


def norm_stats(x):
    return x.mean(0, keepdims=True).astype(np.float32), (x.std(0, keepdims=True) + 1e-6).astype(np.float32)


# ----------------------------------------------------------------------- model
class MatchedNet(nn.Module):
    """Backbone shared by all four methods.
       den : net([noisy_action, t_emb + c_emb])  -> predicted epsilon
       c1/c2: net([noisy_action, c_emb])          -> normalized clean action
       bc  : net(c_emb)                           -> normalized clean action
    cond path: Linear(cond->256)-ReLU-Linear(256->256). Time branch (den only):
    Linear(1->256)-ReLU-Linear(256->256), input t/1000, ADDED to c_emb (as in
    train.TrajectoryDiffusion)."""

    def __init__(self, cond_dim, action_dim, use_action, use_time, hidden=HIDDEN, enc_in=0, enc_d=0):
        """cond_dim = width of the conditioning vector AFTER the optional encoder. If enc_d > 0 the first enc_in
        columns of the raw conditioning vector go through Linear(enc_in,128)-ReLU-Linear(128,enc_d) and the remaining
        columns (agent ID) bypass it."""
        super().__init__()
        self.cond_dim, self.action_dim = cond_dim, action_dim
        self.use_action, self.use_time = use_action, use_time
        self.enc_in, self.enc_d = enc_in, enc_d
        if enc_d > 0:
            self.enc = nn.Sequential(nn.Linear(enc_in, 128), nn.ReLU(), nn.Linear(128, enc_d))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        if use_time:
            self.time_mlp = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        in_dim = (action_dim if use_action else 0) + hidden
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))

    def forward(self, cond, x=None, t=None):
        if self.enc_d > 0:
            cond = torch.cat([self.enc(cond[:, :self.enc_in]), cond[:, self.enc_in:]], dim=-1)
        h = self.cond_mlp(cond)
        if self.use_time:
            h = h + self.time_mlp(t.float().unsqueeze(-1) / 1000.0)
        if self.use_action:
            h = torch.cat([x, h], dim=-1)
        return self.net(h)


def make_model(method, cond_dim_, action_dim, enc_in=0, enc_d=0):
    return MatchedNet(cond_dim_, action_dim,
                      use_action=(method != "bc"), use_time=(method == "den"), enc_in=enc_in, enc_d=enc_d)


def model_path(method, scenario, info, agent_tag, seed):
    return os.path.join(MODEL_DIR, "{}_{}_{}_{}_s{}.pt".format(method, scenario, info, agent_tag, seed))


def agent_tag_and_info(scenario, agent, info):
    """File tag for the model that serves `agent`. The listener (and spread) models do not
    depend on the info level, so they are shared across levels."""
    if scenario == "simple_speaker_listener":
        if agent == 0:
            return "speaker", info
        return "listener", "any"
    if scenario in ("simple_adversary", "simple_push"):
        return ("adv", "full") if agent == 0 else ("good", "full")
    return "shared", "full"


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    m = make_model(ck["method"], ck["cond_dim"], ck["action_dim"], ck.get("enc_in", 0), ck.get("enc_d", 0))
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, ck


# --------------------------------------------- fast numpy inference (eval time)
class NumpyNet:
    """float32 numpy copy of MatchedNet for batch-1 inference; mathematically the
    same function as the torch module (checked in smoke_test.py)."""

    def __init__(self, model):
        sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
        self.use_action, self.use_time = model.use_action, model.use_time
        self.A = model.action_dim
        self.enc = "enc.0.weight" in sd
        if self.enc:
            self.eW0, self.eb0 = sd["enc.0.weight"].T, sd["enc.0.bias"]
            self.eW1, self.eb1 = sd["enc.2.weight"].T, sd["enc.2.bias"]
            self.enc_in = self.eW0.shape[0]
        self.cW0, self.cb0 = sd["cond_mlp.0.weight"].T, sd["cond_mlp.0.bias"]
        self.cW1, self.cb1 = sd["cond_mlp.2.weight"].T, sd["cond_mlp.2.bias"]
        W1 = sd["net.0.weight"].T            # [in, 256]
        self.b1 = sd["net.0.bias"]
        self.W2, self.b2 = sd["net.2.weight"].T, sd["net.2.bias"]
        if self.use_action:
            self.W1x, self.W1h = W1[:self.A], W1[self.A:]
        else:
            self.W1h = W1
        if self.use_time:
            tW0, tb0 = sd["time_mlp.0.weight"].T, sd["time_mlp.0.bias"]
            tW1, tb1 = sd["time_mlp.2.weight"].T, sd["time_mlp.2.bias"]
            ts = (np.arange(T_DIFF, dtype=np.float32) / 1000.0)[:, None]
            temb = np.maximum(ts @ tW0 + tb0, 0) @ tW1 + tb1         # [T, 256]
            self.temb_W1h = temb @ self.W1h                           # [T, 256]

    def cond_proj(self, cond):
        """cond: [cond_dim] -> (c_emb @ W1h) [256]; computed once per env step."""
        if self.enc:
            z = np.maximum(cond[:self.enc_in] @ self.eW0 + self.eb0, 0) @ self.eW1 + self.eb1
            cond = np.concatenate([z, cond[self.enc_in:]]).astype(np.float32)
        c = np.maximum(cond @ self.cW0 + self.cb0, 0) @ self.cW1 + self.cb1
        return c @ self.W1h

    def eps(self, x, t, ch):
        """den: predicted epsilon for normalized x [A] at step t (ch = cond_proj)."""
        pre = x @ self.W1x + ch + self.temb_W1h[t] + self.b1
        return np.maximum(pre, 0) @ self.W2 + self.b2

    def direct(self, cond, x=None):
        """c1/c2/bc: normalized clean-action prediction."""
        ch = self.cond_proj(cond)
        pre = ch + self.b1
        if self.use_action:
            pre = pre + x @ self.W1x
        return np.maximum(pre, 0) @ self.W2 + self.b2


def sample_chain(nnet, x, cond, t_start, one_step, rng, alphas, alphas_bar):
    """The existing reverse step of train.diffusion_denoise_action, unchanged:
        x0_hat = (x - sqrt(1-abar_t) * eps_hat) / sqrt(abar_t)
        x = sqrt(alpha_t) * x0_hat + sqrt(1-alpha_t) * z   (t > 0);  x = x0_hat (t = 0)
    run for t = t_start ... 0, no clipping, x_{t_start} = normalized projected noisy action
    (no sqrt(abar) scaling). one_step=True stops after the first evaluation and returns x0_hat.
    alphas / alphas_bar: float64 numpy arrays of length T."""
    ch = nnet.cond_proj(cond)
    for t in range(t_start, -1, -1):
        e = nnet.eps(x, t, ch)
        ab = alphas_bar[t]
        x0_hat = (x - np.sqrt(1.0 - ab) * e) / np.sqrt(ab)
        if one_step or t == 0:
            return x0_hat.astype(np.float32)
        a = alphas[t]
        x = (np.sqrt(a) * x0_hat + np.sqrt(1.0 - a) * rng.standard_normal(x.shape)).astype(np.float32)
    return x


# ============================================================ schedules (S0 / S1)
def schedule(name="linear", T=T_DIFF):
    """Returns float32 torch (betas, alphas, abar).
    linear : S0, the existing schedule (1e-4 -> 0.02).
    cosine : S1, Nichol & Dhariwal cosine (s=0.008), betas clipped to 0.999."""
    if name == "linear":
        return beta_schedule(T)
    assert name == "cosine", name
    s = 0.008
    t = np.arange(T + 1, dtype=np.float64) / T
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    abar = f / f[0]
    betas = np.clip(1.0 - abar[1:] / abar[:-1], 0.0, 0.999)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    return (torch.from_numpy(betas.astype(np.float32)), torch.from_numpy(alphas.astype(np.float32)),
            torch.from_numpy(abar.astype(np.float32)))


def noise_levels(abar):
    """sqrt((1 - abar_t) / abar_t): std of the noise relative to the signal in x_t / sqrt(abar_t)."""
    a = np.asarray(abar, dtype=np.float64)
    return np.sqrt((1.0 - a) / a)


def effective_noise_norm(actions_raw, act_std, sigma, seed=0):
    """Mean over action dims of rms(clip(a + sigma*eps, -1, 1) - a) / act_std  (normalized units,
    after the clip projection). actions_raw [N, d], act_std [d]."""
    rs = np.random.RandomState(seed)
    e = np.clip(actions_raw + sigma * rs.standard_normal(actions_raw.shape), ACT_LOW, ACT_HIGH) - actions_raw
    return float(np.mean(np.sqrt(np.mean(e ** 2, axis=0)) / act_std))


def t_match(level_eff, lv):
    """step whose noise level sqrt((1-abar)/abar) is closest to level_eff."""
    return int(np.argmin(np.abs(lv - level_eff)))


def ddpm_chain(nnet, x, cond, t_start, one_step, rng, betas, alphas, abar):
    """STANDARD DDPM posterior sampler (not the legacy alpha_t re-noising step), epsilon-prediction:
        x0_hat = (x_t - sqrt(1-abar_t) eps_hat) / sqrt(abar_t)
        mean   = (x_t - beta_t / sqrt(1-abar_t) * eps_hat) / sqrt(alpha_t)
        var    = beta_t * (1 - abar_{t-1}) / (1 - abar_t),  no noise at t = 0
    run t = t_start ... 0, no clipping. one_step=True returns x0_hat of the first evaluation.
    betas/alphas/abar: float64 numpy arrays."""
    ch = nnet.cond_proj(cond)
    for t in range(t_start, -1, -1):
        e = nnet.eps(x, t, ch)
        ab = abar[t]
        x0_hat = (x - np.sqrt(1.0 - ab) * e) / np.sqrt(ab)
        if one_step or t == 0:
            return x0_hat.astype(np.float32)
        mean = (x - betas[t] / np.sqrt(1.0 - ab) * e) / np.sqrt(alphas[t])
        var = betas[t] * (1.0 - abar[t - 1]) / (1.0 - ab)
        x = (mean + np.sqrt(var) * rng.standard_normal(x.shape)).astype(np.float32)
    return x


def fault_error_norm(A_ep, fault, level, act_std, seed=0):
    """'Injected noise level' of a fault in normalized units: mean over action dims of
    rms(a~ - a) / act_std, where a~ = clip(fault(a)) is simulated on clean dataset episodes.
    A_ep [E, 25, d] raw clean actions of one agent. fault in {gauss, biased, delay, stuck}
    (same semantics as action_attacks.ActionAttack: biased = mean +1; delay k returns a_{t-k}, or a_0 for t<k;
    stuck-at freezes each dim, with prob p, at its first value)."""
    rs = np.random.RandomState(seed)
    E_, T_, d = A_ep.shape
    if fault == "gauss":
        out = A_ep + level * rs.standard_normal(A_ep.shape)
    elif fault == "biased":
        out = A_ep + 1.0 + level * rs.standard_normal(A_ep.shape)
    elif fault == "delay":
        k = int(level)
        idx = np.maximum(np.arange(T_) - k, 0)
        out = A_ep[:, idx, :]
    elif fault == "stuck":
        mask = rs.rand(E_, 1, d) < level
        out = np.where(mask, A_ep[:, :1, :], A_ep)
    else:
        raise ValueError(fault)
    out = np.clip(out, ACT_LOW, ACT_HIGH)
    err = (out - A_ep).reshape(-1, d)
    return float(np.mean(np.sqrt(np.mean(err ** 2, axis=0)) / act_std))
