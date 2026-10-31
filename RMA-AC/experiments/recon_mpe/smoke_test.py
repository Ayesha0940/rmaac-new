"""
smoke_test.py -- numerical checks that need no environment:
  1. NumpyNet == torch MatchedNet for bc / c1 / c2 / den checkpoints,
  2. sample_chain (numpy, fast) == a literal torch port of train.diffusion_denoise_action (same injected noise),
     full chain and one-step,
  3. E[(clip(a + sigma*eps) - a)^2] on the dataset actions vs sigma^2 (projection caps the error).
Usage: python smoke_test.py <model_dir> <scenario> <info> <seed>
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


class FixedNoise:
    def __init__(self, zs):
        self.zs, self.i = zs, 0

    def standard_normal(self, shape):
        z = self.zs[self.i]; self.i += 1
        return z


def torch_reference(model, x, cond, t_start, one_step, zs, alphas, alphas_bar):
    """Literal port of train.diffusion_denoise_action (loop and update unchanged), H=1, noise injected."""
    x = torch.from_numpy(x)[None]
    cond = torch.from_numpy(cond)[None]
    zi = 0
    for t in reversed(range(t_start + 1)):
        eps_pred = model(cond, x=x, t=torch.tensor([t]))
        alpha, alpha_bar = alphas[t], alphas_bar[t]
        x0_hat = (x - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
        if one_step:
            return x0_hat[0].detach().numpy()
        if t > 0:
            x = torch.sqrt(alpha) * x0_hat + torch.sqrt(1 - alpha) * torch.from_numpy(zs[zi])[None]
            zi += 1
        else:
            x = x0_hat
    return x[0].detach().numpy()


def main():
    model_dir, scenario, info, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    C.MODEL_DIR = model_dir
    _, al, ab = C.beta_schedule()
    rng = np.random.RandomState(0)
    worst = 0.0
    n_agents = len(C.SCENARIOS[scenario]["obs_dims"])
    agents = [0, 1] if scenario == "simple_speaker_listener" else [0]
    for agent in agents:
        tag, info_f = C.agent_tag_and_info(scenario, agent, info)
        for m in C.METHODS:
            model, ck = C.load_model(C.model_path(m, scenario, info_f, tag, seed))
            nn_ = C.NumpyNet(model)
            cd, ad = ck["cond_dim"], ck["action_dim"]
            cond = rng.standard_normal(cd).astype(np.float32)
            x = rng.standard_normal(ad).astype(np.float32)
            if m == "den":
                for t_start in (10, 20, 40):
                    for one in (False, True):
                        zs = [rng.standard_normal(ad).astype(np.float32) for _ in range(t_start + 1)]
                        ref = torch_reference(model, x, cond, t_start, one, zs, al, ab)
                        got = C.sample_chain(nn_, x, cond, t_start, one, FixedNoise(list(zs)), al.numpy(), ab.numpy())
                        d = float(np.abs(ref - got).max()); worst = max(worst, d)
                        print("agent", agent, "den t_start", t_start, "one_step", one, "max|diff|", d)
                # eps check
                t = 17
                e_t = model(torch.from_numpy(cond)[None], x=torch.from_numpy(x)[None], t=torch.tensor([t]))[0].detach().numpy()
                e_n = nn_.eps(x, t, nn_.cond_proj(cond))
                d = float(np.abs(e_t - e_n).max()); worst = max(worst, d)
                print("agent", agent, "den eps(t=17) max|diff|", d)
            else:
                ref = model(torch.from_numpy(cond)[None], x=torch.from_numpy(x)[None] if m != "bc" else None)[0].detach().numpy()
                got = nn_.direct(cond, x if m != "bc" else None)
                d = float(np.abs(ref - got).max()); worst = max(worst, d)
                print("agent", agent, m, "max|diff|", d)
    print("WORST |diff| = {:.2e}".format(worst))

    # projection check on dataset actions (all agents, all steps)
    data = C.load_agent_data(scenario)
    A = np.concatenate([a.ravel() for _, a in data])
    print("clean action bounds: [{:.4f}, {:.4f}] (assert within [-1,1])".format(A.min(), A.max()))
    assert A.min() >= C.ACT_LOW and A.max() <= C.ACT_HIGH
    r = np.random.RandomState(1)
    for s in (0.0, 0.4, 1.0, 2.0, 3.0):
        e = r.standard_normal(A.shape) * s
        m = np.mean((np.clip(A + e, -1, 1) - A) ** 2)
        print("sigma {:.1f}: E[(clip(a+sigma*eps)-a)^2] = {:.4f}   sigma^2 = {:.4f}".format(s, m, s * s))


if __name__ == "__main__":
    main()
