"""Standard-DDPM-posterior sampler (numpy, common.ddpm_chain) vs a literal torch port (same injected noise),
S1 cosine schedule, full chain and one-step, scaled input. Usage: python smoke_test2.py <model_dir> <scenario> <seed>"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


class Fixed:
    def __init__(self, zs): self.zs, self.i = zs, 0
    def standard_normal(self, shape):
        z = self.zs[self.i]; self.i += 1; return z


def torch_ref(model, x, cond, t_start, one_step, zs, betas, alphas, abar):
    x = torch.from_numpy(x.astype(np.float64))[None]
    cond = torch.from_numpy(cond)[None]
    zi = 0
    for t in reversed(range(t_start + 1)):
        eps = model(cond, x=x.float(), t=torch.tensor([t])).double()
        ab = abar[t]
        x0 = (x - np.sqrt(1 - ab) * eps) / np.sqrt(ab)
        if one_step or t == 0:
            return x0[0].numpy()
        mean = (x - betas[t] / np.sqrt(1 - ab) * eps) / np.sqrt(alphas[t])
        var = betas[t] * (1 - abar[t - 1]) / (1 - ab)
        x = mean + np.sqrt(var) * torch.from_numpy(zs[zi].astype(np.float64))[None]; zi += 1
    return x[0].numpy()


torch.set_grad_enabled(False)
model_dir, scen, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
C.MODEL_DIR = model_dir
b, al, ab = [t.numpy().astype(np.float64) for t in C.schedule("cosine")]
rng = np.random.RandomState(0); worst = 0.0
tag, info_f = C.agent_tag_and_info(scen, 1 if scen == "simple_speaker_listener" else 0, "hidden")
model, ck = C.load_model(C.model_path("den_s1", scen, info_f, tag, seed))
nn_ = C.NumpyNet(model)
for t_start in (12, 41, 62, 79):
    for one in (False, True):
        cond = rng.standard_normal(ck["cond_dim"]).astype(np.float32)
        x = (np.sqrt(ab[t_start]) * rng.standard_normal(ck["action_dim"])).astype(np.float32)
        zs = [rng.standard_normal(ck["action_dim"]).astype(np.float32) for _ in range(t_start + 1)]
        ref = torch_ref(model, x, cond, t_start, one, zs, b, al, ab)
        got = C.ddpm_chain(nn_, x, cond, t_start, one, Fixed(list(zs)), b, al, ab)
        d = float(np.abs(ref - got).max()); worst = max(worst, d)
        print("t_start", t_start, "one_step", one, "max|diff| {:.2e}".format(d))
print("WORST {:.2e}".format(worst))
