"""Numpy inference (NumpyNet incl. encoder) vs torch for information-level models. Usage: python smoke_test3.py <model_dir>"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import info_levels as IL
torch.set_grad_enabled(False)
DATA = dict((t, C.load_agent_data(t)) for t in ("simple_spread", "simple_adversary", "simple_speaker_listener"))


def agent_i(task, agent):
    return agent

md = sys.argv[1]; C.MODEL_DIR = md
rs = np.random.RandomState(0); worst = 0.0
cases = [("simple_spread", 0, "shared", "bottleneck-2"), ("simple_spread", 1, "shared", "coarse-8"), ("simple_spread", 2, "shared", "removed"),
         ("simple_adversary", 1, "good", "bottleneck-4"), ("simple_adversary", 2, "good", "removed"),
         ("simple_speaker_listener", 0, "speaker", "bottleneck-1"), ("simple_speaker_listener", 0, "speaker", "removed"),
         ("simple_speaker_listener", 1, "listener", "bottleneck-4")]
for task, agent, role, lv in cases:
    tag, info_f = C.agent_tag_and_info(task, agent, "full")
    for m in ("bc", "den_s1"):
        model, ck = C.load_model(C.model_path(m + IL.SUFFIX[lv], task, info_f, tag, 0))
        nn_ = C.NumpyNet(model)
        lay = ck["level_layout"]
        od = IL.LAYOUT[(task, role)]["obs_dim"]
        obs = DATA[task][agent_i(task, agent)][0][rs.randint(0, 50000)][None]        # real observation row
        n = ck["norm"][agent]
        cond = IL.build_cond_level(lay, obs, n["obs_mean"], n["obs_std"], agent)[0]
        assert cond.shape[0] == lay["cond_in"], (cond.shape, lay["cond_in"])
        x = rs.randn(ck["action_dim"]).astype(np.float32)
        if m == "bc":
            ref = model(torch.from_numpy(cond)[None])[0].numpy(); got = nn_.direct(cond)
        else:
            t = 37
            ref = model(torch.from_numpy(cond)[None], x=torch.from_numpy(x)[None], t=torch.tensor([t]))[0].numpy()
            got = nn_.eps(x, t, nn_.cond_proj(cond))
        d = float(np.abs(ref - got).max()); worst = max(worst, d)
        print("{:<24} agent {} {:<13} {:<7} cond_in {:>2} enc {}->{}  max|diff| {:.1e}".format(task, agent, lv, m, lay["cond_in"], ck["enc_in"], ck["enc_d"], d))
print("WORST {:.2e}".format(worst))
