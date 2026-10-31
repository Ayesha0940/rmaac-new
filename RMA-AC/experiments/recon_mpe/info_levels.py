"""
info_levels.py -- information-level definitions for the MPE sweep (Step 0: spec + printout only, no training).

Per (task, agent role) an observation layout (name -> indices), the TASK FIELDS that are reduced, and the
levels that apply. A level is one of
  full            keep every index
  removed         drop the task-field indices
  coarse-K        task-field 2-D relative positions -> K-sector one-hot each (K = 8 or 4); other indices kept
  bottleneck-d    the FULL own observation -> learned encoder -> d dims (trained jointly); the one-hot agent ID bypasses it
Sector of a 2-D relative position (dx, dy): theta = atan2(dy, dx) in [-pi, pi); sector = floor((theta + pi) / (2 pi / K)),
so edges lie on the axes for K = 4 and on axes + diagonals for K = 8.
Kept continuous fields are z-scored (train-set mean/std); sector one-hots are 0/1 and not normalized.
No other field in any of these observations is computed from the task fields (checked below).
"""
import sys

LAYOUT = {
    # simple_speaker_listener
    ("simple_speaker_listener", "speaker"): dict(
        obs_dim=3, fields=[("goal_colour", [0, 1, 2])], task=["goal_colour"], pair_fields=[],
        levels=["full", "removed", "bottleneck-1", "bottleneck-2"]),
    ("simple_speaker_listener", "listener"): dict(
        obs_dim=11, fields=[("own_vel", [0, 1]), ("landmark_rel_pos[0..2]", [2, 3, 4, 5, 6, 7]), ("received_message", [8, 9, 10])],
        task=["received_message"], pair_fields=[],
        levels=["full", "removed", "bottleneck-1", "bottleneck-2", "bottleneck-4"]),
    # simple_adversary: good agents only (agents 1, 2); the adversary is never faulted or corrected
    ("simple_adversary", "good"): dict(
        obs_dim=10, fields=[("goal_landmark_rel_pos", [0, 1]), ("landmark_rel_pos[0..1]", [2, 3, 4, 5]), ("other_agents_rel_pos[2]", [6, 7, 8, 9])],
        task=["goal_landmark_rel_pos"], pair_fields=[("goal_landmark_rel_pos", [0, 1])],
        levels=["full", "coarse-8", "coarse-4", "removed", "bottleneck-1", "bottleneck-2", "bottleneck-4"]),
    # simple_spread: one shared network for the 3 agents (ID one-hot over 3)
    ("simple_spread", "shared"): dict(
        obs_dim=18, fields=[("own_vel", [0, 1]), ("own_pos", [2, 3]), ("landmark_rel_pos[0..2]", [4, 5, 6, 7, 8, 9]),
                            ("other_agents_rel_pos[2]", [10, 11, 12, 13]), ("comm[2x2] (constant 0, all agents silent)", [14, 15, 16, 17])],
        task=["landmark_rel_pos[0..2]"], pair_fields=[("landmark_rel_pos[0]", [4, 5]), ("landmark_rel_pos[1]", [6, 7]), ("landmark_rel_pos[2]", [8, 9])],
        levels=["full", "coarse-8", "coarse-4", "removed", "bottleneck-1", "bottleneck-2", "bottleneck-4"],
        # named drop level: the agent no longer sees where the other agents are (assignment becomes a 3-way choice)
        drop_levels={"others-removed": ["other_agents_rel_pos[2]", "comm[2x2] (constant 0, all agents silent)"]}),
    # simple_push (Keep-away): good agent (agent 1). Own colour has +0.5 on channel goal_index+1 (encodes the goal);
    # landmark colours are fixed per landmark index (constant across episodes).
    ("simple_push", "good"): dict(
        obs_dim=19, fields=[("own_vel", [0, 1]), ("goal_rel_pos", [2, 3]), ("own_colour (encodes goal)", [4, 5, 6]),
                            ("landmark_rel_pos[0..1]", [7, 8, 9, 10]), ("landmark_colours[0..1]", [11, 12, 13, 14, 15, 16]),
                            ("other_agent_rel_pos", [17, 18])],
        task=["goal_rel_pos", "own_colour (encodes goal)"], pair_fields=[], levels=["full", "goal-hidden"],
        drop_levels={"goal-hidden": ["goal_rel_pos", "own_colour (encodes goal)"]}),
}
ID_DIM = {("simple_speaker_listener", "speaker"): 0, ("simple_speaker_listener", "listener"): 0,
          ("simple_adversary", "good"): 3, ("simple_spread", "shared"): 3, ("simple_push", "good"): 2}


def level_spec(task, role, level):
    L = LAYOUT[(task, role)]
    od = L["obs_dim"]
    task_idx = sorted(i for n, ix in L["fields"] if n in L["task"] for i in ix)
    keep_all = list(range(od))
    if level == "full":
        return dict(kind="select", keep=keep_all, replaced=[], onehot=[], dropped=[], dim=od)
    if level == "removed":
        keep = [i for i in keep_all if i not in task_idx]
        return dict(kind="select", keep=keep, replaced=[], onehot=[], dropped=task_idx, dim=len(keep))
    if level.startswith("coarse-"):
        K = int(level.split("-")[1])
        keep = [i for i in keep_all if i not in task_idx]
        return dict(kind="coarse", keep=keep, replaced=task_idx, onehot=[(n, ix, K) for n, ix in L["pair_fields"]],
                    dropped=[], dim=len(keep) + K * len(L["pair_fields"]))
    if level.startswith("bottleneck-"):
        d = int(level.split("-")[1])
        return dict(kind="bottleneck", keep=keep_all, replaced=[], onehot=[], dropped=[], dim=d, d=d)
    if level in L.get("drop_levels", {}):
        drop = sorted(i for n, ix in L["fields"] if n in L["drop_levels"][level] for i in ix)
        keep = [i for i in keep_all if i not in drop]
        return dict(kind="select", keep=keep, replaced=[], onehot=[], dropped=drop, dim=len(keep))
    raise ValueError(level)


def fmt_idx(ix):
    if not ix:
        return "[]"
    out, s, p = [], ix[0], ix[0]
    for i in ix[1:] + [None]:
        if i is not None and i == p + 1:
            p = i
            continue
        out.append(str(s) if s == p else "{}-{}".format(s, p))
        if i is not None:
            s = p = i
    return "[" + ",".join(out) + "]"


def describe():
    order = [("simple_speaker_listener", "speaker"), ("simple_speaker_listener", "listener"), ("simple_adversary", "good"), ("simple_spread", "shared")]
    print("Index notation: a-b = inclusive range of observation indices; ID = one-hot agent id.")
    for (task, role) in order:
        L = LAYOUT[(task, role)]
        idn = ID_DIM[(task, role)]
        print("\n=== {} / {}   (own obs dim {}, ID input dim {}{})".format(task, role, L["obs_dim"], idn,
              "" if idn else "; single-agent model, no ID input except a constant ID when the input would be empty"))
        print("    fields: " + "; ".join("{} {}".format(n, fmt_idx(ix)) for n, ix in L["fields"]))
        print("    task field(s) reduced: {}".format(", ".join(L["task"])))
        for lv in L["levels"]:
            s = level_spec(task, role, lv)
            if s["kind"] == "select" and lv == "full":
                txt = "keeps {} (all)".format(fmt_idx(s["keep"]))
            elif s["kind"] == "select":
                txt = "keeps {}; drops {}".format(fmt_idx(s["keep"]), fmt_idx(s["dropped"]))
            elif s["kind"] == "coarse":
                txt = "keeps {}; replaces {} by {}".format(fmt_idx(s["keep"]), fmt_idx(s["replaced"]),
                      " + ".join("{}-sector one-hot of {} {}".format(K, n, fmt_idx(ix)) for n, ix, K in s["onehot"]))
            else:
                txt = "full obs {} -> encoder -> d={} (encoder Linear({},128)-ReLU-Linear(128,{}), trained jointly)".format(
                    fmt_idx(s["keep"]), s["d"], L["obs_dim"], s["d"])
            extra = ""
            if lv == "removed" and not s["keep"]:
                extra = "   [empty input: constant one-hot ID only]"
            print("  {:<13} -> reduced input dim {:>2} (+{} ID): {}{}".format(lv, s["dim"], idn, txt, extra))


if __name__ == "__main__":
    describe()


# ============================================================ feature builders (used by training and evaluation)
import numpy as np

# simple_adversary: coarse levels dropped (a goal-direction sector would identify which landmark is the goal)
LEVELS_USED = {
    ("simple_speaker_listener", "speaker"): ["full", "removed", "bottleneck-1", "bottleneck-2"],
    ("simple_speaker_listener", "listener"): ["full", "removed", "bottleneck-1", "bottleneck-2", "bottleneck-4"],
    ("simple_adversary", "good"): ["full", "removed", "bottleneck-1", "bottleneck-2", "bottleneck-4"],
    ("simple_spread", "shared"): ["full", "coarse-8", "coarse-4", "removed", "bottleneck-1", "bottleneck-2", "bottleneck-4"],
}
SUFFIX = {"full": "", "removed": "_Lremoved", "bottleneck-1": "_Lbn1", "bottleneck-2": "_Lbn2", "bottleneck-4": "_Lbn4",
          "coarse-8": "_Lc8", "coarse-4": "_Lc4", "goal-hidden": "_Lgh", "others-removed": "_Lor"}
# agents corrected (and faulted) per (task, varied role)
CORRECTED = {("simple_speaker_listener", "speaker"): [0], ("simple_speaker_listener", "listener"): [1],
             ("simple_adversary", "good"): [1, 2], ("simple_spread", "shared"): [0, 1, 2]}


def role_for(task, agent):
    if task == "simple_speaker_listener":
        return "speaker" if agent == 0 else "listener"
    if task == "simple_push":
        return "good" if agent == 1 else "adv"
    return "good" if task == "simple_adversary" else "shared"


def sector_onehot(dx, dy, K):
    theta = np.arctan2(dy, dx)                                  # [-pi, pi]
    k = np.clip(np.floor((theta + np.pi) / (2 * np.pi / K)).astype(int), 0, K - 1)
    oh = np.zeros((len(dx), K), dtype=np.float32)
    oh[np.arange(len(dx)), k] = 1.0
    return oh


def cond_layout(task, role, level):
    """Widths for a level: dict(spec, idn, cond_in, model_cond, enc_in, enc_d). idn = ID input width actually used
    (a constant 2-wide ID when the input would otherwise be empty, i.e. speaker 'removed')."""
    spec = level_spec(task, role, level)
    idn = ID_DIM[(task, role)]
    if spec["dim"] == 0 and idn == 0:
        idn = 2
    od = LAYOUT[(task, role)]["obs_dim"]
    if spec["kind"] == "bottleneck":
        return dict(spec=spec, idn=idn, cond_in=od + idn, model_cond=spec["d"] + idn, enc_in=od, enc_d=spec["d"])
    return dict(spec=spec, idn=idn, cond_in=spec["dim"] + idn, model_cond=spec["dim"] + idn, enc_in=0, enc_d=0)


def build_cond_level(lay, obs_raw, obs_mean, obs_std, agent_idx):
    """obs_raw [B, od] (raw); returns the conditioning vector [B, cond_in] float32:
    reduced features (kept fields z-scored with train-set stats; sector one-hots unscaled; for bottleneck the full
    z-scored obs) followed by the one-hot agent ID (global agent index for adversary / spread)."""
    spec, idn = lay["spec"], lay["idn"]
    z = (obs_raw - obs_mean) / obs_std
    if spec["kind"] == "bottleneck":
        feats = z
    else:
        parts = [z[:, spec["keep"]]] if spec["keep"] else []
        for _, ix, K in spec["onehot"]:
            parts.append(sector_onehot(obs_raw[:, ix[0]], obs_raw[:, ix[1]], K))
        feats = np.concatenate(parts, axis=1) if parts else np.zeros((obs_raw.shape[0], 0), dtype=np.float32)
    oh = np.zeros((obs_raw.shape[0], idn), dtype=np.float32)
    if idn > 0:
        oh[:, agent_idx if idn > 2 or agent_idx < idn else 0] = 1.0
    return np.concatenate([feats, oh], axis=1).astype(np.float32)
