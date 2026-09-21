# SPDX-License-Identifier: GPL-3.0
# Uses the initial_buffer library of cl_initial_buffer
# (https://github.com/uzh-rpg/cl_initial_buffer, GNU GPL v3.0); this package
# is distributed under the same licence (LICENSE-GPL-3.0).
"""Validation pass for the P-ISB locomotion runs, matching the reference paper.

WHY TRAINING REWARD IS NOT THE ANSWER. Messikommer et al. (ICRA 2024) Section V:

    "During training, the robot is initialized in the middle of the whole
     environment and needs to explore the other terrain types. However, during
     validation, we initialize the robot in the center of all of the different
     terrain types, which enables us to directly assess how well the robot
     learned to walk on the different terrains."

Their Table I and Fig. 3(a) -- including the headline 18.3% -- are VALIDATION
numbers. Reading Train/mean_reward out of the tensorboard log measures something
else, and measures it unfairly: an ISB arm resets robots mid-terrain, so its
training reward is collected under a different initial-state distribution than
vanilla's, the training-versus-deployment mismatch discussed in the paper.
Every arm has to be scored from the same fixed
starts, which is what this script does.

NO RETRAINING IS NEEDED. rsl_rl saves a checkpoint every 50 iterations, so the
whole Fig. 3(a) curve can be reconstructed after the fact from model_*.pt.

ONE SIM, MANY CHECKPOINTS. IsaacGym does not reliably support creating and
destroying several sims inside one process, so the environment is built once and
each checkpoint is loaded into it in turn. That also guarantees every arm is
scored in an identical world, which a per-checkpoint rebuild would not.

THE TERRAIN ASSIGNMENT IS PINNED EVERY STEP. legged_gym runs a terrain
curriculum that promotes or demotes a robot on reset. During validation that
would silently move robots off the patch they were assigned to, so the level and
type -- and the env_origins derived from them -- are re-pinned each step. A
robot that falls therefore restarts on the SAME terrain it was being tested on.

Usage
  python pisb_legged/validate_legged.py
  python pisb_legged/validate_legged.py --runs pisb_runs --steps 1000
  python pisb_legged/validate_legged.py --iters 500,1000,1500,2000
"""
import argparse
import glob
import json
import os
import sys

from isaacgym import gymapi  # noqa: F401  MUST precede torch
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402
from legged_gym.envs import task_registry   # noqa: E402
from legged_gym.utils import get_args        # noqa: E402
from rsl_rl.runners import OnPolicyRunner    # noqa: E402

CKPT = "model_%d.pt"


def parse_known():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--runs", default="pisb_runs")
    p.add_argument("--task", default="anymal_c_rough")
    p.add_argument("--val_envs", type=int, default=200,
                   help="one per terrain patch: num_rows x num_cols = 10 x 20")
    p.add_argument("--steps", type=int, default=2000,
                   help="validation steps per checkpoint. MUST exceed the episode "
                        "timeout, which for anymal is 20 s at 0.02 s control dt = "
                        "1000 steps. At exactly 1000 every environment is still "
                        "mid-episode at the cutoff and gets scored on a PARTIAL "
                        "return, so the metric silently measures survival within "
                        "the window instead of episode reward: vanilla banked "
                        "19.36 with 173 of 200 environments unterminated while "
                        "obs banked 7.91 with 54. 2000 guarantees every "
                        "environment completes at least one full episode.")
    p.add_argument("--iters", default="",
                   help="comma-separated checkpoint iterations; default is every "
                        "checkpoint found")
    p.add_argument("--easy_max_level", type=int, default=4,
                   help="levels 0..N are Easy, above are Hard. The paper splits "
                        "by [10]'s terrain parametrisation; this is that split")
    p.add_argument("--out", default="")
    # legged_gym's get_args() re-parses sys.argv through gymutil with a STRICT
    # parser, so any flag of ours still sitting there aborts the run with
    # "unrecognized arguments". Consume ours and hand gymutil only the rest.
    a, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return a


def pin_terrain(env, levels, types):
    """Hold every robot on its assigned patch, curriculum notwithstanding."""
    env.terrain_levels[:] = levels
    env.terrain_types[:] = types
    if hasattr(env, "terrain_origins") and env.terrain_origins is not None:
        env.env_origins[:] = env.terrain_origins[levels, types]


def validate(env, policy, levels, types, steps, device):
    """-> (per-env list of completed episode returns, mean over all)."""
    n = env.num_envs
    pin_terrain(env, levels, types)
    obs, _ = env.reset(), None
    obs = obs[0] if isinstance(obs, tuple) else obs

    cur = torch.zeros(n, dtype=torch.float, device=device)
    per_env = [[] for _ in range(n)]
    # Length-independent companion metric. Mean episode return rewards a policy
    # both for scoring well per step AND for surviving longer, and those two are
    # impossible to separate when episodes are truncated by the scoring window.
    # Reward per step is immune to that, so a disagreement between the two is a
    # direct signal that the comparison is being driven by survival time.
    total_rew = torch.zeros(n, dtype=torch.float, device=device)

    with torch.no_grad():
        for _ in range(steps):
            actions = policy(obs.detach())
            obs, _priv, rews, dones, _infos = env.step(actions.detach())
            cur += rews
            total_rew += rews
            done_idx = (dones > 0).nonzero(as_tuple=False).flatten()
            for i in done_idx.tolist():
                per_env[i].append(float(cur[i]))
                cur[i] = 0.0
            # Re-pin AFTER the step so the curriculum's promotion on reset is
            # undone before the next episode starts on this env.
            pin_terrain(env, levels, types)

    # An env that never terminated still carries information; count its partial
    # return rather than dropping the env entirely, and record how often this
    # happened so the caller can judge.
    n_partial = 0
    for i in range(n):
        if not per_env[i]:
            per_env[i].append(float(cur[i]))
            n_partial += 1
    return per_env, n_partial, (total_rew / float(steps)).cpu().numpy()


def main():
    # Same precision as training (see train_pisb.disable_tf32).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    a = parse_known()
    root = a.runs if os.path.isabs(a.runs) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), a.runs)
    frags = sorted(glob.glob(os.path.join(root, "frag", "*.json")))
    if not frags:
        sys.exit("ABORT: no fragments in %s/frag" % root)

    want = [int(x) for x in a.iters.split(",") if x.strip()] if a.iters else None

    # ------------------------------------------------------------- the env
    # Built ONCE, from the stock task: validation runs the policy, never the
    # buffer, so the ISB-enabled variant would only add machinery that must then
    # be disabled.
    args = get_args()
    args.task = a.task
    args.headless = True
    args.num_envs = a.val_envs
    env_cfg, train_cfg = task_registry.get_cfgs(name=a.task)
    env_cfg.env.num_envs = a.val_envs
    # Domain randomisation off: validation compares policies, not luck.
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env, env_cfg = task_registry.make_env(name=a.task, args=args, env_cfg=env_cfg)
    device = env.device

    rows = env_cfg.terrain.num_rows
    cols = env_cfg.terrain.num_cols
    idx = torch.arange(a.val_envs, device=device)
    levels = (idx % rows).long()
    types = ((idx // rows) % cols).long()
    print("validation grid: %d envs over %d levels x %d types  (easy <= %d)"
          % (a.val_envs, rows, cols, a.easy_max_level))

    train_cfg.runner.resume = False
    runner = OnPolicyRunner(env, class_to_dict(train_cfg), None, device=str(device))

    lv = levels.cpu().numpy()
    easy_mask = lv <= a.easy_max_level
    results = []

    for fp in frags:
        meta = json.load(open(fp))
        log_dir = meta.get("log_dir", "")
        arm, seed = meta["arm"], meta["seed"]
        cks = sorted(glob.glob(os.path.join(log_dir, "model_*.pt")),
                     key=lambda p: int(os.path.basename(p)[6:-3]))
        if not cks:
            print("  WARN no checkpoints for %s_s%s" % (arm, seed))
            continue
        for ck in cks:
            it = int(os.path.basename(ck)[6:-3])
            if want is not None and it not in want:
                continue
            runner.load(ck, load_optimizer=False)
            policy = runner.get_inference_policy(device=device)
            per_env, n_partial, per_step = validate(env, policy, levels, types,
                                                    a.steps, device)
            mean_env = np.array([np.mean(v) for v in per_env], float)
            n_eps = np.array([len(v) for v in per_env], float)
            rec = {"arm": arm, "seed": seed, "iter": it,
                   "reward_overall": float(mean_env.mean()),
                   "reward_easy": float(mean_env[easy_mask].mean()),
                   "reward_hard": float(mean_env[~easy_mask].mean()),
                   "per_step_overall": float(per_step.mean()),
                   "per_step_easy": float(per_step[easy_mask].mean()),
                   "per_step_hard": float(per_step[~easy_mask].mean()),
                   "mean_episodes_per_env": float(n_eps.mean()),
                   "n_envs": int(len(mean_env)),
                   "n_partial_episodes": int(n_partial)}
            results.append(rec)
            print("  %-8s s%-3d it=%-5d overall %7.3f  easy %7.3f  hard %7.3f"
                  "  per-step %6.4f  eps/env %4.2f  partial %3d"
                  % (arm, seed, it, rec["reward_overall"],
                     rec["reward_easy"], rec["reward_hard"],
                     rec["per_step_overall"], rec["mean_episodes_per_env"],
                     n_partial))

    out = a.out or os.path.join(root, "validation.json")
    json.dump(results, open(out, "w"), indent=2)
    print()
    print("wrote %s  (%d checkpoint evaluations)" % (out, len(results)))
    print("Aggregate with: python pisb_legged/aggregate_legged.py "
          "--validation %s" % out)


def class_to_dict(obj):
    """legged_gym's helper, inlined so this script does not depend on the exact
    module path it lives in across legged_gym versions."""
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


if __name__ == "__main__":
    main()
