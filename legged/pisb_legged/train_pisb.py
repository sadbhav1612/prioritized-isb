# SPDX-License-Identifier: GPL-3.0
# Uses the initial_buffer library of cl_initial_buffer
# (https://github.com/uzh-rpg/cl_initial_buffer, GNU GPL v3.0); this package
# is distributed under the same licence (LICENSE-GPL-3.0).
"""Entry point for the legged-locomotion experiments.

The reference paper validates the CL-Buffer on this domain (Rudin et al.
legged_gym, IsaacGym, 1024 environments, 2000 iterations), and this script
reproduces that setup.

IMPORT ORDER IS LOAD-BEARING. IsaacGym refuses to initialise if torch was
imported first ("PyTorch was imported before isaacgym modules"), and
pisb_legged.state_buffer imports torch at module scope. isaacgym therefore comes
first here and the package imports come last; do not reorder.

Usage
  python pisb_legged/train_pisb.py --pisb_arm pisb --seed 0 --headless \\
      --task anymal_c_rough --max_iterations 2000
  python pisb_legged/train_pisb.py --pisb_arm vanilla --seed 0 --headless

Arms: vanilla (no buffer, stock rsl_rl), random, obs, cl, pisb.
`random`, `obs` and `cl` are the authors' own three sampling strategies, run
through their installed `initial_buffer` library unmodified.
"""
import isaacgym                                            # noqa: F401  FIRST
from legged_gym import LEGGED_GYM_ROOT_DIR                 # noqa: F401
from legged_gym.envs import *                              # noqa: F401,F403
from legged_gym.envs.anymal_c.anymal import Anymal
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict         # noqa: F401
from isaacgym import gymutil

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pisb_legged.integration import (                      # noqa: E402
    ISBLeggedRobot, ISBPPO, ISBOnPolicyRunner)
from pisb_legged.state_buffer import ARMS                  # noqa: E402


class ISBAnymal(ISBLeggedRobot, Anymal):
    """Anymal with the ISB reset override.

    Both parents derive from LeggedRobot, so the MRO puts ISBLeggedRobot.reset_idx
    first and its super() call lands in Anymal, then LeggedRobot. Nothing in
    either upstream class is edited.
    """
    pass


PISB_ARGS = [
    {"name": "--pisb_arm", "type": str, "default": "pisb",
     "help": "vanilla | random | obs | cl | pisb"},
    {"name": "--pisb_alpha", "type": float, "default": 0.7,
     "help": "priority exponent, pisb arm only"},
    {"name": "--pisb_p", "type": float, "default": 0.8,
     "help": "probability an episode starts from the ISB (paper p = 0.8)"},
    {"name": "--pisb_k", "type": int, "default": 64,
     "help": "k-means clusters (paper K = 64)"},
    {"name": "--pisb_n", "type": int, "default": 256,
     "help": "states added to the ISB per rollout (paper N = 256)"},
    {"name": "--pisb_capacity", "type": int, "default": 200000,
     "help": "visited-state buffer capacity"},
    {"name": "--pisb_start_row", "type": int, "default": -1,
     "help": "terrain ROW (difficulty level) of the single-start patch; -1 is "
             "the middle of the grid. The middle of legged_gym's grid is "
             "pyramid_stairs DESCENDING with 14 cm steps and a 3 m platform at "
             "its centre, which traps the robot on that platform: measured "
             "spread over 400 rollout phases is 3.2 m for every arm, buffer or "
             "not. The reference's robot 'needs to explore the other terrain "
             "types', which requires a patch it can walk off, so a flat start "
             "(row 0, col 0 is pyramid_sloped at difficulty 0) is the setting "
             "in which an initial-state buffer can carry the robot outward"},
    {"name": "--pisb_start_col", "type": int, "default": -1,
     "help": "terrain COL (type) of the single-start patch; -1 is the middle"},
    {"name": "--pisb_single_start", "type": int, "default": 0,
     "help": "1 = pin EVERY environment to one terrain patch for training, "
             "reproducing the reference paper's setup: 'the robot is "
             "initialized in the middle of the whole environment and needs to "
             "explore the other terrain types'. legged_gym does NOT do this by "
             "default -- it assigns terrain_types by env index, so all 20 types "
             "are covered from step one and the curriculum walks each robot "
             "through the difficulty levels. Under that default every arm "
             "already gets full terrain exposure and an initial-state buffer "
             "has nothing left to contribute, which is the most likely reason "
             "our locomotion numbers invert the reference's. With this flag the "
             "buffer becomes the ONLY source of terrain diversity, as in [3]."},
    {"name": "--pisb_min_ts", "type": int, "default": 15,
     "help": "PRE-SELECTION (paper Sec V): drop visited states closer than this "
             "to their episode start. Distinct from --pisb_cl_min_ts."},
    {"name": "--pisb_cl_min_ts", "type": int, "default": 8,
     "help": "ProjectionBuffer min_timesteps: the minimum SUB-TRAJECTORY length "
             "create_train_data will accept (toy_example.py uses 8). With "
             "num_steps_per_env = 24, setting this to 15 leaves only 9 usable "
             "steps and the sub-trajectory search almost always fails."},
    {"name": "--pisb_prio_raw", "type": int, "default": 1,
     "help": "1 = priority |A| from raw GAE (returns - values), as the paper "
             "states and as drone racing uses. 0 = the normalised "
             "storage.advantages, |A - mean|/std, which produced the runs "
             "reported before 2026-09-14."},
    {"name": "--pisb_fifo_uniform", "type": int, "default": 1,
     "help": "1 = when a rollout exceeds --pisb_capacity keep a uniform "
             "subsample. 0 = keep the tail, which drops the first timesteps of "
             "every rollout; that produced the runs reported before 2026-09-14."},
    {"name": "--pisb_reset_all", "type": int, "default": 1,
     "help": "1 = after an ISB restore, write root and DOF state to the sim for "
             "ALL envs reset that step in one call. 0 = write only the restored "
             "envs, which makes IsaacGym drop every nominal reset in the same "
             "step; that produced the runs made before 2026-09-15."},
    {"name": "--pisb_onmap_filter", "type": int, "default": 1,
     "help": "1 = only states whose base is 5 m inside the terrain grid and not "
             "below the ground may enter the visited buffer. 0 = no such check: "
             "restore chains drift robots off the map, where they free-fall; "
             "that produced the runs made before 2026-09-15."},
    {"name": "--pisb_tag", "type": str, "default": "",
     "help": "suffix for the run name and log dir, e.g. '_prio'. Needed to run "
             "two variants of one arm at one seed: without it both write "
             "logs/<exp>/<arm>_s<seed>, and run_local.sh deletes that directory "
             "before training, so the second variant erases the first."},
    {"name": "--pisb_out", "type": str, "default": "",
     "help": "directory for the run fragment; defaults to the log dir"},
]


def get_isb_args():
    """legged_gym's get_args() with the --pisb_* flags appended.

    get_args() hardcodes its custom_parameters list and takes no arguments, so
    the list is reproduced here plus ours and handed to the same
    gymutil.parse_arguments. The post-parse device fixups are copied verbatim
    from helpers.get_args; omitting them leaves sim_device unset and the env
    builds on the wrong device.
    """
    custom = [
        {"name": "--task", "type": str, "default": "anymal_c_rough"},
        {"name": "--resume", "action": "store_true", "default": False},
        {"name": "--experiment_name", "type": str},
        {"name": "--run_name", "type": str},
        {"name": "--load_run", "type": str},
        {"name": "--checkpoint", "type": int},
        {"name": "--headless", "action": "store_true", "default": False},
        {"name": "--horovod", "action": "store_true", "default": False},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {"name": "--num_envs", "type": int},
        {"name": "--seed", "type": int, "default": 0},
        {"name": "--max_iterations", "type": int},
    ] + PISB_ARGS

    args = gymutil.parse_arguments(description="P-ISB legged locomotion",
                                   custom_parameters=custom)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    if args.rl_device.lower() == "gpu":
        args.rl_device = "cuda:0"
    return args


def disable_tf32():
    """Force full FP32 on Ampere GPUs.

    torch 1.10 enables TF32 for matmul and cuDNN by default on compute
    capability >= 8.0 (the RTX 3060 is 8.6), so every Linear layer runs in
    reduced precision. Called before any CUDA work so every run on every GPU
    computes in the same precision.
    """
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.backends.cuda.matmul.allow_tf32


def main():
    tf32 = disable_tf32()
    args = get_isb_args()
    if args.pisb_arm not in ARMS:
        sys.exit("ABORT: --pisb_arm must be one of %s" % (ARMS,))

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    train_cfg.runner.run_name = "%s_s%d%s" % (args.pisb_arm, args.seed, args.pisb_tag)

    # The curriculum has to go BEFORE make_env: _get_env_origins reads it while
    # building the sim, and with it off legged_gym stops calling
    # _update_terrain_curriculum on reset, so a pinned robot stays pinned.
    if args.pisb_single_start:
        env_cfg.terrain.curriculum = False

    # Register an ISB-enabled variant of the requested task rather than mutating
    # the existing registration, so a stock run is still one command away.
    isb_task = "%s_isb" % args.task
    task_registry.register(isb_task, ISBAnymal, env_cfg, train_cfg)

    env, env_cfg = task_registry.make_env(name=isb_task, args=args,
                                          env_cfg=env_cfg)

    if args.pisb_single_start:
        # Pin every environment to ONE patch, the centre of the terrain grid.
        # Turning the curriculum off is not enough on its own: legged_gym then
        # spreads terrain_levels uniformly over ALL rows instead of 0..5, which
        # gives MORE terrain exposure rather than less. terrain_types are
        # assigned by env index either way. Both have to be overridden for the
        # robot to start "in the middle of the whole environment".
        lvl = (env_cfg.terrain.num_rows // 2 if args.pisb_start_row < 0
               else min(args.pisb_start_row, env_cfg.terrain.num_rows - 1))
        typ = (env_cfg.terrain.num_cols // 2 if args.pisb_start_col < 0
               else min(args.pisb_start_col, env_cfg.terrain.num_cols - 1))
        print("[ISB] single start pinned to terrain patch (row %d, col %d)"
              % (lvl, typ), flush=True)
        env.terrain_levels[:] = lvl
        env.terrain_types[:] = typ
        env.env_origins[:] = env.terrain_origins[lvl, typ]
        print("PISB-LEGGED SINGLE START: every env pinned to terrain "
              "(level %d, type %d) of a %dx%d grid; curriculum disabled"
              % (lvl, typ, env_cfg.terrain.num_rows, env_cfg.terrain.num_cols))

    # rsl_rl resolves its algorithm class with a bare eval() of the config
    # string, evaluated in on_policy_runner's OWN module globals. So the class
    # has to be injected there specifically; putting it in rsl_rl.algorithms is
    # not enough. This is a runtime attribute set, not an edit to their file.
    train_cfg.runner.algorithm_class_name = "ISBPPO"
    import rsl_rl.runners.on_policy_runner as _opr
    _opr.ISBPPO = ISBPPO

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    log_dir = os.path.join(log_root, "%s_s%d%s" % (args.pisb_arm, args.seed,
                                                   args.pisb_tag))
    runner = ISBOnPolicyRunner(env, _train_cfg_dict(train_cfg), log_dir,
                               device=args.rl_device)
    runner.init_isb(arm=args.pisb_arm, alpha=args.pisb_alpha,
                    p_reset=args.pisb_p, n_isb=args.pisb_n,
                    k_clusters=args.pisb_k, capacity=args.pisb_capacity,
                    seed=args.seed, min_timesteps=args.pisb_min_ts,
                    cl_min_timesteps=args.pisb_cl_min_ts,
                    prio_raw=bool(args.pisb_prio_raw),
                    fifo_uniform=bool(args.pisb_fifo_uniform),
                    reset_all=bool(args.pisb_reset_all),
                    onmap_filter=bool(args.pisb_onmap_filter))

    print("PISB-LEGGED EFFECTIVE: arm=%s alpha=%s K=%d N=%d p=%s g_min_ts=%d "
          "envs=%d obs_dim=%d" %
          (args.pisb_arm, args.pisb_alpha, args.pisb_k, args.pisb_n,
           env.isb_prob, args.pisb_min_ts, env.num_envs, env.num_obs))

    iters = args.max_iterations or train_cfg.runner.max_iterations
    runner.learn(num_learning_iterations=iters, init_at_random_ep_len=True)

    frag_dir = args.pisb_out or log_dir
    os.makedirs(frag_dir, exist_ok=True)
    with open(os.path.join(frag_dir, "fragment.json"), "w") as f:
        json.dump({"arm": args.pisb_arm, "seed": args.seed, "task": args.task,
                   "alpha": args.pisb_alpha, "K": args.pisb_k, "N": args.pisb_n,
                   "p": float(env.isb_prob), "min_ts": args.pisb_min_ts,
                   "tf32": int(bool(tf32)),
                   "prio_raw": int(args.pisb_prio_raw),
                   "fifo_uniform": int(args.pisb_fifo_uniform),
                   "reset_all": int(args.pisb_reset_all),
                   "onmap_filter": int(args.pisb_onmap_filter),
                   "iters": iters, "num_envs": env.num_envs,
                   "log_dir": log_dir}, f, indent=2)
    print("wrote %s/fragment.json" % frag_dir)


def _train_cfg_dict(train_cfg):
    from legged_gym.utils.helpers import class_to_dict
    return class_to_dict(train_cfg)


if __name__ == "__main__":
    main()
