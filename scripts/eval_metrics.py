#!/usr/bin/env python3
"""Evaluate a saved racing policy and print the exact metrics needed for
the paper's Table~\ref{tab:eval}: mean / max gates per episode, crash
percentage, median + 95th-percentile per-step speed (m/s).

Usage:
  python3 eval_metrics.py -w ./saved/splitS.zip \\
      --track flightlib/configs/SplitS.yaml --num_rollouts 60
  python3 eval_metrics.py -w ./saved/alpha.zip \\
      --track flightlib/configs/AlphaPilot.yaml --num_rollouts 60
"""
import argparse
import io
import os
import sys

import numpy as np
import yaml
from ruamel.yaml import YAML

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
FLIGHTRL = os.path.dirname(THIS_DIR)
FLIGHTMARE = os.path.dirname(FLIGHTRL)
sys.path.insert(0, os.path.join(FLIGHTMARE, "flightlib", "build"))
sys.path.insert(0, FLIGHTRL)
os.environ.setdefault("FLIGHTMARE_PATH", FLIGHTMARE)

from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common.misc_util import set_global_seeds

from flightgym import QuadrotorRacingEnv_v0
from rpg_baselines.envs.racing_env_wrapper import RacingEnvVec
from rpg_baselines.ppo.ppo2 import PPO2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-w", "--weight", required=True, help="PPO2 .zip")
    p.add_argument("--track", default=None,
                   help="Track yaml (default: whatever quadrotor_racing_env.yaml points at)")
    p.add_argument("--num_rollouts", type=int, default=60)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions stochastically (default: deterministic).")
    p.add_argument("--num_envs", type=int, default=100)
    p.add_argument("--num_threads", type=int, default=10)
    p.add_argument("--success_gates", type=int, default=None,
                   help="Gate count counted as a success (default: n_gates = "
                        "one full lap). Use a higher value for a "
                        "discriminating success metric on multi-lap tracks.")
    p.add_argument("--json_out", default=None,
                   help="If set, write all metrics to this JSON file "
                        "(for the comparison-grid plotter).")
    p.add_argument("--from_start", action="store_true",
                   help="Reset every rollout deterministically from the START "
                        "LINE (clean success rate), instead of the training "
                        "70%% mid-track random init.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed python/numpy/tensorflow and the env's C++ RNG "
                        "before loading the policy, so that --stochastic runs "
                        "reproduce bit-for-bit. Default (unset): every "
                        "invocation draws fresh action noise, so re-scoring "
                        "the same checkpoint gives a different 50-rollout "
                        "sample each time (expected Monte-Carlo variance).")
    return p.parse_args()


def predict_action(model, obs, deterministic):
    return model.step(obs, None, None, deterministic=deterministic)[0]


def make_env(num_envs, num_threads):
    yml = YAML(typ="unsafe", pure=True)
    cfg_path = os.path.join(FLIGHTMARE, "flightlib/configs/vec_env.yaml")
    with open(cfg_path) as f:
        cfg = yml.load(f)
    cfg["env"]["num_envs"] = num_envs
    cfg["env"]["num_threads"] = num_threads
    cfg["env"]["render"] = "no"
    buf = io.StringIO()
    yml.dump(cfg, buf)
    impl = QuadrotorRacingEnv_v0(buf.getvalue(), False)
    return RacingEnvVec(impl)


def load_track(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    n = sum(1 for _ in data.get("gates", {}).values())
    return n


def main():
    args = parse_args()
    print(f"Model: {args.weight}")

    if args.seed is not None:
        # Must happen before PPO2.load() builds the TF graph: the Gaussian
        # policy's sample() op bakes in its RNG seed at graph-construction
        # time, so seeding afterwards would not affect it.
        set_global_seeds(args.seed)

    racing_env = make_env(args.num_envs, args.num_threads)
    racing_env._from_start = bool(args.from_start)  # clean start-line resets
    if args.seed is not None:
        racing_env.wrapper.setSeed(args.seed)

    if args.track is not None:
        track_path = args.track
        if not os.path.isabs(track_path):
            track_path = os.path.join(FLIGHTMARE, track_path)
        for i in range(racing_env.num_envs):
            racing_env.wrapper.loadTrackFromFile(i, track_path)
    else:
        with open(os.path.join(
            FLIGHTMARE,
            "flightlib/configs/quadrotor_racing_env.yaml")) as f:
            rcfg = yaml.safe_load(f)
        track_path = rcfg.get("track", {}).get(
            "yaml_path", "flightlib/configs/CircularLoop.yaml")
        if not os.path.isabs(track_path):
            track_path = os.path.join(FLIGHTMARE, track_path)

    n_gates = load_track(track_path)
    print(f"Track: {track_path}  ({n_gates} gates)")

    model = PPO2.load(args.weight)

    weight_path = os.path.abspath(args.weight)
    weight_dir = os.path.dirname(weight_path)
    weight_stem = os.path.splitext(os.path.basename(weight_path))[0]
    candidates = [weight_dir,
                  os.path.join(weight_dir, weight_stem),
                  weight_path[:-len('.zip')]
                  if weight_path.endswith('.zip') else None]
    obs_rms_dir = next((c for c in candidates
                        if c and os.path.exists(
                            os.path.join(c, "obs_rms.pkl"))), None)
    if obs_rms_dir is None:
        print("ERROR: no obs_rms.pkl found, policy will see raw obs",
              file=sys.stderr)
        sys.exit(2)
    env = VecNormalize(racing_env, norm_obs=True, norm_reward=False,
                       clip_obs=10.0, gamma=0.99)
    env.load_running_average(obs_rms_dir)
    env.training = False
    print(f"obs_rms from: {obs_rms_dir}")

    cur_idx = racing_env._extraInfoNames.index("current_gate")
    passed_idx = racing_env._extraInfoNames.index("gates_passed")
    crashed_idx = racing_env._extraInfoNames.index("gate_crashed")
    box_idx = (racing_env._extraInfoNames.index("box_exit")
               if "box_exit" in racing_env._extraInfoNames else None)

    per_ep_gates = []
    per_ep_crash = []
    per_ep_box = []
    per_ep_len = []
    all_speeds = []
    all_bodyrate = []  # ||omega|| each step -> smoothness proxy (lower=smoother)

    for r in range(args.num_rollouts):
        obs = env.reset()
        last_passed = 0
        crashed = False
        box_exit = False
        for step in range(args.max_steps):
            act = predict_action(model, obs,
                                 deterministic=not args.stochastic)
            obs, _, done, _ = env.step(act)
            raw = racing_env._observation[0]
            all_speeds.append(float(np.linalg.norm(raw[0:3])))
            # obs layout: [v(3), a(3), R(9), w(3), gates...] -> w = raw[15:18]
            all_bodyrate.append(float(np.linalg.norm(raw[15:18])))
            last_passed = int(racing_env._extraInfo[0, passed_idx])
            if done[0]:
                crashed = bool(racing_env._extraInfo[0, crashed_idx] > 0.5)
                if box_idx is not None:
                    box_exit = bool(racing_env._extraInfo[0, box_idx] > 0.5)
                break
        per_ep_gates.append(last_passed)
        per_ep_crash.append(int(crashed))
        per_ep_box.append(int(box_exit))
        per_ep_len.append(step + 1)

    g = np.asarray(per_ep_gates)
    box = np.asarray(per_ep_box)
    br = np.asarray(all_bodyrate)
    s = np.asarray(all_speeds)
    crash = np.asarray(per_ep_crash)
    L = np.asarray(per_ep_len)

    print()
    print("=" * 60)
    print(f"Rollouts : {len(g)}  (stochastic={args.stochastic}, "
          f"seed={args.seed})")
    print(f"Gates / ep  mean={g.mean():.2f}   max={g.max()}   "
          f"min={g.min()}   std={g.std():.2f}")
    # Success rate = fraction of rollouts reaching the success threshold
    # (default = n_gates = one full lap). This is the paper's headline
    # drone-racing metric; raise --success_gates for a discriminating
    # threshold on multi-lap tracks.
    succ_gates = args.success_gates if args.success_gates else n_gates
    n_complete = int((g >= succ_gates).sum())
    success_rate = 100.0 * (g >= succ_gates).mean()
    print(f"Success rate (>={succ_gates} gates): "
          f"{success_rate:.1f}%  ({n_complete}/{len(g)})")
    # Full termination breakdown — "crash ratio" alone is misleading because
    # a collapsed policy ends by FLYING OFF (box_exit), not a gate crash, and
    # a smooth policy ends by timeout (neither). clean = neither crash nor box.
    clean = 1.0 - np.clip(crash + box, 0, 1)
    print(f"Crash ratio (gate): {100*crash.mean():.1f}%   "
          f"Box-exit (flew off): {100*box.mean():.1f}%   "
          f"Clean end (timeout/laps): {100*clean.mean():.1f}%")
    # Smoothness proxy: mean body-rate magnitude (rad/s). Lower = smoother.
    print(f"Body-rate ||w|| (smoothness, lower=smoother): "
          f"mean={br.mean():.2f}  median={np.median(br):.2f}  "
          f"p95={np.percentile(br,95):.2f} rad/s")
    print(f"Episode length  mean={L.mean():.1f} steps  "
          f"({L.mean()*0.02:.2f} s)")
    print(f"Speed [m/s]  median={np.median(s):.2f}   "
          f"p95={np.percentile(s, 95):.2f}   "
          f"max={s.max():.2f}   mean={s.mean():.2f}")
    print()
    print("LaTeX table row:")
    track_name = os.path.basename(track_path).replace(".yaml", "")
    print(f"  {track_name:12s} & {n_gates} & "
          f"{g.mean():.2f} & {g.max()} & "
          f"{100*crash.mean():.1f} & "
          f"{np.median(s):.2f} / {np.percentile(s, 95):.2f} \\\\")

    if args.json_out:
        import json
        metrics = {
            "track": track_name, "n_gates": int(n_gates),
            "success_gates": int(succ_gates), "rollouts": int(len(g)),
            "stochastic": bool(args.stochastic), "seed": args.seed,
            "success_rate": float(success_rate),
            "gates_mean": float(g.mean()), "gates_max": float(g.max()),
            "gates_std": float(g.std()),
            "crash_pct": float(100 * crash.mean()),
            "box_exit_pct": float(100 * box.mean()),
            "clean_pct": float(100 * (1.0 - np.clip(crash + box, 0, 1)).mean()),
            "bodyrate_mean": float(br.mean()),
            "bodyrate_p95": float(np.percentile(br, 95)),
            "speed_median": float(np.median(s)),
            "speed_p95": float(np.percentile(s, 95)),
            "ep_time_s": float(L.mean() * 0.02),
            # per-rollout gates reached: lets any downstream analysis compute
            # SR at an arbitrary gate/lap threshold as (gates >= thr).mean().
            "gates_per_rollout": [int(x) for x in g],
        }
        with open(args.json_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote metrics JSON to {args.json_out}")


if __name__ == "__main__":
    main()
