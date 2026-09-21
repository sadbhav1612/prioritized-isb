#!/usr/bin/env python3
"""Extract per-lap times from rollouts of a trained racing policy.

Racing's canonical metric is LAP TIME, not gate count. On the circular
multi-lap track, gates_passed is cumulative, so a full lap completes every N
gates. We record the sim-time at each gate pass and take the duration of each
"flying lap" (the time between completing gate kN and (k+1)N) — i.e. steady-
state laps, independent of where the episode happened to start.

Reports, over all flying laps pooled across the given rollouts:
  best (fastest) lap, median lap, #laps, and the speed distribution.

Usage:
  python3 scripts/lap_times.py -w runs/<run>/saved/<stamp>.zip \\
      --track flightlib/configs/SplitS.yaml --num_rollouts 50 [--json_out f.json]
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
from flightgym import QuadrotorRacingEnv_v0
from rpg_baselines.envs.racing_env_wrapper import RacingEnvVec
from rpg_baselines.ppo.ppo2 import PPO2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-w", "--weight", required=True)
    p.add_argument("--track", default="flightlib/configs/SplitS.yaml")
    p.add_argument("--num_rollouts", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=2500)
    p.add_argument("--num_envs", type=int, default=25)
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--sim_dt", type=float, default=0.02)
    p.add_argument("--json_out", default=None)
    return p.parse_args()


def n_gates(track):
    path = track if os.path.isabs(track) else os.path.join(FLIGHTMARE, track)
    return len(yaml.safe_load(open(path)).get("gates", {}))


def main():
    args = parse_args()
    yml = YAML(typ="unsafe", pure=True)
    cfg = yml.load(open(os.path.join(FLIGHTMARE, "flightlib/configs/vec_env.yaml")))
    cfg["env"]["num_envs"] = args.num_envs
    cfg["env"]["num_threads"] = args.num_threads
    cfg["env"]["render"] = "no"
    buf = io.StringIO(); yml.dump(cfg, buf)
    racing_env = RacingEnvVec(QuadrotorRacingEnv_v0(buf.getvalue(), False))
    track = args.track if os.path.isabs(args.track) else \
        os.path.join(FLIGHTMARE, args.track)
    for i in range(racing_env.num_envs):
        racing_env.wrapper.loadTrackFromFile(i, track)
    N = n_gates(args.track)

    model = PPO2.load(args.weight)
    wp = os.path.abspath(args.weight)
    cands = [os.path.dirname(wp), os.path.splitext(wp)[0]]
    rms = next((c for c in cands if os.path.exists(os.path.join(c, "obs_rms.pkl"))), None)
    if rms is None:
        print("ERROR: no obs_rms.pkl", file=sys.stderr); sys.exit(2)
    env = VecNormalize(racing_env, norm_obs=True, norm_reward=False,
                       clip_obs=10.0, gamma=0.99)
    env.load_running_average(rms); env.training = False

    passed_idx = racing_env._extraInfoNames.index("gates_passed")
    speeds, lap_times = [], []
    for r in range(args.num_rollouts):
        obs = env.reset()
        pass_steps = []
        last = 0
        for step in range(args.max_steps):
            act = model.step(obs, None, None, deterministic=True)[0]
            obs, _, done, _ = env.step(act)
            speeds.append(float(np.linalg.norm(racing_env._observation[0][0:3])))
            gp = int(racing_env._extraInfo[0, passed_idx])
            while gp > last:           # may pass >1 gate per logged step
                pass_steps.append(step)
                last += 1
            if done[0]:
                break
        # Flying laps: time between completing gate kN and (k+1)N. Skip the
        # first partial lap (start-dependent) by starting at the N-th pass.
        for k in range(1, len(pass_steps) // N):
            dt_steps = pass_steps[(k + 1) * N - 1] - pass_steps[k * N - 1]
            lap_times.append(dt_steps * args.sim_dt)

    lt = np.asarray(lap_times)
    sp = np.asarray(speeds)
    print(f"\nFlying laps: {len(lt)}")
    if len(lt):
        print(f"Lap time [s]  best={lt.min():.2f}  median={np.median(lt):.2f}  "
              f"mean={lt.mean():.2f}  p25={np.percentile(lt,25):.2f}")
    print(f"Speed [m/s]  median={np.median(sp):.2f}  p95={np.percentile(sp,95):.2f}  "
          f"max={sp.max():.2f}")
    if args.json_out:
        import json
        json.dump({
            "n_laps": int(len(lt)),
            "lap_best": float(lt.min()) if len(lt) else None,
            "lap_median": float(np.median(lt)) if len(lt) else None,
            "lap_mean": float(lt.mean()) if len(lt) else None,
            "speed_median": float(np.median(sp)),
            "speed_max": float(sp.max()),
        }, open(args.json_out, "w"), indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
