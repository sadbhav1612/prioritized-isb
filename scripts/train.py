#!/usr/bin/env python3
"""Train a near-time-optimal racing policy.

Implements the training loop from "Autonomous Drone Racing with Deep
Reinforcement Learning" (Song et al., 2021):

- per-env progress / safety / body-rate reward (in C++ env)
- distributed initialization across path segments (in C++ env)
- random track curriculum: complexity rises as crash ratio falls
- single-track or random-track training selectable via --mode

The PPO trainer is the existing TF1 stable-baselines wrapper. Track
randomization and curriculum stepping happen through a PPO callback.
"""

import argparse
import io
import os

import numpy as np
import tensorflow as tf
from ruamel.yaml import YAML

from stable_baselines import logger
from stable_baselines.common.vec_env import VecNormalize

import rpg_baselines.common.util as U
from rpg_baselines.common.policies import MlpPolicy
from rpg_baselines.envs import racing_env_wrapper as racing_wrapper
from rpg_baselines.ppo.ppo2 import PPO2
from rpg_baselines.ppo.ppo2_test import test_model
from rpg_baselines.track import TrackGenerator

from flightgym import QuadrotorRacingEnv_v0


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--train', type=int, default=1,
                   help='1 to train, 0 to test a saved model')
    p.add_argument('--render', type=int, default=0)
    p.add_argument('--mode', choices=['fixed', 'random'], default='fixed',
                   help='fixed = single-track training (uses track yaml from '
                        'racing env config), random = curriculum-driven '
                        'random track training')
    p.add_argument('--num_envs', type=int, default=100)
    p.add_argument('--num_threads', type=int, default=10)
    p.add_argument('--total_timesteps', type=int, default=25_000_000)
    p.add_argument('--n_steps', type=int, default=250)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save_dir', type=str,
                   default=os.path.dirname(os.path.realpath(__file__)))
    p.add_argument('-w', '--weight', type=str,
                   default='./saved/quadrotor_racing.zip')
    # Curriculum knobs (random mode)
    p.add_argument('--crash_threshold', type=float, default=0.2,
                   help='Crash ratio under which complexity gets bumped up.')
    p.add_argument('--complexity_step', type=float, default=0.05)
    p.add_argument('--complexity_init', type=float, default=0.0)
    p.add_argument('--norm_obs', type=int, default=1,
                   help='1 = wrap with VecNormalize for online z-score '
                        'observation normalization (paper Sec III-B.2). '
                        'Strongly recommended; raw obs spans wildly different '
                        'scales (vel ~20, R ~1, omega ~10, sphere coords).')
    # Shaping anneal: linearly decay non-paper bootstrap rewards
    # (survival/gate_pass/pos_track/vel_track/align) from their yaml
    # values to 0 between anneal_start_frac and anneal_end_frac of
    # total_timesteps. Final phase trains under paper Eq 4 only.
    p.add_argument('--anneal_start_frac', type=float, default=0.0,
                   help='Fraction of total_timesteps at which to begin '
                        'annealing shaping coefficients (default 0 = '
                        'start immediately).')
    p.add_argument('--anneal_end_frac', type=float, default=0.6,
                   help='Fraction of total_timesteps at which shaping '
                        'coefficients reach 0 (default 0.6).')
    # Floor on gate_pass_bonus. Other shaping decays to 0 over the
    # anneal window; gate_pass_bonus is the only discrete "you cleared
    # gate N" signal and the policy needs it to learn chaining past
    # gate 3-4. 0 = paper-faithful (no floor); >0 keeps the signal
    # alive permanently.
    p.add_argument('--gate_pass_floor', type=float, default=50.0,
                   help='Minimum value for gate_pass_bonus even after '
                        'anneal reaches 0. 0 = strict paper Eq 4.')
    # Past-trajectory init (paper Sec III-C.2 phase 2). Ramps from 0 at
    # past_traj_ramp_start * total_timesteps up to past_traj_target at
    # past_traj_ramp_end * total_timesteps.
    p.add_argument('--past_traj_target', type=float, default=0.4,
                   help='Final probability of resampling reset state '
                        'from the past-trajectory buffer (0 = disable, '
                        'paper phase-2 default ~0.3-0.5).')
    p.add_argument('--past_traj_ramp_start', type=float, default=0.1,
                   help='Fraction of total_timesteps before past-traj '
                        'sampling starts ramping up (avoids feeding '
                        'the policy junk states during bootstrap).')
    p.add_argument('--past_traj_ramp_end', type=float, default=0.4,
                   help='Fraction of total_timesteps at which past-traj '
                        'sampling reaches past_traj_target.')
    p.add_argument('--prioritize_alpha', type=float, default=None,
                   help='P-ISB prioritization exponent for reset-state '
                        'sampling. None = use the yaml default; 0 = uniform '
                        '(Random-Buffer baseline); 0.4-1.0 = prioritized. '
                        'Set this per run for the P-ISB vs Random-Buffer A/B.')
    # --- P-ISB Milestone 2 (Python-owned buffer) -------------------
    p.add_argument('--pisb_python', type=int, default=0,
                   help='1 = enable the Milestone-2 Python-owned Prioritized '
                        'Initial State Buffer (exact |advantage| priorities + '
                        'diversity-preserving k-means sampling). Uses '
                        '--prioritize_alpha as the priority exponent.')
    p.add_argument('--pisb_pool', type=int, default=256,
                   help='Reset-pool size pushed to each env per refresh.')
    p.add_argument('--pisb_k', type=int, default=64,
                   help='k-means clusters for diversity-preserving sampling.')
    p.add_argument('--pisb_capacity', type=int, default=20000,
                   help='Python buffer capacity (visited states).')
    p.add_argument('--pisb_gate_min', type=int, default=1,
                   help='Only buffer states with gate_idx >= this.')
    return p


def _load_shaping_init():
    """Read the racing env yaml and capture initial shaping coefficients
    so the anneal callback can multiply them by a decay factor.
    """
    yaml = YAML(typ='unsafe', pure=True)
    path = os.path.join(os.environ['FLIGHTMARE_PATH'],
                        'flightlib/configs/quadrotor_racing_env.yaml')
    with open(path) as f:
        cfg = yaml.load(f)
    rl = cfg.get('rl', {}) or {}
    return dict(
        survival=float(rl.get('survival_bonus', 0.0)),
        gate_pass=float(rl.get('gate_pass_bonus', 0.0)),
        pos_track=float(rl.get('pos_track_coeff', 0.0)),
        vel_track=float(rl.get('vel_track_coeff', 0.0)),
        align=float(rl.get('align_coeff', 0.0)),
    )


def make_env(args):
    yaml = YAML(typ='unsafe', pure=True)
    cfg_path = os.path.join(os.environ['FLIGHTMARE_PATH'],
                            'flightlib/configs/vec_env.yaml')
    with open(cfg_path) as f:
        cfg = yaml.load(f)

    cfg['env']['num_envs'] = args.num_envs if args.train else 1
    cfg['env']['num_threads'] = args.num_threads if args.train else 1
    cfg['env']['render'] = 'yes' if args.render else 'no'

    buf = io.StringIO()
    yaml.dump(cfg, buf)
    impl = QuadrotorRacingEnv_v0(buf.getvalue(), False)
    inner = racing_wrapper.RacingEnvVec(impl)
    env = inner
    if getattr(args, 'norm_obs', 1):
        # Online z-score normalization (paper Sec III-B.2). Reward is left
        # raw so progress values don't get rescaled out of meaningful range.
        env = VecNormalize(env, norm_obs=True, norm_reward=False,
                           clip_obs=10.0, gamma=0.99)
    # Always return a handle to the inner racing env so curriculum / track
    # randomization callbacks can call racing-specific methods.
    return env, inner


def _log_gate_stats(racing_env, header=False):
    """Print + log per-iteration gate-passing stats. Stdout line shows
    counts you can eyeball; logger.logkv writes them into progress.csv.
    """
    stats = racing_env.iteration_stats()
    cum = racing_env.cumulative_stats()
    if header:
        print('  iter | episodes | gates_passed | avg_gates/ep | best_ep | '
              'crash_ratio | cum_max')
    print(f'         | {stats["episodes"]:>8d} | {stats["total_gates"]:>12d}'
          f' | {stats["avg_gates_per_ep"]:>12.2f}'
          f' | {stats["max_gates_in_ep"]:>7d}'
          f' | {stats["crash_ratio"]:>10.2%}'
          f' | {cum["cum_max_gates_in_ep"]:>7d}')
    logger.logkv_mean('iter/episodes', stats['episodes'])
    logger.logkv_mean('iter/gates_passed', stats['total_gates'])
    logger.logkv_mean('iter/avg_gates_per_ep', stats['avg_gates_per_ep'])
    logger.logkv_mean('iter/max_gates_in_ep', stats['max_gates_in_ep'])
    logger.logkv_mean('iter/crash_ratio', stats['crash_ratio'])
    logger.logkv('cum/total_gates', cum['cum_total_gates'])
    logger.logkv('cum/max_gates_in_ep', cum['cum_max_gates_in_ep'])
    # P-ISB: surface the buffer's priority distribution so prioritization is
    # visible during training. Wrapped so a build without the binding is a
    # no-op rather than a crash.
    try:
        ps = racing_env.buffer_priority_stats()
        logger.logkv('pisb/buf_priority_mean', ps['mean'])
        logger.logkv('pisb/buf_priority_max', ps['max'])
        logger.logkv('pisb/buf_fill', ps['fill'])
    except Exception:
        pass


def build_callback(racing_env, generator, args, shaping_init):
    """Per-update callback. Always logs gate stats. If a track generator
    is provided, also resamples tracks and steps the curriculum. Also
    linearly decays the non-paper shaping coefficients from their yaml
    values to 0 between args.anneal_start_frac and args.anneal_end_frac
    so the final training stage matches paper Eq 4.
    """

    # PPO update count corresponding to one full pass through
    # total_timesteps. Used to compute the anneal fraction.
    total_updates = max(1, args.total_timesteps // max(1, args.num_envs * args.n_steps))
    start_frac = max(0.0, min(1.0, args.anneal_start_frac))
    end_frac = max(start_frac + 1e-6, min(1.0, args.anneal_end_frac))

    state = {
        'last_resample_update': 0,
        'updates_per_resample': 1,
        'header_printed': False,
    }

    def _anneal_mult(update):
        frac = update / total_updates
        if frac <= start_frac:
            return 1.0
        if frac >= end_frac:
            return 0.0
        return 1.0 - (frac - start_frac) / (end_frac - start_frac)

    # Past-trajectory init ramp (paper Sec III-C.2 phase 2): 0 -> target
    # over [ramp_start, ramp_end] fraction of training. Held at target
    # afterward. Avoids feeding the policy junk mid-flight states
    # during bootstrap when the buffer would mostly contain crashes.
    pt_start = max(0.0, min(1.0, args.past_traj_ramp_start))
    pt_end = max(pt_start + 1e-6, min(1.0, args.past_traj_ramp_end))
    pt_target = max(0.0, min(1.0, args.past_traj_target))

    def _past_traj_prob(update):
        frac = update / total_updates
        if frac <= pt_start:
            return 0.0
        if frac >= pt_end:
            return pt_target
        return pt_target * (frac - pt_start) / (pt_end - pt_start)

    def _callback(local_vars, _global_vars):
        update = local_vars.get('update', 0)
        # Always log gate stats so the user can see if the policy is
        # making progress regardless of training mode.
        if not state['header_printed']:
            _log_gate_stats(racing_env, header=True)
            state['header_printed'] = True
        else:
            _log_gate_stats(racing_env, header=False)

        # Anneal non-paper shaping coefficients toward 0. mult=1 keeps
        # the yaml bootstrap values; mult=0 leaves only paper Eq 4
        # (progress + safety + body-rate + crash terminal).
        # gate_pass_bonus is floored at args.gate_pass_floor so the
        # discrete "you cleared gate N" signal stays alive even after
        # other shaping has fully decayed — required for the policy to
        # learn chaining past gate 3-4 on long tracks like AlphaPilot.
        mult = _anneal_mult(update)
        gate_pass = max(shaping_init['gate_pass'] * mult,
                        args.gate_pass_floor)
        racing_env.set_shaping_coeffs(
            shaping_init['survival'] * mult,
            gate_pass,
            shaping_init['pos_track'] * mult,
            shaping_init['vel_track'] * mult,
            shaping_init['align'] * mult,
        )
        logger.logkv_mean('anneal/mult', mult)
        logger.logkv_mean('anneal/gate_pass', gate_pass)

        # Ramp past-trajectory init probability per paper Sec III-C.2.
        pt_prob = _past_traj_prob(update)
        racing_env.set_past_traj_prob(pt_prob)
        logger.logkv_mean('past_traj/prob', pt_prob)

        # P-ISB Milestone 2 controller stats (Python-owned buffer).
        model = local_vars.get('self', None)
        pisb = getattr(model, 'pisb', None) if model is not None else None
        if pisb is not None:
            st = pisb.stats()
            logger.logkv('pisb/py_fill', st['fill'])
            logger.logkv('pisb/py_pool', st['pool'])
            logger.logkv('pisb/py_prio_mean', st['prio_mean'])
            logger.logkv('pisb/py_prio_max', st['prio_max'])

        crash_ratio = racing_env.crash_ratio()
        n_eps = racing_env.iteration_stats()['episodes']

        if generator is not None:
            logger.logkv_mean('complexity', generator.complexity)
            # bump complexity when we are confidently below threshold
            if n_eps >= max(racing_env.num_envs, 16) \
                    and crash_ratio < args.crash_threshold:
                generator.step_complexity(args.complexity_step)
            # resample tracks across all envs every iteration
            if update - state['last_resample_update'] \
                    >= state['updates_per_resample']:
                racing_env.randomize_tracks(generator)
                racing_env.reset()
                state['last_resample_update'] = update

        racing_env.reset_iteration_stats()
        return True

    return _callback


def main():
    args = parser().parse_args()
    env, racing_env = make_env(args)
    configure_random_seed(args.seed, env=env)

    # P-ISB: override the yaml prioritization exponent for this run if the
    # flag was given (the A/B knob). Left at the yaml default otherwise.
    if getattr(args, 'prioritize_alpha', None) is not None:
        racing_env.set_prioritize_alpha(args.prioritize_alpha)
        print(f"P-ISB prioritize_alpha = {args.prioritize_alpha}")

    if args.train:
        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(rsg_root, 'saved')
        saver = U.ConfigurationSaver(log_dir=log_dir)

        # Warm-start: if --weight points to a saved zip, load it as the
        # starting policy instead of training from scratch. Useful for
        # fine-tuning a curriculum-trained model on a specific track.
        ckpt = args.weight if (args.weight and os.path.exists(args.weight)) \
            else None
        if ckpt:
            print(f"Warm-starting from {ckpt}")
            # ent_coef must be passed through PPO2.load as a kwarg —
            # stable_baselines bakes ent_coef into the TF graph during
            # setup_model(), so a post-load assignment to model.ent_coef
            # has no effect on the loss. Loading with ent_coef in kwargs
            # makes __dict__.update apply it BEFORE setup_model() runs.
            # Paper uses ent_coef=0. Earlier we used 0.003 to keep
            # exploration alive while the policy bootstrapped past
            # gate 1. Once the policy reliably passes 2+ gates the
            # entropy bonus starts to dominate the (now smaller)
            # reward gradient and log_std grows unbounded — observed
            # as policy_entropy climbing 1.4 -> 3.2 -> 8.1 across
            # successive warm-starts. ent_coef=0 lets the policy
            # commit to a fast trajectory.
            model = PPO2.load(ckpt, env=env, tensorboard_log=saver.data_dir,
                              verbose=1, ent_coef=0.0)
            model.n_steps = args.n_steps
        else:
            model = PPO2(
                tensorboard_log=saver.data_dir,
                policy=MlpPolicy,
                policy_kwargs=dict(
                    net_arch=[dict(pi=[128, 128], vf=[128, 128])],
                    act_fun=tf.nn.tanh,  # paper-faithful (tanh policy head)
                ),
                env=env,
                lam=0.95,
                gamma=0.99,
                n_steps=args.n_steps,
                # Paper uses 0. We use 0.003 to keep enough exploration
                # to discover multi-gate chains while not letting PPO
                # blow up the action stddev. At 0.01 we measured σ→1.17
                # (above the [-1,1] clip range) and the policy never
                # committed to any strategy — entropy bonus dominated all
                # other gradients in the reward. 0.003 is the same value
                # used in the warm-start path below; it's what the env
                # author landed on empirically.
                ent_coef=0.003,
                learning_rate=3e-4,
                vf_coef=0.5,
                max_grad_norm=0.5,
                # 4 minibatches per epoch * 10 epochs = 40 SGD steps per
                # PPO update, vs the previous 1*10 = 10. At 25000 samples
                # per batch, nminibatches=1 was averaging across the
                # entire batch in one giant step — burning gradient
                # information. Stable-baselines default is 4.
                nminibatches=4,
                noptepochs=10,
                cliprange=0.2,
                verbose=1,
            )

        # P-ISB Milestone 2: attach the Python-owned buffer controller. The
        # PPO runner picks it up via model.pisb (no-op otherwise). C++-side
        # prioritize_alpha is forced to 0 so the pool is drawn uniformly —
        # all prioritization happens in Python here.
        if args.pisb_python:
            from rpg_baselines.pisb import PISBController
            alpha = 0.7 if args.prioritize_alpha is None else args.prioritize_alpha
            racing_env.set_prioritize_alpha(0.0)
            model.pisb = PISBController(
                racing_env, alpha=alpha, n_pool=args.pisb_pool,
                k_clusters=args.pisb_k, capacity=args.pisb_capacity,
                gate_min=args.pisb_gate_min, seed=args.seed)
            print(f"P-ISB (Python) enabled: alpha={alpha} pool={args.pisb_pool} "
                  f"k={args.pisb_k} capacity={args.pisb_capacity}")

        # Always attach a callback so we get gate-pass stats in stdout
        # and progress.csv. Track randomization is conditional on mode.
        generator = None
        if args.mode == 'random':
            generator = TrackGenerator()
            generator.set_complexity(args.complexity_init)
            # Track randomization touches racing-specific methods, so go
            # through the inner env (bypassing VecNormalize).
            racing_env.randomize_tracks(generator)
            env.reset()
        # Capture initial shaping coefficients from yaml so the anneal
        # callback can decay them toward 0.
        shaping_init = _load_shaping_init()
        callback = build_callback(racing_env, generator, args, shaping_init)

        logger.configure(folder=saver.data_dir)
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_dir=saver.data_dir,
            logger=logger,
        )
        model.save(saver.data_dir)
        # Persist the VecNormalize statistics so render/eval can apply the
        # same z-scoring as training.
        if hasattr(env, 'save_running_average'):
            env.save_running_average(saver.data_dir)
    else:
        model = PPO2.load(args.weight)
        # If a VecNormalize stats dir sits next to the checkpoint, load it.
        weight_dir = os.path.dirname(os.path.abspath(args.weight))
        if hasattr(env, 'load_running_average') \
                and os.path.exists(os.path.join(weight_dir, 'obs_rms.pkl')):
            env.load_running_average(weight_dir)
            env.training = False  # freeze stats during eval
        test_model(env, model, render=args.render)


if __name__ == '__main__':
    main()
