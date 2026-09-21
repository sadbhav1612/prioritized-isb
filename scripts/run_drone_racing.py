#!/usr/bin/env python3
"""Helpers shared by the P-ISB training runners.

Extracted from the racing training script that produced every result in the
paper. The five functions the runners use (parser, configure_random_seed,
_load_shaping_init, make_env, build_callback) are unchanged; flags and logging
branches that belong to other experiments were removed. The environment is the
Flightmare QuadrotorRacingEnv_v0 (C++, Python bindings `flightgym`).

The callback implements two schedules shared by every method:
  * the reset probability rho ramps linearly from 0 at --past_traj_ramp_start
    to --past_traj_target at --past_traj_ramp_end (fractions of training);
  * the shaping terms (survival, position/velocity tracking, alignment) decay
    linearly to 0 between --anneal_start_frac and --anneal_end_frac, and the
    gate-pass bonus to --gate_pass_floor.
"""
import argparse
import io
import json
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
    p.add_argument('--ckpt_every', type=int, default=0,
                   help='Save a checkpoint every this many PPO updates '
                        '(0 = only at the end).')
    # Kept only so run_pisb_ral.py can refuse it; the P-ISB protocol is
    # passed explicitly on the command line.
    p.add_argument('--paper_preset', type=int, default=0)
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


def _save_projection(pisb, stem):
    """Persist the contrastive projection next to a checkpoint.

    Nothing used to save it, so a finished run's zip held only the PPO policy
    and the embedding that run actually used was unrecoverable. Any later
    analysis of the embedding then had to RECONSTRUCT one by retraining a
    fresh projection on rollouts from the converged policy, which answers a
    different question. Writing ~50 kB per checkpoint removes that gap.

    Silent no-op for every non-cl embed, which has no projection.
    """
    proj = getattr(pisb, 'proj', None) if pisb is not None else None
    if proj is None:
        return
    try:
        proj.torch.save(proj.net.state_dict(), stem + '_clproj.pt')
        with open(stem + '_clproj.json', 'w') as f:
            json.dump(dict(getattr(pisb, '_cl', {}),
                           embed=getattr(pisb, 'embed', None)), f, indent=2)
    except Exception as exc:                     # never kill a run over a dump
        print(f"WARNING: could not save CL projection: {exc}")


def build_callback(racing_env, generator, args, shaping_init, save_dir=None):
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
            # CL-Buffer only: latest soft-nearest-neighbor loss.
            if st.get('cl_loss') is not None:
                logger.logkv('isb/cl_loss', st['cl_loss'])
            # Reset-coverage diagnostics, logged when the controller reports them.
            for k in ('reset_H', 'reset_eff', 'reset_gatecov', 'reset_pos_spread',
                      'prio_eff', 'prio_H'):
                if k in st:
                    logger.logkv('isb/' + k, st[k])

        # Periodic checkpoint. Runs used to save only at the very end, so
        # "how many env steps to reach success X" could only be answered with
        # iter/avg_gates_per_ep, which depends on the reset distribution and so
        # is not comparable across methods with different rho. With
        # intermediate checkpoints the question can be answered properly, by
        # evaluating each one from the start line.
        ckpt_every = int(getattr(args, 'ckpt_every', 0) or 0)
        if ckpt_every > 0 and save_dir and update > 0 and update % ckpt_every == 0:
            stem = os.path.join(save_dir, f'ckpt_{update:06d}')
            try:
                if model is not None:
                    model.save(stem)
                _save_projection(pisb, stem)
            except Exception as exc:
                print(f"WARNING: checkpoint at update {update} failed: {exc}")

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


