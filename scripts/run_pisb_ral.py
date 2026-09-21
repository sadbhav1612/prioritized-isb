#!/usr/bin/env python3
"""Training entry point for P-ISB and the baseline reset-selection arms.

Reuses the argument parser, environment builder and training callback of
scripts/run_drone_racing.py by import, and adds three things:

  1. The buffer. `model.pisb` is a PISBRALController from rpg_baselines/pisb_ral.
     The PPO runner calls model.pisb.update(...) after computing the GAE
     advantages and before the gradient step (see INTEGRATION.md).

  2. The log directory, `args.save_dir/saved`.

  3. Optionally, a decay of the reset probability rho back toward zero late in
     training (--pr_rho_decay_start/_end), one of the ablations in the paper.
     The shared callback ramps rho up to its target and holds; the decay
     wrapper overwrites rho after it runs.

Arms (--pr_arm):

  vanilla  never resets from a buffer; every reset comes from the environment's
           base distribution (30% at the race start, 70% along a randomly
           chosen gate segment), which every other arm also uses for the
           resets it does not draw from its buffer.
  random   uniform draw from the visited set        (reference 'random')
  obs      cosine k-means on raw observations, n/K nearest each centroid
                                                    (reference 'observations')
  cl       the same on the learned contrastive embedding
                                                    (reference 'network')
  pisb     P-ISB: z-scored observations, Euclidean k-means, then a draw within
           each cluster with probability proportional to (|A|+1e-3)^alpha
  signed   P-ISB with max(A, 0) in place of |A|
  refresh  P-ISB with every stored priority re-scored against the current
           critic each time the pool is rebuilt

`obs` and `cl` are the reference's own two sampling strategies and differ ONLY
in the projection, which isolates what the learned embedding buys. P-ISB at
alpha=0 (`--pr_arm pisb --pr_alpha 0`) is a different estimator from `obs`: a
different normalisation, a different k-means and a random within-cluster draw.

Usage (one Split-S run of the main comparison)
  python scripts/run_pisb_ral.py --pr_arm pisb --pr_alpha 0.7 --seed 0 \
      --num_envs 100 --num_threads 10 --n_steps 250 \
      --total_timesteps 150000000 --mode fixed --norm_obs 1 \
      --past_traj_target 0.4 --past_traj_ramp_start 0.1 --past_traj_ramp_end 0.4 \
      --pr_pool 256 --pr_k 64 --pr_capacity 20000 --pr_gate_min 1 \
      --ckpt_every 500 --save_dir runs/pisb_s0
"""

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_drone_racing as rdr                       # noqa: E402

from stable_baselines import logger                  # noqa: E402
import rpg_baselines.common.util as U                # noqa: E402
from rpg_baselines.common.policies import MlpPolicy  # noqa: E402
from rpg_baselines.ppo.ppo2 import PPO2              # noqa: E402
from rpg_baselines.track import TrackGenerator       # noqa: E402
from rpg_baselines.pisb_ral import PISBRALController, ARMS  # noqa: E402


def parser():
    """Shared parser plus the P-ISB flags. Every added flag carries a --pr_
    prefix so it can never collide with a flag the shared runner adds later."""
    p = rdr.parser()
    g = p.add_argument_group('P-ISB')
    g.add_argument('--pr_arm', choices=list(ARMS), default='pisb',
                   help='Reset-selection arm. See the module docstring.')
    g.add_argument('--pr_alpha', type=float, default=0.7,
                   help='Priority exponent alpha for --pr_arm pisb. 0 recovers '
                        'a uniform within-cluster draw, which is the submitted '
                        "paper's observation-clustering baseline.")
    g.add_argument('--pr_pool', type=int, default=256,
                   help='n: reset-pool size pushed to the envs each update.')
    g.add_argument('--pr_k', type=int, default=64,
                   help='K: number of clusters.')
    g.add_argument('--pr_capacity', type=int, default=20000,
                   help='N: capacity of the visited-state FIFO V.')
    g.add_argument('--pr_gate_min', type=int, default=1,
                   help='g_min: only buffer states at or past this gate index. '
                        '0 disables the pre-selection filter entirely.')
    g.add_argument('--pr_refresh_every', type=int, default=1,
                   help='Rebuild the reset pool every this many PPO updates.')
    # --- reference CL constants; defaults ARE the published values -----------
    g.add_argument('--pr_cl_embed_dim', type=int, default=64)
    g.add_argument('--pr_cl_hidden', type=int, default=64)
    g.add_argument('--pr_cl_temp', type=float, default=0.5)
    g.add_argument('--pr_cl_mining', type=int, default=10,
                   help='nr_mining_samples: size of the positive and negative '
                        'sets mined per contrastive step.')
    g.add_argument('--pr_cl_lr', type=float, default=1e-4)
    g.add_argument('--pr_cl_train_data', type=int, default=256,
                   help='n_train_data: candidate states mined among per step.')
    g.add_argument('--pr_cl_delta', type=int, default=0,
                   help='1 = mine on the CHANGE in improvement between steps '
                        'rather than its level. Identical to the level under a '
                        'once-per-rollout schedule, which is what this runner '
                        'uses; see rpg_baselines/pisb_ral/ref_cl.py.')
    # --- optional: anneal rho back toward zero late in training --------------------
    g.add_argument('--pr_rho_decay_start', type=float, default=1.0,
                   help='Fraction of training at which rho starts decaying '
                        'toward 0. Default 1.0 = no decay, reproducing the '
                        'submitted schedule exactly.')
    g.add_argument('--pr_rho_decay_end', type=float, default=1.0,
                   help='Fraction of training at which rho reaches 0.')
    return p


def _wrap_rho_decay(inner_cb, racing_env, args, total_updates):
    """Wrap the shared callback so rho decays after it has been set.

    The shared callback computes its ramped rho and calls set_past_traj_prob.
    Running afterwards and setting it again is enough to override, and leaves
    the shared code untouched.
    """
    d0 = max(0.0, min(1.0, args.pr_rho_decay_start))
    d1 = max(d0 + 1e-6, min(1.0, args.pr_rho_decay_end))
    target = max(0.0, min(1.0, args.past_traj_target))

    def _cb(local_vars, global_vars):
        out = inner_cb(local_vars, global_vars)
        update = local_vars.get('update', 0)
        frac = update / max(1, total_updates)
        if frac > d0:
            rho = target * max(0.0, 1.0 - (frac - d0) / (d1 - d0))
            racing_env.set_past_traj_prob(rho)
            logger.logkv_mean('past_traj/prob_decayed', rho)
        return out

    return _cb


def _wrap_buffer_stats(inner_cb):
    """Log the FIFO turnover statistics the shared callback drops.

    VisitedBuffer.stats() computes `buf_added` and `buf_turnover_iters`, but
    run_drone_racing.py logs only fill/pool/prio_mean/prio_max plus a few
    guarded keys, so both are discarded. buf_turnover_iters is the staleness
    bound of the stored priorities -- capacity divided by the current write rate,
    i.e. how many iterations the buffer needs to replace itself -- and it is
    not recoverable after the fact, so it has to be logged while training runs.

    """
    def _cb(local_vars, global_vars):
        out = inner_cb(local_vars, global_vars)
        # Same handle the shared callback uses: PPO2.learn passes its own
        # `self` through locals().
        pisb = getattr(local_vars.get('self', None), 'pisb', None)
        if pisb is not None:
            st = pisb.stats()
            for k in ('buf_added', 'buf_turnover_iters'):
                if k in st:
                    logger.logkv('pisb/' + k, st[k])
        return out

    return _cb


def main():
    args = parser().parse_args()
    if args.paper_preset:
        sys.exit("ABORT: --paper_preset is not used by this runner; pass the "
                 "protocol explicitly (see README.md).")
    if not args.train:
        sys.exit("ABORT: this runner trains only. Evaluate with "
                 "scripts/eval_metrics.py and scripts/lap_times.py.")
    if not args.save_dir:
        sys.exit("ABORT: --save_dir is required so packed jobs stay separate.")

    env, racing_env = rdr.make_env(args)
    rdr.configure_random_seed(args.seed, env=env)

    # This runner owns its log dir, so it does not matter whether the shared
    # runner on this machine honours --save_dir.
    log_dir = os.path.join(args.save_dir, 'saved')
    saver = U.ConfigurationSaver(log_dir=log_dir)

    model = PPO2(
        tensorboard_log=saver.data_dir,
        policy=MlpPolicy,
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])],
            act_fun=tf.nn.tanh,
        ),
        env=env,
        lam=0.95,
        gamma=0.995,
        n_steps=args.n_steps,
        ent_coef=0.003,
        learning_rate=3e-4,
        vf_coef=0.5,
        max_grad_norm=0.5,
        nminibatches=4,
        noptepochs=10,
        cliprange=0.2,
        verbose=1,
    )

    # All reset selection happens in Python, so the C++ prioritisation exponent
    # is forced to 0 and the C++ reset draws uniformly from the pool we push.
    racing_env.set_prioritize_alpha(0.0)

    if args.pr_arm == 'vanilla':
        args.past_traj_target = 0.0
        print("P-ISB arm: vanilla (no initial-state buffer)")
    else:
        model.pisb = PISBRALController(
            racing_env, arm=args.pr_arm, alpha=args.pr_alpha,
            n_pool=args.pr_pool, k_clusters=args.pr_k,
            capacity=args.pr_capacity, gate_min=args.pr_gate_min,
            refresh_every=args.pr_refresh_every, seed=args.seed,
            cl_embed_dim=args.pr_cl_embed_dim, cl_hidden=args.pr_cl_hidden,
            cl_temp=args.pr_cl_temp, cl_mining=args.pr_cl_mining,
            cl_lr=args.pr_cl_lr, cl_train_data=args.pr_cl_train_data,
            cl_use_delta=bool(args.pr_cl_delta))
        # Read back from the CONTROLLER, not from args. A sweep guard greps this
        # line to prove the requested K / n / capacity actually reached the
        # buffer; printing args would only prove the flag was parsed.
        print("PISB-RAL EFFECTIVE: arm=%s alpha=%s K=%d n=%d capacity=%d "
              "g_min=%d rho=%s" %
              (model.pisb.arm, model.pisb.alpha, model.pisb.k,
               model.pisb.n_pool, model.pisb.buf.capacity,
               model.pisb.buf.gate_min, args.past_traj_target))
        if args.pr_arm == 'refresh':
            # The ONLY arm that needs the critic. Priorities are re-scored
            # against the current value function every time the pool is
            # rebuilt, the prioritized-replay analogue of refreshing priorities. Bound to the live model, so it tracks training rather
            # than freezing at insertion.
            def _value_fn(obs, _m=model):
                return _m.value(np.asarray(obs, np.float32))
            model.pisb.value_fn = _value_fn
            # Prove it works NOW rather than silently falling back to frozen
            # |A| for the whole run -- a silent fallback would make this arm a
            # duplicate of `pisb` and the ablation meaningless.
            try:
                _odim = int(model.observation_space.shape[0])
                _probe = _value_fn(np.zeros((2, _odim), np.float32))
                print("PISB-RAL REFRESH: critic attached, V probe shape %s"
                      % (np.asarray(_probe).shape,))
            except Exception as _e:
                raise SystemExit(
                    "ABORT: --pr_arm refresh could not call model.value(): %s\n"
                    "       Without a critic this arm silently degrades to the "
                    "frozen-priority `pisb` arm and the ablation is void." % _e)

        if args.pr_arm in ('cl', 'clpisb'):
            print("PISB-RAL CL: embed_dim=%d hidden=%d temp=%s mining=%d lr=%s "
                  "train_data=%d delta=%d" %
                  (args.pr_cl_embed_dim, args.pr_cl_hidden, args.pr_cl_temp,
                   args.pr_cl_mining, args.pr_cl_lr, args.pr_cl_train_data,
                   args.pr_cl_delta))

    generator = None
    if args.mode == 'random':
        generator = TrackGenerator()
        generator.set_complexity(args.complexity_init)
        racing_env.randomize_tracks(generator)
        env.reset()

    shaping_init = rdr._load_shaping_init()
    callback = rdr.build_callback(racing_env, generator, args, shaping_init,
                                  save_dir=saver.data_dir)
    if args.pr_rho_decay_start < 1.0:
        total_updates = max(1, args.total_timesteps //
                            (args.n_steps * args.num_envs))
        callback = _wrap_rho_decay(callback, racing_env, args, total_updates)
        print("P-ISB: rho decays to 0 over [%.2f, %.2f] of training" %
              (args.pr_rho_decay_start, args.pr_rho_decay_end))
    callback = _wrap_buffer_stats(callback)

    logger.configure(folder=saver.data_dir)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        log_dir=saver.data_dir,
        logger=logger,
    )
    model.save(saver.data_dir)

    pisb = getattr(model, 'pisb', None)
    if pisb is not None:
        pisb.save_projection(os.path.join(saver.data_dir, 'final'))
        # The measured FIFO turnover, i.e. the staleness bound of the priorities.
        print("PISB-RAL FINAL BUFFER: %s" % pisb.stats())
    if hasattr(env, 'save_running_average'):
        env.save_running_average(saver.data_dir)


if __name__ == '__main__':
    main()
