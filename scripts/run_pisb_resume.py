#!/usr/bin/env python3
"""Continue a finished P-ISB run from 150M to 300M environment steps.

Each 150M run is continued for a further 150M steps from its OWN final policy,
so the from-start curve runs unbroken from 0 to 300M and its first half is
exactly the 150M result. Retraining from scratch at 300M would not do that:
the rho ramp and the shaping anneal are fractions of the budget, so a 300M run
is a different schedule, not a longer one.

WHAT A RESUME MUST PRESERVE, AND HOW
  PPO settings        read from the zip and checked against the protocol below;
                      learning rate 3e-4 and clip 0.2 are constants in this
                      protocol, so no schedule restarts.
  Obs normaliser      obs_rms.pkl / ret_rms.pkl are loaded into VecNormalize
                      before the first step and keep updating, as before.
  Timestep counter    stable-baselines does not store it, so it is set to the
                      original budget and learn() runs with
                      reset_num_timesteps=False; progress.csv continues at 150M.
  Shared schedules    run_drone_racing.build_callback computes rho and the
                      shaping anneal as update / total_updates, and PPO2.learn
                      restarts `update` at 1. A naive resume would restart the
                      rho ramp AND bring the annealed shaping terms back at full
                      strength, i.e. train on a different reward. The callback
                      is therefore built with the ORIGINAL budget and fed
                      update + updates_done, so every fraction is >= 1 and both
                      schedules hold their end-of-run values (rho = target,
                      shaping multiplier 0). Checkpoints inherit the offset and
                      are written as ckpt_006500 ... ckpt_012000.
  CL projection       final_clproj.npz is loaded into the projection before its
                      first training step, so the embedding is not relearned.

WHAT IS NOT PRESERVED, STATED RATHER THAN HIDDEN
  Adam moments of PPO and of the CL projection are not stored, so both restart.
  The visited-state FIFO is not stored; it refills within about five updates
  (capacity 20000 against 14k-25k inserts per update). RNG streams cannot be
  continued. progress.csv continues across the boundary, so any transient is
  visible in the data rather than assumed away.

Nothing shared is edited: the parser, env builder and callback are reused by
import, exactly as scripts/run_pisb_ral.py does.

Usage
  python scripts/run_pisb_resume.py --pr_arm pisb --pr_alpha 0.7 --seed 0 \\
      --train 1 --mode fixed --norm_obs 1 --num_envs 100 --num_threads 10 \\
      --n_steps 250 --total_timesteps 150000000 --ckpt_every 500 \\
      --past_traj_target 0.4 --past_traj_ramp_start 0.1 --past_traj_ramp_end 0.4 \\
      --pr_pool 256 --pr_k 64 --pr_capacity 20000 --pr_gate_min 1 \\
      --rs_zip  runs/pisb_s0/saved/<stamp>.zip \\
      --rs_dir  runs/pisb_s0/saved/<stamp> \\
      --save_dir runs/pisb_s0_ext
  --total_timesteps is the LENGTH OF THE CONTINUATION, not the new total.
"""
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_drone_racing as rdr                        # noqa: E402
import run_pisb_ral as rpr                            # noqa: E402

from stable_baselines import logger                   # noqa: E402
import rpg_baselines.common.util as U                 # noqa: E402
from rpg_baselines.ppo.ppo2 import PPO2               # noqa: E402
from rpg_baselines.pisb_ral import PISBRALController  # noqa: E402

# The PPO constants scripts/run_pisb_ral.py trains with. The loaded zip is
# checked against them, so a resume can never silently continue a run that was
# trained under different settings.
EXPECT = dict(gamma=0.995, lam=0.95, ent_coef=0.003, vf_coef=0.5,
              max_grad_norm=0.5, nminibatches=4, noptepochs=10)
CONST_SCHEDULES = (('learning_rate', 3e-4), ('cliprange', 0.2))


def parser():
    p = rpr.parser()
    g = p.add_argument_group('resume')
    g.add_argument('--rs_zip', required=True,
                   help='Final policy of the run being continued.')
    g.add_argument('--rs_dir', required=True,
                   help="That run's directory: obs_rms.pkl, ret_rms.pkl and, "
                        "for the cl arm, final_clproj.npz.")
    g.add_argument('--rs_updates_done', type=int, default=6000,
                   help='PPO updates the original run completed.')
    g.add_argument('--rs_orig_timesteps', type=int, default=150000000,
                   help='Budget the original run was trained with. The shared '
                        'schedules are evaluated against THIS, not against '
                        '--total_timesteps.')
    return p


def _offset_update(inner_cb, k):
    """Show the shared callback update + k instead of update. See docstring."""
    def _cb(local_vars, global_vars):
        lv = dict(local_vars)
        lv['update'] = local_vars.get('update', 0) + k
        return inner_cb(lv, global_vars)
    return _cb


def _check_model(model, args):
    bad = {}
    for k, want in EXPECT.items():
        got = float(getattr(model, k))
        if abs(got - want) > 1e-9:
            bad[k] = (got, want)
    if int(model.n_steps) != int(args.n_steps):
        bad['n_steps'] = (model.n_steps, args.n_steps)
    if int(model.n_envs) != int(args.num_envs):
        bad['n_envs'] = (model.n_envs, args.num_envs)
    # After a first learn() both are stored as schedule callables. A constant
    # schedule gives the same value at every progress fraction; anything else
    # would restart on resume and is refused.
    for name, want in CONST_SCHEDULES:
        v = getattr(model, name)
        vals = [float(v(f)) if callable(v) else float(v) for f in (1.0, 0.5, 0.0)]
        if any(abs(x - want) > 1e-12 for x in vals):
            bad[name] = (vals, want)
    if bad:
        sys.exit("ABORT: %s does not match the P-ISB protocol: %s"
                 % (args.rs_zip, bad))


def main():
    args = parser().parse_args()
    if args.paper_preset:
        sys.exit("ABORT: --paper_preset is not used by this runner.")
    if not args.train or not args.save_dir:
        sys.exit("ABORT: --train 1 and --save_dir are required.")
    if args.mode != 'fixed':
        sys.exit("ABORT: only --mode fixed is supported; the runs continued are fixed-track.")
    if args.pr_arm == 'refresh' or args.pr_rho_decay_start < 1.0:
        sys.exit("ABORT: the refresh arm and rho decay are not part of the continued runs "
                 "and are not supported by this runner.")
    if not os.path.isfile(args.rs_zip):
        sys.exit("ABORT: no policy at %s" % args.rs_zip)
    if not os.path.isfile(os.path.join(args.rs_dir, 'obs_rms.pkl')):
        sys.exit("ABORT: no obs_rms.pkl in %s; the policy would be fed "
                 "unnormalised observations." % args.rs_dir)
    clproj = os.path.join(args.rs_dir, 'final_clproj.npz')
    if args.pr_arm in ('cl', 'clpisb') and not os.path.isfile(clproj):
        sys.exit("ABORT: no final_clproj.npz in %s; the cl arm would restart "
                 "its embedding from scratch." % args.rs_dir)

    env, racing_env = rdr.make_env(args)
    rdr.configure_random_seed(args.seed, env=env)

    if not hasattr(env, 'load_running_average'):
        sys.exit("ABORT: the env is not VecNormalize; pass --norm_obs 1.")
    env.load_running_average(args.rs_dir)
    cnt = float(np.asarray(env.obs_rms.count))
    print("PISB-RESUME OBS_RMS: count=%.0f |mean|=%.4f  from %s"
          % (cnt, float(np.linalg.norm(env.obs_rms.mean)), args.rs_dir))
    if cnt < 1e4:
        sys.exit("ABORT: loaded obs_rms has count %.0f; that is not the "
                 "normaliser of a 150M-step run." % cnt)

    log_dir = os.path.join(args.save_dir, 'saved')
    saver = U.ConfigurationSaver(log_dir=log_dir)
    model = PPO2.load(args.rs_zip, env=env, tensorboard_log=saver.data_dir)
    _check_model(model, args)
    print("PISB-RESUME MODEL: %s  n_steps=%d n_envs=%d  lr/clip constant"
          % (args.rs_zip, model.n_steps, model.n_envs))

    # Same as the original runner: all reset selection happens in Python.
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
        # Same wording as run_pisb_ral.py so an existing guard grepping this
        # line keeps working on resumed runs.
        print("PISB-RAL EFFECTIVE: arm=%s alpha=%s K=%d n=%d capacity=%d "
              "g_min=%d rho=%s" %
              (model.pisb.arm, model.pisb.alpha, model.pisb.k,
               model.pisb.n_pool, model.pisb.buf.capacity,
               model.pisb.buf.gate_min, args.past_traj_target))

        if args.pr_arm in ('cl', 'clpisb'):
            # The controller builds the projection lazily on its first update
            # and only if none exists, so building it here and loading the
            # saved weights means the first training step starts from the
            # trained embedding.
            import torch
            from rpg_baselines.pisb_ral.ref_cl import RefProjectionBuffer
            w = np.load(clproj)
            obs_dim = int(w['0.weight'].shape[1])
            proj = RefProjectionBuffer(obs_dim=obs_dim, **model.pisb._cl_cfg)
            sd = proj.net.state_dict()
            mism = sorted(set(sd) ^ set(w.files)) + \
                [k for k in sd if k in w.files and tuple(sd[k].shape) != w[k].shape]
            if mism:
                sys.exit("ABORT: saved projection does not fit the network: %s" % mism)
            proj.net.load_state_dict({k: torch.as_tensor(w[k]) for k in w.files})
            model.pisb.proj = proj
            print("PISB-RESUME CL: projection loaded (%d tensors, obs_dim=%d)"
                  % (len(w.files), obs_dim))

    # The shared callback sees the ORIGINAL budget and an offset update, so rho
    # and the shaping anneal hold their end-of-run values. Copied AFTER the
    # vanilla branch so its rho target of 0 carries over.
    cb_args = copy.copy(args)
    cb_args.total_timesteps = int(args.rs_orig_timesteps)
    shaping_init = rdr._load_shaping_init()
    callback = rdr.build_callback(racing_env, None, cb_args, shaping_init,
                                  save_dir=saver.data_dir)
    callback = _offset_update(callback, int(args.rs_updates_done))
    callback = rpr._wrap_buffer_stats(callback)
    print("PISB-RESUME SCHEDULE: callback budget=%d, update offset=%d, "
          "continuation=%d steps" % (cb_args.total_timesteps,
                                     args.rs_updates_done, args.total_timesteps))

    model.num_timesteps = int(args.rs_orig_timesteps)
    logger.configure(folder=saver.data_dir)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        log_dir=saver.data_dir,
        logger=logger,
        reset_num_timesteps=False,
    )
    model.save(saver.data_dir)

    pisb = getattr(model, 'pisb', None)
    if pisb is not None:
        pisb.save_projection(os.path.join(saver.data_dir, 'final'))
        print("PISB-RAL FINAL BUFFER: %s" % pisb.stats())
    env.save_running_average(saver.data_dir)


if __name__ == '__main__':
    main()
