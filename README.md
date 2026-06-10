# Prioritized Initial-State Buffer (P-ISB)

Reference implementation accompanying the paper
*Prioritized Initial-State Buffers for Agile Drone Racing*
(under double-anonymous review).

P-ISB decides **where** an on-policy agent is reset from. Each visited state is
scored by the magnitude of the critic's advantage, |A|, and resets are drawn
more often from high-advantage states, while a k-means clustering step preserves
broad coverage. The method adds no new network, reuses the advantage PPO already
computes, and recovers the vanilla, random, and observation-clustering buffers as
the limiting cases (alpha=0, K=1, reset-prob=0).

This release provides the core buffer, the training/evaluation entry points, and
the track configurations needed to reproduce the headline Split-S results.

---

## Contents

```
.
├── pisb.py            # core: PrioritizedInitialStateBuffer + PISBController
├── INTEGRATION.md     # how P-ISB hooks into the PPO/GAE loop
├── requirements.txt
├── configs/
│   ├── SplitS.yaml    # 7-gate Split-S track (deep quantitative validation)
│   ├── Ellipse.yaml   # 8-gate ellipse track
│   ├── AlphaPilot.yaml# AlphaPilot circuit
│   ├── racing.yaml    # racing-env config (reward + dynamics)
│   └── env.yaml       # vectorised-env config
└── scripts/
    ├── train.py       # PPO training with the P-ISB flags
    └── eval.py        # deterministic from-start evaluation (lap time, speed, smoothness)
```

## Requirements

```
python 3.7
tensorflow 1.15
stable-baselines 2.10
numpy, scipy, matplotlib, pyyaml, ruamel.yaml
```

```bash
pip install -r requirements.txt
```

The training/evaluation scripts run on top of a `QuadrotorRacingEnv_v0` C++
environment layered on the Flightmare base simulator
([Song et al., 2020](https://uzh-rpg.github.io/flightmare/)) with its Python
bindings on `PYTHONPATH`, and on the accompanying `rpg_baselines` Python package
(PPO runner, env wrapper, policies). Place `pisb.py` inside that package as
`rpg_baselines/pisb.py` so that `from rpg_baselines.pisb import PISBController`
resolves. `pisb.py` itself is dependency-light (NumPy only) and can be inspected
or reused independently of the simulator; `INTEGRATION.md` documents the three
hook points so it can be ported to any PPO implementation.

## Quick start

Train P-ISB on the Split-S track:

```bash
python scripts/train.py --train 1 --mode fixed \
    --num_envs 100 --num_threads 10 --total_timesteps 150000000 \
    --n_steps 250 --norm_obs 1 --seed 0 \
    --pisb_python 1 --prioritize_alpha 0.3
```

Key P-ISB flags: `--pisb_python 1` enables the Python-owned buffer;
`--prioritize_alpha` sets the prioritization exponent alpha;
`--pisb_pool/_k/_capacity/_gate_min` configure the reset pool. Setting
`--prioritize_alpha 0` recovers the observation-clustering baseline.

Evaluate a trained checkpoint from the fixed race start:

```bash
python scripts/eval.py --weight <checkpoint.zip> --from_start --json_out metrics.json
```

See `INTEGRATION.md` for how the buffer plugs into the PPO advantage computation.
