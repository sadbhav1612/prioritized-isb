# Integrating P-ISB into a PPO loop

P-ISB acts only on where episodes start. It uses the on-policy GAE advantage
as its priority and changes neither the loss nor the gradient step, so the
data stay on-policy and need no importance-sampling correction. It touches
three points.

## 1. Attach a controller

```python
from rpg_baselines.pisb_ral import PISBRALController

model.pisb = PISBRALController(racing_env, arm="pisb", alpha=0.7, n_pool=256,
                               k_clusters=64, capacity=20000, gate_min=1,
                               seed=seed)
```

`arm` selects the reset rule: `random`, `obs`, `cl` (the reference's three
strategies), `pisb`, or the variants `signed` and `refresh`. Plain PPO attaches
no controller and sets the reset probability to 0.

## 2. Feed it each rollout, before the gradient step

During collection, read the re-initable simulator state of every environment
*before* `env.step`, so that it aligns with the observation and advantage of
that step (a done environment is reset inside `env.step`):

```python
pisb = getattr(model, "pisb", None)
for t in range(n_steps):
    ...
    if pisb is not None:
        mb_full_states.append(pisb.get_full_states())   # [N, 17]
    obs, rewards, dones, infos = env.step(actions)
```

After GAE, while the arrays are still time-major `[T, N]`, push the rollout:

```python
# mb_advs: GAE advantages [T, N], signed (the controller takes |A| itself)
# dv:      discounted sum of future advantages within each episode, [T, N],
#          dv[t] = A[t] + gamma * (1 - done[t+1]) * dv[t+1]; the CL-Buffer
#          trains its embedding on it
# mb_returns = mb_advs + mb_values
pisb.update(fs.reshape(T * N, -1),           # full states
            mb_obs.reshape(T * N, -1),       # observations, the clustering features
            mb_advs.reshape(T * N),
            embed_signal=dv.reshape(T * N),
            recover_signal=mb_returns.reshape(T * N))
```

`update` keeps the states at or past gate `gate_min`, appends them to the
visited-state FIFO with priority |A|, rebuilds the reset pool (k-means over the
buffer, a uniform cluster draw, then a draw within the cluster proportional to
(|A| + 1e-3)^alpha) and pushes it to the environments. Because this happens
before the PPO update, the pool used in iteration k+1 carries priorities from
the critic of iteration k. When one call inserts more states than the capacity,
the buffer keeps the most recent ones in this time-major order.

## 3. Environment hooks

The racing environment exposes:

- `get_full_states()` -> `[num_envs, 17]`: position (3), orientation
  quaternion (4), linear velocity (3), body rates (3), gate index (1) and the
  last-passed-gate record (3);
- `set_reset_pool(env_idx, pool)`: store an `[n, 17]` pool of candidate reset
  states for one environment (the controller calls it for every environment);
- `set_past_traj_prob(rho)`: on reset, draw from the pool with probability
  rho, otherwise from the base distribution (30% at the race start, 70% at a
  uniform point along a random gate segment, moving along it at 3 m/s).
  Evaluation resets from the race start only.

The training callback ramps rho and anneals the reward shaping through
`set_past_traj_prob` and `set_shaping_coeffs` (see `scripts/run_drone_racing.py`).

## Limiting cases

- `rho = 0`: no buffer resets, plain PPO on the base reset distribution;
- `arm="pisb", alpha=0`: a uniform draw within clusters, a clustering-only
  buffer. It is a different estimator from `arm="obs"`, which takes the states
  nearest each centroid of a cosine k-means;
- `k_clusters=1`: priority sampling over the whole buffer.
