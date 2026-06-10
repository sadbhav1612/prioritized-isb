# Integrating P-ISB into the PPO loop

P-ISB lives on the algorithm side so its priorities use the exact on-policy
learning signal (the GAE advantage). Integration touches three points in a
standard PPO runner; none change the gradient step.

### 1. Attach the controller

After building the policy, attach a controller so the runner can find it:

```python
from pisb import PISBController
model.pisb = PISBController(racing_env, alpha=0.3, n_pool=256,
                            k_clusters=64, capacity=20000, gate_min=1)
```

### 2. Feed it the rollout each PPO update

After advantages are computed by GAE (delta_t = r_t + gamma*V(o_{t+1})*(1-done) - V(o_t);
A_t = delta_t + gamma*lambda*(1-done)*A_{t+1}), push the flattened rollout:

```python
pisb = getattr(model, "pisb", None)
if pisb is not None:
    # full_states: [M, 17] re-initable simulator states  (pos, quat, vel, omega, gate_idx, last_passed)
    # obs:         [M, 26] observations used as clustering features
    # advantages:  [M]     priority = |advantage|
    pisb.update(full_states.reshape(-1, 17),
                obs.reshape(-1, obs.shape[-1]),
                advantages.reshape(-1))
```

`update()` adds the batch to the rolling buffer, then (every `refresh_every`
updates) rebuilds the reset pool by k-means clustering on the observations and
priority-weighted sampling within clusters, and pushes the pool to every env.

### 3. Environment hooks

The C++ racing env must expose:

- `get_full_states()` -> `[num_envs, 17]` current re-initable states,
- `set_reset_pool(env_idx, pool)` -> store an `[n, 17]` pool of candidate
  reset states for that env,
- `reset()` draws from the pool with probability `past_traj_prob` (ramped by the
  training callback), else from the fixed race start.

Because the pool already encodes the prioritized + diverse distribution, the
uniform C++ draw reproduces exactly the distribution Python constructed.

### Limiting cases (sanity checks)

- `alpha = 0`            -> uniform within clusters == observation-clustering buffer
- `k_clusters = 1`       -> flat priority sampling == prioritized-replay-style resets
- `past_traj_prob = 0`   -> never reset from the buffer == vanilla fixed start
