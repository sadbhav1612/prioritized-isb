"""Visited-state buffer and the five reset-selection arms, for legged locomotion.

The reference paper (Messikommer et al., ICRA 2024) validates the CL-Buffer on
this setup, which is reproduced here: Rudin et al. legged_gym on IsaacGym, 1024 environments, 2000
iterations, terrain of stairs up/down, rough, smooth, discrete and flat.

SELF-CONTAINED. Nothing here is imported by legged_gym or rsl_rl; this package
only ever imports FROM them. No file in either upstream tree is modified.

THE CL AND OBS ARMS USE THE AUTHORS' OWN LIBRARY. `initial_buffer` from
https://github.com/uzh-rpg/cl_initial_buffer is imported verbatim rather than
reimplemented, so on this task the baseline IS the published method, with only the reset-selection
rule differing between arms. `sampling_strategy` maps directly:

    'random'       -> Random-Buffer   (paper Eq 2)
    'observations' -> Obs-Buffer      (paper Eq 3-4)
    'network'      -> CL-Buffer       (paper Eq 8-10)

P-ISB is the one arm that is ours: same k-means over the same embedding, but
states are drawn WITHIN each cluster with probability proportional to
(|A| + eps)^alpha instead of taking the N/K nearest the centre. alpha = 0
recovers a uniform within-cluster draw.

WHAT A "STATE" IS HERE, and why it is stored relative to the terrain.
A legged_gym root state is in world coordinates and only means anything next to
the terrain patch it was recorded on. Restoring a raw world position into an
environment sitting on a different patch drops the robot into the wrong terrain,
silently. So each entry stores the base pose RELATIVE to its env_origin plus the
terrain_level and terrain_type it came from; on restore the target environment is
moved onto that patch first, then the robot is placed at origin + offset. The
command is stored too, since the reward is command-following and a state without
its command is not the same task.

PRE-SELECTION follows the paper's Section V for locomotion, verbatim:
  - exclude states less than 15 timesteps from their episode start
  - exclude states whose accumulated reward is below zero, which indicates the
    command following is not being performed
Their Fig. 3(d) ablation shows both filters together train best.
"""
import numpy as np
import torch

ARMS = ("vanilla", "random", "obs", "cl", "pisb")

# Paper Section IV-A / V constants.
N_ISB = 256          # states added to the ISB per roll-out phase
K_CLUSTERS = 64      # k-means clusters
P_RESET = 0.8        # probability an episode starts from the ISB
MIN_TIMESTEPS = 15   # locomotion pre-selection: distance from episode start
EPS_PRIO = 1e-3      # priority floor, matching the drone-racing P-ISB


class LeggedStateBuffer:
    """Rolling FIFO of visited states V, with the paper's pre-selection.

    Layout of one stored state, all on the training device:
        root_rel   [13]  base pose with position RELATIVE to env_origin,
                         then quaternion, linear velocity, angular velocity
        dof_pos    [D]
        dof_vel    [D]
        commands   [C]   the command being tracked at that instant
        terrain    [2]   (terrain_level, terrain_type) as floats
    """

    def __init__(self, capacity, num_dof, num_commands, device, fifo_uniform=True):
        self.capacity = int(capacity)
        self.num_dof = int(num_dof)
        self.num_commands = int(num_commands)
        self.device = device
        # When one rollout exceeds capacity (24 steps x 1024 envs = 24,576
        # against 20,000), keep a UNIFORM subsample of it. The original code
        # kept states[-capacity:], and because rows are flattened time-first
        # that silently discarded the first ~4.5 timesteps of EVERY rollout.
        # False reproduces the tail behaviour of the runs already reported.
        self.fifo_uniform = bool(fifo_uniform)
        self.dim = 13 + 2 * self.num_dof + self.num_commands + 2

        self.states = torch.zeros(self.capacity, self.dim, device=device)
        self.obs = None                       # allocated on first add
        self.prio = torch.zeros(self.capacity, device=device)
        self.write = 0
        self.fill = 0
        self.added_total = 0
        self.last_added = 0

    # ------------------------------------------------------------- capture
    @staticmethod
    def encode(env):
        """Snapshot every environment's current state as one [num_envs, dim] row.

        Called once per simulation step from the rollout loop, so it must stay
        cheap: this is pure tensor slicing and one subtraction, no host sync.
        """
        root_rel = env.root_states.clone()
        root_rel[:, :3] -= env.env_origins            # terrain-relative position
        terrain = torch.stack(
            (env.terrain_levels.float(), env.terrain_types.float()), dim=1) \
            if hasattr(env, "terrain_levels") else \
            torch.zeros(env.num_envs, 2, device=env.device)
        return torch.cat((root_rel, env.dof_pos, env.dof_vel,
                          env.commands[:, :env.cfg.commands.num_commands],
                          terrain), dim=1)

    def _plausible(self, states):
        """True for rows that are finite and physically reachable.

        Layout is [root(13) | dof_pos(D) | dof_vel(D) | commands(C) | terrain(2)]
        with root = pos(3), quat(4), lin vel(3), ang vel(3), so the velocity
        blocks are at fixed offsets and no env handle is needed here.
        """
        ok = torch.isfinite(states).all(dim=1)
        if states.shape[1] < 13:
            return ok
        # THE QUATERNION IS THE DANGEROUS FIELD. cols 3:7 are the base
        # orientation, and legged_gym feeds it straight into quat_rotate_inverse
        # for the projected gravity and the base velocities. A zero or
        # denormalised quaternion there produces NaN in the OBSERVATION, which
        # survives normalization.clip_observations (clamp passes NaN through
        # untouched) and reaches the actor, whose output then fails the Normal
        # distribution's Real() constraint in only the affected dimensions --
        # exactly the signature seen, where the visible columns look ordinary.
        # A zero row is finite and slow, so the magnitude checks below never
        # catch it; only the norm does.
        qn = states[:, 3:7].norm(dim=1)
        ok &= (qn > 0.5) & (qn < 2.0)
        ok &= states[:, 7:10].abs().amax(dim=1) < LIN_VEL_MAX
        ok &= states[:, 10:13].abs().amax(dim=1) < ANG_VEL_MAX
        d = self.num_dof if hasattr(self, "num_dof") else None
        if d and states.shape[1] >= 13 + 2 * d:
            ok &= states[:, 13 + d:13 + 2 * d].abs().amax(dim=1) < DOF_VEL_MAX
        return ok

    def add(self, states, obs, prio, keep_mask):
        """Append the rows selected by `keep_mask` (the pre-selection filter)."""
        if keep_mask is not None:
            states, obs, prio = states[keep_mask], obs[keep_mask], prio[keep_mask]

        # PHYSICAL-PLAUSIBILITY FILTER, on top of the paper's pre-selection.
        # The paper filters on episode step and accumulated reward only. Neither
        # excludes a state captured while a robot is already blowing up, and
        # such a state is poison: restore it later and the policy sees inputs far
        # outside its training distribution, produces larger actions, stores
        # worse states, and the loop closes. Seven of seventeen cells died this
        # way on 2026-09-02 with `Expected parameter loc ... to satisfy Real()`,
        # at iterations 347 to 1883, while vanilla -- which never restores -- did
        # not fail once.
        # This is numerical hygiene, not a change to the method: a diverged state
        # is not a state the paper intends the buffer to hold either.
        ok = self._plausible(states)
        n_drop = int((~ok).sum())
        if n_drop:
            prev = getattr(self, "n_implausible", 0)
            self.n_implausible = prev + n_drop
            # SAY SO. A silent filter is untestable: the previous version of this
            # guard could not be distinguished from a no-op, which cost a whole
            # debugging cycle. Print the first occurrence and then every 50th
            # call, with a breakdown of WHICH check rejected the row, so the log
            # shows whether the guard is doing anything and what it is catching.
            self.n_drop_calls = getattr(self, "n_drop_calls", 0) + 1
            if self.n_drop_calls == 1 or self.n_drop_calls % 50 == 0:
                nf = int((~torch.isfinite(states).all(dim=1)).sum())
                qn = states[:, 3:7].norm(dim=1)
                bq = int(((qn <= 0.5) | (qn >= 2.0)).sum())
                print("[ISB] dropped %d/%d implausible states "
                      "(non-finite %d, bad quaternion %d) -- %d total"
                      % (n_drop, states.shape[0], nf, bq, self.n_implausible),
                      flush=True)
            states, obs, prio = states[ok], obs[ok], prio[ok]

        m = states.shape[0]
        self.last_added = m
        if m == 0:
            return
        if self.obs is None:
            self.obs = torch.zeros(self.capacity, obs.shape[1], device=self.device)
        if m > self.capacity:
            if self.fifo_uniform:
                # uniform over the whole rollout, so no timestep is favoured
                keep = torch.randperm(m, device=states.device)[:self.capacity]
            else:
                # legacy: the tail, i.e. the LAST timesteps of the rollout
                keep = torch.arange(m - self.capacity, m, device=states.device)
            states, obs, prio = states[keep], obs[keep], prio[keep]
            m = self.capacity
        idx = (self.write + torch.arange(m, device=self.device)) % self.capacity
        self.states[idx] = states
        self.obs[idx] = obs
        self.prio[idx] = prio.abs()
        self.write = int((self.write + m) % self.capacity)
        self.fill = int(min(self.capacity, self.fill + m))
        self.added_total += m

    def view(self):
        return (self.states[:self.fill],
                self.obs[:self.fill] if self.obs is not None else None,
                self.prio[:self.fill])

    def stats(self):
        return {"buf_fill": self.fill,
                "buf_added": self.last_added,
                "buf_turnover_iters": self.capacity / max(1, self.last_added)}


# Bounds for the plausibility filter. Deliberately generous -- roughly an order
# of magnitude past anything a walking ANYmal reaches, so a legitimate recovery
# or a stumble is kept and only a numerical blow-up is dropped. The command
# range is +-1 m/s, and a fall tops out near 10 m/s.
LIN_VEL_MAX = 20.0     # m/s,   root linear velocity
ANG_VEL_MAX = 30.0     # rad/s, root angular velocity
DOF_VEL_MAX = 50.0     # rad/s, joint velocity (ANYmal's own limit is ~20)


# ON-MAP FILTER (added 2026-09-15). Every restore starts a fresh 20 s episode
# from wherever the stored robot stood, so a chain of restores lets a robot
# drift without bound from its patch. Past the 25 m flat border there is no
# ground: the robot free-falls for the rest of the episode, rew_lin_vel_z grows
# as vz^2 (-1e4 to -1e30 per episode in every default-protocol buffer cell),
# and PhysX eventually returns inf/NaN. diag_explode.py found every remaining
# explosion to be exactly this: bases 5 m below the ground, mostly on level 0,
# the row at the grid's edge. Vanilla never drifts (it restarts within 1 m of
# its patch centre) and single start sits 40+ m from any edge.
# A state is eligible only if its base is MAP_MARGIN inside the patch grid and
# not below the local ground. One episode can then cover at most
# sqrt(2) m/s * 20 s = 28.3 m, less than MAP_MARGIN + border (30 m).
MAP_MARGIN = 5.0             # m inside the patch grid
MIN_BASE_CLEARANCE = -0.5    # m, base height above the terrain sample under it


def on_map_mask(env, margin=MAP_MARGIN):
    """[num_envs] bool: base inside the terrain grid and not below the ground."""
    if not hasattr(env, "height_samples") or not hasattr(env, "terrain"):
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    cfg = env.terrain.cfg
    xy = env.root_states[:, :2]
    # The mesh is shifted by -border_size, so the patch grid spans
    # [0, rows*length] x [0, cols*width] in world coordinates.
    L = cfg.num_rows * cfg.terrain_length
    W = cfg.num_cols * cfg.terrain_width
    ok = (xy[:, 0] >= margin) & (xy[:, 0] <= L - margin) & \
         (xy[:, 1] >= margin) & (xy[:, 1] <= W - margin)
    # Same sampling as LeggedRobot._get_heights.
    p = ((xy + cfg.border_size) / cfg.horizontal_scale).long()
    px = p[:, 0].clamp(0, env.height_samples.shape[0] - 2)
    py = p[:, 1].clamp(0, env.height_samples.shape[1] - 2)
    ground = env.height_samples[px, py].float() * cfg.vertical_scale
    ok &= (env.root_states[:, 2] - ground) > MIN_BASE_CLEARANCE
    return ok


def preselect_mask(episode_step, accumulated_reward, min_timesteps=MIN_TIMESTEPS):
    """Paper Section V, locomotion: drop states too close to their episode start
    and states whose accumulated reward has gone negative.

    Both are element-wise over the flattened [T*N] rollout.
    """
    return (episode_step >= min_timesteps) & (accumulated_reward >= 0.0)


# --------------------------------------------------------------- selection
def select_indices(arm, projection_buffer, obs, prio, n=N_ISB, k=K_CLUSTERS,
                   alpha=0.7, rng=None):
    """Return the indices of the states to load into the ISB.

    random / obs / cl delegate to the authors' ProjectionBuffer, so those three
    arms are their published method unmodified. Only `pisb` is ours.
    """
    if arm in ("random", "obs", "cl"):
        return projection_buffer.create_initial_state_buffer(obs, buffer_length=n)

    if arm != "pisb":
        raise ValueError("unknown arm %r" % arm)

    # P-ISB: cluster exactly as the Obs-Buffer does, then draw within each
    # cluster proportional to priority^alpha rather than taking the nearest to
    # the centre. Reusing their clustering call keeps the diversity axis
    # identical, so the ONLY difference from the obs arm is the draw.
    embs = projection_buffer.project_obs(obs)
    dists, labels = projection_buffer.cluster_embeddings(
        embs, nr_clusters=k, cluster_algo=projection_buffer.cluster_algo)
    labels = np.asarray(labels).ravel()
    p = prio.detach().cpu().numpy().ravel() if torch.is_tensor(prio) else np.asarray(prio).ravel()
    rng = rng or np.random.default_rng(0)

    per = max(1, n // max(1, k))
    chosen = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        w = np.power(p[members] + EPS_PRIO, alpha)
        tot = w.sum()
        take = min(per, len(members))
        chosen.append(rng.choice(members, size=take, replace=False,
                                 p=(w / tot) if tot > 0 else None))
    if not chosen:
        return np.arange(min(n, len(p)))
    idx = np.concatenate(chosen)
    if len(idx) < n:                       # even allocation under-filled
        rest = np.setdiff1d(np.arange(len(p)), idx)
        if len(rest):
            idx = np.concatenate(
                [idx, rng.choice(rest, size=min(n - len(idx), len(rest)),
                                 replace=False)])
    return idx[:n]


# ----------------------------------------------------------------- restore
def apply_state(env, env_ids, states):
    """Place the given environments at the given stored states.

    Order matters. The terrain patch is set FIRST, because env_origins is what
    the relative base position is measured against; writing the root state
    before moving the origin would place the robot correctly relative to the
    patch it is about to leave.
    """
    if len(env_ids) == 0:
        return
    D, C = env.num_dof, env.cfg.commands.num_commands

    if hasattr(env, "terrain_levels") and hasattr(env, "terrain_origins"):
        lvl = states[:, 13 + 2 * D + C].long().clamp_(0, env.max_terrain_level - 1)
        typ = states[:, 13 + 2 * D + C + 1].long().clamp_(
            0, env.terrain_origins.shape[1] - 1)
        env.terrain_levels[env_ids] = lvl
        env.terrain_types[env_ids] = typ
        env.env_origins[env_ids] = env.terrain_origins[lvl, typ]

    root = states[:, :13].clone()
    root[:, :3] += env.env_origins[env_ids]           # back to world coordinates
    env.root_states[env_ids] = root
    env.dof_pos[env_ids] = states[:, 13:13 + D]
    env.dof_vel[env_ids] = states[:, 13 + D:13 + 2 * D]
    env.commands[env_ids, :C] = states[:, 13 + 2 * D:13 + 2 * D + C]

    # SYNC THE "LAST" BUFFERS TO THE STATE WE JUST TELEPORTED INTO.
    # super().reset_idx() zeroes last_dof_vel and last_root_vel, then the lines
    # above overwrite dof_vel and the root velocities with the stored ones. On
    # the very next step legged_gym computes
    #     _reward_dof_acc = sum(((last_dof_vel - dof_vel) / dt) ** 2)
    # which, with last_dof_vel still 0 and dt = 0.02, fabricates an acceleration
    # of 50x the restored joint velocity -- a large penalty spike on EVERY ISB
    # reset that the nominal start never incurs. _reward_lin_vel_z and the base
    # acceleration terms have the same problem via last_root_vel. Copying the
    # restored values across makes the first post-reset step continuous, so the
    # reward compares like with like between an ISB start and a nominal one.
    if hasattr(env, "last_dof_vel"):
        env.last_dof_vel[env_ids] = env.dof_vel[env_ids]
    if hasattr(env, "last_root_vel"):
        env.last_root_vel[env_ids] = env.root_states[env_ids, 7:13]
