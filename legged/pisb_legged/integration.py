"""Wiring the initial-state buffer into legged_gym / rsl_rl.

Three subclasses and nothing else. `legged_gym` and `rsl_rl` are NOT modified;
this file only inherits from them.

    ISBLeggedRobot   overrides reset_idx  -> episodes may start from the ISB
    ISBPPO           overrides update     -> trains the projection once per epoch
    ISBOnPolicyRunner overrides learn     -> captures states, builds the ISB

ORDER OF OPERATIONS, taken from the reference toy_example.py:

    build ISB from the visited buffer      (create_initial_state_buffer)
    collect rollout                        (env resets draw from the ISB)
    compute returns
    create_train_data(rewards, obs, dones) (fixes 256 sub-trajectories)
    for each of the 10 PPO epochs:
        values = critic(train sample obs)  <- re-evaluated, critic has moved
        train_step(values)                 <- differences against prev_improvement

That last loop is the part a one-call-per-rollout implementation gets wrong.
`create_train_data` resets `prev_improvement` to None, so the FIRST train_step
of a rollout differences against zeros and reduces to the improvement LEVEL;
only from the second call does the mining signal become the DELTA the paper
describes. Calling it once per rollout therefore reproduces the reference's
first epoch and nothing else. rsl_rl's update() has no explicit epoch boundary
-- its generator yields num_learning_epochs * num_mini_batches batches back to
back -- so the boundary is recovered by counting batches.
"""
import numpy as np
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym import gymtorch
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner

from pisb_legged.state_buffer import (
    LeggedStateBuffer, apply_state, preselect_mask, select_indices, on_map_mask,
    N_ISB, K_CLUSTERS, P_RESET, MIN_TIMESTEPS,
)


# ===========================================================================
class ISBLeggedRobot(LeggedRobot):
    """LeggedRobot whose episodes may restart from a buffered state.

    The ISB is a plain tensor pushed in by the runner. With probability p each
    resetting environment is placed at a sampled entry instead of the nominal
    start pose.

    WHY super().reset_idx IS STILL CALLED. It clears episode_length_buf,
    last_actions, feet_air_time and the episode-sum logging, resamples the
    command, and steps the terrain curriculum. Skipping it to write a state
    directly would leave stale buffers behind and silently corrupt the reward
    bookkeeping. So the nominal reset runs first and the ISB state is written
    over the top, after which the root and DOF tensors are pushed to the sim
    again for every environment reset this step (see isb_reset_all).
    """

    isb_states = None          # [M, dim] or None
    isb_prob = 0.0             # p; 0 disables, which is the vanilla arm
    # ONE INDEXED WRITE PER STEP (fixed 2026-09-15). IsaacGym applies only the
    # LAST set_actor_root_state_tensor_indexed / set_dof_state_tensor_indexed
    # call before simulate. legged_gym writes every resetting env, then the ISB
    # restore used to write again for the buffer-restored subset only, so every
    # NOMINAL reset in the same step was silently dropped: diag_explode.py
    # measured 20% of nominal resets still 8-10 m from the written pose one step
    # later, buffer restores 6 mm. Every default-protocol buffer cell also showed
    # rew_lin_vel_z of -1e4 to -1e30 while vanilla and single start never did.
    # True re-issues both writes for ALL envs reset this step. False reproduces
    # the runs made before the fix.
    isb_reset_all = True
    _isb_step_ids = None       # env ids passed to this step's reset_idx

    # Counters, class-level so a run reports them once at the end.
    n_bad_obs = 0
    n_bad_obs_steps = 0
    n_hard_resets = 0
    n_unrecoverable = 0

    def compute_observations(self):
        """Stock observations, then a non-finite tripwire.

        WHY. A NaN reaching the actor makes torch's Normal reject its own `loc`
        with "Expected parameter loc ... to satisfy the constraint Real()", and
        the traceback points at rsl_rl, thousands of steps downstream of
        whatever produced it. Worse, normalization.clip_observations uses clamp,
        and clamp passes NaN through unchanged, so the usual safety net does not
        catch this one. Seven of seventeen cells died this way on 2026-09-02 and
        the tracebacks named nothing useful.
        This check names the step, the environments and the observation columns
        at the moment of first corruption, then zeroes the offending entries so
        the run continues and produces a diagnosable log instead of a stack
        trace. If the counter stays at zero the cause is elsewhere -- in the
        optimizer rather than the data -- which is equally worth knowing.
        """
        super().compute_observations()
        bad = ~torch.isfinite(self.obs_buf)
        if bool(bad.any()):
            n = int(bad.sum())
            ISBLeggedRobot.n_bad_obs += n
            ISBLeggedRobot.n_bad_obs_steps += 1
            if ISBLeggedRobot.n_bad_obs_steps <= 3:
                envs = torch.unique(bad.any(dim=1).nonzero().flatten())[:8]
                cols = torch.unique(bad.any(dim=0).nonzero().flatten())[:16]
                print("[ISB] NON-FINITE OBSERVATION: %d entries, envs %s, "
                      "cols %s (obs_dim=%d, isb_prob=%.2f)"
                      % (n, envs.tolist(), cols.tolist(),
                         self.obs_buf.shape[1], float(self.isb_prob)),
                      flush=True)
            self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0,
                                            posinf=0.0, neginf=0.0)

            # SANITISING THE OBSERVATION IS NOT ENOUGH. The corruption is in the
            # SIMULATOR state, not the observation: env 214 of random_s13 showed
            # 221 of 235 dims non-finite on three consecutive steps, including
            # columns 0-8, which are base linear velocity, angular velocity and
            # projected gravity -- all derived from root_states. A zeroed
            # observation hides that while the env stays broken for the rest of
            # training, feeding fabricated zeros into every rollout until the
            # policy diverges. That is exactly how random_s0, obs_s7 and
            # random_s13 died 25-31 minutes in.
            # Hard-reset the affected environments through the PARENT reset, so
            # they restart from the nominal pose and never from the buffer --
            # the buffer is where the bad state came from.
            bad_envs = bad.any(dim=1).nonzero(as_tuple=False).flatten()
            if len(bad_envs):
                ISBLeggedRobot.n_hard_resets += int(len(bad_envs))

                # RESET DOES NOT CLEAR self.actions, AND self.actions IS IN THE
                # OBSERVATION. legged_robot.py concatenates self.actions into
                # obs_buf (:218) but reset_idx only zeroes last_actions (:171).
                # So a single non-finite action pins that environment's action
                # columns non-finite for the REST OF TRAINING, and every hard
                # reset reports "STILL non-finite after refresh" while changing
                # nothing. obs_s13 died that way at iteration 412 after eight
                # futile resets. The same holds for any per-env tensor carried
                # across steps, so sanitise them all BEFORE the reset -- the
                # reset's own arithmetic (root_states = base_init + env_origins)
                # would otherwise propagate a NaN origin straight back in.
                fixed = self._isb_sanitise(bad_envs)

                # env_origins is the one carried tensor that must be REBUILT
                # rather than zeroed: zero is a real coordinate, and it would
                # silently teleport the robot to the corner of the terrain grid.
                if getattr(self, "custom_origins", False) and \
                        hasattr(self, "terrain_origins"):
                    org = self.env_origins[bad_envs]
                    if not bool(torch.isfinite(org).all()):
                        self.env_origins[bad_envs] = self.terrain_origins[
                            self.terrain_levels[bad_envs],
                            self.terrain_types[bad_envs]]
                        fixed.append("env_origins(rebuilt)")

                LeggedRobot.reset_idx(self, bad_envs)
                # Again after the reset: _reset_root_states adds a random offset
                # and _resample_commands draws new commands, either of which can
                # re-derive from a value we have not reached yet.
                fixed += self._isb_sanitise(bad_envs)
                # The parent reset above made its own indexed writes for
                # bad_envs only, which replace this step's reset writes. Re-issue
                # one write covering both sets (see isb_reset_all).
                if self.isb_reset_all:
                    step_ids = self._isb_step_ids
                    ids = bad_envs if step_ids is None or len(step_ids) == 0 \
                        else torch.unique(torch.cat((step_ids, bad_envs)))
                    self._isb_push_states(ids)

                # RESETTING root_states IS NOT ENOUGH ON ITS OWN. The observation
                # is assembled from DERIVED tensors -- base_lin_vel, base_ang_vel,
                # projected_gravity, measured_heights -- which legged_gym refreshes
                # only in post_physics_step. Recomputing observations straight
                # after reset_idx therefore rebuilds them from the stale NaN
                # derivatives and the environment stays corrupt, which is exactly
                # what happened: env 214 and then env 715 tripped the detector on
                # three consecutive steps each. Refresh the derived quantities
                # here, for all envs since it only runs when corruption is found.
                from isaacgym.torch_utils import quat_rotate_inverse
                self.base_quat[:] = self.root_states[:, 3:7]
                self.base_lin_vel[:] = quat_rotate_inverse(
                    self.base_quat, self.root_states[:, 7:10])
                self.base_ang_vel[:] = quat_rotate_inverse(
                    self.base_quat, self.root_states[:, 10:13])
                self.projected_gravity[:] = quat_rotate_inverse(
                    self.base_quat, self.gravity_vec)
                if getattr(self.cfg.terrain, "measure_heights", False):
                    self.measured_heights = self._get_heights()

                super().compute_observations()
                still = ~torch.isfinite(self.obs_buf)
                self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0,
                                                posinf=0.0, neginf=0.0)
                print("[ISB] hard-reset %d env(s) %s; %s after refresh "
                      "(%d hard resets total)%s"
                      % (len(bad_envs), bad_envs[:8].tolist(),
                         "STILL non-finite" if bool(still.any()) else "clean",
                         ISBLeggedRobot.n_hard_resets,
                         ("; repaired " + ", ".join(fixed)) if fixed else ""),
                      flush=True)

                # UNRECOVERABLE MEANS STOP, NOT CARRY ON. If the observation is
                # still non-finite after the state has been fully rebuilt, the
                # corruption is somewhere this repair does not reach, and every
                # subsequent step trains on fabricated zeros for that
                # environment. A cell that finishes on fabricated data is worse
                # than a cell that fails, because it lands in the table looking
                # like a result. Fail loudly and name the columns instead.
                if bool(still.any()):
                    ISBLeggedRobot.n_unrecoverable += 1
                    if ISBLeggedRobot.n_unrecoverable >= 3:
                        cols = torch.unique(
                            still.any(dim=0).nonzero().flatten()).tolist()
                        envs = torch.unique(
                            still.any(dim=1).nonzero().flatten()).tolist()
                        raise RuntimeError(
                            "[ISB] UNRECOVERABLE non-finite observation after "
                            "%d full state repairs. envs=%s cols=%s "
                            "(obs_dim=%d). The corruption is outside the "
                            "per-env tensors this repair rebuilds; do not "
                            "treat a run past this point as data."
                            % (ISBLeggedRobot.n_unrecoverable, envs[:8],
                               cols[:24], self.obs_buf.shape[1]))
                else:
                    ISBLeggedRobot.n_unrecoverable = 0

    # Tensors excluded from the sweep below: obs_buf is sanitised by the caller,
    # and the reward/termination buffers are recomputed from scratch every step,
    # so zeroing them here would hide rather than fix a problem.
    _ISB_SKIP = ("obs_buf", "privileged_obs_buf", "rew_buf", "reset_buf",
                 "time_out_buf", "episode_length_buf")

    def _isb_sanitise(self, env_ids):
        """Zero non-finite entries in every per-env float tensor on the env.

        Deliberately generic rather than a hand-listed set of attributes. The
        hand-listed version is what let this bug through: self.actions was not
        on the list, and it is the one tensor that both feeds the observation
        directly and survives reset_idx. Anything shaped (num_envs, ...) and
        floating point is state carried across steps, so anything matching that
        is worth clearing. Returns the names repaired, for the log line.
        """
        fixed = []
        for name in dir(self):
            if name.startswith("__") or name in self._ISB_SKIP:
                continue
            try:
                t = getattr(self, name)
            except Exception:
                continue                       # properties that raise; skip
            if not torch.is_tensor(t) or not t.is_floating_point():
                continue
            if t.dim() == 0 or t.shape[0] != self.num_envs:
                continue
            sub = t[env_ids]
            n = int((~torch.isfinite(sub)).sum())
            if n:
                t[env_ids] = torch.nan_to_num(sub, nan=0.0, posinf=0.0,
                                              neginf=0.0)
                fixed.append("%s(%d)" % (name, n))
        return fixed

    def _isb_push_states(self, ids):
        """Write root and DOF state of `ids` to the sim, as ONE call each."""
        ids32 = ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(ids32), len(ids32))
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(ids32), len(ids32))

    def reset_idx(self, env_ids):
        # Recorded on every call, including empty ones, so the hard-reset path
        # in compute_observations always sees THIS step's resets.
        self._isb_step_ids = env_ids
        super().reset_idx(env_ids)
        if (self.isb_states is None or len(env_ids) == 0
                or self.isb_prob <= 0.0 or self.isb_states.shape[0] == 0):
            return

        take = torch.rand(len(env_ids), device=self.device) < self.isb_prob
        chosen = env_ids[take]
        if len(chosen) == 0:
            return

        pick = torch.randint(0, self.isb_states.shape[0], (len(chosen),),
                             device=self.device)
        apply_state(self, chosen, self.isb_states[pick])

        # root_states / dof_state now hold the nominal pose for the envs not
        # chosen and the restored state for the chosen ones, so one write over
        # all of env_ids applies both. Writing `chosen` alone would replace
        # legged_gym's write and drop the nominal resets (see isb_reset_all).
        self._isb_push_states(env_ids if self.isb_reset_all else chosen)


# ===========================================================================
class ISBPPO(PPO):
    """PPO that also trains the contrastive projection, once per epoch."""

    projection_buffer = None       # set by the runner; None for non-cl arms
    isb_cl_losses = None

    def update(self):
        pb = self.projection_buffer
        if pb is None or getattr(pb, "sampling_strategy", None) != "network":
            return super().update()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        self.isb_cl_losses = []

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs)

        n_batches = 0
        for batch in generator:
            # An epoch boundary is every num_mini_batches batches. Fire the
            # projection step FIRST, so the values it sees are the critic as it
            # stands at the start of the epoch, matching the reference's
            # evaluate-then-train_step ordering.
            if n_batches % self.num_mini_batches == 0:
                self._projection_step()
            n_batches += 1
            mean_value_loss, mean_surrogate_loss = self._ppo_batch(
                batch, mean_value_loss, mean_surrogate_loss)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return mean_value_loss / num_updates, mean_surrogate_loss / num_updates

    def _projection_step(self):
        """One contrastive update on the FIXED sub-trajectory set."""
        pb = self.projection_buffer
        if pb.train_data is None:
            return
        with torch.no_grad():
            obs = pb.get_train_sample_observs()
            values = self.actor_critic.evaluate(obs).squeeze(-1)
        loss = pb.train_step(values.detach().cpu().numpy())
        if loss is not None:
            self.isb_cl_losses.append(float(loss))

    def _ppo_batch(self, batch, mean_value_loss, mean_surrogate_loss):
        """The body of rsl_rl's update loop, unchanged in substance.

        Reproduced rather than called because rsl_rl fuses the generator loop
        and the loss computation into one method with no seam to hook.
        """
        (obs_batch, critic_obs_batch, actions_batch, target_values_batch,
         advantages_batch, returns_batch, old_actions_log_prob_batch,
         old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch) = batch

        self.actor_critic.act(obs_batch, masks=masks_batch,
                              hidden_states=hid_states_batch[0])
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
        value_batch = self.actor_critic.evaluate(
            critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        entropy_batch = self.actor_critic.entropy

        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5)
                    + (torch.square(old_sigma_batch)
                       + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif self.desired_kl / 2.0 > kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.learning_rate

        ratio = torch.exp(actions_log_prob_batch
                          - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (
                value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        loss = (surrogate_loss + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                       self.max_grad_norm)
        self.optimizer.step()
        return (mean_value_loss + value_loss.item(),
                mean_surrogate_loss + surrogate_loss.item())


# ===========================================================================
class ISBOnPolicyRunner(OnPolicyRunner):
    """Runner that captures visited states and rebuilds the ISB each iteration.

    The arm is passed in as `isb_arm`. `vanilla` skips all of it and reduces to
    stock rsl_rl, which is the point: the baseline arm runs the unmodified
    algorithm, so any difference between arms is the buffer and nothing else.
    """

    isb_stat = None

    def init_isb(self, arm="pisb", alpha=0.7, p_reset=P_RESET, n_isb=N_ISB,
                 k_clusters=K_CLUSTERS, capacity=200000, seed=0,
                 min_timesteps=MIN_TIMESTEPS, cl_min_timesteps=8,
                 prio_raw=True, fifo_uniform=True, reset_all=True,
                 onmap_filter=True):
        self.isb_arm = arm
        self.isb_onmap = bool(onmap_filter)       # see state_buffer.on_map_mask
        self.isb_n_offmap = 0
        self.isb_n_offmap_calls = 0
        # Set on the env for every arm, vanilla included: its hard-reset path
        # has the same single-write issue.
        self.env.isb_reset_all = bool(reset_all)
        self.isb_prio_raw = bool(prio_raw)
        self.isb_fifo_uniform = bool(fifo_uniform)
        self.isb_alpha = float(alpha)
        self.isb_n = int(n_isb)
        self.isb_k = int(k_clusters)
        self.isb_min_ts = int(min_timesteps)
        self.isb_rng = np.random.default_rng(int(seed))
        self.env.isb_prob = 0.0 if arm == "vanilla" else float(p_reset)

        if arm == "vanilla":
            self.isb_buf = None
            self.alg.projection_buffer = None
            return

        self.isb_buf = LeggedStateBuffer(
            capacity=capacity, num_dof=self.env.num_dof,
            num_commands=self.env.cfg.commands.num_commands,
            device=self.device, fifo_uniform=self.isb_fifo_uniform)

        # The authors' library, unmodified. 'network' is the CL-Buffer,
        # 'observations' the Obs-Buffer, 'random' the Random-Buffer. P-ISB
        # borrows the 'observations' projection and replaces only the draw.
        from initial_buffer.algorithms.projection_buffer import ProjectionBuffer
        strategy = {"random": "random", "obs": "observations",
                    "cl": "network", "pisb": "observations"}[arm]
        self.projection_buffer = ProjectionBuffer(
            device=self.device,
            obs_dim=self.env.num_obs,
            advantage_gamma=self.alg.gamma,
            gae_lambda=self.alg.lam,
            nr_clusters=self.isb_k,
            sampling_strategy=strategy,
            # NOT self.isb_min_ts. That one is the paper's Sec V visited-state
            # pre-selection (15 steps from episode start). This one is the
            # minimum sub-trajectory length create_train_data will accept, which
            # toy_example.py sets to 8. num_steps_per_env is 24, so passing 15
            # here would leave 9 usable steps and starve the search.
            min_timesteps=int(cl_min_timesteps),
        )
        self.alg.projection_buffer = self.projection_buffer

    # ------------------------------------------------------------------
    def _rebuild_isb(self):
        """create_initial_state_buffer over the visited buffer -> env.isb_states."""
        if self.isb_buf is None or self.isb_buf.fill < self.isb_n:
            return
        states, obs, prio = self.isb_buf.view()
        idx = select_indices(self.isb_arm, self.projection_buffer, obs, prio,
                             n=self.isb_n, k=self.isb_k, alpha=self.isb_alpha,
                             rng=self.isb_rng)
        idx = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=self.device)
        self.env.isb_states = states[idx].clone()
        self.isb_stat = dict(self.isb_buf.stats())
        self.isb_stat["isb_pool"] = int(self.env.isb_states.shape[0])
        self.isb_stat["isb_unique"] = int(torch.unique(idx).numel())

    def _push_rollout(self, states_TN, obs_TN, ep_step_TN, acc_rew_TN, onmap_TN=None):
        """Pre-select, then append the rollout to the visited buffer.

        Priority is |A|, the paper's p_i = |A_i|.

        THIS IS CALLED AFTER compute_returns, and rsl_rl's compute_returns
        normalises storage.advantages in place to (A - mean) / std
        (rollout_storage.py:137). Reading that tensor gives |A - mean| / std,
        which ranks states by distance from the rollout-MEAN advantage rather
        than from zero -- not what the paper states, and not what the
        drone-racing P-ISB uses (ppo2.py reads raw GAE before normalising).
        returns and values are left unnormalised, so returns - values is the
        raw GAE advantage. prio_raw=False reproduces the reported runs.
        """
        if self.isb_buf is None:
            return
        if self.isb_prio_raw:
            adv = (self.alg.storage.returns - self.alg.storage.values).squeeze(-1)
        else:
            adv = self.alg.storage.advantages.squeeze(-1)      # legacy, normalised
        flat = lambda x: x.reshape(-1, x.shape[-1]) if x.dim() == 3 else x.reshape(-1)
        keep = preselect_mask(flat(ep_step_TN), flat(acc_rew_TN), self.isb_min_ts)
        if onmap_TN is not None:
            on = flat(onmap_TN)
            n_off = int((keep & ~on).sum())
            self.isb_n_offmap += n_off
            self.isb_n_offmap_calls += 1
            if n_off and (self.isb_n_offmap == n_off
                          or self.isb_n_offmap_calls % 100 == 0):
                print("[ISB] off-map filter dropped %d/%d pre-selected states "
                      "-- %d total" % (n_off, int(keep.sum()), self.isb_n_offmap),
                      flush=True)
            keep = keep & on
        self.isb_buf.add(flat(states_TN), flat(obs_TN), flat(adv), keep)

    # ------------------------------------------------------------------
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        from collections import deque
        import statistics, time, os
        from torch.utils.tensorboard import SummaryWriter

        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos, rewbuffer, lenbuffer = [], deque(maxlen=100), deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        T, N = self.num_steps_per_env, self.env.num_envs
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # ISB is rebuilt BEFORE the rollout, as in toy_example.py, so this
            # iteration's resets already draw from it.
            self._rebuild_isb()

            cap_states = torch.zeros(T, N, self.isb_buf.dim, device=self.device) \
                if self.isb_buf is not None else None
            cap_obs = torch.zeros(T, N, self.env.num_obs, device=self.device) \
                if self.isb_buf is not None else None
            cap_step = torch.zeros(T, N, device=self.device)
            cap_acc = torch.zeros(T, N, device=self.device)
            cap_onmap = torch.ones(T, N, dtype=torch.bool, device=self.device) \
                if (self.isb_buf is not None and self.isb_onmap) else None

            with torch.inference_mode():
                for i in range(T):
                    if self.isb_buf is not None:
                        cap_states[i] = LeggedStateBuffer.encode(self.env)
                        cap_obs[i] = obs
                        if cap_onmap is not None:
                            cap_onmap[i] = on_map_mask(self.env)
                    cap_step[i] = self.env.episode_length_buf.float()
                    cap_acc[i] = cur_reward_sum

                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (obs.to(self.device),
                                                       critic_obs.to(self.device),
                                                       rewards.to(self.device),
                                                       dones.to(self.device))
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time(); collection_time = stop - start; start = stop
                self.alg.compute_returns(critic_obs)

            self._push_rollout(cap_states, cap_obs, cap_step, cap_acc, cap_onmap)

            # Fix the 256 sub-trajectories this update's projection steps use.
            # Only the CL arm trains a network; the others have nothing to fit.
            if self.isb_arm == "cl":
                self.projection_buffer.create_train_data(
                    rollout_rewards=self.alg.storage.rewards.squeeze(-1).cpu().numpy(),
                    rollout_observations=cap_obs,
                    rollout_episode_starts=self.alg.storage.dones.squeeze(-1).cpu().numpy(),
                )

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time(); learn_time = stop - start
            if self.isb_arm != "vanilla":
                st = getattr(self, "isb_stat", None)
                cl = getattr(self.alg, "isb_cl_losses", None)
                print("  isb[%s] fill=%s added/iter=%s pool=%s unique=%s%s"
                      % (self.isb_arm,
                         st.get("buf_fill") if st else "-",
                         st.get("buf_added") if st else "-",
                         st.get("isb_pool") if st else "not built yet",
                         st.get("isb_unique") if st else "-",
                         ("  cl_steps=%d loss=%.4f" % (len(cl), cl[-1]))
                         if cl else ""))
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir,
                               'model_{}.pt'.format(self.current_learning_iteration)))
