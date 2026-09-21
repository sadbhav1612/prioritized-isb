"""The reset-selection arms, in one file.

All five attach as `model.pisb` and expose the signature the PPO runner already
calls, so NO shared code changes:

    update(states, feats, advantages, from_start=None, embed_signal=None,
           recover_signal=None)

`advantages` is the sampling priority |A|; `embed_signal` is dv, the discounted
sum of future GAE advantages, which is what the reference CL trains its
embedding on. `from_start` and `recover_signal` are accepted and ignored; the
PPO runner passes them for other buffer variants.

THE ARMS

  vanilla  no buffer at all. Never instantiated; the runner sets the reset
           probability to 0 and attaches nothing, so every reset comes from
           the environment's base distribution (30% at the race start, 70%
           along a randomly chosen gate segment).
  random   uniform draw from the visited set. Reference
           sampling_strategy='random'.
  obs      cosine k-means on RAW L2-normalised observations, then the n/K
           nearest each centroid (n = pool size). Reference sampling_strategy='observations'.
  cl       the same, on the learned contrastive embedding. Reference
           sampling_strategy='network'. See ref_cl.py.
  pisb     P-ISB: z-scored observations, Euclidean k-means, then
           a draw within each cluster with probability proportional to
           (|A| + 1e-3)^alpha.

WHY obs AND cl SHARE A SELECTION RULE BUT pisb DOES NOT. `obs` and `cl` are the
reference's own two sampling strategies and differ ONLY in the projection, which
isolates what the learned embedding buys. P-ISB's contribution is the within-cluster priority draw, so
giving it centroid selection would delete the method. P-ISB at alpha=0 (a
uniform draw within its clusters) is NOT the same estimator as the reference's
obs baseline, and the two are kept as separate arms.
"""
import numpy as np

from rpg_baselines.pisb_ral.buffers import (
    VisitedBuffer, STATE_DIM, zscore, l2norm,
    kmeans_euclidean, kmeans_cosine,
    nearest_per_center, priority_per_cluster,
)

# The last three are variant arms. Both are P-ISB with ONLY the
# priority scalar changed -- same clustering, same alpha, same pool size -- so
# any difference is attributable to the score and nothing else.
#   signed  : max(A, 0) instead of |A|, an alternative priority score
#   refresh : |ret - V_now(s)|, recomputed each iteration against the current
#             critic, the prioritized-replay analogue of refreshing priorities
#   clpisb  : the reference CL embedding for clustering and P-ISB's priority
#             draw for selection. Not reported in the paper.
ARMS = ('vanilla', 'random', 'obs', 'cl', 'pisb', 'signed', 'refresh',
        'clpisb')


class PISBRALController:
    """One controller, five arms, selected by `arm`."""

    def __init__(self, racing_env, arm='pisb', alpha=0.7, n_pool=256,
                 k_clusters=64, capacity=20000, gate_min=1, refresh_every=1,
                 seed=0,
                 # reference CL constants; defaults ARE the published values
                 cl_embed_dim=64, cl_hidden=64, cl_temp=0.5, cl_mining=10,
                 cl_lr=1e-4, cl_train_data=256, cl_use_delta=False):
        assert arm in ARMS, arm
        assert arm != 'vanilla', "vanilla attaches no controller"
        self.env = racing_env
        self.arm = arm
        self.alpha = float(alpha)
        self.n_pool = int(n_pool)
        self.k = int(k_clusters)
        self.refresh_every = int(refresh_every)
        self.rng = np.random.default_rng(int(seed))
        self.buf = VisitedBuffer(capacity=capacity, state_dim=STATE_DIM,
                                 gate_min=gate_min)
        self.updates = 0
        self.last_pool_size = 0
        self.proj = None
        # Set by scripts/run_pisb_ral.py for the `refresh` arm only: a callable
        # obs -> V(obs) using the CURRENT critic. Left None for every other arm,
        # and `refresh` degrades to frozen |A| rather than crashing if it is
        # never attached.
        self.value_fn = None
        self._cl_cfg = dict(embed_dim=cl_embed_dim, hidden=cl_hidden,
                            temp=cl_temp, n_mining=cl_mining, lr=cl_lr,
                            n_train_data=cl_train_data, seed=seed,
                            use_delta=cl_use_delta)

    # ------------------------------------------------ runner-facing hooks
    def get_full_states(self):
        return self.env.get_full_states()

    def update(self, states, feats, advantages, from_start=None,
               embed_signal=None, recover_signal=None):
        dv = embed_signal if embed_signal is not None else advantages
        # recover_signal is the return-to-go; the `refresh` arm holds it as the
        # fixed target it re-scores against. Harmless (unused) for every other
        # arm, and None-safe inside add().
        self.buf.add(states, feats, advantages, dv=dv, ret=recover_signal)

        if self.arm in ('cl', 'clpisb'):
            if self.proj is None and self.buf.feats is not None:
                from rpg_baselines.pisb_ral.ref_cl import RefProjectionBuffer
                self.proj = RefProjectionBuffer(
                    obs_dim=self.buf.feats.shape[1], **self._cl_cfg)
            if self.proj is not None:
                # Trained on the CURRENT rollout, matching the reference, which
                # builds its train data from the rollout just collected rather
                # than from the whole visited buffer.
                self.proj.train_step(np.asarray(feats, np.float32),
                                     np.asarray(dv, np.float32))

        self.updates += 1
        if self.updates % self.refresh_every == 0:
            pool = self._sample_pool()
            self.last_pool_size = len(pool)
            if len(pool) > 0:
                for i in range(self.env.num_envs):
                    self.env.set_reset_pool(i, pool)

    # ----------------------------------------------------------- the pool
    def _sample_pool(self):
        S, F, P, _dv = self.buf.view()
        n = self.n_pool
        if len(S) == 0:
            return np.zeros((0, STATE_DIM), np.float32)

        if self.arm == 'random':
            idx = self.rng.choice(len(S), size=min(n, len(S)), replace=False)
            return S[idx].copy()

        if self.arm == 'obs':
            Xn = l2norm(F)
            labels, centers, kk = kmeans_cosine(
                Xn, self.k, seed=int(self.rng.integers(1 << 30)))
            idx = nearest_per_center(Xn, labels, centers, kk, n, self.rng)

        elif self.arm == 'cl':
            if self.proj is None:        # net not built yet -> uniform
                idx = self.rng.choice(len(S), size=min(n, len(S)), replace=False)
            else:
                idx = self.proj.select(F, n, self.k)

        elif self.arm == 'clpisb':
            if self.proj is None:        # net not built yet -> uniform, as cl
                idx = self.rng.choice(len(S), size=min(n, len(S)), replace=False)
            else:
                idx = self.proj.select_priority(F, P, n, self.k,
                                                self.alpha, self.rng)

        else:                            # pisb as submitted, and its variants
            labels, _c, kk = kmeans_euclidean(
                zscore(F), self.k, seed=int(self.rng.integers(1 << 30)))
            if self.arm in ('signed', 'refresh'):
                P = self._variant_priority(F, P)
            idx = priority_per_cluster(P, labels, kk, n, self.alpha, self.rng)

        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) < n:                 # even allocation under-filled
            rest = np.setdiff1d(np.arange(len(S)), idx)
            if len(rest):
                extra = self.rng.choice(
                    rest, size=min(n - len(idx), len(rest)), replace=False)
                idx = np.concatenate([idx, extra])
        return S[idx[:n]].copy()

    def _variant_priority(self, F, frozen):
        """The priority scalar for the two variant arms.

        Everything else about the draw is identical to `pisb` -- same k-means on
        the same z-scored features, same alpha, same per-cluster allocation --
        so a difference between these arms and `pisb` isolates the SCORE.
        """
        raw, ret = self.buf.view_variant()

        if self.arm == 'signed':
            # Positive advantage only. |A| treats "the critic badly
            # under-estimated this state" and "badly over-estimated it" as
            # equally worth revisiting. Clipping at zero keeps
            # only the first, and states the critic over-values fall back to
            # the eps floor inside priority_per_cluster rather than competing.
            return np.maximum(raw, 0.0)

        # refresh: re-score against the CURRENT critic each time the pool is
        # rebuilt, holding the insertion-time return-to-go as the target. That
        # is a present-tense advantage estimate, so priorities track the critic
        # instead of freezing, the prioritized-replay analogue. Falls back to the frozen |A| if no critic was attached.
        if self.value_fn is None:
            return frozen
        try:
            v_now = np.asarray(self.value_fn(F), np.float32).ravel()
        except Exception:
            return frozen
        if v_now.shape[0] != ret.shape[0]:
            return frozen
        return np.abs(ret - v_now)

    # ---------------------------------------------------------- diagnostics
    def stats(self):
        s = self.buf.stats()
        s["pool"] = self.last_pool_size
        s["arm"] = self.arm
        if self.arm in ('cl', 'clpisb') and self.proj is not None:
            s["cl_loss"] = self.proj.last_loss
            s["cl_steps"] = self.proj.steps
        return s

    def save_projection(self, stem):
        """Persist the learned embedding next to the policy. The projection is
        part of the method, not a diagnostic: without it the trained embedding
        cannot be inspected after the run."""
        if self.arm not in ('cl', 'clpisb') or self.proj is None:
            return None
        import json
        w = self.proj.state_dict_numpy()
        np.savez(stem + "_clproj.npz", **w)
        with open(stem + "_clproj.json", "w") as f:
            json.dump({k: list(v.shape) for k, v in w.items()}, f, indent=2)
        return stem + "_clproj.npz"
