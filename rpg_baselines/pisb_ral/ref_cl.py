# SPDX-License-Identifier: GPL-3.0
# Port of the projection network and loss of cl_initial_buffer
# (https://github.com/uzh-rpg/cl_initial_buffer), licensed under the GNU GPL
# v3.0; this file is distributed under the same licence (LICENSE-GPL-3.0).
"""Faithful port of the published Contrastive Initial State Buffer.

Ported from the reference `ProjectionBuffer` / `SoftNearestNeighborLoss`
(Messikommer et al., ICRA 2024), so that the CL-Buffer baseline is the
published method. This file matches the reference on all of the following:

    projection net       [obs_dim, 64, 64] Tanh, then Linear(64, 64)
    embedding dim        64
    optimiser            Adam, lr 1e-4
    loss                 soft-nearest-neighbour, temperature 0.5
    loss form            ONE anchor (the first positive), num and den are
                         MEANS, and den = mean(exp(neg/T)) + num
    mining               top / bottom 10 by improvement
    clustering           cosine k-means on L2-normalised embeddings
    selection            N/K nearest each centroid, ordered

ONE DOCUMENTED DEPARTURE, and the reason for it. The reference mines on
`gradient_improvement = improvement - prev_improvement`, differencing across
successive `train_step` calls over a FIXED set of 256 sub-trajectories held for
the rollout. `create_train_data` resets `prev_improvement` to None at the start
of each rollout, so the FIRST `train_step` of a rollout differences against
zeros and `gradient_improvement` is exactly `improvement`. Under a
once-per-rollout training schedule the two are therefore identical, and that is
the schedule this runner uses: one projection step per PPO update. `use_delta`
exposes the alternative for the case where the reference's loop turns out to
call `train_step` several times per rollout; it keeps the previous update's
improvement for the states it can match and differences against it.

`improvement` itself is not recomputed here. The PPO runner already produces
    dv[t] = A_t + gamma * (1 - done) * dv[t+1]
which is the discounted sum of future GAE advantages from t, i.e. exactly the
reference's `(advantages * gamma_t).sum(1)` for a sub-trajectory starting at t.
It arrives as `embed_signal`.
"""
import numpy as np

from rpg_baselines.pisb_ral.buffers import l2norm


class SoftNearestNeighborLoss:
    """The reference loss, structure preserved.

        num  = mean_j exp(cos(a, p_j) / T)
        den  = mean_j exp(cos(a, n_j) / T) + num
        loss = -log(num / den)

    Note this is NOT the textbook soft-nearest-neighbour form: there is a single
    anchor, the aggregations are means rather than sums, and the numerator is
    added into the denominator rather than the denominator ranging over all
    non-self pairs. Those differences change the gradient, so they are kept.
    """

    def __init__(self, torch, temperature=0.5):
        self.torch = torch
        self.temperature = float(temperature)

    def __call__(self, anchor, pos, neg):
        torch = self.torch
        # Rows reaching here are already L2-normalised by project_obs, so a dot
        # product IS the cosine similarity; normalising again is a no-op that
        # the reference performs defensively.
        a = torch.nn.functional.normalize(anchor, p=2, dim=-1)
        p = torch.nn.functional.normalize(pos, p=2, dim=-1)
        n = torch.nn.functional.normalize(neg, p=2, dim=-1)
        num_sim = (a * p).sum(-1)            # broadcasts [1,D] against [P,D]
        den_sim = (a * n).sum(-1)
        num = torch.exp(num_sim / self.temperature).mean(0)
        den = torch.exp(den_sim / self.temperature).mean(0) + num
        return -torch.log(num / den)


class RefProjectionBuffer:
    """The reference projection network, its contrastive training step, and its
    cluster-representative selection.

    Interface is deliberately narrow: train_step(feats, improvement) and
    select(feats, n, k) -> indices. The visited-state FIFO lives outside, in
    buffers.VisitedBuffer, exactly as the reference keeps its buffer outside
    ProjectionBuffer.
    """

    def __init__(self, obs_dim, embed_dim=64, hidden=64, temp=0.5,
                 n_mining=10, lr=1e-4, n_train_data=256, seed=0,
                 use_delta=False):
        import torch                       # lazy: only the CL arm needs torch
        import torch.nn as nn
        self.torch = torch
        torch.manual_seed(int(seed))

        # Reference setup_projection_network(): cluster_net_arch = [obs, 64, 64]
        # builds Linear+Tanh for each successive pair, then one final Linear to
        # cluster_embedding_dim. Two Tanh hidden layers, linear output.
        arch = [int(obs_dim), int(hidden), int(hidden)]
        layers = []
        last = arch[0]
        for out in arch[1:]:
            layers += [nn.Linear(last, out), nn.Tanh()]
            last = out
        layers.append(nn.Linear(last, int(embed_dim)))
        self.net = nn.Sequential(*layers)

        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(lr))
        self.loss_fn = SoftNearestNeighborLoss(torch, temperature=temp)
        self.n_mining = int(n_mining)
        self.n_train_data = int(n_train_data)
        self.use_delta = bool(use_delta)
        self.rng = np.random.default_rng(int(seed))
        self.prev_improvement = None
        self.prev_feats = None
        self.last_loss = None
        self.steps = 0

    # ---------------------------------------------------------------- embed
    def project(self, feats, grad=False):
        """project_obs(): forward pass then L2 normalisation."""
        torch = self.torch
        x = torch.as_tensor(np.asarray(feats, np.float32))
        if grad:
            return torch.nn.functional.normalize(self.net(x), p=2, dim=-1)
        with torch.no_grad():
            z = torch.nn.functional.normalize(self.net(x), p=2, dim=-1)
        return z.numpy()

    # ----------------------------------------------------------- train step
    def train_step(self, feats, improvement):
        """One contrastive step. `improvement` is the discounted advantage sum
        per candidate state, the reference's policy_improvement() output."""
        torch = self.torch
        feats = np.asarray(feats, np.float32)
        improvement = np.asarray(improvement, np.float32).ravel()
        n = len(feats)
        if n < 2 * self.n_mining + 1:
            return None

        # Reference create_train_data() draws n_train_data sub-trajectories from
        # the rollout and mines among THOSE, not among every timestep. Sampling
        # the same number of candidate starts keeps the mining pool the same
        # size, so the top/bottom 10 are as selective as the reference's.
        take = min(self.n_train_data, n)
        cand = self.rng.choice(n, size=take, replace=False)
        f_c, imp = feats[cand], improvement[cand]

        signal = imp
        if self.use_delta:
            # See the module docstring. Only reachable when explicitly asked
            # for; the default schedule makes delta and level identical.
            if self.prev_improvement is not None \
                    and len(self.prev_improvement) == len(imp):
                signal = imp - self.prev_improvement
            else:
                signal = imp - np.zeros_like(imp)
            self.prev_improvement = imp.copy()

        k = self.n_mining
        # Reference uses argpartition to grab the k largest and k smallest.
        pos_ids = np.argpartition(signal, -k)[-k:]      # highest improvement
        neg_ids = np.argpartition(signal, k)[:k]        # lowest improvement

        pos = self.project(f_c[pos_ids], grad=True)
        neg = self.project(f_c[neg_ids], grad=True)
        # cluster_train_step(): anchor is the FIRST positive, the remaining
        # positives are the positive set.
        loss = self.loss_fn(pos[0:1], pos[1:], neg)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.last_loss = float(loss.item())
        self.steps += 1
        return self.last_loss

    # ------------------------------------------------------------- selection
    def select(self, feats, n, k):
        """create_initial_state_buffer() for sampling_strategy='network':
        project, cosine-cluster, take the N/K nearest each centroid."""
        from rpg_baselines.pisb_ral.buffers import kmeans_cosine, nearest_per_center
        emb = self.project(feats)                    # already L2-normalised
        labels, centers, kk = kmeans_cosine(
            emb, k, seed=int(self.rng.integers(1 << 30)))
        return nearest_per_center(l2norm(emb), labels, centers, kk, n, self.rng)

    def select_priority(self, feats, priorities, n, k, alpha, rng, eps=1e-3):
        """The `clpisb` selection: this embedding's clusters, P-ISB's draw.

        Identical to select() up to and including the cosine k-means on the
        L2-normalised embedding, then replaces the reference's
        nearest-to-centroid rule with the submitted method's
        (|A| + eps)^alpha within-cluster draw.

        HOLDING THE CLUSTERING FIXED IS THE POINT. Because the projection, the
        metric and the k-means seed stream are all shared with select(), a
        difference between `clpisb` and `cl` is attributable to the selection
        rule and to nothing else. That is the comparison the reference paper's
        own conclusion asks for -- "future work can extend our CL-Buffer to
        consider a prioritized sampling from the initial state buffer" -- and
        it is the one the submitted P-ISB never ran, because it clusters raw
        observations instead.
        """
        from rpg_baselines.pisb_ral.buffers import (
            kmeans_cosine, priority_per_cluster)
        emb = self.project(feats)                    # already L2-normalised
        labels, _centers, kk = kmeans_cosine(
            emb, k, seed=int(self.rng.integers(1 << 30)))
        return priority_per_cluster(np.asarray(priorities, np.float32),
                                    labels, kk, n, alpha, rng, eps=eps)

    def state_dict_numpy(self):
        """Projection weights, so the trained embedding can be inspected after
        the run rather than being lost with the process."""
        return {k: v.detach().cpu().numpy()
                for k, v in self.net.state_dict().items()}
