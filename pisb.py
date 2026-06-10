"""Prioritized Initial State Buffer (P-ISB) — Milestone 2, Python-owned.

The buffer lives on the algorithm side (the PPO loop) so priorities can use
the EXACT learning signal from each rollout (|GAE advantage| or |TD-error|),
unlike the C++ MVP which only had a critic-free reward-surprise proxy.

Pipeline per PPO update:
  1. add(states, feats, priorities): append the rollout's per-step full
     states (re-initable payloads), clustering features (the observations),
     and priorities (e.g. |advantage|) into a rolling buffer.
  2. sample_pool(n, alpha, k): build the reset pool with diversity-preserving
     two-level sampling — k-means cluster on feats (diversity), allocate pool
     slots evenly across clusters, then within each cluster draw states with
     probability ∝ priority^alpha (focus). alpha=0 reduces to the Obs-Buffer;
     k=1 reduces to flat priority sampling.

The caller pushes the returned pool to each env via set_reset_pool(); the C++
reset() then draws from it uniformly (with past_traj_prob), so the uniform
C++ draw reproduces the distribution Python constructed here.
"""
import numpy as np


def _kmeans(X, k, iters=10, seed=0):
    """Tiny k-means++ (numpy-only, no sklearn dependency). Returns labels."""
    rng = np.random.default_rng(seed)
    n = len(X)
    k = max(1, min(k, n))
    # k-means++ init.
    centers = np.empty((k, X.shape[1]), dtype=X.dtype)
    centers[0] = X[rng.integers(n)]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for c in range(1, k):
        probs = d2 / max(d2.sum(), 1e-12)
        centers[c] = X[rng.choice(n, p=probs)]
        d2 = np.minimum(d2, ((X - centers[c]) ** 2).sum(1))
    labels = np.zeros(n, dtype=np.int64)
    x2 = (X ** 2).sum(1)
    for _ in range(iters):
        # Assign via ||x-c||^2 = x^2 - 2 x.c + c^2 (avoids an [n,k,feat] temp).
        dists = x2[:, None] - 2.0 * (X @ centers.T) + (centers ** 2).sum(1)[None, :]
        new_labels = dists.argmin(1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        # Update.
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = X[m].mean(0)
    return labels, k


class PrioritizedInitialStateBuffer:
    def __init__(self, capacity=20000, state_dim=17, feat_dim=None,
                 k_clusters=64, normalize_feats=True, seed=0):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.feat_dim = feat_dim  # allocated lazily on first add() if None
        self.k_clusters = int(k_clusters)
        self.normalize_feats = normalize_feats
        self.rng = np.random.default_rng(seed)

        self.states = np.zeros((self.capacity, self.state_dim), np.float32)
        self.feats = None if feat_dim is None else \
            np.zeros((self.capacity, int(feat_dim)), np.float32)
        self.priorities = np.zeros(self.capacity, np.float32)
        self.write = 0
        self.fill = 0

    def add(self, states, feats, priorities, gate_min=1):
        """Append a flat batch. Optionally drop pre-gate_min states so the
        buffer biases toward mid/late-track states (matches the C++ filter
        and the paper's drone-racing pre-selection)."""
        states = np.asarray(states, np.float32)
        feats = np.asarray(feats, np.float32)
        priorities = np.abs(np.asarray(priorities, np.float32))
        if gate_min is not None and gate_min > 0:
            # gate_idx is index 13 of the full state.
            keep = states[:, 13] >= gate_min
            states, feats, priorities = states[keep], feats[keep], priorities[keep]
        m = len(states)
        if m == 0:
            return
        if self.feats is None:
            self.feat_dim = feats.shape[1]
            self.feats = np.zeros((self.capacity, self.feat_dim), np.float32)
        idx = (self.write + np.arange(m)) % self.capacity
        self.states[idx] = states
        self.feats[idx] = feats
        self.priorities[idx] = priorities
        self.write = int((self.write + m) % self.capacity)
        self.fill = int(min(self.capacity, self.fill + m))

    def sample_pool(self, n, alpha, k=None):
        """Return [n, state_dim] reset states via diversity-preserving
        prioritization. Falls back gracefully when the buffer is small."""
        if self.fill == 0:
            return np.zeros((0, self.state_dim), np.float32)
        n = int(n)
        f = self.feats[:self.fill]
        s = self.states[:self.fill]
        p = self.priorities[:self.fill]
        k = self.k_clusters if k is None else int(k)

        feat = f
        if self.normalize_feats:
            mu = f.mean(0)
            sd = f.std(0) + 1e-6
            feat = (f - mu) / sd

        labels, k = _kmeans(feat, k, seed=int(self.rng.integers(1 << 30)))

        # Even slot allocation across clusters = diversity (paper's N/K).
        clusters = [np.where(labels == c)[0] for c in range(k)]
        clusters = [c for c in clusters if len(c) > 0]
        if not clusters:
            return np.zeros((0, self.state_dim), np.float32)
        per = max(1, n // len(clusters))

        chosen = []
        for members in clusters:
            w = np.power(p[members] + 1e-3, alpha)
            tot = w.sum()
            prob = (w / tot) if tot > 0 else None
            take = min(per, len(members))
            pick = self.rng.choice(members, size=take, replace=False, p=prob)
            chosen.append(pick)
        chosen = np.concatenate(chosen)
        # Top up to n if even allocation under-filled (priority-weighted).
        if len(chosen) < n:
            w = np.power(p + 1e-3, alpha)
            w[chosen] = 0.0
            tot = w.sum()
            if tot > 0:
                extra = self.rng.choice(
                    self.fill, size=min(n - len(chosen), int((w > 0).sum())),
                    replace=False, p=w / tot)
                chosen = np.concatenate([chosen, extra])
        return s[chosen[:n]].copy()

    def stats(self):
        p = self.priorities[:self.fill]
        return {
            "fill": int(self.fill),
            "prio_mean": float(p.mean()) if self.fill else 0.0,
            "prio_max": float(p.max()) if self.fill else 0.0,
        }


class PISBController:
    """Glue between the PPO runner and the P-ISB buffer.

    Each PPO update the runner calls update() with the rollout's per-step
    full states, observation features, and |advantage| priorities. The
    controller fills the buffer and, every refresh_every updates, samples a
    fresh prioritized+diverse reset pool and pushes it to every env (the C++
    reset() then draws from it with prob past_traj_prob — ramped by the
    existing training callback). Attach as `model.pisb` so the runner finds it.
    """

    def __init__(self, racing_env, alpha=0.7, n_pool=256, k_clusters=64,
                 capacity=20000, gate_min=1, refresh_every=1, seed=0):
        self.env = racing_env
        self.alpha = float(alpha)
        self.n_pool = int(n_pool)
        self.k = int(k_clusters)
        self.gate_min = int(gate_min)
        self.refresh_every = int(refresh_every)
        self.buf = PrioritizedInitialStateBuffer(
            capacity=capacity, state_dim=17, feat_dim=None,
            k_clusters=k_clusters, seed=seed)
        self.updates = 0
        self.last_pool_size = 0

    def get_full_states(self):
        return self.env.get_full_states()

    def update(self, states, feats, advantages):
        """states [M,17], feats [M,F], advantages [M] (priority = |adv|)."""
        self.buf.add(states, feats, advantages, gate_min=self.gate_min)
        self.updates += 1
        if self.updates % self.refresh_every == 0:
            pool = self.buf.sample_pool(self.n_pool, self.alpha, self.k)
            self.last_pool_size = len(pool)
            if len(pool) > 0:
                for i in range(self.env.num_envs):
                    self.env.set_reset_pool(i, pool)

    def stats(self):
        s = self.buf.stats()
        s["pool"] = self.last_pool_size
        return s
