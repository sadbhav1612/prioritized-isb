"""Visited-state buffer and clustering primitives shared by every
reset-selection arm.

Two clustering conventions live side by side, and the difference is deliberate:

  z-score + Euclidean k-means   -> what P-ISB does (a per-dimension
                                   mean/std standardisation, not L2).
  L2-normalise + cosine k-means -> what the published CL-Buffer reference does
                                   (fast_pytorch_kmeans with mode='cosine' on
                                   projected, L2-normalised embeddings).

Using one for both would either make the baseline unfaithful or silently
redefine P-ISB, so each arm keeps its own.
"""
import numpy as np

# 17-d full state layout (C++ getFullState):
#   [pos(0:3), quat(3:7), vel(7:10), omega(10:13), gate_idx(13), last_passed(14:17)]
STATE_DIM = 17
GATE_IDX_COL = 13


# --------------------------------------------------------------- normalisation
def zscore(X, eps=1e-6):
    """Per-dimension standardisation. This is what the submitted P-ISB applies
    to observations before k-means ((f - mu) / (sd + 1e-6), as in the first
    release's pisb.py, kept in this repository's history)."""
    X = np.asarray(X, np.float32)
    mu = X.mean(0)
    sd = X.std(0) + eps
    return (X - mu) / sd


def l2norm(X, eps=1e-12):
    """Project rows onto the unit sphere. Required before cosine k-means, and
    what the reference's project_obs() does via F.normalize(..., p=2)."""
    X = np.asarray(X, np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


# ------------------------------------------------------------------- k-means
def _kmeanspp_init(X, k, rng, sqdist):
    """k-means++ seeding, shared by both k-means variants. `sqdist(X, c)` gives
    the per-row distance to a single centre under the relevant geometry."""
    n = len(X)
    centers = np.empty((k, X.shape[1]), dtype=np.float32)
    centers[0] = X[rng.integers(n)]
    d = sqdist(X, centers[0])
    for c in range(1, k):
        tot = d.sum()
        probs = (d / tot) if tot > 1e-12 else None
        centers[c] = X[rng.choice(n, p=probs)] if probs is not None \
            else X[rng.integers(n)]
        d = np.minimum(d, sqdist(X, centers[c]))
    return centers


def kmeans_euclidean(X, k, iters=10, seed=0):
    """Euclidean k-means++ on z-scored features. Returns (labels, centers, k).

    This is the SUBMITTED P-ISB clustering, reproduced from the first release's pisb.py including
    its early-exit on label stability and its 10-iteration cap.
    """
    X = np.asarray(X, np.float32)
    rng = np.random.default_rng(int(seed))
    n = len(X)
    k = max(1, min(int(k), n))
    centers = _kmeanspp_init(X, k, rng, lambda A, c: ((A - c) ** 2).sum(1))
    labels = np.zeros(n, dtype=np.int64)
    x2 = (X ** 2).sum(1)
    for _ in range(iters):
        # ||x-c||^2 = x^2 - 2 x.c + c^2, avoiding an [n, k, d] temporary.
        d = x2[:, None] - 2.0 * (X @ centers.T) + (centers ** 2).sum(1)[None, :]
        new_labels = d.argmin(1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = X[m].mean(0)
    return labels, centers, k


def kmeans_cosine(X, k, iters=10, seed=0):
    """Cosine k-means++ on L2-normalised rows. Returns (labels, centers, k),
    with centers themselves L2-normalised.

    Reproduces fast_pytorch_kmeans(mode='cosine') as the reference uses it:
    assignment maximises cosine similarity, the centroid update is the mean of
    the assigned unit vectors renormalised back onto the sphere, and the
    distance the reference reports is 1 - cos.
    """
    X = l2norm(X)
    rng = np.random.default_rng(int(seed))
    n = len(X)
    k = max(1, min(int(k), n))
    # clip at 0: for the centre row itself the dot product of a unit vector
    # with itself can land a few ulp above 1.0, and k-means++ needs a
    # non-negative distance to build its sampling distribution.
    def _cos_d(A, c):
        return np.maximum(0.0, 1.0 - (A @ c))
    centers = l2norm(_kmeanspp_init(X, k, rng, _cos_d))
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        sims = X @ centers.T                    # cosine, rows are unit norm
        new_labels = sims.argmax(1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = X[m].mean(0)
        centers = l2norm(centers)
    return labels, centers, k


# ------------------------------------------------------ within-cluster choice
def nearest_per_center(Xn, labels, centers, k, n, rng):
    """Reference selection: within each cluster keep the N/K members CLOSEST to
    the centre, ordered by distance, then flatten.

    This is create_initial_state_buffer():
        k = buffer_length // nr_clusters
        take the k smallest cluster distances, sort them, flatten
    The reference's own comment on the return is "Important: ordering", so the
    per-cluster sort is kept rather than treated as incidental. Deterministic
    given the clustering: no sampling happens here at all.
    """
    centers_n = l2norm(centers)
    per = max(1, int(n // max(1, k)))
    chosen = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        # distance = 1 - cos, so ordering by -sim is ordering by distance.
        sims = Xn[members] @ centers_n[c]
        take = min(per, len(members))
        chosen.append(members[np.argsort(-sims)[:take]])
    if not chosen:
        return np.array([], dtype=np.int64)
    idx = np.concatenate(chosen)
    if len(idx) > n:
        idx = idx[:n]
    return idx


def priority_per_cluster(priorities, labels, k, n, alpha, rng, eps=1e-3):
    """Submitted P-ISB selection: even slots per populated cluster (diversity),
    then draw WITHIN each cluster with probability proportional to
    (priority + eps)^alpha (focus). alpha = 0 gives a uniform draw, which is the
    observation-clustering corner. Tops up to n from the priority-weighted
    remainder when the even allocation under-fills.

    Reproduced from the first release's pisb.py, including eps = 1e-3 and replace=False.
    """
    clusters = [np.where(labels == c)[0] for c in range(k)]
    clusters = [c for c in clusters if len(c) > 0]
    if not clusters:
        return np.array([], dtype=np.int64)
    per = max(1, n // len(clusters))
    chosen = []
    for members in clusters:
        w = np.power(priorities[members] + eps, alpha)
        tot = w.sum()
        prob = (w / tot) if tot > 0 else None
        take = min(per, len(members))
        chosen.append(rng.choice(members, size=take, replace=False, p=prob))
    idx = np.concatenate(chosen)
    if len(idx) < n:
        w = np.power(priorities + eps, alpha)
        w[idx] = 0.0
        tot = w.sum()
        if tot > 0:
            extra = rng.choice(len(priorities),
                               size=min(n - len(idx), int((w > 0).sum())),
                               replace=False, p=w / tot)
            idx = np.concatenate([idx, extra])
    return idx[:n]


# --------------------------------------------------------------- the buffer V
class VisitedBuffer:
    """Rolling FIFO of visited states with their clustering features and two
    scalar streams: the sampling priority |A| and the embedding signal DeltaV.

    TURNOVER. Capacity is
    20000 while one PPO iteration pushes 14k to 25k post-gate_min states, so the
    whole buffer is replaced roughly every iteration and no priority in it is
    more than about one critic update old. `added_total` is recorded so the
    turnover can be measured rather than assumed.
    """

    def __init__(self, capacity=20000, state_dim=STATE_DIM, gate_min=1,
                 fifo_uniform=False, seed=0):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.gate_min = int(gate_min)
        # See add(). When one iteration inserts more than `capacity` states,
        # the default keeps the most recent ones; every reported racing result
        # uses this default. fifo_uniform=True keeps a uniform subsample
        # instead.
        self.fifo_uniform = bool(fifo_uniform)
        self._rng = np.random.default_rng(int(seed))
        self.states = np.zeros((self.capacity, self.state_dim), np.float32)
        self.feats = None                 # allocated on first add
        self.prio = np.zeros(self.capacity, np.float32)
        self.dv = np.zeros(self.capacity, np.float32)
        # Two extra streams for the variant arms. Both stay zero
        # unless `add` is given them, so every existing arm is unaffected.
        #   raw : the SIGNED advantage, before the abs() below. The `signed`
        #         arm needs the sign that |A| discards.
        #   ret : the return-to-go recorded at insertion. The `refresh` arm
        #         re-derives a priority as |ret - V_now(s)|, which is a current
        #         advantage estimate against a fixed target: the
        #         prioritized-replay analogue of refreshing priorities.
        self.raw = np.zeros(self.capacity, np.float32)
        self.ret = np.zeros(self.capacity, np.float32)
        self.write = 0
        self.fill = 0
        self.added_total = 0
        self.last_added = 0

    def add(self, states, feats, priorities, dv=None, ret=None):
        states = np.asarray(states, np.float32)
        feats = np.asarray(feats, np.float32)
        # Keep the signed values before abs(): ppo2 hands us mb_advs, which is
        # signed, and every existing arm wants the magnitude. `raw` preserves
        # the sign for the variant arms without changing what `prio` means.
        raw = np.asarray(priorities, np.float32).copy()
        priorities = np.abs(np.asarray(priorities, np.float32))
        dv = np.zeros(len(states), np.float32) if dv is None \
            else np.asarray(dv, np.float32)
        ret = np.zeros(len(states), np.float32) if ret is None \
            else np.asarray(ret, np.float32)

        if self.gate_min > 0:
            keep = states[:, GATE_IDX_COL] >= self.gate_min
            states, feats = states[keep], feats[keep]
            priorities, dv = priorities[keep], dv[keep]
            raw, ret = raw[keep], ret[keep]

        m = len(states)
        self.last_added = m
        if m == 0:
            return
        if self.feats is None:
            self.feats = np.zeros((self.capacity, feats.shape[1]), np.float32)
        # A batch larger than the buffer would wrap onto itself, so it has to
        # be cut to `capacity`. KNOWN BIAS IN THE LEGACY PATH: ppo2 flattens the
        # rollout time-first, fs.reshape(T*N, -1) on [T, N, ...], so the "most
        # recent" rows of ONE batch are just its LAST TIMESTEPS -- every row in
        # a batch is equally recent. With T=250, N=100 and 14k-25k rows passing
        # gate_min, the tail drops the first ~0-50 timesteps whenever a rollout
        # exceeds capacity. It hits all buffer arms equally, so it cannot make
        # one arm beat another, but it does skew every buffer toward late
        # states. fifo_uniform=True keeps a uniform subsample instead.
        if m > self.capacity:
            if self.fifo_uniform:
                keep = self._rng.permutation(m)[:self.capacity]
            else:
                keep = np.arange(m - self.capacity, m)            # legacy tail
            states, feats = states[keep], feats[keep]
            priorities, dv = priorities[keep], dv[keep]
            raw, ret = raw[keep], ret[keep]
            m = self.capacity
        idx = (self.write + np.arange(m)) % self.capacity
        self.states[idx] = states
        self.feats[idx] = feats
        self.prio[idx] = priorities
        self.dv[idx] = dv
        self.raw[idx] = raw
        self.ret[idx] = ret
        self.write = int((self.write + m) % self.capacity)
        self.fill = int(min(self.capacity, self.fill + m))
        self.added_total += m

    def view(self):
        f = self.feats[:self.fill] if self.feats is not None else None
        return self.states[:self.fill], f, self.prio[:self.fill], self.dv[:self.fill]

    def view_variant(self):
        """The signed advantage and the insertion-time return-to-go, for the
        `signed` and `refresh` arms. Separate from view() so the four-tuple
        every existing arm unpacks keeps its shape."""
        return self.raw[:self.fill], self.ret[:self.fill]

    def stats(self):
        p = self.prio[:self.fill]
        return {
            # MUST be "fill", not "buf_fill": the training callback indexes
            # st['fill'] without a guard. "buf_fill" is the name of the log
            # column, not of this key.
            "fill": int(self.fill),
            "buf_added": int(self.last_added),
            # Iterations for the FIFO to fully replace itself at the current
            # write rate: the staleness bound of the stored priorities.
            "buf_turnover_iters": float(self.capacity / max(1, self.last_added)),
            "prio_mean": float(p.mean()) if self.fill else 0.0,
            "prio_max": float(p.max()) if self.fill else 0.0,
        }
