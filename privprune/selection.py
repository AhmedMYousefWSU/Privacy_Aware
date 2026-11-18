"""Privacy-aware token selection.

    max_x  sum_i x_i R_i - λ sum_i x_i P_i   s.t.  sum_i x_i = K,  x_i ∈ {0,1}
is solved exactly by keeping the top-K tokens of S_i = R_i - λ P_i.
"""
import numpy as np


def normalize(r, mode="rank", eps=1e-8):
    """Map raw relevance to [0,1] so it is on the same scale as P.

    minmax: (r - min) / (max - min). Sensitive to attention-sink outliers: a single huge value
            squashes every other token towards 0, which silently makes λ much stronger.
    rank:   r_i -> rank(r_i)/(N-1). Uniform on [0,1] for every image/question, so λ has the same
            meaning everywhere. Used by default; minmax is kept as an ablation.
    """
    r = np.asarray(r, dtype=np.float64)
    if mode == "minmax":
        return (r - r.min()) / (r.max() - r.min() + eps)
    if mode == "rank":
        ranks = np.empty(len(r))
        ranks[np.argsort(r, kind="stable")] = np.arange(len(r))
        return ranks / max(len(r) - 1, 1)
    raise ValueError(mode)


def num_keep(ratio, n=576):
    return max(1, int(round(ratio * n)))


def select(method, k, R=None, P=None, lam=0.0, tau=None, rng=None, n=576):
    """Return sorted indices of kept tokens.

    method:
      full           all tokens (K ignored)
      random         uniform random K
      relevance      top-K of R                (λ = 0; standard task-driven pruning)
      privacy_only   top-K of -P               (drop the most private tokens, ignore the task)
      soft           top-K of R - λP           (proposed)
      hard           top-K of R among tokens with P < τ; never admits P ≥ τ, may return < K
    """
    if method == "full":
        return np.arange(n)
    if method == "random":
        rng = rng or np.random.default_rng(0)
        return np.sort(rng.choice(n, size=k, replace=False))
    if method == "relevance":
        s = np.asarray(R, dtype=np.float64)
    elif method == "privacy_only":
        s = -np.asarray(P, dtype=np.float64)
    elif method == "soft":
        s = np.asarray(R, dtype=np.float64) - lam * np.asarray(P, dtype=np.float64)
    elif method == "hard":
        s = np.asarray(R, dtype=np.float64).copy()
        s[np.asarray(P) >= tau] = -np.inf
        k = min(k, int(np.isfinite(s).sum()))
    else:
        raise ValueError(method)
    if k <= 0:
        return np.zeros(0, dtype=int)
    # stable tie-break by index so results are deterministic
    order = np.lexsort((np.arange(n), -s))
    return np.sort(order[:k])


def keep_to_hex(keep, n=576):
    m = np.zeros(n, dtype=bool)
    m[np.asarray(keep, dtype=int)] = True
    return np.packbits(m).tobytes().hex()


def hex_to_keep(h, n=576):
    return np.flatnonzero(np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))[:n])
