"""Core-Set selection (Sener & Savarese, 2018) via greedy k-center on a feature space."""
import numpy as np


def k_center_greedy(features, labelled_mask, k):
    """Pick k unlabelled points that greedily minimise the max distance to the labelled set.

    features: (N, D) array, labelled_mask: (N,) bool. Returns k indices into `features`.
    """
    assert (~labelled_mask).sum() >= k, "not enough unlabelled points"
    feats = features / np.linalg.norm(features, axis=1, keepdims=True)
    if labelled_mask.any():
        min_dist = _min_dist(feats, feats[labelled_mask])
    else:
        min_dist = np.full(len(feats), np.inf)
    min_dist[labelled_mask] = -1.0
    picked = []
    for _ in range(k):
        i = int(np.argmax(min_dist))
        picked.append(i)
        min_dist = np.minimum(min_dist, _min_dist(feats, feats[i:i + 1]))
        min_dist[i] = -1.0
        min_dist[labelled_mask] = -1.0
    return np.array(picked)


def _min_dist(a, b):
    d = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.clip(d, 0, None)).min(1)
