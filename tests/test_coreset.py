import numpy as np

from al_steel.coreset import k_center_greedy


def test_picks_one_point_per_far_cluster():
    rng = np.random.default_rng(0)
    centers = np.eye(4) * 10 + 1  # four well separated directions
    feats = np.concatenate([c + rng.normal(0, 0.01, (25, 4)) for c in centers])
    labelled = np.zeros(len(feats), bool)
    labelled[0] = True  # cluster 0 already covered
    picked = k_center_greedy(feats, labelled, 3)
    assert sorted(picked // 25) == [1, 2, 3]


def test_never_returns_labelled_or_duplicates():
    rng = np.random.default_rng(1)
    feats = rng.normal(size=(200, 8))
    labelled = rng.random(200) < 0.3
    picked = k_center_greedy(feats, labelled, 50)
    assert len(set(picked)) == 50
    assert not labelled[picked].any()
