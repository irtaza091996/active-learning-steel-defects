import numpy as np

from al_steel.data import split_indices


def test_split_is_stratified_disjoint_and_fixed():
    labels = np.repeat(np.arange(6), 240)
    tr, te = split_indices(labels)
    assert len(np.intersect1d(tr, te)) == 0
    assert len(tr) + len(te) == len(labels)
    assert np.all(np.bincount(labels[te]) == 48)
    tr2, te2 = split_indices(labels)
    assert np.array_equal(te, te2)
