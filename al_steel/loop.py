"""Pool-based active learning on frozen features with BaaL.

A small MLP head with dropout sits on top of fixed backbone features. BaaL's MCDropoutModule keeps
dropout active at inference, so `predict_on_dataset(..., iterations=T)` returns T stochastic forward
passes that the BALD and Entropy heuristics turn into uncertainty scores.
"""
import numpy as np
import torch
from baal.active import ActiveLearningDataset
from baal.active.heuristics import BALD, BatchBALD, Entropy
from baal.bayesian.dropout import MCDropoutModule
from baal.modelwrapper import ModelWrapper, TrainingArgs
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import TensorDataset

from .coreset import k_center_greedy

STRATEGIES = ["random", "entropy", "bald", "batchbald", "coreset"]


def make_head(in_dim, n_classes, hidden=256, p=0.5):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, n_classes))


def _wrapper(in_dim, n_classes, epochs, seed):
    torch.manual_seed(seed)
    model = MCDropoutModule(make_head(in_dim, n_classes))
    args = TrainingArgs(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
        criterion=nn.CrossEntropyLoss(), batch_size=32, epoch=epochs, workers=0, use_cuda=False,
        replicate_in_memory=False)  # loop over MC passes instead of copying each batch T times
    return ModelWrapper(model, args)


def mc_probs(wrapper, dataset, iterations):
    """(N, C, T) softmax probabilities from T MC-dropout passes."""
    logits = wrapper.predict_on_dataset(dataset, iterations=iterations, verbose=False)
    return torch.softmax(torch.as_tensor(logits), dim=1).numpy()


def run_active_learning(features, labels, train_idx, test_idx, strategy, seed,
                        init=12, query=12, rounds=10, mc_iter=20, epochs=60):
    """Run one active learning curve. Returns (rows, picks): per-round metrics and queried train indices."""
    assert strategy in STRATEGIES
    rng = np.random.default_rng(seed)
    mean, std = features[train_idx].mean(0), features[train_idx].std(0) + 1e-6
    x = torch.as_tensor((features - mean) / std, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    train_x = x[train_idx]
    train_set = TensorDataset(train_x, y[train_idx])
    test_set = TensorDataset(x[test_idx], y[test_idx])
    n_classes = int(labels.max()) + 1

    al = ActiveLearningDataset(train_set, random_state=seed)
    al.label_randomly(init)
    heuristic = {"bald": BALD(), "entropy": Entropy(), "batchbald": BatchBALD(num_samples=query)}.get(strategy)

    rows, picks = [], [np.flatnonzero(al.labelled).tolist()]
    for r in range(rounds + 1):
        wrapper = _wrapper(x.shape[1], n_classes, epochs, seed * 1000 + r)
        wrapper.train_on_dataset(al)
        pred = mc_probs(wrapper, test_set, mc_iter).mean(-1).argmax(1)
        true = labels[test_idx]
        rows.append({"strategy": strategy, "seed": seed, "round": r, "n_labelled": int(al.n_labelled),
                     "accuracy": accuracy_score(true, pred), "macro_f1": f1_score(true, pred, average="macro")})
        if r == rounds:
            break
        pool_to_train = np.flatnonzero(~al.labelled)  # pool position -> train index
        if strategy == "random":
            chosen = rng.choice(len(pool_to_train), query, replace=False)
        elif strategy == "coreset":
            chosen_train = k_center_greedy(train_x.numpy(), al.labelled.copy(), query)
            chosen = np.searchsorted(pool_to_train, chosen_train)
        else:
            ranks, _ = heuristic.get_ranks(mc_probs(wrapper, al.pool, mc_iter))
            chosen = np.asarray(ranks[:query])
        assert len(set(chosen.tolist())) == query, f"{strategy} returned duplicate picks"
        picks.append(pool_to_train[chosen].tolist())
        al.label(chosen.tolist())
    return rows, picks


def full_data_accuracy(features, labels, train_idx, test_idx, seed, mc_iter=20, epochs=60):
    """Upper bound: the same head trained on every training label."""
    rows, _ = run_active_learning(features, labels, train_idx, test_idx, "random", seed,
                                  init=len(train_idx), rounds=0, mc_iter=mc_iter, epochs=epochs)
    return rows[0]["accuracy"]
