"""Turn results/*.csv into figures and a markdown summary table."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from al_steel.data import CLASSES, split_indices  # noqa: E402

RES = ROOT / "results"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Categorical slots in fixed order (validated: scripts in the dataviz palette, light surface).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
STYLE = {  # random is the baseline, so it gets neutral ink instead of a hue
    "random": dict(color=INK2, marker="o", ls="--", label="Random"),
    "entropy": dict(color=SLOTS[0], marker="s", ls="-", label="Entropy"),
    "bald": dict(color=SLOTS[1], marker="^", ls="-", label="BALD (top-k)"),
    "batchbald": dict(color=SLOTS[2], marker="D", ls="-", label="BatchBALD"),
    "coreset": dict(color=SLOTS[3], marker="v", ls="-", label="Core-Set"),
}
N_TRAIN = 1152  # 80% of the 1,440 images, see al_steel.data.split_indices
TITLES = {"dinov2_vits14": "DINOv2 ViT-S/14 (self-supervised)", "resnet18": "ResNet-18 (ImageNet supervised)"}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def learning_curves(curves, full):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, facecolor=SURFACE)
    for ax, bb in zip(axes, TITLES):
        style_axes(ax)
        sub = curves[curves.backbone == bb]
        for s, st in STYLE.items():
            g = sub[sub.strategy == s].groupby("n_labelled").accuracy
            m, sd = g.mean(), g.std()
            ax.fill_between(m.index, m - sd, m + sd, color=st["color"], alpha=0.12, lw=0)
            ax.plot(m.index, m.values, color=st["color"], ls=st["ls"], lw=2, marker=st["marker"], ms=5,
                    label=st["label"])
        ub = full[full.backbone == bb].accuracy.mean()
        ax.axhline(ub, color=INK, lw=1, ls=":")
        ax.text(sub.n_labelled.max(), ub + 0.006, f"all {N_TRAIN} labels: {ub:.1%}",
                ha="right", va="bottom", fontsize=8.5, color=INK2)
        ax.set_title(TITLES[bb], fontsize=11, color=INK, loc="left")
        ax.set_xlabel("Labelled images", color=INK2, fontsize=10)
        ax.set_ylim(0.35, 1.03)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0].set_ylabel("Test accuracy (mean ± 1 sd, 5 seeds)", color=INK2, fontsize=10)
    axes[1].legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Active learning on NEU-DET steel surface defects", x=0.06, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(RES / "learning_curves.png", dpi=160, facecolor=SURFACE)


def tsne_picks(picks, n_rounds=3):
    d = np.load(ROOT / "features" / "dinov2_vits14.npz")
    feats, labels = d["features"], d["labels"]
    train_idx, _ = split_indices(labels)
    emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(feats[train_idx])
    pos = {g: i for i, g in enumerate(train_idx)}
    strategies = ["random", "bald", "batchbald", "coreset"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), facecolor=SURFACE)
    for ax, s in zip(axes, strategies):
        ax.set_facecolor(SURFACE)
        ax.set_xticks([]), ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(GRID)
        for c, name in enumerate(CLASSES):
            m = labels[train_idx] == c
            ax.scatter(emb[m, 0], emb[m, 1], s=6, color=SLOTS[c], alpha=0.35, lw=0, label=name)
        chosen = [pos[g] for p in picks[f"dinov2_vits14/{s}"][:n_rounds + 1] for g in p]
        ax.scatter(emb[chosen, 0], emb[chosen, 1], s=60, c=[SLOTS[labels[train_idx][i]] for i in chosen],
                   edgecolors=INK, linewidths=1.2, zorder=3)
        covered = len(set(labels[train_idx][chosen]))
        ax.set_title(f"{STYLE[s]['label']}  ·  {covered}/6 classes", fontsize=10.5, color=INK, loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", color=SLOTS[c], ms=7, label=n) for c, n in enumerate(CLASSES)]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=8, label="labelled"))
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(f"First {len(chosen)} labelled images (seed 0) on a t-SNE of DINOv2 features",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(RES / "tsne_picks.png", dpi=160, facecolor=SURFACE)


def labels_to_reach(m, target):
    hit = m[m >= target]
    return int(hit.index.min()) if len(hit) else None


def summary(curves, full):
    rows = []
    for bb in TITLES:
        for s in STYLE:
            g = curves[(curves.backbone == bb) & (curves.strategy == s)].groupby("n_labelled").accuracy
            m, sd = g.mean(), g.std()
            rows.append({"backbone": bb, "strategy": STYLE[s]["label"],
                         "acc@24": f"{m[24]:.1%} ± {sd[24]:.1%}", "acc@48": f"{m[48]:.1%} ± {sd[48]:.1%}",
                         f"acc@{m.index.max()}": f"{m.iloc[-1]:.1%} ± {sd.iloc[-1]:.1%}",
                         "labels to 95%": labels_to_reach(m, 0.95) or "not reached"})
        rows.append({"backbone": bb, "strategy": "All labels (upper bound)",
                     f"acc@{m.index.max()}": f"{full[full.backbone == bb].accuracy.mean():.1%}"})
    table = pd.DataFrame(rows).fillna("")
    table.to_csv(RES / "summary.csv", index=False)
    (RES / "summary.md").write_text(table.to_markdown(index=False))
    print(table.to_markdown(index=False))


if __name__ == "__main__":
    curves = pd.read_csv(RES / "curves.csv")
    full = pd.read_csv(RES / "full_data.csv")
    picks = json.loads((RES / "picks_seed0.json").read_text())["picks"]
    learning_curves(curves, full)
    tsne_picks(picks)
    summary(curves, full)
