# Regenerates the figures of the paper from the archived per-sample outputs and per-seed tables.
# Main comparison: PhoBERT 5-epoch run (logits archived in results/phobert_logits/).
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, f1_score

os.makedirs("figures_t05", exist_ok=True)
LAB = {"negative": 0, "neutral": 1, "positive": 2}
SEED = 42


def ece_and_bins(p, y, n_bins=10):
    """ECE with equal-width bins (first bin closed, others left-open), as in src/train_v40.py."""
    conf, acc = p.max(1), (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, pts = 0.0, []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf >= lo) & (conf <= hi) if b == 0 else (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
            pts.append((conf[m].mean(), acc[m].mean()))
    return ece, np.array(pts)


a = pd.read_csv(f"results/v40_probs/v40_test_seed_{SEED}.csv")
p = pd.read_csv(f"results/phobert_logits/phobert_test_seed_{SEED}.csv")
assert (a.row_index.values == p.row_index.values).all()
y = a.y_true.astype(str).str.lower().str.strip().map(LAB).values
PA = a[["p_negative", "p_neutral", "p_positive"]].values
PP = softmax(p[["logit_0", "logit_1", "logit_2"]].values, axis=1)
T = pd.read_csv("tables/per_seed_all_metrics_5ep.csv").set_index("seed").loc[SEED, "T"]
PT = softmax(p[["logit_0", "logit_1", "logit_2"]].values / T, axis=1)

# 1. Confusion matrices (seed 42)
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for ax, (name, P) in zip(axs, [("AERS-Pro", PA), ("PhoBERT (5 epochs)", PP)]):
    cm = confusion_matrix(y, P.argmax(1))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > 900 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(["Negative", "Neutral", "Positive"])
    ax.set_yticks(range(3)); ax.set_yticklabels(["Negative", "Neutral", "Positive"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{name} (seed {SEED}), F1-macro {f1_score(y, P.argmax(1), average='macro'):.4f}")
plt.tight_layout(); plt.savefig("figures_t05/fig_confusion_matrix.png", dpi=300); plt.close()

# 2. Paired F1 per seed (PhoBERT 5 epochs)
df = pd.read_csv("tables/per_seed_all_metrics_5ep.csv")
x, w = np.arange(len(df)), 0.35
plt.figure(figsize=(10, 5))
plt.bar(x - w / 2, df["f1_a"], w, label="AERS-Pro", color="tab:blue")
plt.bar(x + w / 2, df["f1_p"], w, label="PhoBERT (5 epochs)", color="tab:orange")
plt.xticks(x, df["seed"].astype(str), rotation=45)
plt.ylim(0.8, 0.92); plt.ylabel("F1-macro")
plt.title("Paired F1-macro over 10 seeds"); plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout(); plt.savefig("figures_t05/fig_f1_per_seed.png", dpi=300, bbox_inches="tight"); plt.close()

# 3. Reliability diagram (seed 42); ECE values are computed here, not typed in
plt.figure(figsize=(6, 5))
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
for name, P, mk, col in [("AERS-Pro", PA, "o", "blue"), ("PhoBERT raw", PP, "s", "red"), ("PhoBERT + TS", PT, "^", "green")]:
    e, pts = ece_and_bins(P, y)
    plt.plot(pts[:, 0], pts[:, 1], marker=mk, color=col, label=f"{name} (ECE {e:.3f})")
plt.xlabel("Mean confidence (max class probability)"); plt.ylabel("Accuracy in bin")
plt.title(f"Reliability diagram (seed {SEED}, 10 bins)"); plt.legend(loc="lower right"); plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout(); plt.savefig("figures_t05/fig_reliability_diagram.png", dpi=300); plt.close()
print("figures written to figures_t05/")
