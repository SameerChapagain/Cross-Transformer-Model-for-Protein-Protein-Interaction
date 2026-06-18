import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve
)

npz_path = r"D:\Protein_Protein_Interaction\test_predictions_01.npz"
output_dir = r"D:\Protein_Protein_Interaction\Testing\TEST RESULTS"
os.makedirs(output_dir, exist_ok=True)

data = np.load(npz_path)
probs = data["predictions"].reshape(-1)
labels = data["labels"].reshape(-1).astype(int)

threshold = 0.5
preds = (probs >= threshold).astype(int)

acc = accuracy_score(labels, preds)
precision = precision_score(labels, preds, zero_division=0)
recall = recall_score(labels, preds, zero_division=0)
f1 = f1_score(labels, preds, zero_division=0)
auroc = roc_auc_score(labels, probs)
auprc = average_precision_score(labels, probs)
mcc = matthews_corrcoef(labels, preds)

cm = confusion_matrix(labels, preds, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

thresholds = np.linspace(0, 1, 101)
mcc_scores = [matthews_corrcoef(labels, (probs >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(mcc_scores)]
best_mcc = max(mcc_scores)

# 1. Confusion Matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Pred 0", "Pred 1"])
plt.yticks([0, 1], ["True 0", "True 1"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
plt.show()

# 2. ROC Curve
fpr, tpr, _ = roc_curve(labels, probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
plt.show()

# 3. Precision-Recall Curve
prec, rec, _ = precision_recall_curve(labels, probs)
plt.figure(figsize=(6, 5))
plt.plot(rec, prec, label=f"AUPRC = {auprc:.4f}")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150)
plt.show()

# 4. Probability Distribution
plt.figure(figsize=(6, 5))
plt.hist(probs[labels == 0], bins=50, alpha=0.6, density=True, label="Negative (0)")
plt.hist(probs[labels == 1], bins=50, alpha=0.6, density=True, label="Positive (1)")
plt.title("Prediction Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "probability_distribution.png"), dpi=150)
plt.show()

# 5. Metrics Bar Chart
metrics = {
    "AUROC": auroc,
    "AUPRC": auprc,
    "Accuracy": acc,
    "F1": f1,
    "Sensitivity": sensitivity,
    "Specificity": specificity,
    "Precision": precision,
    "MCC(scaled)": (mcc + 1) / 2,
}

plt.figure(figsize=(9, 5))
bars = plt.bar(metrics.keys(), metrics.values())
plt.ylim(0, 1.1)
plt.title("All Metrics Summary")
plt.ylabel("Score")
plt.xticks(rotation=30)
for bar, val in zip(bars, metrics.values()):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha="center")
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "metrics_summary.png"), dpi=150)
plt.show()

# 6. Normalized Confusion Matrix
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6, 5))
plt.imshow(cm_norm, cmap="Blues")
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Pred 0", "Pred 1"])
plt.yticks([0, 1], ["True 0", "True 1"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm_norm[i, j]:.3f}", ha="center", va="center", fontsize=12)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "normalized_confusion_matrix.png"), dpi=150)
plt.show()

# 7. MCC vs Threshold
plt.figure(figsize=(6, 5))
plt.plot(thresholds, mcc_scores)
plt.axvline(best_threshold, linestyle="--", label=f"Best threshold = {best_threshold:.2f}")
plt.axhline(best_mcc, linestyle="--", label=f"Best MCC = {best_mcc:.4f}")
plt.axvline(0.5, linestyle=":", label="Default threshold = 0.5")
plt.title("MCC vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("MCC")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mcc_vs_threshold.png"), dpi=150)
plt.show()