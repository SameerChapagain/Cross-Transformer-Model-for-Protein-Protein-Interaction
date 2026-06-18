import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix
)

BASE_DIR = os.getcwd()
sys.path.append(BASE_DIR)

from Config.model_config import MODEL_CONFIGS
from Dataset_loader.Loader import create_dataloaders
from PPI_Model.main_model import SiameseCrossTransformer


@torch.no_grad()
def test_model(model, test_loader, device):
    model.eval()

    all_probs = []
    all_labels = []

    for batch in test_loader:
        emb_a = batch["emb_a"].to(device, non_blocking=True)
        emb_b = batch["emb_b"].to(device, non_blocking=True)
        mask_a = batch["mask_a"].to(device, non_blocking=True)
        mask_b = batch["mask_b"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(emb_a, emb_b, mask_a, mask_b).squeeze(-1)
            probs = torch.sigmoid(logits)

        all_probs.extend(probs.cpu().float().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auroc": roc_auc_score(all_labels, all_probs),
        "auprc": average_precision_score(all_labels, all_probs),
        "mcc": matthews_corrcoef(all_labels, all_preds),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    return metrics, all_probs, all_labels


if __name__ == "__main__":
    splits_dir = r"/kaggle/input/datasets/sameerchapagain/pewwnew/4_Dataset_splits/Random splits"
    embedding_dir = r"/kaggle/input/datasets/sameerchapagain/ppidata/PPI/PPI/3_ESM2_Embeddings_generation/esm2_embs_3B"
    checkpoint_path = "/kaggle/input/datasets/sameerchapagain/ckptss/ppi_checkpoints/epoch_007.pt"

    model_size = "medium"
    max_len = 900
    batch_size = 16

    output_dir = "/kaggle/working/test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = create_dataloaders(
        splits_dir=splits_dir,
        embedding_dir=embedding_dir,
        batch_size=batch_size,
        max_len=max_len,
        num_workers=0,
        use_weighted_sampling=False,
    )

    model_config = MODEL_CONFIGS[model_size].copy()
    model_config["max_len"] = max_len

    # Important:
    # Friend's code applies sigmoid outside the model.
    # So model should output raw logits.
    model_config["use_sigmoid_output"] = False

    model = SiameseCrossTransformer(model_config).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False
    )

    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")

    test_metrics, test_probs, test_labels = test_model(
        model=model,
        test_loader=test_loader,
        device=device
    )

    np.savez(
        os.path.join(output_dir, "test_predictions_01.npz"),
        predictions=test_probs,
        labels=test_labels,
    )

    print("\nFINAL TEST RESULTS")
    print("=" * 60)
    for metric, value in test_metrics.items():
        if isinstance(value, float):
            print(f"{metric:15s}: {value:.4f}")
        else:
            print(f"{metric:15s}: {value}")