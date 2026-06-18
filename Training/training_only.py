import os
import sys
import json
import time
import logging
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix
)

# Adjust these imports to your file names
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Config.training_config import training_config
from Dataset_loader.Loader import create_dataloaders
from PPI_Model.main_model import SiameseCrossTransformer, MODEL_CONFIGS

logger = logging.getLogger(__name__)


def format_time(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


class FocalLoss(nn.Module):
    """
    Focal loss for probability outputs.
    This version expects probabilities in [0,1], not raw logits.
    """
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, probs, targets):
        targets = targets.view_as(probs).float()
        probs = probs.clamp(min=self.eps, max=1.0 - self.eps)

        bce = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce
        return loss.mean()


class WeightedBCELoss(nn.Module):
    """
    Weighted BCE for probability outputs.
    """
    def __init__(self, pos_weight=10.0, eps=1e-8):
        super().__init__()
        self.pos_weight = pos_weight
        self.eps = eps

    def forward(self, probs, targets):
        targets = targets.view_as(probs).float()
        probs = probs.clamp(min=self.eps, max=1.0 - self.eps)

        loss = -(
            self.pos_weight * targets * torch.log(probs) +
            (1 - targets) * torch.log(1 - probs)
        )
        return loss.mean()


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warm-up and cosine decay."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = max(warmup_steps, 1)
        self.total_steps = max(total_steps, 1)
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            scale = self.step_count / self.warmup_steps
        else:
            denom = max(self.total_steps - self.warmup_steps, 1)
            progress = (self.step_count - self.warmup_steps) / denom
            progress = min(max(progress, 0.0), 1.0)
            scale = 0.5 * (1 + np.cos(np.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = max(base_lr * scale, self.min_lr)

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


class PPITrainer:
    def __init__(self, config):
        self.config = config

        requested_device = config.get("device", "cuda")
        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        self.output_dir = config.get("output_dir", "checkpoints")
        os.makedirs(self.output_dir, exist_ok=True)

        self._setup_logging()

        # Build model
        model_config = MODEL_CONFIGS[config.get("model_size", "medium")].copy()
        model_config["max_len"] = config.get("max_len", 900)
        self.model = SiameseCrossTransformer(model_config).to(self.device)
        logger.info(f"Model parameters: {self.model.count_parameters():,}")

        # Dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            splits_dir=config["splits_dir"],
            embedding_dir=config["embedding_dir"],
            batch_size=config.get("batch_size", 4),
            max_len=config.get("max_len", 900),
            num_workers=config.get("num_workers", 0),
            use_weighted_sampling=config.get("use_weighted_sampling", True),
        )

        # Loss
        loss_type = config.get("loss", "focal")
        if loss_type == "focal":
            self.criterion = FocalLoss(
                alpha=config.get("focal_alpha", 0.25),
                gamma=config.get("focal_gamma", 2.0)
            )
        elif loss_type == "bce_weighted":
            self.criterion = WeightedBCELoss(
                pos_weight=config.get("pos_weight", 10.0)
            )
        else:
            self.criterion = nn.BCELoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
            betas=(0.9, 0.999),
        )

        # Scheduler
        total_steps = len(self.train_loader) * config.get("epochs", 50)
        warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, warmup_steps, total_steps
        )

        # AMP
        self.use_amp = config.get("use_amp", True) and self.device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None

        # State
        self.best_val_f1 = 0.0
        self.best_val_auroc = 0.0
        self.patience_counter = 0
        self.patience = config.get("patience", 10)
        self.gradient_clip = config.get("gradient_clip", 1.0)
        self.history = defaultdict(list)

    def _setup_logging(self):
        log_file = os.path.join(self.output_dir, "training.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ]
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            emb_a = batch["emb_a"].to(self.device)
            emb_b = batch["emb_b"].to(self.device)
            mask_a = batch["mask_a"].to(self.device)
            mask_b = batch["mask_b"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast(device_type="cuda"):
                    probs = self.model(emb_a, emb_b, mask_a, mask_b).squeeze(-1)
                    loss = self.criterion(probs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                probs = self.model(emb_a, emb_b, mask_a, mask_b).squeeze(-1)
                loss = self.criterion(probs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()
            preds = probs.detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 100 == 0:
                lr = self.scheduler.get_lr()[0]

                elapsed = time.time() - epoch_start_time
                batches_done = batch_idx + 1
                total_batches = len(self.train_loader)

                time_per_batch = elapsed / batches_done
                remaining_batches = total_batches - batches_done
                eta = time_per_batch * remaining_batches

                logger.info(
                    f"  Epoch {epoch} | Batch {batches_done}/{total_batches} | "
                    f"Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                    f"Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}"
                )

        avg_loss = total_loss / max(len(self.train_loader), 1)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        pred_binary = (all_preds >= 0.5).astype(int)

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy_score(all_labels, pred_binary),
            "f1": f1_score(all_labels, pred_binary, zero_division=0),
            "auroc": roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0,
        }

        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader, prefix="Val"):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in dataloader:
            emb_a = batch["emb_a"].to(self.device)
            emb_b = batch["emb_b"].to(self.device)
            mask_a = batch["mask_a"].to(self.device)
            mask_b = batch["mask_b"].to(self.device)
            labels = batch["label"].to(self.device)

            if self.use_amp:
                with autocast(device_type="cuda"):
                    probs = self.model(emb_a, emb_b, mask_a, mask_b).squeeze(-1)
                    loss = self.criterion(probs, labels)
            else:
                probs = self.model(emb_a, emb_b, mask_a, mask_b).squeeze(-1)
                loss = self.criterion(probs, labels)

            total_loss += loss.item()
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(len(dataloader), 1)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        pred_binary = (all_preds >= 0.5).astype(int)

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy_score(all_labels, pred_binary),
            "precision": precision_score(all_labels, pred_binary, zero_division=0),
            "recall": recall_score(all_labels, pred_binary, zero_division=0),
            "f1": f1_score(all_labels, pred_binary, zero_division=0),
            "auroc": roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0,
            "auprc": average_precision_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0,
            "mcc": matthews_corrcoef(all_labels, pred_binary),
        }

        cm = confusion_matrix(all_labels, pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["tp"] = int(tp)
        metrics["fp"] = int(fp)
        metrics["tn"] = int(tn)
        metrics["fn"] = int(fn)

        logger.info(f"\n{prefix} Metrics:")
        logger.info(f"  Loss:        {metrics['loss']:.4f}")
        logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
        logger.info(f"  Precision:   {metrics['precision']:.4f}")
        logger.info(f"  Recall:      {metrics['recall']:.4f}")
        logger.info(f"  F1:          {metrics['f1']:.4f}")
        logger.info(f"  AUROC:       {metrics['auroc']:.4f}")
        logger.info(f"  AUPRC:       {metrics['auprc']:.4f}")
        logger.info(f"  MCC:         {metrics['mcc']:.4f}")
        logger.info(f"  Specificity: {metrics['specificity']:.4f}")
        logger.info(f"  Confusion:   TP={tp} FP={fp} TN={tn} FN={fn}")

        return metrics, all_preds, all_labels

    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }

        latest_path = os.path.join(self.output_dir, "checkpoint_latest.pt")
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(self.output_dir, "checkpoint_best.pt")
            torch.save(checkpoint, best_path)
            logger.info(
                f"  ★ New best model saved! (F1: {metrics['f1']:.4f}, AUROC: {metrics['auroc']:.4f})"
            )

    def train(self):
        epochs = self.config.get("epochs", 50)

        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info(f"  Model:      Siamese Cross-Transformer ({self.config.get('model_size', 'medium')})")
        logger.info(f"  Parameters: {self.model.count_parameters():,}")
        logger.info(f"  Epochs:     {epochs}")
        logger.info(f"  Batch size: {self.config.get('batch_size', 4)}")
        logger.info(f"  LR:         {self.config.get('learning_rate', 1e-4)}")
        logger.info(f"  Device:     {self.device}")
        logger.info("=" * 60)

        training_start_time = time.time()
        epoch_times = []

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics, _, _ = self.evaluate(self.val_loader, "Val")

            for key, value in train_metrics.items():
                self.history[f"train_{key}"].append(value)
            for key, value in val_metrics.items():
                self.history[f"val_{key}"].append(value)

            elapsed = time.time() - start_time
            epoch_times.append(elapsed)

            total_elapsed = time.time() - training_start_time
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = epochs - epoch
            total_eta = avg_epoch_time * remaining_epochs

            logger.info(
                f"\nEpoch {epoch}/{epochs} Finished | "
                f"Epoch Time: {format_time(elapsed)} | "
                f"Total Elapsed: {format_time(total_elapsed)} | "
                f"Training ETA: {format_time(total_eta)}\n"
                f"  Train Loss: {train_metrics['loss']:.4f} | "
                f"Train F1: {train_metrics['f1']:.4f} | "
                f"Train AUROC: {train_metrics['auroc']:.4f}\n"
                f"  Val Loss:   {val_metrics['loss']:.4f} | "
                f"Val F1:   {val_metrics['f1']:.4f} | "
                f"Val AUROC:   {val_metrics['auroc']:.4f}"
            )

            is_best = val_metrics["f1"] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics["f1"]
                self.best_val_auroc = val_metrics["auroc"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, val_metrics, is_best)

            if self.patience_counter >= self.patience:
                logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                break

        with open(os.path.join(self.output_dir, "training_history.json"), "w") as f:
            json.dump(
                {k: [float(v) for v in vals] for k, vals in self.history.items()},
                f,
                indent=2
            )

        total_training_time = time.time() - training_start_time
        logger.info("=" * 60)
        logger.info(f"Training finished in {format_time(total_training_time)}")
        logger.info("=" * 60)

        return self.history


# ====================================
# MAIN TRAINING CONFIGURATION
# ====================================
if __name__ == "__main__":
    trainer = PPITrainer(training_config)
    trainer.train()
