"""
PyTorch Dataset for PPI pairs with ESM-2 per-residue embeddings.
Handles variable-length sequences with padding and attention masks.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PPIDataset(Dataset):
    """
    Loads protein pairs and their ESM-2 embeddings.
    
    Each sample returns:
        - emb_A: [max_len, embed_dim] padded embedding for protein A
        - emb_B: [max_len, embed_dim] padded embedding for protein B
        - mask_A: [max_len] boolean attention mask for protein A
        - mask_B: [max_len] boolean attention mask for protein B
        - label: 0 or 1
    """
    
    def __init__(self, 
                 pairs_file,
                 embedding_dir,
                 max_len=900,           # Max residues to keep (truncate longer)
                 esm_layer=36,          # Which ESM layer to use (last = 36 for 3B)
                 cache_embeddings=True, # Cache in RAM for speed
                 embedding_dim=2560):   # ESM-2 3B embedding dimension
        
        self.embedding_dir = Path(embedding_dir)
        self.max_len = max_len
        self.esm_layer = esm_layer
        self.embedding_dim = embedding_dim
        self.cache_embeddings = cache_embeddings
        self._cache = {}
        
        # Load pairs
        self.pairs = pd.read_csv(pairs_file, sep="\t", header=None,
                                 names=["protein1", "protein2", "label"])
        
        # Verify all embeddings exist
        self._verify_and_filter()
        
        logger.info(f"Dataset loaded: {len(self.pairs)} pairs, "
                    f"max_len={max_len}, embed_dim={embedding_dim}")
    
    def _verify_and_filter(self):
        """Remove pairs where embeddings are missing."""
        valid_mask = self.pairs.apply(
            lambda row: (self.embedding_dir / f"{row['protein1']}.pt").exists() and
                        (self.embedding_dir / f"{row['protein2']}.pt").exists(),
            axis=1
        )
        n_before = len(self.pairs)
        self.pairs = self.pairs[valid_mask].reset_index(drop=True)
        n_removed = n_before - len(self.pairs)
        if n_removed > 0:
            logger.warning(f"Removed {n_removed} pairs due to missing embeddings")
    
    def _load_embedding(self, protein_id):
        """Load and process a single protein's ESM-2 embedding."""
        if self.cache_embeddings and protein_id in self._cache:
            return self._cache[protein_id]
        
        emb_path = self.embedding_dir / f"{protein_id}.pt"
        data = torch.load(emb_path, map_location="cpu")
        
        # Extract the specific layer's representation
        # Shape: [seq_len, embed_dim]
        emb = data["representations"][self.esm_layer]
        
        # Truncate if necessary
        if emb.shape[0] > self.max_len:
            emb = emb[:self.max_len]
        
        if self.cache_embeddings:
            self._cache[protein_id] = emb
        
        return emb
    
    def _pad_embedding(self, emb):
        """Pad embedding to max_len and create attention mask."""
        seq_len = emb.shape[0]
        
        if seq_len >= self.max_len:
            # Already truncated in _load_embedding
            padded = emb[:self.max_len]
            mask = torch.ones(self.max_len, dtype=torch.bool)
        else:
            # Pad with zeros
            padding = torch.zeros(self.max_len - seq_len, self.embedding_dim)
            padded = torch.cat([emb, padding], dim=0)
            mask = torch.zeros(self.max_len, dtype=torch.bool)
            mask[:seq_len] = True
        
        return padded, mask
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        
        # Load embeddings
        emb_a = self._load_embedding(row["protein1"])
        emb_b = self._load_embedding(row["protein2"])
        
        # Pad and get masks
        emb_a, mask_a = self._pad_embedding(emb_a)
        emb_b, mask_b = self._pad_embedding(emb_b)
        
        label = torch.tensor(row["label"], dtype=torch.float32)
        
        return {
            "emb_a": emb_a,       # [max_len, embed_dim]
            "emb_b": emb_b,       # [max_len, embed_dim]
            "mask_a": mask_a,     # [max_len]
            "mask_b": mask_b,     # [max_len]
            "label": label,       # scalar
        }
    
    def get_class_weights(self):
        """Compute class weights for imbalanced dataset."""
        labels = self.pairs["label"].values
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        weight_pos = n_neg / (n_pos + n_neg)
        weight_neg = n_pos / (n_pos + n_neg)
        return torch.tensor([weight_neg, weight_pos], dtype=torch.float32)
    
    def get_sample_weights(self):
        """Per-sample weights for WeightedRandomSampler."""
        labels = self.pairs["label"].values
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        weights = np.where(labels == 1, 1.0 / n_pos, 1.0 / n_neg)
        return torch.tensor(weights, dtype=torch.float64)


def create_dataloaders(splits_dir, embedding_dir, batch_size=4, max_len=900, 
                       num_workers=0, use_weighted_sampling=True):
    """Create train/val/test dataloaders with proper sampling."""
    
    train_dataset = PPIDataset(
        os.path.join(splits_dir, "train_pairs.tsv"),
        embedding_dir, max_len=max_len, cache_embeddings=False  # Too large for RAM
    )
    val_dataset = PPIDataset(
        os.path.join(splits_dir, "val_pairs.tsv"),
        embedding_dir, max_len=max_len, cache_embeddings=False
    )
    test_dataset = PPIDataset(
        os.path.join(splits_dir, "test_pairs.tsv"),
        embedding_dir, max_len=max_len, cache_embeddings=False
    )
    
    # Weighted sampling for training to handle class imbalance
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders(
        splits_dir=r"/kaggle/input/datasets/sameerchapagain/ppidata/PPI/PPI/4_Dataset_splits/Random splits",
        embedding_dir=r"/kaggle/input/datasets/sameerchapagain/ppidata/PPI/PPI/3_ESM2_Embeddings_generation/esm2_embs_3B",
        batch_size=12,
        max_len=900,
        num_workers=4,
        use_weighted_sampling=True
    )

    batch = next(iter(train_loader))
    print("emb_a shape:", batch["emb_a"].shape)
    print("emb_b shape:", batch["emb_b"].shape)
    print("mask_a shape:", batch["mask_a"].shape)
    print("mask_b shape:", batch["mask_b"].shape)
    print("label shape:", batch["label"].shape)
