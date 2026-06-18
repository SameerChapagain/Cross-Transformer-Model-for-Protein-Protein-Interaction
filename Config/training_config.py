training_config = {
        # Data
        "splits_dir": r"/kaggle/input/datasets/yashodajoshi12/ppi-data/PPI/4_Dataset_splits/Clustured_splits",
        "embedding_dir": r"/kaggle/input/datasets/yashodajoshi12/ppi-data/PPI/3_ESM2_Embeddings_generation/esm2_embs_3B",

        # Model
        "model_size": "small",
        "max_len": 900,

        # Training
        "epochs": 15,
        "batch_size": 12,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "gradient_clip": 1.0,

        # Loss
        "loss": "bce",          # "focal", "bce_weighted", or "bce"
        "focal_alpha": 0.75,
        "focal_gamma": 2.0,
        "pos_weight": 10.0,

        # Sampling
        "use_weighted_sampling": True,

        # Regularization
        "patience": 10,
        "use_amp": True,

        # System
        "device": "cuda",         # auto-falls back to cpu if cuda unavailable
        "num_workers": 4,
        "output_dir": "checkpoints/run_001",
    }
