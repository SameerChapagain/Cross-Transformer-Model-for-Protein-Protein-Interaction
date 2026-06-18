# Protein-Protein Interaction Prediction using Deep Learning

A sequence-based deep learning framework for predicting **protein-protein interactions (PPIs)** using **ESM-2 protein language model embeddings** and a **Siamese Cross-Transformer architecture**.

This project aims to predict whether two proteins interact by learning both individual protein sequence features and inter-protein residue-level relationships through cross-attention.

---

## Project Overview

Protein-protein interactions are essential for cellular signaling, gene regulation, disease mechanisms, drug discovery, and systems biology. However, experimental PPI detection methods such as yeast two-hybrid screening, co-immunoprecipitation, and affinity purification-mass spectrometry are often expensive, time-consuming, and incomplete.

This project develops a computational PPI prediction framework that uses:

- **ESM-2 embeddings** to represent protein sequences at residue level
- **Siamese Cross-Transformer architecture** to model relationships between two proteins
- **Cross-attention mechanism** to capture interaction-relevant residue dependencies
- **Binary classification** to predict interacting and non-interacting protein pairs

---

## Key Features

- Sequence-based PPI prediction
- Uses residue-level **ESM-2 3B embeddings**
- Siamese architecture with shared input projection
- Symmetric cross-attention between protein pairs
- Attention-based pooling for protein-level representation
- Handles variable-length protein sequences using padding and attention masks
- Supports imbalanced dataset training using weighted sampling
- Evaluated using AUROC, AUPRC, MCC, F1-score, precision, recall, sensitivity, and specificity

---

## Model Architecture

The proposed model follows a Siamese Cross-Transformer design:

1. Protein sequences are converted into ESM-2 embeddings.
2. Input embeddings are projected from 2560 dimensions to a lower hidden dimension.
3. Self-attention captures intra-protein sequence features.
4. Cross-attention captures residue-level relationships between two proteins.
5. Attention pooling converts residue-level features into protein-level vectors.
6. Feature fusion is performed using:
   - Protein A representation
   - Protein B representation
   - Absolute difference
   - Element-wise product
7. A classifier predicts the final interaction probability.

---

## Dataset

The dataset was prepared using high-confidence human protein-protein interaction data.

- **Positive interactions:** collected from STRING database
- **Protein sequences:** retrieved from UniProt
- **Species:** Homo sapiens
- **Filtering:** high-confidence interactions with score greater than 0.9
- **Redundancy reduction:** MMseqs2
- **Negative sampling:** sequence-based negative pair generation
- **Train/validation/test split:** 80/10/10

---

## Methodology

The overall workflow includes:

1. Collect STRING PPI records and UniProt FASTA sequences
2. Filter human protein pairs with high confidence scores
3. Remove incomplete, duplicate, and redundant entries
4. Generate residue-level ESM-2 embeddings
5. Prepare train, validation, and test splits
6. Load embeddings with truncation, padding, and attention masks
7. Train Siamese Cross-Transformer model
8. Evaluate on unseen test protein pairs
9. Generate metrics and visualization plots

---

## Results

The model was evaluated on an independent test set of unseen protein pairs.

| Metric | Score |
|---|---:|
| AUROC | 0.965 |
| AUPRC | 0.866 |
| Accuracy | 0.959 |
| F1-Score | 0.791 |
| Sensitivity / Recall | 0.855 |
| Specificity | 0.969 |
| Precision | 0.736 |
| MCC | 0.7856 |

The model showed strong ability to distinguish interacting and non-interacting protein pairs, even under class imbalance.

---

## Technologies Used

- Python
- PyTorch
- ESM-2
- MMseqs2
- scikit-learn
- NumPy
- pandas
- matplotlib
- seaborn
- Google Colab Pro
- Kaggle GPU environment

---

## Repository Structure

```text
PPI-Prediction/
│
├── Config/
│   ├── model_config.py
│   └── training_config.py
│
├── Dataset_loader/
│   └── Loader.py
│
├── PPI_Model/
│   └── main_model.py
│
├── scripts/
│   ├── dataset_creation.py
│   ├── generate_esm_embeddings.py
│   ├── split_data.py
│   ├── train.py
│   ├── test.py
│   └── plot_results.py
│
├── results/
│   ├── metrics/
│   └── plots/
│
├── checkpoints/
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/SameerChapagain/Cross-Transformer-Model-for-Protein-Protein-Interaction
cd your-repository-name
```

Install required dependencies:

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn biopython tqdm
```

Install ESM:

```bash
pip install fair-esm
```

---

## Usage

### 1. Prepare Dataset

```bash
python scripts/dataset_creation.py
```

### 2. Generate ESM-2 Embeddings

```bash
python scripts/generate_esm_embeddings.py
```

### 3. Split Dataset

```bash
python scripts/split_data.py
```

### 4. Train Model

```bash
python scripts/train.py
```

### 5. Test Model

```bash
python scripts/test.py
```

### 6. Plot Results

```bash
python scripts/plot_results.py
```

---

## Future Work

Future improvements include:

- Using `pos_weight` in BCEWithLogitsLoss to reduce false-negative predictions
- Extracting and visualizing cross-attention maps for residue-level interpretability
- Testing the model on virus-host PPI datasets
- Applying the framework to South Asian or region-specific protein datasets when available
- Developing a simple web-based PPI prediction tool

---

## Applications

This framework can support:

- Drug discovery
- Biomarker identification
- Disease pathway analysis
- Protein function prediction
- Systems biology
- Pathogen-host interaction mapping

---

## Authors

Final Year Biomedical Engineering Project

- Pawana Gyawali
- Pranisha Bhattarai
- Pushpa Thapa
- Sachin Subedi
- Sameer Chapagain
- Yashasvi Bashyal

Supervised by:

- Asst. Prof. Dipesh Sapkota
- Assoc. Prof. Rishi Baniya

---

## License

This project is intended for academic and research purposes.

---

## Acknowledgement

This work was completed as a final-year Biomedical Engineering project under Purbanchal University, Faculty of Engineering, National Institute of Engineering and Technology, Lalitpur, Nepal.
