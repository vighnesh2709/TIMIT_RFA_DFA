# TIMIT_RFA_DFA

## Overview
This repository contains experimental code for training neural networks on the TIMIT speech corpus using three learning rules:
- Backpropagation (BP)
- Random Feedback Alignment (RFA)
- Direct Feedback Alignment (DFA)

The project covers MFCC preprocessing, frame-level alignments, tensorized datasets, and training pipelines for all three methods.

---

## Repository Structure

```text
.
├── src/
|   ├── kaldi_scipts
|   │   ├── FeatureExtraction.sh
│   ├── utils/
│   │   ├── process_timit.py
│   │   └── load_dataset.py
│   ├── model/
│   │   ├── backprop.py
│   │   ├── RFA.py
│   │   └── DFA.py
│   ├── train_bp.py
│   ├── train_RFA.py
│   ├── train_DFA.py
│   └── main.py
├── data/
│   ├── raw/
│   ├── exp/
│   ├── processed_13/
│   ├── processed_39/
│   └── processed_*
├── results/                   - Stores intermediate and flops results
├── requirements.txt
└── README.md

```

## Data Directory

The `data/` directory must exist at the repository root.

Contents:
- `raw/`  
  Original TIMIT corpus.
- `exp/`  
  Feature-extracted text files (MFCCs and frame-level alignments).
- `processed_13/`  
  Tensorized datasets for 13-dimensional MFCC features.
- `processed_39/`  
  Tensorized datasets for 39-dimensional MFCC features.
- `processed_*`  
  Tensorized datasets for additional feature configurations.

Each `processed_*` directory contains:
- `X.pt` — feature tensor of shape `[N, D]`
- `Y.pt` — label tensor of shape `[N]`

---

## Pipeline Summary
1. Load MFCC features and frame-level alignments.
2. Verify frame-to-label correspondence.
3. Convert features and labels into tensors.
4. Optionally apply temporal splicing.
5. Train models using BP, RFA, and DFA.
6. Record accuracy and runtime statistics.

---

## Source Files and Functions

### src/main.py
Purpose: Orchestrates preprocessing and training runs for 13- and 39-dimensional MFCC features using BP, RFA, and DFA.

Functions:
- `main()`  
  Loads MFCCs and alignments, checks dataset consistency, writes tensors, runs training pipelines for all methods, and writes a summary of results.

---

### src/train_bp.py
Purpose: Train an MLP using standard backpropagation.

Functions:
- `train_bp(train_loader, val_loader, num_feats, num_pdfs)`  
  Trains a backpropagation-based MLP for 100 epochs using SGD and returns maximum train and validation accuracy.

---

### src/train_RFA.py
Purpose: Train an MLP using Random Feedback Alignment.

Functions:
- `train_rfa(X, Y, num_feats, num_pdfs)`  
  Splits tensors into train/validation sets, applies fixed random feedback matrices, manually updates weights, and returns maximum train and validation accuracy.

---

### src/train_DFA.py
Purpose: Train an MLP using Direct Feedback Alignment.

Functions:
- `train_dfa(X, Y, num_feats, num_pdfs)`  
  Splits tensors into train/validation sets, applies direct random feedback from output to hidden layers, manually updates weights, and returns maximum train and validation accuracy.

---

### src/model/backprop.py
Purpose: Defines the MLP architecture used for backpropagation.

Classes:
- `MLP`  
  Three-layer fully connected network with ReLU activations.

Methods:
- `MLP.forward(x)`  
  Returns output logits.

---

### src/model/RFA.py
Purpose: Defines the MLP architecture used for Random Feedback Alignment.

Classes:
- `RFA_MLP`  
  Three-layer MLP exposing intermediate activations for manual RFA updates.

Methods:
- `RFA_MLP.forward(x)`  
  Returns `(a1, h1, a2, h2, a3)`.

---

### src/model/DFA.py
Purpose: Defines the MLP architecture used for Direct Feedback Alignment.

Classes:
- `DFA_MLP`  
  Three-layer MLP exposing intermediate activations for manual DFA updates.

Methods:
- `DFA_MLP.forward(x)`  
  Returns `(a1, h1, a2, h2, a3)`.

---

### src/utils/load_dataset.py
Purpose: Load tensor datasets and optionally apply temporal frame splicing.

Functions:
- `splice_data(X, context)`  
  Concatenates context frames around each time step.
- `prep_dataset(vector_size, context=5, splicing=False)`  
  Loads `X.pt` and `Y.pt`, optionally applies splicing, splits into train/validation sets, and returns PyTorch DataLoaders.

---

### src/utils/process_timit.py
Purpose: Parse Kaldi-style MFCC and alignment exports, verify consistency, and write tensors.

Functions:
- `load_alignments(path)`  
  Loads frame-level alignment labels keyed by utterance ID.
- `load_mfcc(path)`  
  Loads MFCC features keyed by utterance ID.
- `check_dataset()`  
  Verifies frame-to-label correspondence.
- `write_tensor(vector_size)`  
  Writes concatenated feature and label tensors to `data/processed_<vector_size>/`.

---

### src/kaldi_scripts/featureExtraction.sh
Purpose: Kaldi helper script to export CMVN-normalized MFCCs, delta features, and alignments to text files for downstream preprocessing.

---

## Notes
- File paths in the code assume a fixed local filesystem layout.
- The code is intended for research experiments, not production use.
- Reproducibility is not enforced unless random seeds are explicitly set.

