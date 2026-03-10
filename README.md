# TIMIT_RFA_DFA

A research codebase for comparing **Backpropagation (BP)**, **Random Feedback Alignment (RFA)**, and **Direct Feedback Alignment (DFA)** on frame-level acoustic modeling with the **TIMIT** speech corpus.

---

## 1) What this project does

This repository implements an end-to-end experimental pipeline:

1. Export frame-level MFCC features and alignments from Kaldi outputs.
2. Validate frame/label consistency.
3. Convert features + labels into PyTorch tensors.
4. Optionally apply temporal frame splicing (context windows).
5. Train MLP acoustic models with:
   - Standard BP (autograd + SGD)
   - RFA (manual random feedback propagation)
   - DFA (manual direct feedback propagation)
6. Compare accuracy and runtime across 13-dim (mono) and 39-dim (tri1+delta) feature settings.

The main training script also evaluates both a **3-layer baseline** and a **larger 4-layer v2** network for each learning rule.

---

## 2) Repository layout

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── main.py
│   ├── train_bp.py
│   ├── train_bp_v2.py
│   ├── train_RFA.py
│   ├── train_RFA_v2.py
│   ├── train_DFA.py
│   ├── train_DFA_v2.py
│   ├── model/
│   │   ├── backprop.py
│   │   ├── backprop_v2.py
│   │   ├── RFA.py
│   │   ├── RFA_v2.py
│   │   ├── DFA.py
│   │   └── DFA_v2.py
│   ├── utils/
│   │   ├── process_timit.py
│   │   └── load_dataset.py
│   └── kaldi_scripts/
│       └── featureExtraction.sh
└── results/
    ├── results_comparison.txt
    ├── result.txt
    ├── result_with_batchnorm.txt
    ├── results_4_layers.txt
    ├── backprop_profile_results/
    ├── rfa_profile_results/
    └── weights/
```

---

## 3) Methods implemented

### Backpropagation (BP)
- Implemented in `src/train_bp.py` and `src/train_bp_v2.py`.
- Uses standard PyTorch autograd with `nn.CrossEntropyLoss` + SGD.

### Random Feedback Alignment (RFA)
- Implemented in `src/train_RFA.py` and `src/train_RFA_v2.py`.
- Uses fixed random feedback matrices for hidden-layer error propagation.
- Performs explicit manual weight updates inside `torch.no_grad()`.

### Direct Feedback Alignment (DFA)
- Implemented in `src/train_DFA.py` and `src/train_DFA_v2.py`.
- Uses direct random projections from output-layer error to hidden layers.
- Performs explicit manual updates for all trainable layers.

All three methods are compared on the same TIMIT-derived tensors and similar optimization settings (batch size 256, learning rate 1e-3 by default in training files).

---

## 4) Data and expected directory structure

By default, code expects a `data/` folder at the repository root. You can override this with `TIMIT_DATA_DIR`.

Expected structure:

```text
data/
├── feature_extracted/
│   └── export_feats/
│       ├── mfcc_mono.txt
│       ├── mfcc_tri1.txt
│       ├── labels_mono.txt
│       └── labels_tri1.txt
├── processed_13/
│   ├── X.pt
│   └── Y.pt
└── processed_39/
    ├── X.pt
    └── Y.pt
```

### Meaning of files
- `mfcc_mono.txt`: 13-dim CMVN MFCC frames (Kaldi text archive).
- `mfcc_tri1.txt`: MFCC + delta features (39 dims).
- `labels_*.txt`: frame-level PDF IDs (Kaldi `ali-to-pdf`).
- `X.pt`: tensor of shape `[N, D]`.
- `Y.pt`: tensor of shape `[N]`, integer class labels.

---

## 5) Kaldi feature/alignment export

Use the helper script:

```bash
cd src/kaldi_scripts
bash featureExtraction.sh
```

What it does:
- Applies CMVN to training features.
- Exports 13-dim MFCC (`mfcc_mono.txt`).
- Exports 39-dim features with deltas (`mfcc_tri1.txt`).
- Converts mono and tri1 alignments to PDF labels.

> Note: `featureExtraction.sh` assumes a functional Kaldi setup (`path.sh`, model/alignment dirs, and train data files).

---

## 6) Preprocessing and tensor generation

`src/utils/process_timit.py` provides:
- `load_mfcc(path)`
- `load_alignments(path)`
- `check_dataset()` to verify per-utterance frame/label count match
- `write_tensor(vector_size)` to write `X.pt`, `Y.pt`

`src/utils/load_dataset.py` provides:
- optional frame splicing with context window (`splice_data`)
- train/validation split and PyTorch `DataLoader` construction (`prep_dataset`)

Important behavior:
- If splicing is enabled, the spliced `X` is saved back to `processed_<vector_size>/X.pt`.
- Main script uses context size `5`, so effective input dimensions are:
  - 13-dim case: `13 × (2×5 + 1) = 143`
  - 39-dim case: `39 × (2×5 + 1) = 429`

---

## 7) Training entry point

Run everything from preprocessing to model comparison with:

```bash
python src/main.py
```

`main.py` workflow:
1. Process 13-dim features (`mfcc_mono.txt` + `labels_mono.txt`).
2. Process 39-dim features (`mfcc_tri1.txt` + `labels_tri1.txt`).
3. Train BP / RFA / DFA for 13-dim setup (3-layer + v2 4-layer variants).
4. Train BP / RFA / DFA for 39-dim setup (3-layer + v2 4-layer variants).
5. Write summary file to `results/results_comparison.txt`.

Default in `main.py` is currently `epochs = 5` (can be edited directly).

---

## 8) Model variants

### Baseline (3-layer hidden stack)
- BP: `src/model/backprop.py`
- RFA: `src/model/RFA.py`
- DFA: `src/model/DFA.py`
- Hidden width is 512 in the 3-layer RFA/DFA trainers.

### Larger v2 (4-layer hidden stack)
- BP: `src/model/backprop_v2.py`
- RFA: `src/model/RFA_v2.py`
- DFA: `src/model/DFA_v2.py`
- Hidden width is 1024 in v2 models.

---

## 9) TensorBoard logging

Training scripts instantiate `SummaryWriter()` and log:
- scalar accuracy values
- gradient/error histograms

After running training, inspect logs with:

```bash
tensorboard --logdir runs
```

---

## 10) Results included in this repository

The `results/` folder includes previously generated experiment artifacts, such as:
- method comparison summaries (`results_comparison.txt`, `results_4_layers.txt`)
- alternate run outputs (`result.txt`, `result_with_batchnorm.txt`)
- profiling outputs (`backprop_profile_results/`, `rfa_profile_results/`)
- saved weight statistics (`results/weights/`)

The profiling text files indicate example compute cost measurements (FLOPs) for BP and RFA steps.

---

## 11) Reproducibility notes and caveats

This is a research/prototyping codebase. Before using for publishable benchmarking, consider:

- **Random seeds** are not globally fixed in the current scripts.
- **Path assumptions** exist (Kaldi layout, data export locations).
- Some historical result files may come from different experiment settings.
- Splicing writes transformed tensors back to disk; avoid accidental double-processing.
- `requirements.txt` is currently empty; install dependencies manually.

Recommended core dependencies:
- Python 3.10+
- PyTorch
- tqdm
- tensorboard
- Kaldi tools (for feature/alignment export stage)

---

## 12) Quickstart checklist

1. Prepare Kaldi/TIMIT training data and alignments.
2. Export features + labels via `src/kaldi_scripts/featureExtraction.sh`.
3. Ensure files are in `data/feature_extracted/export_feats/` (or set `TIMIT_DATA_DIR`).
4. Run `python src/main.py`.
5. Read `results/results_comparison.txt`.
6. (Optional) inspect TensorBoard logs.

---

## 13) Suggested next improvements

- Add a fully pinned `requirements.txt`.
- Add CLI arguments for epochs, LR, batch size, and data directory.
- Save checkpoints and per-epoch metrics in structured JSON/CSV.
- Add deterministic seeding + experiment config files.
- Add unit tests for preprocessing and shape consistency.

---

If you want, I can also generate a **publication-style README** variant with experiment tables, equations for BP/RFA/DFA updates, and a reproducibility section tailored for a paper appendix.
