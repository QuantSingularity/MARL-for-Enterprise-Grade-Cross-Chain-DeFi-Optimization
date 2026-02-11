# File Manifest

This document describes all files in the MARL_CrossChain_DeFi_Proposal package.

## Top-Level Files

| File                     | Purpose                                              |
| ------------------------ | ---------------------------------------------------- |
| `DELIVERY_NOTE.txt`      | Summary of deliverables and quick start instructions |
| `artifact_manifest.json` | File checksums and metadata                          |
| `LICENSE`                | License information (MIT for code, CC-BY for text)   |

## Proposal Directory (`proposal/`)

### Source Files

| File             | Purpose                                   | Lines |
| ---------------- | ----------------------------------------- | ----- |
| `main.tex`       | Main LaTeX proposal document (6-10 pages) | ~600  |
| `references.bib` | BibTeX bibliography (40+ citations)       | ~350  |
| `Makefile`       | Build automation for PDF generation       | ~50   |
| `README.md`      | Instructions for building the proposal    | ~50   |

### Generated Files

| File       | Purpose               |
| ---------- | --------------------- |
| `main.pdf` | Compiled proposal PDF |
| `main.aux` | LaTeX auxiliary file  |
| `main.log` | LaTeX log file        |
| `main.toc` | Table of contents     |

### Figures Directory (`proposal/figs/`)

TikZ figures are generated inline in the LaTeX source. No external figure files required.

## Code Directory (`code/`)

### Core Implementation (`src/`)

#### Environment (`src/envs/`)

| File                 | Purpose                     | Lines |
| -------------------- | --------------------------- | ----- |
| `__init__.py`        | Package initialization      | ~10   |
| `cross_chain_env.py` | Main environment simulator  | ~500  |
| `demo_env.py`        | Demo script for environment | ~150  |

#### Agents (`src/agents/`)

| File               | Purpose                       | Lines |
| ------------------ | ----------------------------- | ----- |
| `__init__.py`      | Package initialization        | ~15   |
| `qmix.py`          | QMIX agent implementation     | ~400  |
| `mappo.py`         | MAPPO agent implementation    | ~350  |
| `baselines.py`     | Baseline agents (Random, IQL) | ~250  |
| `communication.py` | Attention-based communication | ~250  |
| `gnn_encoder.py`   | GNN state encoder             | ~350  |

#### Training (`src/train/`)

| File                 | Purpose              | Lines |
| -------------------- | -------------------- | ----- |
| `train_synthetic.py` | Main training script | ~450  |

#### Evaluation (`src/eval/`)

| File               | Purpose                         | Lines |
| ------------------ | ------------------------------- | ----- |
| `evaluate_demo.py` | Evaluation script with plotting | ~350  |

#### Data (`src/data/`)

| File                    | Purpose                  | Lines |
| ----------------------- | ------------------------ | ----- |
| `generate_synthetic.py` | Synthetic data generator | ~400  |

### Notebooks (`notebooks/`)

| File                        | Purpose                   |
| --------------------------- | ------------------------- |
| `01_environment_demo.ipynb` | Environment demonstration |
| `02_training_demo.ipynb`    | Training walkthrough      |

### Tests (`tests/`)

| File             | Purpose                | Lines |
| ---------------- | ---------------------- | ----- |
| `test_env.py`    | Environment unit tests | ~200  |
| `test_agents.py` | Agent unit tests       | ~250  |

### Configuration (`configs/`)

| File        | Purpose            |
| ----------- | ------------------ |
| `demo.yaml` | Demo configuration |

### Root Code Files

| File               | Purpose                       |
| ------------------ | ----------------------------- |
| `requirements.txt` | Python dependencies           |
| `cli.sh`           | Command-line interface script |

## Documentation Directory (`docs/`)

| File                    | Purpose                        |
| ----------------------- | ------------------------------ |
| `README.md`             | Main documentation             |
| `manifest.md`           | This file - file manifest      |
| `required_resources.md` | Hardware/software requirements |

## Figures Directory (`figures/`)

Contains vector/high-resolution PNG figures for the proposal:

- Architecture diagrams
- Timeline Gantt charts
- Environment illustrations

## Total Statistics

- **LaTeX source**: ~1000 lines
- **Python code**: ~3500 lines
- **Total files**: 30+ files
- **Documentation**: 5+ markdown files

## Checksums

See `artifact_manifest.json` for MD5 checksums of all files.
