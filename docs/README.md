# MARL Cross-Chain DeFi Proposal - Documentation

## Quick Start

### Installation

```bash
# Navigate to code directory
cd code

# Install dependencies
pip install -r requirements.txt
```

### Run Demo (End-to-End, ~10 minutes on CPU)

```bash
# Run complete pipeline
./cli.sh all

# Or step by step:
./cli.sh data      # Generate synthetic data
./cli.sh train     # Train agents
./cli.sh eval      # Evaluate agents
```

### Build Proposal PDF

```bash
cd ../proposal
make all
```

## Repository Structure

```
MARL_CrossChain_DeFi_Proposal/
├── proposal/              # LaTeX proposal source
│   ├── main.tex          # Main proposal document
│   ├── references.bib    # Bibliography
│   ├── Makefile          # Build automation
│   └── figs/             # Figures
├── code/                  # Implementation
│   ├── src/
│   │   ├── envs/         # Environment simulator
│   │   ├── agents/       # MARL agents (QMIX, MAPPO)
│   │   ├── train/        # Training scripts
│   │   ├── eval/         # Evaluation scripts
│   │   └── data/         # Data generation
│   ├── notebooks/        # Jupyter notebooks
│   ├── tests/            # Unit tests
│   ├── configs/          # Configuration files
│   └── cli.sh            # CLI script
├── docs/                  # Documentation
│   ├── README.md         # This file
│   ├── manifest.md       # File manifest
│   └── required_resources.md  # Resource requirements
├── figures/               # Proposal figures
├── DELIVERY_NOTE.txt      # Delivery summary
├── artifact_manifest.json # File checksums
└── LICENSE               # License files
```

## Running Individual Components

### Environment Demo

```bash
cd code
python src/envs/demo_env.py --steps 20 --seed 42
```

### Generate Synthetic Data

```bash
python src/data/generate_synthetic.py \
    --output-dir data/synthetic \
    --n-steps 1000 \
    --n-swaps 500 \
    --seed 42
```

### Train QMIX Agent

```bash
python src/train/train_synthetic.py \
    --config configs/demo.yaml \
    --agent qmix \
    --output-dir checkpoints
```

### Train MAPPO Agent

```bash
python src/train/train_synthetic.py \
    --config configs/demo.yaml \
    --agent mappo \
    --output-dir checkpoints
```

### Evaluate Trained Agent

```bash
python src/eval/evaluate_demo.py \
    --checkpoint checkpoints/qmix_model.pth \
    --agent-type qmix \
    --n-episodes 20 \
    --output-dir results
```

### Run Unit Tests

```bash
# Using pytest
pytest tests/ -v

# Or basic tests
./cli.sh test
```

## Jupyter Notebooks

1. **01_environment_demo.ipynb** - Interactive environment demonstration
2. **02_training_demo.ipynb** - Training and evaluation walkthrough

Launch with:

```bash
cd code/notebooks
jupyter notebook
```

## Configuration

Edit `code/configs/demo.yaml` to customize:

- Environment parameters (chains, bridges, pools)
- Training hyperparameters
- Evaluation settings

## Troubleshooting

### Import Errors

Make sure you're in the correct directory:

```bash
cd code
python -c "from src.envs.cross_chain_env import CrossChainEnv; print('OK')"
```

### Out of Memory

Reduce batch size in config:

```yaml
training:
  batch_size: 8 # Reduce from 16
```

### Slow Training

Reduce episodes for demo:

```yaml
training:
  n_episodes: 20 # Reduce from 50
```

## Citation

If using this code, please cite:

```bibtex
@misc{chainmarl2024,
  title={Multi-Agent Reinforcement Learning for Cross-Chain Liquidity and Routing Optimization},
  author={PhD Research Proposal},
  year={2024}
}
```

## License

- Code: MIT License
- Proposal text: CC-BY 4.0
