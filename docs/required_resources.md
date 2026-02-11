# Required Resources

## Hardware Requirements

### Minimum (Demo Mode)

For running the quick demo on synthetic data:

| Resource | Requirement                 |
| -------- | --------------------------- |
| CPU      | 4 cores (any modern x86_64) |
| RAM      | 8 GB                        |
| Storage  | 2 GB free space             |
| GPU      | Not required (CPU-only)     |

### Recommended (Development)

For full development and larger-scale experiments:

| Resource | Requirement                                       |
| -------- | ------------------------------------------------- |
| CPU      | 8+ cores                                          |
| RAM      | 32 GB                                             |
| Storage  | 50 GB free space (for datasets)                   |
| GPU      | NVIDIA GPU with 8+ GB VRAM (V100, RTX 3080, etc.) |

### Production Scale

For experiments with 10+ chains and full historical data:

| Resource | Requirement                  |
| -------- | ---------------------------- |
| CPU      | 32+ cores                    |
| RAM      | 128 GB                       |
| Storage  | 1 TB SSD                     |
| GPU      | Multiple A100s or equivalent |

## Software Requirements

### Core Dependencies

| Package | Version | Purpose                 |
| ------- | ------- | ----------------------- |
| Python  | 3.10+   | Language runtime        |
| PyTorch | 2.0+    | Deep learning framework |
| NumPy   | 1.24+   | Numerical computing     |
| Pandas  | 2.0+    | Data manipulation       |

### RL/ML Libraries

| Package         | Version | Purpose                  |
| --------------- | ------- | ------------------------ |
| Gymnasium       | 0.28+   | Environment API          |
| PettingZoo      | 1.24+   | Multi-agent environments |
| torch-geometric | 2.3+    | Graph neural networks    |

### Blockchain/Web3

| Package     | Version | Purpose              |
| ----------- | ------- | -------------------- |
| web3.py     | 6.0+    | Ethereum interaction |
| eth-account | 0.8+    | Account management   |

### Optional Dependencies

| Package | Version | Purpose             |
| ------- | ------- | ------------------- |
| Jupyter | 1.0+    | Notebooks           |
| wandb   | 0.15+   | Experiment tracking |
| pytest  | 7.3+    | Testing             |

### LaTeX (for Proposal)

| Tool         | Version | Purpose           |
| ------------ | ------- | ----------------- |
| Tectonic     | 0.15+   | LaTeX compiler    |
| Or: TeX Live | 2023+   | Alternative LaTeX |

## Cloud Resources

### Estimated Compute for Full Experiments

| Phase            | GPU Hours | Cost (USD)        |
| ---------------- | --------- | ----------------- |
| Development      | 500       | $500-1000         |
| Full Training    | 5,000     | $5,000-10,000     |
| Ablation Studies | 2,000     | $2,000-4,000      |
| **Total**        | **7,500** | **$7,500-15,000** |

_Based on AWS p3.2xlarge (V100) pricing at $3.06/hour_

### Data API Costs

| Service        | Tier      | Monthly Cost       |
| -------------- | --------- | ------------------ |
| Alchemy        | Growth    | $49-199            |
| Infura         | Developer | $0-225             |
| Dune Analytics | Plus      | $300               |
| **Total**      |           | **$350-725/month** |

## Data Storage Requirements

### Synthetic Data (Demo)

| Dataset     | Size       |
| ----------- | ---------- |
| Chain data  | ~10 MB     |
| Bridge data | ~5 MB      |
| Swap events | ~20 MB     |
| **Total**   | **~35 MB** |

### Real Historical Data (Full Scale)

| Dataset         | Size        |
| --------------- | ----------- |
| Ethereum blocks | ~500 GB     |
| Arbitrum blocks | ~100 GB     |
| Optimism blocks | ~50 GB      |
| Polygon blocks  | ~200 GB     |
| Bridge logs     | ~50 GB      |
| DEX traces      | ~200 GB     |
| **Total**       | **~1.1 TB** |

## Network Requirements

| Use Case       | Bandwidth                 |
| -------------- | ------------------------- |
| Demo/Synthetic | Minimal (offline capable) |
| API Access     | 10 Mbps stable            |
| Full Node Sync | 100 Mbps+                 |

## Time Estimates

### Demo Pipeline (~10 minutes on CPU)

| Step                         | Time       |
| ---------------------------- | ---------- |
| Data generation              | 30 seconds |
| QMIX training (50 episodes)  | 3 minutes  |
| MAPPO training (50 episodes) | 3 minutes  |
| Evaluation                   | 2 minutes  |
| Plotting                     | 1 minute   |

### Full Training (with GPU)

| Phase               | Time           |
| ------------------- | -------------- |
| Data collection     | 1-2 weeks      |
| Environment setup   | 1 week         |
| Baseline training   | 1 week         |
| QMIX/MAPPO training | 2-4 weeks      |
| Ablation studies    | 2 weeks        |
| Evaluation          | 1 week         |
| **Total**           | **8-12 weeks** |

## Fallback Plans

### If GPU Unavailable

1. Use smaller model architectures
2. Reduce batch sizes
3. Train for fewer episodes
4. Use cloud GPU credits (AWS, GCP, Colab)

### If API Rate Limited

1. Use synthetic data (provided)
2. Cache API responses
3. Use public datasets (Dune Analytics)
4. Partner with data providers

### If Storage Limited

1. Stream data during training
2. Use cloud storage (S3, GCS)
3. Compress historical data
4. Sample from full dataset

## Getting Access

### Free Tier Options

| Service        | Free Tier                |
| -------------- | ------------------------ |
| Alchemy        | 100M compute units/month |
| Infura         | 100,000 requests/day     |
| Dune Analytics | Community access         |
| Google Colab   | Free GPU (T4)            |
| AWS            | $300 credits (new users) |

### Academic Discounts

- AWS Educate: $100-500 credits
- GitHub Student Pack: Various credits
- Many API providers offer academic rates

## Contact for Resources

For collaboration or resource sharing:

- DeFi protocols (LayerZero, Stargate)
- Academic institutions
- Research groups
- Industry partners
