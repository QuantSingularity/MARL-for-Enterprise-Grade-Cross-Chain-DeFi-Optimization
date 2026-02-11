# PhD Research Proposal: Multi-Agent Reinforcement Learning for Cross-Chain DeFi

## Building the PDF

### Prerequisites

- [Tectonic](https://tectonic-typesetting.github.io/) LaTeX engine
- Or any standard LaTeX distribution (TeX Live, MiKTeX)

### Build Instructions

#### Using Make (Recommended)

```bash
# Build the PDF (runs tectonic twice for cross-references)
make all

# Quick build (single pass)
make quick

# Clean auxiliary files
make clean

# View the PDF
make view
```

#### Using Tectonic Directly

```bash
# First pass
tectonic main.tex

# Second pass (for cross-references)
tectonic main.tex
```

#### Using Traditional LaTeX

```bash
# Compile
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Files in This Directory

- `main.tex` - Main LaTeX source file (proposal content)
- `references.bib` - BibTeX bibliography file (40+ citations)
- `Makefile` - Build automation
- `main.pdf` - Generated proposal PDF (6-10 pages)
- `figs/` - Directory for figures (TikZ diagrams generated inline)

## Proposal Summary

This PhD research proposal develops **ChainMARL**: a multi-agent reinforcement learning framework for optimizing cross-chain decentralized finance (DeFi) operations. The proposal covers:

1. **Problem Formalization** - POSG formulation of cross-chain DeFi optimization
2. **Methodology** - QMIX, MAPPO, attention-based communication, GNN encoders
3. **Risk Modeling** - VaR/CVaR constraints and Bayesian failure predictors
4. **Evaluation Plan** - Metrics, baselines, and experimental design
5. **Timeline** - 12-month research schedule with milestones
6. **Resources** - Compute, data access, and feasibility assessment

The proposal includes novel contributions in attention-based communication for heterogeneous agents, GNN-based cross-chain state encoding, and risk-aware MARL training.

## Target Venues

This research is suitable for submission to:

- Top AI/ML conferences: NeurIPS, ICML, ICLR
- Blockchain/DeFi venues: ACM AFT, FC, IEEE S&P
- Multi-agent systems: AAMAS, AAAI

## Citation

If referencing this proposal, please cite:

```bibtex
@misc{chainmarl2024,
  title = {Multi-Agent Reinforcement Learning for Cross-Chain Liquidity and Routing Optimization},
  author = {PhD Research Proposal},
  year = {2024},
  note = {PhD Research Proposal}
}
```
