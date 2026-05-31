# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MultistageOT is a Python framework for trajectory inference in single-cell RNA-sequencing data using multistage optimal transport. The core model is in `packages/models/multistageot.py`.

## Installation

```bash
pip install numpy==1.24.3
pip install scanpy==1.9.3
pip install jupyter
```

## Architecture

### Core Module
- `packages/models/multistageot.py` — Contains the `MultistageOT` class implementing the optimal transport model with methods:
  - `.fit(data)` — Fit model to data (accepts pandas DataFrame)
  - `.pseudotemporal_order()` — Infer pseudotime
  - `.cell_fate_probabilities(fate_groups)` — Infer fate probabilities (fate_groups: dict mapping fate labels to cell index arrays)

### Visualization Tools
- `packages/models/tools.py` — Plotting utilities:
  - `plot_couplings()` — Visualize cell-cell coupling matrices
  - `plot_fate_probabilities_with_pie_charts()` — Visualize fate probabilities as pie charts

### Data & Notebooks
- `data/` — Input datasets (synthetic_data/, real_data/)
- `scripts/notebooks/` — Jupyter notebooks:
  - `synthetic_data/main.ipynb` — Demo on synthetic data
  - `real_data/paul2015/main.ipynb` — Application to Paul et al. (2015) hematopoiesis data

## Key Dependencies

- Python 3.9
- numpy 1.24.3 (compatible with scanpy 1.9.3)
- scanpy 1.9.3
- pandas
- matplotlib

## Common Usage Pattern

```python
from packages.models.multistageot import MultistageOT
import pandas as pd

# Instantiate with cell labels
msot = MultistageOT(
    initial_cells=[0],  # least mature cells
    terminal_cells=[5], # most mature cells
    n_groups=4,
    epsilon=0.01
)

# Fit and infer
msot.fit(data)
pt = msot.pseudotemporal_order()
cfp = msot.cell_fate_probabilities({'Fate1': [5]})
```
