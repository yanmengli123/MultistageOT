# Data Download Guide

This document provides information on how to download all data used in the MultistageOT project.

## Main Data Repository (Zenodo)

**URL**: https://zenodo.org/records/17233337

**Total compressed size**: ~17.8 GB
**Total raw data size**: ~249 GB

### Download Method

1. Open browser and navigate to: https://zenodo.org/records/17233337
2. Download the file: `tronstad_manuscript_reproducibility_package.zip`
3. Extract to `D:\MultistageOT-main\data\zenodo_data\`

---

## Individual Dataset Sources

### 1. Weinreb et al. (2020) Data

**Paper**: [Science DOI](https://www.science.org/doi/10.1126/science.aaw3381)

**Gene Expression Omnibus (GEO)**:
```
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140802
```

**GitHub (AllonKleinLab)**:
```
https://github.com/AllonKleinLab/paper-data/tree/master/Lineage_tracing_on_transcriptional_landscapes_links_state_to_fate_during_differentiation
```

**Download via command line**:
```bash
# Clone GitHub repo (may be large)
git clone https://github.com/AllonKleinLab/paper-data.git
```

---

### 2. Dahlin et al. (2018) Data

**Paper**: [Blood DOI](https://ashpublications.org/blood/article/131/21/e1/37145/A-single-cell-hematopoietic-landscape-resolves-8)

**Gene Expression Omnibus (GEO)**:
```
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE107727
```

**Download via command line**:
```bash
# Download from GEO using prefetch (NCBI SRA Toolkit)
prefetch SRP100062
```

---

### 3. Paul et al. (2015) Data

**Paper**: [Cell DOI](https://doi.org/10.1016/j.cell.2015.11.013)

This dataset is **built into Scanpy** - no separate download needed:

```python
import scanpy as sc
adata = sc.datasets.paul15()
```

**Local backup** (already in this repository):
```
data/real_data/paul2015/20250129_paul2015_umap.csv
```

---

### 4. Synthetic Data

Already included in this repository:
```
data/synthetic_data/two_dimensional_data.csv
data/synthetic_data/README.txt
```

---

## Quick Start - Minimal Download

For running the MultistageOT demo, you only need:

1. **Synthetic data** - Already in this repository ✅
2. **Paul et al. data** - Already in this repository ✅
3. **Optional** - Download from Zenodo only if you need to reproduce ALL manuscript results

---

## Download Scripts

### Download Zenodo package
```bash
# Using wget
wget https://zenodo.org/records/17233337/files/tronstad_manuscript_reproducibility_package.zip

# Using curl
curl -L -o tronstad_manuscript_reproducibility_package.zip https://zenodo.org/records/17233337/files/tronstad_manuscript_reproducibility_package.zip
```

### Download Weinreb data from GitHub
```bash
git clone https://github.com/AllonKleinLab/paper-data.git
```

---

## Data Citation

If you use these datasets, please cite the original publications:

- **Paul et al. (2015)**: Cell 163, 1664-1674
- **Weinreb et al. (2020)**: Science 367, eaaw3381
- **Dahlin et al. (2018)**: Blood 131, e1-e16

---

## Support

For issues with data download, contact the original dataset authors or open an issue on the MultistageOT GitHub repository.