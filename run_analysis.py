"""
MultistageOT - Run analysis and generate output images
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import sys
import os

# Add project root to path
sys.path.append(".")

from packages.models.multistageot import MultistageOT
import packages.models.tools

# Create output directory
os.makedirs("results/synthetic_data", exist_ok=True)

# =============================================================================
# Load and preprocess data
# =============================================================================
print("Loading data...")
DATA_PATH = "data/synthetic_data/two_dimensional_data.csv"
df = pd.read_csv(DATA_PATH, header=None, names=['x', 'y', 'class'])

print(f"Data shape: {df.shape}")
print(f"Class distribution:\n{df['class'].value_counts().sort_index()}")

# Normalize to unit square
df[['x', 'y']] = (df[['x', 'y']] - df[['x', 'y']].min()) / (df[['x', 'y']].max() - df[['x', 'y']].min())

# Re-label non-initial and non-intermediate as terminal (class 2)
df.loc[(df['class'] != 1) & (df['class'] != 0), 'class'] = 2

# Prepare input data
input_data = df.iloc[:, :2]
initial_cells = df.loc[df['class'] == 0].index.tolist()
terminal_cells = df.loc[df['class'] == 2].index.tolist()

print(f"\nInitial cells: {initial_cells}")
print(f"Terminal cells: {terminal_cells}")
print(f"Intermediate cells: {len(df.loc[df['class'] == 1])}")

# Define fate groups (terminal cell indices)
fates = {"Fate 1": [90, 91], "Fate 2": [92, 93, 94], "Fate 3": [95, 96, 97, 98], "Fate 4": [99, 100, 101, 102, 103, 104]}

# =============================================================================
# Plot 1: Original data
# =============================================================================
print("\nGenerating Plot 1: Original data...")
fig, ax = plt.subplots(figsize=(6, 6))
size = 90
ax.scatter(df['x'], df['y'], s=1.25 * size, linewidth=2, facecolor='none', edgecolor='k')
ax.scatter(df['x'], df['y'], s=size, color=[0.9, 0.9, 0.9])
ax.scatter(df.loc[df['class'] == 0]['x'], df.loc[df['class'] == 0]['y'], s=0.5 * size, color='tab:blue', edgecolor='k', label='Initial')
ax.scatter(df.loc[df['class'] == 2]['x'], df.loc[df['class'] == 2]['y'], s=0.5 * size, color='tab:orange', marker='X', edgecolor='k', label='Terminal')
sbn.despine(left=True, bottom=True)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Synthetic Data - Initial (blue) and Terminal (orange) cells')
ax.legend(loc='best')
plt.tight_layout()
plt.savefig("results/synthetic_data/01_original_data.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/synthetic_data/01_original_data.png")

# =============================================================================
# Initialize and fit MultistageOT model
# =============================================================================
print("\nInitializing MultistageOT model...")
msot = MultistageOT(
    initial_cells=initial_cells,
    terminal_cells=terminal_cells,
    n_groups=10,
    epsilon=0.0125
)

print("Fitting model...")
msot.fit(input_data, verbose=True, patience=10, tolerance=1e-4)

print("\nRunning proximal Sinkhorn to reduce regularization...")
msot.proximal_sinkhorn(epsilon_threshold=0.006, patience=10)

# =============================================================================
# Plot 2: Pseudotime ordering
# =============================================================================
print("\nGenerating Plot 2: Pseudotime ordering...")
pt = msot.pseudotemporal_order()
fig, ax = plt.subplots(figsize=(6, 6))
size = 90
ax.scatter(df['x'], df['y'], s=1.25 * size, linewidth=2, facecolor='none', edgecolor='k')
scatter = ax.scatter(df['x'], df['y'], c=pt, s=size, cmap='Blues', edgecolor='none')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Pseudotime Ordering')
plt.colorbar(scatter, ax=ax, label='Pseudotime')
sbn.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig("results/synthetic_data/02_pseudotime_ordering.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/synthetic_data/02_pseudotime_ordering.png")

# =============================================================================
# Plot 3: Transport stages
# =============================================================================
print("\nGenerating Plot 3: Transport stages...")
mu = msot.marginals()
cell_group = msot.max_marginal_groups()

df_max = df.copy()
df_max.loc[mu.index, 'max_stage'] = np.argmax(mu.values, axis=1) + 1
df_max.loc[msot.initial_cells, 'max_stage'] = 0
df_max.loc[msot.terminal_cells, 'max_stage'] = 11

fig, ax = plt.subplots(figsize=(6, 6))
size = 90
ax.scatter(df['x'], df['y'], s=1.25 * size, facecolor='none', edgecolor='k')
scatter = ax.scatter(df['x'], df['y'], s=size, edgecolor='none', c=df_max['max_stage'], cmap='Paired')
ax.scatter(df.loc[df['class'] == 0]['x'], df.loc[df['class'] == 0]['y'], color='k', s=size * 0.5)
ax.scatter(df.loc[df['class'] == 2]['x'], df.loc[df['class'] == 2]['y'], color='k', s=size * 0.5, marker='X')
cb = plt.colorbar(scatter, ax=ax)
cb.ax.set_title('Stage')
sbn.despine(left=True, bottom=True)
plt.xticks([])
plt.yticks([])
ax.set_title('Most Active Transport Stage per Cell')
plt.tight_layout()
plt.savefig("results/synthetic_data/03_transport_stages.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/synthetic_data/03_transport_stages.png")

# =============================================================================
# Plot 4: Cell fate probabilities
# =============================================================================
print("\nGenerating Plot 4: Cell fate probabilities...")
transition_matrix = msot.transition_matrix()
cell_fate_probs = msot.cell_fate_probabilities(fate_groups=fates, transition_matrix=transition_matrix)

fig, ax = plt.subplots(figsize=(6, 6))
packages.models.tools.plot_fate_probabilities_with_pie_charts(
    ax, df[['x', 'y']], cell_fate_probs,
    entropy_fade=False, frac=1, s=124,
    plot_order=['Fate 3', 'Fate 2', 'Fate 1', 'Fate 4'],
    background=120
)
ax.get_legend().remove()
sbn.despine(bottom=True, left=True)
plt.xticks([])
plt.yticks([])
ax.set_title('Cell Fate Probabilities')
plt.tight_layout()
plt.savefig("results/synthetic_data/04_cell_fate_probabilities.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/synthetic_data/04_cell_fate_probabilities.png")

# =============================================================================
# Plot 5: Transition matrix
# =============================================================================
print("\nGenerating Plot 5: Transition matrix...")
fig, ax = plt.subplots(figsize=(6, 6))
plt.imshow(transition_matrix, cmap='inferno')
plt.colorbar(label='Transition Probability')
plt.xticks([])
plt.yticks([])
ax.set_title('Cell-Cell Transition Probability Matrix')
plt.tight_layout()
plt.savefig("results/synthetic_data/05_transition_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/synthetic_data/05_transition_matrix.png")

# =============================================================================
# Plot 6: Stage distribution histogram
# =============================================================================
print("\nGenerating Plot 6: Stage distribution histogram...")
fig, ax = plt.subplots(figsize=(6, 4))
sbn.histplot(df_max.loc[msot.intermediate_cells, 'max_stage'], ax=ax, bins=10)
ax.set_title('Distribution of Most Active Transport Stage')
ax.set_xlabel('Stage')
ax.set_ylabel('Number of Cells')
sbn.despine()
plt.tight_layout()
plt.savefig("results/synthetic_data/06_stage_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/synthetic_data/06_stage_distribution.png")

# =============================================================================
# Summary statistics
# =============================================================================
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nPseudotime range: {pt.min():.4f} to {pt.max():.4f}")
print(f"\nMean fate probabilities:")
for fate in cell_fate_probs.columns:
    print(f"  {fate}: {cell_fate_probs[fate].mean():.4f}")

print(f"\nAll images saved to: results/synthetic_data/")
print("\nGenerated files:")
for f in sorted(os.listdir("results/synthetic_data")):
    if f.endswith(".png"):
        print(f"  - {f}")