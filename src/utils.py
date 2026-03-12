# src/utils.py
# Helpers for plotting and saving results

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def plot_convergence(energy_history, optimal_energy, save_path):
    """Plots VQE convergence — energy vs iteration."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(energy_history, color='steelblue', linewidth=2, label='VQE Energy')
    ax.axhline(y=optimal_energy, color='red', linestyle='--',
               linewidth=1.5, label=f'Converged: {optimal_energy:.4f} Ha')
    ax.axhline(y=-1.1372, color='green', linestyle=':',
               linewidth=1.5, label='Exact FCI: -1.1372 Ha')

    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel('Energy (Hartree)', fontsize=13)
    ax.set_title('VQE Convergence — H₂ Ground State Energy', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_energy_landscape(bond_lengths, energies, save_path):
    """Plots energy vs bond length — the potential energy surface."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(bond_lengths, energies, marker='o', color='darkorange',
            linewidth=2.5, markersize=7, label='VQE Energy')
    ax.axvline(x=0.735, color='gray', linestyle='--',
               alpha=0.7, label='Equilibrium (0.735 Å)')

    min_idx = np.argmin(energies)
    ax.annotate(f'Min: {energies[min_idx]:.4f} Ha',
                xy=(bond_lengths[min_idx], energies[min_idx]),
                xytext=(bond_lengths[min_idx] + 0.2, energies[min_idx] + 0.05),
                fontsize=9, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xlabel('Bond Length (Å)', fontsize=13)
    ax.set_ylabel('Ground State Energy (Hartree)', fontsize=13)
    ax.set_title('H₂ Potential Energy Surface — VQE', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_reps_vs_energy(reps_list, energies, save_path):
    """Plots ansatz depth (reps) vs converged energy."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(reps_list, energies, marker='s', color='seagreen',
            linewidth=2.5, markersize=8, label='VQE Energy')
    ax.axhline(y=-1.1372, color='red', linestyle='--',
               linewidth=1.5, label='Exact FCI: -1.1372 Ha')

    ax.set_xlabel('Ansatz Repetitions (depth)', fontsize=13)
    ax.set_ylabel('Converged Energy (Hartree)', fontsize=13)
    ax.set_title('Ansatz Depth vs Energy Accuracy', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


def save_results(data_dict, filepath):
    """Saves results to CSV."""
    df = pd.DataFrame(data_dict)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")