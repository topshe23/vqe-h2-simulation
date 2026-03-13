# src/utils.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import os


def plot_convergence(energy_history, optimal_energy, save_path):
    """Publication quality VQE convergence plot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    iterations = list(range(len(energy_history)))

    ax.plot(iterations, energy_history, color='steelblue',
            linewidth=2, label='VQE Energy', zorder=3)
    ax.axhline(y=optimal_energy, color='red', linestyle='--',
               linewidth=1.8, label=f'Converged: {optimal_energy:.4f} Ha', zorder=2)
    ax.axhline(y=-1.137274, color='green', linestyle=':',
               linewidth=1.8, label='Exact FCI: -1.1372 Ha', zorder=2)

    # Shade convergence zone
    ax.axhspan(optimal_energy - 0.001, optimal_energy + 0.001,
               alpha=0.15, color='red')

    # Annotate convergence point
    conv_iter = next(i for i, e in enumerate(energy_history) if abs(e - optimal_energy) < 0.001)
    ax.annotate(f'Converged at\niteration {conv_iter}',
                xy=(conv_iter, optimal_energy),
                xytext=(conv_iter + 10, optimal_energy + 0.04),
                fontsize=9, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

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
    """Publication quality potential energy surface plot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(bond_lengths, energies, marker='o', color='darkorange',
            linewidth=2.5, markersize=8, label='VQE Energy', zorder=3)

    # Mark minimum
    min_idx = np.argmin(energies)
    ax.axvline(x=0.735, color='gray', linestyle='--',
               alpha=0.7, linewidth=1.5, label='Experimental equilibrium (0.735 Å)')
    ax.scatter([bond_lengths[min_idx]], [energies[min_idx]],
               color='red', s=120, zorder=5, label=f'VQE minimum: {energies[min_idx]:.4f} Ha')
    ax.annotate(f'  Min: {energies[min_idx]:.4f} Ha\n  at {bond_lengths[min_idx]:.2f} Å',
                xy=(bond_lengths[min_idx], energies[min_idx]),
                xytext=(bond_lengths[min_idx] + 0.3, energies[min_idx] + 0.15),
                fontsize=9, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

    # Shade repulsion and dissociation zones
    ax.axvspan(0.3, 0.6, alpha=0.08, color='red', label='Repulsion zone')
    ax.axvspan(1.8, 2.5, alpha=0.08, color='blue', label='Dissociation zone')

    ax.set_xlabel('H-H Bond Length (Å)', fontsize=13)
    ax.set_ylabel('Ground State Energy (Hartree)', fontsize=13)
    ax.set_title('H₂ Potential Energy Surface — VQE Simulation', fontsize=15, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_reps_vs_energy(reps_list, energies, save_path):
    """Publication quality ansatz depth vs energy plot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['steelblue' if abs(e - (-1.137274)) < 0.001 else 'darkorange' for e in energies]

    bars = ax.bar(reps_list, [abs(e) for e in energies],
                  color=colors, edgecolor='white', linewidth=1.5, zorder=3)

    ax.axhline(y=1.137274, color='red', linestyle='--',
               linewidth=1.8, label='Exact FCI: -1.1372 Ha', zorder=2)

    # Annotate each bar
    for i, (rep, energy) in enumerate(zip(reps_list, energies)):
        ax.text(rep, abs(energy) + 0.005, f'{energy:.4f}',
                ha='center', fontsize=9, fontweight='bold')

    exact_patch = mpatches.Patch(color='steelblue', label='Reached exact FCI')
    ax.legend(handles=[exact_patch], fontsize=10)

    ax.set_xlabel('Ansatz Repetitions (depth)', fontsize=13)
    ax.set_ylabel('|Energy| (Hartree)', fontsize=13)
    ax.set_title('Ansatz Depth vs Energy Accuracy — H₂ VQE', fontsize=15, fontweight='bold')
    ax.set_xticks(reps_list)
    ax.set_ylim(1.10, 1.16)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


def save_results(data_dict, filepath):
    df = pd.DataFrame(data_dict)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")