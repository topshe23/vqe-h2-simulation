# src/experiment.py
# Runs 3 VQE experiments and generates results

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.circuit import build_ansatz
from src.simulator import run_vqe, get_hamiltonian, H2_HAMILTONIAN
from src.utils import (plot_convergence, plot_energy_landscape,
                       plot_reps_vs_energy, save_results)


def experiment_vqe_convergence():
    """
    Experiment 1: Run VQE and track energy convergence.
    Shows how energy drops iteration by iteration.
    """
    print("\nExperiment 1: VQE Convergence...")
    qc, _ = build_ansatz(reps=1)
    result = run_vqe(qc, H2_HAMILTONIAN, max_iter=300)

    print(f"  Optimal energy: {result['optimal_energy']:.6f} Ha")
    print(f"  Exact FCI:      -1.137274 Ha")
    print(f"  Error:          {abs(result['optimal_energy'] - (-1.137274)):.6f} Ha")
    print(f"  Iterations:     {result['iterations']}")

    return result


def experiment_bond_length_sweep():
    """
    Experiment 2: Sweep H-H bond length from 0.3 to 2.5 Angstrom.
    Traces the potential energy surface of H2.
    """
    print("\nExperiment 2: Bond Length Sweep...")

    bond_lengths = np.linspace(0.3, 2.5, 12)
    energies = []

    for bl in bond_lengths:
        hamiltonian = get_hamiltonian(bond_length=bl)
        qc, _ = build_ansatz(reps=1)
        result = run_vqe(qc, hamiltonian, max_iter=200)
        energies.append(result['optimal_energy'])
        print(f"  bond_length={bl:.2f}A -> energy={result['optimal_energy']:.6f} Ha")

    return list(bond_lengths), energies


def experiment_reps_vs_accuracy():
    """
    Experiment 3: How does ansatz depth affect accuracy?
    More reps = more parameters = closer to exact answer.
    """
    print("\nExperiment 3: Ansatz Depth vs Accuracy...")

    reps_list = [1, 2, 3, 4]
    energies = []

    for reps in reps_list:
        qc, _ = build_ansatz(reps=reps)
        result = run_vqe(qc, H2_HAMILTONIAN, max_iter=300)
        energies.append(result['optimal_energy'])
        print(f"  reps={reps} ({qc.num_parameters} params) -> energy={result['optimal_energy']:.6f} Ha")

    return reps_list, energies


if __name__ == "__main__":

    # --- Experiment 1: VQE Convergence ---
    conv_result = experiment_vqe_convergence()

    plot_convergence(
        conv_result['energy_history'],
        conv_result['optimal_energy'],
        'results/plots/vqe_convergence.png'
    )
    save_results(
        {'iteration': list(range(len(conv_result['energy_history']))),
         'energy': conv_result['energy_history']},
        'results/data/convergence_results.csv'
    )

    # --- Experiment 2: Bond Length Sweep ---
    bond_lengths, energies_bl = experiment_bond_length_sweep()

    plot_energy_landscape(
        bond_lengths, energies_bl,
        'results/plots/energy_landscape.png'
    )
    save_results(
        {'bond_length': bond_lengths, 'energy': energies_bl},
        'results/data/bond_length_results.csv'
    )

    # --- Experiment 3: Reps vs Accuracy ---
    reps_list, energies_reps = experiment_reps_vs_accuracy()

    plot_reps_vs_energy(
        reps_list, energies_reps,
        'results/plots/reps_vs_energy.png'
    )
    save_results(
        {'reps': reps_list, 'energy': energies_reps},
        'results/data/reps_results.csv'
    )

    print("\nAll experiments done!")
    print("Plots saved to results/plots/")
    print("CSVs saved to results/data/")