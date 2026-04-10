# readout_noise_experiment.py
# VQE H2 with measurement (readout) error — for Ritajit Majumdar, IBM Quantum

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError
from qiskit import transpile
import sys
sys.path.insert(0, '.')
from src.circuit import build_ansatz

# --- Hamiltonian ---
H2_HAMILTONIAN = SparsePauliOp.from_list([
    ("II", -0.580295),
    ("ZI", -0.22575),
    ("IZ",  0.17407),
    ("ZZ",  0.12091),
    ("XX",  0.17407),
])

EXACT_ENERGY = -1.137274  # FCI reference


def compute_energy_ideal(circuit, parameter_values):
    """Exact statevector energy — no noise."""
    param_dict = dict(zip(circuit.parameters, parameter_values))
    bound = circuit.assign_parameters(param_dict)
    sv = Statevector(bound)
    return sv.expectation_value(H2_HAMILTONIAN).real


def compute_energy_with_readout(circuit, parameter_values,
                                 readout_error_prob, shots=8192):
    """
    Energy with readout (measurement) error.
    Readout error: with probability p, a 0 is flipped to 1 or vice versa
    during measurement. This is the most common error on real hardware.
    """
    # Build readout error model
    noise_model = NoiseModel()

    # p(1|0) = prob of reading 1 when qubit is 0
    # p(0|1) = prob of reading 0 when qubit is 1
    p = readout_error_prob
    readout = ReadoutError([[1-p, p], [p, 1-p]])
    noise_model.add_all_qubit_readout_error(readout)

    # Bind parameters
    param_dict = dict(zip(circuit.parameters, parameter_values))
    bound = circuit.assign_parameters(param_dict)

    # We need to measure each Pauli term separately
    energy = 0.0
    for pauli_str, coeff in zip(
        [term[0] for term in H2_HAMILTONIAN.to_list()],
        [term[1] for term in H2_HAMILTONIAN.to_list()]
    ):
        exp_val = _measure_pauli_noisy(
            bound, pauli_str, noise_model, shots)
        energy += coeff.real * exp_val

    return energy


def _measure_pauli_noisy(circuit, pauli_str, noise_model, shots):
    """
    Measures expectation value of a Pauli string with readout noise.
    Rotates to correct basis before measuring.
    """
    from qiskit import QuantumCircuit
    n = len(pauli_str)

    # Build measurement circuit
    meas_circuit = circuit.copy()
    meas_circuit.remove_final_measurements(inplace=False)

    qc = QuantumCircuit(n, n)
    qc.compose(meas_circuit, inplace=True)

    # Rotate to measurement basis
    for i, p in enumerate(reversed(pauli_str)):
        if p == 'X':
            qc.h(i)
        elif p == 'Y':
            qc.sdg(i)
            qc.h(i)

    qc.measure_all()

    sim = AerSimulator(noise_model=noise_model)
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # Compute expectation value
    total = sum(counts.values())
    exp_val = 0.0
    for bitstring, count in counts.items():
        # Remove spaces from bitstring
        bits = bitstring.replace(' ', '')[-n:]
        # Parity: +1 if even number of 1s, -1 if odd
        # Only count qubits where Pauli is not I
        parity = 1
        for i, p in enumerate(reversed(pauli_str)):
            if p != 'I':
                parity *= (-1) ** int(bits[i])
        exp_val += parity * count / total

    return exp_val


def run_vqe_with_noise(readout_prob, n_runs=5):
    """
    Runs VQE multiple times with given readout error.
    Returns list of converged energies for mean/std calculation.
    """
    qc, _ = build_ansatz(reps=1)
    converged_energies = []

    for run in range(n_runs):
        energy_history = []
        np.random.seed(run * 7)
        initial_params = np.random.uniform(0, np.pi, qc.num_parameters)

        def cost(params):
            if readout_prob == 0.0:
                e = compute_energy_ideal(qc, params)
            else:
                e = compute_energy_with_readout(qc, params, readout_prob)
            energy_history.append(e)
            return e

        result = minimize(cost, initial_params, method='COBYLA',
                          options={'maxiter': 200, 'rhobeg': 0.3})
        converged_energies.append(result.fun)

    return converged_energies


def main():
    print("VQE H2 — Readout Noise Experiment")
    print(f"Exact FCI energy: {EXACT_ENERGY} Ha")
    print("="*50)

    # Noise levels as requested by Dr. Majumdar
    noise_levels = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    n_runs = 5  # runs per noise level for statistics

    all_means = []
    all_stds = []
    all_energies = {}

    for p in noise_levels:
        label = "Ideal" if p == 0.0 else f"p={p}"
        print(f"\nRunning VQE with readout error {label}...")
        energies = run_vqe_with_noise(p, n_runs=n_runs)
        mean_e = np.mean(energies)
        std_e = np.std(energies)
        all_means.append(mean_e)
        all_stds.append(std_e)
        all_energies[p] = energies
        print(f"  Energies: {[round(e,6) for e in energies]}")
        print(f"  Mean: {mean_e:.6f} Ha")
        print(f"  Std:  {std_e:.6f} Ha")
        print(f"  Error from FCI: {abs(mean_e - EXACT_ENERGY):.6f} Ha")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean energy vs noise level with error bars
    colors = ['green'] + ['steelblue'] * (len(noise_levels) - 1)
    ax1.errorbar(noise_levels, all_means, yerr=all_stds,
                 fmt='o-', color='steelblue', linewidth=2.5,
                 markersize=8, capsize=5, capthick=2,
                 label='VQE Converged Energy')
    ax1.axhline(y=EXACT_ENERGY, color='red', linestyle='--',
                linewidth=1.8, label=f'Exact FCI: {EXACT_ENERGY} Ha')
    ax1.scatter([0.0], [all_means[0]], color='green',
                s=120, zorder=5, label=f'Ideal: {all_means[0]:.4f} Ha')

    ax1.set_xlabel('Readout Error Probability (p)', fontsize=13)
    ax1.set_ylabel('Converged Energy (Hartree)', fontsize=13)
    ax1.set_title('VQE H2 — Effect of Readout Error on\nConverged Energy',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mean error from FCI vs noise level
    errors_from_fci = [abs(m - EXACT_ENERGY) for m in all_means]
    ax2.bar(noise_levels, errors_from_fci,
            color=['green'] + ['steelblue'] * (len(noise_levels)-1),
            width=0.008, edgecolor='white')
    ax2.set_xlabel('Readout Error Probability (p)', fontsize=13)
    ax2.set_ylabel('|Energy Error| from FCI (Hartree)', fontsize=13)
    ax2.set_title('VQE H2 — Energy Deviation from FCI\nvs Readout Error',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for i, (p, err, std) in enumerate(
            zip(noise_levels, errors_from_fci, all_stds)):
        ax2.text(p, err + 0.001, f'{err:.4f}\n±{std:.4f}',
                 ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('results/plots/readout_noise_vqe.png', dpi=150)
    plt.close()
    print("\nPlot saved to results/plots/readout_noise_vqe.png")

    # --- Save CSV ---
    df = pd.DataFrame({
        'readout_error_prob': noise_levels,
        'mean_energy': all_means,
        'std_energy': all_stds,
        'error_from_fci': errors_from_fci
    })
    df.to_csv('results/data/readout_noise_results.csv', index=False)
    print("CSV saved to results/data/readout_noise_results.csv")
    print("\nSummary Table:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()