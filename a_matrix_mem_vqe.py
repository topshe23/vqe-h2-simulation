# a_matrix_mem_vqe.py
# VQE H2 with A-matrix Measurement Error Mitigation
#
# Approach:
# - Use SamplerV2 to get raw measurement counts
# - Manually build circuits for each Pauli basis measurement
# - Apply A-matrix mitigation to correct noisy counts
# - Compute expectation value from mitigated counts
# - Compare VQE convergence with and without mitigation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from numpy import kron, dot
from numpy.linalg import inv

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.result import sampled_expectation_value, marginal_counts
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError
from qiskit_ibm_runtime import SamplerV2, Session

import sys
sys.path.insert(0, '.')
from src.circuit import build_ansatz

# H2 Hamiltonian
H2_HAMILTONIAN = SparsePauliOp.from_list([
    ("II", -0.580295),
    ("ZI", -0.22575),
    ("IZ",  0.17407),
    ("ZZ",  0.12091),
    ("XX",  0.17407),
])

EXACT_ENERGY = -1.137274
SHOTS = 8192


def make_noisy_backend(p):
    """
    AerSimulator with symmetric readout error probability p.
    Uses transpose convention as in Dr. Majumdar's notebook.
    """
    noise_model = NoiseModel()
    if p > 0.0:
        a_matrix = [[1-p, p], [p, 1-p]]
        roerror = ReadoutError(np.transpose(a_matrix))
        noise_model.add_all_qubit_readout_error(roerror)
    return AerSimulator(noise_model=noise_model)


def build_a_matrix(p):
    """
    Builds the 2-qubit A matrix for symmetric noise p.
    A_total = A_qubit0 ⊗ A_qubit1
    """
    a_single = np.array([[1-p, p], [p, 1-p]])
    return kron(a_single, a_single)


def build_pauli_circuits(ansatz, params):
    """
    Builds measurement circuits for each Pauli term in H2 Hamiltonian.
    For each Pauli string, rotates to the correct measurement basis.

    H2 Hamiltonian terms: II, ZI, IZ, ZZ, XX
    - II: no measurement needed (constant -0.580295)
    - ZI, IZ, ZZ: measure in Z basis (default, no rotation)
    - XX: apply H gate to both qubits before measuring
    """
    # Bind parameters
    param_dict = dict(zip(ansatz.parameters, params))
    bound = ansatz.assign_parameters(param_dict)

    # Remove measurements from ansatz if any
    bound_no_meas = bound.copy()

    circuits = {}

    # ZZ basis circuit (also used for ZI and IZ)
    qc_zz = QuantumCircuit(2)
    qc_zz.compose(bound_no_meas, inplace=True)
    qc_zz.measure_all()
    circuits['ZZ'] = qc_zz

    # XX basis circuit — apply H to both qubits
    qc_xx = QuantumCircuit(2)
    qc_xx.compose(bound_no_meas, inplace=True)
    qc_xx.h(0)
    qc_xx.h(1)
    qc_xx.measure_all()
    circuits['XX'] = qc_xx

    return circuits


def get_noisy_counts(circuits, sampler):
    """Runs circuits with Sampler and returns counts."""
    pubs = [(qc,) for qc in circuits.values()]
    job = sampler.run(pubs)
    results = job.result()

    counts = {}
    for key, result in zip(circuits.keys(), results):
        counts[key] = result.data.meas.get_counts()
    return counts


def counts_to_prob_vector(counts, shots):
    """Converts counts dict to probability vector [p00, p01, p10, p11]."""
    return np.array([
        counts.get('00', 0) / shots,
        counts.get('01', 0) / shots,
        counts.get('10', 0) / shots,
        counts.get('11', 0) / shots
    ])


def apply_mitigation(counts, A_inv, shots):
    """
    Applies A-matrix mitigation to raw counts.
    Returns mitigated counts dict.
    """
    noisy_prob = counts_to_prob_vector(counts, shots)
    mitigated_quasi = dot(A_inv, noisy_prob)
    return {
        '00': mitigated_quasi[0],
        '01': mitigated_quasi[1],
        '10': mitigated_quasi[2],
        '11': mitigated_quasi[3]
    }


def compute_energy_from_counts(counts_zz, counts_xx,
                                mitigate=False, A_inv=None):
    """
    Computes H2 energy from raw measurement counts.

    H2 = -0.580295·II + (-0.22575)·ZI + 0.17407·IZ
          + 0.12091·ZZ + 0.17407·XX

    For each Pauli term:
    - II: constant = -0.580295 (no measurement)
    - ZI: measure qubit 1 in Z basis (marginalize from ZZ circuit)
    - IZ: measure qubit 0 in Z basis (marginalize from ZZ circuit)
    - ZZ: measure both qubits in Z basis
    - XX: measure both qubits in X basis (H gate + Z measurement)
    """
    if mitigate and A_inv is not None:
        counts_zz = apply_mitigation(counts_zz, A_inv, SHOTS)
        counts_xx = apply_mitigation(counts_xx, A_inv, SHOTS)

    # II term — constant
    energy = -0.580295

    # ZZ term
    obs_zz = SparsePauliOp('ZZ')
    energy += 0.12091 * sampled_expectation_value(counts_zz, obs_zz)

    # ZI term — marginalize qubit 1 from ZZ counts
    counts_zi = marginal_counts(counts_zz, indices=[1])
    obs_z = SparsePauliOp('Z')
    energy += -0.22575 * sampled_expectation_value(counts_zi, obs_z)

    # IZ term — marginalize qubit 0 from ZZ counts
    counts_iz = marginal_counts(counts_zz, indices=[0])
    energy += 0.17407 * sampled_expectation_value(counts_iz, obs_z)

    # XX term
    obs_xx = SparsePauliOp('ZZ')  # after H rotation, XX becomes ZZ
    energy += 0.17407 * sampled_expectation_value(counts_xx, obs_xx)

    return energy


def run_vqe(p, initial_params, mitigate=False):
    """
    Runs VQE using SamplerV2 with optional A-matrix mitigation.
    """
    backend = make_noisy_backend(p)
    A_inv = inv(build_a_matrix(p)) if mitigate else None

    ansatz, _ = build_ansatz(reps=1)
    energy_history = []
    iteration = [0]

    with Session(backend=backend) as session:
        sampler = SamplerV2(mode=session)
        sampler.options.default_shots = SHOTS

        def cost_function(params):
            circuits = build_pauli_circuits(ansatz, params)
            counts = get_noisy_counts(circuits, sampler)

            energy = compute_energy_from_counts(
                counts['ZZ'], counts['XX'],
                mitigate=mitigate, A_inv=A_inv
            )

            energy_history.append(energy)
            iteration[0] += 1
            if iteration[0] % 20 == 0:
                mit_label = "mitigated" if mitigate else "noisy"
                print(f"    [{mit_label}] iter {iteration[0]}: "
                      f"E = {energy:.6f} Ha")
            return energy

        result = minimize(
            cost_function,
            initial_params.copy(),
            method='COBYLA',
            options={'maxiter': 300, 'rhobeg': 0.3}
        )

    return {
        'converged_energy': result.fun,
        'energy_history': energy_history,
        'iterations': len(energy_history)
    }


def main():
    print("VQE H2 — A-matrix Measurement Error Mitigation")
    print(f"Exact FCI energy: {EXACT_ENERGY} Ha")
    print("=" * 55)

    # Fixed initial parameters
    ansatz, _ = build_ansatz(reps=1)
    np.random.seed(42)
    initial_params = np.random.uniform(0, np.pi, ansatz.num_parameters)
    print(f"Fixed initial params: {np.round(initial_params, 4)}\n")

    noise_levels = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    summary_rows = []
    all_results = {}

    for p in noise_levels:
        label = f"p={p:.2f}"
        print(f"\n--- Noise level {label} ---")

        # Run without mitigation
        print(f"Running VQE [noisy, {label}]...")
        res_noisy = run_vqe(p, initial_params, mitigate=False)

        # Run with mitigation (skip for ideal case)
        if p > 0.0:
            print(f"Running VQE [mitigated, {label}]...")
            res_mitigated = run_vqe(p, initial_params, mitigate=True)
        else:
            res_mitigated = res_noisy  # ideal = mitigated when p=0

        error_noisy = abs(res_noisy['converged_energy'] - EXACT_ENERGY)
        error_mitigated = abs(res_mitigated['converged_energy'] - EXACT_ENERGY)

        print(f"  Noisy:     {res_noisy['converged_energy']:.6f} Ha "
              f"(error: {error_noisy:.6f})")
        print(f"  Mitigated: {res_mitigated['converged_energy']:.6f} Ha "
              f"(error: {error_mitigated:.6f})")

        summary_rows.append({
            'readout_error_prob': p,
            'energy_noisy': res_noisy['converged_energy'],
            'energy_mitigated': res_mitigated['converged_energy'],
            'error_noisy': error_noisy,
            'error_mitigated': error_mitigated
        })
        all_results[p] = (res_noisy, res_mitigated)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Converged energy comparison
    ax1 = axes[0]
    noise_vals = [r['readout_error_prob'] for r in summary_rows]
    energies_noisy = [r['energy_noisy'] for r in summary_rows]
    energies_mitigated = [r['energy_mitigated'] for r in summary_rows]

    ax1.plot(noise_vals, energies_noisy, 'o-', color='steelblue',
             linewidth=2.5, markersize=8, label='Without mitigation')
    ax1.plot(noise_vals, energies_mitigated, 's-', color='green',
             linewidth=2.5, markersize=8, label='With A-matrix MEM')
    ax1.axhline(y=EXACT_ENERGY, color='red', linestyle='--',
                linewidth=1.8, label=f'Exact FCI: {EXACT_ENERGY} Ha')

    ax1.set_xlabel('Readout Error Probability (p)', fontsize=13)
    ax1.set_ylabel('Converged Energy (Hartree)', fontsize=13)
    ax1.set_title('VQE H2 — Effect of A-matrix MEM\non Converged Energy',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error from FCI comparison
    ax2 = axes[1]
    errors_noisy = [r['error_noisy'] for r in summary_rows]
    errors_mitigated = [r['error_mitigated'] for r in summary_rows]

    x = np.arange(len(noise_vals))
    width = 0.35
    ax2.bar(x - width/2, errors_noisy, width, color='steelblue',
            label='Without mitigation', alpha=0.85)
    ax2.bar(x + width/2, errors_mitigated, width, color='green',
            label='With A-matrix MEM', alpha=0.85)

    ax2.set_xlabel('Readout Error Probability (p)', fontsize=13)
    ax2.set_ylabel('|Error from FCI| (Hartree)', fontsize=13)
    ax2.set_title('Energy Error Comparison\nWith vs Without MEM',
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'p={p:.2f}' for p in noise_vals])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/plots/a_matrix_mem_vqe.png', dpi=150)
    plt.close()
    print("\nPlot saved to results/plots/a_matrix_mem_vqe.png")

    # Save CSV
    df = pd.DataFrame(summary_rows)
    df.to_csv('results/data/a_matrix_mem_results.csv', index=False)
    print("CSV saved to results/data/a_matrix_mem_results.csv")

    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()