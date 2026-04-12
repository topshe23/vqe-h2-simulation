# readout_noise_estimator.py
# VQE H2 — Readout Noise Analysis using Estimator Primitive
#
# Approach:
# - Plain AerSimulator with custom ReadoutError noise model
#   (no gate noise — isolates measurement error effect cleanly)
# - EstimatorV2 from qiskit_ibm_runtime handles Pauli basis
#   rotations and expectation value computation internally
# - Session mode wraps the VQE optimization loop, as appropriate
#   for iterative algorithms where each step depends on the last
# - Fixed initial parameters across all noise levels so that
#   observed variance comes only from noise, not starting point
# - std extracted directly from Estimator result per iteration
#   (no need to run multiple times to estimate variance)
#
# Experiment:
# - Sweep readout error probability p from 0.00 to 0.05
# - For each p, run VQE and record converged energy, std, error from FCI
# - Plot convergence curves and energy deviation vs noise level

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError
from qiskit_ibm_runtime import EstimatorV2, Session

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


def make_noisy_backend(readout_error_prob):
    """
    Plain AerSimulator with only readout noise.
    No gate noise — isolates measurement error effect cleanly.
    """
    noise_model = NoiseModel()

    if readout_error_prob > 0.0:
        p = readout_error_prob
        readout = ReadoutError([[1 - p, p], [p, 1 - p]])
        noise_model.add_all_qubit_readout_error(readout)

    sim = AerSimulator(noise_model=noise_model)
    return sim

def transpile_for_backend(circuit, hamiltonian, backend):
    """
    Transpiles circuit and observable to match backend's
    instruction set architecture (ISA).
    Required by EstimatorV2 from qiskit_ibm_runtime.
    """
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)
    isa_hamiltonian = hamiltonian.apply_layout(isa_circuit.layout)
    return isa_circuit, isa_hamiltonian


def run_vqe_with_session(readout_prob, initial_params, shots=8192):
    """
    Runs VQE using EstimatorV2 inside a Session.

    Session mode is appropriate for VQE because:
    - VQE is iterative — COBYLA calls cost function 100+ times
    - Each call depends on the previous result
    - Session keeps the connection open across all iterations
    - No re-queuing between iterations

    Args:
        readout_prob: readout error probability
        initial_params: fixed numpy array — same across all noise levels
        shots: shots per Estimator call

    Returns:
        dict with results
    """
    # Build noisy backend
    backend = make_noisy_backend(readout_prob)

    # Build and transpile circuit + hamiltonian
    qc, _ = build_ansatz(reps=1)
    isa_circuit, isa_hamiltonian = transpile_for_backend(
        qc, H2_HAMILTONIAN, backend)

    energy_history = []
    std_history = []
    iteration = [0]

    # Session mode — keeps connection open for all VQE iterations
    with Session(backend=backend) as session:
        estimator = EstimatorV2(mode=session)
        estimator.options.default_shots = shots

        def cost_function(params):
            # PUB format: (circuit, observable, parameter_values)
            job = estimator.run([(isa_circuit, isa_hamiltonian, params)])
            result = job.result()[0]

            energy = float(result.data.evs)
            std    = float(result.data.stds)

            energy_history.append(energy)
            std_history.append(std)
            iteration[0] += 1

            if iteration[0] % 20 == 0:
                print(f"    iter {iteration[0]}: "
                      f"E = {energy:.6f} Ha, std = {std:.6f}")

            return energy

        result = minimize(
            cost_function,
            initial_params.copy(),
            method='COBYLA',
            options={'maxiter': 300, 'rhobeg': 0.3}
        )

    return {
        'converged_energy': result.fun,
        'std_at_convergence': std_history[-1] if std_history else 0.0,
        'energy_history': energy_history,
        'std_history': std_history,
        'iterations': len(energy_history)
    }


def main():
    print("VQE H2 — Readout Noise Experiment")
    print("Using EstimatorV2 + Session (qiskit_ibm_runtime)")
    print(f"Exact FCI energy: {EXACT_ENERGY} Ha")
    print("=" * 55)

    # Fixed initial parameters — noise is the only variable
    qc, _ = build_ansatz(reps=1)
    np.random.seed(42)
    initial_params = np.random.uniform(0, np.pi, qc.num_parameters)
    print(f"Fixed initial params: {np.round(initial_params, 4)}\n")

    noise_levels = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    results = {}
    summary_rows = []

    for p in noise_levels:
        label = "Ideal" if p == 0.0 else f"p={p:.2f}"
        print(f"Running VQE [{label}]...")
        res = run_vqe_with_session(p, initial_params, shots=8192)
        results[p] = res

        error_fci = abs(res['converged_energy'] - EXACT_ENERGY)
        print(f"  Converged energy:  {res['converged_energy']:.6f} Ha")
        print(f"  Std at convergence:{res['std_at_convergence']:.6f}")
        print(f"  Error from FCI:    {error_fci:.6f} Ha")
        print(f"  Iterations:        {res['iterations']}\n")

        summary_rows.append({
            'readout_error_prob': p,
            'converged_energy': res['converged_energy'],
            'std_at_convergence': res['std_at_convergence'],
            'error_from_fci': error_fci,
            'iterations': res['iterations']
        })

    # --- Plot 1: Convergence curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['green', 'steelblue', 'royalblue',
              'cornflowerblue', 'mediumpurple', 'red']
    labels = ['Ideal (p=0.00)', 'p=0.01', 'p=0.02',
              'p=0.03', 'p=0.04', 'p=0.05']

    ax1 = axes[0]
    for (p, res), color, label in zip(results.items(), colors, labels):
        ax1.plot(res['energy_history'], color=color,
                 linewidth=1.8, alpha=0.85, label=label)
    ax1.axhline(y=EXACT_ENERGY, color='black', linestyle='--',
                linewidth=1.5, label=f'Exact FCI: {EXACT_ENERGY} Ha')
    ax1.set_xlabel('Iteration', fontsize=13)
    ax1.set_ylabel('Energy (Hartree)', fontsize=13)
    ax1.set_title("VQE Convergence Under Readout Error\n"
                  "(FakeNairobiV2 + custom readout noise)",
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Converged energy vs noise ---
    ax2 = axes[1]
    noise_vals = [r['readout_error_prob'] for r in summary_rows]
    energies   = [r['converged_energy'] for r in summary_rows]
    stds       = [r['std_at_convergence'] for r in summary_rows]
    errors_fci = [r['error_from_fci'] for r in summary_rows]

    ax2.errorbar(noise_vals, energies, yerr=stds,
                 fmt='o-', color='steelblue', linewidth=2.5,
                 markersize=8, capsize=5, capthick=2,
                 label='Converged Energy ± std')
    ax2.axhline(y=EXACT_ENERGY, color='red', linestyle='--',
                linewidth=1.8, label=f'Exact FCI: {EXACT_ENERGY} Ha')

    for p, e, err in zip(noise_vals, energies, errors_fci):
        ax2.annotate(f'+{err:.4f} Ha',
                     xy=(p, e),
                     xytext=(p + 0.002, e + 0.008),
                     fontsize=8, color='gray')

    ax2.set_xlabel('Readout Error Probability (p)', fontsize=13)
    ax2.set_ylabel('Converged Energy (Hartree)', fontsize=13)
    ax2.set_title('Effect of Readout Error on\nVQE Converged Energy',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/plots/readout_noise_estimator.png', dpi=150)
    plt.close()
    print("Plot saved to results/plots/readout_noise_estimator.png")

    # --- Save CSV ---
    df = pd.DataFrame(summary_rows)
    df.to_csv('results/data/readout_noise_estimator.csv', index=False)
    print("CSV saved to results/data/readout_noise_estimator.csv")
    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()