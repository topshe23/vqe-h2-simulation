# src/simulator.py
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize


# Verified H2 Hamiltonian — STO-3G, Jordan-Wigner mapping
# True ground state eigenvalue = -1.137274 Ha
H2_HAMILTONIAN = SparsePauliOp.from_list([
    ("II", -0.580295),
    ("ZI", -0.22575),
    ("IZ",  0.17407),
    ("ZZ",  0.12091),
    ("XX",  0.17407),
])


def get_hamiltonian(bond_length=0.735):
    """
    H2 Hamiltonian scaled for different bond lengths.
    Uses physically motivated scaling of each coefficient.
    """
    r = bond_length
    r0 = 0.735

    # Nuclear repulsion increases as bond shortens
    nuclear = 0.7199 / r

    # Electronic integrals scale with overlap
    overlap = np.exp(-0.5 * abs(r - r0))

    hamiltonian = SparsePauliOp.from_list([
        ("II", -1.05237325 * overlap + nuclear - 0.7199/r0),
        ("IZ",  0.39793742 * overlap),
        ("ZI", -0.39793742 * overlap),
        ("ZZ", -0.01128010 * overlap),
        ("XX",  0.18093120 * overlap),
    ])
    return hamiltonian


def compute_energy(circuit, hamiltonian, parameter_values):
    """Computes <ψ(θ)|H|ψ(θ)> using statevector."""
    param_dict = dict(zip(circuit.parameters, parameter_values))
    bound_circuit = circuit.assign_parameters(param_dict)
    statevector = Statevector(bound_circuit)
    energy = statevector.expectation_value(hamiltonian).real
    return energy


def run_vqe(circuit, hamiltonian, initial_params=None, max_iter=300):
    """Runs VQE with COBYLA optimizer."""
    n_params = circuit.num_parameters
    if initial_params is None:
        np.random.seed(42)
        initial_params = np.random.uniform(0, np.pi, n_params)

    energy_history = []
    iteration = [0]

    def cost_function(params):
        energy = compute_energy(circuit, hamiltonian, params)
        energy_history.append(energy)
        iteration[0] += 1
        if iteration[0] % 20 == 0:
            print(f"  Iteration {iteration[0]}: energy = {energy:.6f} Ha")
        return energy

    print(f"Starting VQE with {n_params} parameters...")
    result = minimize(
        cost_function,
        initial_params,
        method='COBYLA',
        options={'maxiter': max_iter, 'rhobeg': 0.3}
    )

    return {
        'optimal_energy': result.fun,
        'optimal_params': result.x,
        'energy_history': energy_history,
        'converged': result.success,
        'iterations': len(energy_history)
    }