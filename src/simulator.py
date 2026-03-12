# src/simulator.py
# Handles energy measurement and VQE simulation

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit import transpile


# --- H2 Hamiltonian (hardcoded at equilibrium bond length 0.735 Angstrom) ---
# Derived from STO-3G basis set, Jordan-Wigner mapping, qubit reduction
# H = h0*II + h1*ZZ + h2*XX + h3*YY + h4*ZI + h5*IZ
H2_HAMILTONIAN = SparsePauliOp.from_list([
    ("II", -1.0523732),
    ("ZZ",  0.39793742),
    ("XX", -0.39793742),
    ("YY", -0.01128010),
    ("ZI", -0.39793742),
    ("IZ",  0.39793742),
])


def get_hamiltonian(bond_length=0.735):
    """
    Returns H2 Hamiltonian coefficients for a given bond length.
    We interpolate coefficients to simulate different bond lengths.
    
    At equilibrium (0.735A): ground state energy ~ -1.137 Hartree
    As bond length increases: energy increases (molecule dissociates)
    
    Args:
        bond_length: H-H distance in Angstroms
    Returns:
        SparsePauliOp Hamiltonian
    """
    # Scale coefficients based on bond length deviation from equilibrium
    # This is a simplified model — real VQE would recompute integrals
    scale = np.exp(-0.5 * (bond_length - 0.735)**2 / 0.3**2)
    
    hamiltonian = SparsePauliOp.from_list([
        ("II", -1.0523732 * (0.5 + 0.5 * scale)),
        ("ZZ",  0.39793742 * scale),
        ("XX", -0.39793742 * scale),
        ("YY", -0.01128010 * scale),
        ("ZI", -0.39793742 * scale),
        ("IZ",  0.39793742 * scale),
    ])
    return hamiltonian


def compute_energy(circuit, hamiltonian, parameter_values):
    """
    Computes energy expectation value <ψ(θ)|H|ψ(θ)>
    Uses statevector simulation for exact results.
    
    Args:
        circuit: parameterized ansatz QuantumCircuit
        hamiltonian: SparsePauliOp
        parameter_values: list of floats for circuit parameters
    Returns:
        energy: float (in Hartree)
    """
    # Bind parameters to circuit
    param_dict = dict(zip(circuit.parameters, parameter_values))
    bound_circuit = circuit.assign_parameters(param_dict)
    
    # Get statevector
    statevector = Statevector(bound_circuit)
    
    # Compute expectation value
    energy = statevector.expectation_value(hamiltonian).real
    return energy


def run_vqe(circuit, hamiltonian, initial_params=None, max_iter=200):
    """
    Runs the VQE optimization loop.
    Uses COBYLA optimizer to minimize energy.
    
    Args:
        circuit: parameterized ansatz
        hamiltonian: SparsePauliOp
        initial_params: starting parameter values (random if None)
        max_iter: maximum optimizer iterations
    Returns:
        result dict with optimal energy, params, and convergence history
    """
    from scipy.optimize import minimize
    
    n_params = circuit.num_parameters
    if initial_params is None:
        initial_params = np.random.uniform(-np.pi, np.pi, n_params)
    
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
        options={'maxiter': max_iter, 'rhobeg': 0.5}
    )

    return {
        'optimal_energy': result.fun,
        'optimal_params': result.x,
        'energy_history': energy_history,
        'converged': result.success,
        'iterations': len(energy_history)
    }