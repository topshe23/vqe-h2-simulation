# src/circuit.py
# Builds the VQE ansatz circuit for H2 simulation

from qiskit.circuit import QuantumCircuit, ParameterVector
import numpy as np


def build_hf_initial_state():
    """
    Hartree-Fock initial state for H2.
    H2 has 2 electrons in 4 spin-orbitals.
    We use 2 qubits after qubit reduction.
    HF state = |01> in Jordan-Wigner mapping.
    """
    qc = QuantumCircuit(2)
    qc.x(0)  # Put qubit 0 in |1> — represents first electron
    return qc


def build_ansatz(reps=1):
    """
    Builds a parameterized ansatz circuit for VQE.
    Uses RY rotations + CNOT entanglement (hardware efficient ansatz).

    Args:
        reps: number of repetition layers (more reps = more expressive)

    Returns:
        qc: parameterized QuantumCircuit
        params: ParameterVector (the angles we'll optimize)
    """
    n_qubits = 2
    n_params = n_qubits * (reps + 1)
    params = ParameterVector('θ', n_params)

    qc = QuantumCircuit(n_qubits)

    # Start from HF state
    qc.x(0)
    qc.barrier(label='HF Init')

    # Variational layers
    param_idx = 0
    for rep in range(reps):
        # Rotation layer
        for qubit in range(n_qubits):
            qc.ry(params[param_idx], qubit)
            param_idx += 1
        # Entanglement layer
        qc.cx(0, 1)
        qc.barrier(label=f'Layer {rep+1}')

    # Final rotation layer
    for qubit in range(n_qubits):
        qc.ry(params[param_idx], qubit)
        param_idx += 1

    return qc, params


def draw_circuit(qc, save_path='images/circuit.png'):
    """Saves circuit diagram."""
    fig = qc.draw(output='mpl', fold=-1)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Circuit saved to {save_path}")


if __name__ == "__main__":
    qc, params = build_ansatz(reps=1)
    print("Ansatz circuit created!")
    print(f"Parameters: {list(params)}")
    print(f"Circuit depth: {qc.depth()}")
    print(f"Gate counts: {qc.count_ops()}")
    draw_circuit(qc)
    print(qc.draw(output='text'))