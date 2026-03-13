from qiskit.circuit import QuantumCircuit, ParameterVector
import numpy as np


def build_ansatz(reps=1):
    """
    RyRz hardware efficient ansatz for H2.
    2 qubits, full rotation + entanglement layers.
    """
    n_qubits = 2
    n_params = 2 * n_qubits * (reps + 1)
    params = ParameterVector('θ', n_params)
    qc = QuantumCircuit(n_qubits)

    # HF initial state
    qc.x(0)
    qc.barrier(label='HF')

    idx = 0
    for rep in range(reps + 1):
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
            qc.rz(params[idx], q)
            idx += 1
        if rep < reps:
            qc.cx(0, 1)
            qc.barrier(label=f'Layer {rep+1}')

    return qc, params


def draw_circuit(qc, save_path='images/circuit.png'):
    fig = qc.draw(output='mpl', fold=-1)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Circuit saved to {save_path}")


if __name__ == "__main__":
    qc, params = build_ansatz(reps=1)
    print(f"Parameters: {qc.num_parameters}")
    print(f"Depth: {qc.depth()}")
    draw_circuit(qc)
    print(qc.draw(output='text'))