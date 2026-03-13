import numpy as np
import sys
sys.path.insert(0, '.')
from src.circuit import build_ansatz
from src.simulator import run_vqe, get_hamiltonian
from src.utils import plot_energy_landscape, save_results

bond_lengths = np.linspace(0.3, 2.5, 12)
energies = []

for bl in bond_lengths:
    hamiltonian = get_hamiltonian(bond_length=bl)
    qc, _ = build_ansatz(reps=1)
    result = run_vqe(qc, hamiltonian, max_iter=200)
    energies.append(result['optimal_energy'])
    print(f"bond_length={bl:.2f}A -> energy={result['optimal_energy']:.6f} Ha")

plot_energy_landscape(list(bond_lengths), energies, 'results/plots/energy_landscape.png')
save_results({'bond_length': list(bond_lengths), 'energy': energies}, 'results/data/bond_length_results.csv')
print("Done!")