from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator, Statevector

from forward_qft import build_recursive_qft, build_standard_qft, qft_on_amplitudes


# Verify that both QFT constructions match the expected transform.
def main() -> None:
    for num_qubits in range(1, 6):
        reference = QuantumCircuit(num_qubits)
        reference.append(QFTGate(num_qubits), range(num_qubits))

        standard_qft = build_standard_qft(num_qubits)
        recursive_qft = build_recursive_qft(num_qubits)

        assert Operator(standard_qft).equiv(Operator(reference))
        assert Operator(recursive_qft).equiv(Operator(reference))

        amplitudes = np.arange(1, 2**num_qubits + 1, dtype=float)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)

        circuit = QuantumCircuit(num_qubits)
        circuit.initialize(amplitudes, range(num_qubits))
        circuit.compose(standard_qft, inplace=True)

        recursive_circuit = QuantumCircuit(num_qubits)
        recursive_circuit.initialize(amplitudes, range(num_qubits))
        recursive_circuit.compose(recursive_qft, inplace=True)

        expected = qft_on_amplitudes(amplitudes)
        actual = Statevector.from_instruction(circuit).data
        recursive_actual = Statevector.from_instruction(recursive_circuit).data
        assert np.allclose(actual, expected)
        assert np.allclose(recursive_actual, expected)

    print("All standard QFT method and recursive QFT checks passed.")


if __name__ == "__main__":
    main()
