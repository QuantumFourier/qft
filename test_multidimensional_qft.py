from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from multidimensional_qft import (
    build_multidimensional_qft,
    expected_multidimensional_qft_state,
    padded_shape,
    prepare_multidimensional_input,
)


# Verify the multidimensional QFT against the classical multidimensional DFT.
def main() -> None:
    for shape in [(2, 2), (4, 2), (2, 2, 2), (3, 2)]:
        sample_array = np.arange(1, np.prod(shape) + 1, dtype=float).reshape(shape, order="F")
        _, amplitudes = prepare_multidimensional_input(sample_array)
        expected_state = expected_multidimensional_qft_state(sample_array)
        total_qubits = int(np.log2(np.prod(padded_shape(shape))))

        for method in ("standard", "recursive"):
            multidimensional_qft = build_multidimensional_qft(shape, method=method)
            circuit = QuantumCircuit(total_qubits)
            circuit.initialize(amplitudes, range(total_qubits))
            circuit.compose(multidimensional_qft, inplace=True)
            actual_state = Statevector.from_instruction(circuit).data
            assert np.allclose(actual_state, expected_state)

    print("All multidimensional QFT checks passed.")


if __name__ == "__main__":
    main()
