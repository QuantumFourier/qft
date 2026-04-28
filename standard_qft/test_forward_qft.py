from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator, Statevector

PARENT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PARENT_DIR / "src"
for candidate in (PARENT_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from qft.standard_qft import build_recursive_qft, build_standard_qft, qft_on_amplitudes


def assert_qft_matches_reference(num_qubits: int) -> None:
    reference = QuantumCircuit(num_qubits)
    reference.append(QFTGate(num_qubits), range(num_qubits))

    standard_qft = build_standard_qft(num_qubits)
    recursive_qft = build_recursive_qft(num_qubits)

    assert Operator(standard_qft).equiv(Operator(reference))
    assert Operator(recursive_qft).equiv(Operator(reference))

    amplitudes = np.arange(1, 2**num_qubits + 1, dtype=float)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)

    standard_circuit = QuantumCircuit(num_qubits)
    standard_circuit.initialize(amplitudes, range(num_qubits))
    standard_circuit.compose(standard_qft, inplace=True)

    recursive_circuit = QuantumCircuit(num_qubits)
    recursive_circuit.initialize(amplitudes, range(num_qubits))
    recursive_circuit.compose(recursive_qft, inplace=True)

    expected = qft_on_amplitudes(amplitudes)
    actual = Statevector.from_instruction(standard_circuit).data
    recursive_actual = Statevector.from_instruction(recursive_circuit).data
    assert np.allclose(actual, expected)
    assert np.allclose(recursive_actual, expected)


@pytest.mark.parametrize("num_qubits", range(1, 6))
def test_qft_constructions_match_reference(num_qubits: int) -> None:
    assert_qft_matches_reference(num_qubits)


def test_zero_qubit_qft_builders_return_identity_circuits() -> None:
    assert build_standard_qft(0).num_qubits == 0
    assert build_recursive_qft(0).num_qubits == 0


@pytest.mark.parametrize("builder", (build_standard_qft, build_recursive_qft))
def test_negative_qubit_counts_are_rejected(builder) -> None:
    with pytest.raises(ValueError):
        builder(-1)


# Verify that both QFT constructions match the expected transform.
def main() -> None:
    for num_qubits in range(1, 6):
        assert_qft_matches_reference(num_qubits)

    print("All standard QFT method and recursive QFT checks passed.")


if __name__ == "__main__":
    main()
