from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator, Statevector

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qft import build_recursive_qft, build_standard_qft, qft, qft_on_amplitudes
from qft.recursive import qft as recursive_qft_alias
from qft.standard import qft as standard_qft_alias


@pytest.mark.parametrize("num_qubits", range(0, 6))
def test_standard_package_circuit_matches_qiskit_reference(num_qubits: int) -> None:
    package_circuit = build_standard_qft(num_qubits)

    if num_qubits == 0:
        assert package_circuit.num_qubits == 0
        return

    reference = QuantumCircuit(num_qubits)
    reference.append(QFTGate(num_qubits), range(num_qubits))
    assert Operator(package_circuit).equiv(Operator(reference))


@pytest.mark.parametrize("num_qubits", range(0, 6))
def test_recursive_package_circuit_matches_qiskit_reference(num_qubits: int) -> None:
    package_circuit = build_recursive_qft(num_qubits)

    if num_qubits == 0:
        assert package_circuit.num_qubits == 0
        return

    reference = QuantumCircuit(num_qubits)
    reference.append(QFTGate(num_qubits), range(num_qubits))
    assert Operator(package_circuit).equiv(Operator(reference))


def test_qft_alias_returns_same_circuit() -> None:
    assert Operator(qft(4)).equiv(Operator(build_standard_qft(4)))
    assert Operator(standard_qft_alias(4)).equiv(Operator(build_standard_qft(4)))
    assert Operator(recursive_qft_alias(4)).equiv(Operator(build_recursive_qft(4)))


def test_amplitude_transform_matches_circuit_action() -> None:
    amplitudes = np.arange(1, 9, dtype=float)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)

    circuit = QuantumCircuit(3)
    circuit.initialize(amplitudes, range(3))
    circuit.compose(build_standard_qft(3), inplace=True)

    actual = Statevector.from_instruction(circuit).data
    expected = qft_on_amplitudes(amplitudes)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("num_qubits", [-1, -2])
def test_negative_qubits_are_rejected(num_qubits: int) -> None:
    with pytest.raises(ValueError):
        build_standard_qft(num_qubits)
